# backend/reconstruction/recon_engine.py
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from io import BytesIO
import base64
import sys
import struct
import time

import numpy as np
from PIL import Image
import torch
import cv2

cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _encode_png_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_ply_from_depth_rgb(image_pil, depth: np.ndarray, stride: int = 2) -> str:
    """
    Genera PLY ASCII (x y z r g b) desde depth + imagen RGB.
    depth: HxW float32
    Retorna base64 string del PLY.
    """
    rgb = np.array(image_pil).astype(np.uint8)  # HxW x3
    H, W = depth.shape[:2]

    # intrinsics aproximados (demo)
    fx = fy = 1.2 * max(H, W)
    cx = W * 0.5
    cy = H * 0.5

    pts = []
    cols = []

    for v in range(0, H, stride):
        for u in range(0, W, stride):
            z = float(depth[v, u])
            if not np.isfinite(z):
                continue
            # back-project
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            r, g, b = rgb[v, u].tolist()
            pts.append((x, y, z))
            cols.append((r, g, b))

    n = len(pts)
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])

    lines = [header]
    for (x, y, z), (r, g, b) in zip(pts, cols):
        lines.append(f"{x} {y} {z} {r} {g} {b}")

    ply_text = "\n".join(lines).encode("utf-8")
    return base64.b64encode(ply_text).decode("utf-8")


class DepthAnythingEngine:
    def __init__(self):
        self._model = None
        self._device = "cpu"  # por ahora
        self._loaded = False

        # rutas
        self._root = Path(__file__).resolve().parents[2]  # project root (…/SAM3_PROJECT)
        self._da2_dir = self._root / "external" / "Depth-Anything-V2"
        self._weights = self._da2_dir / "checkpoints" / "depth_anything_v2_vits.pth"  # AJUSTA SI CAMBIA
        # encoder por defecto (debe coincidir con el checkpoint usado)
        self.encoder = "vits"

    def load(self):
        """
        Carga Depth-Anything-V2 desde la carpeta local external/Depth-Anything-V2
        (NO torch.hub.load).
        """
        if self._loaded:
            return
        if not self._da2_dir.exists():
            raise FileNotFoundError(f"Depth-Anything-V2 folder not found: {self._da2_dir}")
        if not self._weights.exists():
            raise FileNotFoundError(
                f"Weights not found: {self._weights}\n"
                f"Create checkpoints/ and drop the .pth there (see Depth-Anything-V2 README)."
            )

        # 1) inyecta el repo al sys.path para poder importar
        sys.path.insert(0, str(self._da2_dir))

        # 2) imports del repo (si alguno falla, te digo EXACTO qué archivo abrir para corregirlo)
        #    En la mayoría de implementaciones V2, el modelo está en depth_anything_v2/dpt.py
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        # 3) Construir modelo usando config que coincida con el checkpoint
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        MODEL_CONFIGS = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        # Asegurar que el encoder por defecto coincide con el checkpoint
        # WEIGHTS_PATH tal como se pidió
        WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "external" / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vits.pth"

        # Construir modelo con la configuración adecuada
        model = DepthAnythingV2(**MODEL_CONFIGS[self.encoder])

        # Cargar checkpoint (estricto, debe coincidir con la config)
        ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model.load_state_dict(ckpt, strict=True)
        model = model.to(self._device).eval()

        self._model = model
        self._loaded = True

    @torch.no_grad()
    def predict_depth(self, image_pil: Image.Image) -> np.ndarray:
        """
        Devuelve depth como float32 HxW (no métrico; relativo).
        """
        self.load()
        assert self._model is not None

        # --- Preprocess simple y robusto (funciona bien si el repo no trae transforms listos) ---
        img = np.array(image_pil.convert("RGB"), dtype=np.uint8)
        h0, w0 = img.shape[:2]

        # resize a un tamaño razonable (CPU friendly) PERO múltiplo de patch=14
        patch = 14
        target_long = 770  # valor cualquiera, nosotros lo "cuantizamos"
        scale = target_long / max(h0, w0)

        nh = int(round(h0 * scale))
        nw = int(round(w0 * scale))

        # cuantizar a múltiplos de 14 (mínimo 14)
        nh = max(patch, (nh // patch) * patch)
        nw = max(patch, (nw // patch) * patch)

        # evitar que se vaya a 0 por imágenes pequeñas
        if nh < patch:
            nh = patch
        if nw < patch:
            nw = patch

        img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # to tensor + normalize (ImageNet)
        x = torch.from_numpy(img_rs).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # mover tensor al device antes de normalizar y crear mean/std en el mismo device
        x = x.to(self._device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # forward (la mayoría de V2 retorna 1xHxW o 1x1xHxW)
        y = self._model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        if y.ndim == 4:
            y = y[:, 0, :, :]
        depth_rs = y[0].detach().float().cpu().numpy()

        # upsample back to original size
        depth = cv2.resize(depth_rs, (w0, h0), interpolation=cv2.INTER_CUBIC)
        return depth.astype(np.float32)

    def depth_to_png_b64(self, depth: np.ndarray) -> str:
        """
        Normaliza depth->0..255 y aplica colormap para visualizar.
        """
        d = depth.copy()
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # robust percentiles (evita outliers)
        lo = np.percentile(d, 2.0)
        hi = np.percentile(d, 98.0)
        if hi <= lo + 1e-6:
            hi = lo + 1e-6

        dn = (d - lo) / (hi - lo)
        dn = np.clip(dn, 0.0, 1.0)
        u8 = (dn * 255.0).astype(np.uint8)

        # colormap (turbo se ve muy bien)
        cm = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return _encode_png_b64(pil)

    def reconstruct_fast(
        self,
        image_pil: Image.Image,
        make_ply: bool = True,
        max_res: int = 1024,
        stride: int = 4,
    ) -> dict:
        t0 = time.perf_counter()
        img = image_pil.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_res) / float(max(w, h)))
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
        t_resize = time.perf_counter()

        print("[FASTRECON] engine: START predict_depth")
        depth = self.predict_depth(img)
        t_depth = time.perf_counter()
        print("[FASTRECON] engine: END predict_depth", f"{t_depth - t_resize:.3f}s")

        depth_png_b64 = self.depth_to_png_b64(depth)
        t_png = time.perf_counter()

        h, w = depth.shape[:2]
        meta = {
            "depth_kind": "relative",
            "note": "DepthAnythingV2 outputs relative depth (scale not metric).",
            "w": int(w),
            "h": int(h),
        }

        stride = max(1, int(stride))
        ply_b64 = ""
        ply_preview_b64 = ""

        if make_ply:
            print("[FASTRECON] engine: START ply (ascii preview + binary)")
            ply_preview_bytes, meta_pc = _depth_to_ply_bytes(img, depth, stride=stride, format="ascii")
            t_ply_prev = time.perf_counter()
            ply_bytes, _ = _depth_to_ply_bytes(img, depth, stride=stride, format="binary")
            t_ply_bin = time.perf_counter()
            print(
                "[FASTRECON] engine: END ply",
                f"ascii={t_ply_prev - t_png:.3f}s",
                f"binary={t_ply_bin - t_ply_prev:.3f}s",
                f"total={t_ply_bin - t_png:.3f}s",
            )
            print(
                "[FASTRECON] engine: ply sizes",
                f"preview_ascii={len(ply_preview_bytes) / 1024 / 1024:.2f}MB",
                f"binary={len(ply_bytes) / 1024 / 1024:.2f}MB",
            )

            meta["pc"] = meta_pc

            MAX_PREVIEW_BYTES = 6 * 1024 * 1024   # 6MB (viewer)
            MAX_PLY_BYTES = 8 * 1024 * 1024       # 8MB (download por JSON)

            if len(ply_preview_bytes) <= MAX_PREVIEW_BYTES:
                ply_preview_b64 = base64.b64encode(ply_preview_bytes).decode("utf-8")
            else:
                meta["warning_preview"] = (
                    f"Preview PLY (ASCII) muy grande ({len(ply_preview_bytes) / 1024 / 1024:.1f} MB). "
                    "Sube stride."
                )

            if len(ply_bytes) <= MAX_PLY_BYTES:
                ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")
            else:
                meta["warning"] = (
                    f"PLY binario muy grande ({len(ply_bytes) / 1024 / 1024:.1f} MB). "
                    "Sube stride (8-16) o baja max_res."
                )

        t_end = time.perf_counter()
        meta["timings"] = {
            "resize_s": float(t_resize - t0),
            "predict_depth_s": float(t_depth - t_resize),
            "depth_png_s": float(t_png - t_depth),
            "ply_preview_ascii_s": float((t_ply_prev - t_png) if make_ply else 0.0),
            "ply_binary_s": float((t_ply_bin - t_ply_prev) if make_ply else 0.0),
            "total_s": float(t_end - t0),
        }

        return {
            "depth_png_b64": depth_png_b64,
            "ply_b64": ply_b64,
            "ply_preview_b64": ply_preview_b64,
            "meta": meta,
        }


def _depth_to_pointcloud_arrays(
    image_pil: Image.Image,
    depth: np.ndarray,
    stride: int = 4,
    fov_deg: float = 60.0,
):
    """
    Vectorizado: devuelve (xyz float32 Nx3, rgb uint8 Nx3, meta dict)
    depth = relativo; lo normalizamos con percentiles y lo invertimos para pseudo-Z.
    """
    rgb_img = np.array(image_pil.convert("RGB"), dtype=np.uint8)
    h, w = depth.shape[:2]
    s = max(1, int(stride))

    # --- normalize depth robust ---
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo = np.percentile(d, 2.0)
    hi = np.percentile(d, 98.0)
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    dn = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    z = (1.0 - dn).astype(np.float32)  # near -> larger

    # --- sample grid (stride) ---
    z_s = z[0:h:s, 0:w:s]
    rgb_s = rgb_img[0:h:s, 0:w:s, :]

    # mask valid
    m = z_s > 1e-6
    if not np.any(m):
        # empty pointcloud
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        meta = {"num_points": 0, "pc_stride": s, "depth_norm": {"p2": float(lo), "p98": float(hi), "z_inverted": True}}
        return xyz, rgb, meta

    # --- intrinsics approx (pinhole) ---
    fov = np.deg2rad(float(fov_deg))
    fx = 0.5 * w / np.tan(0.5 * fov)
    fy = fx
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    # coordinates for sampled grid
    us = np.arange(0, w, s, dtype=np.float32)
    vs = np.arange(0, h, s, dtype=np.float32)
    U, V = np.meshgrid(us, vs)  # Hs x Ws

    zz = z_s[m].astype(np.float32)
    uu = U[m].astype(np.float32)
    vv = V[m].astype(np.float32)

    X = (uu - cx) / fx * zz
    Y = -((vv - cy) / fy * zz)
    Z = zz

    xyz = np.stack([X, Y, Z], axis=1).astype(np.float32)
    rgb = rgb_s[m].astype(np.uint8)  # Nx3

    meta = {
        "pc_stride": s,
        "fov_deg": float(fov_deg),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "depth_norm": {"p2": float(lo), "p98": float(hi), "z_inverted": True},
        "num_points": int(xyz.shape[0]),
    }
    return xyz, rgb, meta


def _pointcloud_to_ply_ascii(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    """
    ASCII PLY: util para debug / viewer web si quieres mantenerlo simple.
    """
    n = int(xyz.shape[0])
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # formatea con fixed decimals (evita scientific)
    lines = [header]
    # xyz float32, rgb uint8
    for i in range(n):
        x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
        r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    return "".join(lines).encode("utf-8")


def _depth_to_pointcloud_ply_ascii(
    image_pil: Image.Image,
    depth: np.ndarray,
    stride: int = 4,
    fov_deg: float = 60.0,
) -> tuple[bytes, dict]:
    rgb = np.array(image_pil.convert("RGB"), dtype=np.uint8)
    h, w = depth.shape[:2]

    s = max(1, int(stride))

    # depth -> normalized (robusto)
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo = np.percentile(d, 2.0)
    hi = np.percentile(d, 98.0)
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    dn = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    z = (1.0 - dn).astype(np.float32)  # near -> larger

    # sample grid (stride)
    vv = np.arange(0, h, s, dtype=np.int32)
    uu = np.arange(0, w, s, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)  # shape (hv, wv)

    Z = z[V, U].astype(np.float32)
    mask = Z > 1e-6

    U_f = U[mask].astype(np.float32)
    V_f = V[mask].astype(np.float32)
    Z_f = Z[mask]

    # intrinsics aprox
    fov = np.deg2rad(float(fov_deg))
    fx = 0.5 * w / np.tan(0.5 * fov)
    fy = fx
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    X = (U_f - cx) / fx * Z_f
    Y = -((V_f - cy) / fy * Z_f)

    C = rgb[V_f.astype(np.int32), U_f.astype(np.int32)]  # Nx3 uint8

    n = int(Z_f.shape[0])
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # arma texto rápido
    lines = [header]
    for i in range(n):
        r, g, b = int(C[i, 0]), int(C[i, 1]), int(C[i, 2])
        lines.append(f"{X[i]:.6f} {Y[i]:.6f} {Z_f[i]:.6f} {r} {g} {b}\n")

    ply_bytes = "".join(lines).encode("utf-8")

    meta = {
        "pc_stride": int(s),
        "fov_deg": float(fov_deg),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "depth_norm": {"p2": float(lo), "p98": float(hi), "z_inverted": True},
        "num_points": int(n),
    }
    return ply_bytes, meta


def _pointcloud_to_ply_binary_le(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    """
    PLY binario little-endian (CloudCompare-friendly y mucho mas chico).
    """
    n = int(xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    # pack vectorizado (mucho mas rapido que for/struct.pack)
    xyz_f = np.asarray(xyz, dtype=np.float32)
    rgb_u = np.asarray(rgb, dtype=np.uint8)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    data["x"] = xyz_f[:, 0]
    data["y"] = xyz_f[:, 1]
    data["z"] = xyz_f[:, 2]
    data["red"] = rgb_u[:, 0]
    data["green"] = rgb_u[:, 1]
    data["blue"] = rgb_u[:, 2]

    return header + data.tobytes()


def _depth_to_ply_bytes(
    image_pil: Image.Image,
    depth: np.ndarray,
    stride: int = 4,
    format: str = "binary",  # "binary" o "ascii"
) -> tuple[bytes, dict]:
    """
    Devuelve (ply_bytes, meta_pc). Default: binary (CloudCompare).
    """
    if format == "ascii":
        return _depth_to_pointcloud_ply_ascii(image_pil=image_pil, depth=depth, stride=stride, fov_deg=60.0)

    xyz, rgb, meta_pc = _depth_to_pointcloud_arrays(
        image_pil=image_pil,
        depth=depth,
        stride=stride,
        fov_deg=60.0,
    )
    return _pointcloud_to_ply_binary_le(xyz, rgb), meta_pc


# Alias for compatibility with imports elsewhere (e.g. some code expects ReconEngine)
ReconEngine = DepthAnythingEngine
