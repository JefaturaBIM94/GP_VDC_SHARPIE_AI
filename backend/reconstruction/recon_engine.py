# backend/reconstruction/recon_engine.py
from __future__ import annotations

from pathlib import Path
from io import BytesIO
import base64
import sys

import numpy as np
from PIL import Image
import torch
import cv2


def _encode_png_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ReconEngine:
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

    def reconstruct_fast(self, image_pil: Image.Image, make_ply: bool = False) -> dict:
        depth = self.predict_depth(image_pil)
        depth_b64 = self.depth_to_png_b64(depth)

        # por ahora: SOLO depth (para que ya lo veas). PLY lo hacemos en el siguiente paso.
        return {
            "depth_map_b64": depth_b64,
            "ply_b64": "",
            "meta": {
                "depth_kind": "relative",
                "note": "DepthAnythingV2 outputs relative depth (scale not metric)."
            },
        }


# Alias for compatibility with imports elsewhere (e.g. routes.py expects DepthAnythingEngine)
DepthAnythingEngine = ReconEngine
