# backend/reconstruction/da3_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import tempfile

import numpy as np
from PIL import Image
import torch

from .recon_engine import _encode_png_b64  # reutiliza tu helper actual (o muévelo a utils)
import cv2


@dataclass
class DA3Result:
    depth: np.ndarray          # [H,W] float32
    conf: np.ndarray | None    # [H,W] float32
    intrinsics: np.ndarray | None  # [3,3]
    extrinsics: np.ndarray | None  # [3,4]
    ply_bytes: bytes | None


class DepthAnything3Engine:
    def __init__(self, model_id: str = "depth-anything/da3-giant", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self):
        if self._model is not None:
            return
        from depth_anything_3.api import DepthAnything3  # noqa
        self._model = DepthAnything3.from_pretrained(self.model_id).to(device=self.device)

    @torch.no_grad()
    def infer_one(self, image_pil: Image.Image, export_format: str = "ply") -> DA3Result:
        self.load()
        assert self._model is not None

        # DA3 acepta rutas, PIL o numpy; usamos PIL
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            pred = self._model.inference(
                images=[image_pil.convert("RGB")],
                export_dir=str(out_dir),
                export_format=export_format,
            )

            depth = np.asarray(pred.depth[0], dtype=np.float32)
            conf = np.asarray(pred.conf[0], dtype=np.float32) if getattr(pred, "conf", None) is not None else None
            intr = np.asarray(pred.intrinsics[0], dtype=np.float32) if getattr(pred, "intrinsics", None) is not None else None
            extr = np.asarray(pred.extrinsics[0], dtype=np.float32) if getattr(pred, "extrinsics", None) is not None else None

            ply_bytes = None
            if export_format == "ply":
                # DA3 exporta archivos en export_dir; buscamos el primer .ply
                ply_files = list(out_dir.rglob("*.ply"))
                if ply_files:
                    ply_bytes = ply_files[0].read_bytes()

        return DA3Result(depth=depth, conf=conf, intrinsics=intr, extrinsics=extr, ply_bytes=ply_bytes)

    def depth_to_png_b64(self, depth: np.ndarray) -> str:
        d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        lo = np.percentile(d, 2.0)
        hi = np.percentile(d, 98.0)
        if hi <= lo + 1e-6:
            hi = lo + 1e-6
        dn = (d - lo) / (hi - lo)
        dn = np.clip(dn, 0.0, 1.0)
        u8 = (dn * 255.0).astype(np.uint8)
        cm = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
        return _encode_png_b64(Image.fromarray(rgb))

    def reconstruct_fast(self, image_pil: Image.Image, max_res: int = 1024, return_ply: bool = True) -> dict:
        img = image_pil.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_res) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

        res = self.infer_one(img, export_format=("ply" if return_ply else "npz"))
        depth_png_b64 = self.depth_to_png_b64(res.depth)

        meta = {
            "depth_kind": "da3",
            "w": int(res.depth.shape[1]),
            "h": int(res.depth.shape[0]),
            "has_conf": res.conf is not None,
            "has_intrinsics": res.intrinsics is not None,
            "has_extrinsics": res.extrinsics is not None,
        }

        ply_b64 = ""
        if return_ply and res.ply_bytes is not None:
            MAX_PLY_BYTES = 12 * 1024 * 1024
            if len(res.ply_bytes) > MAX_PLY_BYTES:
                meta["warning"] = f"PLY muy grande ({len(res.ply_bytes)/1024/1024:.1f} MB)."
            else:
                ply_b64 = base64.b64encode(res.ply_bytes).decode("utf-8")

        return {"depth_png_b64": depth_png_b64, "ply_b64": ply_b64, "meta": meta}
