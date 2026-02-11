# backend/reconstruction/routes.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image, ImageOps
import time

from .recon_engine import DepthAnythingEngine

router = APIRouter()

_engine = DepthAnythingEngine()


class FastReconResponse(BaseModel):
    depth_png_b64: str
    ply_b64: Optional[str] = None
    ply_preview_b64: Optional[str] = None
    meta: Optional[dict] = None


@router.post("/api/reconstruct-fast", response_model=FastReconResponse)
async def reconstruct_fast(
    image: UploadFile = File(...),
    make_ply: bool = Form(True),
    stride: int = Form(4),
    max_res: int = Form(1024),
):
    print("[FASTRECON] request received")
    print("[FASTRECON] make_ply:", make_ply)
    print("[FASTRECON] max_res:", max_res, "stride:", stride)

    t0 = time.perf_counter()
    content = await image.read()
    t1 = time.perf_counter()
    print("[FASTRECON] bytes:", len(content))
    try:
        image_pil = Image.open(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image upload") from e
    image_pil2 = ImageOps.exif_transpose(image_pil)  # match browser orientation
    if image_pil2 is None:
        image_pil2 = image_pil
    image_pil = image_pil2.convert("RGB")
    t2 = time.perf_counter()

    result = _engine.reconstruct_fast(
        image_pil=image_pil,
        make_ply=make_ply,
        stride=stride,
        max_res=max_res,
    )
    t3 = time.perf_counter()

    print(
        f"[FASTRECON] read={t1-t0:.3f}s open={t2-t1:.3f}s engine={t3-t2:.3f}s total={t3-t0:.3f}s "
        f"make_ply={make_ply} stride={stride} max_res={max_res}"
    )
    print("[FASTRECON] reconstruct_fast done in", f"{t3-t0:.3f}", "sec")

    # Map engine keys to API response (frontend expects depth_png_b64 and meta)
    return {
        "depth_png_b64": result.get("depth_map_b64", result.get("depth_png_b64", "")),
        "ply_b64": result.get("ply_b64", ""),
        "ply_preview_b64": result.get("ply_preview_b64", ""),
        "meta": result.get("meta", {}),
    }
