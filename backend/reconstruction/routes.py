# backend/reconstruction/routes.py
from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image

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
    content = await image.read()
    image_pil = Image.open(BytesIO(content)).convert("RGB")

    result = _engine.reconstruct_fast(
        image_pil=image_pil,
        make_ply=make_ply,
        stride=stride,
        max_res=max_res,
    )

    # Map engine keys to API response (frontend expects depth_png_b64 and meta)
    return {
        "depth_png_b64": result.get("depth_map_b64", result.get("depth_png_b64", "")),
        "ply_b64": result.get("ply_b64", ""),
        "ply_preview_b64": result.get("ply_preview_b64", ""),
        "meta": result.get("meta", {}),
    }
