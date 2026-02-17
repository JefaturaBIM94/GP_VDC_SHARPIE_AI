# backend/video_routes.py
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from PIL import Image

from .video_store import VideoSessionStore

# IMPORTANTE:
# Ajusta estos imports a tu estructura real. La idea es reutilizar EXACTAMENTE
# la misma función/engine que ya usas en /segment_image.
from .sam3_engine import SAM3Engine  # o donde tengas tu engine real


router = APIRouter(prefix="/video", tags=["video"])
store = VideoSessionStore()

# Inicializa el engine una vez (MVP)
sam_engine = SAM3Engine(device="cuda")  # o "cpu" / autodetect


class VideoProcessResponse(BaseModel):
    session_id: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


class SegmentFrameRequest(BaseModel):
    session_id: str
    frame_idx: int
    prompt: str
    threshold: float = 0.35


@router.post("/process", response_model=VideoProcessResponse)
async def process_video(file: UploadFile = File(...)):
    # Guardar el video temporalmente
    ext = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    tmp_path = os.path.join(os.getcwd(), f"_tmp_upload_{os.getpid()}_{id(file)}{ext}")

    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        sess = store.create_session_from_upload(tmp_path)
        return VideoProcessResponse(
            session_id=sess.session_id,
            fps=sess.fps,
            frame_count=sess.frame_count,
            width=sess.width,
            height=sess.height,
            duration_s=sess.duration_s,
        )
    except Exception as e:
        # Si falla, limpia el tmp
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/frame/{session_id}/{frame_idx}")
def get_frame(session_id: str, frame_idx: int):
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session_id no existe")

    if frame_idx < 0 or frame_idx >= sess.frame_count:
        raise HTTPException(status_code=400, detail="frame_idx fuera de rango")

    path = sess.frame_path(frame_idx)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Frame no encontrado en disco")

    return FileResponse(path, media_type="image/jpeg")


@router.post("/segment_frame")
def segment_frame(req: SegmentFrameRequest):
    sess = store.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session_id no existe")

    if req.frame_idx < 0 or req.frame_idx >= sess.frame_count:
        raise HTTPException(status_code=400, detail="frame_idx fuera de rango")

    frame_path = sess.frame_path(req.frame_idx)
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame no encontrado")

    # Cargar PIL
    img = Image.open(frame_path).convert("RGB")

    # Reutiliza tu lógica actual de segmentación (mismo output que /segment_image)
    # OJO: aquí debes construir el mismo SegmentResponse que ya consume tu frontend.
    # Si tu engine actual ya te regresa overlay/id_map/labels, usa eso.
    segment_response = sam_engine.segment_image(img, req.prompt, req.threshold)

    if segment_response is None:
        # Regresa estructura vacía compatible con el front
        return {
            "labels": [],
            "scores": [],
            "overlay_image_b64": None,
            "id_map_rgb_b64": None,
        }

    return segment_response
