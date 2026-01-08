# api_server.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from io import BytesIO
import base64
import uuid

import numpy as np
from PIL import Image

from sam3_engine import Sam3Engine

app = FastAPI(title="SAM3 GP RD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Sam3Engine()


class InstanceLabel(BaseModel):
    id: int
    class_name: str
    cx: float  # [0,1]
    cy: float  # [0,1]
    score: float
    area_px: int
    color: str  # "#rrggbb"


class SegmentResponse(BaseModel):
    session_id: str
    threshold: float
    overlay_image_b64: str
    id_map_b64: str  # PNG 16-bit: pixel = instance_id (0=fondo)
    classes_counts: Dict[str, int]
    labels: List[InstanceLabel]


def _deterministic_color(i: int) -> tuple[int, int, int]:
    # Paleta determinista por id (no aleatoria por request)
    # Se ve estable en multi-prompt.
    rng = np.random.default_rng(1000 + i)
    c = rng.integers(low=60, high=240, size=(3,), endpoint=True)
    return int(c[0]), int(c[1]), int(c[2])


def _encode_png_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_id_map_16bit_b64(id_map_u16: np.ndarray) -> str:
    """
    id_map_u16: HxW uint16
    """
    if id_map_u16.dtype != np.uint16:
        id_map_u16 = id_map_u16.astype(np.uint16)

    # PIL soporta 16-bit con modo I;16
    pil = Image.fromarray(id_map_u16, mode="I;16")
    return _encode_png_b64(pil)


def _overlay_and_maps(
    image_pil: Image.Image,
    per_prompt_results: List[dict],
    alpha: float = 0.55,
):
    """
    per_prompt_results: [{class_name, masks_np, scores_np}, ...]
    Genera:
      - overlay_b64 (RGB)
      - id_map_b64 (I;16) pixel->instance_id
      - labels: List[InstanceLabel]
      - classes_counts: Dict[str,int]
    """
    img_np = np.array(image_pil.convert("RGB"), copy=True)
    h, w, _ = img_np.shape

    id_map = np.zeros((h, w), dtype=np.uint16)
    labels: List[InstanceLabel] = []
    classes_counts: Dict[str, int] = {}

    next_id = 1

    for pr in per_prompt_results:
        cls = pr["class_name"]
        masks_np: np.ndarray = pr["masks_np"]
        scores_np: np.ndarray = pr["scores_np"]

        if masks_np is None or scores_np is None or masks_np.shape[0] == 0:
            continue

        classes_counts[cls] = classes_counts.get(cls, 0) + int(masks_np.shape[0])

        for k in range(masks_np.shape[0]):
            mask = masks_np[k]
            score = float(scores_np[k])

            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue

            inst_id = next_id
            next_id += 1

            # id_map: escribir id (si hay overlap, el último gana)
            id_map[mask_bool] = inst_id

            # color estable por inst_id
            r, g, b = _deterministic_color(inst_id)

            # overlay
            img_np[mask_bool] = (
                alpha * np.array([r, g, b], dtype=np.float32)
                + (1 - alpha) * img_np[mask_bool].astype(np.float32)
            ).astype(np.uint8)

            # métricas
            area_px = int(mask_bool.sum())
            ys, xs = np.where(mask_bool)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            labels.append(
                InstanceLabel(
                    id=inst_id,
                    class_name=cls,
                    cx=float(cx / w),
                    cy=float(cy / h),
                    score=score,
                    area_px=area_px,
                    color=f"#{r:02x}{g:02x}{b:02x}",
                )
            )

    overlay_b64 = _encode_png_b64(Image.fromarray(img_np))
    id_map_b64 = _encode_id_map_16bit_b64(id_map)

    # Ordenar labels por clase y luego id (más legible)
    labels.sort(key=lambda x: (x.class_name, x.id))

    return overlay_b64, id_map_b64, labels, classes_counts


@app.post("/api/segment", response_model=SegmentResponse)
async def segment_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),        # soporta CSV: "columns,walls,floors"
    threshold: float = Form(0.5),
):
    content = await image.read()
    image_pil = Image.open(BytesIO(content)).convert("RGB")

    prompts = [p.strip() for p in (prompt or "").split(",") if p.strip()]
    if not prompts:
        prompts = ["columns"]

    per_prompt_results: List[dict] = []

    for p in prompts:
        masks_np, scores_np = engine.segment_image(
            image_pil=image_pil,
            text_prompt=p,
            score_threshold=threshold,
        )
        if masks_np is None or scores_np is None or masks_np.shape[0] == 0:
            continue

        per_prompt_results.append(
            {"class_name": p.strip().lower(), "masks_np": masks_np, "scores_np": scores_np}
        )

    session_id = str(uuid.uuid4())

    overlay_b64, id_map_b64, labels, classes_counts = _overlay_and_maps(
        image_pil=image_pil,
        per_prompt_results=per_prompt_results,
        alpha=0.55,
    )

    return SegmentResponse(
        session_id=session_id,
        threshold=threshold,
        overlay_image_b64=overlay_b64 if labels else "",
        id_map_b64=id_map_b64,
        classes_counts=classes_counts,
        labels=labels,
    )
