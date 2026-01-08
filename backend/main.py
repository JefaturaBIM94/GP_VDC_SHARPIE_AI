# backend/main.py
import os
import io
import uuid
import base64

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from backend.sam3_engine import Sam3Engine
from backend.session_store import (
    create_session,
    save_session,
    load_session,
    list_sessions,
    IMAGES_DIR,
)

app = FastAPI(title="GPC SAM3 Backend")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Motor SAM3 global ----------
sam3_engine = Sam3Engine()


@app.get("/ping")
def ping():
    return {"status": "ok", "message": "SAM3 backend vivo"}


# ---------- Utilidades internas ----------

def random_color():
    return np.array([
        np.random.randint(80, 256),
        np.random.randint(80, 256),
        np.random.randint(80, 256),
    ], dtype=np.float32)


def overlay_masks_on_image(image_pil, masks_np, scores_np):
    """
    Devuelve:
    - overlay_pil
    - objects: lista de dicts {id, area, score, color}
    """
    image_np = np.array(image_pil).astype(np.float32)
    h, w, _ = image_np.shape

    n_instances = masks_np.shape[0]
    alpha = 0.5

    objects = []

    for idx in range(n_instances):
        mask = masks_np[idx]
        score = float(scores_np[idx])

        mask_bool = mask > 0.5
        if not mask_bool.any():
            continue

        color = random_color()
        color_str = f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"

        image_np[mask_bool] = (
            (1.0 - alpha) * image_np[mask_bool] + alpha * color
        )

        area_pixels = int(mask_bool.sum())

        objects.append({
            "id": idx,
            "area": area_pixels,
            "score": round(score, 3),
            "color": color_str,
        })

    overlay_pil = Image.fromarray(image_np.astype(np.uint8))
    return overlay_pil, objects


def save_uploaded_image(image_pil):
    """
    Guarda la imagen original en data/images y devuelve el nombre de archivo.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    img_id = str(uuid.uuid4())
    filename = f"{img_id}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    image_pil.save(path, format="JPEG")
    return filename


def build_sessions_table():
    """
    Devuelve info resumida de todas las sesiones registradas.
    """
    sessions = list_sessions()
    rows = []
    for s in sessions:
        sid = s.get("session_id", "")
        img = s.get("image_filename", "")
        cc = s.get("classes_counts", {})
        cc_str = ", ".join(f"{k}: {v}" for k, v in cc.items()) if cc else "-"
        rows.append({
            "session_id": sid,
            "image_filename": img,
            "classes_counts": cc,
            "classes_counts_str": cc_str,
        })
    return rows


# ---------- ENDPOINT PRINCIPAL DE SEGMENTACIÓN ----------

@app.post("/api/segment")
async def api_segment(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.5),
    session_id: str = Form(""),
    force_new_session: bool = Form(False),
):
    """
    Recibe:
    - file: imagen (UploadFile)
    - prompt: texto
    - threshold: score mínimo
    - session_id: opcional
    - force_new_session: si True, ignora session_id y crea una nueva

    Devuelve JSON con:
    - session_id
    - overlay_image_b64
    - objects: [{id, area, score, color}, ...]
    - class_counts: {clase: total}
    - summary: texto
    - sessions_history: listado de sesiones (para panel de historial)
    """
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    # Manejo de sesión
    if not session_id or force_new_session:
        # Nueva sesión
        img_filename = save_uploaded_image(image_pil)
        session_data = create_session(image_filename=img_filename)
    else:
        # Reutilizar
        try:
            session_data = load_session(session_id)
            img_filename = session_data["image_filename"]
        except FileNotFoundError:
            img_filename = save_uploaded_image(image_pil)
            session_data = create_session(image_filename=img_filename)

    session_id = session_data["session_id"]

    # Ejecutar SAM3
    masks_np, scores_np = sam3_engine.segment_image(
        image_pil=image_pil,
        text_prompt=prompt,
        score_threshold=threshold
    )

    if masks_np is None or masks_np.shape[0] == 0:
        summary = f'Prompt "{prompt}" -> 0 objetos detectados (threshold={threshold:.2f}).'
        return {
            "session_id": session_id,
            "overlay_image_b64": None,
            "objects": [],
            "class_counts": session_data["classes_counts"],
            "summary": summary,
            "sessions_history": build_sessions_table(),
        }

    # Colorear instancias
    overlay_pil, objects = overlay_masks_on_image(
        image_pil=image_pil,
        masks_np=masks_np,
        scores_np=scores_np
    )

    total_instances = len(objects)

    # Actualizar sesión
    segment_entry = {
        "class_name": prompt.strip(),
        "threshold": float(threshold),
        "num_objects": int(total_instances),
    }
    session_data["segments"].append(segment_entry)

    cls = prompt.strip()
    session_data["classes_counts"][cls] = session_data["classes_counts"].get(cls, 0) + int(total_instances)

    save_session(session_data)

    # Serializar imagen overlay a base64
    buf = io.BytesIO()
    overlay_pil.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    overlay_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Resumen de texto
    summary_lines = [
        f"Session ID: {session_id}",
        f"Imagen: {img_filename}",
        f'Prompt: "{prompt}"',
        f"Threshold: {threshold:.2f}",
        f"Objetos detectados (esta corrida): {total_instances}",
        "",
        "Conteo acumulado por clase en esta sesión:",
    ]
    for k, v in session_data["classes_counts"].items():
        summary_lines.append(f"- {k}: {v} piezas")

    summary = "\n".join(summary_lines)

    return {
        "session_id": session_id,
        "overlay_image_b64": overlay_b64,
        "objects": objects,
        "class_counts": session_data["classes_counts"],
        "summary": summary,
        "sessions_history": build_sessions_table(),
    }


@app.get("/api/sessions")
def api_sessions():
    """
    Devuelve el historial resumido de sesiones.
    """
    return build_sessions_table()
