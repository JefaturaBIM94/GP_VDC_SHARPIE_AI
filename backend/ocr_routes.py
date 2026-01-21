# backend/ocr_routes.py
from __future__ import annotations

import io
from typing import List, Dict, Any

import pandas as pd
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from backend.ocr_engine import OcrEngine


router = APIRouter(prefix="/api", tags=["ocr"])
ocr_engine = OcrEngine(languages=["en"])  # códigos alfanuméricos


@router.post("/keys/upload")
async def upload_keys(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    # Lee excel desde bytes
    df = pd.read_excel(io.BytesIO(content))
    # Heurística: tomar primera columna no vacía
    col = df.columns[0]
    keys = df[col].dropna().astype(str).tolist()
    ocr_engine.set_known_keys(keys)
    return {"count": len(keys), "column_used": str(col)}


@router.post("/ocr/batch")
async def ocr_batch(
    files: List[UploadFile] = File(...),
    use_preprocess: bool = Form(True),
) -> Dict[str, Any]:
    if len(files) > 10:
        return {"error": "Máximo 10 imágenes por batch en este POC."}

    items = []
    for f in files:
        content = await f.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
        res = ocr_engine.process_image(
            filename=f.filename or "image",
            pil_img=pil,
            use_preprocess=bool(use_preprocess),
        )
        items.append({
            "filename": res.filename,
            "raw_text": res.raw_text,
            "candidates": res.candidates,
            "matched_key": res.matched_key,
            "match_score": res.match_score,
        })

    return {"items": items}


@router.post("/ocr/export/xlsx")
async def export_xlsx(payload: Dict[str, Any]) -> StreamingResponse:
    items = payload.get("items", [])
    df = pd.DataFrame(items)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="OCR")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="ocr_concentrado.xlsx"'},
    )
