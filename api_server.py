# api_server.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Literal
from io import BytesIO
import base64
import uuid
import re

import numpy as np
from PIL import Image

from sam3_engine import Sam3Engine
from backend.reconstruction.routes import router as reconstruction_router

# OCR
import cv2
import easyocr

# Optional external postprocess module (won't break if missing)
_HAS_OCR_POSTPROCESS = False
try:
    from ocr_postprocess import pick_best_codes, stitch_common_patterns, clean_token  # type: ignore
    _HAS_OCR_POSTPROCESS = True
except Exception:
    pick_best_codes = None  # type: ignore
    stitch_common_patterns = None  # type: ignore
    clean_token = None  # type: ignore


app = FastAPI(title="SAM3 GP RD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register reconstruction routes (reconstruct-fast)
app.include_router(reconstruction_router)

engine = Sam3Engine()

# OCR reader (1 sola vez)
ocr_reader = easyocr.Reader(["en"], gpu=False)


# ==========================
# SAM3 MODELS
# ==========================
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

    # Para UI moderna: overlay separado (RGBA) para dibujar ENCIMA de la original
    overlay_rgba_b64: str = ""  # PNG RGBA, fondo transparente

    # Compat / legacy: overlay ya “horneado” sobre la original
    overlay_image_b64: str = ""  # PNG RGB

    id_map_b64: str  # PNG 16-bit: pixel = instance_id (0=fondo)
    id_map_rgb_b64: str   # <--- NUEVO

    # Conteos + labels
    classes_counts: Dict[str, int]
    labels: List[InstanceLabel]




class CompareSegmentResponse(BaseModel):
    session_id: str
    threshold: float
    prompt: str
    left: SegmentResponse
    right: SegmentResponse


def _deterministic_color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(1000 + i)
    c = rng.integers(low=60, high=240, size=(3,), endpoint=True)
    return int(c[0]), int(c[1]), int(c[2])


def _encode_png_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_id_map_16bit_b64(id_map_u16: np.ndarray) -> str:
    if id_map_u16.dtype != np.uint16:
        id_map_u16 = id_map_u16.astype(np.uint16)
    pil = Image.fromarray(id_map_u16, mode="I;16")
    return _encode_png_b64(pil)


def _encode_id_map_rgb_b64(id_map_u16: np.ndarray) -> str:
    # id_map_u16: uint16 (0 fondo, >0 instancias)
    if id_map_u16.dtype != np.uint16:
        id_map_u16 = id_map_u16.astype(np.uint16)

    h, w = id_map_u16.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (id_map_u16 & 0xFF).astype(np.uint8)
    rgb[..., 1] = ((id_map_u16 >> 8) & 0xFF).astype(np.uint8)
    rgb[..., 2] = ((id_map_u16 >> 16) & 0xFF).astype(np.uint8)

    pil = Image.fromarray(rgb, mode="RGB")
    return _encode_png_b64(pil)



def _overlay_and_maps(
    image_pil: Image.Image,
    per_prompt_results: List[dict],
    alpha: float = 0.55,
):
    """
    Retorna:
    - overlay_rgb_b64: overlay “horneado” (RGB)
    - overlay_rgba_b64: overlay transparente (RGBA) para dibujar sobre la original
    - id_map_b64: 16-bit
    - labels
    - classes_counts
    """
    img_np = np.array(image_pil.convert("RGB"), copy=True)
    h, w, _ = img_np.shape

    # overlay RGBA (transparente)
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)

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

            id_map[mask_bool] = inst_id

            r, g, b = _deterministic_color(inst_id)

            # 1) Overlay “horneado” (legacy)
            img_np[mask_bool] = (
                alpha * np.array([r, g, b], dtype=np.float32)
                + (1 - alpha) * img_np[mask_bool].astype(np.float32)
            ).astype(np.uint8)

            # 2) Overlay RGBA transparente (nuevo)
            overlay_rgba[mask_bool, 0] = r
            overlay_rgba[mask_bool, 1] = g
            overlay_rgba[mask_bool, 2] = b
            overlay_rgba[mask_bool, 3] = int(max(0, min(255, round(alpha * 255))))

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

    overlay_rgb_b64 = _encode_png_b64(Image.fromarray(img_np))
    overlay_rgba_b64 = _encode_png_b64(Image.fromarray(overlay_rgba, mode="RGBA"))
    id_map_b64 = _encode_id_map_16bit_b64(id_map)
    id_map_rgb_b64 = _encode_id_map_rgb_b64(id_map)

    labels.sort(key=lambda x: (x.class_name, x.id))
    return overlay_rgb_b64, id_map_b64, id_map_rgb_b64, labels, classes_counts



def _segment_core(image_pil: Image.Image, prompt_csv: str, threshold: float) -> SegmentResponse:
    prompts = [p.strip() for p in (prompt_csv or "").split(",") if p.strip()]
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

    overlay_rgb_b64, id_map_b64, id_map_rgb_b64, labels, classes_counts = _overlay_and_maps(
        image_pil=image_pil,
        per_prompt_results=per_prompt_results,
        alpha=0.55,
    )
    return SegmentResponse(
        session_id=session_id,
        threshold=threshold,
        overlay_rgba_b64="",
        overlay_image_b64=overlay_rgb_b64 if labels else "",
        id_map_b64=id_map_b64,
        id_map_rgb_b64=id_map_rgb_b64,
        classes_counts=classes_counts,
        labels=labels,
        class_name=",".join([p.strip().lower() for p in prompts]),
        num_objects=len(labels),
    )


# ==========================
# SAM3 endpoint (SAMVIEWER)
# ==========================
@app.post("/api/segment", response_model=SegmentResponse)
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.5),
):
    content = await file.read()
    image_pil = Image.open(BytesIO(content)).convert("RGB")
    return _segment_core(image_pil=image_pil, prompt_csv=prompt, threshold=threshold)


# ==========================
# SAM3 COMPARE endpoint (2 imágenes)
# ==========================
@app.post("/api/segment-compare", response_model=CompareSegmentResponse)
async def segment_compare(
    file_left: UploadFile = File(...),
    file_right: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.5),
):
    content_l = await file_left.read()
    content_r = await file_right.read()

    img_l = Image.open(BytesIO(content_l)).convert("RGB")
    img_r = Image.open(BytesIO(content_r)).convert("RGB")

    left = _segment_core(image_pil=img_l, prompt_csv=prompt, threshold=threshold)
    right = _segment_core(image_pil=img_r, prompt_csv=prompt, threshold=threshold)

    return CompareSegmentResponse(
        session_id=str(uuid.uuid4()),
        threshold=threshold,
        prompt=prompt,
        left=left,
        right=right,
    )


# ==========================
# OCR helpers (pipeline fuerte para claves en metal)
# ==========================
def _bgr_to_b64png(bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return _encode_png_b64(pil)


def _normalize_separators(s: str) -> str:
    s = (s or "")
    s = s.replace("—", "-").replace("–", "-").replace("•", "-").replace("·", "-")
    s = s.replace(":", "-").replace(".", "-")  # '.' puede significar '-'
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_token(s: str) -> str:
    s = (s or "").upper()
    s = _normalize_separators(s).upper()
    s = re.sub(r"[^A-Z0-9_-]", "", s)
    return s.strip("-_")


# If external ocr_postprocess not found, bind names to internal versions
if clean_token is None:
    clean_token = _clean_token  # type: ignore


_CODE_RE = re.compile(
    r"\b(?=[A-Z0-9_-]{4,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+(?:[-_][A-Z0-9]+){1,4}\b"
)


def _stitch_common_patterns(txt: str) -> str:
    t = _normalize_separators(txt).upper()
    t = re.sub(r"\b([A-Z]+\d)\s+([A-Z]+\d)\s+(\d{1,3})\b", r"\1-\2-\3", t)
    t = re.sub(r"\b([A-Z]+\d)\s+([A-Z0-9]{2,6})\s+(\d{1,3})\b", r"\1-\2-\3", t)
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"\s*_\s*", "_", t)
    return t


if stitch_common_patterns is None:
    stitch_common_patterns = _stitch_common_patterns  # type: ignore


def _extract_codes(raw_text: str) -> List[str]:
    txt = _stitch_common_patterns(raw_text or "")
    found = _CODE_RE.findall(txt)
    out, seen = [], set()
    for c in found:
        c = _clean_token(c)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _bbox_center_x(bbox) -> float:
    pts = np.array(bbox, dtype=np.float32)
    return float(np.mean(pts[:, 0]))


def _assemble_left_to_right(dets: list) -> str:
    toks = []
    dets_sorted = sorted(dets, key=lambda d: _bbox_center_x(d[0]) if d and d[0] is not None else 1e9)
    for (bbox, text, conf) in dets_sorted:
        if not isinstance(text, str):
            continue
        t = _clean_token(text)
        if not t:
            continue
        toks.append(t)
    return " ".join(toks).strip()


def _union_boxes(dets: list, w: int, h: int, pad_frac: float = 0.10) -> Optional[Tuple[int, int, int, int]]:
    xs, ys = [], []
    for (bbox, text, conf) in dets:
        if not bbox:
            continue
        if conf is None or float(conf) < 0.20:
            continue
        pts = np.array(bbox, dtype=np.float32)
        xs.extend(list(pts[:, 0]))
        ys.extend(list(pts[:, 1]))

    if not xs or not ys:
        return None

    x0 = int(max(0, min(xs)))
    y0 = int(max(0, min(ys)))
    x1 = int(min(w - 1, max(xs)))
    y1 = int(min(h - 1, max(ys)))

    pad_x = int((x1 - x0 + 1) * pad_frac)
    pad_y = int((y1 - y0 + 1) * pad_frac)

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return (x0, y0, x1, y1)


def _estimate_text_angle(gray: np.ndarray) -> float:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(mag, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        length = float(np.hypot(dx, dy))
        angles.append(ang)
        weights.append(length)

    if not angles:
        return 0.0

    angles = np.array(angles, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    order = np.argsort(angles)
    angles_s = angles[order]
    weights_s = weights[order]
    cdf = np.cumsum(weights_s) / (weights_s.sum() + 1e-6)
    idx = int(np.searchsorted(cdf, 0.5))
    dominant = float(angles_s[min(max(idx, 0), len(angles_s) - 1)])

    if abs(dominant) < 1.0 or abs(dominant) > 35.0:
        return 0.0
    return -dominant


def _rotate_affine(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-3:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)
    return cv2.warpAffine(bgr, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _add_white_frame(bgr: np.ndarray, pad_frac: float = 0.10) -> np.ndarray:
    h, w = bgr.shape[:2]
    pad = int(max(h, w) * pad_frac)
    out = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    out[pad:pad + h, pad:pad + w] = bgr
    return out


def _preprocess_variants_for_ocr(bgr: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    out.append(g1)

    # --- (A) realce de relieve: black-hat (muy útil en grabado) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(g1, cv2.MORPH_BLACKHAT, kernel)
    out.append(blackhat)

    # --- (B) normalización de iluminación: divide by blur (barato y efectivo) ---
    blur_big = cv2.GaussianBlur(g1, (0, 0), 9.0)
    # evita división por 0
    norm = cv2.divide(g1, blur_big + 1, scale=255)
    out.append(norm)

    blur = cv2.GaussianBlur(g1, (0, 0), 1.2)
    unsharp = cv2.addWeighted(g1, 1.8, blur, -0.8, 0)
    out.append(unsharp)

    up = cv2.resize(unsharp, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    out.append(up)

    gx = cv2.Sobel(unsharp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(unsharp, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out.append(mag)

    mag_up = cv2.resize(mag, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    out.append(mag_up)

    thr = cv2.adaptiveThreshold(
        unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    out.append(thr)
    out.append(255 - thr)

    return out


def _pattern_score_from_text(text_lr: str) -> float:
    txt = _stitch_common_patterns(text_lr or "")
    codes = _CODE_RE.findall(txt)
    if not codes:
        return 0.0

    score = 0.0
    for c in codes:
        cc = _clean_token(c)
        if not cc:
            continue
        score += 3.0
        if cc.count("-") >= 2:
            score += 1.0
        if "HN" in cc:
            score += 0.5
        if re.search(r"[A-Z]+\d", cc):
            score += 0.5
    return score


def _run_easyocr(proc_img: np.ndarray) -> list:
    return ocr_reader.readtext(
        proc_img,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.",
        text_threshold=0.6,
        low_text=0.35,
        link_threshold=0.4,
        contrast_ths=0.1,
        adjust_contrast=0.6,
    )


def _apply_orientation(bgr: np.ndarray, k_rot90: int, flip: bool) -> np.ndarray:
    out = np.ascontiguousarray(np.rot90(bgr, k=k_rot90))
    if flip:
        out = cv2.flip(out, 1)
    return out


OcrStatus = Literal["OK", "DUDOSO", "RECHAZADO"]


def _aggregate_confidence(dets: list) -> float:
    """
    Confianza agregada [0..1] usando detections EasyOCR.
    Pondera por longitud de token limpio y por conf.
    """
    if not dets:
        return 0.0

    total_w = 0.0
    total = 0.0
    for (bbox, text, conf) in dets:
        try:
            c = float(conf)
        except Exception:
            continue
        t = clean_token(text)  # type: ignore
        if not t:
            continue
        # peso por longitud del token (capado)
        w = float(min(6, max(1, len(t))))
        total += c * w
        total_w += w

    if total_w <= 0:
        return 0.0

    # promedio ponderado
    return max(0.0, min(1.0, total / total_w))


def _ambiguity_penalty(dets: list) -> float:
    """
    Penaliza ambigüedad típica en grabado:
    - demasiados tokens (ruido)
    - tokens muy parecidos repetidos
    Devuelve penalización [0..0.35]
    """
    if not dets:
        return 0.35

    toks = []
    for (_, text, conf) in dets:
        t = clean_token(text)  # type: ignore
        if t:
            toks.append(t)

    if not toks:
        return 0.35

    # ruido por demasiados tokens
    n = len(toks)
    p = 0.0
    if n >= 10:
        p += 0.20
    elif n >= 7:
        p += 0.12
    elif n >= 5:
        p += 0.07

    # repetición exacta
    uniq = len(set(toks))
    if uniq <= max(1, n // 2):
        p += 0.10

    # penaliza confusiones típicas en posiciones críticas
    joined = "".join(toks)
    if any(ch in joined for ch in ["O", "0", "I", "1", "L"]):
        p += 0.05

    return max(0.0, min(0.35, p))


def _pattern_bonus(codes: list[str]) -> float:
    """
    Bonus por tener 1+ claves válidas.
    Devuelve [0..0.25]
    """
    if not codes:
        return 0.0
    # más bonus si hay separadores (familias tipo H-IC-6-0)
    c0 = codes[0]
    seps = c0.count("-") + c0.count("_")
    b = 0.12 + min(0.13, 0.05 * seps)
    return max(0.0, min(0.25, b))


def _final_conf_and_status(dets: list, codes: list[str]) -> tuple[float, OcrStatus]:
    """
    Conf final: base_conf + bonus - penalty, clamped.
    Status por umbrales.
    """
    base = _aggregate_confidence(dets)
    bonus = _pattern_bonus(codes)
    pen = _ambiguity_penalty(dets)

    conf = base + bonus - pen
    conf = max(0.0, min(1.0, conf))

    # Umbrales recomendados (ajustables)
    if codes and conf >= 0.78:
        return conf, "OK"
    if codes and conf >= 0.60:
        return conf, "DUDOSO"
    # si no hay codes pero hay buena conf, igual dudoso (texto parcial)
    if (not codes) and conf >= 0.70:
        return conf, "DUDOSO"
    return conf, "RECHAZADO"


def _best_ocr_on_bgr(bgr_in: np.ndarray) -> dict:
    best_global = {
        "score": -1.0,
        "dets": [],
        "text_lr": "",
        "raw_text": "",
        "codes": [],
        "preview_bgr": bgr_in,
        "roi_xyxy": [0, 0, bgr_in.shape[1] - 1, bgr_in.shape[0] - 1],
        "orientation": {"rot90_k": 0, "flip_h": False, "skew_deg": 0.0},
    }

    orientations = [(k, f) for k in [0, 1, 2, 3] for f in [False, True]]

    for k_rot90, flip_h in orientations:
        bgr = _apply_orientation(bgr_in, k_rot90, flip_h)
        bgr_canvas = _add_white_frame(bgr, pad_frac=0.10)

        base_gray = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = clahe.apply(base_gray)

        dets0 = _run_easyocr(base)
        h0, w0 = bgr_canvas.shape[:2]
        roi = _union_boxes(dets0, w0, h0, pad_frac=0.15)
        if roi is None:
            roi = (0, 0, w0 - 1, h0 - 1)

        x0, y0, x1, y1 = roi
        bgr_roi0 = bgr_canvas[y0:y1 + 1, x0:x1 + 1].copy()

        gray_roi0 = cv2.cvtColor(bgr_roi0, cv2.COLOR_BGR2GRAY)
        skew = _estimate_text_angle(gray_roi0)
        bgr_roi = _rotate_affine(bgr_roi0, angle_deg=skew)

        best_local = {"score": -1.0, "dets": [], "text_lr": "", "raw_text": "", "preview_bgr": bgr_roi}

        for proc in _preprocess_variants_for_ocr(bgr_roi):
            dets = _run_easyocr(proc)
            text_lr = _assemble_left_to_right(dets)
            raw_text = " ".join([d[1] for d in dets if isinstance(d[1], str)])

            sc = _pattern_score_from_text(text_lr)
            if sc <= 0:
                conf_sum = sum([float(d[2]) for d in dets if len(d) >= 3])
                sc = conf_sum * 0.10

            if sc > best_local["score"]:
                best_local = {
                    "score": float(sc),
                    "dets": dets,
                    "text_lr": text_lr,
                    "raw_text": raw_text,
                    "preview_bgr": bgr_roi,
                }

        text_lr_norm = _stitch_common_patterns(best_local["text_lr"] or "")
        codes_base = _extract_codes(text_lr_norm)

        pp_debug = None
        if _HAS_OCR_POSTPROCESS and pick_best_codes is not None:
            try:
                det_tokens_lr = [
                    clean_token(d[1]) for d in sorted(best_local["dets"], key=lambda d: _bbox_center_x(d[0]))
                ]  # type: ignore
                codes_pp, pp_debug = pick_best_codes(text_lr_norm, det_tokens_lr=det_tokens_lr)  # type: ignore
                codes = codes_pp or codes_base
            except Exception:
                codes = codes_base
        else:
            codes = codes_base

        final_score = best_local["score"] + (10.0 if len(codes) > 0 else 0.0)

        if final_score > best_global["score"]:
            best_global = {
                "score": float(final_score),
                "dets": best_local["dets"],
                "text_lr": text_lr_norm,
                "raw_text": _stitch_common_patterns(best_local["raw_text"] or ""),
                "codes": codes,
                "preview_bgr": best_local["preview_bgr"],
                "roi_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "orientation": {"rot90_k": int(k_rot90), "flip_h": bool(flip_h), "skew_deg": float(skew)},
            }
            if pp_debug is not None:
                best_global["postprocess_debug"] = pp_debug

    return best_global


# ==========================
# OCR Batch (claves de montaje)
# ==========================
@app.post("/api/ocr-batch")
async def ocr_batch(images: List[UploadFile] = File(...)):
    if len(images) > 10:
        images = images[:10]

    items = []
    global_codes: List[str] = []

    for f in images:
        content = await f.read()
        pil = Image.open(BytesIO(content)).convert("RGB")
        bgr0 = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        best = _best_ocr_on_bgr(bgr0)

        preview = best["preview_bgr"].copy()
        for (bbox, text, conf) in best["dets"]:
            t = clean_token(text)  # type: ignore
            if not t:
                continue
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(preview, [pts], True, (0, 255, 0), 3)
            cv2.putText(
                preview,
                f"{t} ({float(conf):.2f})",
                (pts[0][0], max(0, pts[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        preview_b64 = _bgr_to_b64png(preview)

        det_json = []
        for (bbox, text, conf) in best["dets"]:
            det_json.append(
                {
                    "text": text,
                    "clean": clean_token(text),  # type: ignore
                    "conf": float(conf),
                    "bbox": [[float(p[0]), float(p[1])] for p in bbox],
                }
            )

        codes = best["codes"] or []
        global_codes.extend(codes)

        final_conf, status = _final_conf_and_status(best["dets"], codes)

        debug_obj = {
            "roi_xyxy_on_canvas": best.get("roi_xyxy"),
            "orientation": best.get("orientation"),
            "final_score": best.get("score"),
        }
        if "postprocess_debug" in best:
            debug_obj["postprocess"] = best["postprocess_debug"]

        items.append(
            {
                "filename": f.filename,
                "raw_text": best["text_lr"] or best["raw_text"] or "",
                "codes": codes,
                "confidence": final_conf,
                "status": status,
                "detections": det_json,
                "preview_b64": preview_b64,
                "debug": debug_obj,
            }
        )

    seen = set()
    unique_codes = []
    for c in global_codes:
        if c not in seen:
            seen.add(c)
            unique_codes.append(c)

    return {"items": items, "unique_codes": unique_codes}
