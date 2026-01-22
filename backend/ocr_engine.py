# backend/ocr_engine.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# OpenCV para prepro
import cv2

# OCR
import easyocr

# Matching
from rapidfuzz import process, fuzz


CODE_REGEX = re.compile(r"[A-Z0-9]+(?:[-_][A-Z0-9]+)+", re.IGNORECASE)

def _normalize(s: str) -> str:
    s = s.strip().upper()
    # Normalizaciones típicas: espacios -> nada
    s = re.sub(r"\s+", "", s)
    # Homogeneizar guiones raros
    s = s.replace("—", "-").replace("–", "-")
    return s


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Prepro robusto para metal: contraste + denoise + binarización."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Aumentar contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise ligero
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Binarización adaptativa (funciona bien con sombras)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    return Image.fromarray(thr)


@dataclass
class OcrItemResult:
    filename: str
    raw_text: str
    candidates: List[str]
    matched_key: Optional[str]
    match_score: Optional[float]


class OcrEngine:
    def __init__(self, languages: List[str] | None = None) -> None:
        self.reader = easyocr.Reader(languages or ["en"], gpu=False)
        self.known_keys: List[str] = []

    def set_known_keys(self, keys: List[str]) -> None:
        self.known_keys = [_normalize(k) for k in keys if str(k).strip()]

    def run_ocr(self, pil_img: Image.Image) -> str:
        # EasyOCR trabaja con numpy
        img = np.array(pil_img.convert("RGB"))
        chunks = self.reader.readtext(img, detail=0)  # solo strings
        return "\n".join(chunks)

    def extract_candidates(self, raw_text: str) -> List[str]:
        raw_text_n = _normalize(raw_text)
        found = CODE_REGEX.findall(raw_text_n)
        # dedupe preservando orden
        out = []
        seen = set()
        for f in found:
            f = _normalize(f)
            if f not in seen:
                out.append(f)
                seen.add(f)
        return out

    def match_best(self, candidates: List[str]) -> Tuple[Optional[str], Optional[float]]:
        if not self.known_keys or not candidates:
            return None, None

        # Probamos el mejor match de cualquier candidato contra known_keys
        best_key = None
        best_score = -1.0

        for c in candidates:
            hit = process.extractOne(
                query=c,
                choices=self.known_keys,
                scorer=fuzz.WRatio
            )
            if hit:
                key, score, _ = hit
                if score > best_score:
                    best_score = float(score)
                    best_key = key

        return best_key, best_score

    def process_image(
        self,
        filename: str,
        pil_img: Image.Image,
        use_preprocess: bool = True
    ) -> OcrItemResult:
        img = preprocess_for_ocr(pil_img) if use_preprocess else pil_img
        raw = self.run_ocr(img)
        candidates = self.extract_candidates(raw)
        mk, ms = self.match_best(candidates)
        return OcrItemResult(
            filename=filename,
            raw_text=raw,
            candidates=candidates,
            matched_key=mk,
            match_score=ms
        )
