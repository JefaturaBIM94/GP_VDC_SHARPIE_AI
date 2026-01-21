# ocr_postprocess.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any

# Regex de clave: requiere letras y dígitos, y 2..5 grupos con separador - o _
CODE_RE = re.compile(
    r"\b(?=[A-Z0-9_-]{4,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+(?:[-_][A-Z0-9]+){1,4}\b"
)

# Confusiones típicas OCR en grabado metálico (solo aplicaremos si ayuda)
# Nota: NO son reemplazos "a ciegas"; se prueban variantes y se elige la mejor por score.
CHAR_ALTS: Dict[str, str] = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "L": "1",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
    "Z": "2",
    "2": "Z",
}

# Tokens frecuentes que son basura o “muy cortos” pero aparecen en dets
STOP_TOKENS = {"", "-", "_", "--", "__"}


def normalize_separators(s: str) -> str:
    s = (s or "")
    s = s.replace("—", "-").replace("–", "-").replace("•", "-").replace("·", "-")
    s = s.replace(":", "-").replace(".", "-")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*_\s*", "_", s)
    return s


def clean_token(s: str) -> str:
    s = (s or "").upper()
    s = normalize_separators(s).upper()
    s = re.sub(r"[^A-Z0-9_-]", "", s)
    return s.strip("-_")


def stitch_common_patterns(txt: str) -> str:
    """
    Re-ensambla casos típicos donde OCR separa por espacios:
    HN1 C1 12  -> HN1-C1-12
    HN1 13T8 10 -> HN1-13T8-10
    """
    t = normalize_separators(txt).upper()

    t = re.sub(r"\b([A-Z]+\d)\s+([A-Z]+\d)\s+(\d{1,3})\b", r"\1-\2-\3", t)
    t = re.sub(r"\b([A-Z]+\d)\s+([A-Z0-9]{2,6})\s+(\d{1,3})\b", r"\1-\2-\3", t)

    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"\s*_\s*", "_", t)
    return t


def extract_codes(text: str) -> List[str]:
    t = stitch_common_patterns(text or "")
    found = CODE_RE.findall(t)
    out: List[str] = []
    seen = set()
    for c in found:
        cc = clean_token(c)
        if cc and cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out


def score_code_like(s: str) -> float:
    """
    Score de “parece clave”:
    - Debe tener letras y dígitos
    - Premia tener 2+ separadores
    - Premia prefijo HN (puedes ajustar a tus familias)
    """
    s = clean_token(s)
    if not s or s in STOP_TOKENS:
        return 0.0
    if not any(ch.isalpha() for ch in s):
        return 0.0
    if not any(ch.isdigit() for ch in s):
        return 0.0

    score = 1.0
    score += 0.8 * min(4, s.count("-") + s.count("_"))
    if "HN" in s:
        score += 0.7
    if re.search(r"[A-Z]+\d", s):
        score += 0.5
    if CODE_RE.search(s):
        score += 2.5
    return score


def generate_variants(token: str, max_variants: int = 24) -> List[str]:
    """
    Genera variantes aplicando sustituciones OCR típicas en pocos caracteres.
    Controlamos el crecimiento combinatorio con max_variants.
    """
    token = clean_token(token)
    if not token:
        return []

    variants = {token}

    # Encuentra posiciones donde hay ambigüedad
    amb_positions = [i for i, ch in enumerate(token) if ch in CHAR_ALTS]

    # BFS limitado: cambia 1 o 2 caracteres ambiguos (suficiente en práctica)
    for i in amb_positions:
        new_set = set(variants)
        for v in variants:
            nv = list(v)
            nv[i] = CHAR_ALTS[nv[i]]
            new_set.add("".join(nv))
        variants = new_set
        if len(variants) >= max_variants:
            break

    # Opcional: una pasada extra para permitir 2 cambios
    if len(variants) < max_variants:
        for i in amb_positions:
            for j in amb_positions:
                if i >= j:
                    continue
                new_set = set(variants)
                for v in list(variants)[: max(1, max_variants // 2)]:
                    nv = list(v)
                    nv[i] = CHAR_ALTS.get(nv[i], nv[i])
                    nv[j] = CHAR_ALTS.get(nv[j], nv[j])
                    new_set.add("".join(nv))
                variants = new_set
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break

    # Normaliza separadores por si acaso
    return [clean_token(v) for v in variants if clean_token(v)]


def pick_best_codes(raw_text: str, det_tokens_lr: List[str] | None = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Entrada:
      - raw_text: texto “crudo” (preferible LR ya armado)
      - det_tokens_lr: tokens individuales (izq->der) si los tienes (opcional)
    Salida:
      - codes: lista final de claves (limpias)
      - debug: info para analizar por qué se eligió eso
    """
    base_text = stitch_common_patterns(raw_text or "")
    base_codes = extract_codes(base_text)

    best_codes = base_codes
    best_score = sum(score_code_like(c) for c in best_codes)

    debug: Dict[str, Any] = {
        "base_text": base_text,
        "base_codes": base_codes,
        "base_score": best_score,
        "tried": [],
        "chosen_score": best_score,
    }

    # Si ya sacamos códigos buenos, normalmente no toques.
    # Pero si está vacío o score muy bajo, intentamos variantes.
    need_search = (len(best_codes) == 0) or (best_score < 3.0)

    # Armamos candidatos: tokens del texto LR o del texto completo
    tokens: List[str] = []
    if det_tokens_lr:
        tokens = [clean_token(t) for t in det_tokens_lr if clean_token(t)]
    else:
        tokens = [clean_token(t) for t in re.split(r"\s+", base_text) if clean_token(t)]

    tokens = [t for t in tokens if t not in STOP_TOKENS]

    # Si no vamos a buscar, igual devolvemos base.
    if not need_search or not tokens:
        return best_codes, debug

    # Estrategia:
    # 1) generar variantes por token
    # 2) re-combinar 2..4 tokens con '-' (porque OCR separa)
    # 3) extraer códigos y quedarnos con el mejor score
    token_variants: List[List[str]] = []
    for t in tokens[:8]:  # limitamos para performance
        token_variants.append(generate_variants(t))

    # Candidatos de string
    candidates_texts = set()
    candidates_texts.add(base_text)

    # Combina tokens contiguos (izq->der) en ventanas
    flat_tokens = [tv[0] if tv else "" for tv in token_variants]  # token "base" limpio
    n = len(flat_tokens)

    for win in [2, 3, 4]:
        for i in range(0, max(0, n - win + 1)):
            chunk = [flat_tokens[i + k] for k in range(win)]
            if any(not c for c in chunk):
                continue
            candidates_texts.add("-".join(chunk))

    # Además prueba sustituciones en el “texto completo”
    # Cambiando algunos tokens por sus variantes top
    for idx, vars_ in enumerate(token_variants):
        for v in vars_[:6]:
            tmp = flat_tokens[:]
            if idx < len(tmp):
                tmp[idx] = v
            # crea 2 combinaciones: con espacio y con guiones
            candidates_texts.add(" ".join(tmp))
            candidates_texts.add("-".join(tmp))

    # Evalúa candidatos
    for cand in list(candidates_texts)[:120]:
        codes = extract_codes(cand)
        sc = sum(score_code_like(c) for c in codes)
        debug["tried"].append({"cand": cand, "codes": codes, "score": sc})
        if sc > best_score:
            best_score = sc
            best_codes = codes
            debug["chosen_score"] = sc

    # Quita duplicados preservando orden
    seen = set()
    final_codes = []
    for c in best_codes:
        if c not in seen:
            seen.add(c)
            final_codes.append(c)

    return final_codes, debug
