# session_store.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid


@dataclass
class SegmentResult:
    class_name: str
    threshold: float
    num_objects: int


class SessionStore:
    """
    Maneja sesiones de análisis:
    - Guarda metadata en data/sessions/*.json
    - Guarda nombres de imagen en data/images/ (el guardado del archivo lo hace api_server)
    """

    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.sessions_dir = self.base_dir / "sessions"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def create_session(self, image_filename: str) -> Dict[str, Any]:
        """
        Crea sesión y devuelve el dict completo (NO string) para evitar inconsistencias.
        """
        session_id = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat()

        session_data: Dict[str, Any] = {
            "session_id": session_id,
            "image_filename": image_filename,
            "created_at": now_iso,
            "segments": [],
            "classes_counts": {},
        }

        with self._session_path(session_id).open("w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def append_segment(
        self,
        session_id: str,
        segment: SegmentResult,
    ) -> Dict[str, Any]:
        """
        Agrega un resultado a una sesión existente y actualiza conteo por clase.
        Devuelve sesión actualizada.
        """
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Sesión {session_id} no encontrada")

        with path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        data.setdefault("segments", []).append(asdict(segment))

        classes_counts: Dict[str, int] = data.setdefault("classes_counts", {})
        cls = segment.class_name
        classes_counts[cls] = classes_counts.get(cls, 0) + int(segment.num_objects)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data

    # ---- Atajo compatible con tu api_server.py (la llamada “por kwargs”) ----
    def append_segment_by_fields(
        self,
        image_filename: str,
        class_name: str,
        threshold: float,
        num_objects: int,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compat: permite al api_server llamar sin construir SegmentResult manualmente.
        - Si session_id viene None -> crea sesión.
        - Si session_id viene -> actualiza esa sesión.
        """
        if session_id is None:
            session_data = self.create_session(image_filename)
            session_id = session_data["session_id"]
        else:
            session_data = self.get_session(session_id)
            if session_data is None:
                session_data = self.create_session(image_filename)
                session_id = session_data["session_id"]

        seg = SegmentResult(
            class_name=class_name,
            threshold=float(threshold),
            num_objects=int(num_objects),
        )
        return self.append_segment(session_id=session_id, segment=seg)

    def list_sessions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fp in self.sessions_dir.glob("*.json"):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
        return out
