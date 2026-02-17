# backend/video_store.py
from __future__ import annotations
import os
import uuid
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import cv2


@dataclass
class VideoSession:
    session_id: str
    video_path: str
    frames_dir: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float

    def frame_path(self, idx: int) -> str:
        return os.path.join(self.frames_dir, f"frame_{idx:06d}.jpg")


class VideoSessionStore:
    """
    Store en memoria (MVP). Para producción: persistencia + cleanup programado.
    """
    def __init__(self) -> None:
        self._sessions: Dict[str, VideoSession] = {}

    def create_session_from_upload(self, tmp_video_path: str) -> VideoSession:
        sid = uuid.uuid4().hex[:12]
        frames_dir = tempfile.mkdtemp(prefix=f"sam3_frames_{sid}_")

        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video con OpenCV.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_s = (frame_count / fps) if fps > 0 else 0.0

        # Extraer frames a JPG (rápido para demo). Si quieres acelerar, baja calidad o samplea.
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
            # JPG quality 85 (balance)
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            idx += 1

        cap.release()

        # Ajustar frame_count real si OpenCV reportó mal
        if idx > 0:
            frame_count = idx
            duration_s = (frame_count / fps) if fps > 0 else 0.0

        sess = VideoSession(
            session_id=sid,
            video_path=tmp_video_path,
            frames_dir=frames_dir,
            fps=float(fps),
            frame_count=frame_count,
            width=width,
            height=height,
            duration_s=duration_s,
        )
        self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> Optional[VideoSession]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if not sess:
            return
        try:
            if os.path.exists(sess.video_path):
                os.remove(sess.video_path)
        except:
            pass
        try:
            if os.path.isdir(sess.frames_dir):
                shutil.rmtree(sess.frames_dir, ignore_errors=True)
        except:
            pass
