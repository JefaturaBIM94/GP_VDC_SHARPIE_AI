import React, { useEffect, useMemo, useRef } from "react";
import type { SegmentResponse } from "../api";

function b64ToImageSrc(b64: string): string {
  // tu backend suele mandar base64 “puro” (sin data:)
  // si ya viene con “data:image/png;base64,” no pasa nada
  if (b64.startsWith("data:")) return b64;
  return `data:image/png;base64,${b64}`;
}

type Props = {
  videoUrl: string;
  // resultados por timestamp (segundos)
  results: Array<{ t: number; seg: SegmentResponse }>;
  playing: boolean;
  onTime?: (t: number) => void;
  opacity?: number; // 0..1 fill opacity
};

export function VideoSegPlayer({
  videoUrl,
  results,
  playing,
  onTime,
  opacity = 0.5,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // ordena por tiempo para búsqueda rápida
  const sorted = useMemo(() => [...results].sort((a, b) => a.t - b.t), [results]);

  // cache de imágenes overlay para evitar re-decode en cada frame
  const overlayCache = useRef<Map<number, HTMLImageElement>>(new Map());

  const getClosest = (t: number) => {
    if (!sorted.length) return null;
    // búsqueda lineal simple (para MVP). Si quieres, luego lo cambiamos a binary search.
    let best = sorted[0];
    let bestd = Math.abs(sorted[0].t - t);
    for (let i = 1; i < sorted.length; i++) {
      const d = Math.abs(sorted[i].t - t);
      if (d < bestd) {
        best = sorted[i];
        bestd = d;
      }
    }
    return best;
  };

  const ensureOverlayImg = (keyT: number, b64: string) => {
    const cache = overlayCache.current;
    if (cache.has(keyT)) return cache.get(keyT)!;
    const img = new Image();
    img.src = b64ToImageSrc(b64);
    cache.set(keyT, img);
    return img;
  };

  const draw = () => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;

    const ctx = c.getContext("2d");
    if (!ctx) return;

    // sincroniza canvas con tamaño real renderizado del video
    const rect = v.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }

    ctx.clearRect(0, 0, c.width, c.height);

    const t = v.currentTime;
    onTime?.(t);

    const closest = getClosest(t);
    if (!closest) return;

    const overlayB64 = closest.seg.overlay_image_b64;
    if (!overlayB64) return;

    const overlayImg = ensureOverlayImg(closest.t, overlayB64);

    // Dibuja overlay con opacidad (fill). Tu overlay ya trae outline, pero aquí forzamos visibilidad.
    // Nota: si el overlay del backend trae alpha muy bajo, este globalAlpha ayuda muchísimo.
    ctx.save();
    ctx.globalAlpha = opacity;
    // overlayImg puede no estar cargada aún; si no está, no dibuja hasta el siguiente tick
    if (overlayImg.complete && overlayImg.naturalWidth > 0) {
      ctx.drawImage(overlayImg, 0, 0, c.width, c.height);
    }
    ctx.restore();
  };

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const onResize = () => draw();
    const onUpdate = () => draw();

    window.addEventListener("resize", onResize);
    v.addEventListener("timeupdate", onUpdate);
    v.addEventListener("seeked", onUpdate);
    v.addEventListener("loadedmetadata", onUpdate);

    // primer draw
    const id = window.setTimeout(draw, 0);

    return () => {
      window.clearTimeout(id);
      window.removeEventListener("resize", onResize);
      v.removeEventListener("timeupdate", onUpdate);
      v.removeEventListener("seeked", onUpdate);
      v.removeEventListener("loadedmetadata", onUpdate);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoUrl, results, opacity]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (playing) v.play().catch(() => {});
    else v.pause();
  }, [playing]);

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        style={{
          width: "100%",
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.08)",
          background: "#000",
          display: "block",
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          // importante: el canvas toma el mismo layout que el video
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          borderRadius: 10,
        }}
      />
    </div>
  );
}
