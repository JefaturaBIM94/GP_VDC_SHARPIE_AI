// frontend/src/components/VideoSegPlayer.tsx
import React, { useEffect, useRef } from "react";
import type { SegmentResponse } from "../api";

type Props = {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  src: string | null;
  segmentData: SegmentResponse | null;
  hoverId?: number;
  onHoverId?: (id: number | null) => void;
  onTimeUpdate?: () => void;
};

export function VideoSegPlayer({ videoRef, src, segmentData, hoverId, onHoverId, onTimeUpdate }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Si no hay fuente, no renderizamos el player para evitar src=""
  if (!src) {
    return null;
  }

  // Render overlay (fill + outline) desde segmentData.overlay_image_b64 o id_map_rgb_b64
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const rect = video.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width));
      canvas.height = Math.max(1, Math.floor(rect.height));
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(video);

    return () => ro.disconnect();
  }, [videoRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Caso 1: el backend ya manda un overlay pre-rendered (PNG b64)
    const overlayB64: string | null = (segmentData as any)?.overlay_image_b64 ?? null;
    if (overlayB64) {
      const img = new Image();
      img.onload = () => {
        // fill con opacidad tipo Meta
        ctx.globalAlpha = 0.45;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
      };
      img.src = overlayB64.startsWith("data:") ? overlayB64 : `data:image/png;base64,${overlayB64}`;
    }
  }, [segmentData, hoverId]);

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <video
        ref={videoRef}
        src={src ?? undefined}
        controls
        onTimeUpdate={onTimeUpdate}
        style={{ width: "100%", borderRadius: 12, border: "1px solid rgba(255,255,255,0.08)" }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
