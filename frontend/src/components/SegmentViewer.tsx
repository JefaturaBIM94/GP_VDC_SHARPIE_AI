// frontend/src/components/SegmentViewer.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import type { InstanceLabel, SegmentResponse } from "../api";

type EdgeMap = Map<number, Uint32Array>;

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) return void setUrl(null);
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

function computeEdgesFromIdMap(ids: Uint32Array, w: number, h: number): EdgeMap {
  const edgesById = new Map<number, number[]>();
  const idx = (x: number, y: number) => y * w + x;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const id = ids[idx(x, y)];
      if (!id) continue;
      const n1 = ids[idx(x - 1, y)];
      const n2 = ids[idx(x + 1, y)];
      const n3 = ids[idx(x, y - 1)];
      const n4 = ids[idx(x, y + 1)];
      if (n1 !== id || n2 !== id || n3 !== id || n4 !== id) {
        if (!edgesById.has(id)) edgesById.set(id, []);
        edgesById.get(id)!.push((x << 16) | y);
      }
    }
  }

  const out: EdgeMap = new Map();
  for (const [id, arr] of edgesById.entries()) out.set(id, new Uint32Array(arr));
  return out;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const h = (hex || "").replace("#", "").trim();
  if (h.length !== 6) return { r: 52, g: 211, b: 153 };
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return { r, g, b };
}

function drawHoverEdges(
  ctx: CanvasRenderingContext2D,
  edges: Uint32Array,
  color: string,
  scaleX: number,
  scaleY: number
) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.save();
  ctx.globalAlpha = 0.95;
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;
  ctx.fillStyle = color;

  for (let i = 0; i < edges.length; i++) {
    const packed = edges[i];
    const x = (packed >>> 16) & 0xffff;
    const y = packed & 0xffff;
    const dx = x * scaleX;
    const dy = y * scaleY;
    ctx.fillRect(dx, dy, Math.max(1, scaleX), Math.max(1, scaleY));
  }
  ctx.restore();
}

export function SegmentViewer({
  title,
  file,
  result,
  loading,
}: {
  title: string;
  file: File | null;
  result: SegmentResponse | null;
  loading?: boolean;
}) {
  const originalSrc = useObjectUrl(file);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hoverCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [hoverId, setHoverId] = useState<number>(0);

  // Decoded id map (uint32 for safety)
  const [idMap, setIdMap] = useState<Uint32Array | null>(null);
  const [mapW, setMapW] = useState(0);
  const [mapH, setMapH] = useState(0);
  const [edgesById, setEdgesById] = useState<EdgeMap | null>(null);

  const labels: InstanceLabel[] = result?.labels ?? [];

  const colorById = useMemo(() => {
    const m = new Map<number, string>();
    for (const l of labels) m.set(l.id, l.color);
    return m;
  }, [labels]);

  const idMapRgbSrc = result?.id_map_rgb_b64 ? `data:image/png;base64,${result.id_map_rgb_b64}` : null;

  // Sync canvas to image displayed size
  const syncCanvases = () => {
    const img = imgRef.current;
    const c1 = overlayCanvasRef.current;
    const c2 = hoverCanvasRef.current;
    if (!img || !c1 || !c2) return;

    const rect = img.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));

    if (c1.width !== w) c1.width = w;
    if (c1.height !== h) c1.height = h;
    if (c2.width !== w) c2.width = w;
    if (c2.height !== h) c2.height = h;
  };

  useEffect(() => {
    const onResize = () => syncCanvases();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Load id_map_rgb and decode to Uint32Array
  useEffect(() => {
    setHoverId(0);
    setIdMap(null);
    setEdgesById(null);
    setMapW(0);
    setMapH(0);

    if (!idMapRgbSrc) return;

    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;

      const w = img.naturalWidth;
      const h = img.naturalHeight;

      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, w, h).data;

      const ids = new Uint32Array(w * h);
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        ids[p] = r + (g << 8) + (b << 16);
      }

      setIdMap(ids);
      setMapW(w);
      setMapH(h);

      const edges = computeEdgesFromIdMap(ids, w, h);
      setEdgesById(edges);
    };
    img.src = idMapRgbSrc;

    return () => {
      cancelled = true;
    };
  }, [idMapRgbSrc]);

  // Draw overlay masks canvas (colors) above original
  useEffect(() => {
    const c = overlayCanvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, c.width, c.height);

    if (!idMap || !mapW || !mapH || !labels.length) return;

    // Build an ImageData at map resolution, then scale to canvas size
    const off = document.createElement("canvas");
    off.width = mapW;
    off.height = mapH;
    const octx = off.getContext("2d", { willReadFrequently: true });
    if (!octx) return;

    const imgData = octx.createImageData(mapW, mapH);
    const out = imgData.data;

    const rgbCache = new Map<number, { r: number; g: number; b: number }>();
    for (const l of labels) rgbCache.set(l.id, hexToRgb(l.color));

    // Alpha del overlay (ajustable)
    const A = 115; // ~0.45

    for (let p = 0; p < idMap.length; p++) {
      const id = idMap[p];
      if (!id) continue;
      const rgb = rgbCache.get(id);
      if (!rgb) continue;

      const i = p * 4;
      out[i] = rgb.r;
      out[i + 1] = rgb.g;
      out[i + 2] = rgb.b;
      out[i + 3] = A;
    }

    octx.putImageData(imgData, 0, 0);

    // Paint scaled
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, c.width, c.height);
  }, [idMap, mapW, mapH, labels]);

  // Draw hover highlight edges
  useEffect(() => {
    const c = hoverCanvasRef.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;

    ctx.clearRect(0, 0, c.width, c.height);

    if (!hoverId || !edgesById || !mapW || !mapH) return;
    const edges = edgesById.get(hoverId);
    if (!edges) return;

    const scaleX = c.width / mapW;
    const scaleY = c.height / mapH;
    const color = colorById.get(hoverId) ?? "#34d399";
    drawHoverEdges(ctx, edges, color, scaleX, scaleY);
  }, [hoverId, edgesById, mapW, mapH, colorById]);

  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!idMap || !imgRef.current || !mapW || !mapH) return;

    const rect = imgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      if (hoverId !== 0) setHoverId(0);
      return;
    }

    const nx = Math.floor((x / rect.width) * mapW);
    const ny = Math.floor((y / rect.height) * mapH);

    const id = idMap[ny * mapW + nx] ?? 0;
    if (id !== hoverId) setHoverId(id);
  };

  return (
    <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex flex-col">
      <div className="px-3 py-2 bg-slate-900/60 flex items-center justify-between">
        <span className="text-[11px] font-semibold text-slate-200">{title}</span>
        <span className="text-[11px] text-slate-400">{labels.length} instancias</span>
      </div>

      <div className="p-3 flex-1">
        <div
          className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden relative flex items-center justify-center"
          onMouseMove={onMouseMove}
          onMouseLeave={() => setHoverId(0)}
        >
          {originalSrc ? (
            <>
              <img
                ref={imgRef}
                src={originalSrc}
                alt={title}
                className={`max-h-[520px] w-auto object-contain ${loading ? "blur-[2px] opacity-60" : ""}`}
                onLoad={() => syncCanvases()}
              />

              {/* overlay masks */}
              <canvas
                ref={overlayCanvasRef}
                className="absolute pointer-events-none z-10"
                style={{ width: "100%", height: "100%" }}
              />

              {/* hover highlight */}
              <canvas
                ref={hoverCanvasRef}
                className="absolute pointer-events-none z-20"
                style={{ width: "100%", height: "100%" }}
              />

              {/* chips ID (opcional) */}
              {labels.map((lbl) => {
                const isActive = hoverId === lbl.id;
                return (
                  <div
                    key={lbl.id}
                    className="absolute z-30"
                    style={{
                      left: `${clamp01(lbl.cx) * 100}%`,
                      top: `${clamp01(lbl.cy) * 100}%`,
                      transform: "translate(-50%, -50%)",
                    }}
                  >
                    <button
                      type="button"
                      onMouseEnter={() => setHoverId(lbl.id)}
                      onMouseLeave={() => setHoverId(0)}
                      className={`px-2 py-1 rounded-md text-[10px] font-extrabold shadow-lg border transition ${
                        isActive ? "border-emerald-300/70" : "border-black/60"
                      }`}
                      style={{
                        background: "rgba(0,0,0,0.78)",
                        color: "white",
                        boxShadow: isActive
                          ? "0 0 0 1px rgba(16,185,129,0.35), 0 0 18px rgba(16,185,129,0.35)"
                          : undefined,
                      }}
                    >
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle"
                        style={{ background: lbl.color }}
                      />
                      ID {lbl.id}
                    </button>
                  </div>
                );
              })}
            </>
          ) : (
            <div className="text-slate-500 text-sm px-6 py-10 text-center">
              Sube una imagen para visualizar.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
