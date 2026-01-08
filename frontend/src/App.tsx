import React, { useEffect, useMemo, useRef, useState } from "react";
import "./index.css";
import { segmentImage, type SegmentResponse, type InstanceLabel } from "./api";

type CountMode = "simple" | "multi";
type ChartType = "donut" | "pie" | "bubble";

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}
function formatClassName(s: string) {
  return (s ?? "").trim().toLowerCase() || "--";
}

/** ======= SVG CHARTS (sin dependencias) ======= */
function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function describeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number
) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

function PieDonutChart({
  data,
  type,
  size = 180,
}: {
  data: { className: string; count: number }[];
  type: "pie" | "donut";
  size?: number;
}) {
  const total = data.reduce((a, b) => a + b.count, 0);
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.42;
  const innerR = size * 0.28;

  const palette = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#22c55e"];

  let angle = 0;
  const slices = data.map((d, i) => {
    const frac = total > 0 ? d.count / total : 0;
    const start = angle;
    const end = angle + frac * 360;
    angle = end;

    const color = palette[i % palette.length];

    const outerArc = describeArc(cx, cy, r, start, end);
    const startOuter = polarToCartesian(cx, cy, r, end);
    const endOuter = polarToCartesian(cx, cy, r, start);

    if (type === "pie") {
      const path = `${outerArc} L ${cx} ${cy} L ${startOuter.x} ${startOuter.y} Z`;
      return { path, color, label: d.className, count: d.count };
    }

    const startInner = polarToCartesian(cx, cy, innerR, end);
    const endInner = polarToCartesian(cx, cy, innerR, start);
    const largeArcFlag = end - start <= 180 ? "0" : "1";

    const path = [
      `M ${startOuter.x} ${startOuter.y}`,
      `A ${r} ${r} 0 ${largeArcFlag} 0 ${endOuter.x} ${endOuter.y}`,
      `L ${endInner.x} ${endInner.y}`,
      `A ${innerR} ${innerR} 0 ${largeArcFlag} 1 ${startInner.x} ${startInner.y}`,
      "Z",
    ].join(" ");

    return { path, color, label: d.className, count: d.count };
  });

  return (
    <div className="flex items-center justify-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="rgba(2,6,23,0.35)" />
        {slices.map((s, idx) => (
          <path
            key={idx}
            d={s.path}
            fill={s.color}
            opacity={0.9}
            stroke="rgba(15,23,42,0.8)"
            strokeWidth={1}
          />
        ))}
        {type === "donut" && (
          <circle cx={cx} cy={cy} r={innerR} fill="rgba(2,6,23,0.85)" />
        )}

        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(226,232,240,0.9)"
          fontSize={16}
          fontWeight={700}
        >
          {total}
        </text>
        <text
          x={cx}
          y={cy + 16}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(148,163,184,0.9)"
          fontSize={10}
        >
          total objects
        </text>
      </svg>
    </div>
  );
}

function BubbleChart({
  data,
  width = 340,
  height = 200,
}: {
  data: { className: string; count: number }[];
  width?: number;
  height?: number;
}) {
  const max = data.reduce((m, d) => Math.max(m, d.count), 1);
  const palette = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#22c55e"];

  const padding = 14;
  const availableW = width - padding * 2;
  const step = data.length > 0 ? availableW / data.length : availableW;

  return (
    <div className="flex items-center justify-center">
      <svg width={width} height={height}>
        <rect x={0} y={0} width={width} height={height} rx={14} fill="rgba(2,6,23,0.35)" />
        {data.map((d, i) => {
          const frac = d.count / max;
          const r = 10 + frac * 34;
          const cx = padding + step * i + step / 2;
          const cy = height / 2;

          const color = palette[i % palette.length];

          return (
            <g key={d.className}>
              <circle cx={cx} cy={cy} r={r} fill={color} opacity={0.85} />
              <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(15,23,42,0.7)" strokeWidth={2} />
              <text
                x={cx}
                y={cy}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="rgba(2,6,23,0.95)"
                fontSize={12}
                fontWeight={800}
              >
                {d.count}
              </text>
              <text
                x={cx}
                y={cy + r + 16}
                textAnchor="middle"
                fill="rgba(148,163,184,0.95)"
                fontSize={10}
              >
                {d.className.length > 10 ? d.className.slice(0, 10) + "…" : d.className}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** ======= Hover Highlight helpers =======
 * Usamos id_map (PNG grayscale) para:
 * - Determinar hoveredId por píxel
 * - Precalcular contornos (edges) por id
 * - Dibujar outline/glow en un canvas TRANSPARENTE
 */
type EdgeMap = Map<number, Uint32Array>; // cada edge = [x,y,x,y,...] packed (x<<16|y) o viceversa

function computeEdgesFromIdMap(idMap: Uint8ClampedArray, w: number, h: number): EdgeMap {
  const edgesById = new Map<number, number[]>();

  const idx = (x: number, y: number) => y * w + x;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const id = idMap[idx(x, y)];
      if (id === 0) continue;

      // edge si algún vecino != id
      const n1 = idMap[idx(x - 1, y)];
      const n2 = idMap[idx(x + 1, y)];
      const n3 = idMap[idx(x, y - 1)];
      const n4 = idMap[idx(x, y + 1)];
      if (n1 !== id || n2 !== id || n3 !== id || n4 !== id) {
        if (!edgesById.has(id)) edgesById.set(id, []);
        edgesById.get(id)!.push((x << 16) | y);
      }
    }
  }

  const out: EdgeMap = new Map();
  for (const [id, arr] of edgesById.entries()) {
    out.set(id, new Uint32Array(arr));
  }
  return out;
}

function drawEdges(
  ctx: CanvasRenderingContext2D,
  edges: Uint32Array,
  color: string,
  scaleX: number,
  scaleY: number
) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Glow + outline (2 pasadas)
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Pass 1: glow suave
  ctx.globalAlpha = 0.85;
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;
  ctx.fillStyle = "rgba(255,255,255,0.0)"; // no fill real, pintamos “puntos”
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;

  // Dibujo por puntos (rápido y suficientemente “Meta-like” para POC)
  // Pintamos rects de 1px escalados -> se ve como borde
  for (let i = 0; i < edges.length; i++) {
    const packed = edges[i];
    const x = (packed >>> 16) & 0xffff;
    const y = packed & 0xffff;
    const dx = x * scaleX;
    const dy = y * scaleY;
    ctx.fillRect(dx, dy, Math.max(1, scaleX), Math.max(1, scaleY));
  }

  // Pass 2: outline más definido
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 0.95;
  ctx.fillStyle = "rgba(255,255,255,0.0)";
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

/** ======= MAIN APP ======= */
export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("columns");
  const [threshold, setThreshold] = useState(0.5);

  const [mode, setMode] = useState<CountMode>("simple");
  const [chartType, setChartType] = useState<ChartType>("donut");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [lastResult, setLastResult] = useState<SegmentResponse | null>(null);
  const [history, setHistory] = useState<SegmentResponse[]>([]);

  // Hover state
  const [hoverId, setHoverId] = useState<number>(0);

  // Refs para canvas hover
  const overlayImgRef = useRef<HTMLImageElement | null>(null);
  const hoverCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // id_map loaded data (grayscale)
  const [idMapData, setIdMapData] = useState<Uint8ClampedArray | null>(null);
  const [idMapW, setIdMapW] = useState<number>(0);
  const [idMapH, setIdMapH] = useState<number>(0);
  const [edgeMap, setEdgeMap] = useState<EdgeMap | null>(null);

  const labels: InstanceLabel[] = lastResult?.labels ?? [];

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setLastResult(null);
    setHistory([]);
    setError(null);

    setHoverId(0);
    setIdMapData(null);
    setIdMapW(0);
    setIdMapH(0);
    setEdgeMap(null);
  };

  const onModeChange = (m: CountMode) => {
    setMode(m);
    setHistory([]);
    if (lastResult && lastResult.num_objects > 0) setHistory([lastResult]);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Primero sube una imagen.");
      return;
    }

    setError(null);
    setLoading(true);
    setHoverId(0);

    try {
      const data = await segmentImage(file, prompt, threshold);
      setLastResult(data);

      if (data.num_objects > 0) {
        setHistory((prev) => (mode === "simple" ? [data] : [...prev, data]));
      }
    } catch (err) {
      console.error(err);
      setError("Error llamando al backend. Verifica Uvicorn en http://127.0.0.1:8000.");
    } finally {
      setLoading(false);
    }
  };

  const originalSrc = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  const overlaySrc = lastResult?.overlay_image_b64
    ? `data:image/png;base64,${lastResult.overlay_image_b64}`
    : null;

  const idMapSrc = lastResult?.id_map_b64
    ? `data:image/png;base64,${lastResult.id_map_b64}`
    : null;

  const aggregatedCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of history) {
      const key = formatClassName(r.class_name);
      counts.set(key, (counts.get(key) ?? 0) + (r.num_objects ?? 0));
    }
    return Array.from(counts.entries())
      .map(([className, count]) => ({ className, count }))
      .sort((a, b) => b.count - a.count);
  }, [history]);

  const totalObjects = aggregatedCounts.reduce((acc, row) => acc + row.count, 0);
  const maxCount = aggregatedCounts.reduce((acc, row) => Math.max(acc, row.count), 0);

  // Map id -> label color (para outline)
  const colorById = useMemo(() => {
    const m = new Map<number, string>();
    for (const l of labels) m.set(l.id, l.color);
    return m;
  }, [labels]);

  /** Cargar id_map PNG y convertirlo a Uint8ClampedArray (canal R) */
  useEffect(() => {
    if (!idMapSrc) {
      setIdMapData(null);
      setIdMapW(0);
      setIdMapH(0);
      setEdgeMap(null);
      return;
    }

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
      const imageData = ctx.getImageData(0, 0, w, h).data;

      // grayscale PNG: R=G=B=id
      const ids = new Uint8ClampedArray(w * h);
      for (let i = 0, p = 0; i < imageData.length; i += 4, p++) {
        ids[p] = imageData[i]; // canal R
      }

      setIdMapData(ids);
      setIdMapW(w);
      setIdMapH(h);

      // precalc edges por id (una sola vez por segmentación)
      const edges = computeEdgesFromIdMap(ids, w, h);
      setEdgeMap(edges);
    };
    img.src = idMapSrc;

    return () => {
      cancelled = true;
    };
  }, [idMapSrc]);

  /** Ajustar el canvas hover a las dimensiones RENDER del <img> (no natural) */
  const syncHoverCanvasToImage = () => {
    const imgEl = overlayImgRef.current;
    const canvas = hoverCanvasRef.current;
    if (!imgEl || !canvas) return;

    const rect = imgEl.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));

    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
  };

  useEffect(() => {
    syncHoverCanvasToImage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlaySrc, loading]);

  useEffect(() => {
    const onResize = () => syncHoverCanvasToImage();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  /** Cuando cambia hoverId, dibuja outline en canvas */
  useEffect(() => {
    const canvas = hoverCanvasRef.current;
    const imgEl = overlayImgRef.current;
    if (!canvas || !imgEl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!hoverId || !edgeMap || !idMapW || !idMapH) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const edges = edgeMap.get(hoverId);
    if (!edges) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const scaleX = canvas.width / idMapW;
    const scaleY = canvas.height / idMapH;

    const color = colorById.get(hoverId) ?? "#34d399";
    drawEdges(ctx, edges, color, scaleX, scaleY);
  }, [hoverId, edgeMap, idMapW, idMapH, colorById]);

  /** Mouse move: samplea id_map por pixel para determinar hoverId */
  const onOverlayMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!idMapData || !overlayImgRef.current || !idMapW || !idMapH) return;

    const imgEl = overlayImgRef.current;
    const rect = imgEl.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      if (hoverId !== 0) setHoverId(0);
      return;
    }

    // Convertir coords render -> coords natural
    const nx = Math.floor((x / rect.width) * idMapW);
    const ny = Math.floor((y / rect.height) * idMapH);

    const idx = ny * idMapW + nx;
    const id = idMapData[idx] ?? 0;

    if (id !== hoverId) setHoverId(id);
  };

  const onOverlayMouseLeave = () => {
    setHoverId(0);
  };

  return (
    <div className="min-h-screen bg-black text-slate-50 flex flex-col">
      {/* HEADER */}
      <header className="w-full border-b border-slate-900 px-8 py-4 flex items-center justify-between bg-gradient-to-r from-black via-slate-950 to-black">
        <div className="flex items-center gap-4">
          <img
            src="/gpc-horizontal-blanco.png"
            alt="GP Construcción"
            className="h-12 w-auto drop-shadow-[0_0_12px_rgba(16,185,129,0.55)]"
          />
          <div className="flex flex-col">
            <span className="text-2xl font-semibold tracking-tight title-3d">
              <span className="text-emerald-400">SHARPIE.AI ·</span> SAM3 Object Tracking
            </span>
            <span className="text-[11px] text-slate-400 tracking-[0.18em] uppercase">
              App by Research + Development · 2025
            </span>
          </div>
        </div>
        <div className="text-[11px] text-slate-500 uppercase tracking-[0.25em]">
          Beta · Local Playground
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="flex-1 p-5 bg-gradient-to-br from-slate-950 via-black to-slate-950">
        <div className="grid grid-cols-12 gap-5 h-[calc(100vh-110px)]">
          {/* Q1 */}
          <section className="col-span-12 lg:col-span-3 h-full rounded-2xl border border-slate-900 bg-slate-950/70 backdrop-blur-md p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">1. Panel de análisis</h2>
              <div className="text-[11px] text-slate-400">v0.2</div>
            </div>

            <div className="space-y-2 text-xs">
              <p className="text-[11px] text-slate-400">
                <span className="font-semibold text-emerald-400">Paso 1 ·</span> Sube una imagen
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={onFileChange}
                className="block w-full text-[11px] file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-[11px] file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-400 cursor-pointer"
              />
            </div>

            <div className="space-y-2 text-xs">
              <p className="text-[11px] text-slate-400">
                <span className="font-semibold text-emerald-400">Paso 2 ·</span> Prompt / clase a segmentar
              </p>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full rounded-md bg-black/40 border border-slate-800 px-3 py-2 text-[12px] focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                placeholder="Ej: columns, walls, floors..."
              />
              <p className="text-[11px] text-slate-500">
                Tip: prompts específicos suelen ayudar (ej. “structural column”, “concrete wall”).
              </p>
            </div>

            <div className="space-y-2 text-xs">
              <p className="text-[11px] text-slate-400">
                <span className="font-semibold text-emerald-400">Paso 3 ·</span> Umbral score (0–1)
              </p>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="flex-1 accent-emerald-500"
                />
                <span className="w-12 text-right text-[11px]">{clamp01(threshold).toFixed(2)}</span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2">
              <span className="text-[11px] text-slate-400 uppercase tracking-widest">Mode</span>
              <div className="inline-flex rounded-full bg-slate-900/80 p-0.5 text-[11px]">
                <button
                  onClick={() => onModeChange("simple")}
                  className={`px-3 py-1 rounded-full transition ${
                    mode === "simple" ? "bg-emerald-500 text-slate-950" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Simple
                </button>
                <button
                  onClick={() => onModeChange("multi")}
                  className={`px-3 py-1 rounded-full transition ${
                    mode === "multi" ? "bg-emerald-500 text-slate-950" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Multi
                </button>
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading || !file}
              className="mt-2 inline-flex items-center justify-center rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-sm font-semibold px-4 py-3 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_0_28px_rgba(16,185,129,0.45)] transition"
            >
              {loading ? "Segmentando..." : "Segmentar"}
            </button>

            {loading && (
              <div className="rounded-xl border border-slate-900 bg-black/40 p-3 render-noise">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] text-slate-300">Rendering preview</span>
                  <span className="text-[11px] text-slate-500">SAM3</span>
                </div>
                <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                  <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
                </div>
                <p className="mt-2 text-[11px] text-slate-500">Segmentando máscaras + generando overlay…</p>
              </div>
            )}

            {error && (
              <div className="text-[11px] text-red-300 bg-red-950/40 border border-red-900 rounded-xl px-3 py-3">
                {error}
              </div>
            )}

            <div className="rounded-2xl border border-slate-900 bg-black/35 p-4">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-slate-400 uppercase tracking-widest">último resultado</span>
                <span className="text-[11px] text-slate-500">
                  {lastResult?.session_id ? `session ${lastResult.session_id}` : "--"}
                </span>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3 text-[11px]">
                <div className="rounded-xl bg-slate-950/60 border border-slate-900 p-3">
                  <div className="text-slate-400">Prompt</div>
                  <div className="text-emerald-300 font-semibold mt-1">{lastResult?.class_name ?? "--"}</div>
                </div>
                <div className="rounded-xl bg-slate-950/60 border border-slate-900 p-3">
                  <div className="text-slate-400">Objetos</div>
                  <div className="text-emerald-300 font-semibold mt-1">{lastResult?.num_objects ?? 0}</div>
                </div>
                <div className="rounded-xl bg-slate-950/60 border border-slate-900 p-3">
                  <div className="text-slate-400">Umbral</div>
                  <div className="text-slate-200 font-semibold mt-1">{(lastResult?.threshold ?? threshold).toFixed(2)}</div>
                </div>
                <div className="rounded-xl bg-slate-950/60 border border-slate-900 p-3">
                  <div className="text-slate-400">Tags</div>
                  <div className="text-slate-200 font-semibold mt-1">{labels.length}</div>
                </div>
              </div>

              {lastResult && lastResult.num_objects === 0 && (
                <div className="mt-3 text-[11px] text-amber-300 bg-amber-950/35 border border-amber-900 rounded-xl px-3 py-2">
                  No se encontraron objetos para ese prompt.
                </div>
              )}
            </div>
          </section>

          {/* Q2 */}
          <section className="col-span-12 lg:col-span-5 h-full rounded-2xl border border-slate-900 bg-slate-950/40 p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">2. Imagen original</h2>
              <span className="text-[11px] text-slate-500">{file ? file.name : "—"}</span>
            </div>

            <div className="flex-1 rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex items-center justify-center">
              {originalSrc ? (
                <img src={originalSrc} alt="Imagen original" className="max-h-full max-w-full object-contain" />
              ) : (
                <div className="text-slate-500 text-sm px-6 text-center">Sube una imagen para comenzar.</div>
              )}
            </div>
          </section>

          {/* Right column */}
          <section className="col-span-12 lg:col-span-4 h-full flex flex-col gap-5">
            {/* Q3 */}
            <div className="flex-1 rounded-2xl border border-slate-900 bg-slate-950/40 p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-100">
                  3. Resultado + Hover Highlight
                </h2>
                <span className="text-[11px] text-slate-500">{labels.length} instancias</span>
              </div>

              <div
                className="flex-1 rounded-2xl border border-slate-900 bg-black/35 overflow-hidden relative flex items-center justify-center"
                onMouseMove={onOverlayMouseMove}
                onMouseLeave={onOverlayMouseLeave}
              >
                {loading && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center">
                    <div className="w-[92%] max-w-[520px] rounded-2xl border border-slate-900 bg-black/45 p-4 render-noise">
                      <div className="text-[11px] text-slate-300 mb-2 flex items-center justify-between">
                        <span>Subsampling preview</span>
                        <span className="text-slate-500">SAM3</span>
                      </div>
                      <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                        <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
                      </div>
                      <div className="mt-3 text-[11px] text-slate-500">
                        Construyendo máscara → overlay → id_map → labels…
                      </div>
                    </div>
                  </div>
                )}

                {overlaySrc ? (
                  <>
                    <img
                      ref={overlayImgRef}
                      src={overlaySrc}
                      alt="Resultado de segmentación"
                      className={`max-h-full max-w-full object-contain ${loading ? "blur-[2px] opacity-60" : ""}`}
                      onLoad={() => syncHoverCanvasToImage()}
                    />

                    {/* Canvas transparente para outline/glow (NO pinta fondo) */}
                    <canvas
                      ref={hoverCanvasRef}
                      className="absolute pointer-events-none z-20"
                      style={{ width: "100%", height: "100%" }}
                    />

                    {/* TAGS */}
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
                  <div className="text-slate-500 text-sm px-6 text-center">
                    Cuando ejecutes <span className="font-semibold">“Segmentar”</span>, aquí verás la segmentación
                    y podrás resaltar máscaras con hover (como Meta).
                  </div>
                )}

                {/* id_map hidden (solo para asegurar que exista / debug opcional) */}
                {idMapSrc && <img src={idMapSrc} alt="id_map" className="hidden" />}
              </div>

              {/* Tabla instancias */}
              {labels.length > 0 && (
                <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden">
                  <div className="px-4 py-3 bg-slate-900/60 flex items-center justify-between">
                    <span className="text-[11px] font-semibold text-slate-200">
                      Instancias (hover para resaltar)
                    </span>
                    <span className="text-[11px] text-slate-400">{labels.length} objetos</span>
                  </div>

                  <div className="grid grid-cols-4 px-4 py-2 text-[11px] text-slate-400 border-t border-slate-900/70">
                    <span>ID</span>
                    <span>Clase</span>
                    <span className="text-right">Score</span>
                    <span className="text-right">Color</span>
                  </div>

                  <div className="max-h-[160px] overflow-auto">
                    {labels.map((lbl) => (
                      <div
                        key={lbl.id}
                        className={`grid grid-cols-4 px-4 py-2 text-[11px] border-t border-slate-900/60 transition ${
                          hoverId === lbl.id ? "bg-emerald-500/10" : ""
                        }`}
                        onMouseEnter={() => setHoverId(lbl.id)}
                        onMouseLeave={() => setHoverId(0)}
                      >
                        <span className="font-semibold text-slate-100">#{lbl.id}</span>
                        <span className="text-slate-300">{formatClassName(lastResult?.class_name ?? "")}</span>
                        <span className="text-right text-slate-200">{lbl.score.toFixed(3)}</span>
                        <span className="text-right">
                          <span
                            className="inline-block w-3 h-3 rounded-full"
                            style={{ background: lbl.color }}
                          />
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Q4 */}
            <div className="h-[340px] rounded-2xl border border-slate-900 bg-slate-950/40 p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-100">4. Conteos + Gráfica</h2>

                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-400">Chart</span>
                  <select
                    value={chartType}
                    onChange={(e) => setChartType(e.target.value as ChartType)}
                    className="text-[11px] bg-black/40 border border-slate-800 rounded-md px-2 py-1 text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  >
                    <option value="donut">Donut</option>
                    <option value="pie">Pie</option>
                    <option value="bubble">Bubble</option>
                  </select>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden">
                <div className="grid grid-cols-3 px-4 py-2 text-[11px] bg-slate-900/60 text-slate-400">
                  <span>Clase</span>
                  <span className="text-right">Objetos</span>
                  <span className="text-right">%</span>
                </div>

                <div className="max-h-[120px] overflow-auto">
                  {aggregatedCounts.length === 0 ? (
                    <div className="px-4 py-3 text-[11px] text-slate-500">
                      Ejecuta segmentaciones para ver conteos aquí.
                    </div>
                  ) : (
                    aggregatedCounts.map((row) => (
                      <div
                        key={row.className}
                        className="grid grid-cols-3 px-4 py-2 text-[11px] border-t border-slate-900/60"
                      >
                        <span className="text-slate-100 font-medium">{row.className}</span>
                        <span className="text-right text-slate-200">{row.count}</span>
                        <span className="text-right text-slate-300">
                          {totalObjects > 0 ? ((row.count / totalObjects) * 100).toFixed(1) : "0.0"}%
                        </span>
                      </div>
                    ))
                  )}
                </div>

                <div className="grid grid-cols-3 px-4 py-2 text-[11px] bg-slate-900/60 border-t border-slate-900/70 font-semibold">
                  <span>Total</span>
                  <span className="text-right">{totalObjects}</span>
                  <span />
                </div>
              </div>

              <div className="flex-1 rounded-2xl border border-slate-900 bg-black/35 p-3">
                {aggregatedCounts.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-[11px] text-slate-500">
                    Sin datos todavía.
                  </div>
                ) : chartType === "bubble" ? (
                  <BubbleChart data={aggregatedCounts} />
                ) : chartType === "pie" ? (
                  <PieDonutChart data={aggregatedCounts} type="pie" />
                ) : (
                  <PieDonutChart data={aggregatedCounts} type="donut" />
                )}

                {aggregatedCounts.length > 0 && maxCount > 0 && (
                  <div className="mt-3 space-y-1.5">
                    {aggregatedCounts.slice(0, 4).map((row) => (
                      <div key={row.className} className="flex items-center gap-2">
                        <span className="w-16 text-[11px] text-slate-400 truncate">{row.className}</span>
                        <div className="flex-1 h-2 rounded-full bg-slate-900 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-emerald-400/80"
                            style={{ width: `${(row.count / maxCount) * 100}%` }}
                          />
                        </div>
                        <span className="w-8 text-right text-[11px] text-slate-200">{row.count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
