// frontend/src/App.tsx
import React, { useEffect, useMemo, useState } from "react";
import "./index.css";
import {
  segmentImage,
  ocrBatch,
  type SegmentResponse,
  type InstanceLabel,
  type OcrBatchResponse,
  type OcrItem,
  type OcrDetection,
} from "./api";
import Sam3CompareView from "./views/Sam3CompareView";
import FastReconstructionView from "./views/FastReconstructionView";
import { SegmentViewer } from "./components/SegmentViewer";
import ResultSidePanel from "./components/ResultSidePanel";

type CountMode = "simple" | "multi";
type ChartType = "donut" | "pie" | "bubble";
type ToolMode = "sam3" | "sam3_compare" | "ocr" | "fast_recon";

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}
function formatClassName(s: string) {
  return (s ?? "").trim().toLowerCase() || "--";
}

/** Hook para URL.createObjectURL con cleanup (evita leaks) */
function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  return url;
}

/** ======= SVG CHARTS (sin dependencias) ======= */
function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
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
        {type === "donut" && <circle cx={cx} cy={cy} r={innerR} fill="rgba(2,6,23,0.85)" />}

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
              <text x={cx} y={cy + r + 16} textAnchor="middle" fill="rgba(148,163,184,0.95)" fontSize={10}>
                {d.className.length > 10 ? d.className.slice(0, 10) + "…" : d.className}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** ======= Hover Highlight helpers (SAM3) ======= */
// (EdgeMap type was moved to SegmentViewer)


/** ======= MAIN APP ======= */
export default function App() {
  // DEBUG flag — set VITE_DEBUG=1 to enable verbose frontend SAM3 logs
  const DEBUG = ((import.meta as any).env?.VITE_DEBUG ?? "0") === "1";

  // Helper: decode id_map_rgb_b64 exactly like SegmentViewer does
  async function decodeIdMapRgbB64(b64: string) {
    return new Promise<{ ids: Uint32Array; w: number; h: number }>((resolve, reject) => {
      try {
        const img = new Image();
        img.onload = () => {
          const w = img.naturalWidth;
          const h = img.naturalHeight;
          const c = document.createElement("canvas");
          c.width = w;
          c.height = h;
          const ctx = c.getContext("2d", { willReadFrequently: true });
          if (!ctx) return reject(new Error("Canvas context unavailable"));
          ctx.drawImage(img, 0, 0);
          const data = ctx.getImageData(0, 0, w, h).data;

          const ids = new Uint32Array(w * h);
          for (let i = 0, p = 0; i < data.length; i += 4, p++) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            ids[p] = r + (g << 8) + (b << 16);
          }

          resolve({ ids, w, h });
        };
        img.onerror = () => reject(new Error("Failed to load id_map image"));
        img.src = `data:image/png;base64,${b64}`;
      } catch (err) {
        reject(err);
      }
    });
  }

  // Helper: map client coords to id-map coords accounting for object-fit:contain letterbox offsets
  function mapClientToIdCoords(
    imgEl: HTMLImageElement | null,
    clientX: number,
    clientY: number,
    mapW: number,
    mapH: number
  ): { nx: number; ny: number } | null {
    if (!imgEl) return null;
    const rect = imgEl.getBoundingClientRect();
    const naturalW = imgEl.naturalWidth || 0;
    const naturalH = imgEl.naturalHeight || 0;
    if (!naturalW || !naturalH) return null;

    // compute display size of the image inside the rect when object-fit: contain is used
    const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
    const dispW = naturalW * scale;
    const dispH = naturalH * scale;
    const offsetX = (rect.width - dispW) / 2;
    const offsetY = (rect.height - dispH) / 2;

    const xInImage = clientX - rect.left - offsetX;
    const yInImage = clientY - rect.top - offsetY;
    if (xInImage < 0 || yInImage < 0 || xInImage > dispW || yInImage > dispH) return null;

    const nx = Math.floor((xInImage / dispW) * mapW);
    const ny = Math.floor((yInImage / dispH) * mapH);
    return { nx, ny };
  }

  // expose mapping helper in DEBUG for interactive testing in console
  
  // Tool selector
  const [toolMode, setToolMode] = useState<ToolMode>("sam3");

  // SAM3 state
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("columns");
  const [threshold, setThreshold] = useState(0.5);

  // DEBUG mapping test
  const [debugMappingActive, setDebugMappingActive] = useState(false);
  const [debugIdMap, setDebugIdMap] = useState<{ ids: Uint32Array; w: number; h: number } | null>(null);
  const [debugHover, setDebugHover] = useState<{ id: number; nx: number; ny: number } | null>(null);

  // expose mapping helper in DEBUG for interactive testing in console
  useEffect(() => {
    if (!DEBUG) return;
    (window as any).__sam3_mapClientToIdCoords = mapClientToIdCoords;

    const onMove = (e: MouseEvent) => {
      if (!debugMappingActive) return;
      const imgEl = document.querySelector('img[alt^="SAM3"]') as HTMLImageElement | null;
      if (!imgEl || !debugIdMap) return;
      const coords = mapClientToIdCoords(imgEl, e.clientX, e.clientY, debugIdMap.w, debugIdMap.h);
      if (!coords) {
        setDebugHover(null);
        return;
      }
      const idx = coords.ny * debugIdMap.w + coords.nx;
      const id = debugIdMap.ids[idx] ?? 0;
      setDebugHover({ id, nx: coords.nx, ny: coords.ny });
    };

    window.addEventListener("mousemove", onMove);

    return () => {
      try {
        delete (window as any).__sam3_mapClientToIdCoords;
      } catch {}
      window.removeEventListener("mousemove", onMove);
    };
  }, [DEBUG, debugMappingActive, debugIdMap]);

  const [mode, setMode] = useState<CountMode>("simple");
  const [chartType, setChartType] = useState<ChartType>("donut");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [lastResult, setLastResult] = useState<SegmentResponse | null>(null);
  const [history, setHistory] = useState<SegmentResponse[]>([]);

  // OCR state
  const [ocrFiles, setOcrFiles] = useState<File[]>([]);
  const [ocrResult, setOcrResult] = useState<OcrBatchResponse | null>(null);
  const [ocrSelectedIdx, setOcrSelectedIdx] = useState<number>(0);

  // Hover state (SAM3)
  const [hoverId, setHoverId] = useState<number | null>(null);

  // legacy hover refs/states removed (SegmentViewer handles id_map and hover)

  const labels: InstanceLabel[] = lastResult?.labels ?? [];

  const resetSam3 = () => {
    setFile(null);
    setLastResult(null);
    setHistory([]);
    setHoverId(null);
  };

  const resetOcr = () => {
    setOcrFiles([]);
    setOcrResult(null);
    setOcrSelectedIdx(0);
  };

  const onToolModeChange = (m: ToolMode) => {
    setToolMode(m);
    setError(null);
    setLoading(false);

    // OCR es independiente
    if (m === "ocr") {
      resetSam3();
      return;
    }

    // sam3 y sam3_compare comparten “familia SAM”
    resetOcr();
    // opcional: si quieres que SAM3 Compare tenga estado propio,
    // NO resetees sam3 aquí. Por ahora, lo dejamos así para no confundir.
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);

    if (toolMode === "sam3") {
      const f = e.target.files?.[0] ?? null;
      setFile(f);
      setLastResult(null);
      setHistory([]);
      setHoverId(null);
    } else {
      const files = Array.from(e.target.files ?? []);
      const limited = files.slice(0, 10);
      setOcrFiles(limited);
      setOcrResult(null);
      setOcrSelectedIdx(0);
    }
  };

  // Guard rail: si cambia el arreglo, asegurar índice válido
  useEffect(() => {
    if (toolMode !== "ocr") return;
    if (ocrFiles.length === 0) {
      setOcrSelectedIdx(0);
      return;
    }
    if (ocrSelectedIdx > ocrFiles.length - 1) {
      setOcrSelectedIdx(0);
    }
  }, [ocrFiles, ocrSelectedIdx, toolMode]);

  useEffect(() => {
    if (toolMode !== "ocr") return;
    if (!ocrResult?.items?.length) return;
    if (ocrSelectedIdx > ocrResult.items.length - 1) setOcrSelectedIdx(0);
  }, [ocrResult, ocrSelectedIdx, toolMode]);

  const onModeChange = (m: CountMode) => {
    setMode(m);
    setHistory([]);
    if (lastResult && (lastResult.labels?.length ?? 0) > 0) setHistory([lastResult]);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Primero sube una imagen.");
      return;
    }

    setError(null);
    setLoading(true);
    setHoverId(null);

    try {
      const data = await segmentImage(file, prompt, threshold);
      setLastResult(data);

      if (DEBUG) {
        console.debug("[SAM3 DEBUG] segmentImage response:", {
          session_id: data.session_id,
          labels: (data.labels ?? []).length,
          has_overlay: !!data.overlay_image_b64,
          has_id_map_rgb: !!data.id_map_rgb_b64,
        });

        // decode id_map_rgb_b64 for quick inspection (same method as SegmentViewer)
        if (data.id_map_rgb_b64) {
          decodeIdMapRgbB64(data.id_map_rgb_b64)
            .then(({ ids, w, h }) => {
              const unique = new Set<number>();
              for (let i = 0; i < Math.min(ids.length, 5000); i++) unique.add(ids[i]);
              console.debug("[SAM3 DEBUG] id_map_rgb sample:", { w, h, sample_unique_approx: unique.size });
              // store for interactive debug mapping if requested
              setDebugIdMap({ ids, w, h });
            })
            .catch((e) => console.warn("[SAM3 DEBUG] id_map decode failed", e));
        }
      }

      if ((data.labels?.length ?? 0) > 0) {
        setHistory((prev) => (mode === "simple" ? [data] : [...prev, data]));
      }
    } catch (err) {
      console.error(err);
      setError("Error llamando al backend. Verifica Uvicorn en http://127.0.0.1:8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleOcr = async () => {
    if (ocrFiles.length === 0) {
      setError("Primero sube 1 a 10 imágenes para OCR.");
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const data = await ocrBatch(ocrFiles);
      setOcrResult(data);
      setOcrSelectedIdx(0);
    } catch (err) {
      console.error(err);
      setError("Error llamando OCR al backend. Verifica el endpoint /api/ocr-batch.");
    } finally {
      setLoading(false);
    }
  };

  // Descarga CSV (POC)
  const downloadOcrCsv = () => {
    if (!ocrResult) return;

    const rows: string[] = [];
    rows.push(["filename", "codes"].join(","));

    for (const item of ocrResult.items) {
      const codes = (item.codes ?? []).join(" | ");
      const safeFilename = `"${(item.filename ?? "").replace(/"/g, '""')}"`;
      const safeCodes = `"${codes.replace(/"/g, '""')}"`;
      rows.push([safeFilename, safeCodes].join(","));
    }

    rows.push("");
    rows.push("unique_codes");
    for (const c of ocrResult.unique_codes ?? []) {
      rows.push(`"${String(c).replace(/"/g, '""')}"`);
    }

    const csv = rows.join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ocr_concentrado.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  // ===== Preview imagen original =====
  const ocrSelectedFile: File | null = useMemo(() => {
    if (toolMode !== "ocr") return null;
    if (!ocrFiles.length) return null;
    const idx = Math.min(Math.max(0, ocrSelectedIdx), ocrFiles.length - 1);
    return ocrFiles[idx] ?? null;
  }, [toolMode, ocrFiles, ocrSelectedIdx]);

  const sam3OriginalSrc = useObjectUrl((toolMode as ToolMode) === "sam3" ? file : null);
  const ocrOriginalSrc = useObjectUrl((toolMode as ToolMode) === "ocr" ? ocrSelectedFile : null);
  const originalSrc = (toolMode as ToolMode) === "sam3" ? sam3OriginalSrc : ocrOriginalSrc;

  // overlay and id_map are handled inside SegmentViewer now

  // ✅ FIX: ya no usamos class_name/num_objects; usamos classes_counts agregadas en history
  const aggregatedCounts = useMemo(() => {
    const counts = new Map<string, number>();

    for (const r of history) {
      const cc = r.classes_counts ?? {};
      for (const [cls, n] of Object.entries(cc)) {
        const key = formatClassName(cls);
        counts.set(key, (counts.get(key) ?? 0) + (n ?? 0));
      }
    }

    return Array.from(counts.entries())
      .map(([className, count]) => ({ className, count }))
      .sort((a, b) => b.count - a.count);
  }, [history]);

  const totalObjects = aggregatedCounts.reduce((acc, row) => acc + row.count, 0);
  const maxCount = aggregatedCounts.reduce((acc, row) => Math.max(acc, row.count), 0);

  // colorById is provided per-view inside SegmentViewer / panels

  // legacy id_map & hover pipeline removed; SegmentViewer handles these tasks now

  // ===== OCR helpers =====
  const ocrSelectedItem: OcrItem | null = useMemo(() => {
    if (!ocrResult?.items?.length) return null;
    const idx = Math.min(Math.max(0, ocrSelectedIdx), ocrResult.items.length - 1);
    return ocrResult.items[idx] ?? null;
  }, [ocrResult, ocrSelectedIdx]);

  const ocrPreviewSrc = useMemo(() => {
    if (!ocrSelectedItem?.preview_b64) return null;
    return `data:image/png;base64,${ocrSelectedItem.preview_b64}`;
  }, [ocrSelectedItem]);

  const ocrDetections: OcrDetection[] = (ocrSelectedItem?.detections ?? []) as OcrDetection[];

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
        <div className="text-[11px] text-slate-500 uppercase tracking-[0.25em]">Beta · Local Playground</div>
      </header>

      {/* MAIN GRID */}
      <main className="flex-1 p-5 bg-gradient-to-br from-slate-950 via-black to-slate-950">
        {/* Tool switcher SIEMPRE visible */}
        <div className="mb-4 flex gap-2">
          <button
            type="button"
            onClick={() => onToolModeChange("sam3")}
            className={`px-4 py-2 rounded-xl border ${toolMode === "sam3" ? "bg-emerald-500 text-black" : "bg-black/30 text-slate-200 border-slate-800"}`}
          >
            SAM3
          </button>

          <button
            type="button"
            onClick={() => onToolModeChange("ocr")}
            className={`px-4 py-2 rounded-xl border ${toolMode === "ocr" ? "bg-emerald-500 text-black" : "bg-black/30 text-slate-200 border-slate-800"}`}
          >
            OCR
          </button>

          <button
            type="button"
            onClick={() => onToolModeChange("fast_recon")}
            className={`px-4 py-2 rounded-xl border ${toolMode === "fast_recon" ? "bg-emerald-500 text-black" : "bg-black/30 text-slate-200 border-slate-800"}`}
          >
            Fast Reconstruction
          </button>

          <button
            type="button"
            onClick={() => onToolModeChange("sam3_compare")}
            className={`px-4 py-2 rounded-xl border ${toolMode === "sam3_compare" ? "bg-emerald-500 text-black" : "bg-black/30 text-slate-200 border-slate-800"}`}
          >
            SAM3COMPARE
          </button>
        </div>

        {/* Contenido */}
        {toolMode === "sam3_compare" ? (
          <Sam3CompareView />
        ) : toolMode === "fast_recon" ? (
          <FastReconstructionView />
        ) : toolMode === "sam3" ? (
          <div className="grid grid-cols-12 gap-4 h-[calc(100vh-110px)]">
            {/* Panel Izquierdo: controles compactos */}
            <div className="col-span-12 lg:col-span-3 h-full rounded-2xl border border-slate-900 bg-slate-950/70 p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-100">Controles</h2>
                <div className="text-[11px] text-slate-400">SAM3</div>
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
                  placeholder="Ej: columns"
                />
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

              <div className="mt-auto">
                <button
                  onClick={handleSubmit}
                  disabled={loading || !file}
                  className="w-full inline-flex items-center justify-center rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-sm font-semibold px-4 py-3 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_0_28px_rgba(16,185,129,0.45)] transition"
                >
                  {loading ? "Segmentando..." : "Segmentar"}
                </button>

                {loading && (
                  <div className="mt-3 rounded-xl border border-slate-900 bg-black/40 p-3 render-noise">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] text-slate-300">Processing</span>
                      <span className="text-[11px] text-slate-500">SAM3</span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                      <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
                    </div>
                    <p className="mt-2 text-[11px] text-slate-500">Segmentando máscaras + generando overlay…</p>
                  </div>
                )}
              </div>
            </div>

            {/* Centro: visor grande */}
            <div className="col-span-12 lg:col-span-6">
              <div className="rounded-2xl border border-slate-900 bg-black/35 p-3 h-full">
                <div className="text-sm opacity-80 mb-2">Resultado + Hover Highlight</div>

                <SegmentViewer
                  title="SAM3"
                  file={file}
                  result={lastResult}
                  hoverId={hoverId ?? undefined}
                  onHoverId={(id) => setHoverId(id === 0 ? null : id)}
                  loading={loading}
                />
              </div>
            </div>

            {/* Derecha: métricas/tabla */}
            <div className="col-span-12 lg:col-span-3">
              <div className="rounded-2xl border border-slate-900 bg-black/35 p-3 h-full flex flex-col gap-3">
                <div className="rounded-xl border border-white/10 bg-black/30 p-3">
                  <div className="text-xs opacity-70">Instancias</div>
                  <div className="text-2xl font-semibold">{lastResult?.labels?.length ?? 0}</div>
                  <div className="text-xs opacity-70 mt-2">Top clases</div>
                  <div className="text-sm">
                    {lastResult?.classes_counts
                      ? Object.entries(lastResult.classes_counts)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 3)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(" · ")
                      : "—"}
                  </div>
                </div>

                <div className="rounded-xl border border-white/10 bg-black/30 p-3">
                  <div className="text-xs opacity-70 mb-2">Hover</div>
                  <div className="text-sm">{hoverId ? `ID ${hoverId}` : "Pasa el mouse sobre una instancia"}</div>
                </div>

                <div className="rounded-xl border border-white/10 bg-black/30 p-3 overflow-auto max-h-[260px]">
                  <div className="text-xs opacity-70 mb-2">Instancias</div>
                  <table className="w-full text-xs">
                    <thead className="opacity-70">
                      <tr>
                        <th className="text-left py-1">ID</th>
                        <th className="text-left py-1">Clase</th>
                        <th className="text-left py-1">Score</th>
                        <th className="text-left py-1">Color</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(lastResult?.labels ?? []).map((it) => (
                        <tr
                          key={it.id}
                          className={`cursor-pointer ${hoverId === it.id ? "bg-white/10" : "hover:bg-white/5"}`}
                          onMouseEnter={() => setHoverId(it.id)}
                          onMouseLeave={() => setHoverId(null)}
                        >
                          <td className="py-1 pr-2">{it.id}</td>
                          <td className="py-1 pr-2">{it.class_name}</td>
                          <td className="py-1 pr-2">{it.score.toFixed(3)}</td>
                          <td className="py-1 pr-2">
                            <span className="inline-block w-3 h-3 rounded" style={{ background: it.color }} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-12 gap-5 h-[calc(100vh-110px)]">
          {/* Q1 */}
          <section className="col-span-12 lg:col-span-3 h-full rounded-2xl border border-slate-900 bg-slate-950/70 backdrop-blur-md p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">1. Panel de análisis</h2>
              <div className="text-[11px] text-slate-400">v0.4</div>
            </div>

            {/* TOOL TOGGLE */}
            <div className="flex items-center justify-between pt-1">
              <span className="text-[11px] text-slate-400 uppercase tracking-widest">Tool</span>
              <div className="inline-flex rounded-full bg-slate-900/80 p-0.5 text-[11px]">
                <button
                  onClick={() => onToolModeChange("sam3")}
                  className={`px-3 py-1 rounded-full transition ${
                    (toolMode as ToolMode) === "sam3" ? "bg-emerald-500 text-slate-950" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  SAM3
                </button>

                <button
                  onClick={() => onToolModeChange("sam3_compare")}
                  className={`px-3 py-1 rounded-full transition ${
                      (toolMode as ToolMode) === "sam3_compare"
                        ? "bg-emerald-500 text-slate-950"
                        : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  SAM3 COMPARE
                </button>

                <button
                  onClick={() => onToolModeChange("fast_recon")}
                  className={`px-3 py-1 rounded-full transition ${
                    (toolMode as ToolMode) === "fast_recon" ? "bg-emerald-500 text-slate-950" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  FAST RECON
                </button>

                <button
                  onClick={() => onToolModeChange("ocr")}
                  className={`px-3 py-1 rounded-full transition ${
                    (toolMode as ToolMode) === "ocr" ? "bg-emerald-500 text-slate-950" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  OCR
                </button>
              </div>
            </div>

            {/* STEP 1 UPLOAD */}
            <div className="space-y-2 text-xs">
              <p className="text-[11px] text-slate-400">
                <span className="font-semibold text-emerald-400">Paso 1 ·</span>{" "}
                {(toolMode as ToolMode) === "sam3" ? "Sube una imagen" : "Sube hasta 10 imágenes (OCR)"}
              </p>
                <input
                type="file"
                accept="image/*"
                multiple={(toolMode as ToolMode) === "ocr"}
                onChange={onFileChange}
                className="block w-full text-[11px] file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-[11px] file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-400 cursor-pointer"
              />
                {DEBUG && (toolMode as ToolMode) === "sam3" && (
                  <div className="mt-2">
                    <button
                      type="button"
                      onClick={async () => {
                        setDebugMappingActive((v) => !v);
                      }}
                      className={`mt-2 w-full inline-flex items-center justify-center rounded-md px-3 py-2 text-[12px] font-semibold ${
                        debugMappingActive ? "bg-emerald-500 text-slate-900" : "bg-slate-800 text-slate-200"
                      }`}
                      title="Toggle DEBUG mapping (requires id_map_rgb_b64)"
                    >
                      {debugMappingActive ? "DEBUG: Mapping ON" : "DEBUG: Mapping OFF"}
                    </button>
                  </div>
                )}
              {(toolMode as ToolMode) === "ocr" && (
                <div className="text-[11px] text-slate-500">
                  Cargadas: <span className="text-slate-200 font-semibold">{ocrFiles.length}</span> / 10
                </div>
              )}
            </div>

            {/* SAM3 CONTROLS */}
            {(toolMode as ToolMode) === "sam3" && (
              <>
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
                        mode === "simple"
                          ? "bg-emerald-500 text-slate-950"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Simple
                    </button>
                    <button
                      onClick={() => onModeChange("multi")}
                      className={`px-3 py-1 rounded-full transition ${
                        mode === "multi"
                          ? "bg-emerald-500 text-slate-950"
                          : "text-slate-400 hover:text-slate-200"
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
                  <div className="mt-3 rounded-xl border border-slate-900 bg-black/40 p-3 render-noise">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] text-slate-300">Processing</span>
                      <span className="text-[11px] text-slate-500">SAM3</span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                      <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
                    </div>
                    <p className="mt-2 text-[11px] text-slate-500">Segmentando máscaras + generando overlay…</p>
                  </div>
                )}
              </>
            )}

            {/* OCR CONTROLS */}
            {toolMode === "ocr" && (
              <>
                <button
                  onClick={handleOcr}
                  disabled={loading || ocrFiles.length === 0}
                  className="mt-2 inline-flex items-center justify-center rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-sm font-semibold px-4 py-3 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_0_28px_rgba(16,185,129,0.45)] transition"
                >
                  {loading ? "Leyendo..." : "Extraer claves (OCR)"}
                </button>

                <div className="flex gap-2">
                  <button
                    onClick={downloadOcrCsv}
                    disabled={!ocrResult}
                    className="inline-flex flex-1 items-center justify-center rounded-xl bg-slate-200/10 hover:bg-slate-200/15 text-slate-100 text-[12px] font-semibold px-3 py-2 border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
                  >
                    Descargar CSV
                  </button>
                  <button
                    onClick={() => {
                      setOcrResult(null);
                      setOcrSelectedIdx(0);
                    }}
                    disabled={!ocrResult}
                    className="inline-flex items-center justify-center rounded-xl bg-slate-200/10 hover:bg-slate-200/15 text-slate-100 text-[12px] font-semibold px-3 py-2 border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
                    title="Limpiar resultados"
                  >
                    Limpiar
                  </button>
                </div>
              </>
            )}

            {loading && (
              <div className="rounded-xl border border-slate-900 bg-black/40 p-3 render-noise">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] text-slate-300">Processing</span>
                  <span className="text-[11px] text-slate-500">{(toolMode as ToolMode) === "sam3" ? "SAM3" : "OCR"}</span>
                </div>
                <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                  <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
                </div>
                <p className="mt-2 text-[11px] text-slate-500">
                    {(toolMode as ToolMode) === "sam3"
                    ? "Segmentando máscaras + generando overlay…"
                    : "Detectando texto + destacando regiones…"}
                </p>
              </div>
            )}

            {error && (
              <div className="text-[11px] text-red-300 bg-red-950/40 border border-red-900 rounded-xl px-3 py-3">
                {error}
              </div>
            )}

            {/* OCR summary box */}
            {toolMode === "ocr" && (
              <div className="rounded-2xl border border-slate-900 bg-black/35 p-4">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-slate-400 uppercase tracking-widest">OCR Resultado</span>
                  <span className="text-[11px] text-slate-500">
                    {ocrResult ? `${ocrResult.unique_codes?.length ?? 0} claves únicas` : "--"}
                  </span>
                </div>

                {ocrResult ? (
                  <>
                    <div className="mt-3 text-[11px] text-slate-500">
                      Procesadas: <span className="text-slate-200 font-semibold">{ocrResult.items.length}</span>{" "}
                      imágenes.
                    </div>

                    <div className="mt-3">
                      {(ocrResult.unique_codes ?? []).length === 0 ? (
                        <div className="text-[11px] text-slate-500">No se detectaron claves con el patrón actual.</div>
                      ) : (
                        <div className="flex flex-wrap gap-2">
                          {(ocrResult.unique_codes ?? []).slice(0, 18).map((c) => (
                            <span
                              key={c}
                              className="text-[11px] px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-200"
                            >
                              {c}
                            </span>
                          ))}
                          {(ocrResult.unique_codes ?? []).length > 18 && (
                            <span className="text-[11px] px-2 py-1 rounded-full bg-slate-200/10 border border-slate-800 text-slate-300">
                              +{(ocrResult.unique_codes ?? []).length - 18} más
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="mt-3 text-[11px] text-slate-500">Ejecuta OCR para ver el resumen.</div>
                )}
              </div>
            )}
          </section>

          {/* Q2 */}
          <section className="col-span-12 lg:col-span-5 h-full rounded-2xl border border-slate-900 bg-slate-950/40 p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">2. Imagen original</h2>
                <span className="text-[11px] text-slate-500">
                {(toolMode as ToolMode) === "sam3"
                  ? file
                    ? file.name
                    : "—"
                  : ocrFiles.length > 0
                    ? `${ocrFiles[Math.min(ocrSelectedIdx, ocrFiles.length - 1)].name} (${ocrSelectedIdx + 1}/${ocrFiles.length})`
                    : "—"}
              </span>
            </div>

            <div className="flex-1 rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex items-center justify-center">
              {originalSrc ? (
                <img src={originalSrc} alt="Imagen original" className="max-h-full max-w-full object-contain" />
              ) : (
                <div className="text-slate-500 text-sm px-6 text-center">
                  {(toolMode as ToolMode) === "sam3"
                    ? "Sube una imagen para comenzar."
                    : "Sube 1 a 10 imágenes para OCR y luego ejecuta extracción."}
                </div>
              )}
            </div>
          </section>

          {/* Right column */}
          <section className="col-span-12 lg:col-span-4 h-full flex flex-col gap-5">
            {/* Q3 */}
            <div className="flex-1 rounded-2xl border border-slate-900 bg-slate-950/40 p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-100">
                  {(toolMode as ToolMode) === "sam3" ? "3. Resultado + Hover Highlight" : "3. OCR Resultados (por imagen)"}
                </h2>
                <span className="text-[11px] text-slate-500">
                  {(toolMode as ToolMode) === "sam3" ? `${labels.length} instancias` : `${ocrResult?.items.length ?? 0} imágenes`}
                </span>
              </div>

              {/* SAM3 */}
              {(toolMode as ToolMode) === "sam3" ? (
                <div className="flex gap-4 h-full">
                  <div className="flex-1 min-w-0">
                    <SegmentViewer
                      title="SAM3 · Imagen"
                      file={file}
                      result={lastResult}
                      hoverId={hoverId ?? undefined}
                      onHoverId={setHoverId}
                      loading={loading}
                    />
                  </div>

                  <ResultSidePanel
                    title="Métricas"
                    result={lastResult}
                    hoverId={hoverId ?? undefined}
                    onHoverId={setHoverId}
                  />
                </div>
              ) : (
                // OCR
                <div className="flex-1 rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex flex-col">
                  {!ocrResult ? (
                    <div className="flex-1 flex items-center justify-center text-slate-500 text-sm px-6 text-center">
                      Sube imágenes y ejecuta <span className="font-semibold mx-1">“Extraer claves (OCR)”</span>.
                    </div>
                  ) : (
                    <div className="flex-1 grid grid-cols-12 gap-3 p-3">
                      {/* Lista de items */}
                      <div className="col-span-5 rounded-2xl border border-slate-900 bg-slate-950/40 overflow-hidden flex flex-col">
                        <div className="px-3 py-2 bg-slate-900/60 flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-slate-200">Imágenes</span>
                          <span className="text-[11px] text-slate-400">{ocrResult.items.length}</span>
                        </div>

                        <div className="grid grid-cols-3 px-3 py-2 text-[11px] text-slate-400 border-t border-slate-900/70">
                          <span>Idx</span>
                          <span>Claves</span>
                          <span className="text-right">#</span>
                        </div>

                        <div className="flex-1 overflow-auto">
                          {ocrResult.items.map((it, idx) => {
                            const active = idx === ocrSelectedIdx;
                            return (
                              <button
                                key={`${it.filename}-${idx}`}
                                className={`w-full text-left grid grid-cols-3 px-3 py-2 text-[11px] border-t border-slate-900/60 transition ${
                                  active ? "bg-emerald-500/10" : "hover:bg-slate-200/5"
                                }`}
                                onClick={() => setOcrSelectedIdx(idx)}
                                type="button"
                              >
                                <span className="text-slate-200 font-semibold">{idx + 1}</span>
                                <span className="text-slate-300 truncate" title={(it.codes ?? []).join(" | ")}>
                                  {(it.codes ?? []).join(" | ") || "--"}
                                </span>
                                <span className="text-right text-slate-200">{(it.codes ?? []).length}</span>
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      {/* Detalle del seleccionado */}
                      <div className="col-span-7 rounded-2xl border border-slate-900 bg-slate-950/40 overflow-hidden flex flex-col">
                        <div className="px-3 py-2 bg-slate-900/60 flex items-center justify-between">
                          <div className="min-w-0">
                            <div className="text-[11px] text-slate-400">Seleccionado</div>
                            <div className="text-[11px] text-slate-200 font-semibold truncate">
                              {ocrSelectedItem?.filename ?? "--"}
                            </div>
                          </div>
                          <div className="text-[11px] text-slate-400">
                            {ocrSelectedIdx + 1}/{ocrResult.items.length}
                          </div>
                        </div>

                        <div className="p-3 flex-1 flex flex-col gap-3 overflow-auto">
                          {/* Preview con boxes */}
                          <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex items-center justify-center">
                            {ocrPreviewSrc ? (
                              <img src={ocrPreviewSrc} alt="OCR preview" className="max-h-[320px] w-auto object-contain" />
                            ) : (
                              <div className="text-[11px] text-slate-500 px-4 py-10 text-center">
                                No hay preview_b64 disponible (backend aún no lo genera).
                              </div>
                            )}
                          </div>

                          {/* Codes */}
                          <div className="rounded-2xl border border-slate-900 bg-black/35 p-3">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-[11px] font-semibold text-slate-200">Claves detectadas</span>
                              <span className="text-[11px] text-slate-400">{(ocrSelectedItem?.codes ?? []).length}</span>
                            </div>

                            {(ocrSelectedItem?.codes ?? []).length === 0 ? (
                              <div className="text-[11px] text-slate-500">--</div>
                            ) : (
                              <div className="flex flex-wrap gap-2">
                                {(ocrSelectedItem?.codes ?? []).map((c) => (
                                  <span
                                    key={c}
                                    className="text-[11px] px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-200"
                                  >
                                    {c}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>

                          {/* Detections table */}
                          <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden">
                            <div className="px-3 py-2 bg-slate-900/60 flex items-center justify-between">
                              <span className="text-[11px] font-semibold text-slate-200">Detections</span>
                              <span className="text-[11px] text-slate-400">{ocrDetections.length}</span>
                            </div>

                            <div className="grid grid-cols-3 px-3 py-2 text-[11px] text-slate-400 border-t border-slate-900/70">
                              <span>Text</span>
                              <span>Clean</span>
                              <span className="text-right">Conf</span>
                            </div>

                            <div className="max-h-[160px] overflow-auto">
                              {ocrDetections.length === 0 ? (
                                <div className="px-3 py-3 text-[11px] text-slate-500">
                                  Sin detecciones (o backend aún no devuelve detections).
                                </div>
                              ) : (
                                ocrDetections.map((d, i) => (
                                  <div
                                    key={`${d.clean || d.text}-${i}`}
                                    className="grid grid-cols-3 px-3 py-2 text-[11px] border-t border-slate-900/60"
                                  >
                                    <span className="text-slate-200 truncate" title={d.text}>
                                      {d.text}
                                    </span>
                                    <span className="text-slate-300 truncate" title={d.clean}>
                                      {d.clean}
                                    </span>
                                    <span className="text-right text-slate-200">{(d.conf ?? 0).toFixed(2)}</span>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>

                          {/* Raw text */}
                          <div className="rounded-2xl border border-slate-900 bg-black/35 p-3">
                            <div className="text-[11px] font-semibold text-slate-200 mb-2">Raw text</div>
                            <pre className="text-[11px] text-slate-300 whitespace-pre-wrap break-words">
                              {ocrSelectedItem?.raw_text ?? "--"}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tabla instancias (SAM3) */}
              {(toolMode as ToolMode) === "sam3" && labels.length > 0 && (
                <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden">
                  <div className="px-4 py-3 bg-slate-900/60 flex items-center justify-between">
                    <span className="text-[11px] font-semibold text-slate-200">Instancias (hover para resaltar)</span>
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
                          onMouseLeave={() => setHoverId(null)}
                      >
                        <span className="font-semibold text-slate-100">#{lbl.id}</span>
                        {/* ✅ FIX: clase por instancia */}
                        <span className="text-slate-300">{formatClassName(lbl.class_name)}</span>
                        <span className="text-right text-slate-200">{lbl.score.toFixed(3)}</span>
                        <span className="text-right">
                          <span className="inline-block w-3 h-3 rounded-full" style={{ background: lbl.color }} />
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Q4 (solo SAM3) */}
            {(toolMode as ToolMode) === "sam3" && (
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
                      <div className="px-4 py-3 text-[11px] text-slate-500">Ejecuta segmentaciones para ver conteos aquí.</div>
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
                    <div className="h-full flex items-center justify-center text-[11px] text-slate-500">Sin datos todavía.</div>
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
            )}
          </section>
          </div>
        )}
      </main>
    </div>
  );
}
