// frontend/src/views/Sam3CompareView.tsx
import React, { useMemo, useState } from "react";
import { segmentImage, type SegmentResponse } from "../api";
import { SegmentViewer } from "../components/SegmentViewer";
import ResultSidePanel from "../components/ResultSidePanel";

export function Sam3CompareView() {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);

  const [prompt, setPrompt] = useState("columns");
  const [threshold, setThreshold] = useState(0.5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [resA, setResA] = useState<SegmentResponse | null>(null);
  const [resB, setResB] = useState<SegmentResponse | null>(null);
  const [hoverIdA, setHoverIdA] = useState<number>(0);
  const [hoverIdB, setHoverIdB] = useState<number>(0);

  const countsA = useMemo(() => resA?.classes_counts ?? {}, [resA]);
  const countsB = useMemo(() => resB?.classes_counts ?? {}, [resB]);

  const totalA = useMemo(() => Object.values(countsA).reduce((a, b) => a + (b ?? 0), 0), [countsA]);
  const totalB = useMemo(() => Object.values(countsB).reduce((a, b) => a + (b ?? 0), 0), [countsB]);

  const runCompare = async () => {
    if (!fileA || !fileB) {
      setError("Sube Imagen A e Imagen B.");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const [a, b] = await Promise.all([
        segmentImage(fileA, prompt, threshold),
        segmentImage(fileB, prompt, threshold),
      ]);
      setResA(a);
      setResB(b);
    } catch (e) {
      console.error(e);
      setError("Error llamando al backend. Verifica Uvicorn en http://127.0.0.1:8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="rounded-2xl border border-slate-900 bg-slate-950/70 p-4 flex flex-col gap-3">
        <div className="grid grid-cols-12 gap-3">
          <div className="col-span-12 lg:col-span-4">
            <div className="text-[11px] text-slate-400 mb-1">Imagen A</div>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFileA(e.target.files?.[0] ?? null)}
              className="block w-full text-[11px] file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-[11px] file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-400 cursor-pointer"
            />
          </div>

          <div className="col-span-12 lg:col-span-4">
            <div className="text-[11px] text-slate-400 mb-1">Imagen B</div>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFileB(e.target.files?.[0] ?? null)}
              className="block w-full text-[11px] file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-[11px] file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-400 cursor-pointer"
            />
          </div>

          <div className="col-span-12 lg:col-span-4 flex flex-col gap-2">
            <div className="grid grid-cols-12 gap-2">
              <div className="col-span-7">
                <div className="text-[11px] text-slate-400 mb-1">Prompt</div>
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full rounded-md bg-black/40 border border-slate-800 px-3 py-2 text-[12px] focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                />
              </div>
              <div className="col-span-5">
                <div className="text-[11px] text-slate-400 mb-1">Threshold</div>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full rounded-md bg-black/40 border border-slate-800 px-3 py-2 text-[12px] focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                />
              </div>
            </div>

            <button
              onClick={runCompare}
              disabled={loading || !fileA || !fileB}
              className="inline-flex items-center justify-center rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-sm font-semibold px-4 py-3 disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_0_28px_rgba(16,185,129,0.45)] transition"
            >
              {loading ? "Corriendo compare..." : "Run SAM3 Compare"}
            </button>

            {loading && (
              <div className="mt-3 rounded-xl border border-slate-900 bg-black/40 p-3 render-noise">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] text-slate-300">Processing</span>
                  <span className="text-[11px] text-slate-500">SAM3 COMPARE</span>
                </div>
                <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
                  <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
                </div>
                <p className="mt-2 text-[11px] text-slate-500">Segmentando máscaras + generando overlay…</p>
              </div>
            )}

            <div className="text-[11px] text-slate-400">
              Total A: <span className="text-slate-200 font-semibold">{totalA}</span>{" "}
              · Total B: <span className="text-slate-200 font-semibold">{totalB}</span>
            </div>
          </div>
        </div>

        {error && (
          <div className="text-[11px] text-red-300 bg-red-950/40 border border-red-900 rounded-xl px-3 py-3">
            {error}
          </div>
        )}
      </div>

      {/* Split view */}
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-6">
          <div className="flex gap-4 h-full">
            <div className="flex-1 min-w-0">
              <SegmentViewer
                title="SAM3 Compare · Imagen A"
                file={fileA}
                result={resA}
                hoverId={hoverIdA}
                onHoverId={setHoverIdA}
                loading={loading}
              />
            </div>

            <ResultSidePanel title="Métricas A" result={resA} hoverId={hoverIdA} onHoverId={setHoverIdA} />
          </div>
        </div>

        <div className="col-span-12 lg:col-span-6">
          <div className="flex gap-4 h-full">
            <div className="flex-1 min-w-0">
              <SegmentViewer
                title="SAM3 Compare · Imagen B"
                file={fileB}
                result={resB}
                hoverId={hoverIdB}
                onHoverId={setHoverIdB}
                loading={loading}
              />
            </div>

            <ResultSidePanel title="Métricas B" result={resB} hoverId={hoverIdB} onHoverId={setHoverIdB} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Sam3CompareView;
