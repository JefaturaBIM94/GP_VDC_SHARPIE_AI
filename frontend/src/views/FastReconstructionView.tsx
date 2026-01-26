// frontend/src/views/FastReconstructionView.tsx
import React, { useMemo, useState } from "react";
import { reconstructFast, type FastReconResponse } from "../api";
import { PointCloudViewer } from "../components/PointCloudViewer";


function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);

  React.useEffect(() => {
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

export default function FastReconstructionView() {
  const [file, setFile] = useState<File | null>(null);
  const [makePly, setMakePly] = useState<boolean>(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [result, setResult] = useState<FastReconResponse | null>(null);

  // Point cloud viewer controls
  const [plyB64, setPlyB64] = useState<string>("");
  const [pcColorMode, setPcColorMode] = useState<"rgb" | "scalar">("rgb");
  const [pcScalar, setPcScalar] = useState<"z" | "depth">("z");
  const [pcDownsample, setPcDownsample] = useState<number>(1);
  const [pointSize, setPointSize] = useState<number>(0.01);

  const originalSrc = useObjectUrl(file);

  const depthSrc = useMemo(() => {
    if (!result?.depth_png_b64) return null;
    return `data:image/png;base64,${result.depth_png_b64}`;
  }, [result]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setError(null);
  };

  const run = async () => {
    if (!file) {
      setError("Primero sube una imagen.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const data = await reconstructFast(file, makePly);
    setResult(data);
    console.log("FAST RECON meta:", data.meta);
    setPlyB64(data.ply_b64 || "");
    } catch (e) {
      console.error(e);
      setError("Error llamando al backend. Verifica /api/reconstruct-fast en Uvicorn.");
    } finally {
      setLoading(false);
    }
  };

  const downloadPly = () => {
    if (!result?.ply_b64) return;
    const bin = atob(result.ply_b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

    const blob = new Blob([bytes], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "fast_reconstruction.ply";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full h-full flex flex-col gap-4">
      {/* Controls */}
      <div className="rounded-2xl border border-slate-900 bg-slate-950/40 p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-100">Fast Reconstruction</div>
          <div className="text-[11px] text-slate-500">Depth Map + (opcional) PLY</div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <input
            type="file"
            accept="image/*"
            onChange={onFileChange}
            className="block text-[11px] file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-[11px] file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-400 cursor-pointer"
          />

          <label className="flex items-center gap-2 text-[11px] text-slate-300">
            <input
              type="checkbox"
              checked={makePly}
              onChange={(e) => setMakePly(e.target.checked)}
              className="accent-emerald-500"
            />
            Generar PLY (preview 3D)
          </label>

          <button
            onClick={run}
            disabled={loading || !file}
            className="ml-auto inline-flex items-center justify-center rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-sm font-semibold px-4 py-2 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {loading ? "Reconstruyendo..." : "Run"}
          </button>

          <button
            onClick={downloadPly}
            disabled={!result?.ply_b64}
            className="inline-flex items-center justify-center rounded-xl bg-slate-200/10 hover:bg-slate-200/15 text-slate-100 text-[12px] font-semibold px-4 py-2 border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            Descargar PLY
          </button>
        </div>

        {error && (
          <div className="text-[11px] text-red-300 bg-red-950/40 border border-red-900 rounded-xl px-3 py-3">
            {error}
          </div>
        )}

        {loading && (
          <div className="rounded-xl border border-slate-900 bg-black/40 p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] text-slate-300">Processing</span>
              <span className="text-[11px] text-slate-500">Fast Reconstruction</span>
            </div>
            <div className="h-2 rounded-full bg-slate-900 overflow-hidden">
              <div className="loading-bar-inner h-full w-1/2 bg-emerald-400/80" />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">Generando depth map + export 3D…</p>
          </div>
        )}
      </div>

      {/* Viewers */}
      <div className="grid grid-cols-12 gap-4 flex-1">
        <div className="col-span-12 lg:col-span-6 rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex items-center justify-center">
          {originalSrc ? (
            <img src={originalSrc} alt="Original" className="max-h-full max-w-full object-contain" />
          ) : (
            <div className="text-slate-500 text-sm px-6 text-center">Sube una imagen para generar reconstrucción.</div>
          )}
        </div>

        <div className="col-span-12 lg:col-span-6 flex flex-col gap-3">
          <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex-1 flex items-center justify-center">
            {depthSrc ? (
              <img src={depthSrc} alt="Depth" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="text-slate-500 text-sm px-6 text-center">Aquí aparecerá el depth map.</div>
            )}
          </div>

          {/* === Point Cloud Panel (debajo del Depth Map) === */}
          <div className="rounded-lg border border-white/10 bg-white/5 p-3">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Point Cloud</h3>

              <div className="flex items-center gap-3 text-xs">
                <label className="flex items-center gap-1">
                  <input
                    type="radio"
                    checked={pcColorMode === "rgb"}
                    onChange={() => setPcColorMode("rgb")}
                  />
                  RGB
                </label>

                <label className="flex items-center gap-1">
                  <input
                    type="radio"
                    checked={pcColorMode === "scalar"}
                    onChange={() => setPcColorMode("scalar")}
                  />
                  Scalar
                </label>

                <select
                  className="rounded bg-black/30 px-2 py-1"
                  value={pcScalar}
                  onChange={(e) => setPcScalar(e.target.value as any)}
                  disabled={pcColorMode !== "scalar"}
                >
                  <option value="z">Height (Z)</option>
                  <option value="depth">Depth</option>
                </select>

                <label className="flex items-center gap-2">
                  Downsample
                  <select
                    className="rounded bg-black/30 px-2 py-1"
                    value={pcDownsample}
                    onChange={(e) => setPcDownsample(parseInt(e.target.value, 10))}
                  >
                    <option value={1}>1x</option>
                    <option value={2}>2x</option>
                    <option value={4}>4x</option>
                    <option value={8}>8x</option>
                  </select>
                </label>
              </div>
            </div>

            <div className="h-[420px] w-full overflow-hidden rounded-md bg-black">
              {plyB64 ? (
                <PointCloudViewer
                  plyB64={plyB64}
                  colorMode={pcColorMode}
                  scalarField={pcScalar}
                  downsample={pcDownsample}
                  pointSize={pointSize}
                  originalImageSrc={originalSrc ?? undefined}
                  meta={result?.meta}
                />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-white/60">
                  Run reconstruction with "make_ply" enabled to preview the point cloud.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
