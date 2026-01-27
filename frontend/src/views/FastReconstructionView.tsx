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

  // New controls: point size, real decimation (stride) and style
  const [pcPointSize, setPcPointSize] = useState<number>(0.015);
  const [pcStride, setPcStride] = useState<number>(2);
  const [pcStyle, setPcStyle] = useState<"points" | "spheres">("points");

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
      <div className="grid grid-cols-12 gap-4 flex-1 min-h-0">
        {/* LEFT 50%: Original + Depth side-by-side */}
        <div className="col-span-12 lg:col-span-6 min-h-0">
          <div className="grid grid-cols-2 gap-4 h-full min-h-0">
            <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex items-center justify-center min-h-0">
              {originalSrc ? (
                <img src={originalSrc} alt="Original" className="h-full w-full object-contain" />
              ) : (
                <div className="text-slate-500 text-sm px-6 text-center">Sube una imagen.</div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden flex items-center justify-center min-h-0">
              {depthSrc ? (
                <img src={depthSrc} alt="Depth" className="h-full w-full object-contain" />
              ) : (
                <div className="text-slate-500 text-sm px-6 text-center">Aquí aparecerá el depth map.</div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT 50%: Point Cloud viewer full */}
        <div className="col-span-12 lg:col-span-6 min-h-0">
          <div className="rounded-2xl border border-slate-900 bg-black/35 overflow-hidden h-full min-h-0 flex flex-col">
            {/* header/controls */}
            <div className="p-3 border-b border-slate-900 flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-100">Point Cloud</div>
              {/* deja aquí tus controles actuales (RGB/Scalar/Downsample/PointSize) */}
            </div>

            <div className="p-3">
              <div className="mt-2 grid grid-cols-12 gap-3 text-xs text-white/80">
                <label className="col-span-6 flex flex-col gap-1">
                  <span>Point size: {pcPointSize.toFixed(3)}</span>
                  <input
                    type="range"
                    min={0.002}
                    max={0.08}
                    step={0.001}
                    value={pcPointSize}
                    onChange={(e) => setPcPointSize(parseFloat(e.target.value))}
                  />
                </label>

                <label className="col-span-6 flex flex-col gap-1">
                  <span>Decimation (stride): {pcStride}</span>
                  <input
                    type="range"
                    min={1}
                    max={12}
                    step={1}
                    value={pcStride}
                    onChange={(e) => setPcStride(parseInt(e.target.value, 10))}
                  />
                </label>

                <label className="col-span-6 flex items-center gap-2">
                  <span>Style</span>
                  <select
                    className="rounded bg-black/30 px-2 py-1"
                    value={pcStyle}
                    onChange={(e) => setPcStyle(e.target.value as any)}
                  >
                    <option value="points">Points</option>
                    <option value="spheres">Spheres (heavy)</option>
                  </select>
                </label>
              </div>
            </div>

            {/* viewer */}
            <div className="flex-1 min-h-0">
              {plyB64 ? (
                <PointCloudViewer
                  plyB64={plyB64}
                  colorMode={pcColorMode}
                  scalarField={pcScalar}
                  downsample={pcDownsample}
                  pointSize={pcPointSize}
                  stride={pcStride}
                  style={pcStyle}
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
