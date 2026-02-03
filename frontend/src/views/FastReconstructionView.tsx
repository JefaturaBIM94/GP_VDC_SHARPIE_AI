// frontend/src/views/FastReconstructionView.tsx
import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import { reconstructFast, type FastReconResponse } from "../api";
import { PointCloudViewer } from "../components/PointCloudViewer";

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

export default function FastReconstructionView() {
  const [file, setFile] = useState<File | null>(null);
  const [makePly, setMakePly] = useState<boolean>(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [result, setResult] = useState<FastReconResponse | null>(null);

  // Point cloud viewer controls
  const [plyB64, setPlyB64] = useState<string>("");
  const [pcColorMode] = useState<"rgb" | "scalar">("rgb");
  const [pcScalar] = useState<"z" | "depth">("z");
  const [pointSize, setPointSize] = useState<number>(0.016);
  const [stride, setStride] = useState<number>(4);
  const [maxRes, setMaxRes] = useState<number>(1024);
  const [style, setStyle] = useState<"points" | "spheres">("points");
  const downsample = 1;

  const originalSrc = useObjectUrl(file);

  const depthSrc = useMemo(() => {
    if (!result?.depth_png_b64) return null;
    return `data:image/png;base64,${result.depth_png_b64}`;
  }, [result]);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setResult(null);
    setPlyB64("");
    setError(null);
  };

  const run = async () => {
    if (!file) {
      setError("Primero sube una imagen.");
      return;
    }
    setResult(null);
    setPlyB64("");
    setError(null);
    setLoading(true);
    try {
      const data = await reconstructFast(file, {
        makePly,
        stride,
        maxRes,
      });
      setResult(data);
      console.log("FAST RECON meta:", data.meta);
      setPlyB64(data.ply_b64 || "");
      if (data.meta?.warning && !data.ply_b64) {
        setError(data.meta.warning);
      }
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
            <p className="mt-2 text-[11px] text-slate-500">Generando depth map + export 3D...</p>
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
                <div className="text-slate-500 text-sm px-6 text-center">Aqui aparecera el depth map.</div>
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
              {/* deja aqui tus controles actuales (RGB/Scalar/Downsample/PointSize) */}
            </div>

            <div className="p-3">
              <div className="mb-3 flex items-center gap-3 text-[11px] text-slate-300">
                <label className="flex items-center gap-2">
                  <span className="text-slate-200">Load PLY</span>
                  <input
                    type="file"
                    accept=".ply"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      const reader = new FileReader();
                      reader.onload = () => {
                        const text = reader.result as string;
                        setPlyB64(btoa(text));
                      };
                      reader.readAsText(f);
                    }}
                    className="block text-[11px]"
                  />
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-200">
                <div className="flex items-center gap-2">
                  <span className="w-20">Point Size</span>
                  <input
                    type="range"
                    min={0.002}
                    max={0.05}
                    step={0.001}
                    value={pointSize}
                    onChange={(e) => setPointSize(parseFloat(e.target.value))}
                  />
                  <span className="w-12 text-right">{pointSize.toFixed(3)}</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="w-28">Stride (server)</span>
                  <input
                    type="range"
                    min={2}
                    max={24}
                    step={1}
                    value={stride}
                    onChange={(e) => setStride(parseInt(e.target.value, 10))}
                  />
                  <span className="w-10 text-right">{stride}x</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="w-20">Max Res</span>
                  <input
                    type="range"
                    min={512}
                    max={2048}
                    step={64}
                    value={maxRes}
                    onChange={(e) => setMaxRes(parseInt(e.target.value, 10))}
                  />
                  <span className="w-14 text-right">{maxRes}px</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="w-12">Style</span>
                  <select
                    className="rounded bg-black/30 px-2 py-1"
                    value={style}
                    onChange={(e) => setStyle(e.target.value as any)}
                  >
                    <option value="points">Points</option>
                    <option value="spheres">Spheres</option>
                  </select>
                </div>
              </div>
            </div>

            {/* viewer */}
            <div className="flex-1 min-h-0">
              <PointCloudViewer
                plyB64={plyB64}
                colorMode={pcColorMode}
                scalarField={pcScalar}
                downsample={downsample}
                pointSize={pointSize}
                stride={stride}
                style={style}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
