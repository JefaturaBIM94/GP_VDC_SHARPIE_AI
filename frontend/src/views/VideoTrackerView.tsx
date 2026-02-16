import React, { useMemo, useRef, useState } from "react";
import { segmentImage } from "../api";
import type { SegmentResponse, InstanceLabel } from "../api";
import { VideoSegPlayer } from "../components/VideoSegPlayer";

type FrameSample = {
  t: number;
  file: File;
  dataUrl: string;
};

function blobToFile(blob: Blob, fileName: string): File {
  return new File([blob], fileName, { type: blob.type || "image/png" });
}

async function captureFrameAsPng(videoEl: HTMLVideoElement): Promise<{ file: File; dataUrl: string }> {
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  if (!w || !h) throw new Error("El video aún no tiene dimensiones (videoWidth/videoHeight = 0).");

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("No se pudo obtener contexto 2D de canvas.");
  ctx.drawImage(videoEl, 0, 0, w, h);

  const dataUrl = canvas.toDataURL("image/png");

  const blob: Blob = await new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob devolvió null"))), "image/png");
  });

  return { file: blobToFile(blob, `frame_${Date.now()}.png`), dataUrl };
}

function waitSeek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Error haciendo seek en el video."));
    };
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = t;
  });
}

async function cropAtPoint(dataUrl: string, cx: number, cy: number, size = 96): Promise<string> {
  const img = new Image();
  img.src = dataUrl;

  await new Promise<void>((resolve) => {
    img.onload = () => resolve();
    img.onerror = () => resolve(); // fallback: si falla, regresamos vacío luego
  });

  if (!img.naturalWidth || !img.naturalHeight) return "";

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const x = Math.round(cx * img.naturalWidth);
  const y = Math.round(cy * img.naturalHeight);

  const half = Math.floor(size / 2);
  const sx = Math.max(0, Math.min(img.naturalWidth - size, x - half));
  const sy = Math.max(0, Math.min(img.naturalHeight - size, y - half));

  ctx.drawImage(img, sx, sy, size, size, 0, 0, size, size);
  return canvas.toDataURL("image/png");
}

export default function VideoTrackerView() {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const videoUrl = useMemo(() => (videoFile ? URL.createObjectURL(videoFile) : ""), [videoFile]);

  const [prompt, setPrompt] = useState("");
  const [threshold, setThreshold] = useState<number>(0.35);

  const [stage, setStage] = useState<"idle" | "preprocessing" | "ready" | "searching">("idle");
  const [progress, setProgress] = useState<{ label: string; p: number }>({ label: "", p: 0 });

  const [samples, setSamples] = useState<FrameSample[]>([]);
  const [results, setResults] = useState<Array<{ t: number; seg: SegmentResponse }>>([]);

  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const [error, setError] = useState("");

  const onPickVideo = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError("");
    setStage("idle");
    setProgress({ label: "", p: 0 });
    setSamples([]);
    setResults([]);
    setPlaying(false);

    const f = e.target.files?.[0] ?? null;
    setVideoFile(f);
  };

  const onPreprocess = async () => {
    setError("");
    const v = videoRef.current;
    if (!v || !videoFile) return;

    try {
      setStage("preprocessing");
      setPlaying(false);
      v.pause();

      // espera metadata
      if (!v.duration || Number.isNaN(v.duration)) {
        await new Promise<void>((resolve) => {
          const onMeta = () => resolve();
          v.addEventListener("loadedmetadata", onMeta, { once: true });
        });
      }

      const duration = v.duration;
      const N = 18; // thumbnails para timeline
      const times = Array.from({ length: N }, (_, i) => (duration * i) / Math.max(1, N - 1));

      const out: FrameSample[] = [];
      for (let i = 0; i < times.length; i++) {
        setProgress({ label: "Preprocessing video...", p: (i / times.length) * 100 });
        await waitSeek(v, times[i]);
        const cap = await captureFrameAsPng(v);
        out.push({ t: times[i], ...cap });
      }

      setSamples(out);
      setProgress({ label: "Preprocess listo", p: 100 });
      setStage("ready");
    } catch (err: any) {
      setError(err?.message ?? "Error en preprocess.");
      setStage("idle");
    }
  };

  const onSearchEntireVideo = async () => {
    setError("");
    const v = videoRef.current;
    if (!v || !samples.length) return;

    try {
      setStage("searching");
      setPlaying(false);
      v.pause();

      const out: Array<{ t: number; seg: SegmentResponse }> = [];
      for (let i = 0; i < samples.length; i++) {
        setProgress({ label: `Searching entire video... (${i + 1}/${samples.length})`, p: ((i + 1) / samples.length) * 100 });

        // usamos el frame sample ya capturado
        const seg = await segmentImage(samples[i].file, prompt, threshold);
        out.push({ t: samples[i].t, seg });
      }

      setResults(out);
      setStage("ready");
      setPlaying(true); // auto-play con overlays, estilo Meta
    } catch (err: any) {
      setError(err?.message ?? "Error buscando en el video.");
      setStage("ready");
    }
  };

  // Panel de analytics: toma resultado más cercano al tiempo actual y construye lista
  const closestSeg = useMemo(() => {
    if (!results.length) return null;
    let best = results[0];
    let d = Math.abs(results[0].t - currentTime);
    for (let i = 1; i < results.length; i++) {
      const nd = Math.abs(results[i].t - currentTime);
      if (nd < d) {
        best = results[i];
        d = nd;
      }
    }
    return best;
  }, [results, currentTime]);

  const closestFrame = useMemo(() => {
    if (!samples.length) return null;
    let best = samples[0];
    let d = Math.abs(samples[0].t - currentTime);
    for (let i = 1; i < samples.length; i++) {
      const nd = Math.abs(samples[i].t - currentTime);
      if (nd < d) {
        best = samples[i];
        d = nd;
      }
    }
    return best;
  }, [samples, currentTime]);

  const analytics = useMemo(() => {
    const seg = closestSeg?.seg;
    if (!seg) return { total: 0, byClass: [] as Array<{ k: string; n: number }>, labels: [] as InstanceLabel[] };
    const labels = seg.labels ?? [];
    const map = new Map<string, number>();
    labels.forEach((l) => map.set(l.class_name, (map.get(l.class_name) ?? 0) + 1));
    const byClass = Array.from(map.entries())
      .map(([k, n]) => ({ k, n }))
      .sort((a, b) => b.n - a.n);
    return { total: labels.length, byClass, labels };
  }, [closestSeg]);

  const isBusy = stage === "preprocessing" || stage === "searching";

  return (
    <div className="page">
      {/* CONTROLS */}
      <div className="toolbar">
        <div className="toolbarRow">
          <input type="file" accept="video/*" onChange={onPickVideo} />

          <button className="btn" onClick={onPreprocess} disabled={!videoFile || isBusy}>
            {stage === "preprocessing" ? "Procesando..." : "Process video"}
          </button>

          <label className="field">
            <span>Prompt</span>
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ej: cars, rebar, column..."
              disabled={stage !== "ready"}
            />
          </label>

          <label className="field" style={{ maxWidth: 140 }}>
            <span>Threshold</span>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              disabled={stage !== "ready"}
            />
          </label>

          <button className="btnRun" onClick={onSearchEntireVideo} disabled={stage !== "ready" || !prompt.trim() || isBusy || !samples.length}>
            Search entire video
          </button>

          <button className="btn" onClick={() => setPlaying((p) => !p)} disabled={!results.length || isBusy}>
            {playing ? "Pause" : "Play"}
          </button>
        </div>

        {isBusy && (
          <div className="progressRow">
            <div className="progressLabel">{progress.label}</div>
            <div className="progressBar">
              <div className="progressFill" style={{ width: `${progress.p}%` }} />
            </div>
          </div>
        )}

        {error && <div className="error">{error}</div>}
      </div>

      {/* LAYOUT: video + analytics */}
      <div className="grid2">
        <div className="card">
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Video preview</div>

          {/* VIDEO ELEMENT (hidden control) para preprocess (seek/capture) */}
          {videoFile && (
            <>
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                style={{
                  width: "100%",
                  maxHeight: 420,
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.08)",
                  marginBottom: 10,
                  background: "#000",
                }}
              />

              {/* PLAYER con overlays (se muestra cuando ya hay resultados) */}
              {results.length > 0 && (
                <VideoSegPlayer
                  videoUrl={videoUrl}
                  results={results}
                  playing={playing}
                  onTime={(t) => setCurrentTime(t)}
                  opacity={0.48} // fill opacity 45-50%
                />
              )}
            </>
          )}

          {/* TIMELINE */}
          {samples.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ opacity: 0.85, fontSize: 12, marginBottom: 8 }}>Timeline (thumbnails)</div>
              <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 6 }}>
                {samples.map((s) => (
                  <button
                    key={s.t}
                    className="thumb"
                    onClick={async () => {
                      const v = videoRef.current;
                      if (!v) return;
                      setPlaying(false);
                      v.pause();
                      await waitSeek(v, s.t);
                      setCurrentTime(s.t);
                    }}
                    title={`${s.t.toFixed(2)}s`}
                  >
                    <img src={s.dataUrl} style={{ width: 96, height: 54, objectFit: "cover", borderRadius: 8 }} />
                    <div style={{ fontSize: 11, opacity: 0.75 }}>{s.t.toFixed(1)}s</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ANALYTICS PANEL */}
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <div style={{ fontWeight: 800 }}>Analytics</div>
            <div style={{ opacity: 0.75, fontSize: 12 }}>t = {currentTime.toFixed(2)}s</div>
          </div>

          <div style={{ marginTop: 10, opacity: 0.85, fontSize: 12 }}>
            Instances: <b>{analytics.total}</b>
          </div>

          <div style={{ marginTop: 10 }}>
            <div style={{ fontWeight: 700, marginBottom: 6, opacity: 0.9 }}>Classes</div>
            <div style={{ display: "grid", gap: 6 }}>
              {analytics.byClass.map((c) => (
                <div key={c.k} className="pillRow">
                  <span className="pillKey">{c.k}</span>
                  <span className="pillVal">{c.n}</span>
                </div>
              ))}
              {!analytics.byClass.length && <div style={{ opacity: 0.6 }}>Sin resultados aún.</div>}
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 700, marginBottom: 6, opacity: 0.9 }}>Instances (preview)</div>

            <div style={{ display: "grid", gap: 10, maxHeight: 420, overflow: "auto", paddingRight: 6 }}>
              {analytics.labels.slice(0, 30).map((l) => (
                <div key={l.id} className="instanceRow">
                  <div
                    className="colorDot"
                    style={{ background: l.color }}
                    title={l.color}
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                      <div style={{ fontWeight: 700 }}>{l.class_name}</div>
                      <div style={{ opacity: 0.75, fontSize: 12 }}>score {l.score.toFixed(2)}</div>
                    </div>
                    <div style={{ opacity: 0.75, fontSize: 12 }}>
                      id {l.id} · area {Math.round(l.area_px)} px²
                    </div>
                  </div>

                  {/* mini-crop alrededor de centroid (cx/cy) usando el frame más cercano */}
                  <MiniCrop frameUrl={closestFrame?.dataUrl ?? ""} cx={l.cx} cy={l.cy} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MiniCrop({ frameUrl, cx, cy }: { frameUrl: string; cx: number; cy: number }) {
  const [src, setSrc] = useState<string>("");

  React.useEffect(() => {
    let alive = true;
    if (!frameUrl) {
      setSrc("");
      return;
    }
    cropAtPoint(frameUrl, cx, cy, 84).then((out) => {
      if (alive) setSrc(out);
    });
    return () => {
      alive = false;
    };
  }, [frameUrl, cx, cy]);

  return (
    <div style={{ width: 84, height: 84, borderRadius: 10, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.03)" }}>
      {src ? <img src={src} style={{ width: "100%", height: "100%", objectFit: "cover" }} /> : null}
    </div>
  );
}
