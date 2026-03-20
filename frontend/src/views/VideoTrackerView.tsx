import React, { useEffect, useMemo, useRef, useState } from "react";
import type { SegmentResponse } from "../api";
import { processVideo, segmentVideoFrame, videoFrameUrl } from "../api";
import { VideoSegPlayer } from "../components/VideoSegPlayer"; // ya lo tienes en tu repo

type VideoSessionState = {
  sessionId: string;
  fps: number;
  frameCount: number;
  duration: number;
  w: number;
  h: number;
};

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

export default function VideoTrackerView() {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const localVideoUrl = useMemo(() => (videoFile ? URL.createObjectURL(videoFile) : null), [videoFile]);

  const [session, setSession] = useState<VideoSessionState | null>(null);
  const [status, setStatus] = useState<"idle" | "processing" | "ready" | "searching" | "error">("idle");
  const [statusMsg, setStatusMsg] = useState<string>("");

  const [prompt, setPrompt] = useState("MACHINERY");
  const [threshold, setThreshold] = useState(0.5);

  const [seg, setSeg] = useState<SegmentResponse | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);

  // Para “Search entire video” (demo): samplea cada N frames
  const [searchProgress, setSearchProgress] = useState<{ i: number; total: number } | null>(null);

  // Throttle de requests en playback
  const lastReqRef = useRef<number>(0);
  const lastFrameRef = useRef<number>(-1);

  const onPickVideo = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setVideoFile(f);
    setSeg(null);
    setHoverId(null);
    setSession(null);
    setSearchProgress(null);

    if (!f) return;

    try {
      setStatus("processing");
      setStatusMsg("Processing video (extracting frames)...");
      const info = await processVideo(f);
      setSession({
        sessionId: info.session_id,
        fps: info.fps,
        frameCount: info.frame_count,
        duration: info.duration_s,
        w: info.width,
        h: info.height,
      });
      setStatus("ready");
      setStatusMsg("Video processed. Ready.");
    } catch (err: any) {
      setStatus("error");
      setStatusMsg(err?.message ?? "Error procesando video.");
    }
  };

  const runFrameSeg = async (frameIdx: number) => {
    if (!session) return;
    if (!prompt.trim()) return;

    const s = await segmentVideoFrame(session.sessionId, frameIdx, prompt.trim(), threshold);
    setSeg(s);
  };

  const onSearchEntireVideo = async () => {
    if (!session) return;
    if (!prompt.trim()) return;

    setStatus("searching");
    setStatusMsg("Searching entire video...");

    // sample cada 8 frames (ajusta: 1 = todo, 4 = más fino)
    const stride = 8;
    const total = Math.ceil(session.frameCount / stride);
    setSearchProgress({ i: 0, total });

    try {
      for (let k = 0, idx = 0; idx < session.frameCount; k++, idx += stride) {
        setSearchProgress({ i: k + 1, total });
        await runFrameSeg(idx);

        // opcional: saltar el player al frame
        const v = videoRef.current;
        if (v) v.currentTime = idx / session.fps;
      }

      setStatus("ready");
      setStatusMsg("Search completed.");
    } catch (err: any) {
      setStatus("error");
      setStatusMsg(err?.message ?? "Error en search.");
    } finally {
      setSearchProgress(null);
    }
  };

  // Playback real: cada ~300ms dispara segmentación del frame actual
  const onTimeUpdate = async () => {
    const v = videoRef.current;
    if (!v || !session) return;
    if (status !== "ready") return;
    if (!prompt.trim()) return;

    const now = Date.now();
    if (now - lastReqRef.current < 300) return;

    const frameIdx = clamp(Math.floor(v.currentTime * session.fps), 0, session.frameCount - 1);
    if (frameIdx === lastFrameRef.current) return;

    lastReqRef.current = now;
    lastFrameRef.current = frameIdx;

    try {
      await runFrameSeg(frameIdx);
    } catch {
      // no rompas playback por un frame fallido
    }
  };

  // UI
  const isReady = status === "ready";
  const canSearch = isReady && !!session && !!prompt.trim();

  return (
    <div className="page">
      <div className="toolbar">
        <div className="toolbarRow">
          <input className="fileInput" type="file" accept="video/*" onChange={onPickVideo} />

          <div className="field">
            <div className="label">Process video</div>
            <div className="hint">{status === "processing" ? "Processing..." : statusMsg}</div>
          </div>

          <div className="field">
            <div className="label">Prompt</div>
            <input
              className="input"
              value={prompt}
              disabled={!isReady}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. machinery, column, rebar..."
            />
          </div>

          <div className="field small">
            <div className="label">Threshold</div>
            <input
              className="input"
              type="number"
              step="0.05"
              min="0"
              max="1"
              disabled={!isReady}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>

          <button className="btnRun" disabled={!canSearch} onClick={onSearchEntireVideo}>
            Search entire video
          </button>

          <div className="spacer" />

          <button
            className="btnGhost"
            disabled={!isReady || !videoRef.current}
            onClick={() => videoRef.current?.play()}
          >
            Play
          </button>
        </div>

        {status === "searching" && searchProgress && (
          <div className="progressRow">
            <div className="progressText">
              Searching entire video... ({searchProgress.i}/{searchProgress.total})
            </div>
            <div className="progressBar">
              <div
                className="progressFill"
                style={{ width: `${Math.round((searchProgress.i / searchProgress.total) * 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="grid2">
        {/* VIDEO + OVERLAY */}
        <div className="card">
          <div className="cardTitle">Video preview</div>

          <div className="videoStage">
            {/* VideoSegPlayer dibuja overlay (fill + outline) encima */}
            <VideoSegPlayer
              videoRef={videoRef}
              src={localVideoUrl}
              segmentData={seg}
              hoverId={hoverId ?? undefined}
              onHoverId={(id) => setHoverId(id)}
              onTimeUpdate={onTimeUpdate}
            />
          </div>
        </div>

        {/* ANALYTICS */}
        <div className="card">
          <div className="cardTitle">Analytics</div>

          <div className="analyticsBlock">
            <div className="kv">
              <span>Instances</span>
              <span>{(seg as any)?.labels?.length ?? 0}</span>
            </div>

            <div className="sectionTitle">Classes</div>

            {(seg as any)?.labels?.length ? (
              <div className="list">
                {(seg as any).labels.map((lab: string, i: number) => {
                  const score = (seg as any)?.scores?.[i];
                  // color demo: si tu SegmentResponse ya trae colors, úsalo
                  const color = (seg as any)?.colors?.[i] ?? `hsl(${(i * 47) % 360} 80% 55%)`;

                  return (
                    <button
                      key={`${lab}-${i}`}
                      className="rowBtn"
                      onMouseEnter={() => setHoverId(i)}
                      onMouseLeave={() => setHoverId(null)}
                    >
                      <span className="swatch" style={{ background: color }} />
                      <span className="rowMain">{lab}</span>
                      <span className="rowMeta">{typeof score === "number" ? score.toFixed(3) : ""}</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="hint">Sin resultados aún.</div>
            )}
          </div>
        </div>
      </div>

      {/* Timeline (placeholder): para MVP solo muestra una fila vacía o mini thumbs si las agregas después */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="cardTitle">Timeline (thumbnails)</div>
        <div className="hint">
          (Siguiente iteración) Generar thumbs en backend y pintarlos aquí como en Meta.
        </div>
      </div>
    </div>
  );
}
