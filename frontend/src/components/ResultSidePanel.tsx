// frontend/src/components/ResultSidePanel.tsx
import { useMemo } from "react";
import type { SegmentResponse, InstanceLabel } from "../api";

function formatClassName(s: string) {
  return (s ?? "").trim().toLowerCase() || "--";
}

function buildCountsFromLabels(labels: InstanceLabel[]) {
  const m = new Map<string, number>();
  for (const l of labels) {
    const k = formatClassName(l.class_name);
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return Array.from(m.entries())
    .map(([className, count]) => ({ className, count }))
    .sort((a, b) => b.count - a.count);
}

function PieMini({
  data,
  size = 120,
}: {
  data: { className: string; count: number }[];
  size?: number;
}) {
  // mini donut/pie simple (SVG) – intencionalmente pequeño
  const total = data.reduce((a, b) => a + b.count, 0);
  const r = size * 0.40;
  const cx = size / 2;
  const cy = size / 2;
  const palette = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#22c55e"];

  let angle = 0;
  const slices = data.slice(0, 6).map((d, i) => {
    const frac = total > 0 ? d.count / total : 0;
    const start = angle;
    const end = angle + frac * 360;
    angle = end;

    const a0 = ((start - 90) * Math.PI) / 180;
    const a1 = ((end - 90) * Math.PI) / 180;

    const x0 = cx + r * Math.cos(a0);
    const y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy + r * Math.sin(a1);

    const large = end - start > 180 ? 1 : 0;

    const path = `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} Z`;
    return { path, color: palette[i % palette.length], label: d.className };
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="rgba(2,6,23,0.35)" />
      {slices.map((s, idx) => (
        <path key={idx} d={s.path} fill={s.color} opacity={0.9} stroke="rgba(15,23,42,0.8)" strokeWidth={1} />
      ))}
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle" fill="rgba(226,232,240,0.9)" fontSize={14} fontWeight={800}>
        {total}
      </text>
    </svg>
  );
}

export default function ResultSidePanel({
  title,
  result,
  hoverId,
  onHoverId,
}: {
  title?: string;
  result: SegmentResponse | null;
  hoverId?: number;
  onHoverId?: (id: number) => void;
}) {
  const labels = result?.labels ?? [];

  const counts = useMemo(() => buildCountsFromLabels(labels), [labels]);
  const total = labels.length;

  const activeLabel = useMemo(() => {
    if (!hoverId) return null;
    return labels.find((x) => x.id === hoverId) ?? null;
  }, [labels, hoverId]);

  return (
    <aside className="w-[320px] min-w-[280px] max-w-[360px] rounded-2xl border border-slate-900 bg-black/35 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="text-[11px] text-slate-400 uppercase tracking-widest">{title ?? "Métricas"}</div>
        <div className="text-[11px] text-slate-500">{total} instancias</div>
      </div>

      {/* Mini chart + top counts */}
      <div className="rounded-2xl border border-slate-900 bg-slate-950/40 p-3 flex gap-3 items-center">
        <PieMini data={counts} />
        <div className="flex-1 min-w-0">
          <div className="text-[11px] text-slate-400">Top clases</div>
          <div className="mt-1 space-y-1">
            {counts.slice(0, 4).map((c) => (
              <div key={c.className} className="flex items-center justify-between gap-2">
                <span className="text-[11px] text-slate-200 truncate">{c.className}</span>
                <span className="text-[11px] text-slate-400">{c.count}</span>
              </div>
            ))}
            {counts.length === 0 && <div className="text-[11px] text-slate-500">—</div>}
          </div>
        </div>
      </div>

      {/* Active hover card */}
      <div className="rounded-2xl border border-slate-900 bg-slate-950/40 p-3">
        <div className="text-[11px] text-slate-400 uppercase tracking-widest">Hover</div>
        {activeLabel ? (
          <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
            <div className="text-slate-400">ID</div>
            <div className="text-slate-200 font-semibold">#{activeLabel.id}</div>

            <div className="text-slate-400">Clase</div>
            <div className="text-slate-200">{formatClassName(activeLabel.class_name)}</div>

            <div className="text-slate-400">Score</div>
            <div className="text-slate-200">{activeLabel.score.toFixed(3)}</div>

            <div className="text-slate-400">Área (px)</div>
            <div className="text-slate-200">{activeLabel.area_px}</div>
          </div>
        ) : (
          <div className="mt-2 text-[11px] text-slate-500">Pasa el mouse sobre una instancia.</div>
        )}
      </div>

      {/* Instances table */}
      <div className="rounded-2xl border border-slate-900 bg-slate-950/40 overflow-hidden flex-1 min-h-[180px]">
        <div className="px-3 py-2 bg-slate-900/60 flex items-center justify-between">
          <span className="text-[11px] font-semibold text-slate-200">Instancias</span>
          <span className="text-[11px] text-slate-400">{labels.length}</span>
        </div>

        <div className="grid grid-cols-4 px-3 py-2 text-[11px] text-slate-400 border-t border-slate-900/70">
          <span>ID</span>
          <span>Clase</span>
          <span className="text-right">Score</span>
          <span className="text-right">Color</span>
        </div>

        <div className="max-h-[260px] overflow-auto">
          {labels.length === 0 ? (
            <div className="px-3 py-3 text-[11px] text-slate-500">—</div>
          ) : (
            labels.map((l) => {
              const active = hoverId === l.id;
              return (
                <button
                  type="button"
                  key={l.id}
                  className={`w-full text-left grid grid-cols-4 px-3 py-2 text-[11px] border-t border-slate-900/60 transition ${
                    active ? "bg-emerald-500/10" : "hover:bg-slate-200/5"
                  }`}
                  onMouseEnter={() => onHoverId?.(l.id)}
                  onMouseLeave={() => onHoverId?.(0)}
                >
                  <span className="text-slate-200 font-semibold">#{l.id}</span>
                  <span className="text-slate-300 truncate">{formatClassName(l.class_name)}</span>
                  <span className="text-right text-slate-200">{l.score.toFixed(3)}</span>
                  <span className="text-right">
                    <span className="inline-block w-3 h-3 rounded-full" style={{ background: l.color }} />
                  </span>
                </button>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}
