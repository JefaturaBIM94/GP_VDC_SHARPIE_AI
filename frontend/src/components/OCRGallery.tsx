import { useMemo, useState } from "react";
import type { OcrItem } from "../api";

function pct(v?: number) {
  if (v === undefined || v === null || Number.isNaN(v)) return "--";
  return `${Math.round(v * 100)}%`;
}

function firstKey(it: OcrItem) {
  const k = (it.codes ?? [])[0];
  return k && k.trim().length ? k : "SIN LECTURA";
}

function statusClass(s?: string) {
  const x = (s ?? "RECHAZADO").toLowerCase();
  if (x === "ok") return "ocrCard--ok";
  if (x === "dudoso") return "ocrCard--dudoso";
  return "ocrCard--rechazado";
}

export default function OCRGallery({ items }: { items: OcrItem[] }) {
  const [active, setActive] = useState<OcrItem | null>(null);

  const cards = useMemo(() => items.slice(0, 10), [items]);

  return (
    <>
      <div className="ocrGrid">
        {cards.map((it, idx) => {
          const thumb = it.preview_b64 ? `data:image/png;base64,${it.preview_b64}` : null;

          return (
            <button
              key={`${it.filename}-${idx}`}
              type="button"
              className={`ocrCard ${statusClass(it.status)}`}
              onClick={() => setActive(it)}
              title={it.filename}
            >
              <div className="ocrThumbWrap">
                {thumb ? (
                  <img className="ocrThumb" src={thumb} alt={it.filename} />
                ) : (
                  <div className="ocrThumbPlaceholder">No preview</div>
                )}
              </div>

              <div className="ocrMeta">
                <div className="ocrKeyRow">
                  <span className="ocrKey">{firstKey(it)}</span>
                  <span className="ocrStatus">{it.status ?? "RECHAZADO"}</span>
                </div>
                <div className="ocrSubRow">
                  <span className="ocrFile">{it.filename}</span>
                  <span className="ocrConf">{pct(it.confidence)}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {active && (
        <div className="ocrModalBackdrop" onClick={() => setActive(null)}>
          <div className="ocrModal" onClick={(e) => e.stopPropagation()}>
            <div className="ocrModalHeader">
              <div className="ocrModalHeaderLeft">
                <div className="ocrModalTitle">{active.filename}</div>
                <div className="ocrModalSubtitle">
                  <b>{firstKey(active)}</b> · {pct(active.confidence)} ·{" "}
                  <span className={`ocrPill ${statusClass(active.status)}`}>
                    {active.status ?? "RECHAZADO"}
                  </span>
                </div>
              </div>
              <button className="ocrClose" onClick={() => setActive(null)}>
                ✕
              </button>
            </div>

            <div className="ocrModalBody">
              <div className="ocrModalImgs">
                {active.preview_b64 ? (
                  <img
                    src={`data:image/png;base64,${active.preview_b64}`}
                    className="ocrBig"
                    alt="preview"
                  />
                ) : (
                  <div className="ocrBigPlaceholder">No preview</div>
                )}
              </div>

              <div className="ocrDetections">
                <div className="ocrSectionTitle">Detections</div>
                <div className="ocrTableWrap">
                  <table className="ocrTable">
                    <thead>
                      <tr>
                        <th>Text</th>
                        <th>Clean</th>
                        <th style={{ textAlign: "right" }}>Conf</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(active.detections ?? []).map((d, i) => (
                        <tr key={`${d.clean || d.text}-${i}`}>
                          <td title={d.text}>{d.text}</td>
                          <td title={d.clean}>{d.clean}</td>
                          <td style={{ textAlign: "right" }}>
                            {typeof d.conf === "number" ? d.conf.toFixed(2) : "--"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="ocrSectionTitle" style={{ marginTop: 10 }}>
                  Raw text
                </div>
                <pre className="ocrRaw">{active.raw_text ?? "--"}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
