// frontend/src/api.ts

export type InstanceLabel = {
  id: number;
  class_name: string;
  cx: number; // [0,1]
  cy: number; // [0,1]
  score: number;
  area_px: number;
  color: string; // "#rrggbb"
};

export type SegmentResponse = {
  session_id: string;
  threshold: number;

  // Mantengo por compatibilidad (tu backend aún lo manda)
  overlay_image_b64: string;

  // Legacy (16-bit PNG). NO confiable para hover si ids > 255.
  id_map_b64: string;

  // Nuevo (RGB 24-bit). Recomendado para hover/overlay.
  id_map_rgb_b64: string;

  classes_counts: Record<string, number>;
  labels: InstanceLabel[];
};

export type OcrDetection = {
  text: string;
  clean: string;
  conf: number;
  bbox: number[][];
};

export type OcrItem = {
  filename: string;
  raw_text: string;
  codes: string[];
  detections: OcrDetection[];
  preview_b64?: string;
  debug?: any;
};

export type OcrBatchResponse = {
  items: OcrItem[];
  unique_codes: string[];
};

const API_BASE =
  (import.meta as any).env?.VITE_API_BASE_URL?.toString()?.trim() ||
  "http://127.0.0.1:8000";

// -------------------- SAM3 --------------------
export async function segmentImage(file: File, prompt: string, threshold: number): Promise<SegmentResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("prompt", prompt);
  fd.append("threshold", String(threshold));

  const res = await fetch(`${API_BASE}/api/segment`, { method: "POST", body: fd });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`segmentImage failed: ${res.status} ${txt}`);
  }
  return (await res.json()) as SegmentResponse;
}

// -------------------- OCR --------------------
export async function ocrBatch(files: File[]): Promise<OcrBatchResponse> {
  const fd = new FormData();
  for (const f of files) fd.append("images", f);

  const res = await fetch(`${API_BASE}/api/ocr-batch`, { method: "POST", body: fd });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`ocrBatch failed: ${res.status} ${txt}`);
  }
  return (await res.json()) as OcrBatchResponse;
}


export type FastReconResponse = {
  depth_png_b64: string;
  ply_b64?: string | null;
  width: number;
  height: number;
};

export async function reconstructFast(file: File, makePly: boolean = true): Promise<FastReconResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("make_ply", String(makePly));

  const res = await fetch(`${API_BASE}/api/reconstruct-fast`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`reconstructFast failed: ${res.status} ${txt}`);
  }

  return (await res.json()) as FastReconResponse;
}
