// frontend/src/api.ts

export type InstanceLabel = {
  id: number;
  class_name: string;      // clase a la que pertenece
  cx: number;              // [0..1]
  cy: number;              // [0..1]
  score: number;
  area_px: number;         // área en pixeles (backend)
  color: string;           // "#rrggbb"
};

export type SegmentResponse = {
  session_id: string;

  // === datos principales ===
  class_name: string;
  num_objects: number;
  threshold: number;

  // === imágenes ===
  overlay_image_b64: string;

  // === analytics ===
  classes_counts: Record<string, number>;

  // === instancias ===
  labels: InstanceLabel[];

  // === FUTURO (NO rompe nada aunque no exista aún) ===
  id_map_b64?: string;     // opcional (hover highlight por pixel)
};

export async function segmentImage(
  file: File,
  prompt: string,
  threshold: number
): Promise<SegmentResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("prompt", prompt);
  form.append("threshold", String(threshold));

  const res = await fetch("http://127.0.0.1:8000/api/segment", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText} - ${text}`);
  }

  return res.json();
}
