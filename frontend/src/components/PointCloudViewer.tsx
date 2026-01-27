import React, { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PLYLoader } from "three-stdlib";

type Props = {
  plyB64?: string;          // base64 del .ply (binario)
  pointSize?: number;       // tamaño del punto
  height?: number;          // altura del visor
  colorMode?: "rgb" | "scalar";
  scalarField?: "z" | "depth";
  downsample?: number;
  originalImageSrc?: string | undefined;
  meta?: any;
};

function b64ToBlobUrl(b64: string, mime = "application/octet-stream") {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

function PointsFromPLY({ url, pointSize, colorMode, scalarField, downsample, imgData, meta }: { url: string; pointSize: number; colorMode?: "rgb" | "scalar"; scalarField?: "z" | "depth"; downsample?: number; imgData?: ImageData | null; meta?: any }) {
  const [geom, setGeom] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    let alive = true;
    const loader = new PLYLoader();
    loader.load(
      url,
      (g) => {
        if (!alive) return;
        g.computeVertexNormals?.();

        try {
          // If requested, override vertex colors using the provided imgData and meta (row-major mapping)
          if (colorMode === "rgb" && imgData && meta) {
            const pos = g.getAttribute("position");
            const count = pos.count;
            const w = meta.w ?? meta.width;
            const h = meta.h ?? meta.height;
            if (w && h && w * h >= count) {
              const cols = new Float32Array(count * 3);
              for (let i = 0; i < count; i++) {
                const px = i % w;
                const py = Math.floor(i / w);
                const idx = (py * w + px) * 4;
                const r = imgData.data[idx] / 255.0;
                const gcol = imgData.data[idx + 1] / 255.0;
                const b = imgData.data[idx + 2] / 255.0;
                cols[i * 3 + 0] = r;
                cols[i * 3 + 1] = gcol;
                cols[i * 3 + 2] = b;
              }
              g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
            }
          }
        } catch (err) {
          console.warn("Failed to apply projected RGB colors:", err);
        }

        setGeom(g);
      },
      undefined,
      (err) => {
        console.error("PLY load error:", err);
      }
    );
    return () => {
      alive = false;
    };
  }, [url, colorMode, scalarField, downsample, imgData, meta]);

  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true, // si el PLY trae color por vértice
      sizeAttenuation: true,
    });
  }, [pointSize]);

  if (!geom) return null;

  // Si no trae color, se lo ponemos “blanco”
  if (!geom.getAttribute("color")) {
    const pos = geom.getAttribute("position");
    const colors = new Float32Array(pos.count * 3);
    colors.fill(1.0);
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }

  // Centrar en origen
  geom.computeBoundingBox();
  const bb = geom.boundingBox;
  if (bb) {
    const center = new THREE.Vector3();
    bb.getCenter(center);
    geom.translate(-center.x, -center.y, -center.z);
  }

  return <points geometry={geom} material={material} />;
}

export const PointCloudViewer: React.FC<Props> = ({
  plyB64,
  pointSize = 0.02,
  height = 520,
  colorMode = "rgb",
  scalarField = "z",
  downsample = 1,
  originalImageSrc,
  meta,
}) => {
  const url = useMemo(() => {
    if (!plyB64) return null;
    return b64ToBlobUrl(plyB64);
  }, [plyB64]);

  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [url]);

  // Load original image into ImageData (resized to meta.w/meta.h) for projected RGB sampling
  const [imgData, setImgData] = useState<ImageData | null>(null);
  useEffect(() => {
    if (!originalImageSrc || !meta) {
      setImgData(null);
      return;
    }
    const w = meta.w ?? meta.width ?? null;
    const h = meta.h ?? meta.height ?? null;
    if (!w || !h) {
      setImgData(null);
      return;
    }

    let alive = true;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (!alive) return;
      try {
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        if (!ctx) return setImgData(null);
        ctx.drawImage(img, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h);
        setImgData(data);
      } catch (err) {
        console.error("Failed to create image data for original image:", err);
        setImgData(null);
      }
    };
    img.onerror = () => setImgData(null);
    img.src = originalImageSrc;
    return () => {
      alive = false;
    };
  }, [originalImageSrc, meta]);

  return (
    <div className="w-full rounded border border-slate-700 overflow-hidden" style={{ height }}>
      <Canvas camera={{ position: [0, 0, 2.5], near: 0.01, far: 5000 }}>
        <ambientLight intensity={0.9} />
        <gridHelper args={[10, 10]} />
        <axesHelper args={[1]} />

        {url && (
          <PointsFromPLY url={url} pointSize={pointSize} colorMode={colorMode} scalarField={scalarField} downsample={downsample} imgData={imgData} meta={meta} />
        )}

        <OrbitControls makeDefault />
      </Canvas>
      {!url && (
        <div className="p-2 text-sm text-slate-400">No hay point cloud aún. Ejecuta “Fast Reconstruction” con make_ply=true.</div>
      )}
    </div>
  );
};
