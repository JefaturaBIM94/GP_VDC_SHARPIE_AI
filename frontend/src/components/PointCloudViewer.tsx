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
};

function b64ToBlobUrl(b64: string, mime = "application/octet-stream") {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

function PointsFromPLY({ url, pointSize }: { url: string; pointSize: number }) {
  const [geom, setGeom] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    let alive = true;
    const loader = new PLYLoader();
    loader.load(
      url,
      (g) => {
        if (!alive) return;
        g.computeVertexNormals?.();
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
  }, [url]);

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

  if (!url) {
    return (
      <div className="w-full rounded border border-slate-700 p-4 text-slate-300">
        No hay point cloud aún. Ejecuta “Fast Reconstruction” con make_ply=true.
      </div>
    );
  }

  return (
    <div className="w-full rounded border border-slate-700 overflow-hidden" style={{ height }}>
      <Canvas camera={{ position: [0, 0, 2.5], near: 0.01, far: 5000 }}>
        <ambientLight intensity={0.9} />
        <OrbitControls makeDefault />
        <PointsFromPLY url={url} pointSize={pointSize} />
      </Canvas>
    </div>
  );
};
