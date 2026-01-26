// frontend/src/components/PointCloudViewer.tsx
import React, { useMemo } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PLYLoader } from "three-stdlib";

export type PointCloudColorMode = "rgb" | "scalar";
export type PointCloudScalarField = "z" | "depth";

export type Props = {
  plyB64: string;
  colorMode: PointCloudColorMode;
  scalarField: PointCloudScalarField;
  downsample: number;
  pointSize?: number;
};

// b64 -> ArrayBuffer (binario)
function b64ToArrayBuffer(b64: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function buildScalarColors(pos: Float32Array, scalarField: PointCloudScalarField) {
  const n = pos.length / 3;
  let min = Infinity, max = -Infinity;

  const scalars = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = pos[i * 3 + 2];
    const s = scalarField === "z" ? z : z; // depth == z por ahora (camera-depth)
    scalars[i] = s;
    min = Math.min(min, s);
    max = Math.max(max, s);
  }

  const range = Math.max(1e-6, max - min);
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (scalars[i] - min) / range; // 0..1
    out[i * 3 + 0] = t;        // R
    out[i * 3 + 1] = 0.2;      // G
    out[i * 3 + 2] = 1.0 - t;  // B
  }
  return out;
}

export function PointCloudViewer({
  plyB64,
  colorMode,
  scalarField,
  downsample,
  pointSize = 0.01,
}: Props) {
  const geom = useMemo(() => {
    const ab = b64ToArrayBuffer(plyB64);

    // 1) Parse PLY (ASCII o binario)
    const loader = new PLYLoader();
    const g0 = loader.parse(ab) as THREE.BufferGeometry;

    // 2) Normaliza atributos
    g0.computeBoundingSphere();

    // Asegura que position exista
    const posAttr = g0.getAttribute("position") as THREE.BufferAttribute | undefined;
    if (!posAttr) return g0;

    const pos = posAttr.array as Float32Array;

    // 3) Downsample
    const stride = Math.max(1, downsample);
    const n = pos.length / 3;

    const posDs = new Float32Array(Math.ceil(n / stride) * 3);
    let k = 0;
    for (let i = 0; i < n; i += stride) {
      posDs[k++] = pos[i * 3 + 0];
      posDs[k++] = pos[i * 3 + 1];
      posDs[k++] = pos[i * 3 + 2];
    }
    const posFinal = posDs.subarray(0, k);

    // 4) Colores: RGB si viene en PLY, si no blanco; o Scalar
    let colFinal: Float32Array;
    const colAttr = g0.getAttribute("color") as THREE.BufferAttribute | undefined;

    if (colorMode === "scalar") {
      colFinal = buildScalarColors(posFinal, scalarField);
    } else if (colAttr && colAttr.array) {
      // Ojo: loader suele dar color en 0..1 ya. Si viniera 0..255, igual se vería “quemado”.
      const col0 = colAttr.array as Float32Array;
      // downsample también en color:
      const colDs = new Float32Array(Math.ceil(n / stride) * 3);
      let c = 0;
      for (let i = 0; i < n; i += stride) {
        colDs[c++] = col0[i * 3 + 0];
        colDs[c++] = col0[i * 3 + 1];
        colDs[c++] = col0[i * 3 + 2];
      }
      colFinal = colDs.subarray(0, c);
    } else {
      colFinal = new Float32Array((posFinal.length / 3) * 3);
      colFinal.fill(1.0);
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(posFinal, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colFinal, 3));
    g.computeBoundingSphere();
    return g;
  }, [plyB64, colorMode, scalarField, downsample]);

  return (
    <Canvas camera={{ position: [0, 0, 2.5], fov: 45 }}>
      <ambientLight intensity={0.8} />
      <points geometry={geom}>
        <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
      </points>
      <OrbitControls makeDefault />
    </Canvas>
  );
}
