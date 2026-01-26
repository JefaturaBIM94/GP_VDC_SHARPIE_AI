import React, { useMemo } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

export type PointCloudColorMode = "rgb" | "scalar";
export type PointCloudScalarField = "z" | "depth";

export type Props = {
  plyB64: string;
  colorMode: PointCloudColorMode;
  scalarField: PointCloudScalarField;
  downsample: number;
  pointSize?: number;
};

// --- helpers ---
function decodeBase64ToText(b64: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function parsePlyAscii(plyText: string) {
  const lines = plyText.split(/\r?\n/);
  let i = 0;

  let vertexCount = 0;
  while (i < lines.length) {
    const line = lines[i].trim();
    if (line.startsWith("element vertex")) {
      const parts = line.split(/\s+/);
      vertexCount = parseInt(parts[2], 10);
    }
    if (line === "end_header") {
      i++;
      break;
    }
    i++;
  }

  const positions: number[] = [];
  const colors: number[] = [];

  for (let v = 0; v < vertexCount && i < lines.length; v++, i++) {
    const parts = lines[i].trim().split(/\s+/);
    if (parts.length < 3) continue;

    const x = parseFloat(parts[0]);
    const y = parseFloat(parts[1]);
    const z = parseFloat(parts[2]);

    positions.push(x, y, z);

    // If PLY contains RGB: x y z r g b
    if (parts.length >= 6) {
      colors.push(
        parseFloat(parts[3]) / 255.0,
        parseFloat(parts[4]) / 255.0,
        parseFloat(parts[5]) / 255.0
      );
    } else {
      colors.push(1, 1, 1);
    }
  }

  return { positions, colors };
}

function buildScalarColors(positions: Float32Array, scalarField: PointCloudScalarField) {
  const n = positions.length / 3;
  const scalars = new Float32Array(n);

  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < n; i++) {
    const z = positions[i * 3 + 2];
    const s = scalarField === "z" ? z : z; // por ahora depth = z; luego lo refinamos si mandas depth real
    scalars[i] = s;
    min = Math.min(min, s);
    max = Math.max(max, s);
  }

  const range = Math.max(1e-6, max - min);
  const out = new Float32Array(n * 3);

  // simple blue->red
  for (let i = 0; i < n; i++) {
    const t = (scalars[i] - min) / range;
    out[i * 3 + 0] = t;
    out[i * 3 + 1] = 0.2;
    out[i * 3 + 2] = 1.0 - t;
  }

  return out;
}

// --- component ---
export function PointCloudViewer({
  plyB64,
  colorMode,
  scalarField,
  downsample,
  pointSize = 0.01,
}: Props) {
  const geom = useMemo(() => {
    const plyText = decodeBase64ToText(plyB64);
    const { positions, colors } = parsePlyAscii(plyText);

    const stride = Math.max(1, downsample);

    const posDs: number[] = [];
    const colDs: number[] = [];

    for (let i = 0; i < positions.length; i += 3 * stride) {
      posDs.push(positions[i], positions[i + 1], positions[i + 2]);
      colDs.push(colors[i], colors[i + 1], colors[i + 2]);
    }

    const posArr = new Float32Array(posDs);
    const colArrRgb = new Float32Array(colDs);
    const colArr = colorMode === "scalar" ? buildScalarColors(posArr, scalarField) : colArrRgb;

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
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
