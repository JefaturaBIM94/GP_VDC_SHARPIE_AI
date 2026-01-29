import React, { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PLYLoader } from "three-stdlib";

export type PointCloudColorMode = "rgb" | "scalar";
export type PointCloudScalarField = "z" | "depth";
export type PointCloudStyle = "points" | "spheres";

export type Props = {
  // data
  plyB64: string;

  // color
  colorMode: PointCloudColorMode;
  scalarField: PointCloudScalarField;

  // decimation (client-side)
  downsample: number;

  // visual controls
  pointSize?: number;
  style?: PointCloudStyle;

  // OPTIONAL: server stride (solo para que TS no truene si lo pasas desde la vista)
  // No lo usamos para decimar aqui (eso ya lo hace el backend), pero lo dejamos por compatibilidad.
  stride?: number;
};

function b64ToBlobUrl(b64: string, mime = "application/octet-stream") {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

function PointsFromPLY({
  url,
  pointSize,
  colorMode,
  scalarField,
  stride,
  downsample,
  style,
}: {
  url: string;
  pointSize: number;
  colorMode?: PointCloudColorMode;
  scalarField?: PointCloudScalarField;
  stride?: number;
  downsample?: number;
  style?: PointCloudStyle;
}) {
  const { camera, controls } = useThree() as any;
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
  }, [url, colorMode, scalarField, stride, downsample]);

  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      sizeAttenuation: true,
    });
  }, [pointSize]);

  const effectiveStride = Math.max(1, downsample ?? 1);
  const effectiveStyle = style ?? "points";

  const decimatedGeom = useMemo(() => {
    if (!geom) return null;
    if (effectiveStride <= 1) return geom;

    const pos = geom.getAttribute("position");
    const col = geom.getAttribute("color");
    const outCount = Math.floor(pos.count / effectiveStride);
    const positions = new Float32Array(outCount * 3);
    const colors = col ? new Float32Array(outCount * 3) : null;

    let outIdx = 0;
    for (let i = 0; i < pos.count; i += effectiveStride) {
      positions[outIdx * 3 + 0] = pos.getX(i);
      positions[outIdx * 3 + 1] = pos.getY(i);
      positions[outIdx * 3 + 2] = pos.getZ(i);
      if (col && colors) {
        colors[outIdx * 3 + 0] = col.getX(i);
        colors[outIdx * 3 + 1] = col.getY(i);
        colors[outIdx * 3 + 2] = col.getZ(i);
      }
      outIdx += 1;
      if (outIdx >= outCount) break;
    }

    const dec = new THREE.BufferGeometry();
    dec.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colors) dec.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return dec;
  }, [geom, effectiveStride]);

  if (!decimatedGeom) return null;

  if (!decimatedGeom.getAttribute("color")) {
    const pos = decimatedGeom.getAttribute("position");
    const colors = new Float32Array(pos.count * 3);
    colors.fill(1.0);
    decimatedGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }

  decimatedGeom.computeBoundingBox();
  const bb = decimatedGeom.boundingBox;
  if (bb) {
    const center = new THREE.Vector3();
    bb.getCenter(center);
    decimatedGeom.translate(-center.x, -center.y, -center.z);
  }
  decimatedGeom.computeBoundingSphere();

  useEffect(() => {
    if (!decimatedGeom?.boundingSphere) return;
    const sphere = decimatedGeom.boundingSphere;
    const radius = Math.max(0.001, sphere.radius);
    const distance = radius * 2.5;

    camera.position.set(0, 0, distance);
    camera.near = Math.max(0.001, distance / 100);
    camera.far = Math.max(1000, distance * 10);
    camera.updateProjectionMatrix();

    if (controls?.target) {
      controls.target.set(0, 0, 0);
    }
    if (controls?.update) controls.update();
  }, [decimatedGeom, camera, controls]);

  return (
    <>
      {effectiveStyle === "points" ? (
        <points geometry={decimatedGeom} material={material} />
      ) : (
        <points geometry={decimatedGeom}>
          <pointsMaterial size={Math.max(pointSize ?? 0.01, 0.02)} vertexColors sizeAttenuation />
        </points>
      )}
    </>
  );
}

export const PointCloudViewer: React.FC<Props> = ({
  plyB64,
  pointSize = 0.02,
  colorMode = "rgb",
  scalarField = "z",
  stride,
  downsample,
  style,
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

  return (
    <Canvas camera={{ position: [0, 0, 2.5], fov: 50, near: 0.001, far: 1000 }}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[3, 5, 2]} intensity={0.6} />

      <gridHelper args={[10, 20]} />
      <axesHelper args={[1.5]} />

      {url && (
        <PointsFromPLY
          url={url}
          pointSize={pointSize}
          colorMode={colorMode}
          scalarField={scalarField}
          stride={stride}
          downsample={downsample}
          style={style}
        />
      )}

      <OrbitControls
        makeDefault
        enableRotate
        enableZoom
        enablePan
        screenSpacePanning
        dampingFactor={0.08}
        enableDamping
        rotateSpeed={0.7}
        zoomSpeed={0.9}
        panSpeed={0.7}
      />
    </Canvas>
  );
};
