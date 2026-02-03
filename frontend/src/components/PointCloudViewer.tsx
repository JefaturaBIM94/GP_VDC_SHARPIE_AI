// frontend/src/components/PointCloudViewer.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import type { MutableRefObject, RefObject } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PLYLoader, OrbitControls as OrbitControlsImpl } from "three-stdlib";

export type PointCloudColorMode = "rgb" | "scalar";
export type PointCloudScalarField = "z" | "depth";
export type PointCloudStyle = "points" | "spheres";

export type Props = {
  plyB64: string;
  colorMode: PointCloudColorMode;
  scalarField: PointCloudScalarField;
  downsample: number; // client-side downsample (1,2,4,8...)
  pointSize?: number;

  // Optional (compat con FastReconstructionView)
  style?: PointCloudStyle;
  stride?: number;
};

type ViewerApi = {
  reset: () => void;
  fit: () => void;
  roll: (rad: number) => void;
  view: (preset: "front" | "top" | "left") => void;
};

// b64 -> ArrayBuffer
function b64ToArrayBuffer(b64: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function buildScalarColors(pos: Float32Array, scalarField: PointCloudScalarField) {
  const n = pos.length / 3;
  let min = Infinity,
    max = -Infinity;

  const scalars = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = pos[i * 3 + 2];
    const s = scalarField === "z" ? z : z; // depth == z por ahora
    scalars[i] = s;
    min = Math.min(min, s);
    max = Math.max(max, s);
  }

  const range = Math.max(1e-6, max - min);
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (scalars[i] - min) / range;
    out[i * 3 + 0] = t;
    out[i * 3 + 1] = 0.2;
    out[i * 3 + 2] = 1.0 - t;
  }
  return out;
}

function InstancedSpheres({ geom, radius }: { geom: THREE.BufferGeometry; radius: number }) {
  const ref = useRef<THREE.InstancedMesh | null>(null);

  const count = useMemo(() => {
    const posAttr = geom.getAttribute("position") as THREE.BufferAttribute | undefined;
    if (!posAttr) return 0;
    return Math.floor((posAttr.array as Float32Array).length / 3);
  }, [geom]);

  useEffect(() => {
    const m = ref.current;
    if (!m) return;

    const posAttr = geom.getAttribute("position") as THREE.BufferAttribute | undefined;
    const colAttr = geom.getAttribute("color") as THREE.BufferAttribute | undefined;
    if (!posAttr || !colAttr) return;

    const pos = posAttr.array as Float32Array;
    const col = colAttr.array as Float32Array;

    const mat = new THREE.Matrix4();
    for (let i = 0; i < count; i++) {
      const x = pos[i * 3 + 0];
      const y = pos[i * 3 + 1];
      const z = pos[i * 3 + 2];
      mat.makeTranslation(x, y, z);
      m.setMatrixAt(i, mat);
    }
    m.instanceMatrix.needsUpdate = true;

    m.instanceColor = new THREE.InstancedBufferAttribute(col, 3);
    m.instanceColor.needsUpdate = true;
  }, [count, geom]);

  if (count <= 0) return null;

  return (
    <instancedMesh ref={ref} args={[undefined as any, undefined as any, count]}>
      <sphereGeometry args={[radius, 10, 10]} />
      <meshStandardMaterial vertexColors roughness={0.6} metalness={0.05} />
    </instancedMesh>
  );
}

/**
 * Binder:
 * - toma camera/controls dentro del Canvas
 * - expone API (reset/fit/roll/view) al overlay (afuera del Canvas)
 */
function ControlsBinder({
  geom,
  controlsRef,
  apiRef,
}: {
  geom: THREE.BufferGeometry | null;
  controlsRef: RefObject<OrbitControlsImpl | null>;
  apiRef: MutableRefObject<ViewerApi | null>;
}) {
  const { camera, invalidate } = useThree();

  useEffect(() => {
    apiRef.current = {
      reset: () => {
        controlsRef.current?.reset();
        invalidate();
      },
      fit: () => {
        if (!geom) return;
        geom.computeBoundingSphere();
        const bs = geom.boundingSphere;
        if (!bs) return;

        const center = bs.center.clone();
        const radius = Math.max(1e-6, bs.radius);

        // cámara a 2.5 radios
        const dir = new THREE.Vector3(0, 0, 1);
        const pos = center.clone().add(dir.multiplyScalar(radius * 2.5));

        camera.position.copy(pos);
        camera.near = radius / 100;
        camera.far = radius * 100;
        camera.updateProjectionMatrix();

        const c = controlsRef.current;
        if (c) {
          c.target.copy(center);
          c.update();
        }
        invalidate();
      },
      roll: (rad: number) => {
        // roll sobre eje Z de cámara
        camera.rotateZ(rad);
        camera.updateProjectionMatrix();
        controlsRef.current?.update();
        invalidate();
      },
      view: (preset) => {
        if (!geom) return;
        geom.computeBoundingSphere();
        const bs = geom.boundingSphere;
        if (!bs) return;

        const center = bs.center.clone();
        const radius = Math.max(1e-6, bs.radius);

        let dir = new THREE.Vector3(0, 0, 1); // front
        if (preset === "top") dir = new THREE.Vector3(0, 1, 0);
        if (preset === "left") dir = new THREE.Vector3(1, 0, 0);

        const pos = center.clone().add(dir.multiplyScalar(radius * 2.5));

        camera.position.copy(pos);
        camera.near = radius / 100;
        camera.far = radius * 100;
        camera.up.set(0, 1, 0);
        camera.lookAt(center);
        camera.updateProjectionMatrix();

        const c = controlsRef.current;
        if (c) {
          c.target.copy(center);
          c.update();
        }
        invalidate();
      },
    };

    return () => {
      apiRef.current = null;
    };
  }, [apiRef, camera, controlsRef, geom, invalidate]);

  return null;
}

export function PointCloudViewer({
  plyB64,
  colorMode,
  scalarField,
  downsample,
  style = "points",
  pointSize = 0.01,
}: Props) {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const apiRef = useRef<ViewerApi | null>(null);

  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(true);

  const geom = useMemo<THREE.BufferGeometry | null>(() => {
    if (!plyB64) return null;

    const ab = b64ToArrayBuffer(plyB64);

    const loader = new PLYLoader();
    const g0 = loader.parse(ab) as THREE.BufferGeometry;
    g0.computeBoundingSphere();

    const posAttr = g0.getAttribute("position") as THREE.BufferAttribute | undefined;
    if (!posAttr) return null;

    const pos = posAttr.array as Float32Array;

    // downsample
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

    // colors
    let colFinal: Float32Array;
    const colAttr = g0.getAttribute("color") as THREE.BufferAttribute | undefined;

    if (colorMode === "scalar") {
      colFinal = buildScalarColors(posFinal, scalarField);
    } else if (colAttr?.array) {
      const col0 = colAttr.array as Float32Array;
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
    <div className="relative w-full h-full overflow-hidden rounded-md bg-black">
      {/* HUD (controles visibles) */}
      <div className="absolute z-10 top-2 left-2 flex flex-wrap items-center gap-2 rounded-lg border border-white/10 bg-black/50 px-2 py-2">
        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.reset()}
          title="Reset camera"
          type="button"
        >
          Reset
        </button>
        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.fit()}
          title="Fit to point cloud"
          disabled={!geom}
          type="button"
        >
          Fit
        </button>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.view("front")}
          disabled={!geom}
          title="Front view"
          type="button"
        >
          Front
        </button>
        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.view("top")}
          disabled={!geom}
          title="Top view"
          type="button"
        >
          Top
        </button>
        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.view("left")}
          disabled={!geom}
          title="Left view"
          type="button"
        >
          Left
        </button>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.roll(Math.PI / 2)}
          title="Roll +90°"
          type="button"
        >
          Roll +90
        </button>
        <button
          className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 border border-white/10"
          onClick={() => apiRef.current?.roll(-Math.PI / 2)}
          title="Roll -90°"
          type="button"
        >
          Roll -90
        </button>

        <div className="w-px h-5 bg-white/10 mx-1" />

        <label className="flex items-center gap-1 text-[11px] text-white/80">
          <input
            type="checkbox"
            className="accent-emerald-500"
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
          />
          Grid
        </label>
        <label className="flex items-center gap-1 text-[11px] text-white/80">
          <input
            type="checkbox"
            className="accent-emerald-500"
            checked={showAxes}
            onChange={(e) => setShowAxes(e.target.checked)}
          />
          Axes
        </label>
      </div>

      {/* hint */}
      <div className="absolute z-10 bottom-2 left-2 text-[11px] text-white/60 bg-black/40 border border-white/10 rounded px-2 py-1">
        LMB: orbit · RMB: pan · Wheel: zoom
      </div>

      <Canvas camera={{ position: [0, 0, 2.5], fov: 45 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[2, 3, 4]} intensity={0.8} />

        <ControlsBinder geom={geom} controlsRef={controlsRef} apiRef={apiRef} />

        {showGrid && <gridHelper args={[10, 10]} />}
        {showAxes && <axesHelper args={[1.5]} />}

        {geom &&
          (style === "spheres" ? (
            <InstancedSpheres geom={geom} radius={Math.max(1e-6, pointSize * 0.5)} />
          ) : (
            <points geometry={geom}>
              <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
            </points>
          ))}

        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.08}
          enablePan
          enableRotate
          enableZoom
        />
      </Canvas>
    </div>
  );
}
