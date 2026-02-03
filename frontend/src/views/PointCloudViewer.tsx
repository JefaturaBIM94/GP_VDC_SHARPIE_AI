// frontend/src/views/PointCloudViewer.tsx
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { CameraControls } from "@react-three/drei";
import { PLYLoader } from "three-stdlib";

export type PointCloudColorMode = "rgb" | "scalar";
export type PointCloudScalarField = "z" | "depth";
export type PointCloudStyle = "points" | "spheres";

export type Props = {
  plyB64: string;
  colorMode: PointCloudColorMode;
  scalarField: PointCloudScalarField;
  downsample: number;
  pointSize?: number;
  style?: PointCloudStyle;
  stride?: number;
  showHud?: boolean;
  enableDamping?: boolean;
  enablePan?: boolean;
  enableZoom?: boolean;
  enableRotate?: boolean;
  speed?: number;
  dollySpeed?: number;
};

function ExposeControls() {
  const { controls } = useThree() as any;
  useEffect(() => {
    (window as any).__pc_controls = controls;
    return () => {
      (window as any).__pc_controls = null;
    };
  }, [controls]);
  return null;
}

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
  showHud = true,
  enableDamping = true,
  enablePan = true,
  enableZoom = true,
  enableRotate = true,
  speed = 1.0,
  dollySpeed = 1.0,
}: Props) {
  const geom = useMemo<THREE.BufferGeometry | null>(() => {
    if (!plyB64) return null;
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

  const btn = "px-2 py-1 text-[11px] rounded bg-black/50 border border-white/10 hover:bg-black/60";
  const pillOn = "bg-emerald-500/20 border-emerald-400/30";
  const pillOff = "bg-black/40 border-white/10";
  const pill = (on: boolean) =>
    `px-2 py-1 text-[11px] rounded border ${on ? pillOn : pillOff} hover:bg-black/60`;
  const actionNone = 0;

  return (
    <div className="relative w-full rounded border border-slate-700 overflow-hidden" style={{ height: "100%" }}>
      {showHud && (
        <div className="absolute z-10 top-2 left-2 right-2 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
          <div className="flex flex-wrap gap-2 pointer-events-auto">
            <button className={btn} onClick={() => (window as any).__pc_controls?.reset(true)}>
              Reset
            </button>
            <button
              className={btn}
              onClick={() => {
                const c = (window as any).__pc_controls;
                if (!c || !geom) return;
                try {
                  const box = new THREE.Box3().setFromBufferAttribute(
                    geom.getAttribute("position") as THREE.BufferAttribute
                  );
                  const size = new THREE.Vector3();
                  box.getSize(size);
                  const center = new THREE.Vector3();
                  box.getCenter(center);
                  c.setLookAt(
                    center.x,
                    center.y,
                    center.z + Math.max(size.length(), 1),
                    center.x,
                    center.y,
                    center.z,
                    true
                  );
                } catch {
                  c.reset(true);
                }
              }}
            >
              Fit
            </button>

            <button className={btn} onClick={() => (window as any).__pc_controls?.rotate(0, 0, Math.PI / 2, true)}>
              Roll +90
            </button>
            <button className={btn} onClick={() => (window as any).__pc_controls?.rotate(0, 0, -Math.PI / 2, true)}>
              Roll -90
            </button>

            <button className={btn} onClick={() => (window as any).__pc_controls?.setLookAt(0, 2.5, 0, 0, 0, 0, true)}>
              Top
            </button>
            <button className={btn} onClick={() => (window as any).__pc_controls?.setLookAt(0, 0, 2.5, 0, 0, 0, true)}>
              Front
            </button>
            <button className={btn} onClick={() => (window as any).__pc_controls?.setLookAt(2.5, 0, 0, 0, 0, 0, true)}>
              Left
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2 pointer-events-auto">
            <span className="text-[11px] text-white/70 mr-2">Controls:</span>
            <span className={pill(enableRotate)}>Rotate</span>
            <span className={pill(enablePan)}>Pan</span>
            <span className={pill(enableZoom)}>Zoom</span>

            <div className="ml-2 hidden lg:block text-[10px] text-white/50">
              LMB: rotate · MMB: dolly · RMB: pan · Wheel: zoom
            </div>
          </div>
        </div>
      )}

      {showHud && (
        <div className="absolute z-10 bottom-2 left-2 right-2 flex flex-wrap gap-3 items-center bg-black/30 border border-white/10 rounded-lg px-3 py-2 pointer-events-auto">
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-white/70 w-16">Speed</span>
            <input
              type="range"
              min={0.2}
              max={2.0}
              step={0.1}
              value={speed}
              readOnly
              className="w-40"
              title="Speed (configurable desde props/estado en la vista)"
            />
            <span className="text-[11px] text-white/60">{speed.toFixed(1)}x</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-white/70 w-16">Zoom</span>
            <input
              type="range"
              min={0.2}
              max={3.0}
              step={0.1}
              value={dollySpeed}
              readOnly
              className="w-40"
              title="Zoom speed (configurable desde props/estado en la vista)"
            />
            <span className="text-[11px] text-white/60">{dollySpeed.toFixed(1)}x</span>
          </div>

          <div className="ml-auto text-[11px] text-white/50">
            Tip: si se "pierde" la nube, usa <b>Fit</b> o <b>Reset</b>.
          </div>
        </div>
      )}

      <Canvas camera={{ position: [0, 0, 2.5], fov: 45, near: 0.01, far: 5000 }} dpr={[1, 1.5]} gl={{ antialias: false, powerPreference: "high-performance" }}>
        <ExposeControls />
        <ambientLight intensity={0.8} />
        <gridHelper args={[10, 10]} />
        {geom && (
          <points geometry={geom}>
            <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
          </points>
        )}
        <CameraControls
          makeDefault
          enabled
          mouseButtons={{
            left: enableRotate ? THREE.MOUSE.ROTATE : actionNone,
            middle: enableZoom ? THREE.MOUSE.DOLLY : actionNone,
            right: enablePan ? THREE.MOUSE.PAN : actionNone,
            wheel: enableZoom ? THREE.MOUSE.DOLLY : actionNone,
          }}
          smoothTime={enableDamping ? 0.15 : 0.0}
          dragToOffset={enablePan}
          dollySpeed={dollySpeed}
        />
      </Canvas>
    </div>
  );
}
