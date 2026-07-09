"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type WheelEvent,
} from "react";

type TransformParams = {
  theta: number;
  logScale: number;
  shear: number;
};

type Matrix2 = {
  a: number;
  b: number;
  c: number;
  d: number;
};

type Camera = {
  yaw: number;
  pitch: number;
  distance: number;
};

type Point3 = [number, number, number];

type ProjectedPoint = {
  x: number;
  y: number;
  depth: number;
};

type Challenge = {
  kicker: string;
  question: string;
  options: readonly string[];
  correct: number;
  target: TransformParams;
  explanation: string;
  observation: string;
};

const IDENTITY: TransformParams = { theta: 0, logScale: 0, shear: 0 };
const DEFAULT_CAMERA: Camera = { yaw: -0.72, pitch: 0.46, distance: 6.4 };

const CHALLENGES: readonly Challenge[] = [
  {
    kicker: "Challenge 01 · rotation",
    question: "A unit square meets a 45° phase rotation. What happens to its area?",
    options: ["It stays exactly 1", "It doubles", "It shrinks by half"],
    correct: 0,
    target: { theta: 45, logScale: 0, shear: 0 },
    explanation:
      "A rotation moves every point but changes neither lengths nor oriented area. det R(45°) = 1, so the square returns as a diamond with the same area.",
    observation: "Look for a diamond, not a larger square.",
  },
  {
    kicker: "Challenge 02 · reciprocal scale",
    question: "Set log-scale to +0.70. Which phase direction grows?",
    options: [
      "q grows; p contracts",
      "Both directions grow",
      "p grows; q contracts",
    ],
    correct: 0,
    target: { theta: 0, logScale: 0.7, shear: 0 },
    explanation:
      "The scale is reciprocal: diag(eˢ, e⁻ˢ). At s = 0.70, q grows by 2.01× while p contracts to 0.50×. Their product remains 1.",
    observation: "Watch the green face stretch thin without gaining area.",
  },
  {
    kicker: "Challenge 03 · composition",
    question: "M = C · S · R rotates, reciprocally scales, then chirp-shears. What is det M?",
    options: ["Exactly 1", "Approximately 1.2", "It depends on the rotation"],
    correct: 0,
    target: { theta: 20, logScale: 0.25, shear: 1.2 },
    explanation:
      "Each factor has determinant one. Determinants multiply, so det(C · S · R) = 1 · 1 · 1—even when the final grid looks dramatically slanted.",
    observation: "Shape is negotiable. Symplectic area is not.",
  },
] as const;

const COMPARISONS = {
  same: {
    label: "Same width",
    subhead: "A100 · text8 · trunk width 1024",
    denseName: "Dense @1024",
    lctName: "LCT @1024",
    dense: { params: 50.7, throughput: 48.4, loss: 1.3351 },
    lct: { params: 33.9, throughput: 60.1, loss: 1.4097 },
    verdict:
      "At the same trunk width, LCT is 1.24× faster and uses 33% fewer parameters—but its mean best validation loss is 0.075 nats worse.",
    caution: "A speed result, not a quality win.",
  },
  matched: {
    label: "Parameter matched",
    subhead: "A100 · text8 · 34M-parameter budget",
    denseName: "Dense @840",
    lctName: "LCT @1024",
    dense: { params: 34.2, throughput: 64.5, loss: 1.2947 },
    lct: { params: 33.9, throughput: 60.1, loss: 1.4097 },
    verdict:
      "At the same parameter budget, the narrower dense model is 1.07× faster and 0.115 nats better. It wins on both axes that matter.",
    caution: "This is the fair-control conclusion.",
  },
} as const;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function matrixFor({ theta, logScale, shear }: TransformParams): Matrix2 {
  const radians = (theta * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const stretch = Math.exp(logScale);
  const squeeze = Math.exp(-logScale);

  // Match SymplecticLCTLayer exactly: M = C(k) · S(s) · R(theta),
  // with R applied first to column vectors, followed by S and then C.
  // Every factor has determinant one, so this composition does too.
  const a = stretch * cosine;
  const b = stretch * sine;
  return {
    a,
    b,
    c: shear * a - sine * squeeze,
    d: shear * b + cosine * squeeze,
  };
}

function applyMatrix(matrix: Matrix2, q: number, p: number): [number, number] {
  return [matrix.a * q + matrix.b * p, matrix.c * q + matrix.d * p];
}

function formatSigned(value: number) {
  const stable = Math.abs(value) < 0.0005 ? 0 : value;
  return stable.toFixed(3);
}

function PhaseSpaceCanvas({
  params,
  camera,
  setCamera,
}: {
  params: TransformParams;
  camera: Camera;
  setCamera: React.Dispatch<React.SetStateAction<Camera>>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const matrix = useMemo(() => matrixFor(params), [params]);
  const determinant = matrix.a * matrix.d - matrix.b * matrix.c;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const width = Math.max(320, bounds.width);
      const height = Math.max(420, bounds.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);

      if (canvas.width !== Math.round(width * dpr)) {
        canvas.width = Math.round(width * dpr);
      }
      if (canvas.height !== Math.round(height * dpr)) {
        canvas.height = Math.round(height * dpr);
      }

      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);

      const backdrop = context.createRadialGradient(
        width * 0.46,
        height * 0.44,
        10,
        width * 0.5,
        height * 0.5,
        Math.max(width, height) * 0.72,
      );
      backdrop.addColorStop(0, "#162a26");
      backdrop.addColorStop(0.55, "#0a1715");
      backdrop.addColorStop(1, "#050b0a");
      context.fillStyle = backdrop;
      context.fillRect(0, 0, width, height);

      const project = ([x, y, z]: Point3): ProjectedPoint => {
        const cosYaw = Math.cos(camera.yaw);
        const sinYaw = Math.sin(camera.yaw);
        const yawX = cosYaw * x - sinYaw * z;
        const yawZ = sinYaw * x + cosYaw * z;
        const cosPitch = Math.cos(camera.pitch);
        const sinPitch = Math.sin(camera.pitch);
        const pitchY = cosPitch * y - sinPitch * yawZ;
        const pitchZ = sinPitch * y + cosPitch * yawZ;
        const perspective = camera.distance / Math.max(2.4, camera.distance + pitchZ);
        const scale = Math.min(width, height) * 0.235 * perspective;
        return {
          x: width / 2 + yawX * scale,
          y: height / 2 - pitchY * scale,
          depth: pitchZ,
        };
      };

      const strokePath = (
        points: readonly Point3[],
        color: string,
        lineWidth: number,
        dash: number[] = [],
      ) => {
        if (points.length < 2) return;
        context.beginPath();
        const first = project(points[0]);
        context.moveTo(first.x, first.y);
        for (let index = 1; index < points.length; index += 1) {
          const point = project(points[index]);
          context.lineTo(point.x, point.y);
        }
        context.strokeStyle = color;
        context.lineWidth = lineWidth;
        context.setLineDash(dash);
        context.stroke();
        context.setLineDash([]);
      };

      const fillPolygon = (
        points: readonly Point3[],
        fill: string,
        stroke: string,
        lineWidth = 1.5,
      ) => {
        if (points.length < 3) return;
        context.beginPath();
        const first = project(points[0]);
        context.moveTo(first.x, first.y);
        for (let index = 1; index < points.length; index += 1) {
          const point = project(points[index]);
          context.lineTo(point.x, point.y);
        }
        context.closePath();
        context.fillStyle = fill;
        context.fill();
        context.strokeStyle = stroke;
        context.lineWidth = lineWidth;
        context.stroke();
      };

      const drawArrow = (from: Point3, to: Point3, color: string) => {
        strokePath([from, to], color, 2.2);
        const start = project(from);
        const end = project(to);
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        context.beginPath();
        context.moveTo(end.x, end.y);
        context.lineTo(
          end.x - 10 * Math.cos(angle - 0.42),
          end.y - 10 * Math.sin(angle - 0.42),
        );
        context.lineTo(
          end.x - 10 * Math.cos(angle + 0.42),
          end.y - 10 * Math.sin(angle + 0.42),
        );
        context.closePath();
        context.fillStyle = color;
        context.fill();
      };

      const zStart = -1.05;
      const zEnd = 1.05;
      const gridExtent = 1.45;
      const gridSteps = 8;

      // Two phase-space planes: identity below, transformed above.
      for (let plane = 0; plane <= 1; plane += 1) {
        const t = plane;
        const z = plane === 0 ? zStart : zEnd;
        const planeMatrix = matrixFor({
          theta: params.theta * t,
          logScale: params.logScale * t,
          shear: params.shear * t,
        });
        for (let step = 0; step <= gridSteps; step += 1) {
          const position = -gridExtent + (2 * gridExtent * step) / gridSteps;
          const horizontal: Point3[] = [];
          const vertical: Point3[] = [];
          for (let sample = 0; sample <= 24; sample += 1) {
            const moving = -gridExtent + (2 * gridExtent * sample) / 24;
            const [hq, hp] = applyMatrix(planeMatrix, moving, position);
            const [vq, vp] = applyMatrix(planeMatrix, position, moving);
            horizontal.push([hq, hp, z]);
            vertical.push([vq, vp, z]);
          }
          const alpha = plane === 0 ? "rgba(255,255,255,.11)" : "rgba(197,255,109,.17)";
          strokePath(horizontal, alpha, plane === 0 ? 0.8 : 1);
          strokePath(vertical, alpha, plane === 0 ? 0.8 : 1);
        }
      }

      // A family of phase points moving through determinant-one intermediates.
      for (let orbit = 0; orbit < 16; orbit += 1) {
        const angle = (orbit / 16) * Math.PI * 2;
        const inputQ = Math.cos(angle) * 1.05;
        const inputP = Math.sin(angle) * 1.05;
        const points: Point3[] = [];
        for (let step = 0; step <= 30; step += 1) {
          const t = step / 30;
          const intermediate = matrixFor({
            theta: params.theta * t,
            logScale: params.logScale * t,
            shear: params.shear * t,
          });
          const [q, p] = applyMatrix(intermediate, inputQ, inputP);
          points.push([q, p, zStart + (zEnd - zStart) * t]);
        }
        const highlighted = orbit === 2;
        strokePath(
          points,
          highlighted ? "rgba(255,116,82,.9)" : "rgba(168,151,255,.24)",
          highlighted ? 2.4 : 1,
        );
      }

      const circleAt = (at: number): Point3[] => {
        const atMatrix = matrixFor({
          theta: params.theta * at,
          logScale: params.logScale * at,
          shear: params.shear * at,
        });
        const z = zStart + (zEnd - zStart) * at;
        const points: Point3[] = [];
        for (let step = 0; step <= 64; step += 1) {
          const angle = (step / 64) * Math.PI * 2;
          const [q, p] = applyMatrix(atMatrix, Math.cos(angle) * 1.05, Math.sin(angle) * 1.05);
          points.push([q, p, z]);
        }
        return points;
      };

      strokePath(circleAt(0), "rgba(255,209,112,.72)", 1.6, [5, 5]);
      strokePath(circleAt(1), "rgba(197,255,109,.96)", 2.3);

      const unitSquare: [number, number][] = [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
      ];
      const inputSquare: Point3[] = unitSquare.map(([q, p]) => [q, p, zStart]);
      const outputSquare: Point3[] = unitSquare.map(([q, p]) => {
        const [outputQ, outputP] = applyMatrix(matrix, q, p);
        return [outputQ, outputP, zEnd];
      });
      fillPolygon(inputSquare, "rgba(255,209,112,.12)", "rgba(255,209,112,.82)");
      fillPolygon(outputSquare, "rgba(197,255,109,.12)", "rgba(197,255,109,.96)", 2);

      const [qBasisQ, qBasisP] = applyMatrix(matrix, 0.72, 0);
      const [pBasisQ, pBasisP] = applyMatrix(matrix, 0, 0.72);
      drawArrow([0, 0, zEnd], [qBasisQ, qBasisP, zEnd], "#ff7452");
      drawArrow([0, 0, zEnd], [pBasisQ, pBasisP, zEnd], "#a897ff");

      // Flow axis and quiet labels complete the 3D reading.
      drawArrow([0, 0, zStart - 0.35], [0, 0, zEnd + 0.42], "rgba(255,255,255,.34)");
      const label = (text: string, point: Point3, color: string, align: CanvasTextAlign = "left") => {
        const projected = project(point);
        context.fillStyle = color;
        context.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
        context.textAlign = align;
        context.fillText(text, projected.x, projected.y);
      };
      label("INPUT · A = 1", [-1.45, -1.6, zStart], "rgba(255,209,112,.86)");
      label("OUTPUT · A = 1", [-1.45, -1.6, zEnd], "rgba(197,255,109,.92)");
      label("q′", [qBasisQ, qBasisP + 0.12, zEnd], "#ff8c71");
      label("p′", [pBasisQ, pBasisP + 0.12, zEnd], "#b9abff");
      label("composition τ", [0, 0.12, zEnd + 0.56], "rgba(255,255,255,.48)", "center");

      context.fillStyle = "rgba(255,255,255,.035)";
      for (let x = 0; x < width; x += 26) {
        for (let y = 0; y < height; y += 26) {
          context.fillRect(x, y, 1, 1);
        }
      }
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [camera, matrix, params]);

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    setIsDragging(true);
  };

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    setCamera((current) => ({
      ...current,
      yaw: current.yaw + deltaX * 0.008,
      pitch: clamp(current.pitch + deltaY * 0.008, -1.1, 1.1),
    }));
  };

  const releasePointer = (event: PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
      setIsDragging(false);
    }
  };

  const handleWheel = (event: WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    setCamera((current) => ({
      ...current,
      distance: clamp(current.distance + event.deltaY * 0.006, 4.2, 9.5),
    }));
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    const amount = event.shiftKey ? 0.22 : 0.1;
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "Home"].includes(event.key)) {
      event.preventDefault();
    }
    setCamera((current) => {
      if (event.key === "ArrowLeft") return { ...current, yaw: current.yaw - amount };
      if (event.key === "ArrowRight") return { ...current, yaw: current.yaw + amount };
      if (event.key === "ArrowUp") return { ...current, pitch: clamp(current.pitch - amount, -1.1, 1.1) };
      if (event.key === "ArrowDown") return { ...current, pitch: clamp(current.pitch + amount, -1.1, 1.1) };
      if (event.key === "+" || event.key === "=") {
        return { ...current, distance: clamp(current.distance - 0.35, 4.2, 9.5) };
      }
      if (event.key === "-") return { ...current, distance: clamp(current.distance + 0.35, 4.2, 9.5) };
      if (event.key === "Home") return DEFAULT_CAMERA;
      return current;
    });
  };

  return (
    <div className="canvas-stage">
      <canvas
        ref={canvasRef}
        className={isDragging ? "is-dragging" : ""}
        role="img"
        tabIndex={0}
        aria-label={`Interactive three-dimensional phase-space transform. Rotation ${params.theta.toFixed(0)} degrees, log-scale ${params.logScale.toFixed(2)}, chirp shear ${params.shear.toFixed(2)}, determinant ${determinant.toFixed(6)}.`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={releasePointer}
        onPointerCancel={releasePointer}
        onWheel={handleWheel}
        onKeyDown={handleKeyDown}
      />
      <div className="canvas-caption" aria-hidden="true">
        <span>drag to orbit</span>
        <span>scroll to zoom</span>
        <span>arrow keys work too</span>
      </div>
      <div className="camera-controls" aria-label="3D view controls">
        <button
          type="button"
          aria-label="Rotate view left"
          onClick={() => setCamera((current) => ({ ...current, yaw: current.yaw - 0.18 }))}
        >
          ←
        </button>
        <button
          type="button"
          aria-label="Tilt view up"
          onClick={() =>
            setCamera((current) => ({ ...current, pitch: clamp(current.pitch - 0.18, -1.1, 1.1) }))
          }
        >
          ↑
        </button>
        <button
          type="button"
          aria-label="Tilt view down"
          onClick={() =>
            setCamera((current) => ({ ...current, pitch: clamp(current.pitch + 0.18, -1.1, 1.1) }))
          }
        >
          ↓
        </button>
        <button
          type="button"
          aria-label="Rotate view right"
          onClick={() => setCamera((current) => ({ ...current, yaw: current.yaw + 0.18 }))}
        >
          →
        </button>
        <button type="button" className="camera-reset" onClick={() => setCamera(DEFAULT_CAMERA)}>
          reset view
        </button>
      </div>
    </div>
  );
}

function RangeControl({
  id,
  label,
  symbol,
  min,
  max,
  step,
  value,
  unit,
  onChange,
}: {
  id: string;
  label: string;
  symbol: string;
  min: number;
  max: number;
  step: number;
  value: number;
  unit?: string;
  onChange: (value: number) => void;
}) {
  const percentage = ((value - min) / (max - min)) * 100;
  return (
    <div className="range-control">
      <label htmlFor={id}>
        <span>{label}</span>
        <output htmlFor={id}>
          {symbol} {value.toFixed(step < 0.1 ? 2 : 0)}{unit}
        </output>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--range-progress": `${percentage}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

function MatrixReadout({ params }: { params: TransformParams }) {
  const matrix = useMemo(() => matrixFor(params), [params]);
  const determinant = matrix.a * matrix.d - matrix.b * matrix.c;
  return (
    <div className="matrix-readout">
      <div className="matrix-title">
        <span>live operator</span>
        <strong>M = C(k) · S(s) · R(θ)</strong>
      </div>
      <div className="matrix-and-invariant">
        <div className="matrix" aria-label="Current two by two transform matrix">
          <span>{formatSigned(matrix.a)}</span>
          <span>{formatSigned(matrix.b)}</span>
          <span>{formatSigned(matrix.c)}</span>
          <span>{formatSigned(matrix.d)}</span>
        </div>
        <div className="determinant">
          <span>det M</span>
          <strong>{determinant.toFixed(6)}</strong>
          <em>invariant held</em>
        </div>
      </div>
      <div className="area-ledger" aria-label="Area invariant">
        <span>input area <strong>1.000</strong></span>
        <span aria-hidden="true">→</span>
        <span>output area <strong>{Math.abs(determinant).toFixed(3)}</strong></span>
      </div>
    </div>
  );
}

function LearningLoop({
  params,
  animateTo,
}: {
  params: TransformParams;
  animateTo: (target: TransformParams) => Promise<void>;
}) {
  const [challengeIndex, setChallengeIndex] = useState(0);
  const [selection, setSelection] = useState<number | null>(null);
  const [transformed, setTransformed] = useState(false);
  const [understood, setUnderstood] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [complete, setComplete] = useState(false);
  const challenge = CHALLENGES[challengeIndex];

  const currentStep = complete
    ? 3
    : understood
      ? 3
      : transformed
        ? 2
        : selection !== null
          ? 1
          : 0;

  const transform = async () => {
    setIsAnimating(true);
    await animateTo(challenge.target);
    setIsAnimating(false);
    setTransformed(true);
  };

  const advance = async () => {
    if (challengeIndex === CHALLENGES.length - 1) {
      setComplete(true);
      return;
    }
    setChallengeIndex((current) => current + 1);
    setSelection(null);
    setTransformed(false);
    setUnderstood(false);
    await animateTo(IDENTITY);
  };

  const restart = async () => {
    setChallengeIndex(0);
    setSelection(null);
    setTransformed(false);
    setUnderstood(false);
    setComplete(false);
    await animateTo(IDENTITY);
  };

  return (
    <section className="learning-loop" aria-labelledby="learning-loop-title">
      <div className="loop-heading">
        <div>
          <p className="section-label">A feedback loop, not a lecture</p>
          <h2 id="learning-loop-title">Predict. Move it. Explain it.</h2>
        </div>
        <p>
          Three compact challenges turn the live figure into a learning instrument. Wrong guesses are useful:
          the invariant answers back immediately.
        </p>
      </div>

      <ol className="loop-steps" aria-label="Learning loop steps">
        {["Predict", "Transform", "Explain", "Next"].map((label, index) => (
          <li key={label} className={index === currentStep ? "is-current" : index < currentStep ? "is-done" : ""}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            {label}
          </li>
        ))}
      </ol>

      <div className="challenge-body" aria-live="polite">
        {!complete ? (
          <>
            <div className="challenge-index">
              <span>{challenge.kicker}</span>
              <span>{challengeIndex + 1} / {CHALLENGES.length}</span>
            </div>
            <h3>{challenge.question}</h3>

            <div className="challenge-options" role="group" aria-label="Prediction choices">
              {challenge.options.map((option, index) => {
                const selected = selection === index;
                const correct = index === challenge.correct;
                return (
                  <button
                    type="button"
                    key={option}
                    className={selected ? (correct ? "is-correct" : "is-wrong") : ""}
                    aria-pressed={selected}
                    disabled={transformed || isAnimating}
                    onClick={() => setSelection(index)}
                  >
                    <span>{String.fromCharCode(65 + index)}</span>
                    {option}
                  </button>
                );
              })}
            </div>

            {selection !== null && !transformed && (
              <div className={selection === challenge.correct ? "prediction-feedback correct" : "prediction-feedback wrong"}>
                <strong>{selection === challenge.correct ? "Good prediction." : "Useful miss."}</strong>
                <span>
                  {selection === challenge.correct
                    ? "Now make the geometry prove it."
                    : "Run the transform and let the area ledger settle it."}
                </span>
              </div>
            )}

            {selection !== null && !transformed && (
              <button type="button" className="loop-action" onClick={transform} disabled={isAnimating}>
                {isAnimating ? "Moving through phase space…" : "Transform the figure"}
              </button>
            )}

            {transformed && !understood && (
              <div className="explanation-panel">
                <p className="section-label">What just happened</p>
                <p>{challenge.explanation}</p>
                <aside>{challenge.observation}</aside>
                <button type="button" className="loop-action" onClick={() => setUnderstood(true)}>
                  I can explain it
                </button>
              </div>
            )}

            {understood && (
              <div className="next-panel">
                <div>
                  <span>θ {params.theta.toFixed(0)}°</span>
                  <span>s {params.logScale.toFixed(2)}</span>
                  <span>k {params.shear.toFixed(2)}</span>
                  <span>det 1</span>
                </div>
                <button type="button" className="loop-action" onClick={advance}>
                  {challengeIndex === CHALLENGES.length - 1 ? "Finish the loop" : "Next challenge"}
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="loop-complete">
            <span className="complete-mark">3 / 3</span>
            <div>
              <p className="section-label">Loop complete</p>
              <h3>You can deform the shape without spending its area.</h3>
              <p>
                That is the geometric promise. The rest of this page asks the harder empirical question:
                did that structure buy a better language model?
              </p>
            </div>
            <button type="button" className="loop-action" onClick={restart}>Run it again</button>
          </div>
        )}
      </div>
    </section>
  );
}

function MetricBars({
  label,
  unit,
  denseValue,
  lctValue,
  denseName,
  lctName,
  lowerIsBetter = false,
}: {
  label: string;
  unit: string;
  denseValue: number;
  lctValue: number;
  denseName: string;
  lctName: string;
  lowerIsBetter?: boolean;
}) {
  const maximum = Math.max(denseValue, lctValue);
  const denseWins = lowerIsBetter ? denseValue < lctValue : denseValue > lctValue;
  const lctWins = lowerIsBetter ? lctValue < denseValue : lctValue > denseValue;
  return (
    <div className="metric-row">
      <div className="metric-label">
        <strong>{label}</strong>
        <span>{lowerIsBetter ? "lower is better" : "higher is better"}</span>
      </div>
      <div className="bar-pair">
        <div className="bar-line dense">
          <span>{denseName}</span>
          <i style={{ width: `${(denseValue / maximum) * 100}%` }} />
          <b className={denseWins ? "winner" : ""}>{denseValue.toFixed(label === "Parameters" ? 1 : label === "Throughput" ? 1 : 4)}{unit}</b>
        </div>
        <div className="bar-line lct">
          <span>{lctName}</span>
          <i style={{ width: `${(lctValue / maximum) * 100}%` }} />
          <b className={lctWins ? "winner" : ""}>{lctValue.toFixed(label === "Parameters" ? 1 : label === "Throughput" ? 1 : 4)}{unit}</b>
        </div>
      </div>
    </div>
  );
}

function FairComparison() {
  const [mode, setMode] = useState<keyof typeof COMPARISONS>("matched");
  const comparison = COMPARISONS[mode];
  return (
    <section className="comparison-section" id="fair-control" aria-labelledby="comparison-title">
      <div className="comparison-intro">
        <p className="section-label">The control changes the claim</p>
        <h2 id="comparison-title">Fair comparison is an interaction.</h2>
        <p>
          The A100 result sounds positive until the budget is held constant. Toggle the control; the numbers do
          not change, only the question does.
        </p>
      </div>

      <div className="comparison-instrument">
        <div className="comparison-switch" role="tablist" aria-label="Comparison control">
          {(Object.keys(COMPARISONS) as (keyof typeof COMPARISONS)[]).map((key) => (
            <button
              type="button"
              key={key}
              role="tab"
              aria-selected={mode === key}
              className={mode === key ? "is-active" : ""}
              onClick={() => setMode(key)}
            >
              <span>{COMPARISONS[key].label}</span>
              <small>{key === "same" ? "width = 1024" : "params ≈ 34M"}</small>
            </button>
          ))}
        </div>

        <div className="comparison-plot" role="tabpanel">
          <div className="plot-heading">
            <span>{comparison.subhead}</span>
            <span>2 paired seeds · 3,000 steps</span>
          </div>
          <MetricBars
            label="Parameters"
            unit="M"
            denseValue={comparison.dense.params}
            lctValue={comparison.lct.params}
            denseName={comparison.denseName}
            lctName={comparison.lctName}
            lowerIsBetter
          />
          <MetricBars
            label="Throughput"
            unit="k tok/s"
            denseValue={comparison.dense.throughput}
            lctValue={comparison.lct.throughput}
            denseName={comparison.denseName}
            lctName={comparison.lctName}
          />
          <MetricBars
            label="Best val loss"
            unit=""
            denseValue={comparison.dense.loss}
            lctValue={comparison.lct.loss}
            denseName={comparison.denseName}
            lctName={comparison.lctName}
            lowerIsBetter
          />
        </div>

        <div className="comparison-verdict" key={mode}>
          <span>{comparison.caution}</span>
          <p>{comparison.verdict}</p>
        </div>
      </div>
    </section>
  );
}

function EvidenceTable({
  caption,
  rows,
}: {
  caption: string;
  rows: readonly (readonly string[])[];
}) {
  const [head, ...body] = rows;
  return (
    <div className="evidence-table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>{head.map((cell) => <th key={cell}>{cell}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((row) => (
            <tr key={row[0]}>{row.map((cell) => <td key={cell}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PaperSections() {
  return (
    <section className="paper-section" id="evidence" aria-labelledby="paper-title">
      <header className="paper-header">
        <div>
          <p className="section-label">The paper, disclosed progressively</p>
          <h2 id="paper-title">Structure is real. Improvement is not.</h2>
        </div>
        <p className="paper-deck">
          A trainable structured layer, five repaired evaluation bugs, and a negative result that held across two
          substrates, two datasets, two hardware targets, and a 15× parameter range.
        </p>
      </header>

      <div className="paper-abstract">
        <span>Abstract</span>
        <p>
          LCTLinear is an O(N log N) structured alternative to a dense map. In controlled character-level
          NanoGPT studies it did not beat a narrower dense model. It did beat rank-1 factorization at equal
          parameter budgets, and it became faster than same-width dense models at large widths. The useful result
          is the boundary—not a victory lap.
        </p>
      </div>

      <div className="paper-accordions">
        <details open>
          <summary>
            <span className="paper-number">01</span>
            <span>
              <strong>What the figure teaches—and what NanoGPT actually tested</strong>
              <small>Family geometry ≠ learned experiment trajectory</small>
            </span>
            <i aria-hidden="true">+</i>
          </summary>
          <div className="paper-content split-copy">
            <p>
              The live instrument above uses the same decomposition as the Python layer:
              M = C(k) · S(s) · R(θ). Acting on a column vector, it applies rotation first, reciprocal scale
              second, and chirp shear last. It is an explorable map of the LCT family, not a visualization of a
              training trace.
            </p>
            <p>
              The historical NanoGPT arms used fixed Fourier-family cells: the headline
              <code>linear-fourier</code> setting plus fixed-angle <code>linear-frft30</code> and
              <code>linear-frft45</code> controls. The pilot-selected cell was Fourier, unitary, no inverse.
              Learned capacity lived in the spectral diagonal and output structure; the experiment did
              <strong> not</strong> learn the θ / s / k geometry exposed in this teaching figure.
            </p>
            <aside className="honesty-note">
              <strong>Reading rule</strong>
              Do not transfer motion in the demo into a claim about what the language model learned.
            </aside>
          </div>
        </details>

        <details>
          <summary>
            <span className="paper-number">02</span>
            <span>
              <strong>MPS: the fair control overturns the early story</strong>
              <small>4 paired seeds · tinyshakespeare · 2,000 steps</small>
            </span>
            <i aria-hidden="true">+</i>
          </summary>
          <div className="paper-content">
            <p>
              The same-width dense baseline was already an unhealthy control: the smaller dim-212 dense model was
              both faster and 0.22 nats better. Against that parameter-matched baseline, every LCT arm lost in 4/4
              paired seeds by +0.33 to +0.38 nats.
            </p>
            <EvidenceTable
              caption="Original MPS substrate · mean best validation loss across 4 paired seeds"
              rows={[
                ["config", "params", "tok/s", "best val"],
                ["dense @256", "3.26M", "171k", "2.2496"],
                ["dense @212 · matched", "2.25M", "197k", "2.0313"],
                ["linear-fourier", "2.21M", "145k", "2.3619"],
                ["linear-frft30", "2.21M", "102k", "2.4124"],
                ["linear-frft45", "2.21M", "102k", "2.3729"],
              ]}
            />
            <p className="method-note">
              Five harness bugs were repaired before this conclusion: wrong expansion gradients, a frozen lazy
              activation, identical “different” seeds, unpaired batches, and noisy sampled validation.
            </p>
          </div>
        </details>

        <details>
          <summary>
            <span className="paper-number">03</span>
            <span>
              <strong>Standard substrate: the negative result survives repair</strong>
              <small>attention scaling + warmup/cosine · 4 paired seeds</small>
            </span>
            <i aria-hidden="true">+</i>
          </summary>
          <div className="paper-content">
            <p>
              Standard 1/√head_dim attention scaling and a tuned 100-step warmup + cosine schedule made the two
              dense controls agree. The result remained negative: linear-fourier lost to both dense controls in
              4/4 seeds (+0.094 / +0.099 nats). One narrower positive survived: LCT beat a parameter-identical
              rank-1 up-projection by 0.083 ± 0.007 nats.
            </p>
            <EvidenceTable
              caption="Repaired substrate · mean ± sd best validation loss"
              rows={[
                ["config", "params", "tok/s", "best val"],
                ["dense @256", "3.26M", "171k", "1.4656 ± .0103"],
                ["dense @212 · matched", "2.25M", "194k", "1.4702 ± .0050"],
                ["linear-fourier", "2.21M", "145k", "1.5645 ± .0060"],
                ["activation-fourier", "3.26M", "112k", "1.5954 ± .0080"],
                ["rank-1 MLP", "2.21M", "186k", "1.6470 ± .0065"],
              ]}
            />
            <blockquote>
              LCT structure out-buys naive factorization; neither out-buys a narrower dense model.
            </blockquote>
          </div>
        </details>

        <details>
          <summary>
            <span className="paper-number">04</span>
            <span>
              <strong>A100 + text8: scale does not reverse the ordering</strong>
              <small>34–51M params · trunk width 1024 · 2 paired seeds</small>
            </span>
            <i aria-hidden="true">+</i>
          </summary>
          <div className="paper-content">
            <p>
              At 15× the parameter scale, LCT was 1.24× faster than the same-width dense model and the rank-1 arm
              collapsed. But the matched dim-840 dense model was both faster and substantially better than every
              arm. Kernel speed did not become end-to-end advantage.
            </p>
            <EvidenceTable
              caption="A100-SXM4-40GB · text8 · best validation loss shown per seed"
              rows={[
                ["config", "params", "tok/s", "best val · 2 seeds"],
                ["dense @840 · matched", "34.2M", "64.5k", "1.2956 / 1.2938"],
                ["dense @1024", "50.7M", "48.4k", "1.3430 / 1.3271"],
                ["linear-fourier @1024", "33.9M", "60.1k", "1.4087 / 1.4107"],
                ["rank-1 @1024", "33.9M", "64.8k", "2.0406 / 1.8816"],
              ]}
            />
          </div>
        </details>

        <details>
          <summary>
            <span className="paper-number">05</span>
            <span>
              <strong>H100 learnable path: implementation passes, quality does not</strong>
              <small>Exploratory/debug only · 1 seed · 500 steps · not confirmatory</small>
            </span>
            <i aria-hidden="true">+</i>
          </summary>
          <div className="paper-content debug-result">
            <div className="exploratory-banner">
              <strong>Exploratory / debug result</strong>
              <span>tiny Shakespeare</span>
              <span>width 1024</span>
              <span>4 layers / 8 heads</span>
              <span>NVIDIA H100 80GB</span>
              <span>1 seed / 500 steps</span>
              <b>Not confirmatory</b>
            </div>
            <p>
              This short run asks a narrow engineering question: does the canonical
              M = C(k) · S(s) · R(θ) path actually update its transform parameters on GPU while retaining the
              determinant-one constraint? Yes. It does not ask—or support—a positive model-quality claim.
            </p>
            <EvidenceTable
              caption="H100 learnability probe · exact artifact values · best validation loss at 500 steps"
              rows={[
                ["config", "parameters", "tok/s", "val loss"],
                ["dense", "50,782,273", "122,151.85", "1.7425669"],
                ["historical fixed FFT LCT", "34,021,453", "144,000.68", "1.9404578"],
                ["canonical fixed", "34,021,453", "125,735.66", "1.9438160"],
                ["learned canonical", "34,021,453", "63,751.19", "1.9790263"],
              ]}
            />
            <div className="debug-diagnostics" aria-label="Learnable transform debug diagnostics">
              <div>
                <span>canonical initial val · fixed = learned</span>
                <strong>4.3599734438</strong>
              </div>
              <div>
                <span>frozen max parameter Δ</span>
                <strong>0</strong>
              </div>
              <div>
                <span>learned max parameter Δ</span>
                <strong>0.0019682646</strong>
              </div>
              <div>
                <span>max determinant error</span>
                <strong>1.1920929e-7</strong>
              </div>
            </div>
            <div className="debug-conclusion">
              <span>What this establishes</span>
              <p>
                The implementation now truly learns and preserves det = 1 to float32 precision. But the learned
                canonical arm is <strong>0.0352 nats worse</strong> than canonical fixed and runs at 0.507× its
                throughput—roughly half. This validates the mechanism, not the model. There is no positive quality
                claim.
              </p>
            </div>
            <a
              className="artifact-link"
              href="https://github.com/alok/linear_canonical_transform/blob/main/paper/results/modal_h100_learnable_s1.json"
              target="_blank"
              rel="noreferrer"
            >
              Inspect the raw H100 artifact <span aria-hidden="true">↗</span>
            </a>
          </div>
        </details>

        <details>
          <summary>
            <span className="paper-number">06</span>
            <span>
              <strong>The stronger next hypothesis is square projection speed</strong>
              <small>Promising microbenchmark · not yet a model result</small>
            </span>
            <i aria-hidden="true">+</i>
          </summary>
          <div className="paper-content hypothesis-layout">
            <div className="hypothesis-figure" aria-label="Square projection speedups on A100">
              <span>square LCT / dense speed</span>
              <div><i style={{ height: "14%" }} /><b>2.5×</b><small>1024</small></div>
              <div><i style={{ height: "49%" }} /><b>8.7×</b><small>4096</small></div>
              <div><i style={{ height: "100%" }} /><b>17.9×</b><small>8192</small></div>
            </div>
            <div>
              <p>
                The A100 microbenchmark reaches 17.9× forward and 18.3× forward+backward speedup on square 8192
                maps. The tested MLP placement was rectangular (1024 → 4096), where the whole-model advantage
                disappeared against a narrower dense trunk.
              </p>
              <p>
                The sharper experiment is therefore square attention projections at width ≥2048, where the FFT
                structure aligns with the shape that produced the strongest kernel result. The H100 learnability
                probe above still used the rectangular MLP placement; it did not test square attention maps.
              </p>
              <aside className="honesty-note">
                <strong>Status: untested</strong>
                This is the best next hypothesis, not evidence that attention placement works.
              </aside>
            </div>
          </div>
        </details>
      </div>
    </section>
  );
}

export default function Home() {
  const [params, setParams] = useState<TransformParams>({ theta: -28, logScale: 0.25, shear: 0.55 });
  const [camera, setCamera] = useState<Camera>(DEFAULT_CAMERA);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  const animateTo = useCallback(
    (target: TransformParams) => {
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
      const start = params;
      const duration = 720;
      const startedAt = performance.now();
      return new Promise<void>((resolve) => {
        const frame = (now: number) => {
          const progress = clamp((now - startedAt) / duration, 0, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          setParams({
            theta: start.theta + (target.theta - start.theta) * eased,
            logScale: start.logScale + (target.logScale - start.logScale) * eased,
            shear: start.shear + (target.shear - start.shear) * eased,
          });
          if (progress < 1) {
            animationRef.current = requestAnimationFrame(frame);
          } else {
            animationRef.current = null;
            setParams(target);
            resolve();
          }
        };
        animationRef.current = requestAnimationFrame(frame);
      });
    },
    [params],
  );

  const updateParam = (key: keyof TransformParams, value: number) => {
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    setParams((current) => ({ ...current, [key]: value }));
  };

  const setPreset = (next: TransformParams) => {
    void animateTo(next);
  };

  return (
    <main>
      <header className="site-header">
        <a href="#top" className="wordmark" aria-label="LCT interactive paper home">
          <span>LCT</span>
          <small>an interactive paper</small>
        </a>
        <nav aria-label="Page sections">
          <a href="#instrument">Instrument</a>
          <a href="#fair-control">Fair control</a>
          <a href="#evidence">Evidence</a>
        </nav>
        <a
          className="source-link"
          href="https://github.com/alok/linear_canonical_transform"
          target="_blank"
          rel="noreferrer"
        >
          source ↗
        </a>
      </header>

      <div id="top" className="hero-intro">
        <div className="hero-meta">
          <p>Linear canonical transforms</p>
          <span>Alok Singh · 2026</span>
          <span>PyTorch + MLX</span>
        </div>
        <div className="hero-title">
          <p className="section-label">Geometry you can touch. Evidence you can audit.</p>
          <h1>The transform keeps its area. <em>The benchmark keeps us honest.</em></h1>
          <p className="hero-deck">
            Rotate, squeeze, and shear phase space in three dimensions. Then inspect why a mathematically elegant,
            fast structured layer still lost to a well-chosen dense control.
          </p>
        </div>
        <aside className="hero-thesis">
          <span>Verdict</span>
          <p>Not a dense-layer improvement in these experiments.</p>
          <small>Faster at large same-width maps. Better than rank-1 at equal parameters. Neither is the fair-control win.</small>
        </aside>
      </div>

      <section className="phase-lab" id="instrument" aria-labelledby="instrument-title">
        <div className="lab-topline">
          <div>
            <span>Figure 01 · live phase space</span>
            <h2 id="instrument-title">A determinant-one machine</h2>
          </div>
          <p>
            Every intermediate slice is built from determinant-one factors. The 3D axis is composition—not time,
            loss, or a learned NanoGPT trajectory.
          </p>
        </div>

        <div className="lab-grid">
          <PhaseSpaceCanvas params={params} camera={camera} setCamera={setCamera} />
          <aside className="transform-inspector">
            <MatrixReadout params={params} />

            <div className="controls-heading">
              <span>compose the operator</span>
              <button type="button" onClick={() => setPreset(IDENTITY)}>identity</button>
            </div>
            <RangeControl
              id="rotation"
              label="phase rotation"
              symbol="θ"
              min={-90}
              max={90}
              step={1}
              value={params.theta}
              unit="°"
              onChange={(value) => updateParam("theta", value)}
            />
            <RangeControl
              id="log-scale"
              label="reciprocal log-scale"
              symbol="s"
              min={-0.8}
              max={0.8}
              step={0.01}
              value={params.logScale}
              onChange={(value) => updateParam("logScale", value)}
            />
            <RangeControl
              id="shear"
              label="chirp shear"
              symbol="k"
              min={-1.5}
              max={1.5}
              step={0.01}
              value={params.shear}
              onChange={(value) => updateParam("shear", value)}
            />

            <div className="preset-row" aria-label="Transform presets">
              <button type="button" onClick={() => setPreset({ theta: 90, logScale: 0, shear: 0 })}>Fourier</button>
              <button type="button" onClick={() => setPreset({ theta: 0, logScale: 0.62, shear: 0 })}>squeeze</button>
              <button type="button" onClick={() => setPreset({ theta: 0, logScale: 0, shear: 1.05 })}>shear</button>
            </div>

            <div className="factor-proof">
              <span>det C</span><b>1</b>
              <span>det S</span><b>eˢ · e⁻ˢ = 1</b>
              <span>det R</span><b>1</b>
            </div>
          </aside>
        </div>

        <LearningLoop params={params} animateTo={animateTo} />
      </section>

      <FairComparison />
      <PaperSections />

      <section className="closing-note">
        <p className="section-label">The useful boundary</p>
        <blockquote>
          Elegant structure is a hypothesis generator. A matched control is a truth filter.
        </blockquote>
        <div>
          <p>
            Scope: character-level language modeling, 2M–50M parameters. The confirmatory A100 study has two
            paired seeds; the H100 learnable-path run is a one-seed, 500-step debug result. Square attention
            projections remain untested.
          </p>
          <a href="#instrument">Return to the instrument ↑</a>
        </div>
      </section>

      <footer>
        <span>lct-activation</span>
        <span>H100 debug added · square attention projections remain untested</span>
        <a href="https://github.com/alok/linear_canonical_transform" target="_blank" rel="noreferrer">
          code + artifacts ↗
        </a>
      </footer>
    </main>
  );
}
