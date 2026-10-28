"use client";

import {
  PointerEvent as ReactPointerEvent,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Box = [number, number, number, number];
type Mode = "automatic" | "assisted" | "manual";

type Detection = {
  vehicle_type: string;
  confidence_percent: number;
  bounding_box: Box;
};

type DetectionResponse = {
  image_width: number;
  image_height: number;
  detections: Detection[];
};

export type SelectionPayload = {
  boundingBox: Box;
  mask: Blob | null;
  mode: Mode;
};

export type VehicleSelectorHandle = {
  getSelection: () => Promise<SelectionPayload | null>;
};

type Props = {
  file: File;
  previewUrl: string;
  onReadyChange: (ready: boolean) => void;
  onError: (message: string | null) => void;
};

function normalizedBox(box: Box): Box {
  return [
    Math.min(box[0], box[2]),
    Math.min(box[1], box[3]),
    Math.max(box[0], box[2]),
    Math.max(box[1], box[3]),
  ];
}

function intersectionScore(a: Box, b: Box) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
  return intersection / areaA;
}

function canvasBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
}

const VehicleSelector = forwardRef<VehicleSelectorHandle, Props>(function VehicleSelector(
  { file, previewUrl, onReadyChange, onError },
  ref,
) {
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const assistedStartRef = useRef<[number, number] | null>(null);
  const [mode, setMode] = useState<Mode>("automatic");
  const [scan, setScan] = useState<DetectionResponse | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [assistedBox, setAssistedBox] = useState<Box | null>(null);
  const [draftBox, setDraftBox] = useState<Box | null>(null);
  const [brushSize, setBrushSize] = useState(44);
  const [erasing, setErasing] = useState(false);
  const [hasPaint, setHasPaint] = useState(false);
  const [isFinding, setIsFinding] = useState(true);

  const automaticBox = scan?.detections[selectedIndex]?.bounding_box ?? null;
  const activeBox = mode === "automatic" ? automaticBox : mode === "assisted" ? assistedBox : null;

  useEffect(() => {
    let cancelled = false;
    setIsFinding(true);
    setScan(null);
    setSelectedIndex(0);
    setAssistedBox(null);
    setDraftBox(null);
    setMode("automatic");
    setHasPaint(false);
    onReadyChange(false);
    onError(null);
    const body = new FormData();
    body.append("file", file);
    fetch(`${API_URL}/api/v1/detect`, { method: "POST", body })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail ?? "Vehicle search failed.");
        return payload as DetectionResponse;
      })
      .then((payload) => {
        if (cancelled) return;
        setScan(payload);
        if (payload.detections.length === 0) {
          setMode("manual");
          onError("No automatic outline was found. Paint over the vehicle manually.");
        }
      })
      .catch((caught) => {
        if (cancelled) return;
        setMode("manual");
        onError(caught instanceof Error ? caught.message : "Vehicle search failed.");
      })
      .finally(() => {
        if (!cancelled) setIsFinding(false);
      });
    return () => {
      cancelled = true;
    };
  }, [file, onError, onReadyChange]);

  useEffect(() => {
    onReadyChange(Boolean(activeBox) || (mode === "manual" && hasPaint));
  }, [activeBox, hasPaint, mode, onReadyChange]);

  useImperativeHandle(ref, () => ({
    async getSelection() {
      if (mode !== "manual") {
        return activeBox ? { boundingBox: activeBox, mask: null, mode } : null;
      }
      const canvas = canvasRef.current;
      if (!canvas || !hasPaint) return null;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) return null;
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let left = canvas.width;
      let top = canvas.height;
      let right = -1;
      let bottom = -1;
      for (let y = 0; y < canvas.height; y += 2) {
        for (let x = 0; x < canvas.width; x += 2) {
          if (pixels[(y * canvas.width + x) * 4 + 3] > 10) {
            left = Math.min(left, x);
            top = Math.min(top, y);
            right = Math.max(right, x);
            bottom = Math.max(bottom, y);
          }
        }
      }
      if (right < left || bottom < top) return null;
      return {
        boundingBox: [left, top, Math.min(canvas.width, right + 2), Math.min(canvas.height, bottom + 2)],
        mask: await canvasBlob(canvas),
        mode,
      };
    },
  }));

  function imagePoint(event: ReactPointerEvent): [number, number] {
    const image = imageRef.current;
    if (!image) return [0, 0];
    const rect = image.getBoundingClientRect();
    return [
      Math.round(((event.clientX - rect.left) / rect.width) * image.naturalWidth),
      Math.round(((event.clientY - rect.top) / rect.height) * image.naturalHeight),
    ];
  }

  function prepareCanvas() {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas) return;
    if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
    }
  }

  function pointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (mode === "automatic") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = imagePoint(event);
    if (mode === "assisted") {
      assistedStartRef.current = point;
      setDraftBox([point[0], point[1], point[0], point[1]]);
      return;
    }
    prepareCanvas();
    drawingRef.current = true;
    paint(point, point);
  }

  function pointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const point = imagePoint(event);
    if (mode === "assisted" && assistedStartRef.current) {
      setDraftBox(normalizedBox([
        assistedStartRef.current[0],
        assistedStartRef.current[1],
        point[0],
        point[1],
      ]));
    } else if (mode === "manual" && drawingRef.current) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const previous = (canvas.dataset.previous ?? `${point[0]},${point[1]}`).split(",").map(Number);
      paint([previous[0], previous[1]], point);
      canvas.dataset.previous = `${point[0]},${point[1]}`;
    }
  }

  function pointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (mode === "assisted" && draftBox) {
      const drawn = normalizedBox(draftBox);
      const snapped = scan?.detections
        .map((detection) => ({ box: detection.bounding_box, score: intersectionScore(drawn, detection.bounding_box) }))
        .sort((a, b) => b.score - a.score)[0];
      setAssistedBox(snapped && snapped.score > 0.08 ? snapped.box : drawn);
      setDraftBox(null);
      assistedStartRef.current = null;
    }
    drawingRef.current = false;
    if (canvasRef.current) delete canvasRef.current.dataset.previous;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function paint(from: [number, number], to: [number, number]) {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.save();
    context.globalCompositeOperation = erasing ? "destination-out" : "source-over";
    context.strokeStyle = "white";
    context.lineWidth = brushSize;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(from[0], from[1]);
    context.lineTo(to[0], to[1]);
    context.stroke();
    context.restore();
    if (!erasing) setHasPaint(true);
  }

  function clearPaint() {
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    setHasPaint(false);
  }

  function boxStyle(box: Box) {
    const width = scan?.image_width || imageRef.current?.naturalWidth || 1;
    const height = scan?.image_height || imageRef.current?.naturalHeight || 1;
    return {
      left: `${(box[0] / width) * 100}%`,
      top: `${(box[1] / height) * 100}%`,
      width: `${((box[2] - box[0]) / width) * 100}%`,
      height: `${((box[3] - box[1]) / height) * 100}%`,
    };
  }

  return (
    <section className="selector-card" aria-label="Choose the vehicle to analyze">
      <div className="selector-heading">
        <div>
          <span className="eyebrow">Choose the target</span>
          <h3>{isFinding ? "Finding vehicles…" : "Is this the right car?"}</h3>
        </div>
        <span className="selector-count">
          {scan ? `${scan.detections.length} vehicle${scan.detections.length === 1 ? "" : "s"} found` : "Scanning"}
        </span>
      </div>

      <div className="mode-tabs" role="tablist" aria-label="Vehicle selection method">
        <button className={mode === "automatic" ? "active" : ""} disabled={!scan?.detections.length} onClick={() => setMode("automatic")}>Auto find</button>
        <button className={mode === "assisted" ? "active" : ""} onClick={() => setMode("assisted")}>Snap outline</button>
        <button className={mode === "manual" ? "active" : ""} onClick={() => setMode("manual")}>Manual brush</button>
      </div>

      <p className="mode-help">
        {mode === "automatic" && "Click an outlined vehicle. The bright outline is the exact area MotorLens will scan."}
        {mode === "assisted" && "Drag a rough box around a car. Its edges snap to the nearest detected vehicle."}
        {mode === "manual" && "Paint only the vehicle. Adjust the brush for broad bodywork or fine edges; use Eraser to tidy the mask."}
      </p>

      <div
        className={`selection-stage mode-${mode}`}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img ref={imageRef} src={previewUrl} alt="Vehicle selection preview" onLoad={prepareCanvas} draggable={false} />
        {mode === "automatic" && scan?.detections.map((detection, index) => (
          <button
            key={`${detection.bounding_box.join("-")}-${index}`}
            className={`vehicle-box ${selectedIndex === index ? "selected" : ""}`}
            style={boxStyle(detection.bounding_box)}
            onClick={(event) => { event.stopPropagation(); setSelectedIndex(index); }}
            aria-label={`Select ${detection.vehicle_type} ${index + 1}, ${detection.confidence_percent.toFixed(0)} percent detection confidence`}
          >
            <span>{index + 1}</span>
          </button>
        ))}
        {mode === "assisted" && (draftBox || assistedBox) && (
          <div className={`assisted-box ${draftBox ? "drawing" : "snapped"}`} style={boxStyle(draftBox ?? assistedBox!)}>
            <span>{draftBox ? "Release to snap" : "Snapped scan area"}</span>
          </div>
        )}
        <canvas ref={canvasRef} className="paint-canvas" aria-hidden={mode !== "manual"} />
      </div>

      {mode === "manual" && (
        <div className="brush-controls">
          <label>
            <span>Brush size</span>
            <input type="range" min="12" max="120" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
            <output>{brushSize}px</output>
          </label>
          <div className="brush-actions">
            <button className={!erasing ? "active" : ""} onClick={() => setErasing(false)}>Paint</button>
            <button className={erasing ? "active eraser" : ""} onClick={() => setErasing(true)}>Eraser</button>
            <button onClick={clearPaint}>Clear</button>
          </div>
        </div>
      )}
    </section>
  );
});

export default VehicleSelector;
