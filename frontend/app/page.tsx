"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import VehicleSelector, { VehicleSelectorHandle } from "./VehicleSelector";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

type ReferenceMatch = {
  reference_id: string;
  make: string;
  model: string;
  generation: string;
  year_range: string;
  similarity_percent: number;
  source_url: string;
  license: string;
  attribution: string;
};

type VehiclePrediction = {
  make: string;
  model: string;
  generation: string;
  year_range: string;
  confidence_percent: number;
  raw_label: string;
  references: ReferenceMatch[];
  reference_supported: boolean;
};

type PredictionResponse = {
  prediction_id: string;
  detection: {
    vehicle_type: string;
    confidence_percent: number;
    bounding_box: [number, number, number, number];
  };
  classification: {
    predictions: VehiclePrediction[];
    low_confidence: boolean;
    message: string;
    model_id: string;
    limitations: string;
    explanation: string;
    abstained: boolean;
    decision_reason: string;
  };
  attention_image_data_url: string;
};

function confidenceTone(value: number) {
  if (value >= 70) return "strong";
  if (value >= 35) return "medium";
  return "weak";
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const selectorRef = useRef<VehicleSelectorHandle>(null);
  const [showNotice, setShowNotice] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedRank, setSelectedRank] = useState<number | null>(null);
  const [confirmedRank, setConfirmedRank] = useState<number | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [selectionReady, setSelectionReady] = useState(false);

  const handleSelectionReady = useCallback((ready: boolean) => setSelectionReady(ready), []);
  const handleSelectionError = useCallback((message: string | null) => setError(message), []);

  useEffect(() => {
    if (!showNotice) return;
    const dismiss = () => setShowNotice(false);
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") dismiss();
    };
    document.addEventListener("click", dismiss, { once: true });
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("click", dismiss);
      document.removeEventListener("keydown", escape);
    };
  }, [showNotice]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function chooseFile(nextFile: File) {
    setError(null);
    setResult(null);
    setSelectedRank(null);
    setConfirmedRank(null);
    setFeedbackStatus(null);
    setSelectionReady(false);

    if (!ACCEPTED_TYPES.has(nextFile.type)) {
      setError("Choose a JPG, JPEG, PNG, or WEBP image.");
      return;
    }
    if (nextFile.size > MAX_FILE_BYTES) {
      setError("That image is larger than 10 MB. Choose a smaller file.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
  }

  function handleInput(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (nextFile) chooseFile(nextFile);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const nextFile = event.dataTransfer.files?.[0];
    if (nextFile) chooseFile(nextFile);
  }

  async function analyze() {
    if (!file) return;
    setError(null);
    setResult(null);
    setIsAnalyzing(true);
    setSelectedRank(null);
    setConfirmedRank(null);
    setFeedbackStatus(null);
    try {
      const selection = await selectorRef.current?.getSelection();
      if (!selection) throw new Error("Choose or highlight the vehicle before analyzing it.");
      const body = new FormData();
      body.append("file", file);
      body.append("selection_left", String(selection.boundingBox[0]));
      body.append("selection_top", String(selection.boundingBox[1]));
      body.append("selection_right", String(selection.boundingBox[2]));
      body.append("selection_bottom", String(selection.boundingBox[3]));
      if (selection.mask) body.append("selection_mask", selection.mask, "selection.png");
      const response = await fetch(`${API_URL}/api/v1/predict`, {
        method: "POST",
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "The vehicle could not be analyzed.");
      }
      setResult(payload as PredictionResponse);
      setSelectedRank(1);
      requestAnimationFrame(() =>
        document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }),
      );
    } catch (caught) {
      setError(
        caught instanceof TypeError
          ? "The AI service is unavailable. Make sure FastAPI is running on port 8000."
          : caught instanceof Error
            ? caught.message
            : "Something went wrong while analyzing the image.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function sendFeedback(contribute: boolean) {
    if (!result || !selectedRank || !file) return;
    setIsSubmittingFeedback(true);
    setFeedbackStatus(null);
    try {
      const body = new FormData();
      body.append("prediction_id", result.prediction_id);
      body.append("selected_rank", String(selectedRank));
      body.append("user_confidence", "certain");
      body.append("consent_to_training", String(contribute));
      if (contribute) body.append("file", file);
      const response = await fetch(`${API_URL}/api/v1/feedback`, {
        method: "POST",
        body,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Feedback could not be saved.");
      setConfirmedRank(selectedRank);
      setFeedbackStatus(
        contribute
          ? "Thank you. A metadata-free copy is pending review for future training."
          : "Match confirmed. Your image was not stored.",
      );
    } catch (caught) {
      setFeedbackStatus(
        caught instanceof Error ? caught.message : "Feedback could not be saved.",
      );
    } finally {
      setIsSubmittingFeedback(false);
    }
  }

  const bestMatch = result?.classification.predictions[0];

  return (
    <main className="site-shell">
      {showNotice && (
        <div className="notice-layer" role="presentation">
          <section className="privacy-notice" role="dialog" aria-modal="true" aria-label="Image privacy notice">
            <button className="notice-close" onClick={() => setShowNotice(false)} aria-label="Close privacy notice">×</button>
            <span className="eyebrow">Privacy, in plain English</span>
            <h2>Your image is analyzed, not automatically collected.</h2>
            <p>
              We only save a sanitized copy if you later choose <strong>Contribute this image</strong>.
              Clicking anywhere dismisses this note and does not authorize storage.
            </p>
            <span className="dismiss-hint">Click anywhere to continue</span>
          </section>
        </div>
      )}

      <header className="topbar">
        <a className="brand" href="#top" aria-label="MotorLens home">
          <span className="brand-mark">ML</span>
          <span>MotorLens</span>
        </a>
        <div className="model-status"><span /> Local AI pipeline</div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="eyebrow">Vehicle intelligence · Free & local</span>
          <h1>Identify the car.<br /><em>Know the uncertainty.</em></h1>
          <p className="hero-lede">
            Upload one vehicle photo. MotorLens isolates the car, ranks five possible matches,
            and shows the visual evidence behind the result.
          </p>
          <div className="trust-row">
            <span>No paid API</span>
            <span>Top-five matches</span>
            <span>Generation ranges</span>
          </div>
        </div>

        <div className="upload-panel">
          {!previewUrl ? <div
            className={`dropzone ${isDragging ? "dragging" : ""} ${previewUrl ? "has-image" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
            }}
            aria-label="Choose a vehicle image"
          >
            <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={handleInput} hidden />
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="Selected vehicle preview" className="preview-image" />
            ) : (
              <div className="dropzone-empty">
                <span className="camera-glyph">◎</span>
                <strong>Drop a vehicle photo here</strong>
                <span>or click to browse your files</span>
                <small>JPG, PNG or WEBP · 10 MB maximum</small>
              </div>
            )}
            {previewUrl && <span className="replace-chip">Choose another</span>}
          </div> : file ? (
            <>
              <VehicleSelector
                ref={selectorRef}
                file={file}
                previewUrl={previewUrl}
                onReadyChange={handleSelectionReady}
                onError={handleSelectionError}
              />
              <button className="replace-button" onClick={() => inputRef.current?.click()}>
                Choose another photo
              </button>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={handleInput} hidden />
            </>
          ) : null}

          <button className="analyze-button" disabled={!file || !selectionReady || isAnalyzing} onClick={(event) => { event.stopPropagation(); analyze(); }}>
            {isAnalyzing ? <><span className="spinner" /> Inspecting the vehicle…</> : <>Analyze vehicle <span>→</span></>}
          </button>
          <p className="upload-note">The first analysis may take longer while the local models load.</p>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </div>
      </section>

      {result && bestMatch && (
        <section className="results-section" id="results">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Analysis complete</span>
              <h2>Five possibilities, ranked.</h2>
            </div>
            <span className={`confidence-badge ${result.classification.low_confidence ? "uncertain" : "supported"}`}>
              {result.classification.abstained ? "No reliable identification" : result.classification.low_confidence ? "Uncertain result" : "Reference supported"}
            </span>
          </div>

          <div className="result-grid">
            <article className="primary-result">
              <div className="primary-topline">
                <span>Most likely match</span>
                <span>Detection {result.detection.confidence_percent.toFixed(1)}%</span>
              </div>
              <h3>{bestMatch.make}</h3>
              <p className="model-name">{bestMatch.model}</p>
              <div className="generation-block">
                <span>{bestMatch.generation}</span>
                <strong>{bestMatch.year_range}</strong>
              </div>
              <div className="confidence-meter">
                <div className="meter-label"><span>Model confidence</span><strong>{bestMatch.confidence_percent.toFixed(1)}%</strong></div>
                <div className="meter-track"><span className={confidenceTone(bestMatch.confidence_percent)} style={{ width: `${Math.max(3, bestMatch.confidence_percent)}%` }} /></div>
              </div>
              <div className="uncertainty-note">
                <strong>{result.classification.low_confidence ? "Why this is uncertain" : "Why this is supported"}</strong>
                <p>{result.classification.message}</p>
                {result.classification.decision_reason && <p>{result.classification.decision_reason}</p>}
              </div>
            </article>

            <article className="evidence-card">
              <div className="evidence-heading"><span>Visual attention</span><small>Not causal proof</small></div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={result.attention_image_data_url} alt="AI attention overlay showing influential vehicle regions" />
              <p>The red areas show where the transformer concentrated while comparing vehicle shapes and details.</p>
            </article>
          </div>

          <div className="candidate-panel">
            <div className="candidate-heading">
              <div><span className="eyebrow">Help improve the result</span><h3>Which one is it?</h3></div>
              <p>Choose the closest match. You can confirm without storing your photo.</p>
            </div>
            <div className="candidate-list" role="radiogroup" aria-label="Vehicle candidates">
              {result.classification.predictions.map((prediction, index) => {
                const rank = index + 1;
                const selected = selectedRank === rank;
                return (
                  <button key={`${prediction.raw_label}-${rank}`} className={`candidate-row ${selected ? "selected" : ""}`} onClick={() => setSelectedRank(rank)} role="radio" aria-checked={selected}>
                    <span className="rank">{String(rank).padStart(2, "0")}</span>
                    <span className="candidate-name"><strong>{prediction.make} {prediction.model}</strong><small>{prediction.generation} · {prediction.year_range}</small></span>
                    <span className="candidate-score">{prediction.confidence_percent.toFixed(1)}%</span>
                    <span className="radio-dot" />
                  </button>
                );
              })}
            </div>

            <div className="feedback-actions">
              <button className="confirm-button" disabled={!selectedRank || isSubmittingFeedback} onClick={() => sendFeedback(false)}>
                {confirmedRank === selectedRank ? "Confirmed ✓" : "Confirm match — don’t store image"}
              </button>
              {confirmedRank === selectedRank && (
                <button className="contribute-button" disabled={isSubmittingFeedback} onClick={() => sendFeedback(true)}>
                  Contribute this image
                </button>
              )}
              <span>Contribution strips EXIF data and enters a review queue.</span>
            </div>
            {feedbackStatus && <div className="feedback-status" role="status">{feedbackStatus}</div>}
          </div>

          <details className="limitations">
            <summary>Model limitations and technical detail</summary>
            <p>{result.classification.limitations}</p>
            <p>{result.classification.explanation}</p>
            <code>{result.classification.model_id}</code>
          </details>
        </section>
      )}

      <footer>
        <span>MotorLens · Local prototype</span>
        <span>Angle, lighting, damage and modifications can affect results.</span>
      </footer>
    </main>
  );
}
