from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceMatch:
    reference_id: str
    make: str
    model: str
    generation: str
    year_range: str
    similarity: float
    source_url: str
    license: str
    attribution: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "reference_id": self.reference_id,
            "make": self.make,
            "model": self.model,
            "generation": self.generation,
            "year_range": self.year_range,
            "similarity_percent": round(self.similarity * 100, 2),
            "source_url": self.source_url,
            "license": self.license,
            "attribution": self.attribution,
        }


@dataclass(frozen=True)
class VehiclePrediction:
    make: str
    model: str
    year_range: str
    confidence: float
    raw_label: str
    references: tuple[ReferenceMatch, ...] = ()
    reference_supported: bool = False
    generation: str = "Generation not mapped"

    def to_dict(self) -> dict[str, str | float]:
        return {
            "make": self.make,
            "model": self.model,
            "year_range": self.year_range,
            "generation": self.generation,
            "confidence_percent": round(self.confidence * 100, 2),
            "raw_label": self.raw_label,
            "references": [reference.to_dict() for reference in self.references],
            "reference_supported": self.reference_supported,
        }


@dataclass(frozen=True)
class ClassificationResult:
    predictions: tuple[VehiclePrediction, ...]
    low_confidence: bool
    message: str
    model_id: str
    limitations: str
    explanation: str
    abstained: bool = False
    decision_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "low_confidence": self.low_confidence,
            "message": self.message,
            "model_id": self.model_id,
            "limitations": self.limitations,
            "explanation": self.explanation,
            "abstained": self.abstained,
            "decision_reason": self.decision_reason,
        }
