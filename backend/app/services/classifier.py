import re
from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image
from torch.nn import functional as torch_functional
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.app.core.config import (
    CLASSIFIER_MODEL_ID,
    GENERATION_CATALOG_PATH,
    LOW_CLASSIFICATION_CONFIDENCE,
    UNKNOWN_CLASSIFICATION_CONFIDENCE,
    UNKNOWN_CLASSIFICATION_MARGIN,
    REFERENCE_GALLERY_PATH,
    REFERENCE_SUPPORT_THRESHOLD,
    TOP_PREDICTIONS,
)
from backend.app.schemas.prediction import ClassificationResult, VehiclePrediction
from backend.app.services.reference_store import ReferenceGallery
from backend.app.services.generation_catalog import GenerationCatalog

KNOWN_MAKES = tuple(
    sorted(
        {
            "AM General",
            "Acura",
            "Aston Martin",
            "Audi",
            "BMW",
            "Bentley",
            "Bugatti",
            "Buick",
            "Cadillac",
            "Chevrolet",
            "Chrysler",
            "Dodge",
            "FIAT",
            "Ferrari",
            "Ford",
            "GMC",
            "Honda",
            "Hyundai",
            "Infiniti",
            "Isuzu",
            "Jaguar",
            "Jeep",
            "Lamborghini",
            "Land Rover",
            "Lincoln",
            "MINI",
            "Maybach",
            "McLaren",
            "Mercedes-Benz",
            "Mitsubishi",
            "Nissan",
            "Plymouth",
            "Pontiac",
            "Porsche",
            "Ram",
            "Rolls-Royce",
            "Scion",
            "Spyker",
            "Subaru",
            "Suzuki",
            "Tesla",
            "Toyota",
            "Volkswagen",
            "Volvo",
        },
        key=len,
        reverse=True,
    )
)

YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")


class VehicleClassifier(Protocol):
    def classify(self, image: Image.Image) -> ClassificationResult:
        """Return five ranked fine-grained vehicle predictions."""


@dataclass(frozen=True)
class ModelAnalysis:
    result: ClassificationResult
    attention_overlay: Image.Image
    embedding: tuple[float, ...]


def parse_vehicle_label(label: str) -> tuple[str, str, int | None]:
    """Parse Stanford Cars labels such as 'Audi S5 Coupe 2012'."""
    clean_label = " ".join(label.replace("_", " ").split())
    year_match = YEAR_PATTERN.search(clean_label)
    year = int(year_match.group(1)) if year_match else None
    without_year = YEAR_PATTERN.sub("", clean_label).strip(" -,")

    for make in KNOWN_MAKES:
        if without_year.casefold().startswith(make.casefold() + " "):
            model = without_year[len(make) :].strip()
            return make, model or "Unknown model", year

    first, separator, remainder = without_year.partition(" ")
    if separator:
        return first, remainder, year
    return "Unknown make", without_year or label, year


def display_year_range(year: int | None, confidence: float) -> str:
    if year is None:
        return "Generation/year range unavailable"
    if confidence >= LOW_CLASSIFICATION_CONFIDENCE:
        return str(year)
    return f"{year - 2}-{year + 2} (estimated range)"


class HuggingFaceVehicleClassifier:
    def __init__(
        self,
        model_id: str = CLASSIFIER_MODEL_ID,
        device: str | None = None,
        reference_gallery: ReferenceGallery | None = None,
        generation_catalog: GenerationCatalog | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, attn_implementation="eager"
        )
        self.model.to(self.device).eval()
        self.reference_gallery = reference_gallery or ReferenceGallery.load(
            REFERENCE_GALLERY_PATH
        )
        self.generation_catalog = generation_catalog or GenerationCatalog.load(
            GENERATION_CATALOG_PATH
        )

    @torch.inference_mode()
    def classify(self, image: Image.Image) -> ClassificationResult:
        return self.analyze(image).result

    @torch.inference_mode()
    def analyze(self, image: Image.Image) -> ModelAnalysis:
        rgb_image = image.convert("RGB")
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        probabilities = outputs.logits.softmax(dim=-1)[0]
        embedding = tuple(
            float(value) for value in outputs.hidden_states[-1][0, 0].detach().cpu()
        )
        count = min(TOP_PREDICTIONS, probabilities.numel())
        scores, indices = probabilities.topk(count)

        predictions: list[VehiclePrediction] = []
        for score_tensor, index_tensor in zip(scores, indices):
            confidence = float(score_tensor.item())
            class_id = int(index_tensor.item())
            raw_label = self.model.config.id2label.get(
                class_id, self.model.config.id2label.get(str(class_id), f"Class {class_id}")
            )
            make, model, year = parse_vehicle_label(str(raw_label))
            generation_match = self.generation_catalog.lookup(make, model, year)
            references = self.reference_gallery.nearest(
                embedding, make=make, limit=3
            )
            reference_supported = _has_reference_support(model, references)
            predictions.append(
                VehiclePrediction(
                    make=make,
                    model=model,
                    year_range=(
                        generation_match.year_range
                        if generation_match
                        else display_year_range(
                            year, confidence if reference_supported else 0.0
                        )
                    ),
                    confidence=confidence,
                    raw_label=str(raw_label),
                    references=references,
                    reference_supported=reference_supported,
                    generation=(
                        generation_match.generation
                        if generation_match
                        else "Generation not mapped"
                    ),
                )
            )

        low_confidence = (
            predictions[0].confidence < LOW_CLASSIFICATION_CONFIDENCE
            or not predictions[0].reference_supported
        )
        second_confidence = predictions[1].confidence if len(predictions) > 1 else 0.0
        margin = predictions[0].confidence - second_confidence
        abstained, decision_reason = recognition_decision(
            predictions[0].confidence, margin
        )
        if abstained:
            message = (
                "The classifier cannot separate this image from its known classes. "
                "Treat the five matches as suggestions, not an identification."
            )
        elif predictions[0].confidence < LOW_CLASSIFICATION_CONFIDENCE:
            message = "Low model score. Treat all matches and year ranges as possibilities."
        elif not predictions[0].reference_supported:
            message = (
                "The model score is high, but no verified same-model reference corroborates "
                "it. Treat the result as uncertain."
            )
        else:
            message = "The best match is supported by both the model and reference gallery."
        result = ClassificationResult(
            predictions=tuple(predictions),
            low_confidence=low_confidence,
            message=message,
            model_id=self.model_id,
            limitations=(
                "Stanford Cars has limited, mostly older reference classes. A high model "
                "score does not prove that the real-world make, model, or year is correct."
            ),
            explanation=(
                "The attention image shows where the transformer focused; it is not a "
                "causal proof. Reference matches are cosine-similar verified gallery "
                "images and are empty until that gallery has been populated."
            ),
            abstained=abstained,
            decision_reason=decision_reason,
        )
        return ModelAnalysis(
            result=result,
            attention_overlay=_attention_overlay(rgb_image, outputs.attentions),
            embedding=embedding,
        )


def _attention_overlay(
    image: Image.Image, attentions: tuple[torch.Tensor, ...] | None
) -> Image.Image:
    if not attentions:
        return image.copy()
    attention = attentions[-1][0].mean(dim=0)[0, 1:]
    side = int(attention.numel() ** 0.5)
    if side * side != attention.numel():
        return image.copy()
    attention = attention.reshape(1, 1, side, side)
    attention = torch_functional.interpolate(
        attention,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    attention -= attention.min()
    maximum = attention.max()
    if float(maximum) > 0:
        attention /= maximum
    mask = Image.fromarray(attention.mul(255).byte().cpu().numpy())
    red = Image.new("RGB", image.size, (255, 40, 20))
    return Image.composite(red, image, mask.point(lambda value: int(value * 0.55)))


def _has_reference_support(model: str, references: tuple) -> bool:
    normalized_model = _normalize_model_name(model)
    return any(
        _normalize_model_name(reference.model) == normalized_model
        and reference.similarity >= REFERENCE_SUPPORT_THRESHOLD
        for reference in references
    )


def _normalize_model_name(value: str) -> str:
    body_terms = {
        "convertible", "coupe", "hatchback", "minivan", "sedan", "suv", "van", "wagon"
    }
    return " ".join(
        token.casefold()
        for token in value.replace("-", " ").split()
        if token.casefold() not in body_terms
    )


def recognition_decision(top_confidence: float, top_two_margin: float) -> tuple[bool, str]:
    """Conservative baseline abstention policy; calibrate on the Phase 6 benchmark."""
    if top_confidence < UNKNOWN_CLASSIFICATION_CONFIDENCE:
        return True, (
            f"Best score {top_confidence:.1%} is below the provisional "
            f"{UNKNOWN_CLASSIFICATION_CONFIDENCE:.0%} support threshold."
        )
    if top_two_margin < UNKNOWN_CLASSIFICATION_MARGIN:
        return True, (
            f"The two leading classes are only {top_two_margin:.1%} apart; "
            "the image is ambiguous."
        )
    return False, "The result passed the provisional score-and-margin rejection policy."
