from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_LOCALIZED_TARGETS: tuple[str, ...] = ('sky', 'subject')
ACCEPTED_BUCKET = 'accepted'
SUGGESTED_DISCARDED_BUCKET = 'suggested_discarded'
DEFAULT_WORTH_SAVING_THRESHOLD = 0.25


def bucket_for_worth_saving(worth_saving: float, threshold: float) -> str:
    if worth_saving <= threshold:
        return SUGGESTED_DISCARDED_BUCKET
    return ACCEPTED_BUCKET

SLIDER_LIMITS: dict[str, tuple[float, float]] = {
    "temperature": (-100.0, 100.0),
    "tint": (-100.0, 100.0),
    "exposure": (-3.0, 3.0),
    "contrast": (-100.0, 100.0),
    "highlights": (-100.0, 100.0),
    "shadows": (-100.0, 100.0),
    "whites": (-100.0, 100.0),
    "blacks": (-100.0, 100.0),
    "saturation": (-100.0, 100.0),
    "vibrance": (-100.0, 100.0),
    "clarity": (-100.0, 100.0),
}


def clamp_slider(name: str, value: Any) -> float:
    low, high = SLIDER_LIMITS[name]
    numeric = float(value)
    return max(low, min(high, numeric))


def add_slider_values(left: "GlobalAdjustments", right: "GlobalAdjustments") -> "GlobalAdjustments":
    return GlobalAdjustments.from_dict(
        {name: getattr(left, name) + getattr(right, name) for name in SLIDER_LIMITS}
    )


@dataclass(slots=True)
class GlobalAdjustments:
    temperature: float = 0.0
    tint: float = 0.0
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    clarity: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalAdjustments":
        values = {}
        for name in SLIDER_LIMITS:
            values[name] = clamp_slider(name, data.get(name, 0.0))
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SLIDER_LIMITS}


@dataclass(slots=True)
class LocalizedAdjustment:
    target: str
    delta: GlobalAdjustments

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalizedAdjustment":
        return cls(
            target=str(data.get("target", "unknown")),
            delta=GlobalAdjustments.from_dict(data.get("delta", data.get("adjustments", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "delta": self.delta.to_dict()}


@dataclass(slots=True)
class EditPlan:
    baseline_settings: GlobalAdjustments
    global_delta: GlobalAdjustments
    localized_adjustments: list[LocalizedAdjustment] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    worth_saving: float = 1.0
    discard_reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditPlan":
        baseline = GlobalAdjustments.from_dict(data.get("baseline_settings", {}))
        delta_source = data.get("global_delta", data.get("global_adjustments", {}))
        localized = [
            LocalizedAdjustment.from_dict(item)
            for item in data.get("localized_adjustments", [])
            if isinstance(item, dict)
        ]
        return cls(
            baseline_settings=baseline,
            global_delta=GlobalAdjustments.from_dict(delta_source),
            localized_adjustments=localized,
            rationale=str(data.get("rationale", "")),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            worth_saving=max(0.0, min(1.0, float(data.get("worth_saving", 1.0)))),
            discard_reason=str(data.get("discard_reason", "")),
        )

    @property
    def global_adjustments(self) -> GlobalAdjustments:
        return self.final_settings()

    def final_settings(self) -> GlobalAdjustments:
        return add_slider_values(self.baseline_settings, self.global_delta)

    def to_dict(self, worth_saving_threshold: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "baseline_settings": self.baseline_settings.to_dict(),
            "global_delta": self.global_delta.to_dict(),
            "final_settings": self.final_settings().to_dict(),
            "localized_adjustments": [item.to_dict() for item in self.localized_adjustments],
            "rationale": self.rationale,
            "confidence": self.confidence,
            "worth_saving": self.worth_saving,
            "discard_reason": self.discard_reason,
        }
        if worth_saving_threshold is not None:
            payload["worth_saving_threshold"] = worth_saving_threshold
        return payload


EDIT_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "name": "photo_director_edit_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "global_delta",
            "localized_adjustments",
            "rationale",
            "confidence",
            "worth_saving",
            "discard_reason",
        ],
        "properties": {
            "global_delta": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SLIDER_LIMITS.keys()),
                "properties": {
                    "temperature": {"type": "number", "minimum": -100, "maximum": 100},
                    "tint": {"type": "number", "minimum": -100, "maximum": 100},
                    "exposure": {"type": "number", "minimum": -3, "maximum": 3},
                    "contrast": {"type": "number", "minimum": -100, "maximum": 100},
                    "highlights": {"type": "number", "minimum": -100, "maximum": 100},
                    "shadows": {"type": "number", "minimum": -100, "maximum": 100},
                    "whites": {"type": "number", "minimum": -100, "maximum": 100},
                    "blacks": {"type": "number", "minimum": -100, "maximum": 100},
                    "saturation": {"type": "number", "minimum": -100, "maximum": 100},
                    "vibrance": {"type": "number", "minimum": -100, "maximum": 100},
                    "clarity": {"type": "number", "minimum": -100, "maximum": 100},
                },
            },
            "localized_adjustments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target", "delta"],
                    "properties": {
                        "target": {"type": "string", "enum": list(SUPPORTED_LOCALIZED_TARGETS)},
                        "delta": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(SLIDER_LIMITS.keys()),
                            "properties": {
                                "temperature": {"type": "number", "minimum": -100, "maximum": 100},
                                "tint": {"type": "number", "minimum": -100, "maximum": 100},
                                "exposure": {"type": "number", "minimum": -3, "maximum": 3},
                                "contrast": {"type": "number", "minimum": -100, "maximum": 100},
                                "highlights": {"type": "number", "minimum": -100, "maximum": 100},
                                "shadows": {"type": "number", "minimum": -100, "maximum": 100},
                                "whites": {"type": "number", "minimum": -100, "maximum": 100},
                                "blacks": {"type": "number", "minimum": -100, "maximum": 100},
                                "saturation": {"type": "number", "minimum": -100, "maximum": 100},
                                "vibrance": {"type": "number", "minimum": -100, "maximum": 100},
                                "clarity": {"type": "number", "minimum": -100, "maximum": 100},
                            },
                        },
                    },
                },
            },
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "worth_saving": {"type": "number", "minimum": 0, "maximum": 1},
            "discard_reason": {"type": "string"},
        },
    },
}
