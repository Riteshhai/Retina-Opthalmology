from typing import Literal
from pydantic import BaseModel, Field
from qsart.triage import Action

DISCLAIMER = ('Research prototype. Provisional model outputs are not a diagnosis; '
              'specialist review and recapture are recommendations, not completed care.')


class QualityResult(BaseModel):
    probabilities: list[float]
    predicted_class: int
    poor_probability: float = Field(ge=0, le=1)
    defects: dict[str, list[float]] = Field(default_factory=dict)


class UncertaintyResult(BaseModel):
    entropy: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    method: str = 'normalized_entropy'


class Prediction(BaseModel):
    request_id: str
    image_index: int
    model_version: str
    predicted_grade: int = Field(ge=0, le=4)
    grade_probabilities: list[float] = Field(min_length=5, max_length=5)
    referable_probability: float = Field(ge=0, le=1)
    quality: QualityResult
    uncertainty: UncertaintyResult
    action: Action
    reason_codes: list[str]
    second_pass_activated: bool
    processing_time_ms: float = Field(ge=0)
    policy_scope: Literal['single_image', 'cohort']
    disclaimer: str = DISCLAIMER


class BatchPrediction(BaseModel):
    request_id: str
    results: list[Prediction]
    review_rate: float
    recapture_rate: float
    second_pass_rate: float
    budgets: dict[str, float]
    disclaimer: str = DISCLAIMER
