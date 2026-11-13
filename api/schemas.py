from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from qsart.triage_v2 import Action

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
    policy_version: str = '2.0'
    conformal: dict[str, Any] | None = None
    model_warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class BatchPrediction(BaseModel):
    request_id: str
    results: list[Prediction]
    review_rate: float
    recapture_rate: float
    second_pass_rate: float
    budgets: dict[str, float]
    policy_version: str = '2.0'
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    ready: bool
    reason: str | None = None
    conformal_status: str = 'not_configured'
    policy_version: str = '2.0'
    disclaimer: str = DISCLAIMER


class ModelInformation(BaseModel):
    model_config = ConfigDict(extra='allow')
    ready: bool
    policy_version: str = '2.0'
    disclaimer: str = DISCLAIMER


class PolicyOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    review_budget: float = Field(.30, ge=0, le=1)
    recapture_budget: float = Field(.20, ge=0, le=1)
    minimum_quality: float = Field(.50, ge=0, le=1)
    review_risk: float = Field(.50, ge=0, le=1)
    uncertainty_threshold: float = Field(.75, ge=0, le=1)
    referable_grade: int = Field(2, ge=1, le=4)
    severe_grade: int = Field(3, ge=1, le=4)
    severe_risk: float = Field(.25, ge=0, le=1)
    severity_costs: list[float] = Field(default_factory=lambda: [0., 0., 3., 6., 10.], min_length=5, max_length=5)
    review_cost: float = Field(.25, ge=0)
    recapture_cost: float = Field(.20, ge=0)
    uncertainty_weight: float = Field(.5, ge=0)
    quality_weight: float = Field(1., ge=0)


class TriageConfiguration(BaseModel):
    policy_version: str = '2.0'
    actions: list[Action]
    defaults: PolicyOptions
    methods: list[str]
    simulation_datasets: list[str]
    compute_profiles: dict[str, float]
    notes: list[str]


class SimulationScore(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    grade_probabilities: list[float] = Field(min_length=5, max_length=5)
    quality_good_probability: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    conformal_set: list[int] | None = None
    true_grade: int | None = Field(None, ge=0, le=4)


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    dataset_id: Literal['development_selection_v2'] | None = None
    scores: list[SimulationScore] | None = Field(None, min_length=1, max_length=512)
    config: PolicyOptions = Field(default_factory=PolicyOptions)
    method: Literal['plain', 'confidence_only', 'entropy_only', 'severity_only', 'quality_only',
                    'single_budget', 'naive', 'dual_budget'] = 'dual_budget'
    compute_profile: Literal['single', 'budget25', 'all'] = 'single'


class SimulationResponse(BaseModel):
    status: Literal['evaluated', 'overloaded', 'unavailable']
    policy_version: str = '2.0'
    dataset_id: str
    images: int = 0
    actions: list[Action] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    compute_profile: str = 'single'
    scope: str = 'development_exploration'
    warnings: list[str] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    experiment_id: str
    dataset: str
    partition: str
    scope: str
    model_version: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ExperimentList(BaseModel):
    experiments: list[ExperimentSummary]


class ExperimentMetrics(BaseModel):
    experiment_id: str
    scope: str
    metrics: dict[str, Any]
    notes: list[str]
