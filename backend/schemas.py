from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Verdict(str, Enum):
    REFUTED = "REFUTED"
    NEI = "NEI"
    SUPPORTED = "SUPPORTED"


class UserFacingVerdict(str, Enum):
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFORMATION = "NOT_ENOUGH_INFORMATION"
    SUPPORTED = "SUPPORTED"


class PhaseName(str, Enum):
    PHASE1 = "phase1"
    PHASE2 = "phase2"
    PHASE3 = "phase3"
    PHASE4 = "phase4"
    PHASE5 = "phase5"


class PhaseStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"
    SKIPPED = "SKIPPED"


class ExplanationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


class AuthorityTier(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Modality(str, Enum):
    TEXT_ONLY = "TEXT_ONLY"
    TEXT_IMAGE = "TEXT_IMAGE"


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class InferenceRequest(BaseSchema):
    request_id: str
    claim: str = Field(min_length=3, max_length=5000)
    image_path: Optional[str] = None
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not value:
            raise ValueError("request_id cannot be empty")
        return value

    @property
    def modality(self) -> Modality:
        return Modality.TEXT_IMAGE if self.image_path else Modality.TEXT_ONLY


class PhaseError(BaseSchema):
    phase: PhaseName
    error_type: str
    message: str
    recoverable: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class PhaseTiming(BaseSchema):
    phase: PhaseName
    started_at_utc: datetime
    ended_at_utc: datetime
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_time_order(self) -> "PhaseTiming":
        if self.ended_at_utc < self.started_at_utc:
            raise ValueError("ended_at_utc cannot be before started_at_utc")
        return self


class ModelArtifact(BaseSchema):
    component: str
    model_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Phase1Result(BaseSchema):
    status: PhaseStatus
    has_image: bool
    modality: Modality

    text_embedding_dim: Optional[int] = None
    image_embedding_dim: Optional[int] = None
    fused_embedding_dim: Optional[int] = None

    # Embeddings are optional in serialized API output. The adapter may keep
    # tensors internally and only export vectors when diagnostics request them.
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    fused_embedding: Optional[List[float]] = None

    auxiliary_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auxiliary_prediction: Optional[int] = Field(default=None, ge=0, le=1)
    auxiliary_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auxiliary_label: Optional[str] = None

    model: Optional[ModelArtifact] = None
    timing: Optional[PhaseTiming] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[PhaseError] = None

    @model_validator(mode="after")
    def validate_image_consistency(self) -> "Phase1Result":
        if self.modality == Modality.TEXT_ONLY and self.has_image:
            raise ValueError("TEXT_ONLY modality cannot have has_image=True")
        if self.modality == Modality.TEXT_IMAGE and not self.has_image:
            raise ValueError("TEXT_IMAGE modality requires has_image=True")
        return self


class EvidenceLineage(BaseSchema):
    bm25_rank: Optional[int] = Field(default=None, ge=1)
    crossencoder_rank: Optional[int] = Field(default=None, ge=1)
    phase2_rank: Optional[int] = Field(default=None, ge=1)
    phase3_rank: Optional[int] = Field(default=None, ge=1)

    retrieval_query: Optional[str] = None
    retrieval_hop: Optional[int] = Field(default=None, ge=0)

    used_by_phase4: bool = False
    cited_by_phase5: bool = False


class EvidenceRecord(BaseSchema):
    evidence_id: str
    document_id: Optional[str] = None
    title: str = ""
    text: str
    source: str = ""

    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    retrieved_at_utc: Optional[str] = None
    retrieval_origin: Optional[str] = None

    bm25_score: Optional[float] = None
    crossencoder_raw_score: Optional[float] = None
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    authority_tier: Optional[AuthorityTier] = None
    hierarchy_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    combined_score: Optional[float] = None

    hierarchy_classification_source: Optional[str] = None
    hierarchy_cached: Optional[bool] = None

    lineage: EvidenceLineage = Field(default_factory=EvidenceLineage)

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not value:
            raise ValueError("evidence_id cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_hierarchy_fields(self) -> "EvidenceRecord":
        hierarchy_fields = (
            self.authority_tier,
            self.hierarchy_weight,
            self.combined_score,
        )
        any_present = any(value is not None for value in hierarchy_fields)
        all_present = all(value is not None for value in hierarchy_fields)

        if any_present and not all_present:
            raise ValueError(
                "authority_tier, hierarchy_weight and combined_score "
                "must be populated together"
            )
        return self


class RetrievalTrace(BaseSchema):
    query: str
    hop: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)


class Phase2Result(BaseSchema):
    status: PhaseStatus
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    queries_executed: List[str] = Field(default_factory=list)
    retrieval_trace: List[RetrievalTrace] = Field(default_factory=list)
    hop_count: int = Field(default=0, ge=0)
    stop_reason: Optional[str] = None
    evidence_found: bool = False

    retriever_name: str = "BM25"
    reranker_name: Optional[str] = None
    agent_model: Optional[str] = None

    timing: Optional[PhaseTiming] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[PhaseError] = None

    @model_validator(mode="after")
    def validate_evidence_found(self) -> "Phase2Result":
        if self.evidence_found and not self.evidence:
            raise ValueError("evidence_found=True but evidence list is empty")
        return self


class Phase3Result(BaseSchema):
    status: PhaseStatus
    ranked_evidence: List[EvidenceRecord] = Field(default_factory=list)
    input_evidence_count: int = Field(ge=0)
    output_evidence_count: int = Field(ge=0)
    top_k: int = Field(ge=1)

    hierarchy_model: str
    ranking_formula: str = "relevance_score * hierarchy_weight"

    timing: Optional[PhaseTiming] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[PhaseError] = None

    @model_validator(mode="after")
    def validate_counts(self) -> "Phase3Result":
        if self.output_evidence_count != len(self.ranked_evidence):
            raise ValueError(
                "output_evidence_count must match len(ranked_evidence)"
            )
        if self.output_evidence_count > self.input_evidence_count:
            raise ValueError(
                "Phase 3 cannot output more evidence than it received"
            )
        return self


class ClassProbabilities(BaseSchema):
    refuted: float = Field(ge=0.0, le=1.0)
    nei: float = Field(ge=0.0, le=1.0)
    supported: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probability_sum(self) -> "ClassProbabilities":
        total = self.refuted + self.nei + self.supported
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Class probabilities must sum to 1.0; received {total:.6f}"
            )
        return self

    def for_verdict(self, verdict: Verdict) -> float:
        if verdict == Verdict.REFUTED:
            return self.refuted
        if verdict == Verdict.NEI:
            return self.nei
        return self.supported


class PairwiseVerification(BaseSchema):
    scored_text: Optional[str] = None
    evidence_id: str
    probabilities: ClassProbabilities
    predicted_label: Verdict
    raw_logits: Optional[List[float]] = None


class Phase4Result(BaseSchema):
    status: PhaseStatus
    verdict: Verdict
    user_facing_verdict: UserFacingVerdict
    probabilities: ClassProbabilities
    verifier_score: float = Field(ge=0.0, le=1.0)

    pairwise_predictions: List[PairwiseVerification] = Field(default_factory=list)
    evidence_used_ids: List[str] = Field(default_factory=list)

    aggregation_method: str
    label_mapping: Dict[int, Verdict]
    model: ModelArtifact

    verdict_locked: Literal[True] = True

    timing: Optional[PhaseTiming] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[PhaseError] = None

    @model_validator(mode="after")
    def validate_phase4_result(self) -> "Phase4Result":
        expected_user_verdict = {
            Verdict.REFUTED: UserFacingVerdict.REFUTED,
            Verdict.NEI: UserFacingVerdict.NOT_ENOUGH_INFORMATION,
            Verdict.SUPPORTED: UserFacingVerdict.SUPPORTED,
        }[self.verdict]

        if self.user_facing_verdict != expected_user_verdict:
            raise ValueError("user_facing_verdict is inconsistent with verdict")

        expected_score = self.probabilities.for_verdict(self.verdict)
        if abs(expected_score - self.verifier_score) > 1e-6:
            raise ValueError(
                "verifier_score must equal the probability of the selected verdict"
            )
        return self


class GroundingMetrics(BaseSchema):
    semantic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    coverage_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    semantic_pass: Optional[bool] = None
    coverage_pass: Optional[bool] = None
    citations_valid: Optional[bool] = None
    unsupported_numbers_free: Optional[bool] = None
    verdict_match: Optional[bool] = None
    suspicious_identifiers_free: Optional[bool] = None


class Phase5Result(BaseSchema):
    status: PhaseStatus
    explanation_status: ExplanationStatus

    explanation: Optional[str] = None
    cited_evidence_ids: List[str] = Field(default_factory=list)
    grounding: GroundingMetrics = Field(default_factory=GroundingMetrics)

    repair_attempts_used: int = Field(default=0, ge=0)
    quarantine_reason: Optional[str] = None

    model: Optional[ModelArtifact] = None
    timing: Optional[PhaseTiming] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[PhaseError] = None

    @model_validator(mode="after")
    def validate_explanation_state(self) -> "Phase5Result":
        if self.explanation_status == ExplanationStatus.ACCEPTED and not self.explanation:
            raise ValueError(
                "ACCEPTED explanation status requires non-empty explanation"
            )
        if self.explanation_status == ExplanationStatus.QUARANTINED:
            if not self.quarantine_reason:
                raise ValueError(
                    "QUARANTINED explanation requires quarantine_reason"
                )
        return self


class PipelineProvenance(BaseSchema):
    pipeline_version: str
    mode: Literal["faithful", "enhanced"] = "faithful"
    models: List[ModelArtifact] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class DecisionPackage(BaseSchema):
    request: InferenceRequest

    phase1: Phase1Result
    phase2: Phase2Result
    phase3: Phase3Result
    phase4: Phase4Result
    phase5: Phase5Result

    final_verdict: Verdict
    final_user_facing_verdict: UserFacingVerdict
    final_verifier_score: float = Field(ge=0.0, le=1.0)

    evidence: List[EvidenceRecord] = Field(default_factory=list)

    provenance: PipelineProvenance
    total_latency_ms: float = Field(ge=0.0)

    completed_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def enforce_locked_verdict(self) -> "DecisionPackage":
        if self.final_verdict != self.phase4.verdict:
            raise ValueError(
                "Final verdict must exactly equal the locked Phase 4 verdict"
            )
        if self.final_user_facing_verdict != self.phase4.user_facing_verdict:
            raise ValueError(
                "Final user-facing verdict must equal Phase 4 user-facing verdict"
            )
        if abs(self.final_verifier_score - self.phase4.verifier_score) > 1e-6:
            raise ValueError(
                "Final verifier score must equal the Phase 4 verifier score"
            )
        return self


def verdict_to_user_facing(verdict: Verdict) -> UserFacingVerdict:
    return {
        Verdict.REFUTED: UserFacingVerdict.REFUTED,
        Verdict.NEI: UserFacingVerdict.NOT_ENOUGH_INFORMATION,
        Verdict.SUPPORTED: UserFacingVerdict.SUPPORTED,
    }[verdict]
