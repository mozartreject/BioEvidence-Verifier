from __future__ import annotations

import hashlib
import importlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from backend.schemas import (
    EvidenceRecord,
    ExplanationStatus,
    GroundingMetrics,
    ModelArtifact,
    Phase3Result,
    Phase4Result,
    Phase5Result,
    PhaseStatus,
    PhaseTiming,
)


PHASE5_COMPONENT = "phase5_explanation_generator"

DEFAULT_PHASE5_ROOT = Path(
    "~/Downloads/Phase5-main"
).expanduser()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


def _get_phase5_root() -> Path:
    configured = os.environ.get(
        "BIOVERIFY_PHASE5_ROOT"
    )

    if configured:
        root = Path(
            configured
        ).expanduser().resolve()
    else:
        root = (
            DEFAULT_PHASE5_ROOT
            .expanduser()
            .resolve()
        )

    if not root.exists():
        raise FileNotFoundError(
            f"Phase 5 root missing: {root}"
        )

    return root


def _load_phase5_config(
    phase5_root: Path,
) -> Tuple[Dict[str, Any], Path]:
    config_path = (
        phase5_root
        / "config.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Phase 5 config missing: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(
            handle
        )

    if not isinstance(
        cfg,
        dict,
    ):
        raise RuntimeError(
            "Phase 5 config.yaml did not "
            "contain a dictionary."
        )

    if "generation" not in cfg:
        raise RuntimeError(
            "Phase 5 configuration is "
            "missing 'generation'."
        )

    if "grounding" not in cfg:
        raise RuntimeError(
            "Phase 5 configuration is "
            "missing 'grounding'."
        )

    generation = cfg[
        "generation"
    ]

    grounding = cfg[
        "grounding"
    ]

    if not isinstance(
        generation,
        dict,
    ):
        raise RuntimeError(
            "Phase 5 generation config "
            "must be a dictionary."
        )

    if not isinstance(
        grounding,
        dict,
    ):
        raise RuntimeError(
            "Phase 5 grounding config "
            "must be a dictionary."
        )

    expected_generation = {
        "ollama_url":
            "http://localhost:11434/api/chat",
        "num_predict":
            420,
        "request_retries":
            3,
        "repair_attempts":
            2,
        "min_words":
            25,
        "max_words":
            150,
    }

    expected_grounding = {
        "semantic_model":
            "sentence-transformers/all-MiniLM-L6-v2",
        "semantic_threshold":
            0.45,
        "coverage_threshold":
            0.45,
        "fail_on_unsupported_numbers":
            True,
        "fail_on_invalid_citations":
            True,
        "fail_on_verdict_mismatch":
            True,
        "fail_on_suspicious_identifiers":
            True,
    }

    for (
        key,
        expected,
    ) in expected_generation.items():
        if key not in generation:
            raise RuntimeError(
                "Phase 5 generation config "
                f"missing {key!r}."
            )

        actual = generation[
            key
        ]

        if isinstance(
            expected,
            float,
        ):
            if abs(
                float(actual)
                - expected
            ) > 1e-12:
                raise RuntimeError(
                    "Unexpected Phase 5 "
                    f"config {key}: "
                    f"{actual!r}"
                )

        elif actual != expected:
            raise RuntimeError(
                "Unexpected Phase 5 "
                f"config {key}: "
                f"{actual!r}"
            )

    for (
        key,
        expected,
    ) in expected_grounding.items():
        if key not in grounding:
            raise RuntimeError(
                "Phase 5 grounding config "
                f"missing {key!r}."
            )

        actual = grounding[
            key
        ]

        if isinstance(
            expected,
            float,
        ):
            if abs(
                float(actual)
                - expected
            ) > 1e-12:
                raise RuntimeError(
                    "Unexpected Phase 5 "
                    f"config {key}: "
                    f"{actual!r}"
                )

        elif actual != expected:
            raise RuntimeError(
                "Unexpected Phase 5 "
                f"config {key}: "
                f"{actual!r}"
            )

    return (
        cfg,
        config_path,
    )


def _prepare_phase5_imports(
    phase5_root: Path,
):
    root_string = str(
        phase5_root
    )

    if root_string not in sys.path:
        sys.path.insert(
            0,
            root_string,
        )

    importlib.invalidate_caches()

    models_module = (
        importlib.import_module(
            "src.models"
        )
    )

    orchestrator_module = (
        importlib.import_module(
            "src.orchestrator"
        )
    )

    return (
        models_module,
        orchestrator_module,
    )


def _make_phase5_evidence(
    evidence_records: List[
        EvidenceRecord
    ],
    evidence_class: Any,
):
    phase5_evidence = []

    local_to_source: Dict[
        str,
        str,
    ] = {}

    for (
        index,
        evidence,
    ) in enumerate(
        evidence_records,
        1,
    ):
        local_id = (
            f"[E{index}]"
        )

        source_id = str(
            evidence.evidence_id
        ).strip()

        if not source_id:
            raise RuntimeError(
                "Phase 5 received evidence "
                "without an evidence_id."
            )

        text = str(
            evidence.text
            or ""
        ).strip()

        if not text:
            raise RuntimeError(
                f"Evidence {source_id} "
                "has empty text."
            )

        title = str(
            evidence.title
            or ""
        ).strip()

        hierarchy = None

        if (
            evidence.authority_tier
            is not None
        ):
            hierarchy = _enum_value(
                evidence.authority_tier
            )

        hierarchy_weight = (
            float(
                evidence.hierarchy_weight
            )
            if (
                evidence.hierarchy_weight
                is not None
            )
            else None
        )

        retrieval_score = (
            float(
                evidence.relevance_score
            )
            if (
                evidence.relevance_score
                is not None
            )
            else None
        )

        weighted_score = (
            float(
                evidence.combined_score
            )
            if (
                evidence.combined_score
                is not None
            )
            else None
        )

        metadata = {
            "unified_evidence_id":
                source_id,

            "document_id":
                evidence.document_id,

            "bm25_score":
                evidence.bm25_score,

            "crossencoder_raw_score":
                evidence.crossencoder_raw_score,

            "phase2_rank":
                evidence.lineage.phase2_rank,

            "phase3_rank":
                evidence.lineage.phase3_rank,

            "retrieval_query":
                evidence.lineage.retrieval_query,

            "retrieval_hop":
                evidence.lineage.retrieval_hop,
        }

        item = evidence_class(
            local_id=local_id,
            source_id=source_id,
            title=title,
            text=text,
            hierarchy=hierarchy,
            hierarchy_weight=(
                hierarchy_weight
            ),
            retrieval_score=(
                retrieval_score
            ),
            weighted_score=(
                weighted_score
            ),
            source=evidence.source,
            metadata=metadata,
        )

        phase5_evidence.append(
            item
        )

        local_to_source[
            local_id
        ] = source_id

    return (
        phase5_evidence,
        local_to_source,
    )


def _extract_grounding_metrics(
    validation: Dict[
        str,
        Any,
    ],
    semantic_threshold: float,
    coverage_threshold: float,
) -> GroundingMetrics:
    semantic = validation.get(
        "semantic_grounding",
        {},
    )

    semantic_validation = (
        validation.get(
            "semantic_validation",
            {},
        )
    )

    coverage = validation.get(
        "evidence_coverage",
        {},
    )

    semantic_score = (
        semantic_validation.get(
            "validation_mean",
            semantic.get(
                "mean"
            ),
        )
    )

    coverage_score = (
        coverage.get(
            "max"
        )
    )

    semantic_pass = None

    if semantic_score is not None:
        semantic_pass = (
            float(
                semantic_score
            )
            >= semantic_threshold
        )

    coverage_pass = None

    if coverage_score is not None:
        coverage_pass = (
            float(
                coverage_score
            )
            >= coverage_threshold
        )

    invalid_citations = (
        validation.get(
            "invalid_citations",
            [],
        )
    )

    unsupported_numbers = (
        validation.get(
            "unsupported_numbers",
            [],
        )
    )

    suspicious_identifiers = (
        validation.get(
            "suspicious_identifiers",
            [],
        )
    )

    return GroundingMetrics(
        semantic_score=(
            float(
                semantic_score
            )
            if (
                semantic_score
                is not None
            )
            else None
        ),

        coverage_score=(
            float(
                coverage_score
            )
            if (
                coverage_score
                is not None
            )
            else None
        ),

        semantic_pass=(
            semantic_pass
        ),

        coverage_pass=(
            coverage_pass
        ),

        citations_valid=(
            len(
                invalid_citations
            )
            == 0
        ),

        unsupported_numbers_free=(
            len(
                unsupported_numbers
            )
            == 0
        ),

        verdict_match=True,

        suspicious_identifiers_free=(
            len(
                suspicious_identifiers
            )
            == 0
        ),
    )


def _model_artifact(
    cfg: Dict[
        str,
        Any,
    ],
    config_path: Path,
) -> ModelArtifact:
    generation = cfg[
        "generation"
    ]

    grounding = cfg[
        "grounding"
    ]

    return ModelArtifact(
        component=(
            PHASE5_COMPONENT
        ),

        model_name=str(
            generation.get(
                "model",
                "qwen2.5:7b",
            )
        ),

        version=(
            "legacy-phase5-production"
        ),

        metadata={
            "provider":
                generation.get(
                    "provider"
                ),

            "ollama_url":
                generation.get(
                    "ollama_url"
                ),

            "temperature":
                generation.get(
                    "temperature"
                ),

            "top_p":
                generation.get(
                    "top_p"
                ),

            "num_predict":
                generation.get(
                    "num_predict"
                ),

            "timeout_seconds":
                generation.get(
                    "timeout_seconds"
                ),

            "request_retries":
                generation.get(
                    "request_retries"
                ),

            "repair_attempts":
                generation.get(
                    "repair_attempts"
                ),

            "min_words":
                generation.get(
                    "min_words"
                ),

            "max_words":
                generation.get(
                    "max_words"
                ),

            "semantic_model":
                grounding.get(
                    "semantic_model"
                ),

            "semantic_threshold":
                grounding.get(
                    "semantic_threshold"
                ),

            "coverage_threshold":
                grounding.get(
                    "coverage_threshold"
                ),

            "fail_on_unsupported_numbers":
                grounding.get(
                    "fail_on_unsupported_numbers"
                ),

            "fail_on_invalid_citations":
                grounding.get(
                    "fail_on_invalid_citations"
                ),

            "fail_on_verdict_mismatch":
                grounding.get(
                    "fail_on_verdict_mismatch"
                ),

            "fail_on_suspicious_identifiers":
                grounding.get(
                    "fail_on_suspicious_identifiers"
                ),

            "config_path":
                str(
                    config_path
                ),

            "config_sha256":
                _sha256_file(
                    config_path
                ),

            "verdict_source":
                "phase4_locked_verdict",

            "phase5_can_change_verdict":
                False,
        },
    )


class Phase5Adapter:
    def __init__(
        self,
    ):
        self.phase5_root = (
            _get_phase5_root()
        )

        print(
            "Loading Phase 5 production "
            "explanation system..."
        )

        print(
            "Phase 5 root:",
            self.phase5_root,
        )

        (
            self.cfg,
            self.config_path,
        ) = _load_phase5_config(
            self.phase5_root
        )

        (
            self.models_module,
            self.orchestrator_module,
        ) = _prepare_phase5_imports(
            self.phase5_root
        )

        self.Evidence = getattr(
            self.models_module,
            "Evidence",
        )

        self.ClaimRecord = getattr(
            self.models_module,
            "ClaimRecord",
        )

        self.Phase5Orchestrator = getattr(
            self.orchestrator_module,
            "Phase5Orchestrator",
        )

        print(
            "Initializing Phase 5 "
            "orchestrator..."
        )

        self.orchestrator = (
            self.Phase5Orchestrator(
                self.cfg
            )
        )

        generation = self.cfg[
            "generation"
        ]

        grounding = self.cfg[
            "grounding"
        ]

        print(
            "Phase 5 loaded."
        )

        print(
            "Generator:",
            generation.get(
                "model"
            ),
        )

        print(
            "Ollama endpoint:",
            generation.get(
                "ollama_url"
            ),
        )

        print(
            "Repair attempts:",
            generation.get(
                "repair_attempts"
            ),
        )

        print(
            "Semantic threshold:",
            grounding.get(
                "semantic_threshold"
            ),
        )

        print(
            "Coverage threshold:",
            grounding.get(
                "coverage_threshold"
            ),
        )

    def explain(
        self,
        claim: str,
        claim_id: str,
        phase3: Phase3Result,
        phase4: Phase4Result,
    ) -> Phase5Result:
        started = _utc_now()

        start_perf = (
            time.perf_counter()
        )

        claim = str(
            claim
        ).strip()

        claim_id = str(
            claim_id
        ).strip()

        if not claim:
            raise ValueError(
                "Phase 5 claim "
                "cannot be empty."
            )

        if not claim_id:
            raise ValueError(
                "Phase 5 claim_id "
                "cannot be empty."
            )

        if (
            _enum_value(
                phase4.status
            )
            != "SUCCESS"
        ):
            raise RuntimeError(
                "Phase 5 requires "
                "successful Phase 4 output."
            )

        if (
            phase4.verdict_locked
            is not True
        ):
            raise RuntimeError(
                "Phase 4 verdict "
                "is not locked."
            )

        evidence_records = list(
            phase3.ranked_evidence
        )

        if not evidence_records:
            raise RuntimeError(
                "Phase 5 cannot generate "
                "an explanation with "
                "zero evidence."
            )

        phase4_ids = [
            str(
                value
            )
            for value
            in phase4.evidence_used_ids
        ]

        phase3_ids = [
            str(
                item.evidence_id
            )
            for item
            in evidence_records
        ]

        if (
            phase4_ids
            != phase3_ids
        ):
            raise RuntimeError(
                "Phase 3 evidence order "
                "does not match Phase 4 "
                "evidence order."
            )

        (
            phase5_evidence,
            local_to_source,
        ) = _make_phase5_evidence(
            evidence_records,
            self.Evidence,
        )

        verdict = _enum_value(
            phase4.verdict
        )

        record = self.ClaimRecord(
            claim_key=claim_id,
            claim_text=claim,
            verdict=verdict,
            confidence=float(
                phase4.verifier_score
            ),
            dataset=None,
            evidence=(
                phase5_evidence
            ),
            phase4_metadata={
                "verdict":
                    verdict,

                "verdict_locked":
                    True,

                "verifier_score":
                    float(
                        phase4.verifier_score
                    ),

                "probabilities":
                    phase4.probabilities.model_dump(
                        mode="json"
                    ),

                "evidence_used_ids":
                    phase4_ids,

                "aggregation_method":
                    phase4.aggregation_method,

                "model":
                    phase4.model.model_dump(
                        mode="json"
                    ),
            },
        )

        print()
        print(
            "=" * 70
        )

        print(
            "PHASE 5 LOCKED-VERDICT "
            "EXPLANATION"
        )

        print(
            "=" * 70
        )

        print(
            f"Claim ID: {claim_id}"
        )

        print(
            f"Locked verdict: "
            f"{verdict}"
        )

        print(
            "Verifier score:",
            f"{phase4.verifier_score:.6f}",
        )

        print(
            "Evidence supplied:",
            len(
                phase5_evidence
            ),
        )

        for evidence in (
            phase5_evidence
        ):
            print(
                f"  {evidence.local_id} "
                f"-> {evidence.source_id}"
            )

        (
            success,
            failure,
        ) = (
            self.orchestrator
            .generate_one(
                record
            )
        )

        ended = _utc_now()

        latency_ms = (
            time.perf_counter()
            - start_perf
        ) * 1000.0

        timing = PhaseTiming(
            phase="phase5",
            started_at_utc=started,
            ended_at_utc=ended,
            latency_ms=latency_ms,
        )

        model = _model_artifact(
            self.cfg,
            self.config_path,
        )

        if success is None:
            failure = (
                failure
                or {}
            )

            attempts = (
                failure.get(
                    "attempts",
                    [],
                )
            )

            final_errors = (
                failure.get(
                    "final_errors",
                    [],
                )
            )

            fatal_error = (
                failure.get(
                    "fatal_error"
                )
            )

            reasons = []

            if final_errors:
                reasons.extend(
                    str(
                        error
                    )
                    for error
                    in final_errors
                )

            if fatal_error:
                reasons.append(
                    str(
                        fatal_error
                    )
                )

            if not reasons:
                reasons.append(
                    "Phase 5 explanation "
                    "failed production "
                    "validation."
                )

            quarantine_reason = (
                "; ".join(
                    reasons
                )
            )

            print()
            print(
                "Phase 5 explanation: "
                "QUARANTINED"
            )

            print(
                "Reason:",
                quarantine_reason,
            )

            return Phase5Result(
                status=(
                    PhaseStatus.SUCCESS
                ),

                explanation_status=(
                    ExplanationStatus
                    .QUARANTINED
                ),

                explanation=None,

                cited_evidence_ids=[],

                grounding=(
                    GroundingMetrics()
                ),

                repair_attempts_used=(
                    max(
                        0,
                        len(
                            attempts
                        )
                        - 1,
                    )
                ),

                quarantine_reason=(
                    quarantine_reason
                ),

                model=model,

                timing=timing,

                warnings=[
                    "Phase 4 verdict remains "
                    "locked; only the Phase 5 "
                    "explanation was "
                    "quarantined."
                ],

                error=None,
            )

        returned_verdict = str(
            success.get(
                "verdict",
                ""
            )
        ).strip().upper()

        if (
            returned_verdict
            != verdict
        ):
            raise RuntimeError(
                "Phase 5 success payload "
                "changed the locked "
                "Phase 4 verdict."
            )

        explanation = str(
            success.get(
                "explanation",
                ""
            )
        ).strip()

        if not explanation:
            raise RuntimeError(
                "Phase 5 reported success "
                "with an empty explanation."
            )

        validation = success.get(
            "validation"
        )

        if not isinstance(
            validation,
            dict,
        ):
            raise RuntimeError(
                "Phase 5 success payload "
                "is missing validation."
            )

        if not validation.get(
            "passed",
            False,
        ):
            raise RuntimeError(
                "Phase 5 reported success "
                "despite failed validation."
            )

        local_ids = [
            str(
                value
            ).strip()
            for value
            in success.get(
                "evidence_ids_used",
                [],
            )
        ]

        cited_source_ids = []

        for local_id in (
            local_ids
        ):
            if (
                local_id
                not in local_to_source
            ):
                raise RuntimeError(
                    "Phase 5 returned "
                    "unknown citation ID: "
                    f"{local_id}"
                )

            cited_source_ids.append(
                local_to_source[
                    local_id
                ]
            )

        grounding_cfg = (
            self.cfg[
                "grounding"
            ]
        )

        grounding = (
            _extract_grounding_metrics(
                validation=validation,
                semantic_threshold=float(
                    grounding_cfg[
                        "semantic_threshold"
                    ]
                ),
                coverage_threshold=float(
                    grounding_cfg[
                        "coverage_threshold"
                    ]
                ),
            )
        )

        attempts = (
            success.get(
                "generation_metadata",
                {},
            ).get(
                "attempt_count",
                1,
            )
        )

        repair_attempts_used = (
            max(
                0,
                int(
                    attempts
                )
                - 1,
            )
        )

        warnings = [
            str(
                value
            )
            for value
            in validation.get(
                "warnings",
                [],
            )
        ]

        print()
        print(
            "Phase 5 explanation: "
            "ACCEPTED"
        )

        print(
            "Semantic grounding:",
            grounding.semantic_score,
        )

        print(
            "Evidence coverage:",
            grounding.coverage_score,
        )

        print(
            "Repair attempts used:",
            repair_attempts_used,
        )

        print(
            "Original evidence IDs cited:",
            cited_source_ids,
        )

        return Phase5Result(
            status=(
                PhaseStatus.SUCCESS
            ),

            explanation_status=(
                ExplanationStatus.ACCEPTED
            ),

            explanation=explanation,

            cited_evidence_ids=(
                cited_source_ids
            ),

            grounding=grounding,

            repair_attempts_used=(
                repair_attempts_used
            ),

            quarantine_reason=None,

            model=model,

            timing=timing,

            warnings=warnings,

            error=None,
        )


def run_phase5(
    claim: str,
    claim_id: str,
    phase3: Phase3Result,
    phase4: Phase4Result,
) -> Phase5Result:
    adapter = Phase5Adapter()

    return adapter.explain(
        claim=claim,
        claim_id=claim_id,
        phase3=phase3,
        phase4=phase4,
    )