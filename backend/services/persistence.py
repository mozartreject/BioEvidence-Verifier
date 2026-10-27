from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.pipeline_config import RuntimeConfig
from backend.schemas import DecisionPackage


class PersistenceError(RuntimeError):
    pass


class DuplicateRequestError(PersistenceError):
    pass


class AuditPhaseRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    status: str
    latency_ms: Optional[float] = None
    warnings: List[str] = Field(
        default_factory=list
    )


class AuditRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: str = "1.0"

    request_id: str
    claim_sha256: str

    created_at_utc: datetime
    completed_at_utc: datetime

    pipeline_version: str
    pipeline_mode: str

    final_verdict: str
    final_user_facing_verdict: str
    final_verifier_score: float

    explanation_status: str

    phase_statuses: Dict[
        str,
        AuditPhaseRecord,
    ]

    evidence_ids: List[str]
    cited_evidence_ids: List[str]

    model_artifacts: List[
        Dict[str, Any]
    ]

    total_latency_ms: float

    invariant_checks: Dict[
        str,
        bool,
    ]

    decision_package_sha256: str

    persisted_at_utc: datetime


class PersistenceResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str

    inference_path: str
    audit_path: Optional[str]

    decision_package_sha256: str

    inference_bytes: int
    audit_bytes: Optional[int]

    persisted_at_utc: datetime


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)

    return str(value)


def _sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def _sha256_text(
    text: str,
) -> str:
    return _sha256_bytes(
        text.encode("utf-8")
    )


def _safe_request_id(
    request_id: str,
) -> str:
    value = request_id.strip()

    if not value:
        raise PersistenceError(
            "request_id cannot be empty."
        )

    if len(value) > 200:
        raise PersistenceError(
            "request_id exceeds the "
            "200-character persistence limit."
        )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        value,
    ):
        raise PersistenceError(
            "request_id contains unsafe "
            "filesystem characters. "
            "Allowed characters are letters, "
            "numbers, '.', '_' and '-'."
        )

    return value


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
        allow_nan=False,
    )

    return (
        text + "\n"
    ).encode("utf-8")


def _fsync_directory(
    directory: Path,
) -> None:
    try:
        fd = os.open(
            str(directory),
            os.O_RDONLY,
        )
    except OSError:
        return

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_new(
    path: Path,
    payload: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}."
        f"{os.getpid()}.tmp"
    )

    if path.exists():
        raise DuplicateRequestError(
            f"Refusing to overwrite existing "
            f"research artifact: {path}"
        )

    try:
        with temporary_path.open(
            "xb"
        ) as handle:
            handle.write(
                payload
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        if path.exists():
            raise DuplicateRequestError(
                f"Refusing to overwrite existing "
                f"research artifact: {path}"
            )

        os.replace(
            temporary_path,
            path,
        )

        _fsync_directory(
            path.parent
        )

    except Exception:
        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass

        raise


def _phase_record(
    result: Any,
) -> AuditPhaseRecord:
    timing = getattr(
        result,
        "timing",
        None,
    )

    latency_ms = None

    if timing is not None:
        latency_ms = float(
            timing.latency_ms
        )

    warnings = list(
        getattr(
            result,
            "warnings",
            [],
        )
    )

    return AuditPhaseRecord(
        status=_enum_value(
            result.status
        ),
        latency_ms=latency_ms,
        warnings=warnings,
    )


def _extract_checkpoint_sha256(
    model: Any,
) -> Optional[str]:
    metadata = getattr(
        model,
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        return None

    value = metadata.get(
        "checkpoint_sha256"
    )

    if value is None:
        value = metadata.get(
            "config_sha256"
        )

    if value is None:
        return None

    return str(value)


def _model_artifact_summary(
    package: DecisionPackage,
) -> List[Dict[str, Any]]:
    output: List[
        Dict[str, Any]
    ] = []

    for model in (
        package.provenance.models
    ):
        output.append(
            {
                "component":
                    model.component,

                "model_name":
                    model.model_name,

                "checkpoint_path":
                    model.checkpoint_path,

                "checkpoint_epoch":
                    model.checkpoint_epoch,

                "version":
                    model.version,

                "artifact_sha256":
                    _extract_checkpoint_sha256(
                        model
                    ),
            }
        )

    return output


def _build_invariant_checks(
    package: DecisionPackage,
) -> Dict[str, bool]:
    phase2_ids = {
        str(
            evidence.evidence_id
        )
        for evidence
        in package.phase2.evidence
    }

    phase3_ids = [
        str(
            evidence.evidence_id
        )
        for evidence
        in package.phase3.ranked_evidence
    ]

    phase4_ids = [
        str(value)
        for value
        in package.phase4.evidence_used_ids
    ]

    package_ids = [
        str(
            evidence.evidence_id
        )
        for evidence
        in package.evidence
    ]

    cited_ids = [
        str(value)
        for value
        in package.phase5.cited_evidence_ids
    ]

    probability_sum = (
        float(
            package
            .phase4
            .probabilities
            .refuted
        )
        + float(
            package
            .phase4
            .probabilities
            .nei
        )
        + float(
            package
            .phase4
            .probabilities
            .supported
        )
    )

    selected_probability = (
        package
        .phase4
        .probabilities
        .for_verdict(
            package.phase4.verdict
        )
    )

    explanation_status = (
        _enum_value(
            package
            .phase5
            .explanation_status
        )
    )

    citation_lineage_valid = True

    if explanation_status == "ACCEPTED":
        citation_lineage_valid = set(
            cited_ids
        ).issubset(
            set(
                phase4_ids
            )
        )

    quarantined_output_hidden = True

    if explanation_status == "QUARANTINED":
        quarantined_output_hidden = (
            package
            .phase5
            .explanation
            is None
        )

    checks = {
        "all_phases_success":
            all(
                _enum_value(
                    result.status
                )
                == "SUCCESS"
                for result
                in (
                    package.phase1,
                    package.phase2,
                    package.phase3,
                    package.phase4,
                    package.phase5,
                )
            ),

        "phase3_evidence_originates_phase2":
            set(
                phase3_ids
            ).issubset(
                phase2_ids
            ),

        "phase3_phase4_lineage_match":
            phase3_ids
            == phase4_ids,

        "package_phase4_lineage_match":
            package_ids
            == phase4_ids,

        "phase4_verdict_locked":
            package
            .phase4
            .verdict_locked
            is True,

        "final_verdict_matches_phase4":
            package.final_verdict
            == package.phase4.verdict,

        "final_user_verdict_matches_phase4":
            (
                package
                .final_user_facing_verdict
                == package
                .phase4
                .user_facing_verdict
            ),

        "final_score_matches_phase4":
            abs(
                float(
                    package
                    .final_verifier_score
                )
                - float(
                    package
                    .phase4
                    .verifier_score
                )
            )
            <= 1e-6,

        "phase4_probabilities_sum_to_one":
            abs(
                probability_sum
                - 1.0
            )
            <= 1e-6,

        "phase4_selected_probability_matches_score":
            abs(
                float(
                    selected_probability
                )
                - float(
                    package
                    .phase4
                    .verifier_score
                )
            )
            <= 1e-6,

        "phase5_citation_lineage_valid":
            citation_lineage_valid,

        "quarantined_explanation_hidden":
            quarantined_output_hidden,

        "recognized_deployment_mode":
            (
                package
                .provenance
                .mode
                in {"faithful", "enhanced"}
            ),
    }

    return checks


class DecisionPackagePersistence:
    """
    Immutable local persistence layer for
    BiomedicalVerifier research inference.

    Full DecisionPackages and compact audit records
    are stored separately.

    Existing request IDs are never silently
    overwritten.
    """

    def __init__(
        self,
        runtime_config: RuntimeConfig,
    ):
        self.runtime = (
            runtime_config
        )

        self.inference_dir = (
            self.runtime.outputs_dir
            / "inference"
        )

        self.audit_dir = (
            self.runtime.audit_dir
        )

        self.inference_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.audit_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _paths(
        self,
        request_id: str,
    ) -> tuple[
        Path,
        Path,
    ]:
        safe_id = _safe_request_id(
            request_id
        )

        inference_path = (
            self.inference_dir
            / f"{safe_id}.json"
        )

        audit_path = (
            self.audit_dir
            / f"{safe_id}.json"
        )

        return (
            inference_path,
            audit_path,
        )

    def persist(
        self,
        package: DecisionPackage,
    ) -> PersistenceResult:
        request_id = (
            _safe_request_id(
                package
                .request
                .request_id
            )
        )

        (
            inference_path,
            audit_path,
        ) = self._paths(
            request_id
        )

        if inference_path.exists():
            raise DuplicateRequestError(
                "Inference artifact already "
                "exists for request_id="
                f"{request_id}: "
                f"{inference_path}"
            )

        if (
            self.runtime.save_audit_json
            and audit_path.exists()
        ):
            raise DuplicateRequestError(
                "Audit artifact already "
                "exists for request_id="
                f"{request_id}: "
                f"{audit_path}"
            )

        package_object = (
            package.model_dump(
                mode="json"
            )
        )

        package_bytes = (
            _canonical_json_bytes(
                package_object
            )
        )

        package_sha256 = (
            _sha256_bytes(
                package_bytes
            )
        )

        invariant_checks = (
            _build_invariant_checks(
                package
            )
        )

        failed_invariants = [
            name
            for name, passed
            in invariant_checks.items()
            if not passed
        ]

        if failed_invariants:
            raise PersistenceError(
                "DecisionPackage failed "
                "persistence invariant checks: "
                + ", ".join(
                    failed_invariants
                )
            )

        persisted_at = (
            datetime.now(
                timezone.utc
            )
        )

        audit_record = AuditRecord(
            request_id=request_id,

            claim_sha256=(
                _sha256_text(
                    package.request.claim
                )
            ),

            created_at_utc=(
                package
                .request
                .created_at_utc
            ),

            completed_at_utc=(
                package
                .completed_at_utc
            ),

            pipeline_version=(
                package
                .provenance
                .pipeline_version
            ),

            pipeline_mode=(
                package
                .provenance
                .mode
            ),

            final_verdict=(
                _enum_value(
                    package.final_verdict
                )
            ),

            final_user_facing_verdict=(
                _enum_value(
                    package
                    .final_user_facing_verdict
                )
            ),

            final_verifier_score=float(
                package
                .final_verifier_score
            ),

            explanation_status=(
                _enum_value(
                    package
                    .phase5
                    .explanation_status
                )
            ),

            phase_statuses={
                "phase1":
                    _phase_record(
                        package.phase1
                    ),

                "phase2":
                    _phase_record(
                        package.phase2
                    ),

                "phase3":
                    _phase_record(
                        package.phase3
                    ),

                "phase4":
                    _phase_record(
                        package.phase4
                    ),

                "phase5":
                    _phase_record(
                        package.phase5
                    ),
            },

            evidence_ids=[
                str(
                    evidence.evidence_id
                )
                for evidence
                in package.evidence
            ],

            cited_evidence_ids=[
                str(value)
                for value
                in package
                .phase5
                .cited_evidence_ids
            ],

            model_artifacts=(
                _model_artifact_summary(
                    package
                )
            ),

            total_latency_ms=float(
                package
                .total_latency_ms
            ),

            invariant_checks=(
                invariant_checks
            ),

            decision_package_sha256=(
                package_sha256
            ),

            persisted_at_utc=(
                persisted_at
            ),
        )

        audit_bytes = (
            _canonical_json_bytes(
                audit_record.model_dump(
                    mode="json"
                )
            )
        )

        inference_written = False

        try:
            _atomic_write_new(
                inference_path,
                package_bytes,
            )

            inference_written = True

            if (
                self.runtime
                .save_audit_json
            ):
                _atomic_write_new(
                    audit_path,
                    audit_bytes,
                )

        except Exception:
            if inference_written:
                try:
                    inference_path.unlink()
                except OSError:
                    pass

            raise

        return PersistenceResult(
            request_id=request_id,

            inference_path=str(
                inference_path
            ),

            audit_path=(
                str(
                    audit_path
                )
                if (
                    self.runtime
                    .save_audit_json
                )
                else None
            ),

            decision_package_sha256=(
                package_sha256
            ),

            inference_bytes=len(
                package_bytes
            ),

            audit_bytes=(
                len(
                    audit_bytes
                )
                if (
                    self.runtime
                    .save_audit_json
                )
                else None
            ),

            persisted_at_utc=(
                persisted_at
            ),
        )
