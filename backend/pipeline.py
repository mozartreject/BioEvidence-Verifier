from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from backend.schemas import (
    DecisionPackage,
    InferenceRequest,
    ModelArtifact,
    PipelineProvenance,
)

from backend.pipeline_config import (
    PipelineConfig,
    ensure_runtime_directories,
    load_pipeline_config,
    validate_static_paths,
)

from backend.adapters.phase1_adapter import Phase1Adapter
from backend.adapters.phase2_adapter import Phase2Adapter
from backend.adapters.phase3_adapter import Phase3Adapter
from backend.adapters.phase4_adapter import Phase4Adapter
from backend.adapters.biomedical_nli_adapter import BiomedicalNliAdapter
from backend.adapters.phase5_adapter import Phase5Adapter

from backend.services.persistence import (
    DecisionPackagePersistence,
    PersistenceResult,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _enum_value(value):
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _assert_success(
    phase_name: str,
    result,
) -> None:
    status = _enum_value(result.status)

    if status != "SUCCESS":
        error = getattr(
            result,
            "error",
            None,
        )

        raise RuntimeError(
            f"{phase_name} failed with "
            f"status={status}; "
            f"error={error}"
        )


def _collect_models(
    phase1,
    phase4,
    phase5,
) -> List[ModelArtifact]:
    models: List[ModelArtifact] = []

    for result in (
        phase1,
        phase4,
        phase5,
    ):
        model = getattr(
            result,
            "model",
            None,
        )

        if model is not None:
            models.append(model)

    return models


class BiomedicalVerificationPipeline:
    """
    Faithful end-to-end biomedical misinformation
    verification pipeline.

    Scientific execution graph:

        Input
          |
          +---- Phase 1
          |     multimodal diagnostic branch
          |
          +---- Phase 2
                evidence retrieval
                   |
                   v
                Phase 3
                authority hierarchy
                   |
                   v
                Phase 4
                evidence-grounded verifier
                   |
                   v
                Phase 5
                locked-verdict explanation
                   |
                   v
              DecisionPackage
                   |
                   v
             Persistence
             + Audit Record

    Important scientific constraint:

    Phase 1 is executed and preserved for multimodal
    analysis and provenance, but the selected Phase 4
    production verifier was not trained to consume the
    Phase 1 fused representation.

    Therefore Phase 1 does NOT modify the Phase 4
    verdict in faithful deployment mode.

    Phase 4 is the authoritative verdict source.
    Phase 5 cannot modify that verdict.

    A successful inference is persisted only after the
    complete DecisionPackage and all final invariants
    have been validated.
    """

    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        phase1: Optional[Phase1Adapter] = None,
        phase2: Optional[Phase2Adapter] = None,
        phase3: Optional[Phase3Adapter] = None,
        phase4: Optional[Phase4Adapter] = None,
        phase5: Optional[Phase5Adapter] = None,
    ):
        self.config = (
            config
            if config is not None
            else load_pipeline_config()
        )

        ensure_runtime_directories(
            self.config
        )

        path_failures = validate_static_paths(
            self.config
        )

        if path_failures:
            message = "\n".join(
                f"  - {failure}"
                for failure in path_failures
            )

            raise FileNotFoundError(
                "BiomedicalVerifier static "
                "configuration validation failed:\n"
                f"{message}"
            )

        self.phase1 = (
            phase1
            if phase1 is not None
            else Phase1Adapter(
                self.config.phase1,
                self.config.runtime,
            )
        )

        self.phase2 = (
            phase2
            if phase2 is not None
            else Phase2Adapter(self.config)
        )

        self.phase3 = (
            phase3
            if phase3 is not None
            else Phase3Adapter(self.config)
        )

        if self.config.phase4.backend not in {"legacy", "biomedical_nli"}:
            raise ValueError("BIOVERIFY_VERIFIER must be legacy or biomedical_nli")

        self.phase4 = (
            phase4
            if phase4 is not None
            else (BiomedicalNliAdapter(self.config) if self.config.phase4.backend == "biomedical_nli" else Phase4Adapter(self.config))
        )

        self.phase5 = (
            phase5
            if phase5 is not None
            else Phase5Adapter()
        )

        self.persistence = (
            DecisionPackagePersistence(
                self.config.runtime
            )
        )

        self.last_persistence_result: Optional[
            PersistenceResult
        ] = None

        self._loaded = False

        self.load()

    def load(self) -> None:
        """
        Load all adapters that expose an explicit load()
        lifecycle.

        Loading is performed once per pipeline instance.
        The scientific behavior of the individual phase
        adapters is not modified here.
        """

        if self._loaded:
            return

        print()
        print("=" * 80)
        print("BIOMEDICAL VERIFIER — LOADING PIPELINE")
        print("=" * 80)

        adapters = (
            ("Phase 1", self.phase1),
            ("Phase 2", self.phase2),
            ("Phase 3", self.phase3),
            ("Phase 4", self.phase4),
            ("Phase 5", self.phase5),
        )

        for phase_name, adapter in adapters:
            load_method = getattr(
                adapter,
                "load",
                None,
            )

            if not callable(load_method):
                print(
                    f"{phase_name}: "
                    "no explicit load() lifecycle; "
                    "using constructor initialization."
                )
                continue

            print(
                f"{phase_name}: loading..."
            )

            try:
                load_method()
            except Exception as exc:
                raise RuntimeError(
                    f"{phase_name} adapter failed "
                    f"during pipeline loading: "
                    f"{exc}"
                ) from exc

            print(
                f"{phase_name}: loaded."
            )

        self._loaded = True

        print()
        print(
            "Pipeline adapter lifecycle: READY"
        )
        print("=" * 80)

    def run(
        self,
        request: InferenceRequest,
    ) -> DecisionPackage:
        if not self._loaded:
            raise RuntimeError(
                "BiomedicalVerificationPipeline "
                "is not loaded."
            )

        self.last_persistence_result = None

        started = time.perf_counter()

        print()
        print("=" * 80)
        print("BIOMEDICAL VERIFIER")
        print("END-TO-END FAITHFUL INFERENCE")
        print("=" * 80)

        print(
            "Pipeline version:",
            self.config.runtime.pipeline_version,
        )

        print(
            "Request ID:",
            request.request_id,
        )

        print(
            "Claim:",
            request.claim,
        )

        print(
            "Image:",
            (
                request.image_path
                if request.image_path
                else "not supplied"
            ),
        )

        # =================================================
        # PHASE 1
        # =================================================

        print()
        print("=" * 80)
        print(
            "PHASE 1 — MULTIMODAL DIAGNOSTIC"
        )
        print("=" * 80)

        phase1 = self.phase1.infer(
            request
        )

        _assert_success(
            "Phase 1",
            phase1,
        )

        # =================================================
        # PHASE 2
        # =================================================

        print()
        print("=" * 80)
        print(
            "PHASE 2 — EVIDENCE RETRIEVAL"
        )
        print("=" * 80)

        phase2 = self.phase2.run(
            claim=request.claim,
            claim_id=request.request_id,
        )

        _assert_success(
            "Phase 2",
            phase2,
        )

        if not phase2.evidence:
            raise RuntimeError(
                "Phase 2 returned SUCCESS "
                "without evidence."
            )

        # =================================================
        # PHASE 3
        # =================================================

        print()
        print("=" * 80)
        print(
            "PHASE 3 — EVIDENCE HIERARCHY"
        )
        print("=" * 80)

        phase3 = self.phase3.run(
            phase2.evidence
        )

        _assert_success(
            "Phase 3",
            phase3,
        )

        if not phase3.ranked_evidence:
            raise RuntimeError(
                "Phase 3 returned SUCCESS "
                "without ranked evidence."
            )

        # =================================================
        # PHASE 2 -> PHASE 3 CONTRACT
        # =================================================

        phase2_ids = {
            str(evidence.evidence_id)
            for evidence in phase2.evidence
        }

        phase3_ids = [
            str(evidence.evidence_id)
            for evidence
            in phase3.ranked_evidence
        ]

        if not set(
            phase3_ids
        ).issubset(
            phase2_ids
        ):
            raise RuntimeError(
                "Phase 3 introduced evidence "
                "that did not originate in "
                "Phase 2."
            )

        # =================================================
        # PHASE 4
        # =================================================

        print()
        print("=" * 80)
        print(
            "PHASE 4 — VERIFICATION"
        )
        print("=" * 80)

        phase4 = self.phase4.run(
            claim=request.claim,
            evidence_list=(
                phase3.ranked_evidence
            ),
        )

        _assert_success(
            "Phase 4",
            phase4,
        )

        if phase4.verdict_locked is not True:
            raise RuntimeError(
                "Phase 4 returned an unlocked "
                "verdict."
            )

        phase4_ids = [
            str(evidence_id)
            for evidence_id
            in phase4.evidence_used_ids
        ]

        if phase3_ids != phase4_ids:
            raise RuntimeError(
                "Phase 3 -> Phase 4 evidence "
                "lineage mismatch. "
                f"Phase3={phase3_ids}; "
                f"Phase4={phase4_ids}"
            )

        # =================================================
        # PHASE 4 PROBABILITY CONTRACT
        # =================================================

        probabilities = {
            "REFUTED": float(
                phase4.probabilities.refuted
            ),
            "NEI": float(
                phase4.probabilities.nei
            ),
            "SUPPORTED": float(
                phase4.probabilities.supported
            ),
        }

        probability_sum = sum(
            probabilities.values()
        )

        if (
            abs(
                probability_sum
                - 1.0
            )
            > 1e-6
        ):
            raise RuntimeError(
                "Phase 4 probabilities do not "
                "sum to 1.0. "
                f"Observed={probability_sum}"
            )

        verdict_name = _enum_value(
            phase4.verdict
        )

        if verdict_name not in probabilities:
            raise RuntimeError(
                "Unknown Phase 4 verdict: "
                f"{verdict_name}"
            )

        selected_probability = (
            probabilities[
                verdict_name
            ]
        )

        if (
            abs(
                selected_probability
                - float(
                    phase4.verifier_score
                )
            )
            > 1e-6
        ):
            raise RuntimeError(
                "Phase 4 verifier_score does "
                "not equal the selected class "
                "probability."
            )

        # =================================================
        # PHASE 5
        # =================================================

        print()
        print("=" * 80)
        print(
            "PHASE 5 — GROUNDED EXPLANATION"
        )
        print("=" * 80)

        locked_verdict_before = (
            _enum_value(
                phase4.verdict
            )
        )

        locked_score_before = float(
            phase4.verifier_score
        )

        locked_ids_before = list(
            phase4_ids
        )

        phase5 = self.phase5.explain(
            claim=request.claim,
            claim_id=request.request_id,
            phase3=phase3,
            phase4=phase4,
        )

        _assert_success(
            "Phase 5",
            phase5,
        )

        # =================================================
        # PHASE 5 CANNOT MUTATE PHASE 4
        # =================================================

        locked_verdict_after = (
            _enum_value(
                phase4.verdict
            )
        )

        locked_score_after = float(
            phase4.verifier_score
        )

        locked_ids_after = [
            str(value)
            for value
            in phase4.evidence_used_ids
        ]

        if (
            locked_verdict_before
            != locked_verdict_after
        ):
            raise RuntimeError(
                "Phase 5 modified the locked "
                "Phase 4 verdict."
            )

        if (
            abs(
                locked_score_before
                - locked_score_after
            )
            > 1e-12
        ):
            raise RuntimeError(
                "Phase 5 modified the Phase 4 "
                "verifier score."
            )

        if (
            locked_ids_before
            != locked_ids_after
        ):
            raise RuntimeError(
                "Phase 5 modified Phase 4 "
                "evidence lineage."
            )

        if phase4.verdict_locked is not True:
            raise RuntimeError(
                "Phase 4 verdict lock was lost "
                "during Phase 5."
            )

        # =================================================
        # PHASE 5 CITATION LINEAGE
        # =================================================

        explanation_status = (
            _enum_value(
                phase5.explanation_status
            )
        )

        if (
            explanation_status
            == "ACCEPTED"
        ):
            if not phase5.explanation:
                raise RuntimeError(
                    "Phase 5 marked an empty "
                    "explanation as ACCEPTED."
                )

            invalid_cited_ids = (
                set(
                    str(value)
                    for value
                    in phase5.cited_evidence_ids
                )
                - set(
                    phase4_ids
                )
            )

            if invalid_cited_ids:
                raise RuntimeError(
                    "Phase 5 cited evidence "
                    "outside the Phase 4 evidence "
                    "set: "
                    f"{sorted(invalid_cited_ids)}"
                )

        elif (
            explanation_status
            == "QUARANTINED"
        ):
            if phase5.explanation is not None:
                raise RuntimeError(
                    "Quarantined Phase 5 output "
                    "must not expose an "
                    "explanation."
                )

            if not phase5.quarantine_reason:
                raise RuntimeError(
                    "Quarantined Phase 5 output "
                    "has no quarantine reason."
                )

        else:
            raise RuntimeError(
                "Unexpected Phase 5 explanation "
                "status: "
                f"{explanation_status}"
            )

        # =================================================
        # AUTHORITATIVE FINAL DECISION
        # =================================================

        final_verdict = (
            phase4.verdict
        )

        final_user_facing_verdict = (
            phase4.user_facing_verdict
        )

        final_verifier_score = float(
            phase4.verifier_score
        )

        # =================================================
        # MODEL PROVENANCE
        # =================================================

        # Export the actual verifier/explanation lineage for source inspection.
        used_ids = set(phase4.evidence_used_ids)
        cited_ids = set(phase5.cited_evidence_ids)
        for evidence in phase3.ranked_evidence:
            evidence.lineage.used_by_phase4 = evidence.evidence_id in used_ids
            evidence.lineage.cited_by_phase5 = evidence.evidence_id in cited_ids

        models = _collect_models(
            phase1=phase1,
            phase4=phase4,
            phase5=phase5,
        )

        provenance = (
            PipelineProvenance(
                pipeline_version=(
                    self.config
                    .runtime
                    .pipeline_version
                ),
                mode=("enhanced" if self.config.phase4.backend == "biomedical_nli" else "faithful"),
                models=models,
                notes=[
                    (
                        "Final verdict is the "
                        "locked Phase 4 verdict."
                    ),
                    (
                        "Phase 5 is an "
                        "evidence-grounded "
                        "explanation component and "
                        "cannot change the Phase 4 "
                        "verdict."
                    ),
                    (
                        "Phase 1 is executed as a "
                        "multimodal diagnostic "
                        "component. Its output does "
                        "not alter the Phase 4 "
                        "verdict in either deployment mode."
                    ),
                    (
                        "Phase 2 uses claim-anchored retrieval. "
                        f"Index: {self.config.phase2.index_dir}; "
                        f"live PubMed enabled: {self.config.phase2.live_pubmed}. "
                        "See evidence source URLs and Phase 2 warnings for retrieval provenance."
                    ),
                    (
                        "Phase 3 reranks retrieved "
                        "evidence using authority "
                        "hierarchy weighting."
                    ),
                    (
                        "Phase 4 performs the "
                        "authoritative three-class "
                        "REFUTED/NEI/SUPPORTED "
                        "verification."
                    ),
                ],
            )
        )

        completed_at = _utc_now()

        total_latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # =================================================
        # FINAL DECISION PACKAGE
        # =================================================

        package = DecisionPackage(
            request=request,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            phase5=phase5,
            final_verdict=(
                final_verdict
            ),
            final_user_facing_verdict=(
                final_user_facing_verdict
            ),
            final_verifier_score=(
                final_verifier_score
            ),
            evidence=(
                phase3.ranked_evidence
            ),
            provenance=provenance,
            total_latency_ms=(
                total_latency_ms
            ),
            completed_at_utc=(
                completed_at
            ),
        )

        # =================================================
        # FINAL PACKAGE INVARIANTS
        # =================================================

        if (
            _enum_value(
                package.final_verdict
            )
            != _enum_value(
                package.phase4.verdict
            )
        ):
            raise RuntimeError(
                "DecisionPackage final verdict "
                "does not match Phase 4."
            )

        if (
            package
            .final_user_facing_verdict
            != package
            .phase4
            .user_facing_verdict
        ):
            raise RuntimeError(
                "DecisionPackage user-facing "
                "verdict does not match "
                "Phase 4."
            )

        if (
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
            > 1e-6
        ):
            raise RuntimeError(
                "DecisionPackage verifier score "
                "does not match Phase 4."
            )

        package_ids = [
            str(evidence.evidence_id)
            for evidence
            in package.evidence
        ]

        if package_ids != phase4_ids:
            raise RuntimeError(
                "DecisionPackage evidence "
                "lineage does not match "
                "Phase 4."
            )

        if (
            package.provenance.mode
            not in {"faithful", "enhanced"}
        ):
            raise RuntimeError(
                "DecisionPackage is not marked "
                "as faithful deployment."
            )

        # =================================================
        # IMMUTABLE PERSISTENCE / AUDIT COMMIT
        # =================================================

        print()
        print("=" * 80)
        print(
            "PERSISTENCE — RESEARCH RECORD COMMIT"
        )
        print("=" * 80)

        try:
            persistence_result = (
                self.persistence.persist(
                    package
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "DecisionPackage passed scientific "
                "validation but persistence failed. "
                "The inference is NOT considered a "
                "successfully committed research "
                f"record. Cause: {exc}"
            ) from exc

        self.last_persistence_result = (
            persistence_result
        )

        # =================================================
        # FINAL OUTPUT
        # =================================================

        print()
        print("=" * 80)
        print(
            "FINAL DECISION"
        )
        print("=" * 80)

        print(
            "Verdict:",
            _enum_value(
                package.final_verdict
            ),
        )

        print(
            "User-facing verdict:",
            _enum_value(
                package
                .final_user_facing_verdict
            ),
        )

        print(
            "Verifier score:",
            package.final_verifier_score,
        )

        print(
            "Explanation status:",
            explanation_status,
        )

        print(
            "Evidence count:",
            len(
                package.evidence
            ),
        )

        print(
            "Total inference latency:",
            f"{package.total_latency_ms:.2f} ms",
        )

        print()
        print(
            "DecisionPackage validation: PASS"
        )

        print(
            "Persistence commit: PASS"
        )

        print(
            "Inference artifact:",
            persistence_result.inference_path,
        )

        print(
            "Audit artifact:",
            persistence_result.audit_path,
        )

        print(
            "DecisionPackage SHA256:",
            persistence_result.decision_package_sha256,
        )

        print()
        print("=" * 80)
        print(
            "P1 -> P2 -> P3 -> P4 -> P5 "
            "-> DECISION PACKAGE "
            "-> PERSISTENCE: PASS"
        )
        print("=" * 80)

        return package