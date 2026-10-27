from __future__ import annotations

import importlib.util
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.pipeline_config import PipelineConfig, load_pipeline_config
from backend.schemas import (
    AuthorityTier,
    EvidenceRecord,
    Phase3Result,
    PhaseStatus,
    PhaseTiming,
)


DEFAULT_PHASE3_MODEL = "qwen2.5:7b"
DEFAULT_PHASE3_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_TOP_K = 3

RANKING_FORMULA = "relevance_score * hierarchy_weight"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _load_module_from_file(
    module_name: str,
    file_path: Path,
) -> Any:
    """
    Import one Python file under a unique module name.

    Phase 3's production files do not require generic top-level imports
    from one another for the paths used by this adapter, so loading them
    under isolated names prevents collisions with Phase 2 and later
    phases.
    """

    file_path = file_path.resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Phase 3 source file not found: {file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


class Phase3Adapter:
    """
    Adapter for the finalized Phase 3 evidence hierarchy pipeline.

    Input:
        Phase 2 EvidenceRecord objects

    Flow:
        EvidenceRecord
             ↓
        EvidenceClassifier / Qwen 2.5 7B
             ↓
        A / B / C / D
             ↓
        1.0 / 0.7 / 0.4 / 0.2
             ↓
        relevance_score * hierarchy_weight
             ↓
        authority-aware Top-K
             ↓
        Phase3Result

    Scientific constraints:
    -----------------------
    1. Phase 3 does not decide the claim verdict.
    2. Phase 3 does not alter Phase 2 retrieval scores.
    3. Phase 3 preserves all Phase 2 lineage.
    4. The hierarchy classifier uses study design/publication type only.
    5. The actual production classifier uses Ollama /api/chat.
    6. No rule-based fallback is silently introduced.
    7. Classification failures propagate rather than being replaced with
       fabricated tiers.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else load_pipeline_config()
        )

        self.phase_config = self.config.phase3

        self._loaded = False

        self._classifier_module: Any = None
        self._weights_module: Any = None

        self.classifier: Any = None

        self.top_k = int(
            getattr(
                self.phase_config,
                "top_k",
                DEFAULT_TOP_K,
            )
        )

        if self.top_k < 1:
            raise ValueError(
                "Phase 3 top_k must be at least 1"
            )

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        root = Path(
            self.phase_config.root
        ).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"Phase 3 root does not exist: {root}"
            )

        classifier_path = (
            root / "evidence_classifier.py"
        )

        weights_path = (
            root / "hierarchy_weights.py"
        )

        print("Loading Phase 3 evidence classifier...")

        self._classifier_module = (
            _load_module_from_file(
                module_name=(
                    "biomedicalverifier_"
                    "phase3_evidence_classifier"
                ),
                file_path=classifier_path,
            )
        )

        print("Loading Phase 3 hierarchy weights...")

        self._weights_module = (
            _load_module_from_file(
                module_name=(
                    "biomedicalverifier_"
                    "phase3_hierarchy_weights"
                ),
                file_path=weights_path,
            )
        )

        # evidence_classifier.py defines CACHE_PATH as a relative Path.
        # If left untouched, importing it from BiomedicalVerifier would
        # cause it to read/write a new BiomedicalVerifier/data/... cache.
        #
        # Force the classifier to use the actual historical Phase 3 cache.
        authoritative_cache_path = Path(os.environ.get(
            "BIOVERIFY_HIERARCHY_CACHE",
            str(root / "data" / "phase3_output" / "hierarchy_cache.json"),
        )).expanduser().resolve()

        authoritative_cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._classifier_module.CACHE_PATH = (
            authoritative_cache_path
        )

        configured_model = str(
            getattr(
                self.phase_config,
                "ollama_model",
                DEFAULT_PHASE3_MODEL,
            )
        )

        configured_url = str(
            getattr(
                self.phase_config,
                "ollama_url",
                DEFAULT_PHASE3_CHAT_URL,
            )
        )

        # The audited evidence_classifier.py is authoritative and uses
        # /api/chat. Do not silently inherit Phase 3 config.py's stale
        # /api/generate setting.
        if not configured_url.endswith(
            "/api/chat"
        ):
            configured_url = (
                DEFAULT_PHASE3_CHAT_URL
            )

        print(
            "Initializing EvidenceClassifier "
            f"({configured_model})..."
        )

        self.classifier = (
            self._classifier_module.EvidenceClassifier(
                model=configured_model,
                url=configured_url,
            )
        )

        self._loaded = True

        print("Phase 3 loaded.")
        print(
            f"Hierarchy cache: "
            f"{authoritative_cache_path}"
        )
        print(
            f"Ranking formula: {RANKING_FORMULA}"
        )
        print(
            f"Top-K: {self.top_k}"
        )

    @staticmethod
    def _validate_input_evidence(
        evidence: List[EvidenceRecord],
    ) -> None:
        for index, item in enumerate(
            evidence,
            start=1,
        ):
            if item.relevance_score is None:
                raise ValueError(
                    "Phase 3 requires relevance_score "
                    "from Phase 2 for every evidence item. "
                    f"Missing at input evidence #{index} "
                    f"(ID={item.evidence_id})."
                )

            if not item.evidence_id:
                raise ValueError(
                    "Phase 3 received evidence with an empty ID."
                )

            if not item.text.strip():
                raise ValueError(
                    "Phase 3 received evidence with empty text: "
                    f"{item.evidence_id}"
                )

    @staticmethod
    def _classifier_input(
        evidence: EvidenceRecord,
    ) -> Dict[str, Any]:
        """
        Convert the unified EvidenceRecord to the exact legacy classifier
        contract.

        The legacy classifier looks for:
            id
            title
            text
        """

        return {
            "id": evidence.evidence_id,
            "title": evidence.title,
            "text": evidence.text,
        }

    def _classify_one(
        self,
        evidence: EvidenceRecord,
    ) -> Tuple[
        str,
        float,
        str,
        bool,
    ]:
        result = self.classifier.classify(
            self._classifier_input(
                evidence
            )
        )

        if (
            not isinstance(result, tuple)
            or len(result) != 4
        ):
            raise RuntimeError(
                "Phase 3 classifier returned an unexpected "
                f"result for {evidence.evidence_id}: {result!r}"
            )

        tier, weight, source, cached = result

        tier = str(tier).strip().upper()
        weight = float(weight)
        source = str(source)
        cached = bool(cached)

        if tier not in {"A", "B", "C", "D"}:
            raise ValueError(
                "Phase 3 classifier returned invalid tier "
                f"{tier!r} for evidence {evidence.evidence_id}"
            )

        expected_weight = float(
            self._weights_module.HIERARCHY_WEIGHTS[
                tier
            ]
        )

        if abs(
            weight - expected_weight
        ) > 1e-9:
            raise ValueError(
                "Hierarchy classifier weight does not match "
                "canonical hierarchy_weights.py. "
                f"Evidence={evidence.evidence_id}, "
                f"tier={tier}, "
                f"classifier_weight={weight}, "
                f"expected_weight={expected_weight}"
            )

        return (
            tier,
            weight,
            source,
            cached,
        )

    def _build_rankable_item(
        self,
        evidence: EvidenceRecord,
        tier: str,
        weight: float,
        source: str,
        cached: bool,
        input_index: int,
    ) -> Dict[str, Any]:
        """
        Create a dictionary compatible with the finalized
        hierarchy_weights.rank_evidence() implementation.

        The existing Phase 3 ranker expects retrieval relevance in
        evidence["score"], so the unified adapter intentionally supplies
        Phase 2's normalized relevance_score there.

        This preserves the finalized formula:

            relevance_score * hierarchy_weight
        """

        return {
            "id": evidence.evidence_id,
            "score": float(
                evidence.relevance_score
            ),
            "retrieval_score": float(
                evidence.relevance_score
            ),
            "predicted_tier": tier,
            "tier_weight": weight,
            "classification_source": source,
            "classification_cached": cached,
            "_input_index": input_index,
        }

    def _rank(
        self,
        evidence: List[EvidenceRecord],
        classifications: List[
            Tuple[str, float, str, bool]
        ],
    ) -> List[Dict[str, Any]]:
        rankable: List[
            Dict[str, Any]
        ] = []

        for index, (
            item,
            classification,
        ) in enumerate(
            zip(
                evidence,
                classifications,
            )
        ):
            (
                tier,
                weight,
                source,
                cached,
            ) = classification

            rankable.append(
                self._build_rankable_item(
                    evidence=item,
                    tier=tier,
                    weight=weight,
                    source=source,
                    cached=cached,
                    input_index=index,
                )
            )

        ranked = (
            self._weights_module.rank_evidence(
                rankable,
                weighted=True,
                top_k=self.top_k,
            )
        )

        return ranked

    @staticmethod
    def _rebuild_record(
        source_record: EvidenceRecord,
        tier: str,
        hierarchy_weight: float,
        combined_score: float,
        classification_source: str,
        cached: bool,
        phase3_rank: int,
    ) -> EvidenceRecord:
        """
        Preserve the full Phase 2 record and lineage, modifying only the
        fields Phase 3 is scientifically responsible for.
        """

        payload = source_record.model_dump(
            mode="python"
        )

        payload["authority_tier"] = (
            AuthorityTier(tier)
        )

        payload["hierarchy_weight"] = (
            float(hierarchy_weight)
        )

        payload["combined_score"] = (
            float(combined_score)
        )

        payload[
            "hierarchy_classification_source"
        ] = classification_source

        payload[
            "hierarchy_cached"
        ] = cached

        lineage = dict(
            payload["lineage"]
        )

        lineage["phase3_rank"] = (
            phase3_rank
        )

        payload["lineage"] = lineage

        return EvidenceRecord(
            **payload
        )

    def run(
        self,
        evidence: List[EvidenceRecord],
    ) -> Phase3Result:
        if not isinstance(
            evidence,
            list,
        ):
            raise TypeError(
                "evidence must be a list of EvidenceRecord objects"
            )

        if not self._loaded:
            self.load()

        started_at = _utc_now()
        started_perf = time.perf_counter()

        input_count = len(evidence)

        if input_count == 0:
            ended_at = _utc_now()

            return Phase3Result(
                status=PhaseStatus.SUCCESS,
                ranked_evidence=[],
                input_evidence_count=0,
                output_evidence_count=0,
                top_k=self.top_k,
                hierarchy_model=self.classifier.model,
                ranking_formula=RANKING_FORMULA,
                timing=PhaseTiming(
                    phase="phase3",
                    started_at_utc=_iso_utc(
                        started_at
                    ),
                    ended_at_utc=_iso_utc(
                        ended_at
                    ),
                    latency_ms=(
                        time.perf_counter()
                        - started_perf
                    )
                    * 1000.0,
                ),
                warnings=[
                    "Phase 3 received no evidence from Phase 2."
                ],
                error=None,
            )

        for item in evidence:
            if not isinstance(
                item,
                EvidenceRecord,
            ):
                raise TypeError(
                    "Every Phase 3 input item must be "
                    "an EvidenceRecord."
                )

        self._validate_input_evidence(
            evidence
        )

        print()
        print("=" * 70)
        print("PHASE 3 EVIDENCE HIERARCHY")
        print("=" * 70)

        classifications: List[
            Tuple[str, float, str, bool]
        ] = []

        cache_hits = 0
        new_predictions = 0

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            print(
                f"Classifying evidence "
                f"{index}/{input_count}: "
                f"{item.evidence_id}"
            )

            result = self._classify_one(
                item
            )

            (
                tier,
                weight,
                source,
                cached,
            ) = result

            print(
                f"  Tier={tier} "
                f"Weight={weight:.1f} "
                f"Source={source} "
                f"Cached={cached}"
            )

            classifications.append(
                result
            )

            if cached:
                cache_hits += 1
            else:
                new_predictions += 1

        ranked_items = self._rank(
            evidence=evidence,
            classifications=classifications,
        )

        source_by_index = {
            index: record
            for index, record in enumerate(
                evidence
            )
        }

        ranked_records: List[
            EvidenceRecord
        ] = []

        for phase3_rank, ranked in enumerate(
            ranked_items,
            start=1,
        ):
            input_index = int(
                ranked["_input_index"]
            )

            original = source_by_index[
                input_index
            ]

            tier = str(
                ranked["predicted_tier"]
            )

            hierarchy_weight = float(
                ranked["tier_weight"]
            )

            combined_score = float(
                ranked["weighted_score"]
            )

            classification_source = str(
                ranked[
                    "classification_source"
                ]
            )

            cached = bool(
                ranked[
                    "classification_cached"
                ]
            )

            ranked_records.append(
                self._rebuild_record(
                    source_record=original,
                    tier=tier,
                    hierarchy_weight=(
                        hierarchy_weight
                    ),
                    combined_score=(
                        combined_score
                    ),
                    classification_source=(
                        classification_source
                    ),
                    cached=cached,
                    phase3_rank=phase3_rank,
                )
            )

        ended_at = _utc_now()

        warnings: List[str] = []

        if input_count < self.top_k:
            warnings.append(
                "Phase 3 received fewer evidence items than "
                f"configured top_k={self.top_k}."
            )

        print()
        print(
            f"New LLM classifications: "
            f"{new_predictions}"
        )
        print(
            f"Cache hits: {cache_hits}"
        )
        print(
            f"Phase 3 output evidence: "
            f"{len(ranked_records)}"
        )

        return Phase3Result(
            status=PhaseStatus.SUCCESS,
            ranked_evidence=ranked_records,
            input_evidence_count=input_count,
            output_evidence_count=(
                len(ranked_records)
            ),
            top_k=self.top_k,
            hierarchy_model=(
                self.classifier.model
            ),
            ranking_formula=(
                RANKING_FORMULA
            ),
            timing=PhaseTiming(
                phase="phase3",
                started_at_utc=_iso_utc(
                    started_at
                ),
                ended_at_utc=_iso_utc(
                    ended_at
                ),
                latency_ms=(
                    time.perf_counter()
                    - started_perf
                )
                * 1000.0,
            ),
            warnings=warnings,
            error=None,
        )
