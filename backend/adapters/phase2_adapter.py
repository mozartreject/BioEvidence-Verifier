from __future__ import annotations

import importlib
from backend.services.pubmed_search import PubMedSearch
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.pipeline_config import PipelineConfig, load_pipeline_config
from backend.schemas import (
    EvidenceLineage,
    EvidenceRecord,
    Phase2Result,
    PhaseStatus,
    PhaseTiming,
    RetrievalTrace,
)


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REACT_OBSERVATION_TOP_K = 3

LEGACY_MODULE_NAMES = [
    "config",
    "build_faiss_index",
    "retriever",
    "reranker",
    "react_agent",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _sigmoid(value: float) -> float:
    """
    Monotonic normalization of the raw CrossEncoder score.

    This is NOT a calibrated probability and must not be called
    confidence.
    """
    value = float(value)

    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


class Phase2ImportBundle:
    def __init__(
        self,
        bm25_cls: Any,
        reranker_cls: Any,
        react_agent_cls: Any,
        previous_modules: Dict[str, Any],
        root_string: str,
        path_added: bool,
    ) -> None:
        self.BM25Retriever = bm25_cls
        self.CrossEncoderReranker = reranker_cls
        self.ReActAgent = react_agent_cls

        self.previous_modules = previous_modules
        self.root_string = root_string
        self.path_added = path_added


def _prepare_phase2_modules(
    phase2_root: Path,
    index_dir: Optional[Path] = None,
) -> Phase2ImportBundle:
    """
    Load the legacy Phase 2 project while temporarily exposing its
    top-level module names.

    IMPORTANT:
    BM25Retriever.__init__ performs:

        from build_faiss_index import load_faiss_index

    at runtime.

    Therefore the temporary legacy module aliases MUST remain present
    until BM25Retriever, CrossEncoderReranker and ReActAgent have all
    been instantiated.
    """

    phase2_root = phase2_root.resolve()

    required_files = [
        phase2_root / "config.py",
        phase2_root / "build_faiss_index.py",
        phase2_root / "retriever.py",
        phase2_root / "reranker.py",
        phase2_root / "react_agent.py",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required Phase 2 source files are missing:\n"
            + "\n".join(missing)
        )

    previous_modules: Dict[str, Any] = {
        name: sys.modules.get(name)
        for name in LEGACY_MODULE_NAMES
    }

    root_string = str(phase2_root)
    path_added = False

    if root_string not in sys.path:
        sys.path.insert(0, root_string)
        path_added = True

    for name in LEGACY_MODULE_NAMES:
        sys.modules.pop(name, None)

    importlib.invalidate_caches()

    try:
        legacy_config = importlib.import_module("config")
        if index_dir is not None:
            index_dir = Path(index_dir).resolve()
            for filename in ("biobert_faiss.index", "abstract_metadata.pkl"):
                if not (index_dir / filename).is_file():
                    raise FileNotFoundError(f"Configured Phase 2 index file missing: {index_dir / filename}")
            legacy_config.FAISS_DIR = str(index_dir)
            legacy_config.FAISS_INDEX_PATH = str(index_dir / "biobert_faiss.index")
            legacy_config.FAISS_META_PATH = str(index_dir / "abstract_metadata.pkl")
            print(f"BiomedicalVerifier selected index: {index_dir}")
        importlib.import_module("build_faiss_index")

        retriever_module = importlib.import_module(
            "retriever"
        )

        reranker_module = importlib.import_module(
            "reranker"
        )

        react_agent_module = importlib.import_module(
            "react_agent"
        )

        return Phase2ImportBundle(
            bm25_cls=retriever_module.BM25Retriever,
            reranker_cls=reranker_module.CrossEncoderReranker,
            react_agent_cls=react_agent_module.ReActAgent,
            previous_modules=previous_modules,
            root_string=root_string,
            path_added=path_added,
        )

    except Exception:
        _restore_phase2_modules(
            previous_modules=previous_modules,
            root_string=root_string,
            path_added=path_added,
        )
        raise


def _restore_phase2_modules(
    previous_modules: Dict[str, Any],
    root_string: str,
    path_added: bool,
) -> None:
    """
    Restore Python's module namespace after the legacy Phase 2 objects
    have been fully instantiated.

    Existing class instances continue to function because their module
    globals remain attached to the loaded module objects.
    """

    for name in LEGACY_MODULE_NAMES:
        sys.modules.pop(name, None)

    for name, previous in previous_modules.items():
        if previous is not None:
            sys.modules[name] = previous

    if path_added:
        try:
            sys.path.remove(root_string)
        except ValueError:
            pass


class Phase2Adapter:
    """
    Claim-anchored adapter for the existing Phase 2 retrieval implementation.

    Production architecture:

        claim
          ↓
        BM25 Top-50
          ↓
        CrossEncoder reranking
          ↓
        Top-3 evidence visible per ReAct search hop
          ↓
        Qwen 2.5 7B
          ↓
        structured deduplicated evidence

    Scientific constraints:
    -----------------------
    - BM25 scores and CrossEncoder scores remain separate.
    - Raw CrossEncoder output is never called confidence.
    - relevance_score = sigmoid(raw CrossEncoder score).
    - relevance_score is a bounded ranking transformation, not a
      calibrated probability.
    - Optional live PubMed search supplements local evidence with provenance.
    - Source-prefixed IDs preserve local/live document identity.
    - No evidence is fabricated by this adapter.
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

        self.phase_config = self.config.phase2

        self._loaded = False

        self.retriever: Any = None
        self.reranker: Any = None
        self.agent: Any = None
        self.pubmed = PubMedSearch(self.phase_config.pubmed_cache_dir, limit=self.phase_config.live_pubmed_limit) if self.phase_config.live_pubmed else None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        root = self.phase_config.root

        if not root.exists():
            raise FileNotFoundError(
                f"Phase 2 root does not exist: {root}"
            )

        print("Loading Phase 2 legacy modules...")

        imports = _prepare_phase2_modules(
            root,
            index_dir=self.phase_config.index_dir,
        )

        try:
            print("Initializing BM25 retriever...")

            self.retriever = (
                imports.BM25Retriever()
            )

            from collections import Counter
            source_counts = dict(Counter(doc.get("source", "unknown") for doc in self.retriever.docs))
            print(f"Active retrieval corpus: {len(self.retriever.docs)} documents; sources={source_counts}")

            print(
                "Initializing CrossEncoder reranker..."
            )

            self.reranker = (
                imports.CrossEncoderReranker(
                    model_name=RERANKER_MODEL_NAME
                )
            )

            print("Initializing ReAct agent...")

            self.agent = imports.ReActAgent(
                retriever=self.retriever,
            )

        finally:
            _restore_phase2_modules(
                previous_modules=(
                    imports.previous_modules
                ),
                root_string=imports.root_string,
                path_added=imports.path_added,
            )

        self._loaded = True

        print("Phase 2 loaded.")

    def _retrieve_for_query(
        self,
        query: str,
        hop: int,
    ) -> Tuple[
        List[Dict[str, Any]],
        RetrievalTrace,
    ]:
        """
        Execute one Phase 2 local evidence search.

        1. BM25 generates Top-50 candidates.
        2. CrossEncoder reranks candidates.
        3. Top configured reranked set is materialized.
        4. All configured candidates are retained; observations show three.

        Score lineage is retained explicitly.
        """

        candidates = self.retriever.search(
            query,
            k=self.phase_config.bm25_candidate_k,
        )

        if not candidates:
            trace = RetrievalTrace(
                query=query,
                hop=hop,
                candidate_count=0,
                reranked_count=0,
            )

            return [], trace

        bm25_rank_by_id: Dict[str, int] = {}

        for rank, doc in enumerate(
            candidates,
            start=1,
        ):
            doc_id = str(doc["id"])
            bm25_rank_by_id[doc_id] = rank

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=self.phase_config.crossencoder_top_k,
        )

        enriched: List[Dict[str, Any]] = []

        for ce_rank, doc in enumerate(
            reranked,
            start=1,
        ):
            item = dict(doc)

            document_id = str(
                item["id"]
            )

            raw_score = float(
                item["reranker_score"]
            )

            item["_bm25_rank"] = (
                bm25_rank_by_id.get(
                    document_id
                )
            )

            item[
                "_crossencoder_rank"
            ] = ce_rank

            item[
                "_crossencoder_raw_score"
            ] = raw_score

            item[
                "_relevance_score"
            ] = _sigmoid(
                raw_score
            )

            item[
                "_retrieval_query"
            ] = query

            item[
                "_retrieval_hop"
            ] = hop

            enriched.append(item)

        trace = RetrievalTrace(
            query=query,
            hop=hop,
            candidate_count=len(candidates),
            reranked_count=len(reranked),
        )

        return (
            enriched,
            trace,
        )

    @staticmethod
    def _format_observation(
        results: List[Dict[str, Any]],
    ) -> str:
        if not results:
            return "No results found."

        lines: List[str] = []

        for index, result in enumerate(
            results[:REACT_OBSERVATION_TOP_K],
            start=1,
        ):
            source = str(
                result.get(
                    "source",
                    "unknown",
                )
            )

            title = str(
                result.get(
                    "title",
                    "",
                )
            )

            text = str(
                result.get(
                    "text",
                    "",
                )
            )

            lines.append(
                f"Result {index} "
                f"[{source}] "
                f"(ID: {result['id']}): "
                f"{title}"
            )

            lines.append(
                f"  {text[:250]}..."
            )

        return "\n".join(lines)

    @staticmethod
    def _deduplicate(
        collected: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate evidence by document ID.

        If an evidence document appears through multiple ReAct queries,
        retain the occurrence having the strongest CrossEncoder-derived
        relevance score.
        """

        best_by_id: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for item in collected:
            evidence_id = str(
                item["id"]
            )

            existing = best_by_id.get(
                evidence_id
            )

            if existing is None:
                best_by_id[
                    evidence_id
                ] = item
                continue

            new_score = float(
                item[
                    "_relevance_score"
                ]
            )

            old_score = float(
                existing[
                    "_relevance_score"
                ]
            )

            if new_score > old_score:
                best_by_id[
                    evidence_id
                ] = item

        unique = list(
            best_by_id.values()
        )

        unique.sort(
            key=lambda item: (
                float(
                    item[
                        "_relevance_score"
                    ]
                ),
                float(
                    item[
                        "_crossencoder_raw_score"
                    ]
                ),
            ),
            reverse=True,
        )

        return unique

    def _rerank_for_claim(self, claim, collected):
        """Compare every unique candidate against the SAME original claim.

        Query-specific scores are unsuitable for cross-query sorting. Preserve
        retrieval query/rank lineage, but replace final relevance with a single
        claim-conditioned reranking pass. Do not infer truth from relevance.
        """
        unique = self._deduplicate(collected)
        if not unique:
            return []
        ranked = self.reranker.rerank(claim, unique, top_k=len(unique))
        result = []
        for item in ranked:
            item = dict(item)
            raw = float(item["reranker_score"])
            if not math.isfinite(raw):
                raise ValueError("Claim reranker returned a non-finite score")
            item["_crossencoder_raw_score"] = raw
            item["_relevance_score"] = _sigmoid(raw)
            result.append(item)
        return result

    @staticmethod
    def _to_evidence_records(
        evidence: List[Dict[str, Any]],
    ) -> List[EvidenceRecord]:
        records: List[
            EvidenceRecord
        ] = []

        for phase2_rank, item in enumerate(
            evidence,
            start=1,
        ):
            evidence_id = str(
                item["id"]
            )

            record = EvidenceRecord(
                evidence_id=evidence_id,
                document_id=evidence_id,
                title=str(
                    item.get(
                        "title",
                        "",
                    )
                ),
                text=str(
                    item.get(
                        "text",
                        "",
                    )
                ),
                source=str(
                    item.get(
                        "source",
                        "unknown",
                    )
                ),
                source_url=item.get("source_url") or ("https://pubmed.ncbi.nlm.nih.gov/" + evidence_id.split(":", 1)[1] + "/" if evidence_id.startswith("pubmed:") and evidence_id.split(":", 1)[1].isdigit() else None),
                publication_date=item.get("publication_date"),
                retrieved_at_utc=item.get("retrieved_at_utc"),
                retrieval_origin=item.get("retrieval_origin", "local_index"),
                bm25_score=float(item["score"]) if item.get("score") is not None else None,
                crossencoder_raw_score=float(
                    item[
                        "_crossencoder_raw_score"
                    ]
                ),
                relevance_score=float(
                    item[
                        "_relevance_score"
                    ]
                ),
                lineage=EvidenceLineage(
                    bm25_rank=item.get(
                        "_bm25_rank"
                    ),
                    crossencoder_rank=item.get(
                        "_crossencoder_rank"
                    ),
                    phase2_rank=phase2_rank,
                    retrieval_query=item.get(
                        "_retrieval_query"
                    ),
                    retrieval_hop=item.get(
                        "_retrieval_hop"
                    ),
                    used_by_phase4=False,
                    cited_by_phase5=False,
                ),
            )

            records.append(record)

        return records

    def run(
        self,
        claim: str,
        claim_id: str,
    ) -> Phase2Result:
        if not isinstance(
            claim,
            str,
        ):
            raise TypeError(
                "claim must be a string"
            )

        claim = claim.strip()

        if not claim:
            raise ValueError(
                "claim cannot be empty"
            )

        if not isinstance(
            claim_id,
            str,
        ):
            raise TypeError(
                "claim_id must be a string"
            )

        claim_id = claim_id.strip()

        if not claim_id:
            raise ValueError(
                "claim_id cannot be empty"
            )

        if not self._loaded:
            self.load()

        started = _utc_now()
        start_perf = time.perf_counter()

        scratchpad: List[str] = []

        collected: List[
            Dict[str, Any]
        ] = []

        queries_executed: List[str] = []

        traces: List[
            RetrievalTrace
        ] = []

        # Always retrieve the actual claim before allowing query rewriting.
        # Hop 0 is the deterministic original-claim search; agent hops are 1..N.
        initial, initial_trace = self._retrieve_for_query(claim, hop=0)
        collected.extend(initial)
        queries_executed.append(claim)
        traces.append(initial_trace)
        scratchpad.append("Original-claim search:\n" + self._format_observation(initial))
        seen_queries = {" ".join(claim.casefold().split())}
        hop_count = 0

        stop_reason: Optional[str] = None

        warnings: List[str] = []

        warnings.append(f"Retrieval corpus: {self.phase_config.index_dir}; live_pubmed={self.phase_config.live_pubmed}.")
        if self.pubmed is not None:
            try:
                online, online_trace = self.pubmed.search(claim)
                query = online_trace["query"]
                # Keep live abstract provenance when a PMID already exists locally.
                online_ids = {str(item["id"]) for item in online}
                collected = [item for item in collected if str(item["id"]) not in online_ids]
                for item in online:
                    item.update(_relevance_score=0.0, _crossencoder_raw_score=0.0,
                                _retrieval_query=query, _retrieval_hop=0,
                                _bm25_rank=None, _crossencoder_rank=None)
                collected.extend(online)
                queries_executed.append("PubMed: " + query)
                traces.append(RetrievalTrace(query="PubMed: " + query, hop=0, candidate_count=len(online), reranked_count=0))
                warnings.append(f"PubMed search: {len(online)} abstracts; cache_hit={online_trace['cache_hit']}.")
                if not online:
                    warnings.append("PubMed returned no usable abstracts; continuing with local evidence.")
            except Exception as exc:
                warnings.append(f"Live PubMed unavailable ({type(exc).__name__}); local-only fallback used.")
        max_hops = self.phase_config.max_react_hops

        prompt = (
            self.agent.base_prompt.format(
                claim=claim,
                max_hops=max_hops,
            )
        )

        conversation = (
            prompt + "\n\n"
        )

        print()
        print("=" * 70)
        print(
            f"PHASE 2 CLAIM ID: "
            f"{claim_id}"
        )
        print(
            f"CLAIM: {claim[:120]}"
        )
        print("=" * 70)

        while hop_count < max_hops:
            current_hop = (
                hop_count + 1
            )

            print()
            print(
                f"--- ReAct Hop "
                f"{current_hop}/"
                f"{max_hops} ---"
            )

            scratchpad_text = (
                "\n\n".join(
                    scratchpad
                )
                if scratchpad
                else "No previous actions."
            )

            full_prompt = (
                conversation
                + "\nHere is what has happened "
                + "so far:\n"
                + scratchpad_text
                + "\n\nContinue from here."
            )

            response = (
                self.agent.llm.generate(
                    full_prompt,
                    max_tokens=400,
                    temperature=0.1,
                )
            )

            parsed = (
                self.agent.parse_action(
                    response
                )
            )

            thought = str(
                parsed.get(
                    "thought",
                    "",
                )
            ).strip()

            tool = str(
                parsed.get(
                    "tool",
                    "stop",
                )
            ).strip()

            argument = str(
                parsed.get(
                    "argument",
                    "",
                )
            ).strip()

            print(
                f"Action: {tool}"
            )

            if argument:
                print(
                    f"Argument: "
                    f"{argument[:120]}"
                )

            scratchpad.append(
                "Thought: "
                + thought
                + "\nAction: "
                + f"{tool}('{argument}')"
            )

            if tool == "stop":
                stop_reason = (
                    argument
                    if argument
                    else (
                        "Agent requested stop."
                    )
                )
                break

            if tool != "search_pubmed":
                stop_reason = (
                    "Unsupported ReAct tool: "
                    + tool
                )

                warnings.append(
                    stop_reason
                )

                break

            if not argument:
                stop_reason = (
                    "ReAct produced an empty "
                    "retrieval query."
                )

                warnings.append(
                    stop_reason
                )

                break

            normalized_query = " ".join(argument.casefold().split())
            if normalized_query in seen_queries:
                stop_reason = "Repeated retrieval query; no new evidence search."
                break
            seen_queries.add(normalized_query)
            queries_executed.append(argument)

            results, trace = (
                self._retrieve_for_query(
                    query=argument,
                    hop=current_hop,
                )
            )

            traces.append(
                trace
            )

            observation = (
                self._format_observation(
                    results
                )
            )

            scratchpad.append(
                "Observation: "
                + observation
            )

            collected.extend(
                results
            )

            hop_count += 1

            time.sleep(0.2)

        if (
            hop_count >= max_hops
            and stop_reason is None
        ):
            stop_reason = (
                "Maximum ReAct hop limit "
                "reached."
            )

        # Prefer the source-attributed live copy of a duplicated PMID, even if
        # later agent searches also returned its local corpus version.
        live_ids = {str(item["id"]) for item in collected if item.get("retrieval_origin") == "live_pubmed"}
        collected = [item for item in collected if str(item["id"]) not in live_ids or item.get("retrieval_origin") == "live_pubmed"]
        unique_evidence = self._rerank_for_claim(claim, collected)
        # Same-title revisions must not occupy all hierarchy slots. Keep the
        # newest dated record; retain rank order among the representatives.
        representatives = {}
        for item in unique_evidence:
            title = " ".join(str(item.get("title", "")).lower().split()).rstrip(".")
            key = title if len(title) >= 40 else str(item["id"])
            previous = representatives.get(key)
            if previous is None or str(item.get("publication_date") or "") > str(previous.get("publication_date") or ""):
                representatives[key] = item
        kept = {str(item["id"]) for item in representatives.values()}
        collapsed = len(unique_evidence) - len(kept)
        unique_evidence = [item for item in unique_evidence if str(item["id"]) in kept]
        if collapsed:
            warnings.append(f"Collapsed {collapsed} same-title records to reduce repeated-review weighting; these may be revisions of the same study.")
        candidate_count = len(unique_evidence)
        unique_evidence = unique_evidence[:max(1, self.phase_config.final_candidate_k)]
        warnings.append(f"Claim-reranked {candidate_count} unique candidates; retained {len(unique_evidence)} for hierarchy ranking.")
        warnings.append(
            "Final relevance is reranked against the original claim; "
            "crossencoder_rank in lineage records the originating search rank."
        )

        evidence_records = (
            self._to_evidence_records(
                unique_evidence
            )
        )

        ended = _utc_now()

        latency_ms = (
            time.perf_counter()
            - start_perf
        ) * 1000.0

        timing = PhaseTiming(
            phase="phase2",
            started_at_utc=(
                _iso_utc(started)
            ),
            ended_at_utc=(
                _iso_utc(ended)
            ),
            latency_ms=latency_ms,
        )

        return Phase2Result(
            status=PhaseStatus.SUCCESS,
            evidence=evidence_records,
            queries_executed=(
                queries_executed
            ),
            retrieval_trace=traces,
            hop_count=hop_count,
            stop_reason=stop_reason,
            evidence_found=bool(
                evidence_records
            ),
            retriever_name="BM25Okapi",
            reranker_name=(
                RERANKER_MODEL_NAME
            ),
            agent_model="qwen2.5:7b",
            timing=timing,
            warnings=warnings,
            error=None,
        )