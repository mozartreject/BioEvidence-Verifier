from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    HTTPException,
)

from backend.api.dependencies import (
    get_pipeline,
)


router = APIRouter(
    prefix="/system",
    tags=["system"],
)


def _utc_now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


@router.get(
    "/health",
    summary="BiomedicalVerifier health",
)
def health() -> Dict[str, Any]:
    try:
        pipeline = get_pipeline()

        loaded = bool(
            getattr(
                pipeline,
                "_loaded",
                False,
            )
        )

        if not loaded:
            raise RuntimeError(
                "Pipeline exists but is "
                "not loaded."
            )

        return {
            "status": "healthy",
            "pipeline_ready": True,
            "verifier_backend": pipeline.config.phase4.backend,
            "retrieval": {
                "index_dir": str(pipeline.config.phase2.index_dir),
                "document_count": len(pipeline.phase2.retriever.docs),
                "live_pubmed_enabled": pipeline.config.phase2.live_pubmed,
                "final_candidate_limit": pipeline.config.phase2.final_candidate_k,
            },
            "pipeline_version": (
                pipeline
                .config
                .runtime
                .pipeline_version
            ),
            "mode": "enhanced" if pipeline.config.phase4.backend == "biomedical_nli" else "faithful",
            "timestamp_utc": (
                _utc_now_iso()
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error":
                    "PIPELINE_NOT_READY",

                "message":
                    str(exc),
            },
        ) from exc


@router.get(
    "/models",
    summary="Frozen model registry",
)
def models() -> Dict[str, Any]:
    try:
        pipeline = get_pipeline()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error":
                    "PIPELINE_NOT_READY",

                "message":
                    str(exc),
            },
        ) from exc

    registry: List[
        Dict[str, Any]
    ] = []

    phase1 = getattr(
        pipeline.phase1,
        "model_artifact",
        None,
    )

    if phase1 is not None:
        if hasattr(
            phase1,
            "model_dump",
        ):
            registry.append(
                phase1.model_dump(
                    mode="json"
                )
            )

    phase4 = getattr(
        pipeline.phase4,
        "model_artifact",
        None,
    )

    if phase4 is not None:
        if hasattr(
            phase4,
            "model_dump",
        ):
            registry.append(
                phase4.model_dump(
                    mode="json"
                )
            )

    return {
        "pipeline_version": (
            pipeline
            .config
            .runtime
            .pipeline_version
        ),

        "mode": "enhanced" if pipeline.config.phase4.backend == "biomedical_nli" else "faithful",

        "scientific_components": {
            "phase1": {
                "role":
                    "multimodal diagnostic "
                    "representation",

                "affects_final_verdict":
                    False,
            },

            "phase2": {
                "role":
                    "BM25 + CrossEncoder + "
                    "ReAct evidence retrieval",

                "retriever":
                    "BM25Okapi",

                "reranker":
                    "cross-encoder/"
                    "ms-marco-MiniLM-L-6-v2",

                "agent_model":
                    pipeline
                    .config
                    .phase3
                    .ollama_model,
            },

            "phase3": {
                "role":
                    "authority hierarchy "
                    "classification and reranking",

                "model":
                    pipeline
                    .config
                    .phase3
                    .ollama_model,

                "ranking_formula":
                    "relevance_score * "
                    "hierarchy_weight",
            },

            "phase4": {
                "role":
                    "authoritative three-class "
                    "verification",

                "model": pipeline.phase4.model_artifact.model_name,
                "checkpoint": pipeline.phase4.model_artifact.checkpoint_path,
                "backend": pipeline.config.phase4.backend,

                "affects_final_verdict":
                    True,
            },

            "phase5": {
                "role":
                    "locked-verdict grounded "
                    "explanation",

                "model":
                    pipeline
                    .config
                    .phase5
                    .ollama_model,

                "can_change_verdict":
                    False,
            },
        },

        "loaded_model_artifacts":
            registry,
    }
