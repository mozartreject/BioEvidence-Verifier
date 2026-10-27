from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import (
    ClaimRecord,
    Evidence,
    normalize_verdict,
)

from .utils import (
    first,
    normalize_text,
    claim_hash,
)


def load_jsonish(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        output = []

        for i, line in enumerate(
            path.read_text(
                encoding="utf-8"
            ).splitlines(),
            1,
        ):
            if not line.strip():
                continue

            try:
                output.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL "
                    f"{path}:{i}: {exc}"
                ) from exc

        return output

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_predictions(
    path: Path,
    claims_path: Path | None = None,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            rows = [
                dict(row)
                for row in csv.DictReader(
                    handle
                )
            ]

    else:
        obj = load_jsonish(path)

        if isinstance(obj, list):
            rows = obj

        elif isinstance(obj, dict):
            rows = next(
                (
                    obj[key]
                    for key in (
                        "predictions",
                        "items",
                        "results",
                    )
                    if isinstance(
                        obj.get(key),
                        list,
                    )
                ),
                None,
            )

            if rows is None:
                raise ValueError(
                    "Unsupported Phase 4 "
                    "prediction JSON structure"
                )

        else:
            raise ValueError(
                "Unsupported Phase 4 "
                "prediction structure"
            )

    lookup = {}

    if (
        claims_path
        and claims_path.exists()
    ):
        claims = load_jsonish(
            claims_path
        )

        if not isinstance(
            claims,
            list,
        ):
            raise ValueError(
                "claims.jsonl must resolve "
                "to a list"
            )

        for row in claims:
            claim = normalize_text(
                first(
                    row,
                    [
                        "claim_text",
                        "claim",
                        "text",
                    ],
                )
            )

            if not claim:
                continue

            candidates = [
                first(
                    row,
                    ["claim_id"],
                ),
                first(
                    row,
                    ["sample_id"],
                ),
                first(
                    row,
                    ["claim_hash"],
                ),
                claim_hash(claim),
                claim.lower(),
            ]

            for key in candidates:
                if key not in (
                    None,
                    "",
                ):
                    lookup[
                        str(key)
                    ] = row

    output = []

    for raw_original in rows:
        raw = dict(
            raw_original
        )

        claim = normalize_text(
            first(
                raw,
                [
                    "claim_text",
                    "claim",
                    "text",
                ],
            )
        )

        if not claim:
            candidates = [
                first(
                    raw,
                    ["claim_id"],
                ),
                first(
                    raw,
                    ["sample_id"],
                ),
                first(
                    raw,
                    ["claim_hash"],
                ),
            ]

            for key in candidates:
                if (
                    key not in (
                        None,
                        "",
                    )
                    and str(key) in lookup
                ):
                    source = lookup[
                        str(key)
                    ]

                    claim = normalize_text(
                        first(
                            source,
                            [
                                "claim_text",
                                "claim",
                                "text",
                            ],
                        )
                    )

                    for source_key, value in (
                        source.items()
                    ):
                        raw.setdefault(
                            source_key,
                            value,
                        )

                    break

        verdict = normalize_verdict(
            first(
                raw,
                [
                    "verdict",
                    "pred_verdict",
                    "prediction",
                    "pred",
                    "predicted_label",
                    "label_pred",
                ],
            )
        )

        confidence = first(
            raw,
            [
                "confidence",
                "pred_confidence",
                "max_probability",
                "probability",
            ],
        )

        if confidence in (
            None,
            "",
        ):
            probabilities = []

            for key in (
                "prob_refuted",
                "prob_nei",
                "prob_supported",
                "p_refuted",
                "p_nei",
                "p_supported",
            ):
                try:
                    value = raw.get(
                        key
                    )

                    if value not in (
                        None,
                        "",
                    ):
                        probabilities.append(
                            float(value)
                        )

                except Exception:
                    pass

            confidence = (
                max(probabilities)
                if probabilities
                else None
            )

        else:
            confidence = float(
                confidence
            )

        key = str(
            first(
                raw,
                [
                    "claim_id",
                    "sample_id",
                    "claim_hash",
                ],
                claim_hash(claim)
                if claim
                else "",
            )
        )

        output.append(
            {
                "claim_key": key,
                "claim_text": claim,
                "verdict": verdict,
                "confidence": confidence,
                "dataset": first(
                    raw,
                    [
                        "dataset",
                        "source_dataset",
                    ],
                ),
                "raw": raw,
            }
        )

    return output


def _evidence_list(
    obj: Any,
) -> list[dict]:
    if isinstance(
        obj,
        list,
    ):
        return [
            item
            for item in obj
            if isinstance(
                item,
                dict,
            )
        ]

    if isinstance(
        obj,
        dict,
    ):
        for key in (
            "evidence",
            "top_evidence",
            "retrieved_evidence",
            "documents",
            "results",
            "items",
            "top3",
            "selected_evidence",
        ):
            if isinstance(
                obj.get(key),
                list,
            ):
                return [
                    item
                    for item in obj[key]
                    if isinstance(
                        item,
                        dict,
                    )
                ]

    return []


def _float(value):
    try:
        if value in (
            None,
            "",
        ):
            return None

        return float(value)

    except Exception:
        return None


def _parse_evidence(
    raw: dict,
    index: int,
    max_chars: int,
) -> Evidence | None:
    text = normalize_text(
        first(
            raw,
            [
                "text",
                "evidence_text",
                "abstract",
                "content",
                "document",
                "body",
                "snippet",
            ],
        )
    )

    if not text:
        return None

    if len(text) > max_chars:
        text = (
            text[:max_chars].rstrip()
            + " …"
        )

    hierarchy = first(
        raw,
        [
            "hierarchy",
            "tier",
            "class",
            "hierarchy_class",
            "source_tier",
        ],
    )

    source = first(
        raw,
        [
            "source",
            "journal",
            "publisher",
            "source_name",
        ],
    )

    return Evidence(
        local_id=f"[E{index}]",
        source_id=str(
            first(
                raw,
                [
                    "evidence_id",
                    "id",
                    "doc_id",
                    "document_id",
                    "pmid",
                ],
                f"E{index}",
            )
        ),
        title=normalize_text(
            first(
                raw,
                [
                    "title",
                    "document_title",
                    "article_title",
                ],
                "",
            )
        ),
        text=text,
        hierarchy=(
            str(hierarchy)
            if hierarchy
            not in (
                None,
                "",
            )
            else None
        ),
        hierarchy_weight=_float(
            first(
                raw,
                [
                    "hierarchy_weight",
                    "weight",
                    "authority_weight",
                ],
            )
        ),
        retrieval_score=_float(
            first(
                raw,
                [
                    "retrieval_score",
                    "score",
                    "similarity",
                    "rerank_score",
                ],
            )
        ),
        weighted_score=_float(
            first(
                raw,
                [
                    "weighted_score",
                    "ranking_score",
                    "final_score",
                ],
            )
        ),
        source=(
            str(source)
            if source
            not in (
                None,
                "",
            )
            else None
        ),
        metadata=raw,
    )


def _extract_claim_records(
    obj: Any,
    path: Path,
) -> list[dict]:
    """
    Convert supported Phase 2 / Phase 3
    container formats into a list of
    per-claim dictionaries.
    """

    if isinstance(
        obj,
        list,
    ):
        return [
            record
            for record in obj
            if isinstance(
                record,
                dict,
            )
        ]

    if not isinstance(
        obj,
        dict,
    ):
        raise ValueError(
            f"Unsupported evidence "
            f"structure: {path}"
        )

    # Phase 2 format:
    #
    # {
    #   "metadata": {...},
    #   "claims": [...]
    # }
    #
    # Phase 3 format:
    #
    # {
    #   "phase3_metadata": {...},
    #   "claims": [...]
    # }
    if isinstance(
        obj.get("claims"),
        list,
    ):
        return [
            record
            for record in obj[
                "claims"
            ]
            if isinstance(
                record,
                dict,
            )
        ]

    # Generic alternative container formats.
    for key in (
        "results",
        "items",
        "records",
    ):
        if isinstance(
            obj.get(key),
            list,
        ):
            return [
                record
                for record in obj[key]
                if isinstance(
                    record,
                    dict,
                )
            ]

    # Legacy format where claim IDs are
    # already top-level dictionary keys.
    records = []

    for key, value in (
        obj.items()
    ):
        if not isinstance(
            value,
            dict,
        ):
            continue

        if not _evidence_list(
            value
        ):
            continue

        record = dict(
            value
        )

        record.setdefault(
            "claim_id",
            str(key),
        )

        records.append(
            record
        )

    if records:
        return records

    raise ValueError(
        f"No claim records found in "
        f"evidence file: {path}"
    )


def load_evidence_map(
    path: Path,
    max_items: int,
    max_chars: int,
) -> dict[str, list[Evidence]]:
    """
    Load Phase 2 or Phase 3 evidence.

    Supported structures include:

    Phase 2:
        {
            "metadata": {...},
            "claims": [...]
        }

    Phase 3:
        {
            "phase3_metadata": {...},
            "claims": [...]
        }

    Legacy claim-keyed dictionaries are
    also supported.

    Multiple aliases are created for each
    claim:
        - claim_id
        - sample_id
        - claim_key
        - normalized claim hash
        - lowercase normalized claim text
    """

    obj = load_jsonish(
        path
    )

    claim_records = (
        _extract_claim_records(
            obj,
            path,
        )
    )

    output = {}

    for index, record in enumerate(
        claim_records,
        1,
    ):
        claim_text = normalize_text(
            first(
                record,
                [
                    "claim_text",
                    "claim",
                    "text",
                    "statement",
                ],
                "",
            )
        )

        claim_id = first(
            record,
            [
                "claim_id",
                "sample_id",
                "claim_key",
                "id",
            ],
            f"record_{index}",
        )

        evidence = []

        for evidence_index, raw in enumerate(
            _evidence_list(
                record
            )[:max_items],
            1,
        ):
            item = _parse_evidence(
                raw,
                evidence_index,
                max_chars,
            )

            if item:
                evidence.append(
                    item
                )

        aliases = []

        if claim_id not in (
            None,
            "",
        ):
            aliases.append(
                str(claim_id)
            )

        for key in (
            "sample_id",
            "claim_key",
        ):
            value = record.get(
                key
            )

            if value not in (
                None,
                "",
            ):
                aliases.append(
                    str(value)
                )

        if claim_text:
            aliases.append(
                claim_hash(
                    claim_text
                )
            )

            aliases.append(
                claim_text.lower()
            )

        for alias in aliases:
            if alias:
                output[
                    alias
                ] = evidence

    return output


def join_records(
    predictions: list[dict],
    phase3: dict,
    phase2: dict,
):
    """
    Join Phase 4 predictions to evidence.

    Phase 3 is authoritative and preferred.

    Phase 2 is used only as a fallback when
    a Phase 3 evidence match is unavailable.

    Phase 2 and Phase 3 do NOT need to have
    identical claim populations.
    """

    records = []
    unmatched = []

    for prediction in predictions:
        claim = prediction[
            "claim_text"
        ]

        keys = [
            prediction[
                "claim_key"
            ],
            str(
                first(
                    prediction["raw"],
                    ["claim_id"],
                    "",
                )
            ),
            str(
                first(
                    prediction["raw"],
                    ["sample_id"],
                    "",
                )
            ),
            str(
                first(
                    prediction["raw"],
                    ["claim_hash"],
                    "",
                )
            ),
            (
                claim_hash(
                    claim
                )
                if claim
                else ""
            ),
            (
                claim.lower()
                if claim
                else ""
            ),
        ]

        keys = [
            key
            for key in keys
            if key
        ]

        evidence = None
        evidence_source = None
        matched_key = None

        # Phase 3 is the preferred evidence
        # source for Phase 5.
        for key in keys:
            if (
                key in phase3
                and phase3[key]
            ):
                evidence = phase3[
                    key
                ]

                evidence_source = (
                    "phase3"
                )

                matched_key = key

                break

        # Phase 2 is ONLY a fallback.
        if evidence is None:
            for key in keys:
                if (
                    key in phase2
                    and phase2[key]
                ):
                    evidence = phase2[
                        key
                    ]

                    evidence_source = (
                        "phase2"
                    )

                    matched_key = key

                    break

        if not claim:
            unmatched.append(
                {
                    "claim_key": (
                        prediction[
                            "claim_key"
                        ]
                    ),
                    "claim_text": claim,
                    "reason": (
                        "missing_claim_text"
                    ),
                    "candidate_keys": keys,
                }
            )

            continue

        if not evidence:
            unmatched.append(
                {
                    "claim_key": (
                        prediction[
                            "claim_key"
                        ]
                    ),
                    "claim_text": claim,
                    "reason": (
                        "no_matching_evidence"
                    ),
                    "candidate_keys": keys,
                }
            )

            continue

        metadata = dict(
            prediction[
                "raw"
            ]
        )

        metadata[
            "_phase5_evidence_source"
        ] = evidence_source

        metadata[
            "_phase5_evidence_match_key"
        ] = matched_key

        records.append(
            ClaimRecord(
                claim_key=(
                    prediction[
                        "claim_key"
                    ]
                ),
                claim_text=claim,
                verdict=(
                    prediction[
                        "verdict"
                    ]
                ),
                confidence=(
                    prediction[
                        "confidence"
                    ]
                ),
                dataset=(
                    prediction[
                        "dataset"
                    ]
                ),
                evidence=evidence,
                phase4_metadata=metadata,
            )
        )

    return (
        records,
        unmatched,
    )