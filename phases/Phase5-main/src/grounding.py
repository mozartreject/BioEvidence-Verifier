from __future__ import annotations

import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import ClaimRecord


CITATION_RE = re.compile(
    r"\[E\d+\]"
)

PMID_RE = re.compile(
    r"\bPMID\s*:?\s*\d{4,10}\b",
    re.I,
)

DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.I,
)

URL_RE = re.compile(
    r"https?://\S+",
    re.I,
)

SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+"
)

RAW_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
    )
    %?
    (?![A-Za-z0-9])
    """,
    re.VERBOSE,
)


NEI_PATTERNS = [
    re.compile(
        r"\binsufficient\s+evidence\b",
        re.I,
    ),
    re.compile(
        r"\bevidence\s+(?:is|was|remains)\s+insufficient\b",
        re.I,
    ),
    re.compile(
        r"\binsufficient\s+to\s+"
        r"(?:determine|establish|conclude|support|refute)\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+enough\s+evidence\b",
        re.I,
    ),
    re.compile(
        r"\bevidence\s+(?:is|was)\s+not\s+enough\b",
        re.I,
    ),
    re.compile(
        r"\bcannot\s+"
        r"(?:determine|establish|conclude|support|refute)\b",
        re.I,
    ),
    re.compile(
        r"\bcan(?:not|'t)\s+be\s+determined\b",
        re.I,
    ),
    re.compile(
        r"\bunable\s+to\s+"
        r"(?:determine|establish|conclude)\b",
        re.I,
    ),
    re.compile(
        r"\binconclusive\b",
        re.I,
    ),
    re.compile(
        r"\b(?:does|do|did)\s+not\s+establish\b",
        re.I,
    ),
    re.compile(
        r"\b(?:does|do|did)\s+not\s+provide\s+"
        r"(?:sufficient|enough)\s+evidence\b",
        re.I,
    ),
    re.compile(
        r"\b(?:does|do|did)\s+not\s+"
        r"(?:directly\s+)?address\b",
        re.I,
    ),
    re.compile(
        r"\bevidence\s+(?:does|do|did)\s+not\s+"
        r"(?:directly\s+)?address\b",
        re.I,
    ),
    re.compile(
        r"\bevidence\s+(?:is|remains)\s+"
        r"(?:limited|unclear|inconclusive)\b",
        re.I,
    ),
]


NEI_ABSENCE_PATTERNS = [
    re.compile(
        r"\b(?:does|do|did)\s+not\s+"
        r"(?:directly\s+)?address\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+(?:directly\s+)?addressed\b",
        re.I,
    ),
    re.compile(
        r"\bdoes\s+not\s+provide\s+"
        r"(?:direct|sufficient|enough)\b",
        re.I,
    ),
    re.compile(
        r"\bdo\s+not\s+provide\s+"
        r"(?:direct|sufficient|enough)\b",
        re.I,
    ),
    re.compile(
        r"\bdoes\s+not\s+support\s+or\s+refute\b",
        re.I,
    ),
    re.compile(
        r"\bdo\s+not\s+support\s+or\s+refute\b",
        re.I,
    ),
    re.compile(
        r"\bdoes\s+not\s+support\s+the\s+claim\b",
        re.I,
    ),
    re.compile(
        r"\bdo\s+not\s+support\s+the\s+claim\b",
        re.I,
    ),
    re.compile(
        r"\bdoes\s+not\s+refute\s+the\s+claim\b",
        re.I,
    ),
    re.compile(
        r"\bdo\s+not\s+refute\s+the\s+claim\b",
        re.I,
    ),
    re.compile(
        r"\binsufficient\s+evidence\b",
        re.I,
    ),
    re.compile(
        r"\bevidence\s+(?:is|was|remains)\s+insufficient\b",
        re.I,
    ),
    re.compile(
        r"\binsufficient\s+to\s+"
        r"(?:determine|establish|conclude|support|refute)\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+enough\s+evidence\b",
        re.I,
    ),
    re.compile(
        r"\bcannot\s+"
        r"(?:determine|establish|conclude)\b",
        re.I,
    ),
    re.compile(
        r"\binconclusive\b",
        re.I,
    ),
    re.compile(
        r"\bno\s+direct\s+evidence\b",
        re.I,
    ),
    re.compile(
        r"\bno\s+evidence\s+(?:is\s+)?provided\b",
        re.I,
    ),
    re.compile(
        r"\bnone\s+of\s+(?:the|these)\s+"
        r"(?:evidence|studies|sources)\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+relevant\s+to\s+the\s+claim\b",
        re.I,
    ),
    re.compile(
        r"\bunrelated\s+to\s+the\s+claim\b",
        re.I,
    ),
]


def _norm_num(
    value: str,
) -> str:
    return (
        value.lower()
        .replace(",", "")
        .strip()
    )


def _is_identifier_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    """
    Return True when a numeric-looking substring is
    part of an identifier rather than a quantitative
    statement.

    Examples ignored:
        H1N1
        H1N1209
        PD-1
        HER2
        B6
        [E1]
        IL-6
        SARS-CoV-2

    Standalone quantities remain detectable:
        25%
        50
        3.5
        1,000
    """

    before = (
        text[start - 1]
        if start > 0
        else ""
    )

    after = (
        text[end]
        if end < len(text)
        else ""
    )

    if before.isalnum():
        return True

    if after.isalnum():
        return True

    if before in {"-", "_"}:
        previous = (
            text[start - 2]
            if start >= 2
            else ""
        )

        if previous.isalnum():
            return True

    if after in {"-", "_"}:
        following = (
            text[end + 1]
            if end + 1 < len(text)
            else ""
        )

        if following.isalnum():
            return True

    if start >= 2:
        prefix = text[
            max(
                0,
                start - 2,
            ):start
        ]

        if prefix == "[E":
            return True

    return False


def _extract_numbers(
    text: str,
) -> set[str]:
    """
    Extract standalone quantitative values while
    excluding numbers embedded inside biomedical
    identifiers and evidence citations.
    """

    values: set[str] = set()

    for match in RAW_NUMBER_RE.finditer(
        text
    ):
        if _is_identifier_context(
            text,
            match.start(),
            match.end(),
        ):
            continue

        values.add(
            _norm_num(
                match.group(0)
            )
        )

    return values


def _get_claim_text(
    record: ClaimRecord,
) -> str:
    """
    Robustly retrieve claim text from ClaimRecord.

    The project has used slightly different field
    naming conventions across phases, so this avoids
    coupling numeric provenance validation to one
    specific attribute name.
    """

    for attribute in (
        "claim_text",
        "claim",
        "text",
    ):
        value = getattr(
            record,
            attribute,
            None,
        )

        if value is not None:
            value = str(
                value
            ).strip()

            if value:
                return value

    return ""


def _split_sentences(
    explanation: str,
) -> list[str]:
    return [
        sentence.strip()
        for sentence in SENTENCE_RE.split(
            explanation.strip()
        )
        if sentence.strip()
    ]


def _communicates_nei(
    explanation: str,
) -> bool:
    """
    Check whether an NEI explanation explicitly
    communicates insufficient, inconclusive, or
    non-addressing evidence.
    """

    text = (
        explanation
        .strip()
        .lower()
    )

    if not text:
        return False

    return any(
        pattern.search(
            text
        )
        for pattern in NEI_PATTERNS
    )


def _is_nei_absence_sentence(
    sentence: str,
) -> bool:
    """
    Identify a sentence whose primary role is to
    explain that the supplied evidence does not
    establish, address, support, or refute the claim.

    This is used ONLY for locked NEI verdicts.

    Such sentences can legitimately have low cosine
    similarity to the supplied evidence because they
    often mention the claim concept specifically to
    state that it is absent from the evidence.
    """

    text = sentence.strip()

    if not text:
        return False

    return any(
        pattern.search(
            text
        )
        for pattern in NEI_ABSENCE_PATTERNS
    )


class SemanticGrounder:
    def __init__(
        self,
        model_name: str,
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def sentence_support(
        self,
        explanation: str,
        record: ClaimRecord,
    ):
        sentences = _split_sentences(
            explanation
        )

        evidence = [
            f"{item.title}. {item.text}"
            for item in record.evidence
        ]

        if not sentences:
            return {
                "sentences": [],
                "sentence_scores": [],
                "mean": 0.0,
                "minimum": 0.0,
            }

        if not evidence:
            return {
                "sentences": sentences,
                "sentence_scores": [
                    0.0
                    for _ in sentences
                ],
                "mean": 0.0,
                "minimum": 0.0,
            }

        explanation_embeddings = (
            self.model.encode(
                sentences,
                normalize_embeddings=True,
            )
        )

        evidence_embeddings = (
            self.model.encode(
                evidence,
                normalize_embeddings=True,
            )
        )

        similarity = np.matmul(
            explanation_embeddings,
            evidence_embeddings.T,
        )

        best = similarity.max(
            axis=1
        )

        return {
            "sentences": sentences,
            "sentence_scores": [
                float(
                    value
                )
                for value in best
            ],
            "mean": float(
                best.mean()
            ),
            "minimum": float(
                best.min()
            ),
        }

    def evidence_coverage(
        self,
        explanation: str,
        record: ClaimRecord,
    ):
        evidence = [
            f"{item.title}. {item.text}"
            for item in record.evidence
        ]

        if not evidence:
            return {
                "per_evidence": {},
                "mean": 0.0,
                "max": 0.0,
            }

        explanation_embedding = (
            self.model.encode(
                [
                    explanation
                ],
                normalize_embeddings=True,
            )[0]
        )

        evidence_embeddings = (
            self.model.encode(
                evidence,
                normalize_embeddings=True,
            )
        )

        scores = np.matmul(
            evidence_embeddings,
            explanation_embedding,
        )

        return {
            "per_evidence": {
                record.evidence[
                    index
                ].local_id: float(
                    scores[
                        index
                    ]
                )
                for index in range(
                    len(
                        scores
                    )
                )
            },
            "mean": (
                float(
                    scores.mean()
                )
                if len(
                    scores
                )
                else 0.0
            ),
            "max": (
                float(
                    scores.max()
                )
                if len(
                    scores
                )
                else 0.0
            ),
        }


def _compute_semantic_validation(
    record: ClaimRecord,
    semantic: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine the semantic score used for threshold
    validation.

    IMPORTANT:
    - Raw semantic metrics are NEVER overwritten.
    - SUPPORTED and REFUTED always use the raw mean.
    - Only locked NEI explanations may exclude
      explicit evidence-absence/insufficiency
      sentences from threshold computation.
    - At least one ordinary evidence-descriptive
      sentence must remain. If every sentence is an
      absence statement, the raw mean is retained.

    This prevents an NEI explanation from passing
    merely by repeatedly stating that evidence is
    insufficient.
    """

    raw_mean = float(
        semantic.get(
            "mean",
            0.0,
        )
    )

    sentences = semantic.get(
        "sentences",
        [],
    )

    sentence_scores = semantic.get(
        "sentence_scores",
        [],
    )

    result = {
        "raw_mean": raw_mean,
        "validation_mean": raw_mean,
        "used_nei_absence_adjustment": False,
        "excluded_sentence_indices": [],
        "retained_sentence_indices": list(
            range(
                len(
                    sentence_scores
                )
            )
        ),
    }

    if record.verdict != "NEI":
        return result

    if not sentences:
        return result

    if len(
        sentences
    ) != len(
        sentence_scores
    ):
        return result

    excluded_indices = []
    retained_indices = []

    for index, sentence in enumerate(
        sentences
    ):
        if _is_nei_absence_sentence(
            sentence
        ):
            excluded_indices.append(
                index
            )
        else:
            retained_indices.append(
                index
            )

    # Do not allow an explanation composed entirely
    # of absence statements to bypass semantic
    # grounding.
    if not retained_indices:
        return result

    # No adjustment is needed when nothing matched.
    if not excluded_indices:
        return result

    retained_scores = [
        float(
            sentence_scores[
                index
            ]
        )
        for index in retained_indices
    ]

    adjusted_mean = float(
        np.mean(
            retained_scores
        )
    )

    result[
        "validation_mean"
    ] = adjusted_mean

    result[
        "used_nei_absence_adjustment"
    ] = True

    result[
        "excluded_sentence_indices"
    ] = excluded_indices

    result[
        "retained_sentence_indices"
    ] = retained_indices

    return result


def validate_generated(
    record: ClaimRecord,
    output: dict,
    cfg: dict,
    min_words: int,
    max_words: int,
    grounder: SemanticGrounder,
):
    errors = []
    warnings = []

    explanation = str(
        output.get(
            "explanation",
            "",
        )
    ).strip()

    verdict = str(
        output.get(
            "verdict",
            "",
        )
    ).strip().upper()

    ids = output.get(
        "evidence_ids_used",
        [],
    )

    if not isinstance(
        ids,
        list,
    ):
        errors.append(
            "evidence_ids_used must be a list"
        )

        ids = []

    ids = {
        str(
            value
        ).strip()
        for value in ids
    }

    if (
        cfg[
            "fail_on_verdict_mismatch"
        ]
        and verdict
        != record.verdict
    ):
        errors.append(
            "Verdict mismatch: "
            f"expected {record.verdict}, "
            f"got {verdict or '[empty]'}"
        )

    word_count = len(
        explanation.split()
    )

    if word_count < min_words:
        errors.append(
            "Explanation too short: "
            f"{word_count} words"
        )

    if word_count > max_words:
        errors.append(
            "Explanation too long: "
            f"{word_count} words"
        )

    valid_ids = {
        item.local_id
        for item in record.evidence
    }

    cited = set(
        CITATION_RE.findall(
            explanation
        )
    )

    invalid = sorted(
        (
            cited
            | ids
        )
        - valid_ids
    )

    if (
        invalid
        and cfg[
            "fail_on_invalid_citations"
        ]
    ):
        errors.append(
            f"Invalid citations: {invalid}"
        )

    if not cited:
        errors.append(
            "Explanation contains no evidence citation"
        )

    if cited != ids:
        warnings.append(
            "Citations in explanation differ "
            "from evidence_ids_used"
        )

    evidence_text = " ".join(
        f"{item.title} {item.text}"
        for item in record.evidence
    )

    claim_text = _get_claim_text(
        record
    )

    explanation_numbers = (
        _extract_numbers(
            explanation
        )
    )

    evidence_numbers = (
        _extract_numbers(
            evidence_text
        )
    )

    claim_numbers = (
        _extract_numbers(
            claim_text
        )
    )

    allowed_numbers = (
        evidence_numbers
        | claim_numbers
    )

    unsupported_numbers = sorted(
        explanation_numbers
        - allowed_numbers
    )

    if (
        unsupported_numbers
        and cfg[
            "fail_on_unsupported_numbers"
        ]
    ):
        errors.append(
            "Unsupported numeric claims: "
            f"{unsupported_numbers}"
        )

    evidence_text_lower = (
        evidence_text.lower()
    )

    suspicious = []

    for label, regex in [
        (
            "PMID",
            PMID_RE,
        ),
        (
            "DOI",
            DOI_RE,
        ),
        (
            "URL",
            URL_RE,
        ),
    ]:
        for match in regex.findall(
            explanation
        ):
            value = str(
                match
            )

            if (
                value.lower()
                not in evidence_text_lower
            ):
                suspicious.append(
                    {
                        "type":
                            label,

                        "value":
                            value,
                    }
                )

    if (
        suspicious
        and cfg[
            "fail_on_suspicious_identifiers"
        ]
    ):
        errors.append(
            "Unsupported identifiers/URLs: "
            f"{suspicious}"
        )

    if (
        record.verdict == "NEI"
        and not _communicates_nei(
            explanation
        )
    ):
        errors.append(
            "NEI explanation must explicitly "
            "communicate insufficient/"
            "inconclusive evidence"
        )

    semantic = (
        grounder.sentence_support(
            explanation,
            record,
        )
    )

    coverage = (
        grounder.evidence_coverage(
            explanation,
            record,
        )
    )

    semantic_validation = (
        _compute_semantic_validation(
            record,
            semantic,
        )
    )

    semantic_threshold = float(
        cfg[
            "semantic_threshold"
        ]
    )

    coverage_threshold = float(
        cfg[
            "coverage_threshold"
        ]
    )

    semantic_validation_mean = float(
        semantic_validation[
            "validation_mean"
        ]
    )

    if (
        semantic_validation[
            "used_nei_absence_adjustment"
        ]
    ):
        warnings.append(
            "NEI semantic validation excluded "
            "explicit evidence-absence/"
            "insufficiency sentence(s); "
            "raw semantic metrics are preserved"
        )

    if (
        semantic_validation_mean
        < semantic_threshold
    ):
        errors.append(
            "Semantic grounding "
            f"{semantic_validation_mean:.4f} "
            "below threshold "
            f"{semantic_threshold:.4f}"
        )

    if (
        coverage[
            "max"
        ]
        < coverage_threshold
    ):
        errors.append(
            "Evidence coverage "
            f"{coverage['max']:.4f} "
            "below threshold "
            f"{coverage_threshold:.4f}"
        )

    return {
        "passed":
            not errors,

        "errors":
            errors,

        "warnings":
            warnings,

        "word_count":
            word_count,

        "valid_citations":
            sorted(
                cited
                & valid_ids
            ),

        "invalid_citations":
            invalid,

        "unsupported_numbers":
            unsupported_numbers,

        "number_provenance": {
            "claim_numbers":
                sorted(
                    claim_numbers
                ),

            "evidence_numbers":
                sorted(
                    evidence_numbers
                ),

            "explanation_numbers":
                sorted(
                    explanation_numbers
                ),

            "allowed_numbers":
                sorted(
                    allowed_numbers
                ),
        },

        "suspicious_identifiers":
            suspicious,

        "semantic_grounding":
            semantic,

        "semantic_validation":
            semantic_validation,

        "evidence_coverage":
            coverage,
    }