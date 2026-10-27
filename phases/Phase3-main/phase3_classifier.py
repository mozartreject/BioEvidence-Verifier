import json
from pathlib import Path


INPUT_FILE = Path(
    "data/phase3_output/hierarchy_classified.json"
)

OUTPUT_DIR = Path(
    "data/phase3_output"
)

TOP_K = 3


def to_float(value):

    try:
        return float(value)

    except Exception:

        return 0.0


def rank_evidence(
    evidence_list,
    weighted
):

    ranked = []

    for evidence in evidence_list:

        retrieval_score = to_float(
            evidence.get(
                "score",
                0
            )
        )

        hierarchy_weight = to_float(
            evidence.get(
                "hierarchy_weight",
                1.0
            )
        )

        if weighted:

            final_score = (
                retrieval_score *
                hierarchy_weight
            )

        else:

            final_score = retrieval_score

        item = dict(evidence)

        item[
            "retrieval_score"
        ] = retrieval_score

        item[
            "hierarchy_weight"
        ] = hierarchy_weight

        item[
            "final_score"
        ] = final_score

        ranked.append(item)

    ranked.sort(
        key=lambda x:
        x["final_score"],
        reverse=True
    )

    return ranked[:TOP_K]


def main():

    print("=" * 70)
    print("PHASE 3 — WEIGHTED EVIDENCE RANKING")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    weighted = []
    unweighted = []

    for claim in data["claims"]:

        evidence = claim.get(
            "evidence",
            []
        )

        weighted_top3 = rank_evidence(
            evidence,
            weighted=True
        )

        unweighted_top3 = rank_evidence(
            evidence,
            weighted=False
        )

        claim_id = claim.get(
            "claim_id",
            claim.get("id")
        )

        claim_text = claim.get(
            "claim",
            claim.get(
                "claim_text",
                ""
            )
        )

        weighted.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "evidence": weighted_top3
        })

        unweighted.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "evidence": unweighted_top3
        })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    weighted_file = (
        OUTPUT_DIR /
        "weighted_top3.json"
    )

    unweighted_file = (
        OUTPUT_DIR /
        "unweighted_top3.json"
    )

    with open(
        weighted_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            weighted,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        unweighted_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            unweighted,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nClaims processed: "
        f"{len(weighted)}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"\nWeighted:"
    )

    print(
        weighted_file
    )

    print(
        f"\nUnweighted baseline:"
    )

    print(
        unweighted_file
    )


if __name__ == "__main__":
    main()