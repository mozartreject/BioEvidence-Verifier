#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================
# GLOBAL PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

PHASE2_ROOT = Path.home() / "Downloads" / "phase2"

PHASE3_ROOT = Path.home() / "Downloads" / "Phase3-main"

DATA_ROOT = ROOT / "data" / "final"

ORIGINAL_MODEL3_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "phase4"
    / "model3_improved"
    / "best_model3_improved.pt"
)

DEV_90_FILE = (
    ROOT
    / "results"
    / "end_to_end_balanced"
    / "end_to_end_predictions.jsonl"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "e2e_retrieval_conditioned_final"
)

CACHE_DIR = OUTPUT_ROOT / "cache"

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoint"

NEW_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_retrieval_conditioned_verifier.pt"
)

HARD_NEGATIVE_FILE = (
    CACHE_DIR
    / "hard_neutral_negatives.jsonl"
)

FINAL_SUBSET_FILE = (
    CACHE_DIR
    / "fresh_final_pubhealth_balanced.jsonl"
)

FINAL_SUBSET_MANIFEST = (
    CACHE_DIR
    / "fresh_final_pubhealth_manifest.json"
)

FINAL_RETRIEVAL_CACHE = (
    CACHE_DIR
    / "fresh_final_retrieval_phase3.jsonl"
)

DEV_FEATURE_CACHE = (
    CACHE_DIR
    / "dev90_new_model_features.jsonl"
)

FINAL_FEATURE_CACHE = (
    CACHE_DIR
    / "fresh_final_features.jsonl"
)

TRAINING_HISTORY_FILE = (
    OUTPUT_ROOT
    / "training_history.json"
)

TUNING_FILE = (
    OUTPUT_ROOT
    / "dev_fusion_selection.json"
)

FINAL_METRICS_FILE = (
    OUTPUT_ROOT
    / "FINAL_UNTOUCHED_METRICS.json"
)

FINAL_PREDICTIONS_FILE = (
    OUTPUT_ROOT
    / "FINAL_UNTOUCHED_PREDICTIONS.jsonl"
)

FINAL_REPORT_FILE = (
    OUTPUT_ROOT
    / "FINAL_UNTOUCHED_REPORT.txt"
)

AUDIT_FILE = (
    OUTPUT_ROOT
    / "experiment_audit.json"
)


# ============================================================
# EXPERIMENT CONSTANTS
# ============================================================

SEED = 42

FINAL_SELECTION_SEED = 20260912

LABELS = [0, 1, 2]

LABEL_NAMES = {
    0: "REFUTED",
    1: "NEI",
    2: "SUPPORTED",
}

NAME_TO_LABEL = {
    "REFUTED": 0,
    "REFUTE": 0,
    "FALSE": 0,
    "0": 0,

    "NEI": 1,
    "NOT_ENOUGH_INFORMATION": 1,
    "NOT ENOUGH INFORMATION": 1,
    "UNPROVEN": 1,
    "1": 1,

    "SUPPORTED": 2,
    "SUPPORT": 2,
    "TRUE": 2,
    "2": 2,
}

TIER_WEIGHTS = {
    "A": 1.0,
    "B": 0.7,
    "C": 0.4,
    "D": 0.2,
}

BIOMEDICAL_NLI_MODEL = (
    "pritamdeka/PubMedBERT-MNLI-MedNLI"
)

MAX_LENGTH = 384

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16

GRAD_ACCUMULATION = 2

LEARNING_RATE = 8e-6

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_EPOCHS = 3

PATIENCE = 2

LABEL_SMOOTHING = 0.05

DROPOUT = 0.20

HARD_NEGATIVE_TARGET = 1800

HARD_NEGATIVE_SEARCH_CANDIDATES = 8

HARD_NEGATIVE_NEUTRAL_THRESHOLD = 0.65

FINAL_PER_CLASS = 15

PHASE2_INITIAL_K = 50

PHASE2_RERANK_K = 10

PHASE3_CLASSIFY_K = 5

FINAL_EVIDENCE_K = 3


# ============================================================
# BASIC UTILITIES
# ============================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def cleanup_torch() -> None:

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():

        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def normalize_claim(text: str) -> str:

    text = str(text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def sigmoid(value: float) -> float:

    value = float(value)

    if value >= 0:

        z = math.exp(-value)

        return 1.0 / (1.0 + z)

    z = math.exp(value)

    return z / (1.0 + z)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return default


def atomic_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        str(path) + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(text)

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        temp,
        path,
    )


def atomic_json(
    path: Path,
    data: Any,
) -> None:

    atomic_text(
        path,
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )


def atomic_jsonl(
    path: Path,
    rows: list[dict],
) -> None:

    atomic_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
    )


def load_jsonl(
    path: Path,
) -> list[dict]:

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

            except Exception as exc:

                raise RuntimeError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from exc

            rows.append(row)

    return rows


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_split_file(
    split: str,
) -> Path:

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"Missing data directory:\n"
            f"{DATA_ROOT}"
        )

    candidates = []

    for path in DATA_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jsonl",
            ".csv",
            ".json",
        }:
            continue

        stem = path.stem.lower()

        if split.lower() not in stem:
            continue

        if any(
            banned in stem
            for banned in [
                "balanced",
                "manifest",
                "prediction",
                "subset",
            ]
        ):
            continue

        candidates.append(path)

    if not candidates:

        raise FileNotFoundError(
            f"Could not discover {split} "
            f"split under:\n{DATA_ROOT}"
        )

    candidates.sort(
        key=lambda p: (
            0 if p.suffix == ".jsonl" else 1,
            len(p.name),
            p.name,
        )
    )

    chosen = candidates[0]

    print(
        f"{split.upper()} split: {chosen}"
    )

    return chosen


def read_table(
    path: Path,
) -> list[dict]:

    suffix = path.suffix.lower()

    if suffix == ".jsonl":

        return load_jsonl(path)

    if suffix == ".json":

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

        raise RuntimeError(
            f"Expected JSON list in {path}"
        )

    if suffix == ".csv":

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            return list(
                csv.DictReader(handle)
            )

    raise RuntimeError(
        f"Unsupported split: {path}"
    )


# ============================================================
# SCHEMA NORMALIZATION
# ============================================================

CLAIM_KEYS = [
    "claim",
    "claim_text",
    "statement",
    "query",
]

EVIDENCE_KEYS = [
    "evidence",
    "evidence_text",
    "abstract",
    "context",
    "document",
    "premise",
]

LABEL_KEYS = [
    "label",
    "label_id",
    "target",
    "verdict",
    "class",
]

DATASET_KEYS = [
    "dataset",
    "source_dataset",
    "source",
]


def first_present(
    row: dict,
    keys: list[str],
) -> Any:

    for key in keys:

        if key in row:

            value = row[key]

            if value is not None:
                return value

    return None


def parse_label(
    raw: Any,
) -> int:

    if isinstance(raw, bool):

        return (
            2
            if raw
            else 0
        )

    if isinstance(
        raw,
        (int, np.integer),
    ):

        value = int(raw)

        if value in LABELS:
            return value

    if isinstance(raw, float):

        if (
            math.isfinite(raw)
            and int(raw) in LABELS
        ):

            return int(raw)

    text = str(
        raw
    ).strip().upper()

    text = text.replace(
        "-",
        "_",
    )

    if text in NAME_TO_LABEL:

        return NAME_TO_LABEL[text]

    raise ValueError(
        f"Cannot map label: {raw!r}"
    )


def canonical_row(
    row: dict,
) -> dict:

    claim = first_present(
        row,
        CLAIM_KEYS,
    )

    evidence = first_present(
        row,
        EVIDENCE_KEYS,
    )

    label = first_present(
        row,
        LABEL_KEYS,
    )

    dataset_name = first_present(
        row,
        DATASET_KEYS,
    )

    if claim is None:

        raise RuntimeError(
            "Could not locate claim column. "
            f"Columns were: {list(row.keys())}"
        )

    if label is None:

        raise RuntimeError(
            "Could not locate label column. "
            f"Columns were: {list(row.keys())}"
        )

    if evidence is None:

        evidence = (
            "[NO_EVIDENCE_AVAILABLE]"
        )

    return {
        "claim":
            str(claim).strip(),

        "evidence":
            str(evidence).strip()
            or "[NO_EVIDENCE_AVAILABLE]",

        "label":
            parse_label(label),

        "dataset":
            str(
                dataset_name
                or ""
            ).strip(),
    }


def canonicalize_rows(
    raw_rows: list[dict],
) -> list[dict]:

    rows = []

    errors = []

    for index, row in enumerate(
        raw_rows
    ):

        try:

            item = canonical_row(
                row
            )

            item[
                "source_row_index"
            ] = index

            rows.append(item)

        except Exception as exc:

            errors.append(
                (
                    index,
                    str(exc),
                )
            )

    if not rows:

        raise RuntimeError(
            "No dataset rows could be "
            "normalized.\n"
            f"Examples: {errors[:3]}"
        )

    if errors:

        print(
            f"WARNING: skipped "
            f"{len(errors)} malformed rows."
        )

    return rows


# ============================================================
# MODEL 3
# ============================================================

def load_model3_class():

    if str(ROOT) not in sys.path:

        sys.path.insert(
            0,
            str(ROOT),
        )

    if (
        "evaluate_model3_final_test"
        in sys.modules
    ):

        del sys.modules[
            "evaluate_model3_final_test"
        ]

    module = importlib.import_module(
        "evaluate_model3_final_test"
    )

    return (
        getattr(
            module,
            "PubMedBERTVerifier",
        ),
        getattr(
            module,
            "MODEL_NAME",
        ),
    )


def load_model3_checkpoint(
    checkpoint_path: Path,
    target_device: torch.device,
):

    ModelClass, model_name = (
        load_model3_class()
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name
        )
    )

    model = ModelClass()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            f"Checkpoint missing "
            f"model_state_dict:\n"
            f"{checkpoint_path}"
        )

    result = model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=False,
    )

    if result.missing_keys:

        raise RuntimeError(
            "Missing model keys:\n"
            + "\n".join(
                result.missing_keys
            )
        )

    if result.unexpected_keys:

        raise RuntimeError(
            "Unexpected model keys:\n"
            + "\n".join(
                result.unexpected_keys
            )
        )

    model.to(
        target_device
    )

    return (
        model,
        tokenizer,
        model_name,
        checkpoint,
    )


# ============================================================
# BIOMEDICAL NLI
# ============================================================

def load_biomedical_nli(
    target_device: torch.device,
):

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BIOMEDICAL_NLI_MODEL
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BIOMEDICAL_NLI_MODEL
        )
    )

    id2label = {
        int(key):
            str(value).lower()
        for key, value
        in model.config.id2label.items()
    }

    mapping = {}

    for index, name in (
        id2label.items()
    ):

        if "contrad" in name:

            mapping[
                "contradiction"
            ] = index

        elif "entail" in name:

            mapping[
                "entailment"
            ] = index

        elif "neutral" in name:

            mapping[
                "neutral"
            ] = index

    required = {
        "contradiction",
        "entailment",
        "neutral",
    }

    if set(
        mapping.keys()
    ) != required:

        raise RuntimeError(
            f"Could not map NLI labels: "
            f"{id2label}"
        )

    model.to(
        target_device
    )

    model.eval()

    return (
        model,
        tokenizer,
        mapping,
    )


@torch.no_grad()
def nli_batch(
    model,
    tokenizer,
    mapping,
    target_device,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 16,
) -> list[dict]:

    results = []

    for start in range(
        0,
        len(premises),
        batch_size,
    ):

        end = start + batch_size

        encoded = tokenizer(
            premises[
                start:end
            ],
            hypotheses[
                start:end
            ],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        encoded = {
            key:
                value.to(
                    target_device
                )
            for key, value
            in encoded.items()
        }

        logits = model(
            **encoded
        ).logits

        probs = torch.softmax(
            logits,
            dim=-1,
        ).detach().cpu().numpy()

        for row in probs:

            results.append(
                {
                    "contradiction":
                        float(
                            row[
                                mapping[
                                    "contradiction"
                                ]
                            ]
                        ),

                    "entailment":
                        float(
                            row[
                                mapping[
                                    "entailment"
                                ]
                            ]
                        ),

                    "neutral":
                        float(
                            row[
                                mapping[
                                    "neutral"
                                ]
                            ]
                        ),
                }
            )

    return results


# ============================================================
# HARD NEGATIVE GENERATION
# ============================================================

def generate_hard_negatives(
    train_rows: list[dict],
    target_device: torch.device,
) -> list[dict]:

    if HARD_NEGATIVE_FILE.exists():

        rows = load_jsonl(
            HARD_NEGATIVE_FILE
        )

        print(
            f"Using cached hard negatives: "
            f"{len(rows)}"
        )

        return rows

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GENERATING RETRIEVAL-LIKE "
        "HARD NEUTRAL NEGATIVES"
    )

    print(
        "=" * 80
    )

    evidence_bank = [
        row["evidence"]
        for row in train_rows
        if row[
            "evidence"
        ]
        != "[NO_EVIDENCE_AVAILABLE]"
    ]

    evidence_claims = [
        normalize_claim(
            row["claim"]
        )
        for row in train_rows
        if row[
            "evidence"
        ]
        != "[NO_EVIDENCE_AVAILABLE]"
    ]

    if not evidence_bank:

        raise RuntimeError(
            "Training set contains "
            "no usable evidence."
        )

    selected_indices = list(
        range(
            len(train_rows)
        )
    )

    random.Random(
        SEED
    ).shuffle(
        selected_indices
    )

    selected_indices = (
        selected_indices[
            : min(
                len(selected_indices),
                HARD_NEGATIVE_TARGET
                * 2,
            )
        ]
    )

    queries = [
        train_rows[i][
            "claim"
        ]
        for i in selected_indices
    ]

    print(
        "Building lexical "
        "hard-negative index..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=120000,
        sublinear_tf=True,
    )

    document_matrix = (
        vectorizer.fit_transform(
            evidence_bank
        )
    )

    query_matrix = (
        vectorizer.transform(
            queries
        )
    )

    candidate_pairs = []

    print(
        "Finding topic-similar evidence..."
    )

    for local_index, (
        train_index,
        query_vector,
    ) in enumerate(
        zip(
            selected_indices,
            query_matrix,
        )
    ):

        similarities = (
            query_vector
            @ document_matrix.T
        ).toarray()[0]

        search_k = min(
            HARD_NEGATIVE_SEARCH_CANDIDATES
            * 3,
            len(similarities),
        )

        if search_k <= 0:
            continue

        top_indices = np.argpartition(
            similarities,
            -search_k,
        )[
            -search_k:
        ]

        top_indices = top_indices[
            np.argsort(
                similarities[
                    top_indices
                ]
            )[::-1]
        ]

        current_claim_norm = (
            normalize_claim(
                train_rows[
                    train_index
                ][
                    "claim"
                ]
            )
        )

        chosen = []

        for evidence_index in (
            top_indices
        ):

            if (
                evidence_claims[
                    evidence_index
                ]
                ==
                current_claim_norm
            ):
                continue

            similarity = float(
                similarities[
                    evidence_index
                ]
            )

            if similarity <= 0:
                continue

            chosen.append(
                (
                    int(
                        evidence_index
                    ),
                    similarity,
                )
            )

            if (
                len(chosen)
                >=
                HARD_NEGATIVE_SEARCH_CANDIDATES
            ):
                break

        for evidence_index, score in (
            chosen[:3]
        ):

            candidate_pairs.append(
                {
                    "claim":
                        train_rows[
                            train_index
                        ][
                            "claim"
                        ],

                    "evidence":
                        evidence_bank[
                            evidence_index
                        ],

                    "lexical_score":
                        score,
                }
            )

    if not candidate_pairs:

        raise RuntimeError(
            "No hard-negative candidates "
            "were produced."
        )

    print(
        f"Candidate pairs: "
        f"{len(candidate_pairs)}"
    )

    print(
        "Loading biomedical NLI..."
    )

    (
        nli_model,
        nli_tokenizer,
        nli_mapping,
    ) = load_biomedical_nli(
        target_device
    )

    premises = [
        row["evidence"]
        for row
        in candidate_pairs
    ]

    hypotheses = [
        row["claim"]
        for row
        in candidate_pairs
    ]

    print(
        "Filtering topic-similar "
        "documents for neutral relation..."
    )

    nli_results = nli_batch(
        nli_model,
        nli_tokenizer,
        nli_mapping,
        target_device,
        premises,
        hypotheses,
    )

    accepted = []

    seen_claims = Counter()

    for candidate, nli in zip(
        candidate_pairs,
        nli_results,
    ):

        if (
            nli["neutral"]
            <
            HARD_NEGATIVE_NEUTRAL_THRESHOLD
        ):
            continue

        claim_norm = normalize_claim(
            candidate[
                "claim"
            ]
        )

        if seen_claims[
            claim_norm
        ] >= 1:
            continue

        accepted.append(
            {
                "claim":
                    candidate[
                        "claim"
                    ],

                "evidence":
                    candidate[
                        "evidence"
                    ],

                "label":
                    1,

                "dataset":
                    "SYNTHETIC_RETRIEVAL_NEUTRAL",

                "lexical_score":
                    candidate[
                        "lexical_score"
                    ],

                "nli_neutral":
                    nli[
                        "neutral"
                    ],

                "nli_entailment":
                    nli[
                        "entailment"
                    ],

                "nli_contradiction":
                    nli[
                        "contradiction"
                    ],
            }
        )

        seen_claims[
            claim_norm
        ] += 1

        if (
            len(accepted)
            >= HARD_NEGATIVE_TARGET
        ):
            break

    del nli_model
    del nli_tokenizer

    cleanup_torch()

    if len(accepted) < 300:

        raise RuntimeError(
            "Too few reliable neutral hard "
            f"negatives were found: "
            f"{len(accepted)}"
        )

    atomic_jsonl(
        HARD_NEGATIVE_FILE,
        accepted,
    )

    print(
        f"Accepted neutral "
        f"hard negatives: "
        f"{len(accepted)}"
    )

    return accepted


# ============================================================
# PYTORCH TRAINING DATASET
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        rows: list[dict],
        tokenizer,
    ):

        self.rows = rows

        self.tokenizer = tokenizer

    def __len__(self):

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.rows[
            index
        ]

        encoded = self.tokenizer(
            row["claim"],
            row["evidence"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        encoded[
            "labels"
        ] = int(
            row[
                "label"
            ]
        )

        return encoded


def collate_function(
    tokenizer,
):

    def collate(
        examples,
    ):

        labels = torch.tensor(
            [
                example.pop(
                    "labels"
                )
                for example
                in examples
            ],
            dtype=torch.long,
        )

        batch = tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )

        batch[
            "labels"
        ] = labels

        return batch

    return collate


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:

    (
        precision,
        recall,
        f1,
        support,
    ) = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    return {
        "n":
            len(y_true),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=LABELS,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "prediction_distribution":
            dict(
                Counter(
                    LABEL_NAMES[x]
                    for x in y_pred
                )
            ),

        "per_class": {
            LABEL_NAMES[
                label
            ]: {
                "precision":
                    float(
                        precision[i]
                    ),

                "recall":
                    float(
                        recall[i]
                    ),

                "f1":
                    float(
                        f1[i]
                    ),

                "support":
                    int(
                        support[i]
                    ),
            }
            for i, label
            in enumerate(
                LABELS
            )
        },

        "confusion_matrix":
            cm.tolist(),
    }


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    target_device,
) -> dict:

    model.eval()

    y_true = []
    y_pred = []

    losses = []

    criterion = (
        nn.CrossEntropyLoss()
    )

    for batch in loader:

        labels = batch.pop(
            "labels"
        ).to(
            target_device
        )

        inputs = {
            key:
                value.to(
                    target_device
                )
            for key, value
            in batch.items()
        }

        logits = model(
            **inputs
        )

        loss = criterion(
            logits,
            labels,
        )

        predictions = torch.argmax(
            logits,
            dim=-1,
        )

        y_true.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        y_pred.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        losses.append(
            float(
                loss.item()
            )
        )

    output = metrics(
        y_true,
        y_pred,
    )

    output[
        "loss"
    ] = float(
        np.mean(losses)
    )

    return output


# ============================================================
# RETRIEVAL-CONDITIONED TRAINING
# ============================================================

def train_retrieval_conditioned_model(
    train_rows: list[dict],
    val_rows: list[dict],
    hard_negatives: list[dict],
    target_device: torch.device,
) -> None:

    if NEW_CHECKPOINT.exists():

        print(
            "\nUsing existing retrieval-conditioned "
            "checkpoint:"
        )

        print(
            NEW_CHECKPOINT
        )

        return

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRAINING RETRIEVAL-CONDITIONED "
        "PHASE 4 VERIFIER"
    )

    print(
        "=" * 80
    )

    augmented_train = (
        list(train_rows)
        + list(hard_negatives)
    )

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        augmented_train
    )

    print(
        f"Original train: "
        f"{len(train_rows)}"
    )

    print(
        f"Hard neutral negatives: "
        f"{len(hard_negatives)}"
    )

    print(
        f"Augmented train: "
        f"{len(augmented_train)}"
    )

    print(
        "Augmented label distribution:",
        Counter(
            LABEL_NAMES[
                row["label"]
            ]
            for row
            in augmented_train
        ),
    )

    (
        model,
        tokenizer,
        model_name,
        original_checkpoint,
    ) = load_model3_checkpoint(
        ORIGINAL_MODEL3_CHECKPOINT,
        target_device,
    )

    train_dataset = PairDataset(
        augmented_train,
        tokenizer,
    )

    val_dataset = PairDataset(
        val_rows,
        tokenizer,
    )

    collate = collate_function(
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    counts = np.asarray(
        [
            sum(
                row["label"] == label
                for row
                in augmented_train
            )
            for label in LABELS
        ],
        dtype=float,
    )

    inverse = (
        counts.sum()
        / (
            len(LABELS)
            * counts
        )
    )

    class_weights = np.sqrt(
        inverse
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=target_device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    update_steps_per_epoch = math.ceil(
        len(train_loader)
        /
        GRAD_ACCUMULATION
    )

    total_update_steps = (
        update_steps_per_epoch
        * MAX_EPOCHS
    )

    warmup_steps = int(
        total_update_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
    )

    history = []

    best_macro_f1 = -1.0

    no_improvement = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        print(
            f"\nEpoch "
            f"{epoch}/{MAX_EPOCHS}"
        )

        model.train()

        train_losses = []

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            labels = batch.pop(
                "labels"
            ).to(
                target_device
            )

            inputs = {
                key:
                    value.to(
                        target_device
                    )
                for key, value
                in batch.items()
            }

            logits = model(
                **inputs
            )

            loss = criterion(
                logits,
                labels,
            )

            scaled_loss = (
                loss
                /
                GRAD_ACCUMULATION
            )

            scaled_loss.backward()

            train_losses.append(
                float(
                    loss.item()
                )
            )

            should_update = (
                step
                % GRAD_ACCUMULATION
                == 0
                or
                step
                == len(train_loader)
            )

            if should_update:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            if (
                step % 200 == 0
                or step
                == len(train_loader)
            ):

                print(
                    f"  step "
                    f"{step}/"
                    f"{len(train_loader)} "
                    f"loss="
                    f"{np.mean(train_losses[-200:]):.4f}"
                )

        val_metrics = evaluate_model(
            model,
            val_loader,
            target_device,
        )

        epoch_record = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    np.mean(
                        train_losses
                    )
                ),

            "validation":
                val_metrics,
        }

        history.append(
            epoch_record
        )

        atomic_json(
            TRAINING_HISTORY_FILE,
            history,
        )

        print(
            "Validation:"
        )

        print(
            f"  Accuracy   = "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro-F1   = "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Weighted-F1= "
            f"{val_metrics['weighted_f1']:.4f}"
        )

        if (
            val_metrics[
                "macro_f1"
            ]
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                val_metrics[
                    "macro_f1"
                ]
            )

            no_improvement = 0

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "validation_macro_f1":
                        best_macro_f1,

                    "base_checkpoint":
                        str(
                            ORIGINAL_MODEL3_CHECKPOINT
                        ),

                    "hard_negative_count":
                        len(
                            hard_negatives
                        ),

                    "training_configuration": {
                        "learning_rate":
                            LEARNING_RATE,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "label_smoothing":
                            LABEL_SMOOTHING,

                        "class_weights":
                            class_weights
                            .detach()
                            .cpu()
                            .tolist(),
                    },
                },
                NEW_CHECKPOINT,
            )

            print(
                "  NEW BEST CHECKPOINT"
            )

        else:

            no_improvement += 1

            if (
                no_improvement
                >= PATIENCE
            ):

                print(
                    "Early stopping."
                )

                break

    del model
    del tokenizer

    cleanup_torch()


# ============================================================
# FRESH FINAL SUBSET
# ============================================================

def create_fresh_final_subset(
    test_rows: list[dict],
) -> list[dict]:

    if FINAL_SUBSET_FILE.exists():

        rows = load_jsonl(
            FINAL_SUBSET_FILE
        )

        print(
            f"Using frozen fresh final "
            f"subset: {len(rows)}"
        )

        return rows

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CREATING NEW UNTOUCHED "
        "FINAL PUBHEALTH SUBSET"
    )

    print(
        "=" * 80
    )

    development_rows = load_jsonl(
        DEV_90_FILE
    )

    dev_claims = {
        normalize_claim(
            row["claim"]
        )
        for row
        in development_rows
    }

    pubhealth = [
        row
        for row
        in test_rows
        if (
            "PUBHEALTH"
            in row[
                "dataset"
            ].upper()
            and
            normalize_claim(
                row["claim"]
            )
            not in dev_claims
        )
    ]

    by_claim = {}

    conflicts = set()

    for row in pubhealth:

        key = normalize_claim(
            row[
                "claim"
            ]
        )

        if key not in by_claim:

            by_claim[key] = row

        elif (
            by_claim[
                key
            ][
                "label"
            ]
            != row[
                "label"
            ]
        ):

            conflicts.add(
                key
            )

    unique_rows = [
        row
        for key, row
        in by_claim.items()
        if key not in conflicts
    ]

    grouped = {
        label: [
            row
            for row
            in unique_rows
            if row[
                "label"
            ] == label
        ]
        for label in LABELS
    }

    print(
        "Available fresh "
        "PUBHEALTH claims:"
    )

    for label in LABELS:

        print(
            f"  "
            f"{LABEL_NAMES[label]}: "
            f"{len(grouped[label])}"
        )

    available_minimum = min(
        len(
            grouped[label]
        )
        for label
        in LABELS
    )

    per_class = min(
        FINAL_PER_CLASS,
        available_minimum,
    )

    if per_class < 10:

        raise RuntimeError(
            "Fewer than 10 untouched "
            "PUBHEALTH examples remain "
            "for at least one class."
        )

    rng = random.Random(
        FINAL_SELECTION_SEED
    )

    selected = []

    for label in LABELS:

        group = list(
            grouped[label]
        )

        rng.shuffle(
            group
        )

        selected.extend(
            group[
                :per_class
            ]
        )

    rng.shuffle(
        selected
    )

    output_rows = []

    for index, row in enumerate(
        selected
    ):

        output_rows.append(
            {
                "index":
                    index,

                "claim":
                    row[
                        "claim"
                    ],

                "true_label":
                    row[
                        "label"
                    ],

                "true_label_name":
                    LABEL_NAMES[
                        row[
                            "label"
                        ]
                    ],

                "dataset":
                    "PUBHEALTH",
            }
        )

    atomic_jsonl(
        FINAL_SUBSET_FILE,
        output_rows,
    )

    manifest = {
        "seed":
            FINAL_SELECTION_SEED,

        "per_class":
            per_class,

        "n":
            len(
                output_rows
            ),

        "distribution":
            dict(
                Counter(
                    row[
                        "true_label_name"
                    ]
                    for row
                    in output_rows
                )
            ),

        "excluded_development_claims":
            len(
                dev_claims
            ),

        "sha256":
            sha256_file(
                FINAL_SUBSET_FILE
            ),

        "note":
            (
                "This subset excludes all "
                "90 claims used in V1/V2/V3 "
                "development experiments."
            ),
    }

    atomic_json(
        FINAL_SUBSET_MANIFEST,
        manifest,
    )

    print(
        "Fresh final subset:"
    )

    print(
        manifest
    )

    return output_rows


# ============================================================
# PROJECT IMPORT ISOLATION
# ============================================================

@contextlib.contextmanager
def project_import_path(
    project_root: Path,
):

    old_path = list(
        sys.path
    )

    sys.path.insert(
        0,
        str(project_root),
    )

    try:

        yield

    finally:

        sys.path[:] = old_path


def clear_modules(
    names: list[str],
) -> None:

    for name in names:

        if name in sys.modules:

            del sys.modules[
                name
            ]


# ============================================================
# PHASE 2 + PHASE 3 FRESH RETRIEVAL
# ============================================================

def build_fresh_retrieval_cache(
    final_rows: list[dict],
) -> list[dict]:

    if FINAL_RETRIEVAL_CACHE.exists():

        rows = load_jsonl(
            FINAL_RETRIEVAL_CACHE
        )

        print(
            f"Using cached Phase2/3 "
            f"final retrieval: "
            f"{len(rows)} claims"
        )

        return rows

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RUNNING PHASE 2 + PHASE 3 "
        "ON FRESH FINAL CLAIMS"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Phase 2
    # --------------------------------------------------------

    clear_modules(
        [
            "config",
            "retriever",
            "reranker",
            "build_faiss_index",
        ]
    )

    with project_import_path(
        PHASE2_ROOT
    ):

        retriever_module = (
            importlib.import_module(
                "retriever"
            )
        )

        reranker_module = (
            importlib.import_module(
                "reranker"
            )
        )

        BM25Retriever = getattr(
            retriever_module,
            "BM25Retriever",
        )

        CrossEncoderReranker = (
            getattr(
                reranker_module,
                "CrossEncoderReranker",
            )
        )

        bm25 = BM25Retriever()

        reranker = (
            CrossEncoderReranker()
        )

    # --------------------------------------------------------
    # Phase 3
    # --------------------------------------------------------

    clear_modules(
        [
            "config",
            "evidence_classifier",
            "hierarchy_weights",
        ]
    )

    with project_import_path(
        PHASE3_ROOT
    ):

        classifier_module = (
            importlib.import_module(
                "evidence_classifier"
            )
        )

        classifier = (
            classifier_module
            .get_classifier(
                use_llm=True
            )
        )

    output_rows = []

    for position, row in enumerate(
        final_rows,
        start=1,
    ):

        claim = row[
            "claim"
        ]

        print(
            f"\n[{position}/"
            f"{len(final_rows)}] "
            f"{claim[:100]}"
        )

        initial = bm25.search(
            claim,
            k=PHASE2_INITIAL_K,
        )

        reranked = reranker.rerank(
            claim,
            initial,
            top_k=PHASE2_RERANK_K,
        )

        prepared = []

        for evidence in reranked:

            item = dict(
                evidence
            )

            raw_score = safe_float(
                item.get(
                    "reranker_score",
                    item.get(
                        "score",
                        0.0,
                    ),
                )
            )

            relevance = sigmoid(
                raw_score
            )

            item[
                "raw_reranker_score"
            ] = raw_score

            item[
                "retrieval_score"
            ] = relevance

            item[
                "score"
            ] = relevance

            prepared.append(
                item
            )

        for evidence in prepared[
            :PHASE3_CLASSIFY_K
        ]:

            tier, weight, source, cached = (
                classifier.classify(
                    evidence
                )
            )

            tier = str(
                tier
            ).upper()

            if tier not in TIER_WEIGHTS:

                raise RuntimeError(
                    f"Invalid Phase 3 tier: "
                    f"{tier}"
                )

            evidence[
                "predicted_tier"
            ] = tier

            evidence[
                "hierarchy_weight"
            ] = float(
                weight
            )

            evidence[
                "weighted_score"
            ] = (
                evidence[
                    "retrieval_score"
                ]
                *
                float(
                    weight
                )
            )

            evidence[
                "phase3_classifier_source"
            ] = source

            evidence[
                "phase3_cached"
            ] = bool(
                cached
            )

        classified = prepared[
            :PHASE3_CLASSIFY_K
        ]

        weighted = sorted(
            classified,
            key=lambda x:
                x[
                    "weighted_score"
                ],
            reverse=True,
        )[
            :FINAL_EVIDENCE_K
        ]

        unweighted = sorted(
            classified,
            key=lambda x:
                x[
                    "retrieval_score"
                ],
            reverse=True,
        )[
            :FINAL_EVIDENCE_K
        ]

        seen = set()

        candidates = []

        for evidence in (
            weighted
            + unweighted
        ):

            identifier = str(
                evidence.get(
                    "id",
                    evidence.get(
                        "source_id",
                        (
                            evidence.get(
                                "title",
                                ""
                            )
                            + evidence.get(
                                "text",
                                ""
                            )
                        ),
                    ),
                )
            )

            if identifier in seen:
                continue

            seen.add(
                identifier
            )

            candidates.append(
                evidence
            )

        output_rows.append(
            {
                "index":
                    row["index"],

                "claim":
                    claim,

                "true_label":
                    row[
                        "true_label"
                    ],

                "true_label_name":
                    row[
                        "true_label_name"
                    ],

                "weighted_top3":
                    weighted,

                "unweighted_top3":
                    unweighted,

                "candidates":
                    candidates,
            }
        )

        atomic_jsonl(
            FINAL_RETRIEVAL_CACHE,
            output_rows,
        )

    del bm25
    del reranker
    del classifier

    cleanup_torch()

    return output_rows


# ============================================================
# EVIDENCE HELPERS
# ============================================================

def evidence_text(
    evidence: dict,
) -> str:

    title = str(
        evidence.get(
            "title",
            "",
        )
        or ""
    ).strip()

    text = str(
        evidence.get(
            "text",
            evidence.get(
                "abstract",
                "",
            ),
        )
        or ""
    ).strip()

    if title and text:

        return (
            title
            + ". "
            + text
        )

    if text:
        return text

    if title:
        return title

    return (
        "[NO_EVIDENCE_AVAILABLE]"
    )


def evidence_id(
    evidence: dict,
) -> str:

    for key in [
        "id",
        "source_id",
        "doc_id",
        "pmid",
    ]:

        value = evidence.get(
            key
        )

        if value:

            return (
                f"{key}:"
                f"{value}"
            )

    text = evidence_text(
        evidence
    )

    return (
        "hash:"
        +
        hashlib.sha256(
            text.encode(
                "utf-8"
            )
        ).hexdigest()[
            :16
        ]
    )


def merged_candidates_from_dev(
    row: dict,
) -> list[dict]:

    seen = set()

    output = []

    for evidence in (
        row[
            "weighted_top3"
        ]
        +
        row[
            "unweighted_top3"
        ]
    ):

        identifier = evidence_id(
            evidence
        )

        if identifier in seen:
            continue

        seen.add(
            identifier
        )

        output.append(
            dict(
                evidence
            )
        )

    return output


# ============================================================
# PAIR FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def phase4_batch_predict(
    model,
    tokenizer,
    target_device,
    claims: list[str],
    evidences: list[str],
    batch_size: int = 16,
) -> list[list[float]]:

    output = []

    for start in range(
        0,
        len(claims),
        batch_size,
    ):

        end = start + batch_size

        encoded = tokenizer(
            claims[
                start:end
            ],
            evidences[
                start:end
            ],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        encoded = {
            key:
                value.to(
                    target_device
                )
            for key, value
            in encoded.items()
        }

        logits = model(
            **encoded
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        for probability in (
            probabilities
        ):

            output.append(
                [
                    float(x)
                    for x
                    in probability
                ]
            )

    return output


def build_feature_cache(
    source_rows: list[dict],
    output_file: Path,
    model,
    tokenizer,
    nli_model,
    nli_tokenizer,
    nli_mapping,
    target_device,
    development: bool,
) -> list[dict]:

    if output_file.exists():

        rows = load_jsonl(
            output_file
        )

        print(
            f"Using feature cache: "
            f"{output_file}"
        )

        return rows

    claims = []

    evidence_texts = []

    metadata = []

    for row in source_rows:

        if development:

            candidates = (
                merged_candidates_from_dev(
                    row
                )
            )

            gold = int(
                row[
                    "true_label"
                ]
            )

        else:

            candidates = row[
                "candidates"
            ]

            gold = int(
                row[
                    "true_label"
                ]
            )

        for evidence in candidates:

            claims.append(
                row[
                    "claim"
                ]
            )

            evidence_texts.append(
                evidence_text(
                    evidence
                )
            )

            metadata.append(
                {
                    "claim_index":
                        int(
                            row[
                                "index"
                            ]
                        ),

                    "claim":
                        row[
                            "claim"
                        ],

                    "true_label":
                        gold,

                    "evidence":
                        evidence,
                }
            )

    print(
        f"Computing Phase4 probabilities "
        f"for {len(metadata)} pairs..."
    )

    p4_probabilities = (
        phase4_batch_predict(
            model,
            tokenizer,
            target_device,
            claims,
            evidence_texts,
        )
    )

    print(
        "Computing biomedical "
        "NLI probabilities..."
    )

    nli_probabilities = nli_batch(
        nli_model,
        nli_tokenizer,
        nli_mapping,
        target_device,
        evidence_texts,
        claims,
    )

    grouped = {}

    for item, p4, nli in zip(
        metadata,
        p4_probabilities,
        nli_probabilities,
    ):

        index = item[
            "claim_index"
        ]

        evidence = item[
            "evidence"
        ]

        if index not in grouped:

            grouped[index] = {
                "index":
                    index,

                "claim":
                    item[
                        "claim"
                    ],

                "true_label":
                    item[
                        "true_label"
                    ],

                "pairs":
                    [],
            }

        retrieval_score = safe_float(
            evidence.get(
                "retrieval_score",
                evidence.get(
                    "score",
                    0.5,
                ),
            ),
            0.5,
        )

        hierarchy_weight = safe_float(
            evidence.get(
                "hierarchy_weight",
                evidence.get(
                    "tier_weight",
                    0.2,
                ),
            ),
            0.2,
        )

        grouped[index][
            "pairs"
        ].append(
            {
                "evidence_id":
                    evidence_id(
                        evidence
                    ),

                "title":
                    str(
                        evidence.get(
                            "title",
                            "",
                        )
                    ),

                "retrieval_score":
                    retrieval_score,

                "hierarchy_weight":
                    hierarchy_weight,

                "phase4":
                    p4,

                "nli":
                    nli,
            }
        )

    output_rows = [
        grouped[index]
        for index
        in sorted(
            grouped.keys()
        )
    ]

    atomic_jsonl(
        output_file,
        output_rows,
    )

    return output_rows


# ============================================================
# FUSION MODEL
# ============================================================

def normalize_probabilities(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        1e-12,
        None,
    )

    return (
        values
        /
        values.sum()
    )


def fuse_claim(
    row: dict,
    config: dict,
) -> np.ndarray:

    pair_vectors = []

    weights = []

    for pair in row[
        "pairs"
    ]:

        p4 = normalize_probabilities(
            pair[
                "phase4"
            ]
        )

        nli = pair[
            "nli"
        ]

        nli_vector = np.asarray(
            [
                nli[
                    "contradiction"
                ],

                nli[
                    "neutral"
                ],

                nli[
                    "entailment"
                ],
            ],
            dtype=float,
        )

        nli_vector = (
            nli_vector
            /
            nli_vector.sum()
        )

        alpha = config[
            "phase4_weight"
        ]

        vector = (
            alpha
            * p4
            +
            (
                1.0
                - alpha
            )
            * nli_vector
        )

        vector[0] *= (
            1.0
            +
            config[
                "relation_boost"
            ]
            * nli[
                "contradiction"
            ]
        )

        vector[1] *= (
            1.0
            +
            config[
                "neutral_boost"
            ]
            * nli[
                "neutral"
            ]
        )

        vector[2] *= (
            1.0
            +
            config[
                "relation_boost"
            ]
            * nli[
                "entailment"
            ]
        )

        vector = (
            vector
            /
            vector.sum()
        )

        retrieval = max(
            0.0,
            min(
                1.0,
                pair[
                    "retrieval_score"
                ],
            ),
        )

        authority = max(
            0.0,
            min(
                1.0,
                pair[
                    "hierarchy_weight"
                ],
            ),
        )

        quality = math.sqrt(
            (
                0.6
                +
                0.4
                * retrieval
            )
            *
            (
                0.6
                +
                0.4
                * authority
            )
        )

        pair_vectors.append(
            vector
        )

        weights.append(
            quality
        )

    stacked = np.vstack(
        pair_vectors
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights /= (
        weights.sum()
    )

    consensus = (
        stacked
        *
        weights[
            :,
            None
        ]
    ).sum(
        axis=0
    )

    class_max = np.max(
        stacked,
        axis=0,
    )

    max_weight = config[
        "class_max_weight"
    ]

    final = (
        (
            1.0
            - max_weight
        )
        * consensus
        +
        max_weight
        * class_max
    )

    final = (
        final
        /
        final.sum()
    )

    return final


# ============================================================
# DEV HYPERPARAMETER SELECTION
# ============================================================

def select_fusion_config(
    dev_features: list[dict],
) -> dict:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SELECTING FUSION ON "
        "DEVELOPMENT 90 ONLY"
    )

    print(
        "=" * 80
    )

    phase4_weights = [
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
    ]

    relation_boosts = [
        0.0,
        0.15,
        0.30,
        0.45,
    ]

    neutral_boosts = [
        0.0,
        0.10,
        0.20,
        0.30,
    ]

    class_max_weights = [
        0.0,
        0.15,
        0.30,
    ]

    y_true = [
        int(
            row[
                "true_label"
            ]
        )
        for row
        in dev_features
    ]

    candidates = []

    for phase4_weight in (
        phase4_weights
    ):

        for relation_boost in (
            relation_boosts
        ):

            for neutral_boost in (
                neutral_boosts
            ):

                for class_max_weight in (
                    class_max_weights
                ):

                    config = {
                        "phase4_weight":
                            phase4_weight,

                        "relation_boost":
                            relation_boost,

                        "neutral_boost":
                            neutral_boost,

                        "class_max_weight":
                            class_max_weight,
                    }

                    predictions = [
                        int(
                            np.argmax(
                                fuse_claim(
                                    row,
                                    config,
                                )
                            )
                        )
                        for row
                        in dev_features
                    ]

                    result = metrics(
                        y_true,
                        predictions,
                    )

                    candidates.append(
                        {
                            "config":
                                config,

                            "metrics":
                                result,
                        }
                    )

    candidates.sort(
        key=lambda x: (
            x[
                "metrics"
            ][
                "macro_f1"
            ],
            x[
                "metrics"
            ][
                "accuracy"
            ],
        ),
        reverse=True,
    )

    best = candidates[0]

    atomic_json(
        TUNING_FILE,
        {
            "best":
                best,

            "top_10":
                candidates[:10],

            "selection_metric":
                (
                    "macro_f1 then accuracy"
                ),

            "selection_data":
                (
                    "90 previously-used "
                    "development claims"
                ),
        },
    )

    print(
        "BEST DEVELOPMENT CONFIG:"
    )

    print(
        json.dumps(
            best,
            indent=2,
        )
    )

    return best[
        "config"
    ]


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_final(
    final_features: list[dict],
    config: dict,
) -> tuple[
    dict,
    list[dict],
]:

    y_true = []

    y_pred = []

    predictions = []

    for row in final_features:

        probability = fuse_claim(
            row,
            config,
        )

        prediction = int(
            np.argmax(
                probability
            )
        )

        gold = int(
            row[
                "true_label"
            ]
        )

        y_true.append(
            gold
        )

        y_pred.append(
            prediction
        )

        predictions.append(
            {
                "index":
                    row[
                        "index"
                    ],

                "claim":
                    row[
                        "claim"
                    ],

                "true_label":
                    gold,

                "true_label_name":
                    LABEL_NAMES[
                        gold
                    ],

                "predicted_label":
                    prediction,

                "predicted_label_name":
                    LABEL_NAMES[
                        prediction
                    ],

                "correct":
                    bool(
                        prediction
                        == gold
                    ),

                "probabilities": {
                    "REFUTED":
                        float(
                            probability[0]
                        ),

                    "NEI":
                        float(
                            probability[1]
                        ),

                    "SUPPORTED":
                        float(
                            probability[2]
                        ),
                },

                "evidence":
                    row[
                        "pairs"
                    ],
            }
        )

    return (
        metrics(
            y_true,
            y_pred,
        ),
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        SEED
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_device = device()

    print(
        "=" * 80
    )

    print(
        "MASTER RETRIEVAL-CONDITIONED "
        "END-TO-END EXPERIMENT"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {target_device}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The old 90 claims are now "
        "DEVELOPMENT data."
    )

    print(
        "The final balanced subset will "
        "exclude all of them."
    )

    # --------------------------------------------------------
    # Load Phase 4 splits
    # --------------------------------------------------------

    train_file = discover_split_file(
        "train"
    )

    val_file = discover_split_file(
        "val"
    )

    test_file = discover_split_file(
        "test"
    )

    train_rows = canonicalize_rows(
        read_table(
            train_file
        )
    )

    val_rows = canonicalize_rows(
        read_table(
            val_file
        )
    )

    test_rows = canonicalize_rows(
        read_table(
            test_file
        )
    )

    print(
        f"\nTrain rows: "
        f"{len(train_rows)}"
    )

    print(
        f"Validation rows: "
        f"{len(val_rows)}"
    )

    print(
        f"Test rows: "
        f"{len(test_rows)}"
    )

    # --------------------------------------------------------
    # Hard negatives
    # --------------------------------------------------------

    hard_negatives = (
        generate_hard_negatives(
            train_rows,
            target_device,
        )
    )

    # --------------------------------------------------------
    # Train new verifier
    # --------------------------------------------------------

    train_retrieval_conditioned_model(
        train_rows,
        val_rows,
        hard_negatives,
        target_device,
    )

    # --------------------------------------------------------
    # Freeze fresh final claims BEFORE tuning
    # --------------------------------------------------------

    fresh_final = (
        create_fresh_final_subset(
            test_rows
        )
    )

    # --------------------------------------------------------
    # Retrieve Phase2 + rank Phase3
    # for untouched final data
    # --------------------------------------------------------

    fresh_retrieval = (
        build_fresh_retrieval_cache(
            fresh_final
        )
    )

    # --------------------------------------------------------
    # Load newly trained verifier
    # + biomedical NLI
    # --------------------------------------------------------

    print(
        "\nLoading retrieval-conditioned "
        "verifier..."
    )

    (
        verifier,
        verifier_tokenizer,
        verifier_model_name,
        verifier_checkpoint,
    ) = load_model3_checkpoint(
        NEW_CHECKPOINT,
        target_device,
    )

    verifier.eval()

    print(
        "Loading biomedical NLI..."
    )

    (
        nli_model,
        nli_tokenizer,
        nli_mapping,
    ) = load_biomedical_nli(
        target_device
    )

    # --------------------------------------------------------
    # Development feature extraction
    # --------------------------------------------------------

    dev90 = load_jsonl(
        DEV_90_FILE
    )

    dev_features = (
        build_feature_cache(
            dev90,
            DEV_FEATURE_CACHE,
            verifier,
            verifier_tokenizer,
            nli_model,
            nli_tokenizer,
            nli_mapping,
            target_device,
            development=True,
        )
    )

    # --------------------------------------------------------
    # Legitimately tune fusion ON DEV ONLY
    # --------------------------------------------------------

    best_config = (
        select_fusion_config(
            dev_features
        )
    )

    # --------------------------------------------------------
    # Fresh final feature extraction
    # --------------------------------------------------------

    final_features = (
        build_feature_cache(
            fresh_retrieval,
            FINAL_FEATURE_CACHE,
            verifier,
            verifier_tokenizer,
            nli_model,
            nli_tokenizer,
            nli_mapping,
            target_device,
            development=False,
        )
    )

    # --------------------------------------------------------
    # ONE final untouched evaluation
    # --------------------------------------------------------

    final_metrics, predictions = (
        evaluate_final(
            final_features,
            best_config,
        )
    )

    atomic_jsonl(
        FINAL_PREDICTIONS_FILE,
        predictions,
    )

    atomic_json(
        FINAL_METRICS_FILE,
        {
            "metrics":
                final_metrics,

            "frozen_fusion_config":
                best_config,

            "verifier_checkpoint":
                str(
                    NEW_CHECKPOINT
                ),

            "fresh_subset":
                str(
                    FINAL_SUBSET_FILE
                ),

            "fresh_subset_sha256":
                sha256_file(
                    FINAL_SUBSET_FILE
                ),
        },
    )

    audit = {
        "development_set":
            (
                "The original 90 balanced "
                "PUBHEALTH claims used for "
                "V1/V2/V3."
            ),

        "final_set":
            (
                "New balanced PUBHEALTH "
                "claims excluding all "
                "development claims."
            ),

        "hard_negative_generation":
            (
                "Lexically similar evidence "
                "filtered using biomedical NLI "
                "for neutral relation."
            ),

        "phase2_final":
            (
                "BM25 Top50 followed by "
                "cross-encoder reranking Top10."
            ),

        "phase3_final":
            (
                "Qwen 2.5 7B evidence hierarchy "
                "classification on Top5; "
                "authority-weighted selection."
            ),

        "phase4_final":
            (
                "Retrieval-conditioned PubMedBERT "
                "initialized from Model3."
            ),

        "relation_model":
            BIOMEDICAL_NLI_MODEL,

        "fusion_selected_on":
            "development 90 only",

        "final_labels_used_for_tuning":
            False,

        "target_accuracy":
            0.85,

        "target_is_guaranteed":
            False,
    }

    atomic_json(
        AUDIT_FILE,
        audit,
    )

    # --------------------------------------------------------
    # Human report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "FINAL UNTOUCHED "
        "END-TO-END VERIFICATION RESULTS"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        f"N: "
        f"{final_metrics['n']}"
    )

    lines.append(
        f"Accuracy: "
        f"{final_metrics['accuracy']:.4f}"
    )

    lines.append(
        f"Macro-F1: "
        f"{final_metrics['macro_f1']:.4f}"
    )

    lines.append(
        f"Weighted-F1: "
        f"{final_metrics['weighted_f1']:.4f}"
    )

    lines.append("")

    lines.append(
        "Prediction distribution:"
    )

    lines.append(
        str(
            final_metrics[
                "prediction_distribution"
            ]
        )
    )

    lines.append("")

    lines.append(
        "Per-class:"
    )

    for name in [
        "REFUTED",
        "NEI",
        "SUPPORTED",
    ]:

        values = (
            final_metrics[
                "per_class"
            ][
                name
            ]
        )

        lines.append(
            f"{name:10s} "
            f"P="
            f"{values['precision']:.4f} "
            f"R="
            f"{values['recall']:.4f} "
            f"F1="
            f"{values['f1']:.4f} "
            f"N="
            f"{values['support']}"
        )

    lines.append("")

    lines.append(
        "Confusion Matrix "
        "(rows actual, columns predicted)"
    )

    for matrix_row in (
        final_metrics[
            "confusion_matrix"
        ]
    ):

        lines.append(
            str(
                matrix_row
            )
        )

    lines.append("")

    lines.append(
        "Frozen fusion configuration:"
    )

    lines.append(
        json.dumps(
            best_config,
            indent=2,
        )
    )

    atomic_text(
        FINAL_REPORT_FILE,
        "\n".join(
            lines
        )
        + "\n",
    )

    # --------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL UNTOUCHED "
        "END-TO-END RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"N: "
        f"{final_metrics['n']}"
    )

    print(
        f"Accuracy:    "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1:    "
        f"{final_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1: "
        f"{final_metrics['weighted_f1']:.4f}"
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        final_metrics[
            "prediction_distribution"
        ]
    )

    print(
        "\nPer-class metrics:"
    )

    for name in [
        "REFUTED",
        "NEI",
        "SUPPORTED",
    ]:

        values = (
            final_metrics[
                "per_class"
            ][
                name
            ]
        )

        print(
            f"{name:10s} "
            f"P="
            f"{values['precision']:.4f} "
            f"R="
            f"{values['recall']:.4f} "
            f"F1="
            f"{values['f1']:.4f} "
            f"N="
            f"{values['support']}"
        )

    print(
        "\nConfusion Matrix:"
    )

    for matrix_row in (
        final_metrics[
            "confusion_matrix"
        ]
    ):

        print(
            matrix_row
        )

    print(
        "\nFinal frozen fusion config:"
    )

    print(
        best_config
    )

    print(
        "\nOutput:"
    )

    print(
        FINAL_REPORT_FILE
    )

    print(
        FINAL_METRICS_FILE
    )

    print(
        FINAL_PREDICTIONS_FILE
    )

    print(
        AUDIT_FILE
    )

    print(
        "\nTARGET CHECK:"
    )

    if (
        final_metrics[
            "accuracy"
        ]
        >= 0.85
    ):

        print(
            "✓ Accuracy target >= 0.85 achieved."
        )

    else:

        print(
            "✗ Accuracy target >= 0.85 "
            "was NOT achieved."
        )

    print(
        "\nExperiment complete."
    )


if __name__ == "__main__":
    main()
