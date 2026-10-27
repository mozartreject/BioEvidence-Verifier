from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


LABEL_TO_ID = {
    "REFUTED": 0,
    "NEI": 1,
    "SUPPORTED": 2,
}


ID_TO_LABEL = {
    0: "REFUTED",
    1: "NEI",
    2: "SUPPORTED",
}


class Phase4Dataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        claim_max_length: int = 128,
        evidence_max_length: int = 256,
    ) -> None:

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}"
            )

        self.tokenizer = tokenizer
        self.claim_max_length = claim_max_length
        self.evidence_max_length = evidence_max_length

        self.examples = self._load_jsonl()

        if not self.examples:
            raise ValueError(
                f"Dataset is empty: {self.path}"
            )

        self._validate()


    def _load_jsonl(
        self,
    ) -> List[Dict[str, Any]]:

        examples = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    example = json.loads(line)

                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {self.path}, "
                        f"line {line_number}: {error}"
                    )

                if not isinstance(example, dict):
                    raise TypeError(
                        f"Line {line_number} in {self.path} "
                        "is not a JSON object."
                    )

                examples.append(example)

        return examples


    def _validate(
        self,
    ) -> None:

        sample_ids = set()

        required_fields = {
            "sample_id",
            "dataset",
            "claim_text",
            "evidence_text",
            "verdict",
            "label",
        }

        for index, example in enumerate(
            self.examples
        ):

            missing = (
                required_fields
                - set(example.keys())
            )

            if missing:
                raise ValueError(
                    f"Example {index} missing fields: "
                    f"{sorted(missing)}"
                )

            sample_id = example["sample_id"]

            if not isinstance(sample_id, str):
                raise TypeError(
                    f"sample_id must be string at "
                    f"example {index}"
                )

            if sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: {sample_id}"
                )

            sample_ids.add(sample_id)

            label = example["label"]

            if label not in {0, 1, 2}:
                raise ValueError(
                    f"Invalid label {label} "
                    f"for {sample_id}"
                )

            expected_verdict = ID_TO_LABEL[
                label
            ]

            if example["verdict"] != expected_verdict:
                raise ValueError(
                    f"Label/verdict mismatch for "
                    f"{sample_id}. "
                    f"label={label}, "
                    f"verdict={example['verdict']}"
                )

            claim = str(
                example["claim_text"]
            ).strip()

            if not claim:
                raise ValueError(
                    f"Empty claim for {sample_id}"
                )


    def __len__(
        self,
    ) -> int:

        return len(self.examples)


    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        example = self.examples[index]

        claim = str(
            example["claim_text"]
        ).strip()

        evidence = str(
            example.get(
                "evidence_text",
                "",
            )
        ).strip()

        claim_tokens = self.tokenizer(
            claim,
            max_length=self.claim_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        evidence_tokens = self.tokenizer(
            evidence,
            max_length=self.evidence_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "sample_id": example["sample_id"],
            "dataset": example["dataset"],

            "claim_input_ids":
                claim_tokens[
                    "input_ids"
                ].squeeze(0),

            "claim_attention_mask":
                claim_tokens[
                    "attention_mask"
                ].squeeze(0),

            "evidence_input_ids":
                evidence_tokens[
                    "input_ids"
                ].squeeze(0),

            "evidence_attention_mask":
                evidence_tokens[
                    "attention_mask"
                ].squeeze(0),

            "label":
                torch.tensor(
                    int(example["label"]),
                    dtype=torch.long,
                ),
        }

        if (
            "token_type_ids"
            in claim_tokens
        ):

            item[
                "claim_token_type_ids"
            ] = claim_tokens[
                "token_type_ids"
            ].squeeze(0)

        if (
            "token_type_ids"
            in evidence_tokens
        ):

            item[
                "evidence_token_type_ids"
            ] = evidence_tokens[
                "token_type_ids"
            ].squeeze(0)

        return item


def build_tokenizer(
    model_name: str,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:

        if tokenizer.eos_token is not None:
            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        else:
            raise ValueError(
                "Tokenizer has no pad token "
                "or eos token."
            )

    return tokenizer