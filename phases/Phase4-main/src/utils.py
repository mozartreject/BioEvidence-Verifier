from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def get_device() -> torch.device:

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and
        torch.backends.mps.is_available()
    ):

        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def load_yaml(
    path: str | Path,
) -> Dict[str, Any]:

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):

        raise TypeError(
            "Configuration root "
            "must be a dictionary."
        )

    return config


def atomic_json_dump(
    obj: Any,
    path: str | Path,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temporary_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                obj,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

            file.write(
                "\n"
            )

            file.flush()

            os.fsync(
                file.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    except Exception:

        try:
            os.remove(
                temporary_path
            )

        except OSError:
            pass

        raise


def count_parameters(
    model: torch.nn.Module,
) -> Dict[str, int]:

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total":
            total,

        "trainable":
            trainable,

        "frozen":
            total - trainable,
    }