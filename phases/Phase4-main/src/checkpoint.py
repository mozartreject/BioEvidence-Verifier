from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def load_torch_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=map_location,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint root to be a dictionary."
        )

    return checkpoint


def get_model_state_dict(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    state_dict = checkpoint["model_state_dict"]

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "'model_state_dict' is not a mapping."
        )

    return state_dict


def extract_prefixed_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> OrderedDict[str, torch.Tensor]:

    prefix_with_dot = prefix + "."

    output = OrderedDict()

    for key, tensor in state_dict.items():

        if key.startswith(prefix_with_dot):

            stripped_key = key[
                len(prefix_with_dot):
            ]

            output[stripped_key] = tensor

    if not output:
        raise KeyError(
            f"No checkpoint parameters found "
            f"for prefix '{prefix}'."
        )

    return output


def validate_phase1_checkpoint(
    state_dict: Mapping[str, torch.Tensor],
) -> None:

    required_keys = {
        "no_image_token",
        "text_proj.weight",
        "text_proj.bias",
        "img_proj.weight",
        "img_proj.bias",
        "fusion.0.weight",
        "fusion.0.bias",
        "fusion.1.weight",
        "fusion.1.bias",
    }

    missing = (
        required_keys
        - set(state_dict.keys())
    )

    if missing:
        raise KeyError(
            "Phase 1 checkpoint is missing required "
            f"parameters: {sorted(missing)}"
        )

    expected_shapes = {
        "no_image_token": (768,),
        "text_proj.weight": (768, 768),
        "text_proj.bias": (768,),
        "img_proj.weight": (768, 768),
        "img_proj.bias": (768,),
        "fusion.0.weight": (768, 1536),
        "fusion.0.bias": (768,),
        "fusion.1.weight": (768,),
        "fusion.1.bias": (768,),
    }

    for key, expected_shape in expected_shapes.items():

        actual_shape = tuple(
            state_dict[key].shape
        )

        if actual_shape != expected_shape:

            raise ValueError(
                f"Unexpected shape for {key}: "
                f"{actual_shape}; expected "
                f"{expected_shape}"
            )


def checkpoint_summary(
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:

    state_dict = get_model_state_dict(
        checkpoint
    )

    parameter_elements = sum(
        tensor.numel()
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor)
    )

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "tensor_count": len(state_dict),
        "parameter_elements": parameter_elements,
    }
