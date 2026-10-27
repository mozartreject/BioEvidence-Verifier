from __future__ import annotations

import asyncio
from typing import Optional

from backend.pipeline import (
    BiomedicalVerificationPipeline,
)


_pipeline: Optional[
    BiomedicalVerificationPipeline
] = None

_inference_lock = asyncio.Lock()


def initialize_pipeline() -> (
    BiomedicalVerificationPipeline
):
    global _pipeline

    if _pipeline is None:
        _pipeline = (
            BiomedicalVerificationPipeline()
        )

    return _pipeline


def get_pipeline() -> (
    BiomedicalVerificationPipeline
):
    if _pipeline is None:
        raise RuntimeError(
            "BiomedicalVerificationPipeline "
            "has not been initialized."
        )

    return _pipeline


def shutdown_pipeline() -> None:
    global _pipeline

    _pipeline = None


def get_inference_lock() -> asyncio.Lock:
    return _inference_lock
