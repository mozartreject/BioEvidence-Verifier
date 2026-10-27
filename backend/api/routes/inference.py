from __future__ import annotations
from typing import Optional

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.api.dependencies import (
    get_inference_lock,
    get_pipeline,
)

from backend.services.persistence import (
    DuplicateRequestError,
)

from backend.schemas import (
    DecisionPackage,
    InferenceRequest,
)


router = APIRouter(
    prefix="/inferences",
    tags=["inference"],
)


# ============================================================
# MULTIMODAL UPLOAD CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

UPLOAD_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_IMAGE_BYTES = (
    15 * 1024 * 1024
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def _safe_request_id(
    request_id: str,
) -> str:
    safe = "".join(
        char
        if (
            char.isalnum()
            or char in "-_"
        )
        else "_"
        for char in request_id
    )

    safe = safe.strip("_")

    if not safe:
        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "INVALID_REQUEST_ID",

                "message":
                    "request_id contains no "
                    "usable characters.",
            },
        )

    return safe


async def _save_uploaded_image(
    image: UploadFile,
    request_id: str,
) -> Path:
    content_type = (
        image.content_type
        or ""
    ).lower()

    if (
        content_type
        not in ALLOWED_IMAGE_TYPES
    ):
        raise HTTPException(
            status_code=415,
            detail={
                "error":
                    "UNSUPPORTED_IMAGE_TYPE",

                "message":
                    "Image must be PNG, "
                    "JPEG or WEBP.",
            },
        )

    data = await image.read(
        MAX_IMAGE_BYTES + 1
    )

    if not data:
        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "EMPTY_IMAGE",

                "message":
                    "Uploaded image is empty.",
            },
        )

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error":
                    "IMAGE_TOO_LARGE",

                "message":
                    "Image exceeds the "
                    "15 MB upload limit.",
            },
        )

    extension = (
        ALLOWED_IMAGE_TYPES[
            content_type
        ]
    )

    safe_id = _safe_request_id(
        request_id
    )

    destination = (
        UPLOAD_DIR
        / f"{safe_id}{extension}"
    )

    try:
        with destination.open(
            "wb"
        ) as handle:
            handle.write(data)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "IMAGE_SAVE_FAILED",

                "message":
                    f"Unable to save uploaded "
                    f"image: {exc}",
            },
        ) from exc

    return destination.resolve()


def _load_json_file(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error":
                    "RECORD_NOT_FOUND",

                "message":
                    f"Record not found: "
                    f"{path.name}",
            },
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            value = json.load(
                handle
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "PERSISTED_RECORD_INVALID",

                "message":
                    f"Unable to read "
                    f"{path.name}: {exc}",
            },
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "PERSISTED_RECORD_INVALID",

                "message":
                    "Persisted JSON root "
                    "must be an object.",
            },
        )

    return value


# ============================================================
# CREATE INFERENCE
# ============================================================

@router.post(
    "",
    response_model=DecisionPackage,
    status_code=(
        status.HTTP_201_CREATED
    ),
    summary=(
        "Run biomedical verification"
    ),
)
async def create_inference(
    request_id: str = Form(...),
    claim: str = Form(...),
    image: Optional[UploadFile] = File(
        default=None
    ),
) -> DecisionPackage:

    request_id = request_id.strip()
    claim = claim.strip()

    if not request_id:
        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "INVALID_REQUEST_ID",

                "message":
                    "request_id cannot "
                    "be empty.",
            },
        )

    if len(request_id) > 200:
        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "INVALID_REQUEST_ID",

                "message":
                    "request_id exceeds "
                    "200 characters.",
            },
        )

    if len(claim) < 3:
        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "INVALID_CLAIM",

                "message":
                    "Claim must contain at "
                    "least 3 characters.",
            },
        )

    if len(claim) > 5000:
        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "INVALID_CLAIM",

                "message":
                    "Claim exceeds "
                    "5000 characters.",
            },
        )

    pipeline = get_pipeline()

    image_path: Path | None = None

    if image is not None:
        image_path = (
            await _save_uploaded_image(
                image,
                request_id,
            )
        )

        print()
        print(
            "Multimodal image received:"
        )
        print(
            image_path
        )

    request = InferenceRequest(
        request_id=request_id,
        claim=claim,
        image_path=(
            str(image_path)
            if image_path
            else None
        ),
        created_at_utc=(
            datetime.now(
                timezone.utc
            )
        ),
    )

    lock = get_inference_lock()

    async with lock:
        try:
            package = (
                await asyncio.to_thread(
                    pipeline.run,
                    request,
                )
            )

        except DuplicateRequestError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error":
                        "DUPLICATE_REQUEST_ID",

                    "message":
                        str(exc),
                },
            ) from exc

        except RuntimeError as exc:
            cause = exc.__cause__

            if isinstance(
                cause,
                DuplicateRequestError,
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error":
                            "DUPLICATE_REQUEST_ID",

                        "message":
                            str(cause),
                    },
                ) from exc

            raise HTTPException(
                status_code=500,
                detail={
                    "error":
                        "INFERENCE_FAILED",

                    "message":
                        str(exc),
                },
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error":
                        "INFERENCE_FAILED",

                    "message":
                        str(exc),
                },
            ) from exc

        finally:
            if image is not None:
                await image.close()

    return package


# ============================================================
# GET PERSISTED INFERENCE
# ============================================================

@router.get(
    "/{request_id}",
    response_model=DecisionPackage,
    summary="Get persisted inference",
)
def get_inference(
    request_id: str,
) -> DecisionPackage:
    pipeline = get_pipeline()

    try:
        (
            inference_path,
            _,
        ) = pipeline.persistence._paths(
            request_id
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "INVALID_REQUEST_ID",

                "message":
                    str(exc),
            },
        ) from exc

    raw = _load_json_file(
        inference_path
    )

    try:
        return (
            DecisionPackage
            .model_validate(
                raw
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error":
                    "PERSISTED_RECORD_SCHEMA_ERROR",

                "message":
                    str(exc),
            },
        ) from exc


# ============================================================
# GET AUDIT RECORD
# ============================================================

@router.get(
    "/{request_id}/audit",
    summary=(
        "Get persisted audit record"
    ),
)
def get_audit(
    request_id: str,
) -> Dict[str, Any]:
    pipeline = get_pipeline()

    try:
        (
            _,
            audit_path,
        ) = pipeline.persistence._paths(
            request_id
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "INVALID_REQUEST_ID",

                "message":
                    str(exc),
            },
        ) from exc

    return _load_json_file(
        audit_path
    )
