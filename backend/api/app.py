from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.api.dependencies import (
    initialize_pipeline,
    shutdown_pipeline,
)

from backend.api.routes.inference import (
    router as inference_router,
)

from backend.api.routes.system import (
    router as system_router,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    print()
    print("=" * 80)
    print(
        "BIOMEDICAL VERIFIER RESEARCH API"
    )
    print(
        "Initializing frozen inference engine..."
    )
    print("=" * 80)

    initialize_pipeline()

    print()
    print(
        "Research API pipeline: READY"
    )
    print("=" * 80)

    yield

    print()
    print(
        "BiomedicalVerifier API shutting down."
    )

    shutdown_pipeline()


app = FastAPI(
    title=(
        "BiomedicalVerifier Research API"
    ),
    description=(
        "Evidence-grounded biomedical "
        "misinformation verification API. "
        "The authoritative verdict is produced "
        "by the frozen Phase 4 verifier. "
        "Phase 5 explanations cannot modify "
        "that verdict."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Content-Type",
        "Accept",
    ],
)


app.include_router(
    system_router,
    prefix="/api/v1",
)

app.include_router(
    inference_router,
    prefix="/api/v1",
)


@app.get(
    "/",
    tags=["root"],
)
def root():
    return {
        "service":
            "BiomedicalVerifier Research API",

        "version":
            "0.1.0",

        "api":
            "/api/v1",

        "documentation":
            "/docs",

        "mode":
            "faithful",
    }
