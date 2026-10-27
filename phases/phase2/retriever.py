"""
Advanced retrieval module for SciFact.

Retrievers:
    1. BM25
    2. BioBERT + FAISS
    3. Hybrid BM25 + BioBERT using Reciprocal Rank Fusion (RRF)

Designed for scientific claim -> evidence retrieval.
"""

import os
import pickle
import re

import faiss
import numpy as np
import torch

from rank_bm25 import BM25Okapi

from config import (
    BIOMODEL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    TOP_K_RETRIEVE,
)

from build_faiss_index import BioBERTEncoder


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize_scientific_text(text):
    """
    Scientific-aware tokenizer.

    Keeps:
        words
        numbers
        decimals
        hyphenated terms
        gene/protein names
        abbreviations

    Example:
        "IL-6 inhibits TNF-α in 3.5-fold experiments"

    becomes approximately:
        ["il-6", "inhibits", "tnf", "α", "3.5", "fold", ...]
    """

    text = str(text).lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Preserve scientific words / hyphenated expressions
    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*"
        r"|[α-ωΑ-Ω]"
        r"|[+-]"
        r"|[0-9]+(?:\.[0-9]+)?",
        text,
    )

    return tokens


# ============================================================
# BM25
# ============================================================

class BM25Retriever:
    """
    Strong lexical baseline for SciFact.

    Uses scientific-aware tokenization and BM25Okapi.
    """

    def __init__(self):

        from build_faiss_index import load_faiss_index

        _, metadata = load_faiss_index()

        self.docs = metadata["abstracts"]

        self.corpus_texts = [
            d["text"]
            for d in self.docs
        ]

        print(
            f"Building BM25 index over "
            f"{len(self.docs)} documents..."
        )

        self.tokenized_corpus = [
            tokenize_scientific_text(text)
            for text in self.corpus_texts
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print(
            f"BM25 retriever ready with "
            f"{len(self.docs)} documents."
        )

    def search(
        self,
        query,
        k=TOP_K_RETRIEVE,
    ):

        tokenized_query = tokenize_scientific_text(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        results = []

        for idx in top_indices:

            doc = self.docs[idx]

            results.append(
                {
                    "id": str(doc["id"]),
                    "title": doc["title"],
                    "text": doc["text"],
                    "source": doc.get(
                        "source",
                        "unknown",
                    ),
                    "score": float(
                        scores[idx]
                    ),
                }
            )

        return results


# ============================================================
# BIOBERT + FAISS
# ============================================================

class BioBERTRetriever:
    """
    Dense retrieval using BioBERT + FAISS.

    BioBERT embeddings are L2-normalized,
    so FAISS IndexFlatIP performs cosine similarity.
    """

    _instance = None

    def __new__(
        cls,
        *args,
        **kwargs,
    ):

        if cls._instance is None:

            cls._instance = super().__new__(
                cls
            )

            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        device=None,
    ):

        if self._initialized:
            return

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        print(
            "Initializing BioBERT Retriever..."
        )

        self.encoder = BioBERTEncoder(
            BIOMODEL,
            device=device,
        )

        self.index, self.metadata = (
            self._load_faiss()
        )

        self.documents = (
            self.metadata["abstracts"]
        )

        self._initialized = True

        print(
            "BioBERT retriever ready."
        )

    def _load_faiss(self):

        if not os.path.exists(
            FAISS_INDEX_PATH
        ):

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{FAISS_INDEX_PATH}"
            )

        index = faiss.read_index(
            FAISS_INDEX_PATH
        )

        with open(
            FAISS_META_PATH,
            "rb",
        ) as f:

            metadata = pickle.load(f)

        return index, metadata

    def search(
        self,
        query,
        k=TOP_K_RETRIEVE,
    ):

        query_embedding = self.encoder.encode(
            [query],
            batch_size=1,
        )

        distances, indices = (
            self.index.search(
                query_embedding,
                k,
            )
        )

        results = []

        for dist, idx in zip(
            distances[0],
            indices[0],
        ):

            if idx < 0:
                continue

            if idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            results.append(
                {
                    "id": str(doc["id"]),
                    "title": doc["title"],
                    "text": doc["text"],
                    "source": doc.get(
                        "source",
                        "unknown",
                    ),
                    "score": float(dist),
                }
            )

        return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

class HybridRetriever:
    """
    Hybrid BM25 + BioBERT retriever.

    Uses Reciprocal Rank Fusion (RRF).

    RRF score:

        score(d) =
            1 / (rrf_k + rank_bm25)
            +
            1 / (rrf_k + rank_biobert)

    Documents appearing in both rankings
    receive particularly strong scores.
    """

    def __init__(
        self,
        candidate_k=50,
        rrf_k=60,
    ):

        print("\n" + "=" * 70)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 70)

        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever()

        self.biobert = BioBERTRetriever()

        print(
            "\nHybrid retriever ready."
        )

    def search(
        self,
        query,
        k=10,
    ):

        # ----------------------------------------------------
        # Retrieve candidates
        # ----------------------------------------------------

        bm25_results = self.bm25.search(
            query,
            k=self.candidate_k,
        )

        biobert_results = self.biobert.search(
            query,
            k=self.candidate_k,
        )

        # ----------------------------------------------------
        # Store documents
        # ----------------------------------------------------

        documents = {}

        for result in bm25_results:

            documents[
                str(result["id"])
            ] = result

        for result in biobert_results:

            documents[
                str(result["id"])
            ] = result

        # ----------------------------------------------------
        # RRF scoring
        # ----------------------------------------------------

        rrf_scores = {}

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            doc_id = str(result["id"])

            rrf_scores[doc_id] = (
                rrf_scores.get(
                    doc_id,
                    0.0,
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        # BioBERT ranking
        for rank, result in enumerate(
            biobert_results,
            start=1,
        ):

            doc_id = str(result["id"])

            rrf_scores[doc_id] = (
                rrf_scores.get(
                    doc_id,
                    0.0,
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        # ----------------------------------------------------
        # Build final results
        # ----------------------------------------------------

        results = []

        for doc_id in ranked_ids[:k]:

            doc = documents[doc_id]

            results.append(
                {
                    "id": doc_id,
                    "title": doc["title"],
                    "text": doc["text"],
                    "source": doc.get(
                        "source",
                        "unknown",
                    ),
                    "score": float(
                        rrf_scores[doc_id]
                    ),
                }
            )

        return results