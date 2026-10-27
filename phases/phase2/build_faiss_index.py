"""
Build FAISS index from abstracts.
Uses BioBERT with mean pooling + L2 normalization for cosine similarity.
"""

import os
import pickle
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from config import BIOMODEL, FAISS_INDEX_PATH, FAISS_META_PATH
from download_pubmed import load_all_abstracts


class BioBERTEncoder:
    def __init__(self, model_name=BIOMODEL, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"Loading BioBERT: {model_name} on {device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("BioBERT loaded.")
    
    def encode(self, texts, batch_size=32):
        """
        Encode texts using BioBERT mean pooling.

        Returns contiguous, float32, L2-normalized
        embeddings suitable for FAISS IndexFlatIP.
        """

        if not texts:
            return np.empty(
                (0, self.model.config.hidden_size),
                dtype=np.float32,
            )

        print("[ENCODE 1] entered encode()", flush=True)
        all_embeddings = []

        for i in range(
            0,
            len(texts),
            batch_size,
        ):
            print("[ENCODE 2] entering batch", flush=True)

            batch_texts = texts[
                i:i + batch_size
            ]

            print("[ENCODE 3] batch created", flush=True)

            print("[ENCODE 4] before tokenizer", flush=True)

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            print("[ENCODE 5] tokenizer PASS", flush=True)

            input_ids = encoded[
                "input_ids"
            ].to(self.device)

            print("[ENCODE 6] input_ids PASS", flush=True)

            attention_mask = encoded[
                "attention_mask"
            ].to(self.device)

            print("[ENCODE 7] attention_mask PASS", flush=True)
            print("[ENCODE 8] before model forward", flush=True)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            print("[ENCODE 9] model forward PASS", flush=True)

            token_embeddings = (
                outputs.last_hidden_state
            )

            print("[ENCODE 10] hidden state PASS", flush=True)

            print("[ENCODE 11] before mask expansion", flush=True)

            mask_expanded = (
                attention_mask
                .unsqueeze(-1)
                .to(
                    dtype=token_embeddings.dtype
                )
            )

            print("[ENCODE 12] mask expansion PASS", flush=True)

            sum_embeddings = torch.sum(
                token_embeddings
                * mask_expanded,
                dim=1,
            )

            print("[ENCODE 13] torch.sum PASS", flush=True)

            sum_mask = torch.clamp(
                mask_expanded.sum(dim=1),
                min=1e-9,
            )

            print("[ENCODE 14] sum_mask PASS", flush=True)

            embeddings = (
                sum_embeddings
                / sum_mask
            )

            print("[ENCODE 15] division PASS", flush=True)

            print("[ENCODE 16] before NumPy conversion", flush=True)

            batch_embeddings = (
                embeddings
                .detach()
                .cpu()
                .numpy()
            )

            print("[ENCODE 17] NumPy conversion PASS", flush=True)

            all_embeddings.append(
                batch_embeddings
            )

        print("[ENCODE 18] batches complete", flush=True)

        embeddings = np.concatenate(
            all_embeddings,
            axis=0,
        )

        print("[ENCODE 19] concatenate PASS", flush=True)

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        # NumPy L2 normalization avoids an unnecessary
        # native FAISS operation during query encoding.
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        embeddings = (
            embeddings
            / np.clip(
                norms,
                1e-12,
                None,
            )
        )

        return np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )


def build_faiss_index():
    print("=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)
    
    abstracts = load_all_abstracts()
    
    encoder = BioBERTEncoder(BIOMODEL)
    
    texts = [abs_dict['text'] for abs_dict in abstracts]
    ids = [abs_dict['id'] for abs_dict in abstracts]
    
    print(f"\nEncoding {len(texts)} abstracts...")
    embeddings = encoder.encode(texts, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Building FAISS index (IndexFlatIP)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Index size: {index.ntotal}")
    
    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    
    # Save metadata
    metadata = {'ids': ids, 'abstracts': abstracts}
    with open(FAISS_META_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata: {FAISS_META_PATH}")
    
    # Test search
    print("\nTesting index with sample query...")
    test_query = "Aspirin reduces risk of heart attack"
    test_emb = encoder.encode([test_query])
    distances, indices = index.search(test_emb, k=3)
    
    print(f"\nTop 3 results for: '{test_query}'")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  {i+1}. [{ids[idx]}] Score: {dist:.4f}")
        print(f"      {abstracts[idx]['title'][:80]}...")
    
    print("\n" + "=" * 70)
    print("INDEX BUILDING COMPLETE")
    print("=" * 70)
    return index, metadata


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"Index not found at {FAISS_INDEX_PATH}. Run: python build_faiss_index.py"
        )
    
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(FAISS_META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors")
    return index, metadata


if __name__ == "__main__":
    build_faiss_index()
