"""Dataset classes for Phase 1 multimodal training."""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Dict, List


class TripletDataset(Dataset):
    """
    Contrastive triplet dataset using Honest OOC structure.
    ASSUMES: label 0 = true/REAL caption, label 1 = falsified/FAKE caption
    (Matches corrected preprocessor: Honest OOC original 1->0, 2->1)
    """
    def __init__(self, csv_path: str, tokenizer,
                 image_transform: Optional[Callable] = None, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.triplets = self._build_triplets()
        print(f"[TripletDataset] Built {len(self.triplets)} triplets from {csv_path}")
        
    def _build_triplets(self) -> List[Dict]:
        triplets = []
        image_col = "image_path" if "image_path" in self.df.columns else "image"
        caption_col = "caption" if "caption" in self.df.columns else "claim_text"
        label_col = "label" if "label" in self.df.columns else "rating"
        
        # Only honest_ooc has paired true+falsified captions per image
        df_ho = self.df[self.df.get("source_dataset", "") == "honest_ooc"].copy()
        if len(df_ho) == 0:
            print("WARNING: No honest_ooc samples found. TripletDataset will be empty.")
            return triplets
            
        grouped = df_ho.groupby(image_col)
        for img_path, group in grouped:
            # FIXED: label 0 = true, label 1 = falsified (matches preprocessor)
            true_caps = group[group[label_col] == 0][caption_col].tolist()
            fake_caps = group[group[label_col] == 1][caption_col].tolist()
            if len(true_caps) > 0 and len(fake_caps) > 0:
                triplets.append({
                    "image_path": img_path,
                    "true_caption": str(true_caps[0]),
                    "fake_caption": str(fake_caps[0]),
                })
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        trip = self.triplets[idx]
        pix = self._load_image(trip["image_path"])
        
        true_enc = self.tokenizer(
            trip["true_caption"], max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        fake_enc = self.tokenizer(
            trip["fake_caption"], max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "true_ids": true_enc["input_ids"].squeeze(0),
            "true_mask": true_enc["attention_mask"].squeeze(0),
            "fake_ids": fake_enc["input_ids"].squeeze(0),
            "fake_mask": fake_enc["attention_mask"].squeeze(0),
            "pix": pix,
            "image_path": trip["image_path"],
        }
    
    def _load_image(self, path: str):
        try:
            if os.path.exists(str(path)):
                img = Image.open(path).convert("RGB")
                if self.image_transform:
                    return self.image_transform(img)
        except Exception:
            pass
        return torch.zeros(3, 224, 224)


class BinaryMMDataset(Dataset):
    """
    Binary classification dataset. Uses the 'label' column from CSV directly.
    CRITICAL: No source-based label overwriting.
    """
    def __init__(self, csv_path: str, tokenizer,
                 image_transform: Optional[Callable] = None, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        
        required = ["claim_text", "image_path", "source_dataset", "label"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col} in {csv_path}")
        
        self.df["label"] = self.df["label"].astype(int)
        unique_labels = sorted(self.df["label"].unique())
        print(f"[BinaryMMDataset] Loaded {len(self.df)} samples from {csv_path}")
        print(f"  Unique labels: {unique_labels} (MUST be [0, 1])")
        
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"CRITICAL: Labels contain values other than 0 and 1: {unique_labels}")
        
        # CRITICAL: Print per-source distribution to catch mapping errors
        print(f"  Per-source label distribution:")
        for src in self.df["source_dataset"].unique():
            src_df = self.df[self.df["source_dataset"] == src]
            counts = src_df["label"].value_counts().sort_index().to_dict()
            print(f"    {src}: {counts}")
            if len(counts) == 1:
                print(f"      ⚠️  WARNING: {src} is 100% single label! Verify preprocessing.")
        
        # Validate images
        self.df = self.df[self.df["image_path"].notna()].reset_index(drop=True)
        valid_mask = [os.path.exists(str(p)) for p in self.df["image_path"]]
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"  After image validation: {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pix = self._load_image(row["image_path"])
        
        enc = self.tokenizer(
            str(row["claim_text"]), max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        source = row.get("source_dataset", "unknown")
        label_value = int(row["label"])
        
        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "pix": pix,
            "y": torch.tensor(label_value, dtype=torch.float),
            "source": source,
            "image_path": row["image_path"],
        }
    
    def _load_image(self, path: str):
        try:
            if os.path.exists(str(path)):
                img = Image.open(path).convert("RGB")
                if self.image_transform:
                    return self.image_transform(img)
        except Exception:
            pass
        return torch.zeros(3, 224, 224)


# ============================================================
# BASELINE DATASETS
# ============================================================

class TextOnlyDataset(Dataset):
    """Text-only baseline."""
    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.df = self.df[self.df["image_path"].notna()].reset_index(drop=True)
        valid_mask = [os.path.exists(str(p)) for p in self.df["image_path"]]
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"[TextOnlyDataset] {len(self.df)} samples")
        for src in self.df["source_dataset"].unique():
            counts = self.df[self.df["source_dataset"] == src]["label"].value_counts().sort_index().to_dict()
            print(f"  {src}: {counts}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["claim_text"]), max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "y": torch.tensor(int(row["label"]), dtype=torch.float),
            "source": row.get("source_dataset", "unknown"),
        }


class ImageOnlyDataset(Dataset):
    """Image-only baseline."""
    def __init__(self, csv_path: str, image_transform: Optional[Callable] = None):
        self.df = pd.read_csv(csv_path)
        self.image_transform = image_transform
        
        self.df = self.df[self.df["image_path"].notna()].reset_index(drop=True)
        valid_mask = [os.path.exists(str(p)) for p in self.df["image_path"]]
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"[ImageOnlyDataset] {len(self.df)} samples")
        for src in self.df["source_dataset"].unique():
            counts = self.df[self.df["source_dataset"] == src]["label"].value_counts().sort_index().to_dict()
            print(f"  {src}: {counts}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pix = self._load_image(row["image_path"])
        return {
            "pix": pix,
            "y": torch.tensor(int(row["label"]), dtype=torch.float),
            "source": row.get("source_dataset", "unknown"),
        }
    
    def _load_image(self, path: str):
        try:
            if os.path.exists(str(path)):
                img = Image.open(path).convert("RGB")
                if self.image_transform:
                    return self.image_transform(img)
        except Exception:
            pass
        return torch.zeros(3, 224, 224)


def collate_fn_contrastive(batch):
    return {
        "true_ids": torch.stack([b["true_ids"] for b in batch]),
        "true_mask": torch.stack([b["true_mask"] for b in batch]),
        "fake_ids": torch.stack([b["fake_ids"] for b in batch]),
        "fake_mask": torch.stack([b["fake_mask"] for b in batch]),
        "pix": torch.stack([b["pix"] for b in batch]),
        "image_path": [b["image_path"] for b in batch],
    }


def collate_fn_binary(batch):
    return {
        "ids": torch.stack([b["ids"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "pix": torch.stack([b["pix"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "source": [b["source"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def collate_fn_text_only(batch):
    return {
        "ids": torch.stack([b["ids"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "source": [b["source"] for b in batch],
    }


def collate_fn_image_only(batch):
    return {
        "pix": torch.stack([b["pix"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "source": [b["source"] for b in batch],
    }