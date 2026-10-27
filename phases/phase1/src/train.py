#!/usr/bin/env python3
"""Phase 1 Training Script - UPDATED FOR MACRO-F1, CM PLOTS, AND RETRIEVAL."""
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config
from src.models import ContrastiveFusion, TextOnlyClassifier, ImageOnlyClassifier
from src.dataset import (
    TripletDataset, BinaryMMDataset, TextOnlyDataset, ImageOnlyDataset,
    collate_fn_contrastive, collate_fn_binary, collate_fn_text_only, collate_fn_image_only
)
from src.losses import FocalLoss, ContrastiveLoss
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    set_seed, count_parameters, freeze_backbone, unfreeze_top_layers,
    print_metrics, save_checkpoint, EarlyStopping, save_json
)

def build_model(cfg):
    if cfg.training.mode == "text_only":
        model = TextOnlyClassifier(biobert_name=cfg.model.biobert, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
        print(f"[TextOnly] Trainable parameters: {count_parameters(model):,}")
    elif cfg.training.mode == "image_only":
        model = ImageOnlyClassifier(vit_name=cfg.model.vit, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
        print(f"[ImageOnly] Trainable parameters: {count_parameters(model):,}")
    else:
        model = ContrastiveFusion(biobert_name=cfg.model.biobert, vit_name=cfg.model.vit, fusion_type=cfg.model.fusion_type, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
        if cfg.model.freeze_biobert: freeze_backbone(model.biobert, "biobert")
        else: unfreeze_top_layers(model.biobert, "biobert", cfg.model.unfreeze_biobert_top_n)
        if cfg.model.freeze_vit: freeze_backbone(model.vit, "vit")
        else: unfreeze_top_layers(model.vit, "vit", cfg.model.unfreeze_vit_top_n)
        print(f"[Multimodal] Trainable parameters: {count_parameters(model):,}")
    return model

def build_dataloaders(cfg, tokenizer):
    mode = cfg.training.mode
    if mode == "contrastive":
        train_ds = TripletDataset(os.path.join(cfg.data.processed_dir, "train_multimodal.csv"), tokenizer, get_train_transforms(cfg.training.use_augmentation))
        val_ds = TripletDataset(os.path.join(cfg.data.processed_dir, "val_multimodal.csv"), tokenizer, get_val_transforms())
        test_ds = TripletDataset(os.path.join(cfg.data.processed_dir, "test_multimodal.csv"), tokenizer, get_val_transforms())
        collate_fn = collate_fn_contrastive
    elif mode == "text_only":
        train_ds = TextOnlyDataset(os.path.join(cfg.data.processed_dir, "train_multimodal.csv"), tokenizer)
        val_ds = TextOnlyDataset(os.path.join(cfg.data.processed_dir, "val_multimodal.csv"), tokenizer)
        test_ds = TextOnlyDataset(os.path.join(cfg.data.processed_dir, "test_multimodal.csv"), tokenizer)
        collate_fn = collate_fn_text_only
    elif mode == "image_only":
        train_ds = ImageOnlyDataset(os.path.join(cfg.data.processed_dir, "train_multimodal.csv"), get_train_transforms(cfg.training.use_augmentation))
        val_ds = ImageOnlyDataset(os.path.join(cfg.data.processed_dir, "val_multimodal.csv"), get_val_transforms())
        test_ds = ImageOnlyDataset(os.path.join(cfg.data.processed_dir, "test_multimodal.csv"), get_val_transforms())
        collate_fn = collate_fn_image_only
    else:
        train_ds = BinaryMMDataset(os.path.join(cfg.data.processed_dir, "train_multimodal.csv"), tokenizer, get_train_transforms(cfg.training.use_augmentation))
        val_ds = BinaryMMDataset(os.path.join(cfg.data.processed_dir, "val_multimodal.csv"), tokenizer, get_val_transforms())
        test_ds = BinaryMMDataset(os.path.join(cfg.data.processed_dir, "test_multimodal.csv"), tokenizer, get_val_transforms())
        collate_fn = collate_fn_binary
    
    if mode in ("binary", "text_only", "image_only"):
        fake_count = sum(1 for idx in range(len(train_ds)) if train_ds[idx]["y"].item() == 1.0)
        real_count = len(train_ds) - fake_count
        total = fake_count + real_count
        print(f"\n[WeightedSampler] Train set: Fake={fake_count}, Real={real_count}")
        weight_for_fake = total / (2.0 * fake_count) if fake_count > 0 else 1.0
        weight_for_real = total / (2.0 * real_count) if real_count > 0 else 1.0
        sample_weights = [weight_for_fake if train_ds[idx]["y"].item() == 1.0 else weight_for_real for idx in range(len(train_ds))]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, sampler=sampler, num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory, collate_fn=collate_fn, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory, collate_fn=collate_fn, drop_last=True)
    
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def train_epoch_contrastive(model, loader, optimizer, criterion, device, scaler, cfg, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        true_ids, true_mask = batch["true_ids"].to(device), batch["true_mask"].to(device)
        fake_ids, fake_mask = batch["fake_ids"].to(device), batch["fake_mask"].to(device)
        pix = batch["pix"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp and device.type == "cuda"):
            fused_true, fused_fake = model.forward_contrastive(true_ids, true_mask, fake_ids, fake_mask, pix)
            loss = criterion(fused_true, fused_fake)
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(loader)

def validate_contrastive(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validate]"):
            true_ids, true_mask = batch["true_ids"].to(device), batch["true_mask"].to(device)
            fake_ids, fake_mask = batch["fake_ids"].to(device), batch["fake_mask"].to(device)
            pix = batch["pix"].to(device)
            fused_true, fused_fake = model.forward_contrastive(true_ids, true_mask, fake_ids, fake_mask, pix)
            loss = criterion(fused_true, fused_fake)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch_binary(model, loader, optimizer, criterion, device, scaler, cfg, epoch):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        ids, mask, pix, y = batch.get("ids"), batch.get("mask"), batch.get("pix"), batch["y"].to(device)
        if ids is not None: ids = ids.to(device)
        if mask is not None: mask = mask.to(device)
        if pix is not None: pix = pix.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp and device.type == "cuda"):
            logits = model(ids, mask, pix)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix({"loss": loss.item()})
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    train_f1 = f1_score(all_labels, preds, zero_division=0, average='macro')
    return total_loss / len(loader), train_f1

def validate_binary(model, loader, criterion, device, cfg):
    model.eval()
    total_loss, all_probs, all_labels, all_sources = 0.0, [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validate]"):
            ids, mask, pix, y = batch.get("ids"), batch.get("mask"), batch.get("pix"), batch["y"].to(device)
            if ids is not None: ids = ids.to(device)
            if mask is not None: mask = mask.to(device)
            if pix is not None: pix = pix.to(device)
            logits = model(ids, mask, pix)
            loss = criterion(logits, y)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_sources.extend(batch["source"])
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    best_thresh, best_f1 = 0.5, 0.0
    if len(np.unique(all_probs)) >= 2:
        for thresh in np.arange(cfg.evaluation.threshold_search_min, cfg.evaluation.threshold_search_max, cfg.evaluation.threshold_search_step):
            preds_t = (all_probs > thresh).astype(int)
            f1_t = f1_score(all_labels, preds_t, zero_division=0, average='macro')
            if f1_t > best_f1: best_f1, best_thresh = f1_t, thresh
    val_preds = (all_probs > best_thresh).astype(int)
    metrics = {
        "loss": total_loss / len(loader), "threshold": best_thresh,
        "accuracy": accuracy_score(all_labels, val_preds),
        "precision_macro": precision_score(all_labels, val_preds, zero_division=0, average='macro'),
        "recall_macro": recall_score(all_labels, val_preds, zero_division=0, average='macro'),
        "f1_macro": f1_score(all_labels, val_preds, zero_division=0, average='macro'),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    if len(np.unique(all_labels)) == 2:
        metrics["precision_fake"] = precision_score(all_labels, val_preds, pos_label=1, zero_division=0)
        metrics["recall_fake"] = recall_score(all_labels, val_preds, pos_label=1, zero_division=0)
        metrics["precision_real"] = precision_score(all_labels, val_preds, pos_label=0, zero_division=0)
        metrics["recall_real"] = recall_score(all_labels, val_preds, pos_label=0, zero_division=0)
    metrics["per_source"] = {src: accuracy_score(all_labels[np.array([s == src for s in all_sources])], val_preds[np.array([s == src for s in all_sources])]) for src in set(all_sources) if np.array([s == src for s in all_sources]).sum() > 0}
    return metrics

def test_binary(model, loader, device, threshold, cfg, run_name, ckpt_dir):
    model.eval()
    all_probs, all_labels, all_sources = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Test]"):
            ids, mask, pix, y = batch.get("ids"), batch.get("mask"), batch.get("pix"), batch["y"].to(device)
            if ids is not None: ids = ids.to(device)
            if mask is not None: mask = mask.to(device)
            if pix is not None: pix = pix.to(device)
            logits = model(ids, mask, pix)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_sources.extend(batch["source"])
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    test_preds = (all_probs > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(all_labels, test_preds),
        "precision_macro": precision_score(all_labels, test_preds, zero_division=0, average='macro'),
        "recall_macro": recall_score(all_labels, test_preds, zero_division=0, average='macro'),
        "f1_macro": f1_score(all_labels, test_preds, zero_division=0, average='macro'),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    metrics["per_source"] = {src: accuracy_score(all_labels[np.array([s == src for s in all_sources])], test_preds[np.array([s == src for s in all_sources])]) for src in set(all_sources) if np.array([s == src for s in all_sources]).sum() > 0}
    cm = confusion_matrix(all_labels, test_preds)
    report = classification_report(all_labels, test_preds, target_names=["Fake (0)", "Real (1)"], digits=4, output_dict=True)
    
    # SAVE CONFUSION MATRIX PLOT
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake (0)', 'Real (1)'], yticklabels=['Fake (0)', 'Real (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {cfg.training.mode}')
    cm_path = os.path.join(ckpt_dir, f"confusion_matrix_{cfg.training.mode}_{run_name}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix plot to {cm_path}")
    
    return metrics, all_probs, all_labels, cm, report

def evaluate_contrastive_retrieval(model, loader, device, cfg, ckpt_dir):
    """Calculates Retrieval Metrics (Recall@1, 5, 10) safely using existing embeddings."""
    model.eval()
    all_text_embs, all_img_embs = [], []
    print("\n" + "="*70)
    print("EXTRACTING EMBEDDINGS FOR RETRIEVAL METRICS")
    print("="*70)
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Extracting Embeddings]"):
            true_ids, true_mask = batch["true_ids"].to(device), batch["true_mask"].to(device)
            pix = batch["pix"].to(device)
            text_emb = F.normalize(model.encode_text(true_ids, true_mask), dim=-1)
            img_emb = F.normalize(model.encode_image(pix), dim=-1)
            all_text_embs.append(text_emb.cpu())
            all_img_embs.append(img_emb.cpu())
    all_text_embs, all_img_embs = torch.cat(all_text_embs, dim=0), torch.cat(all_img_embs, dim=0)
    sim_matrix = torch.matmul(all_text_embs, all_img_embs.T)
    N = sim_matrix.size(0)
    labels = torch.arange(N).to(sim_matrix.device)
    t2i_preds = torch.argsort(sim_matrix, dim=1, descending=True)
    i2t_preds = torch.argsort(sim_matrix, dim=0, descending=True).T
    metrics = {}
    for k in [1, 5, 10]:
        if k > N: continue
        metrics[f"Text_to_Image_Recall@{k}"] = (t2i_preds == labels.unsqueeze(1)).float()[:, :k].sum().item() / N
        metrics[f"Image_to_Text_Recall@{k}"] = (i2t_preds == labels.unsqueeze(1)).float()[:, :k].sum().item() / N
    print("\n📊 RETRIEVAL METRICS (Cross-Modal Alignment Quality):")
    for k, v in metrics.items(): print(f"  {k:25s}: {v:.4f}")
    save_json(metrics, os.path.join(ckpt_dir, f"retrieval_metrics_{cfg.training.mode}.json"))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Multimodal Encoder Training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["contrastive", "binary", "text_only", "image_only"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fusion", choices=["concat", "cross_attn", "bilinear"])
    parser.add_argument("--freeze_backbones", action="store_true")
    parser.add_argument("--unfreeze_biobert", type=int)
    parser.add_argument("--unfreeze_vit", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if args.mode: cfg.training.mode = args.mode
    if args.epochs: cfg.training.epochs = args.epochs
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.fusion: cfg.model.fusion_type = args.fusion
    if args.freeze_backbones: cfg.model.freeze_biobert, cfg.model.freeze_vit = True, True
    if args.unfreeze_biobert is not None: cfg.model.freeze_biobert, cfg.model.unfreeze_biobert_top_n = False, args.unfreeze_biobert
    if args.unfreeze_vit is not None: cfg.model.freeze_vit, cfg.model.unfreeze_vit_top_n = False, args.unfreeze_vit
    
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else cfg.logging.checkpoint_dir
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    set_seed(cfg.system.seed)
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    tokenizer = BertTokenizer.from_pretrained(cfg.model.biobert)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, tokenizer)
    model = build_model(cfg).to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    total_steps = len(train_loader) * cfg.training.epochs // cfg.training.accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(total_steps * cfg.training.warmup_ratio), T_mult=2)
    
    if cfg.training.mode == "contrastive": criterion = ContrastiveLoss(margin=cfg.training.contrastive_margin)
    else:
        fake_count = sum(1 for idx in range(len(train_loader.dataset)) if train_loader.dataset[idx]["y"].item() == 1.0)
        real_count = len(train_loader.dataset) - fake_count
        pos_weight_value = real_count / fake_count if fake_count > 0 else 1.0
        criterion = FocalLoss(alpha=cfg.training.focal_alpha, gamma=cfg.training.focal_gamma, pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(device))
    
    scaler = torch.cuda.amp.GradScaler() if cfg.training.use_amp and device.type == "cuda" else None
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, run_name))
    early_stop = EarlyStopping(patience=cfg.training.early_stopping_patience, mode="max")
    
    start_epoch, best_val_f1, best_threshold = 1, 0.0, 0.5
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("metrics", {}).get("f1_macro", 0.0)
    
    print(f"\nSTARTING TRAINING: {cfg.training.mode.upper()} | Epochs: {cfg.training.epochs} | Batch: {cfg.training.batch_size}")
    
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_start = time.time()
        if cfg.training.mode == "contrastive":
            train_loss = train_epoch_contrastive(model, train_loader, optimizer, criterion, device, scaler, cfg, epoch)
            val_loss = validate_contrastive(model, val_loader, criterion, device, cfg)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < getattr(main, "best_val_loss", float("inf")):
                main.best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, {"val_loss": val_loss}, os.path.join(ckpt_dir, f"best_contrastive_{run_name}.pt"), is_best=True)
        else:
            train_loss, train_f1 = train_epoch_binary(model, train_loader, optimizer, criterion, device, scaler, cfg, epoch)
            val_metrics = validate_binary(model, val_loader, criterion, device, cfg)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_metrics['f1_macro']:.4f} (thresh={val_metrics['threshold']:.2f})")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("MacroF1/val", val_metrics["f1_macro"], epoch)
            
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1, best_threshold = val_metrics["f1_macro"], val_metrics["threshold"]
                save_checkpoint(model, optimizer, epoch, val_metrics, os.path.join(ckpt_dir, f"best_{cfg.training.mode}_{run_name}.pt"), is_best=True)
            if early_stop(val_metrics["f1_macro"]): print(f"Early stopping triggered at epoch {epoch}"); break
        scheduler.step()
    writer.close()
    
    if cfg.training.mode in ("binary", "text_only", "image_only"):
        print("\nFINAL TEST EVALUATION")
        test_metrics, test_probs, test_labels, cm, report = test_binary(model, test_loader, device, best_threshold, cfg, run_name, ckpt_dir)
        print(f"\nTest Macro-F1: {test_metrics['f1_macro']:.4f}")
        print_metrics(test_metrics, prefix="Test ")
        results = {"config": {"mode": cfg.training.mode, "epochs": cfg.training.epochs, "batch_size": cfg.training.batch_size}, "best_val_macro_f1": best_val_f1, "best_threshold": best_threshold, "test_metrics": test_metrics, "confusion_matrix": cm.tolist(), "classification_report": report}
        save_json(results, os.path.join(ckpt_dir, f"test_results_{run_name}.json"))
        
    if cfg.training.mode == "binary":
        proj_path = os.path.join(ckpt_dir, f"best_fusion_{run_name}.pt")
        model.save_projection_layers(proj_path)
        
    if cfg.training.mode == "contrastive":
        print("\nFINAL CONTRASTIVE RETRIEVAL EVALUATION")
        evaluate_contrastive_retrieval(model, test_loader, device, cfg, ckpt_dir)

if __name__ == "__main__":
    main()