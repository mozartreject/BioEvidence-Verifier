"""Loss functions for Phase 1."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification mode with optional pos_weight."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # ============================================================
        # FIX: Accept pos_weight to handle class imbalance
        # ============================================================
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce_loss
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Triplet-style contrastive loss for multimodal alignment."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, fused_true: torch.Tensor, fused_fake: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(fused_true - fused_fake, dim=1)
        loss = torch.clamp(self.margin - dist, min=0.0).mean()
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent loss for contrastive learning."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, fused_true: torch.Tensor, fused_fake: torch.Tensor) -> torch.Tensor:
        batch_size = fused_true.size(0)
        fused_true = F.normalize(fused_true, dim=-1)
        fused_fake = F.normalize(fused_fake, dim=-1)
        all_fused = torch.cat([fused_true, fused_fake], dim=0)
        sim_matrix = torch.matmul(all_fused, all_fused.T) / self.temperature
        mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        cos_sim = F.cosine_similarity(fused_true, fused_fake, dim=-1)
        loss = (cos_sim + 1.0).mean()
        return loss