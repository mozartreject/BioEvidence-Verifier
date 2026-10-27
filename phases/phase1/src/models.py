"""Model architectures for Phase 1 Multimodal Encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, text_emb: torch.Tensor, img_emb: torch.Tensor):
        text_seq = text_emb.unsqueeze(1)
        img_seq = img_emb.unsqueeze(1)
        text_attended, _ = self.text_to_img_attn(text_seq, img_seq, img_seq)
        img_attended, _ = self.img_to_text_attn(img_seq, text_seq, text_seq)
        text_attended = text_attended.squeeze(1)
        img_attended = img_attended.squeeze(1)
        fused = self.fusion(torch.cat([text_attended, img_attended], dim=-1))
        return fused


class BilinearFusion(nn.Module):
    def __init__(self, hidden_dim: int = 768, rank: int = 256):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, rank)
        self.project = nn.Linear(rank, hidden_dim)
        
    def forward(self, text_emb: torch.Tensor, img_emb: torch.Tensor):
        fused = self.bilinear(text_emb, img_emb)
        fused = self.project(fused)
        return fused


class ContrastiveFusion(nn.Module):
    def __init__(
        self,
        biobert_name: str = "dmis-lab/biobert-base-cased-v1.1",
        vit_name: str = "google/vit-base-patch16-224",
        fusion_type: str = "concat",
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        self.biobert = BertModel.from_pretrained(biobert_name)
        self.vit = ViTModel.from_pretrained(vit_name)
        
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "cross_attn":
            self.fusion = CrossAttentionFusion(hidden_dim, dropout=dropout)
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(hidden_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.no_image_token = nn.Parameter(torch.randn(hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.text_proj, self.img_proj, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_proj(text_cls)
        return text_proj
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values=pixel_values)
        img_cls = outputs.last_hidden_state[:, 0, :]
        img_proj = self.img_proj(img_cls)
        return img_proj
    
    def fuse(self, text_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused = self.fusion(torch.cat([text_emb, img_emb], dim=-1))
        else:
            fused = self.fusion(text_emb, img_emb)
        return fused
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pixel_values: torch.Tensor, return_components: bool = False):
        text_emb = self.encode_text(input_ids, attention_mask)
        img_emb = self.encode_image(pixel_values)
        text_emb = F.normalize(text_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)
        fused = self.fuse(text_emb, img_emb)
        fused = F.normalize(fused, dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        if return_components:
            return logits, {"text": text_emb, "image": img_emb, "fused": fused}
        return logits
    
    def forward_contrastive(self, true_input_ids: torch.Tensor, true_attention_mask: torch.Tensor,
                            fake_input_ids: torch.Tensor, fake_attention_mask: torch.Tensor,
                            pixel_values: torch.Tensor):
        img_emb = self.encode_image(pixel_values)
        img_emb = F.normalize(img_emb, dim=-1)
        
        text_true = self.encode_text(true_input_ids, true_attention_mask)
        text_true = F.normalize(text_true, dim=-1)
        fused_true = self.fuse(text_true, img_emb)
        fused_true = F.normalize(fused_true, dim=-1)
        
        text_fake = self.encode_text(fake_input_ids, fake_attention_mask)
        text_fake = F.normalize(text_fake, dim=-1)
        fused_fake = self.fuse(text_fake, img_emb)
        fused_fake = F.normalize(fused_fake, dim=-1)
        
        return fused_true, fused_fake
    
    def save_projection_layers(self, path: str):
        torch.save({
            "text_proj": self.text_proj.state_dict(),
            "img_proj": self.img_proj.state_dict(),
            "fusion": self.fusion.state_dict() if hasattr(self.fusion, "state_dict") else None,
            "no_image_token": self.no_image_token,
            "fusion_type": self.fusion_type,
        }, path)
        print(f"Saved projection layers to {path}")
    
    def load_projection_layers(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.text_proj.load_state_dict(checkpoint["text_proj"])
        self.img_proj.load_state_dict(checkpoint["img_proj"])
        if checkpoint.get("fusion") is not None and hasattr(self.fusion, "load_state_dict"):
            self.fusion.load_state_dict(checkpoint["fusion"])
        if "no_image_token" in checkpoint:
            self.no_image_token.data = checkpoint["no_image_token"]
        print(f"Loaded projection layers from {path}")


class TextOnlyClassifier(nn.Module):
    """BioBERT-only baseline for ablation."""
    def __init__(self, biobert_name: str = "dmis-lab/biobert-base-cased-v1.1", 
                 hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.biobert = BertModel.from_pretrained(biobert_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(self, input_ids, attention_mask, pixel_values=None, return_components=False):
        outputs = self.biobert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)


class ImageOnlyClassifier(nn.Module):
    """ViT-only baseline for ablation."""
    def __init__(self, vit_name: str = "google/vit-base-patch16-224",
                 hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, return_components=False):
        outputs = self.vit(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)