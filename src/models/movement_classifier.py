"""
src/models/movement_classifier.py

Temporal camera-movement classifier: frozen CLIP (per-frame embeddings) +
a small trained temporal head (LSTM or Transformer, selectable) + a linear
classifier over the 21-class MOVEMENT_CLASS_VOCAB.

Mirrors the frozen-CLIP forward pattern from static_classifier.py (calls
clip.vision_model() + clip.visual_projection() directly, bypassing
get_image_features() due to the transformers-version API quirk documented
in PROJECT_NOTES.md section 11).

Usage:
    from src.models.movement_classifier import MovementClassifier
    model = MovementClassifier(num_classes=21, temporal_type="lstm")
    logits = model(frames)  # frames: (batch, 16, 3, 224, 224)
"""

import torch
import torch.nn as nn
from transformers import CLIPModel

DEFAULT_CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"
DEFAULT_N_FRAMES = 16


class MovementClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        clip_checkpoint: str = DEFAULT_CLIP_CHECKPOINT,
        temporal_type: str = "lstm",  # "lstm" or "transformer"
        n_frames: int = DEFAULT_N_FRAMES,
        hidden_dim: int = 128,
        num_layers: int = 1,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert temporal_type in ("lstm", "transformer"), temporal_type
        self.temporal_type = temporal_type
        self.n_frames = n_frames

        self.clip = CLIPModel.from_pretrained(clip_checkpoint)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/32

        if temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:  # transformer
            self.pos_embedding = nn.Parameter(torch.zeros(1, n_frames, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Linear(embed_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # frozen backbone always stays in eval mode, regardless of model.train()/.eval()
        return self

    def _embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (batch, seq_len, 3, H, W) -> (batch, seq_len, embed_dim).
        CLIP is frozen, so no gradient is needed through it."""
        b, t, c, h, w = frames.shape
        flat = frames.view(b * t, c, h, w)
        with torch.no_grad():
            vision_out = self.clip.vision_model(pixel_values=flat)
            pooled = vision_out.pooler_output
            embeds = self.clip.visual_projection(pooled)
        return embeds.view(b, t, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        embeds = self._embed_frames(frames)

        if self.temporal_type == "lstm":
            _, (h_n, _) = self.temporal(embeds)
            pooled = h_n[-1]  # last layer's final hidden state: (batch, hidden_dim)
        else:
            x = embeds + self.pos_embedding
            x = self.temporal(x)
            pooled = x.mean(dim=1)  # mean-pool over the sequence

        return self.classifier(pooled)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test -- run with: python -m src.models.movement_classifier
    dummy_frames = torch.randn(2, DEFAULT_N_FRAMES, 3, 224, 224)  # batch of 2

    for temporal_type in ("lstm", "transformer"):
        model = MovementClassifier(num_classes=21, temporal_type=temporal_type)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_frames)
        print(f"[{temporal_type}] output shape: {logits.shape}  "
              f"(expected: (2, 21))")
        print(f"[{temporal_type}] trainable params: {count_trainable_params(model):,}")