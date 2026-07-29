"""
src/models/static_classifier.py — Frozen CLIP + 7 multi-label heads

Architecture (matches PROJECT_NOTES.md section 2):
    Image -> CLIP ViT (FROZEN, no gradients) -> pooled image embedding
           -> 7 independent linear heads (one per static dimension)
           -> each head outputs per-class logits (sigmoid applied at
              inference / BCEWithLogitsLoss during training, per the
              multi-label decision in section 2b)

CLIP itself is never trained here -- only the 7 head layers have
trainable parameters. This is what makes this feasible on a 4GB GTX 1650:
you're only backpropagating through a handful of small linear layers,
not the full CLIP model.

Usage:
    from src.models.static_classifier import StaticShotClassifier
    from src.data.label_encoding import CLASS_VOCAB

    model = StaticShotClassifier(CLASS_VOCAB)
    model.to("cuda")

    # transform to preprocess images before feeding the model:
    transform = model.get_preprocess()

    outputs = model(pixel_values)  # dict: {"Frame Size": logits, ...}
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPImageProcessor

# Base CLIP checkpoint. ViT-B/32 is the smallest common CLIP vision
# transformer -- good starting point for a 4GB GPU. Swap to
# "openai/clip-vit-base-patch16" or "-large-patch14" later if you have
# headroom and want higher accuracy (bigger CLIP embedding = more
# signal for the heads, at the cost of more VRAM + slower forward pass).
DEFAULT_CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"

class ClipPreprocess:
    """
    Picklable replacement for a closure-based transform, so it can be
    sent to DataLoader worker processes on Windows (spawn start method
    can't pickle local/nested functions).
    """
    def __init__(self, clip_checkpoint):
        self.processor = CLIPImageProcessor.from_pretrained(clip_checkpoint)

    def __call__(self, pil_image):
        result = self.processor(images=pil_image, return_tensors="pt")
        return result["pixel_values"].squeeze(0)  # (3, H, W)

class StaticShotClassifier(nn.Module):
    def __init__(self, class_vocab: dict, clip_checkpoint: str = DEFAULT_CLIP_CHECKPOINT):
        """
        Args:
            class_vocab: CLASS_VOCAB dict from label_encoding.py --
                {"Frame Size": [...7 classes...], "Lens Size": [...], ...}
                Determines how many output units each head has.
            clip_checkpoint: HuggingFace model id for the frozen CLIP backbone.
        """
        super().__init__()
        self.dimensions = list(class_vocab.keys())

        # --- Load frozen CLIP backbone ---
        self.clip = CLIPModel.from_pretrained(clip_checkpoint)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()  # keep in eval mode -- no dropout/batchnorm updates ever

        embed_dim = self.clip.config.projection_dim  # e.g. 512 for ViT-B/32

        # --- One linear head per dimension ---
        # Using OrderedDict + ModuleDict so heads are addressable by name
        # (matches the dict keys the Dataset class already returns).
        self.heads = nn.ModuleDict(OrderedDict(
            (dim, nn.Linear(embed_dim, len(classes)))
            for dim, classes in class_vocab.items()
        ))

        # Store checkpoint name for reference / logging
        self.clip_checkpoint = clip_checkpoint

    def get_preprocess(self):
        """
        Returns a callable image transform matching this CLIP checkpoint's
        expected preprocessing (resize, center crop, normalization).
        Pass this as the `transform` argument to ShotDataset.

        Note: CLIPImageProcessor expects a PIL Image and returns a dict;
        we wrap it to return just the pixel_values tensor (squeezed) so
        it drops straight into ShotDataset's expected transform signature.
        """
        return ClipPreprocess(self.clip_checkpoint)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: (batch, 3, H, W) preprocessed image tensor
                (output of get_preprocess(), batched via DataLoader)

        Returns:
            dict of {dimension_name: logits tensor of shape (batch, n_classes)}
            Apply torch.sigmoid() to these for probabilities (multi-label),
            NOT softmax -- per the multi-label decision in section 2b.
        """
        # CLIP frozen -> no gradient tracking needed for this part
        #
        # NOTE: we deliberately do NOT use self.clip.get_image_features()
        # here. In some transformers versions it returns a raw tensor; in
        # others it returns a wrapped output object without the expected
        # attributes (observed: BaseModelOutputWithPooling with no
        # .image_embeds). Calling vision_model + visual_projection
        # directly is exactly what get_image_features does internally,
        # and is stable regardless of that wrapper's API changes.
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output  # (batch, vision_hidden_dim)
            image_features = self.clip.visual_projection(pooled_output)  # (batch, embed_dim)

        # Detach explicitly for clarity/safety, even though no_grad already
        # prevents graph construction -- avoids any accidental leakage if
        # this function is ever called outside a no_grad context upstream.
        image_features = image_features.detach()

        outputs = {}
        for dim, head in self.heads.items():
            outputs[dim] = head(image_features)
        return outputs

    def trainable_parameters(self):
        """Convenience: only the head parameters need an optimizer."""
        return [p for p in self.heads.parameters() if p.requires_grad]


if __name__ == "__main__":
    # Smoke test -- run with: python -m src.models.static_classifier
    from src.data.label_encoding import CLASS_VOCAB

    print("Loading model (downloads CLIP weights on first run)...")
    model = StaticShotClassifier(CLASS_VOCAB)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_frozen = sum(p.numel() for p in model.clip.parameters())
    print(f"Trainable params (heads only): {n_trainable:,}")
    print(f"Frozen params (CLIP backbone): {n_frozen:,}")

    # Fake batch of 2 images, correct CLIP input size (224x224 for ViT-B/32)
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)

    print("\nOutput shapes per dimension:")
    for dim, logits in outputs.items():
        print(f"  {dim}: {tuple(logits.shape)}")