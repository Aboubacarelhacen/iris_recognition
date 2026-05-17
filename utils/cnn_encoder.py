"""
CNN iris embedding inference (supports both v1 and v2 checkpoints).

Two architectures are recognized via the `arch` field in the checkpoint:

  * ``resnet18_pretrained_3ch`` (v2 — Kaggle notebook) — pretrained ResNet18,
    3-channel input (grayscale replicated), ImageNet-normalized.
  * ``resnet18_1ch`` (v1 — older local train_cnn.py) — ResNet18 with 1-channel
    conv1 replacement, no normalization beyond [0, 1].

Inference contract:
    >>> enc = CNNIrisEncoder.load("model/cnn_iris.pt")
    >>> emb = enc.encode(strip_64x512_uint8)   # -> (256,) float32, L2-normalized
    >>> CNNIrisEncoder.cosine_distance(a, b)   # -> float in [0, 2]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- lazy torch
def _torch():
    import torch  # noqa: F401
    return torch


_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------- architectures
def _build_v2_model(num_classes, embed_dim=256):
    """Pretrained ResNet18, 3-channel input — matches train_cnn_kaggle.ipynb v2."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18

    class ArcFace(nn.Module):
        def __init__(self, dim, n_class, s=30.0, m=0.0):
            super().__init__()
            self.s, self.m = s, m
            self.W = nn.Parameter(torch.empty(n_class, dim))
            nn.init.xavier_normal_(self.W)
        def forward(self, x, y=None):
            cos = F.linear(F.normalize(x), F.normalize(self.W))
            if y is None or self.m == 0.0: return cos * self.s
            cos = cos.clamp(-1+1e-7, 1-1e-7)
            sin = (1 - cos.pow(2)).sqrt()
            cm, sm = math.cos(self.m), math.sin(self.m)
            phi = cos*cm - sin*sm
            oh = F.one_hot(y, cos.size(1)).float()
            return (oh*phi + (1-oh)*cos) * self.s

    class IrisNet(nn.Module):
        def __init__(self, n_class, d=256):
            super().__init__()
            b = resnet18(weights=None)   # weights come from checkpoint
            feat = b.fc.in_features; b.fc = nn.Identity()
            self.backbone = b
            self.head = nn.Sequential(nn.Linear(feat, d, bias=False),
                                      nn.BatchNorm1d(d))
            self.arc = ArcFace(d, n_class)
        def embed(self, x):
            return F.normalize(self.head(self.backbone(x)), dim=1)
        def forward(self, x, y=None):
            e = self.embed(x); return self.arc(e, y), e

    return IrisNet(num_classes, embed_dim)


def _build_v1_model(num_classes, embed_dim=256):
    """1-channel conv1 ResNet18 — matches the older train_cnn.py."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18

    class ArcMarginProduct(nn.Module):
        def __init__(self, in_f, out_f, s=30.0, m=0.50):
            super().__init__()
            self.s, self.m = s, m
            self.weight = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.xavier_normal_(self.weight)
        def forward(self, e, y=None):
            cos = F.linear(F.normalize(e), F.normalize(self.weight))
            if y is None: return cos * self.s
            sin = torch.sqrt((1.0 - cos.pow(2)).clamp(0, 1))
            cm, sm = math.cos(self.m), math.sin(self.m)
            phi = cos*cm - sin*sm
            oh = F.one_hot(y, cos.size(1)).float()
            return (oh * phi + (1-oh) * cos) * self.s

    class IrisEmbedding(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            b = resnet18(weights=None)
            b.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            feat = b.fc.in_features; b.fc = nn.Identity()
            self.backbone = b
            self.embed_head = nn.Sequential(nn.Linear(feat, d, bias=False),
                                            nn.BatchNorm1d(d))
        def forward(self, x):
            return F.normalize(self.embed_head(self.backbone(x)), dim=1)

    class IrisModel(nn.Module):
        def __init__(self, n_class, d=256):
            super().__init__()
            self.encoder = IrisEmbedding(d)
            self.arc = ArcMarginProduct(d, n_class)
        def forward(self, x, y=None):
            e = self.encoder(x); return self.arc(e, y), e
        def embed(self, x): return self.encoder(x)

    return IrisModel(num_classes, embed_dim)


def build_model(num_classes, embed_dim=256, arch="resnet18_pretrained_3ch"):
    """Build the encoder model for a given architecture tag."""
    if arch == "resnet18_pretrained_3ch":
        return _build_v2_model(num_classes, embed_dim)
    return _build_v1_model(num_classes, embed_dim)


# ----------------------------------------------------------------- inference
class CNNIrisEncoder:
    """Inference-only wrapper. Auto-detects v1 / v2 architecture."""

    def __init__(self, model, device, arch, embed_dim, threshold=None):
        self.model, self.device = model, device
        self.arch, self.embed_dim = arch, embed_dim
        self.threshold = threshold

    @classmethod
    def load(cls, checkpoint_path):
        torch = _torch()
        ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                          weights_only=False)
        arch = ckpt.get("arch", "resnet18_1ch")
        embed_dim = ckpt.get("embed_dim", 256)
        num_classes = ckpt.get("num_classes", 1000)
        threshold = ckpt.get("threshold")

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        model = build_model(num_classes, embed_dim, arch)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:    print(f"[CNNIrisEncoder] missing keys: {missing[:3]}")
        if unexpected: print(f"[CNNIrisEncoder] unexpected: {unexpected[:3]}")
        model.eval().to(device)
        print(f"[CNNIrisEncoder] loaded {arch}  d={embed_dim}  classes={num_classes}  device={device}")
        return cls(model, device, arch, embed_dim, threshold)

    def _preprocess(self, strip):
        """uint8 (H, W) -> float32 tensor matching the model's expected shape."""
        if strip is None: return None
        arr = np.asarray(strip, dtype=np.float32) / 255.0
        if self.arch == "resnet18_pretrained_3ch":
            # Replicate to 3 channels, ImageNet-normalize
            arr = np.stack([arr, arr, arr], axis=0)             # (3, H, W)
            arr = (arr - _IMNET_MEAN[:, None, None]) / _IMNET_STD[:, None, None]
            arr = arr[None, ...]                                 # (1, 3, H, W)
        else:
            arr = arr[None, None, :, :]                          # (1, 1, H, W)
        return arr

    def encode(self, strip):
        """Return a 1-D L2-normalized embedding for a single 64x512 strip."""
        torch = _torch()
        x = self._preprocess(strip)
        if x is None: return None
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            emb = self.model.embed(t).cpu().numpy()[0]
        return emb.astype(np.float32)

    @staticmethod
    def cosine_distance(a, b):
        """Cosine distance in [0, 2]; identical L2-normalized vectors -> 0."""
        if a is None or b is None: return 1.0
        return float(1.0 - np.dot(a, b))
