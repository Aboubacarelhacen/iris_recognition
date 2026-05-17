"""
Train a CNN iris embedding (ResNet18 + ArcFace) on CASIA-Iris-Thousand.

Works locally (CPU / Apple MPS) and on Kaggle (CUDA T4). Steps:

    1.  Walk the dataset, segment + rubber-sheet unwrap every image, and cache
        the (strip, subject_id) tensors to model/strips_cache.npz so re-runs
        skip the slow CPU work.
    2.  Open-set split: 80% of subjects -> train, 20% -> held-out validation
        (the val subjects are NEVER seen during training, mirroring real
        biometric deployment).
    3.  Train ResNet18 + ArcFace with AdamW + cosine LR.
    4.  On val: compute genuine / impostor cosine-distance distributions, sweep
        thresholds, report EER + AUC.
    5.  Save model/cnn_iris.pt (encoder + arcface weights + meta).

Usage:
    python train_cnn.py                       # local
    python train_cnn.py --epochs 30 --batch 128
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.segmentation import (
    load_image, preprocess_image, detect_iris_and_pupil, normalize_iris,
)
from utils.cnn_encoder import build_model


DEFAULT_DATA = Path("data/CASIA-Iris-Thousand")
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)
CACHE_PATH = MODEL_DIR / "strips_cache.npz"
CKPT_PATH = MODEL_DIR / "cnn_iris.pt"
METRICS_PATH = MODEL_DIR / "cnn_metrics.json"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
STRIP_H, STRIP_W = 64, 512
EMBED_DIM = 256


# ------------------------------------------------------------------ discovery
def _looks_like_subject_dir(p):
    if not p.is_dir(): return False
    for child in p.iterdir():
        if child.is_file() and child.suffix.lower() in IMG_EXT: return True
        if child.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMG_EXT for f in child.iterdir()
        ): return True
    return False


def _find_subjects_root(root):
    cur = root
    for _ in range(4):
        kids = [p for p in cur.iterdir() if p.is_dir()]
        if kids and any(_looks_like_subject_dir(k) for k in kids):
            return cur
        if len(kids) == 1:
            cur = kids[0]; continue
        break
    return cur


def discover(root):
    out = []
    if not root.exists(): return out
    real = _find_subjects_root(root)
    print(f"[discover] subjects root: {real}")
    for sub in sorted(p for p in real.iterdir() if p.is_dir()):
        for path in sub.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMG_EXT:
                out.append((sub.name, path))
    return out


# -------------------------------------------------------- build / cache strips
def encode_strip(path):
    img = load_image(path)
    if img is None: return None
    pre = preprocess_image(img)
    iris, pupil = detect_iris_and_pupil(pre)
    if iris is None or pupil is None: return None
    return normalize_iris(pre, iris, pupil, STRIP_H, STRIP_W)


def build_cache(data_dir, max_per_subject=None):
    """Encode every image to a 64x512 strip and persist to disk."""
    if CACHE_PATH.exists():
        print(f"[cache] reusing {CACHE_PATH}")
        z = np.load(CACHE_PATH, allow_pickle=True)
        return z["strips"], z["labels"], list(z["subjects"])

    pairs = discover(data_dir)
    if not pairs:
        print(f"[ERROR] no images under {data_dir}")
        sys.exit(1)

    strips, labels, subjects = [], [], {}
    per_subj_count = {}
    t0 = time.time()
    for sid, path in tqdm(pairs, desc="encoding strips"):
        if max_per_subject and per_subj_count.get(sid, 0) >= max_per_subject:
            continue
        strip = encode_strip(path)
        if strip is None: continue
        if sid not in subjects: subjects[sid] = len(subjects)
        strips.append(strip.astype(np.uint8))
        labels.append(subjects[sid])
        per_subj_count[sid] = per_subj_count.get(sid, 0) + 1

    strips = np.stack(strips, 0)
    labels = np.asarray(labels, dtype=np.int64)
    subj_list = sorted(subjects, key=lambda k: subjects[k])
    print(f"[cache] {strips.shape} in {time.time()-t0:.0f}s")
    np.savez_compressed(CACHE_PATH, strips=strips, labels=labels,
                        subjects=np.array(subj_list))
    return strips, labels, subj_list


# ----------------------------------------------------------- dataset / loader
def make_loaders(strips, labels, subjects, batch, seed=42):
    """Open-set split by subject; returns train_loader, val_strips, val_labels."""
    import torch
    from torch.utils.data import Dataset, DataLoader

    rng = np.random.default_rng(seed)
    all_subj = np.arange(len(subjects))
    rng.shuffle(all_subj)
    n_val = max(1, int(0.20 * len(all_subj)))
    val_subj = set(all_subj[:n_val].tolist())

    is_val = np.array([lb in val_subj for lb in labels])
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    # Remap train labels to a dense range [0, n_train_classes)
    train_labels_old = labels[train_idx]
    uniq = sorted(set(train_labels_old.tolist()))
    remap = {old: new for new, old in enumerate(uniq)}
    train_labels = np.array([remap[l] for l in train_labels_old], dtype=np.int64)
    n_train_classes = len(uniq)

    print(f"[split] train: {len(train_idx)} imgs / {n_train_classes} subjects   "
          f"val: {len(val_idx)} imgs / {len(val_subj)} subjects")

    class StripDS(Dataset):
        def __init__(self, X, y, augment=False):
            self.X, self.y, self.aug = X, y, augment
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            img = self.X[i].astype(np.float32) / 255.0
            if self.aug:
                # circular angular shift (rotation tolerance)
                shift = np.random.randint(-32, 33)
                img = np.roll(img, shift, axis=1)
                # brightness jitter
                img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
            return torch.from_numpy(img).unsqueeze(0), int(self.y[i])

    train_ds = StripDS(strips[train_idx], train_labels, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    return train_loader, n_train_classes, val_idx, labels[val_idx]


# ----------------------------------------------------------------- evaluation
def evaluate(model, strips_val, labels_val, device, batch=128):
    """Compute embeddings on val, then genuine/impostor cosine distances."""
    import torch
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(strips_val), batch):
            x = strips_val[i:i+batch].astype(np.float32) / 255.0
            t = torch.from_numpy(x).unsqueeze(1).to(device)
            embs.append(model.embed(t).cpu().numpy())
    E = np.concatenate(embs, 0)  # (N, D), already L2-normalized

    # cosine distance = 1 - dot
    sims = E @ E.T
    iu = np.triu_indices_from(sims, k=1)
    sim_pairs = sims[iu]
    same = (labels_val[iu[0]] == labels_val[iu[1]])
    dist = 1.0 - sim_pairs
    genuine = dist[same]; impostor = dist[~same]
    if len(genuine) == 0 or len(impostor) == 0:
        return {"eer": 1.0, "auc": 0.5, "n_gen": int(same.sum()),
                "n_imp": int((~same).sum()), "threshold": 0.5}

    # Balance impostor count for fair sampling
    if len(impostor) > 50 * len(genuine):
        idx = np.random.default_rng(0).choice(len(impostor),
                                              50 * len(genuine), replace=False)
        impostor = impostor[idx]

    thresholds = np.linspace(0, 2, 401)
    far = np.array([(impostor < t).mean() for t in thresholds])
    frr = np.array([(genuine >= t).mean() for t in thresholds])
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2)
    thr = float(thresholds[idx])

    # AUC
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    # higher score = same person -> use similarity (1 - dist)
    y_score = np.concatenate([1 - genuine, 1 - impostor])
    auc = float(roc_auc_score(y_true, y_score))

    return {"eer": eer, "auc": auc, "threshold": thr,
            "n_gen": int(len(genuine)), "n_imp": int(len(impostor)),
            "far_at_eer": float(far[idx]), "frr_at_eer": float(frr[idx])}


# ----------------------------------------------------------------------- main
def pick_device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-per-subject", type=int, default=None,
                        help="optional cap to speed up CPU runs")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    print("=" * 60); print(" CNN Iris Embedding — ArcFace / ResNet18")
    print("=" * 60)

    strips, labels, subjects = build_cache(args.data, args.max_per_subject)
    train_loader, n_classes, val_idx, val_labels = make_loaders(
        strips, labels, subjects, args.batch,
    )

    device = pick_device()
    print(f"[train] device: {device}   classes: {n_classes}   "
          f"epochs: {args.epochs}   batch: {args.batch}")

    model = build_model(n_classes, EMBED_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best = {"eer": 1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _ = model(x, y)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0); n += x.size(0)
            pbar.set_postfix(loss=f"{running/n:.4f}")
        sched.step()

        # quick eval on a subsample of val for speed
        sample = val_idx
        if len(sample) > 4000:
            sample = np.random.default_rng(epoch).choice(val_idx, 4000, replace=False)
        m = evaluate(model, strips[sample],
                     val_labels[: 0] if len(sample) == 0 else labels[sample],
                     device)
        print(f"  val EER={m['eer']*100:.2f}%  AUC={m['auc']:.4f}  "
              f"thr={m['threshold']:.3f}   (n_gen={m['n_gen']}, n_imp={m['n_imp']})")

        if m["eer"] < best["eer"]:
            best = m
            torch.save({
                "model": model.state_dict(),
                "num_classes": n_classes,
                "embed_dim": EMBED_DIM,
                "threshold": m["threshold"],
                "subjects_meta": subjects,
            }, CKPT_PATH)
            print(f"  [+] best EER so far -> saved {CKPT_PATH}")

    # ----- final full eval -----
    print("\n[final] evaluating on full val set...")
    final = evaluate(model, strips[val_idx], val_labels, device)
    print(f"FINAL:  EER={final['eer']*100:.2f}%   AUC={final['auc']:.4f}   "
          f"threshold={final['threshold']:.3f}")
    final["best_during_training"] = best
    METRICS_PATH.write_text(json.dumps(final, indent=2))
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
