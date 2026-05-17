"""
Standalone training script: build the iris recognition model from disk.

Usage:
    python train_model.py
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.segmentation import (
    load_image, preprocess_image, detect_iris_and_pupil, normalize_iris,
)
from utils.iriscode import extract_iriscode
from utils.evaluation import (
    create_genuine_pairs, create_impostor_pairs,
    compute_far_frr, find_eer, compute_auc,
)


DATA_DIR = Path("data/CASIA-Iris-Thousand")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "iris_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _looks_like_subject_dir(p):
    """A subject folder contains images directly or via L/R subdirs."""
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_file() and child.suffix.lower() in IMG_EXT:
            return True
        if child.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMG_EXT
            for f in child.iterdir()
        ):
            return True
    return False


def _find_subjects_root(root):
    """Descend into single-child wrapper folders until we hit subject dirs."""
    cur = root
    for _ in range(4):
        kids = [p for p in cur.iterdir() if p.is_dir()]
        if kids and any(_looks_like_subject_dir(k) for k in kids):
            return cur
        if len(kids) == 1:
            cur = kids[0]
            continue
        break
    return cur


def discover_images(root):
    """Walk `root` and emit (subject_id, image_path) tuples."""
    out = []
    if not root.exists():
        return out
    real_root = _find_subjects_root(root)
    print(f"Scanning subjects under: {real_root}")
    for sub in sorted(p for p in real_root.iterdir() if p.is_dir()):
        subject_id = sub.name
        for path in sub.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMG_EXT:
                out.append((subject_id, path))
    return out


def process_image(path):
    """Full pipeline for a single image — returns the IrisCode or None."""
    img = load_image(path)
    if img is None:
        return None
    pre = preprocess_image(img)
    iris, pupil = detect_iris_and_pupil(pre)
    if iris is None or pupil is None:
        return None
    norm = normalize_iris(pre, iris, pupil)
    if norm is None:
        return None
    return extract_iriscode(norm)


def main():
    print("=" * 60)
    print(" Iris Recognition — Model Training")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"[ERROR] Dataset folder not found: {DATA_DIR}")
        print("        Download CASIA-Iris-Thousand and extract under data/.")
        sys.exit(1)

    images = discover_images(DATA_DIR)
    if not images:
        print(f"[ERROR] No images found under {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(images)} images across "
          f"{len({s for s, _ in images})} subjects.")

    # --- Process every image -----------------------------------------------
    iris_data, skipped = [], 0
    t0 = time.time()
    for subject_id, path in tqdm(images, desc="Encoding"):
        try:
            code = process_image(path)
            if code is None:
                skipped += 1
                continue
            iris_data.append({
                "subject_id": subject_id,
                "image_path": str(path),
                "iriscode": code,
            })
        except Exception as e:
            skipped += 1
            print(f"[skip] {path}: {e}")

    n_proc = len(iris_data)
    print(f"\nProcessed: {n_proc}   Skipped: {skipped}   "
          f"Success rate: {n_proc / max(len(images), 1):.1%}   "
          f"Elapsed: {time.time() - t0:.1f}s")

    if n_proc < 2:
        print("[ERROR] Not enough successful encodings to evaluate.")
        sys.exit(1)

    # --- Evaluation --------------------------------------------------------
    print("\nBuilding genuine pairs...")
    genuine = create_genuine_pairs(iris_data)
    print(f"  genuine pairs: {len(genuine)}")
    print("Building impostor pairs (balanced)...")
    impostor = create_impostor_pairs(iris_data, max_pairs=max(len(genuine), 1000))
    print(f"  impostor pairs: {len(impostor)}")

    thresholds = np.linspace(0.0, 1.0, 101)
    far, frr = compute_far_frr(genuine, impostor, thresholds)
    eer, eer_t, far_e, frr_e = find_eer(far, frr, thresholds)
    auc_value = compute_auc(far, frr)

    print("\n" + "-" * 60)
    print(f"  EER           : {eer:.4f}")
    print(f"  EER threshold : {eer_t:.4f}")
    print(f"  FAR @ EER     : {far_e:.4f}")
    print(f"  FRR @ EER     : {frr_e:.4f}")
    print(f"  AUC           : {auc_value:.4f}")
    print("-" * 60)

    # --- Save --------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"iris_data": iris_data, "threshold": eer_t}, f)
    print(f"Saved model → {MODEL_PATH}")

    metrics = {
        "eer": eer,
        "threshold": eer_t,
        "far_at_eer": far_e,
        "frr_at_eer": frr_e,
        "auc": auc_value,
        "num_subjects": len({d["subject_id"] for d in iris_data}),
        "num_images": n_proc,
        "num_skipped": skipped,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {METRICS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
