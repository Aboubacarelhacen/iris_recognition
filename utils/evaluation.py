"""
Evaluation utilities: genuine/impostor pairs, FAR/FRR sweep, EER, AUC.
"""

import random
from itertools import combinations

import numpy as np
from sklearn.metrics import auc as sk_auc

from .iriscode import hamming_distance


def create_genuine_pairs(iris_data):
    """
    Build the list of HD scores for all same-subject pairs.

    Args:
        iris_data: list of dicts with keys "subject_id" and "iriscode".

    Returns:
        list[float] of Hamming distances.
    """
    by_subject = {}
    for i, item in enumerate(iris_data):
        by_subject.setdefault(item["subject_id"], []).append(i)

    scores = []
    for indices in by_subject.values():
        if len(indices) < 2:
            continue
        for a, b in combinations(indices, 2):
            scores.append(
                hamming_distance(iris_data[a]["iriscode"], iris_data[b]["iriscode"])
            )
    return scores


def create_impostor_pairs(iris_data, max_pairs=None, seed=42):
    """
    Build the list of HD scores for different-subject pairs.

    Random sampling is used so the impostor set can be balanced with
    `max_pairs` (typically the number of genuine pairs).
    """
    n = len(iris_data)
    if n < 2:
        return []

    rng = random.Random(seed)
    target = max_pairs if max_pairs is not None else n * 4
    scores = []
    attempts = 0
    seen = set()

    while len(scores) < target and attempts < target * 20:
        attempts += 1
        i, j = rng.sample(range(n), 2)
        if iris_data[i]["subject_id"] == iris_data[j]["subject_id"]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        scores.append(
            hamming_distance(iris_data[i]["iriscode"], iris_data[j]["iriscode"])
        )
    return scores


def compute_far_frr(genuine_scores, impostor_scores, thresholds):
    """
    Compute FAR / FRR at each threshold.

    FAR(t) = P(impostor HD <  t)   (accepted-as-genuine impostors)
    FRR(t) = P(genuine  HD >= t)   (rejected-as-impostor genuines)
    """
    g = np.asarray(genuine_scores, dtype=float)
    i = np.asarray(impostor_scores, dtype=float)
    far_list, frr_list = [], []
    for t in thresholds:
        far_list.append(float(np.mean(i < t)) if i.size else 0.0)
        frr_list.append(float(np.mean(g >= t)) if g.size else 0.0)
    return far_list, frr_list


def find_eer(far_list, frr_list, thresholds):
    """
    Locate the Equal Error Rate point.

    Returns:
        (eer, eer_threshold, far_at_eer, frr_at_eer)
    """
    far = np.asarray(far_list, dtype=float)
    frr = np.asarray(frr_list, dtype=float)
    diffs = np.abs(far - frr)
    idx = int(np.argmin(diffs))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(thresholds[idx]), float(far[idx]), float(frr[idx])


def compute_auc(far_list, frr_list):
    """
    Compute ROC AUC from FAR/FRR sweep.

    ROC plots TPR = 1 - FRR against FPR = FAR. We sort by FPR before
    integrating to keep sk_auc happy.
    """
    fpr = np.asarray(far_list, dtype=float)
    tpr = 1.0 - np.asarray(frr_list, dtype=float)
    order = np.argsort(fpr)
    return float(sk_auc(fpr[order], tpr[order]))
