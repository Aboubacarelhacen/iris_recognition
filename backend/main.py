"""
FastAPI backend powering the React (TanStack Start) iris-recognition dashboard.

Endpoints (all under /api):

    GET  /api/health                 → liveness + model status
    GET  /api/summary                → Daugman + CNN metrics, dataset stats
    GET  /api/subjects               → list subjects with per-subject quality
    GET  /api/subjects/{id}          → one subject: codes, samples, intra-HD
    POST /api/verify                 → 2 image uploads → match result
    GET  /api/llm-analysis           → cached Ollama Turkish text
    POST /api/llm-analysis/regenerate→ call Ollama, persist, return
    GET  /api/training-history       → CNN per-epoch metrics (for charts)

The Vite dev server proxies /api/* here. Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# --- repo-root sys.path so `from utils.*` works regardless of CWD -------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.segmentation import (                                   # noqa: E402
    load_image, preprocess_image, detect_iris_and_pupil, normalize_iris,
)
from utils.iriscode import (                                       # noqa: E402
    extract_iriscode, hamming_distance, visualize_iriscode,
)

# CNN is optional — keep import lazy & wrapped
def _try_load_cnn():
    try:
        from utils.cnn_encoder import CNNIrisEncoder
        path = ROOT / "model" / "cnn_iris.pt"
        if not path.exists():
            return None
        return CNNIrisEncoder.load(path)
    except Exception as e:
        print(f"[CNN] unavailable: {e}")
        return None


# --------------------------------------------------------------- paths
MODEL_PATH        = ROOT / "model" / "iris_model.pkl"
METRICS_PATH      = ROOT / "model" / "metrics.json"
CNN_METRICS_PATH  = ROOT / "model" / "cnn_metrics.json"
LLM_PATH          = ROOT / "llm_analysis_output.txt"

DEFAULT_HD_THRESHOLD  = 0.32
DEFAULT_CNN_THRESHOLD = 0.84


# --------------------------------------------------------------- app + state
app = FastAPI(title="Iris Recognition API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class State:
    iris_data: Optional[list] = None
    daugman_threshold: float = DEFAULT_HD_THRESHOLD
    daugman_metrics: Optional[dict] = None
    cnn_metrics: Optional[dict] = None
    cnn = None     # CNNIrisEncoder instance
    subjects_index: dict = {}  # subject_id -> list[int] indices into iris_data


STATE = State()


def _safe_json_load(path: Path):
    if not path.exists(): return None
    try:    return json.loads(path.read_text())
    except Exception as e:
        print(f"[load] {path}: {e}"); return None


@app.on_event("startup")
def _startup():
    print("[startup] loading models...")
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        STATE.iris_data = obj.get("iris_data", [])
        STATE.daugman_threshold = float(obj.get("threshold", DEFAULT_HD_THRESHOLD))
        # build subject index
        idx = {}
        for i, it in enumerate(STATE.iris_data):
            idx.setdefault(it["subject_id"], []).append(i)
        STATE.subjects_index = idx
        print(f"[startup] loaded {len(STATE.iris_data)} iris codes "
              f"across {len(idx)} subjects")
    else:
        print(f"[startup] no Daugman model at {MODEL_PATH}")

    STATE.daugman_metrics = _safe_json_load(METRICS_PATH)
    STATE.cnn_metrics = _safe_json_load(CNN_METRICS_PATH)
    STATE.cnn = _try_load_cnn()
    print(f"[startup] CNN: {'loaded' if STATE.cnn else 'unavailable'}")


# --------------------------------------------------------------- helpers
def _decode_pil(buf: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(buf)).convert("L")
    return np.array(img)


def _png_b64(arr: np.ndarray) -> str:
    """Encode a 2D uint8 array as base64 PNG (for <img src='data:image/png;base64,...'>)."""
    if arr is None: return ""
    ok, png = cv2.imencode(".png", arr)
    if not ok: return ""
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


def _run_pipeline(gray: np.ndarray):
    """Return dict with norm/code/emb (None on failure)."""
    pre = preprocess_image(gray)
    iris, pupil = detect_iris_and_pupil(pre)
    if iris is None or pupil is None:
        return None
    norm = normalize_iris(pre, iris, pupil)
    if norm is None: return None
    code = extract_iriscode(norm)
    emb = STATE.cnn.encode(norm) if STATE.cnn is not None else None
    return {"pre": pre, "iris": iris, "pupil": pupil,
            "norm": norm, "code": code, "emb": emb}


def _subject_intra_hd(subject_id: str, max_pairs: int = 20) -> Optional[float]:
    """Average Hamming distance among same-subject codes (consistency metric)."""
    indices = STATE.subjects_index.get(subject_id, [])
    if len(indices) < 2: return None
    items = [STATE.iris_data[i]["iriscode"] for i in indices]
    pairs = list(combinations(items, 2))[:max_pairs]
    if not pairs: return None
    return float(np.mean([hamming_distance(a, b) for a, b in pairs]))


def _quality_level(intra_hd: Optional[float]) -> str:
    """Bucket a subject by intra-subject HD: lower = more consistent = higher quality.

    Thresholds tuned to give a useful spread on CASIA-Iris-Thousand; tighten if
    your dataset's distribution shifts.
    """
    if intra_hd is None: return "unknown"
    if intra_hd < 0.20: return "high"
    if intra_hd < 0.25: return "medium"
    return "low"


# --------------------------------------------------------------- schemas
class HealthOut(BaseModel):
    status: str
    n_subjects: int
    n_codes: int
    has_daugman: bool
    has_cnn: bool


class SummaryOut(BaseModel):
    n_subjects: int
    n_codes: int
    daugman: dict
    cnn: dict
    quality_buckets: dict
    top_consistent_subjects: list


class SubjectRow(BaseModel):
    subject_id: str
    n_codes: int
    intra_hd: Optional[float] = None
    quality: str


class VerifyOut(BaseModel):
    daugman: dict
    cnn: Optional[dict] = None
    artifacts: dict


# --------------------------------------------------------------- routes
@app.get("/api/health", response_model=HealthOut)
def health():
    return HealthOut(
        status="ok",
        n_subjects=len(STATE.subjects_index),
        n_codes=len(STATE.iris_data or []),
        has_daugman=bool(STATE.iris_data),
        has_cnn=STATE.cnn is not None,
    )


@app.get("/api/summary")
def summary():
    n = len(STATE.iris_data or [])
    subj_ids = list(STATE.subjects_index.keys())

    # Per-subject intra-HD on a sample (full would be slow on huge sets)
    sample = subj_ids[:200]
    consistencies = []
    for sid in sample:
        h = _subject_intra_hd(sid, max_pairs=10)
        if h is not None:
            consistencies.append((sid, h, len(STATE.subjects_index[sid])))

    buckets = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for _, h, _ in consistencies:
        buckets[_quality_level(h)] += 1

    consistencies.sort(key=lambda t: t[1])
    top_consistent = [
        {"subject_id": sid, "intra_hd": float(h), "n_codes": k, "quality": _quality_level(h)}
        for sid, h, k in consistencies[:8]
    ]

    daugman_m = STATE.daugman_metrics or {}
    cnn_m = STATE.cnn_metrics or {}
    cnn_thr = cnn_m.get("threshold") or cnn_m.get("thr") or DEFAULT_CNN_THRESHOLD

    return {
        "n_subjects": len(subj_ids),
        "n_codes": n,
        "daugman": {
            "available": bool(STATE.iris_data),
            "eer": daugman_m.get("eer"),
            "threshold": STATE.daugman_threshold,
            "far_at_eer": daugman_m.get("far_at_eer"),
            "frr_at_eer": daugman_m.get("frr_at_eer"),
            "auc": daugman_m.get("auc"),
            "num_images": daugman_m.get("num_images"),
            "num_skipped": daugman_m.get("num_skipped"),
        },
        "cnn": {
            "available": STATE.cnn is not None,
            "eer": cnn_m.get("eer"),
            "threshold": cnn_thr,
            "far_at_eer": cnn_m.get("far_at_eer"),
            "frr_at_eer": cnn_m.get("frr_at_eer"),
            "auc": cnn_m.get("auc"),
            "n_gen": cnn_m.get("n_gen"),
            "n_imp": cnn_m.get("n_imp"),
        },
        "quality_buckets": buckets,
        "top_consistent_subjects": top_consistent,
        "has_llm_analysis": LLM_PATH.exists(),
    }


@app.get("/api/subjects")
def subjects(limit: int = 100, offset: int = 0,
             sort: str = "consistency_asc", q: Optional[str] = None):
    """List subjects with per-subject quality. Sort by consistency or count."""
    rows = []
    for sid, indices in STATE.subjects_index.items():
        if q and q not in sid: continue
        h = _subject_intra_hd(sid, max_pairs=10)
        rows.append({
            "subject_id": sid,
            "n_codes": len(indices),
            "intra_hd": h,
            "quality": _quality_level(h),
        })

    keyf = {
        "consistency_asc":  lambda r: (r["intra_hd"] is None, r["intra_hd"] or 0.0),
        "consistency_desc": lambda r: -(r["intra_hd"] or 0.0),
        "codes_desc":       lambda r: -r["n_codes"],
        "id_asc":           lambda r: r["subject_id"],
    }.get(sort, lambda r: r["subject_id"])
    rows.sort(key=keyf)

    total = len(rows)
    return {"count": total, "items": rows[offset:offset + limit]}


@app.get("/api/subjects/{subject_id}")
def subject_detail(subject_id: str):
    if subject_id not in STATE.subjects_index:
        raise HTTPException(404, f"subject {subject_id} not found")
    indices = STATE.subjects_index[subject_id]
    items = [STATE.iris_data[i] for i in indices]
    h = _subject_intra_hd(subject_id, max_pairs=50)

    # Build small bitmap previews of each code
    codes_preview = []
    for it in items[:24]:
        vis = visualize_iriscode(it["iriscode"])
        codes_preview.append({
            "image_path": it.get("image_path", ""),
            "code_png": _png_b64(vis),
        })

    return {
        "subject_id": subject_id,
        "n_codes": len(indices),
        "intra_hd": h,
        "quality": _quality_level(h),
        "codes": codes_preview,
    }


@app.post("/api/verify", response_model=VerifyOut)
async def verify(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    """Compare two iris images. Returns Daugman HD, CNN cosine, and visual artifacts."""
    if not STATE.iris_data:
        raise HTTPException(503, "Daugman model not loaded")

    a_bytes = await image_a.read()
    b_bytes = await image_b.read()
    try:
        a_gray, b_gray = _decode_pil(a_bytes), _decode_pil(b_bytes)
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    a = _run_pipeline(a_gray)
    b = _run_pipeline(b_gray)
    if a is None or b is None:
        raise HTTPException(422, "segmentation failed on one of the images")

    hd = hamming_distance(a["code"], b["code"])
    cos = None
    if a["emb"] is not None and b["emb"] is not None:
        cos = float(1.0 - np.dot(a["emb"], b["emb"]))

    cnn_thr = (STATE.cnn_metrics or {}).get("threshold") \
        or (STATE.cnn_metrics or {}).get("thr") \
        or (STATE.cnn.threshold if STATE.cnn else DEFAULT_CNN_THRESHOLD)

    return {
        "daugman": {
            "hd": float(hd),
            "threshold": STATE.daugman_threshold,
            "match": bool(hd < STATE.daugman_threshold),
        },
        "cnn": None if cos is None else {
            "cosine_distance": cos,
            "threshold": float(cnn_thr),
            "match": bool(cos < cnn_thr),
        },
        "artifacts": {
            "strip_a": _png_b64(a["norm"]),
            "strip_b": _png_b64(b["norm"]),
            "code_a": _png_b64(visualize_iriscode(a["code"])),
            "code_b": _png_b64(visualize_iriscode(b["code"])),
        },
    }


@app.get("/api/llm-analysis")
def get_llm_analysis():
    if not LLM_PATH.exists():
        return {"available": False, "text": None}
    return {"available": True, "text": LLM_PATH.read_text(encoding="utf-8")}


class RegenIn(BaseModel):
    model: str = "llama3.2:latest"
    timeout: int = 120


@app.post("/api/llm-analysis/regenerate")
def regen_llm(payload: RegenIn = RegenIn()):
    if not STATE.daugman_metrics and not STATE.cnn_metrics:
        raise HTTPException(503, "no metrics available to summarize")

    m = STATE.daugman_metrics or {}
    cm = STATE.cnn_metrics or {}
    prompt = f"""Aşağıdaki iris tanıma sistemi değerlendirme sonuçlarını analiz et ve **Türkçe** bir rapor üret.
Raporu şu başlıklarla yaz:

## Bulgular
## Tartışma
## Sonuç
## Gelecek Çalışmalar

Sayısal sonuçlar:
- Konu (kişi) sayısı: {m.get('num_subjects', '?')}
- İşlenen görüntü sayısı: {m.get('num_images', '?')}
- Atlanılan görüntü sayısı: {m.get('num_skipped', '?')}

Daugman IrisCode:
- EER: {m.get('eer', '?')}
- EER eşiği: {m.get('threshold', '?')}
- AUC: {m.get('auc', '?')}

CNN (ResNet18 + ArcFace):
- EER: {cm.get('eer', '?')}
- Eşik: {cm.get('threshold') or cm.get('thr', '?')}
- AUC: {cm.get('auc', '?')}

Akademik bir tonla, açıklayıcı ve eleştirel bir analiz yap."""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": payload.model, "prompt": prompt, "stream": False},
            timeout=payload.timeout,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, "Ollama not running at localhost:11434")
    except Exception as e:
        raise HTTPException(500, f"Ollama failed: {e}")

    LLM_PATH.write_text(text, encoding="utf-8")
    return {"available": True, "text": text}


@app.get("/api/training-history")
def training_history():
    """Per-epoch CNN metrics for the training-curve chart."""
    cm = STATE.cnn_metrics or {}
    return {"history": cm.get("history", [])}
