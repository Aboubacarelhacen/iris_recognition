"""
Streamlit dashboard for the iris recognition system.

Run:
    streamlit run app.py
"""

import io
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.segmentation import (
    preprocess_image, detect_iris_and_pupil, normalize_iris,
)
from utils.iriscode import extract_iriscode, hamming_distance, visualize_iriscode


MODEL_PATH = Path("model/iris_model.pkl")
METRICS_PATH = Path("model/metrics.json")
CNN_PATH = Path("model/cnn_iris.pt")
CNN_METRICS_PATH = Path("model/cnn_metrics.json")
LLM_PATH = Path("llm_analysis_output.txt")
DEFAULT_THRESHOLD = 0.32
DEFAULT_CNN_THRESHOLD = 0.6


# ---------------------------------------------------------------- styling
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.2rem; border-radius: 12px; border: 1px solid #374151;
    }
    .result-ok  { background:#064e3b; color:#d1fae5; padding:1.5rem;
                  border-radius:12px; text-align:center; font-size:1.6rem;
                  font-weight:700; border:2px solid #10b981;}
    .result-no  { background:#7f1d1d; color:#fee2e2; padding:1.5rem;
                  border-radius:12px; text-align:center; font-size:1.6rem;
                  font-weight:700; border:2px solid #ef4444;}
    h1, h2, h3 { color: #f9fafb; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    """Load pickled iris_data + threshold; return None if missing/broken."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource
def load_metrics():
    """Load metrics.json if present."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return None


@st.cache_resource
def load_cnn():
    """Lazy-load the CNN encoder (returns None if torch / checkpoint missing)."""
    if not CNN_PATH.exists():
        return None
    try:
        from utils.cnn_encoder import CNNIrisEncoder
        return CNNIrisEncoder.load(CNN_PATH)
    except ModuleNotFoundError as e:
        st.warning(f"CNN unavailable — install torch/torchvision to enable. ({e})")
        return None
    except Exception as e:
        st.error(f"Failed to load CNN: {e}")
        return None


@st.cache_resource
def load_cnn_metrics():
    if not CNN_METRICS_PATH.exists():
        return None
    try:
        return json.loads(CNN_METRICS_PATH.read_text())
    except Exception:
        return None


def pil_to_gray(pil_image):
    """Convert an uploaded PIL image to a grayscale numpy array."""
    arr = np.array(pil_image.convert("L"))
    return arr


def full_pipeline(gray_image):
    """Run full pipeline; returns dict with intermediate artifacts or None."""
    pre = preprocess_image(gray_image)
    iris, pupil = detect_iris_and_pupil(pre)
    if iris is None or pupil is None:
        return None
    norm = normalize_iris(pre, iris, pupil)
    if norm is None:
        return None
    code = extract_iriscode(norm)
    return {"pre": pre, "iris": iris, "pupil": pupil, "norm": norm, "code": code}


# ---------------------------------------------------------------- pages
def page_home(metrics, cnn_metrics=None):
    st.title("🔐 Iris Recognition System")
    st.markdown(
        "An implementation of **Daugman's IrisCode** pipeline — segmentation "
        "via Hough circles, polar rubber-sheet normalization, 2-D Gabor "
        "encoding, and rotation-tolerant Hamming-distance matching."
    )

    if metrics is None:
        st.warning(
            "⚠️ Model not trained yet. Run `python train_model.py` "
            "(or execute `iris_recognition.ipynb`) to generate "
            "`model/iris_model.pkl` and `model/metrics.json`."
        )
        return

    st.subheader("System Performance")
    cols = st.columns(4)
    cols[0].metric("EER", f"{metrics['eer']*100:.2f} %")
    cols[1].metric("FAR @ EER", f"{metrics['far_at_eer']*100:.2f} %")
    cols[2].metric("FRR @ EER", f"{metrics['frr_at_eer']*100:.2f} %")
    cols[3].metric("AUC", f"{metrics['auc']:.4f}")

    st.markdown("---")
    st.markdown(
        f"**Subjects:** {metrics.get('num_subjects', '?')}  •  "
        f"**Encoded images:** {metrics.get('num_images', '?')}  •  "
        f"**Decision threshold:** `{metrics['threshold']:.3f}`"
    )

    if cnn_metrics:
        st.markdown("### CNN Embedding (ResNet18 + ArcFace)")
        cc = st.columns(3)
        cc[0].metric("CNN EER",       f"{cnn_metrics['eer']*100:.2f} %")
        cc[1].metric("Threshold",     f"{cnn_metrics.get('threshold') or cnn_metrics.get('thr', 0):.3f}")
        cc[2].metric("AUC",           f"{(cnn_metrics.get('auc') or 0):.4f}")
    else:
        st.info("ℹ️ Train the CNN branch with `python train_cnn.py` "
                "(or run `train_cnn_kaggle.ipynb` on Kaggle GPU) to unlock the "
                "CNN matcher in the Verification page.")

    if st.button("→ Go to Verification"):
        st.session_state["page"] = "Verification"
        st.rerun()


def _decide(score, threshold):
    """Return (match, confidence) given a distance score and threshold."""
    match = score < threshold
    if match:
        conf = max(0.0, min(1.0, 1.0 - score / max(threshold, 1e-6)))
    else:
        conf = max(0.0, min(1.0, (score - threshold) / max(2.0 - threshold, 1e-6)))
    return match, conf


def _verdict_box(match, label, score):
    cls = "result-ok" if match else "result-no"
    icon = "🟢" if match else "🔴"
    text = "SAME PERSON" if match else "DIFFERENT PERSON"
    st.markdown(
        f"<div class='{cls}'>{icon} {text}<br><span style='font-size:0.7em'>"
        f"{label} = {score:.4f}</span></div>",
        unsafe_allow_html=True,
    )


def page_verify(metrics, model, cnn, cnn_metrics):
    st.title("🔍 Iris Verification")

    # ---------- matcher selection ----------
    available = ["Daugman IrisCode"]
    if cnn is not None: available += ["CNN Embedding", "Both (compare)"]
    matcher = st.radio("Matching method", available, horizontal=True)
    if cnn is None:
        st.caption("CNN matcher is hidden — drop `model/cnn_iris.pt` "
                   "(from train_cnn.py or the Kaggle notebook) to enable it.")

    hd_thr = (metrics or {}).get("threshold", DEFAULT_THRESHOLD)
    cnn_thr = ((cnn_metrics or {}).get("threshold")
               or (cnn_metrics or {}).get("thr")
               or (cnn.threshold if cnn is not None else DEFAULT_CNN_THRESHOLD))

    col1, col2 = st.columns(2)
    with col1:
        f1 = st.file_uploader("Iris image #1", type=["jpg", "jpeg", "png", "bmp"], key="i1")
    with col2:
        f2 = st.file_uploader("Iris image #2", type=["jpg", "jpeg", "png", "bmp"], key="i2")

    if not (f1 and f2):
        st.info("Upload two iris images to compare.")
        return

    img1 = Image.open(io.BytesIO(f1.read()))
    img2 = Image.open(io.BytesIO(f2.read()))
    c1, c2 = st.columns(2)
    c1.image(img1, caption="Image #1", use_container_width=True)
    c2.image(img2, caption="Image #2", use_container_width=True)

    if not st.button("🔍 Verify Identity", type="primary"):
        return

    with st.spinner("Segmenting and normalizing both images..."):
        try:
            r1 = full_pipeline(pil_to_gray(img1))
            r2 = full_pipeline(pil_to_gray(img2))
        except Exception as e:
            st.error(f"Processing failed: {e}")
            return

    if r1 is None or r2 is None:
        st.error("Could not segment one of the images — try a clearer iris photo.")
        return

    # ---------- compute scores ----------
    hd = hamming_distance(r1["code"], r2["code"])
    cos = None
    if matcher in ("CNN Embedding", "Both (compare)") and cnn is not None:
        with st.spinner("Computing CNN embeddings..."):
            try:
                e1 = cnn.encode(r1["norm"])
                e2 = cnn.encode(r2["norm"])
                cos = cnn.cosine_distance(e1, e2)
            except Exception as e:
                st.error(f"CNN inference failed: {e}")
                cos = None

    # ---------- verdict ----------
    if matcher == "Daugman IrisCode":
        match, conf = _decide(hd, hd_thr)
        _verdict_box(match, "Hamming Distance", hd)
        st.progress(max(0.0, 1.0 - hd), text=f"Similarity: {(1-hd)*100:.1f} %")
        st.caption(f"Threshold: {hd_thr:.3f} • Confidence: **{conf*100:.1f} %**")

    elif matcher == "CNN Embedding":
        if cos is None:
            st.error("CNN score unavailable.")
            return
        match, conf = _decide(cos, cnn_thr)
        _verdict_box(match, "Cosine Distance", cos)
        st.progress(max(0.0, min(1.0, 1.0 - cos / 2.0)),
                    text=f"Similarity: {(1-cos/2)*100:.1f} %")
        st.caption(f"Threshold: {cnn_thr:.3f} • Confidence: **{conf*100:.1f} %**")

    else:  # Both
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Daugman IrisCode**")
            m, _ = _decide(hd, hd_thr)
            _verdict_box(m, "HD", hd)
            st.caption(f"thr {hd_thr:.3f}")
        with col_b:
            st.markdown("**CNN Embedding**")
            if cos is None:
                st.error("CNN unavailable")
            else:
                m, _ = _decide(cos, cnn_thr)
                _verdict_box(m, "Cosine", cos)
                st.caption(f"thr {cnn_thr:.3f}")
        if cos is not None:
            agree = (hd < hd_thr) == (cos < cnn_thr)
            st.success("Both methods agree ✓" if agree
                       else "Methods disagree — handle with care ⚠️")

    # ---------- artifacts ----------
    st.markdown("#### Normalized Iris Strips")
    c1, c2 = st.columns(2)
    c1.image(r1["norm"], caption="Strip #1", use_container_width=True, clamp=True)
    c2.image(r2["norm"], caption="Strip #2", use_container_width=True, clamp=True)

    st.markdown("#### IrisCodes")
    c1, c2 = st.columns(2)
    c1.image(visualize_iriscode(r1["code"]), caption="Code #1",
             use_container_width=True, clamp=True)
    c2.image(visualize_iriscode(r2["code"]), caption="Code #2",
             use_container_width=True, clamp=True)


def page_explorer(model):
    st.title("📚 Dataset Explorer")
    if model is None:
        st.warning("Model not loaded. Train it first with `python train_model.py`.")
        return

    iris_data = model["iris_data"]
    subjects = sorted({d["subject_id"] for d in iris_data})

    c1, c2 = st.columns(2)
    c1.metric("Subjects", len(subjects))
    c2.metric("Total IrisCodes", len(iris_data))

    sel = st.selectbox("Select a subject", subjects)
    items = [d for d in iris_data if d["subject_id"] == sel]
    st.write(f"**{len(items)}** IrisCodes for subject **{sel}**")

    cols = st.columns(min(4, max(1, len(items))))
    for i, item in enumerate(items[:8]):
        vis = visualize_iriscode(item["iriscode"])
        cols[i % len(cols)].image(vis, caption=Path(item.get("image_path", "")).name,
                                  clamp=True, use_container_width=True)

    # Average intra-subject HD (consistency metric)
    if len(items) >= 2:
        from itertools import combinations
        hds = [hamming_distance(a["iriscode"], b["iriscode"])
               for a, b in combinations(items, 2)]
        st.metric("Average intra-subject HD", f"{np.mean(hds):.4f}")
    else:
        st.info("Need at least 2 samples for an intra-subject HD.")


def page_info(metrics, cnn_metrics=None):
    st.title("ℹ️ System Info")
    st.markdown(
        """
### Daugman's Algorithm — Step by Step

1. **Segmentation** — locate the inner (pupil) and outer (iris) circular
   boundaries using the Hough Circle Transform.
2. **Normalization** — unwrap the annular iris region to a fixed
   rectangle `(radial × angular) = (64 × 512)` via the **rubber sheet** model.
3. **Encoding** — convolve the strip with a bank of **2-D Gabor filters**
   (4 orientations, quadrature pair) and keep the sign of each response →
   a **2048-bit IrisCode**.
4. **Matching** — compare two codes by **normalized Hamming Distance**,
   minimized over a small set of angular shifts to absorb rotation.
5. **Decision** — accept if `HD < threshold`, else reject.

### Pipeline
```
   eye image
       │
   CLAHE preprocess
       │
   HoughCircles  ──►  (iris, pupil)
       │
   Rubber-sheet unwrap  ──►  64 × 512 strip
       │
   Gabor filter bank (4 orientations, quadrature)
       │
   Binarize sign  ──►  2048-bit IrisCode
       │
   Hamming Distance  ──►  decision
```
        """
    )

    if metrics:
        st.subheader("Daugman IrisCode — Metrics")
        st.json(metrics)
    if cnn_metrics:
        st.subheader("CNN Embedding — Metrics")
        st.json(cnn_metrics)

    if LLM_PATH.exists():
        st.subheader("🤖 Ollama LLM Analysis (Türkçe)")
        try:
            st.markdown(LLM_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not read {LLM_PATH}: {e}")


# ---------------------------------------------------------------- main
def main():
    metrics = load_metrics()
    model = load_model()
    cnn = load_cnn()
    cnn_metrics = load_cnn_metrics()

    st.sidebar.title("Navigation")
    pages = ["Home", "Verification", "Explorer", "System Info"]
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    choice = st.sidebar.radio(
        "Page", pages, index=pages.index(st.session_state["page"])
    )
    st.session_state["page"] = choice

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models**")
    st.sidebar.write("Daugman: " + ("✅" if model else "⛔ not trained"))
    st.sidebar.write("CNN:      " + ("✅" if cnn else "⛔ not loaded"))
    if metrics:
        st.sidebar.caption(f"HD  • EER {metrics['eer']*100:.2f}%   "
                           f"AUC {metrics['auc']:.3f}")
    if cnn_metrics:
        st.sidebar.caption(f"CNN • EER {cnn_metrics['eer']*100:.2f}%   "
                           f"thr {cnn_metrics.get('threshold') or cnn_metrics.get('thr', 0):.3f}")

    if choice == "Home":
        page_home(metrics, cnn_metrics)
    elif choice == "Verification":
        page_verify(metrics, model, cnn, cnn_metrics)
    elif choice == "Explorer":
        page_explorer(model)
    elif choice == "System Info":
        page_info(metrics, cnn_metrics)


if __name__ == "__main__":
    main()
