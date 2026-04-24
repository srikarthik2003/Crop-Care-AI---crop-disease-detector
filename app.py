# app.py
import os
import io
import json
import time
import base64
import pathlib
import datetime as dt
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import requests
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Crop Care • AI", page_icon="🌱", layout="wide")

# ---- Paths (edit these) ----
# For local dev: set this to your local .h5 file path
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobileNet_crop_disease_model_v1 (1).h5")

# Remote model URL (set via Streamlit Cloud Secrets as MODEL_URL)
# Leave empty to use LOCAL_MODEL_PATH
MODEL_URL = os.getenv("MODEL_URL", "")

CLASSES_TXT = os.path.join(os.path.dirname(__file__), "Classes.txt")  # one class per line (in training order)
CLASSES_JSON = "classes.json"         # optional: {"class_names": [...]}
PRED_LOG = "predictions_log.csv"      # dashboard data store

# ---- Model input size (should match training) ----
IMG_SIZE = (256, 256)  # MobileNetV2 default in your training
CHANNELS = 3

# ---- Class names (fallback) ----
# IMPORTANT: Replace this with the exact list in training order if you don't provide a file.
CLASS_NAMES = [
    # Example:
     "Corn Crop Diseases", "Cotton Crop Diseases", "Fruit Crop Diseases", "Pulse Crop",
     "Rice plant Diseases", "Tobacco Crop Diseases", "Vegetable Crop Diseases", "Wheat Diseases"
]

# ======================
# UTILITIES
# ======================
def load_class_names():
    """
    Return list of class names in the exact order used during training.
    Priority:
      1) classes.json -> {"class_names": [...]}
      2) classes.txt  -> one class per line
      3) fallback CLASS_NAMES list in this file
    """
    # if os.path.exists(CLASSES_JSON):
    #     with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     names = data.get("class_names", [])
    #     if names:
    #         return names
    # if os.path.exists(CLASSES_TXT):
    #     with open(CLASSES_TXT, "r", encoding="utf-8") as f:
    #         names = [ln.strip() for ln in f.readlines() if ln.strip()]
    #     if names:
    #         return names
    return CLASS_NAMES

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def preprocess_image(pil_img: Image.Image, target_size=(224, 224)):
    """
    - auto-orient using EXIF
    - ensure RGB
    - resize with letterbox (maintain aspect), then center-crop to target
    - scale to [0,1]
    """
    img = ImageOps.exif_transpose(pil_img)
    img = ensure_rgb(img)

    # letterbox to keep aspect ratio
    img.thumbnail(target_size, Image.LANCZOS)
    # paste centered on white canvas to reach exact target_size
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    x = (target_size[0] - img.size[0]) // 2
    y = (target_size[1] - img.size[1]) // 2
    canvas.paste(img, (x, y))
    arr = np.asarray(canvas).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr, canvas  # return processed array and the display image

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-9, None)

def is_diseased(class_name: str) -> bool:
    """Heuristic: decide Healthy vs Diseased based on class name."""
    tokens = class_name.lower()
    unhealthy_keywords = ["disease", "diseases", "blight", "rust", "mildew", "leaf spot", "bacterial", "viral", "wilt", "infect"]
    return any(k in tokens for k in unhealthy_keywords)

def crop_type_from_class(class_name: str) -> str:
    """
    Try to extract crop type from a label like:
    'Rice plant Diseases', 'Wheat Diseases', 'Corn Crop Diseases', 'Tomato___Late_blight'
    """
    if "___" in class_name:
        crop = class_name.split("___")[0].strip()
        return crop
    # lightweight heuristics
    words = class_name.replace("plant", "").replace("Crop", "").replace("Diseases", "").replace("Disease", "")
    words = words.replace("_", " ").strip()
    # take first word as crop (works for 'Rice', 'Wheat', 'Corn', etc.)
    return words.split()[0] if words else class_name

def log_prediction(filename, pred_class, confidence, healthy_flag, crop_type):
    row = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "filename": filename,
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "healthy": bool(healthy_flag),
        "crop_type": crop_type,
    }
    if os.path.exists(PRED_LOG):
        df = pd.read_csv(PRED_LOG)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(PRED_LOG, index=False)

# ======================
# MODEL DOWNLOADER
# ======================
def get_model_path() -> str:
    """
    Return path to local model file.
    Downloads from MODEL_URL to LOCAL_MODEL_PATH if needed.
    """
    if MODEL_URL and not os.path.exists(LOCAL_MODEL_PATH):
        st.info("Downloading model for the first time… this may take a moment.")
        try:
            import os
            # Ensure directory exists
            os.makedirs(os.path.dirname(LOCAL_MODEL_PATH) or ".", exist_ok=True)
            # Use gdown for reliable Google Drive downloads
            gdown.download(MODEL_URL, LOCAL_MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {MODEL_URL}: {e}")
    return LOCAL_MODEL_PATH


# ======================
# CACHED LOADERS
# ======================
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached():
    path = get_model_path()
    return load_model(path)

@st.cache_resource
def get_class_names_cached():
    names = load_class_names()
    # if not names:
    #     st.warning("No class names found. Please fill CLASS_NAMES in the code, or provide classes.txt / classes.json.")
    return names

# ======================
# HEADER / STYLE
# ======================
st.markdown(
    """
    <style>
      .big-metric {font-size: 36px; font-weight: 700; margin-top: -10px;}
      .small-note {font-size: 12px; color:#888;}
      .stButton>button {border-radius: 10px; padding: 0.6rem 1rem;}
      .result-box {padding: 1rem; border: 1px solid #eee; border-radius: 12px; background:#f9fafb;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
# SIDEBAR NAV
# ======================
pages = {
    "Home": "🏠 Home",
    "Dashboard": "📊 Dashboard",
    "About": "ℹ️ About",
}
choice = st.sidebar.radio("Navigate", list(pages.values()))

# Allow TTA toggle in sidebar
use_tta = st.sidebar.checkbox("Use Test-Time Augmentation (TTA)", value=False,
                              help="Runs 5 random flips/zooms and averages predictions.")

# ======================
# HOME
# ======================
if choice == pages["Home"]:
    st.title("🌱 Crop Care — AI Disease Detector & Crop Type Classifier")
    st.write("Upload a leaf image. The model predicts the **class** and infers **Healthy vs Diseased**, plus the **crop type**.")

    model = None
    class_names = get_class_names_cached()

    colA, colB = st.columns([2, 1])
    with colB:
        st.caption("Model file")
        display_path = MODEL_URL if MODEL_URL else LOCAL_MODEL_PATH
        st.code(display_path)
        try:
            model = load_model_cached()
        except Exception as e:
            st.error(f"Model unavailable: {e}")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None and model is not None:
        pil_img = Image.open(uploaded)
        arr, disp_img = preprocess_image(pil_img, IMG_SIZE)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(disp_img, caption="Input", use_container_width=True)

        # Prediction
        with col2:
            st.subheader("Prediction")
            if use_tta:
                preds = []
                # 5 augmentations: identity + flips/zoom crops
                for i in range(5):
                    a = arr.copy()
                    # random horizontal flip
                    if np.random.rand() < 0.5:
                        a = a[:, :, ::-1, :]
                    # tiny random zoom by cropping a border
                    if np.random.rand() < 0.5:
                        pad = np.random.randint(0, 8)
                        a = a[:, pad:IMG_SIZE[0]-pad, pad:IMG_SIZE[1]-pad, :]
                        a = tf.image.resize(a, IMG_SIZE)
                    preds.append(model.predict(a, verbose=0))
                y = np.mean(np.vstack(preds), axis=0)
            else:
                y = model.predict(arr, verbose=0)

            # ...existing code...
            probs = softmax(y)
            if probs.ndim == 1:
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
            elif probs.ndim == 2:
                top_idx = int(np.argmax(probs[0]))
                confidence = float(probs[0][top_idx])
            else:
                top_idx = 0
                confidence = float(probs)
            pred_class = class_names[top_idx] if class_names and top_idx < len(class_names) else f"Class #{top_idx}"
            # ...existing code...

            healthy_flag = not is_diseased(pred_class)
            crop = crop_type_from_class(pred_class)

            # Log to CSV for dashboard
            safe_name = getattr(uploaded, "name", f"upload_{int(time.time())}.png")
            log_prediction(safe_name, pred_class, confidence, healthy_flag, crop)

            # Nice result box
            color = "#090101" if healthy_flag else "#1a140b"
            st.markdown(f"<div class='result-box' style='background:{color}'>"
                        f"<div><b>Predicted Class:</b> {pred_class}</div>"
                        f"<div><b>Crop Type:</b> {crop}</div>"
                        f"<div><b>Health:</b> {'✅ Healthy' if healthy_flag else '⚠️ Diseased'}</div>"
                        f"<div><b>Confidence:</b> {(confidence*100):.2f}%</div>"
                        f"</div>", unsafe_allow_html=True)

            # Helpful tips
            if healthy_flag:
                st.success("Looks healthy. Keep monitoring. Consider periodic checks and proper nutrient balance.")
            else:
                st.warning("Signs of disease detected. Inspect multiple leaves; consider targeted treatment and isolate if necessary.")

    st.markdown("---")

# ======================
# DASHBOARD
# ======================
elif choice == pages["Dashboard"]:
    st.title("📊 Dashboard — Health Overview")

    if not os.path.exists(PRED_LOG):
        st.info("No predictions logged yet. Make some predictions on the Home page first.")
    else:
        df = pd.read_csv(PRED_LOG)
        if df.empty:
            st.info("No predictions yet.")
        else:
            # KPIs
            total = len(df)
            healthy = int(df["healthy"].sum())
            diseased = total - healthy

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Images", total)
            c2.metric("Healthy", healthy)
            c3.metric("Diseased", diseased)

            st.write(" ")
            colA, colB = st.columns(2)

            with colA:
                st.subheader("Health Split")
                pie = df["healthy"].replace({True: "Healthy", False: "Diseased"}).value_counts()
                st.pyplot(pie.plot(kind="pie", autopct="%1.1f%%", ylabel="").get_figure())
            with colB:
                st.subheader("Top Predicted Classes")
                top_classes = df["predicted_class"].value_counts().head(10)
                st.bar_chart(top_classes)

            st.subheader("Per-Crop Counts")
            crop_counts = df["crop_type"].value_counts()
            st.bar_chart(crop_counts)

            st.subheader("Recent Predictions")
            st.dataframe(df.sort_values("timestamp", ascending=False).head(25), use_container_width=True)

            st.download_button(
                "Download Logs (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions_log.csv",
                mime="text/csv",
            )

# ======================
# ABOUT
# ======================
elif choice == pages["About"]:
    st.title("ℹ️ About — Crop Care AI")
    st.markdown("""
### Project Overview
Crop Care AI detects plant diseases from leaf images and identifies the crop type.  
It leverages transfer learning (MobileNetV2 / VGG variants) and careful fine-tuning to deliver reliable predictions even with moderate datasets.

### How It Helps Users
- **Early detection** → reduces yield loss and treatment cost.
- **Actionable** → outputs crop type and health status to guide next steps.
- **Accessible** → lightweight model suitable for phones or low-cost devices.
- **Data-driven** → dashboard summarizes field activity and helps spotting trends.

### Pros
- Fast, easy, and low-cost compared to lab diagnostics.  
- Works offline (once the model is on-device).  
- Extensible: add new crops/diseases by retraining.

### Cons
- Accuracy depends on image quality and dataset coverage.  
- Visually similar diseases (e.g., leaf spots) can confuse models.  
- Not a substitute for expert agronomy in critical cases.

### Best Practices for Users
- Take photos in good light, focus on the leaf, and avoid heavy shadows.
- Capture multiple leaves and angles for reliability.
- Use results as **decision support**, not a sole authority.

### Future Work
- Multi-disease detection on the same leaf.
- Region-aware recommendations (weather, soil, stage).
- On-device continual learning and active learning loops.
- Explainability overlays (heatmaps) to show “why” a decision was made.
""")
    st.info("Have ideas or need a custom feature? Add a feedback button and store messages to a sheet or database.")

st.markdown(
    """
    <style>
    /* Main text */
    .stApp {
        color: white;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700; /* Gold */
    }
    /* Paragraphs */
    p {
        color: lightgrey;
    }
    </style>
    """,
    unsafe_allow_html=True
)
