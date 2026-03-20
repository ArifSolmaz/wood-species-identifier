"""
🌲 Wood Species Identifier — Fast Cloud Edition (ONNX Runtime)
No PyTorch needed — installs in ~1 min instead of ~8 min
FSD-Trained ResNet50 · 112 Species · 97.32% Accuracy
TÜBİTAK 1002 Project
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════╗
# ║  UPDATE THESE GOOGLE DRIVE FILE IDS                      ║
# ║                                                          ║
# ║  1. Share fsd_resnet50.onnx → paste ID below             ║
# ║  2. Share fsd_classes.json  → paste ID below             ║
# ╚══════════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────

ONNX_FILE_ID = "14vTZ8iwgi45JAZZs1UzGY5GfHCeh7hKy"
CLASSES_FILE_ID = "17f2UxdMytTxQQRzaf9ZGFnXkPR328Xxv"

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
ONNX_PATH = MODEL_DIR / "fsd_resnet50.onnx"
CLASSES_PATH = MODEL_DIR / "fsd_classes.json"
CALIBRATION_TEMPERATURE = 1.1
OOD_CONFIDENCE_THRESHOLD = 0.30

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# ─────────────────────────────────────────────────────────────
# Page Config & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Wood Species Identifier", page_icon="🌲",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700; text-align: center; padding: 1rem 0 0.3rem;
        background: linear-gradient(90deg, #2E7D32, #8D6E63);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { text-align: center; color: #888; margin-bottom: 2rem; }
    .prediction-card {
        background: #1a1a2e; border-radius: 14px; padding: 1.8rem;
        border-left: 5px solid #4CAF50; margin-bottom: 1rem;
    }
    .species-name { font-size: 1.5rem; font-weight: 600; color: #4CAF50; }
    .confidence-high { color: #4CAF50; font-weight: 700; }
    .confidence-mid  { color: #FFC107; font-weight: 700; }
    .confidence-low  { color: #F44336; font-weight: 700; }
    .stat-card { background: #16213e; border-radius: 10px; padding: 1rem; text-align: center; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #4CAF50; }
    .stat-label { font-size: 0.8rem; color: #aaa; }
    .ood-warning {
        background: #3e1a1a; border-radius: 14px; padding: 1.5rem;
        border-left: 5px solid #F44336; margin-bottom: 1rem;
    }
    .mini-bar-bg { background: #333; border-radius: 4px; height: 8px; margin-top: 2px; }
    .mini-bar-fill { border-radius: 4px; height: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Download & Load
# ─────────────────────────────────────────────────────────────
def download_from_drive(file_id, dest_path):
    if dest_path.exists():
        return True
    if "PASTE" in file_id:
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest_path), quiet=False)
        return dest_path.exists()
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False


@st.cache_data
def load_species_db():
    db_path = DATA_DIR / "species_database.json"
    if db_path.exists():
        with open(db_path) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_model():
    with st.spinner("Downloading model (~95 MB)... This only happens once."):
        if not download_from_drive(ONNX_FILE_ID, ONNX_PATH):
            return None, None
        if not download_from_drive(CLASSES_FILE_ID, CLASSES_PATH):
            return None, None

    session = ort.InferenceSession(str(ONNX_PATH),
                                    providers=['CPUExecutionProvider'])

    with open(CLASSES_PATH) as f:
        meta = json.load(f)

    return session, meta['classes']


# ─────────────────────────────────────────────────────────────
# Preprocessing (pure numpy — no torch/albumentations needed)
# ─────────────────────────────────────────────────────────────
def preprocess(image):
    """Resize, normalize, and convert to NCHW float32 tensor."""
    img = image.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis, ...]   # Add batch dim → (1, 3, 224, 224)


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
def softmax(x, temperature=1.0):
    x = x / temperature
    e = np.exp(x - np.max(x))
    return e / e.sum()


def check_ood(probs):
    max_conf = float(np.max(probs))
    p = np.clip(probs, 1e-10, 1.0)
    entropy = -float(np.sum(p * np.log(p)))
    norm_ent = entropy / np.log(len(probs))
    is_ood = max_conf < OOD_CONFIDENCE_THRESHOLD or norm_ent > 0.85
    reason = ("Very low confidence — image may not be a wood micrograph"
              if max_conf < OOD_CONFIDENCE_THRESHOLD
              else "High uncertainty — species may not be in database"
              if norm_ent > 0.85 else "OK")
    return {"is_ood": is_ood, "max_confidence": max_conf,
            "normalized_entropy": norm_ent, "reason": reason}


def predict(session, image, classes, top_k=5, use_calibration=True):
    tensor = preprocess(image)
    logits = session.run(None, {"input": tensor})[0][0]
    T = CALIBRATION_TEMPERATURE if use_calibration else 1.0
    probs = softmax(logits, T)
    top_idx = np.argsort(probs)[-top_k:][::-1]
    ood = check_ood(probs)

    results = []
    for idx in top_idx:
        raw = classes[idx]
        parts = raw.split(" ", 1)
        clean = parts[1] if len(parts) > 1 and parts[0].isdigit() else raw
        results.append({"species": clean, "full_label": raw,
                        "confidence": float(probs[idx]), "class_idx": int(idx)})
    return results, ood


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def make_confidence_chart(results):
    species = [r["species"] for r in results][::-1]
    confs = [r["confidence"] * 100 for r in results][::-1]
    colors = ["#4CAF50" if c >= 50 else "#FFC107" if c >= 20 else "#F44336" for c in confs]
    fig = go.Figure(go.Bar(x=confs, y=species, orientation="h", marker_color=colors,
        text=[f"{c:.1f}%" for c in confs], textposition="outside", textfont=dict(size=14)))
    fig.update_layout(xaxis_title="Confidence (%)", xaxis_range=[0, 110],
        height=max(250, len(results) * 58), margin=dict(l=10, r=40, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color="#ddd"),
        yaxis=dict(tickfont=dict(size=13, color="#ddd")))
    return fig

def confidence_color(c):
    return "confidence-high" if c >= 0.7 else "confidence-mid" if c >= 0.4 else "confidence-low"

def confidence_emoji(c):
    return "🟢" if c >= 0.8 else "🟡" if c >= 0.5 else "🔴"

def render_species_info(label, db):
    info = db.get(label, {})
    if not info:
        return
    st.markdown("#### 📖 Species information")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | **Common name** | {info.get('common', 'N/A')} |
        | **Family** | {info.get('family', 'N/A')} |
        | **Group** | {info.get('group', 'N/A')} |
        | **Density** | {info.get('density_gcm3', 'N/A')} g/cm³ |
        | **Durability** | {info.get('durability', 'N/A')} |
        | **Region** | {info.get('region', 'N/A')} |
        | **Uses** | {info.get('uses', 'N/A')} |
        """)
    iawa = info.get("iawa", {})
    if iawa:
        with c2:
            st.markdown("**IAWA anatomical features**")
            st.markdown(f"""
            | Feature | Description |
            |---|---|
            | **Porosity** | {iawa.get('porosity', 'N/A')} |
            | **Vessel arrangement** | {iawa.get('vessel_arrangement', 'N/A')} |
            | **Ray width** | {iawa.get('ray_width', 'N/A')} |
            | **Parenchyma** | {iawa.get('parenchyma', 'N/A')} |
            | **Resin canals** | {iawa.get('resin_canals', 'N/A')} |
            | **Growth rings** | {iawa.get('growth_rings', 'N/A')} |
            """)


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">🔬 Wood Species Identifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">'
                'FSD Microscopic · ResNet50 · 112 Species · 97.32% Accuracy · '
                'IAWA Features · OOD Detection'
                '</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Number of predictions", 1, 10, 5)
        show_species_info = st.checkbox("Show species info & IAWA", value=True)
        use_calibration = st.checkbox("Calibrated confidence", value=True)
        show_ood = st.checkbox("OOD detection warnings", value=True)
        st.divider()
        st.header("ℹ️ Model")
        st.markdown("""
        | | |
        |---|---|
        | **Architecture** | ResNet50 |
        | **Dataset** | FSD Micro (2,240) |
        | **Species** | 112 |
        | **Top-1** | 97.32% |
        | **Top-3** | 99.11% |
        | **Top-5** | 99.70% |
        | **Runtime** | ONNX (fast) |
        """)
        st.divider()
        st.caption("TÜBİTAK 1002 · Streamlit + ONNX Runtime")

    session, classes = load_model()
    species_db = load_species_db()

    if session is None:
        if "PASTE" in ONNX_FILE_ID or "PASTE" in CLASSES_FILE_ID:
            st.error(
                "### ⚠️ Google Drive File IDs not configured!\n\n"
                "1. Run `convert_to_onnx.py` in Colab\n"
                "2. Share `fsd_resnet50.onnx` and `fsd_classes.json` on Drive\n"
                "3. Paste both file IDs in `app.py` lines 25-26")
        else:
            st.error("### ⚠️ Model download failed. Check file IDs and sharing permissions.")
        return

    st.sidebar.success("Model loaded — 🔵 ONNX CPU")

    # Upload
    st.markdown("### 📤 Upload micrograph(s)")
    uploaded_files = st.file_uploader(
        "Drop one or more wood micrograph images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True)

    if not uploaded_files:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        for col, val, lbl in [(c1, "112", "Species"), (c2, "97.3%", "Top-1 Acc"),
                               (c3, "99.7%", "Top-5 Acc")]:
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{val}'
                            f'</div><div class="stat-label">{lbl}</div></div>',
                            unsafe_allow_html=True)
        st.info("👆 Upload wood micrograph image(s) to get started.\n\n"
                "Best with **microscopic cross-sections** at ~100× magnification.")
        return

    # ─── SINGLE IMAGE ────────────────────────────────────
    if len(uploaded_files) == 1:
        file = uploaded_files[0]
        image = Image.open(file)
        col_img, col_gap, col_res = st.columns([1, 0.05, 1.2])

        with col_img:
            st.image(image, caption=file.name, width="stretch")
            st.caption(f"📐 {image.width}×{image.height} · {image.mode}")

        with col_res:
            t0 = time.time()
            results, ood = predict(session, image, classes, top_k, use_calibration)
            ms = (time.time() - t0) * 1000

            if show_ood and ood["is_ood"]:
                st.markdown(
                    f'<div class="ood-warning">'
                    f'<div style="color:#F44336;font-size:1.1rem;font-weight:600;">'
                    f'🚫 Out-of-distribution detected</div>'
                    f'<div style="color:#ccc;margin-top:0.3rem;">{ood["reason"]}</div>'
                    f'<div style="color:#888;font-size:0.8rem;margin-top:0.3rem;">'
                    f'Max conf: {ood["max_confidence"]*100:.1f}% · '
                    f'Entropy: {ood["normalized_entropy"]:.3f}</div></div>',
                    unsafe_allow_html=True)

            top = results[0]
            css = confidence_color(top["confidence"])
            emoji = confidence_emoji(top["confidence"])
            cal_tag = ' <span style="color:#888;font-size:0.75rem;">(calibrated)</span>' if use_calibration else ''

            st.markdown(
                f'<div class="prediction-card">'
                f'<div style="color:#aaa;font-size:0.85rem;">PREDICTED SPECIES</div>'
                f'<div class="species-name">🌿 <i>{top["species"]}</i></div>'
                f'<div style="margin-top:0.8rem;">Confidence: {emoji} '
                f'<span class="{css}" style="font-size:1.4rem;">{top["confidence"]*100:.1f}%</span>'
                f'{cal_tag}</div>'
                f'<div style="color:#666;font-size:0.8rem;margin-top:0.3rem;">'
                f'⚡ {ms:.0f} ms · {top["full_label"]}</div></div>',
                unsafe_allow_html=True)

            st.markdown(f"#### Top-{top_k} predictions")
            st.plotly_chart(make_confidence_chart(results))

        if show_species_info:
            st.markdown("---")
            render_species_info(results[0]["full_label"], species_db)

    # ─── BATCH MODE ──────────────────────────────────────
    else:
        st.markdown(f"### 🔄 Batch — {len(uploaded_files)} images")
        progress = st.progress(0, text="Classifying...")
        batch_results = []
        total_time = 0

        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            t0 = time.time()
            results, ood = predict(session, image, classes, top_k, use_calibration)
            total_time += time.time() - t0
            batch_results.append({"file": file, "image": image,
                                  "results": results, "ood": ood})
            progress.progress((i+1)/len(uploaded_files),
                              text=f"Classified {i+1}/{len(uploaded_files)}")
        progress.empty()

        avg_conf = np.mean([br["results"][0]["confidence"] for br in batch_results])
        ood_count = sum(1 for br in batch_results if br["ood"]["is_ood"])
        unique_sp = len(set(br["results"][0]["species"] for br in batch_results))

        s1, s2, s3, s4 = st.columns(4)
        for col, val, lbl in [(s1, str(len(batch_results)), "Images"),
                               (s2, f"{avg_conf*100:.1f}%", "Avg. confidence"),
                               (s3, str(unique_sp), "Unique species"),
                               (s4, str(ood_count), "OOD warnings")]:
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{val}'
                            f'</div><div class="stat-label">{lbl}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📋 Results")
        rows = []
        for br in batch_results:
            top = br["results"][0]
            rows.append({
                "Image": br["file"].name,
                "Status": "🟢" if not br["ood"]["is_ood"] else "🔴 OOD",
                "Species": top["species"],
                "Confidence": f"{top['confidence']*100:.1f}%",
                "Family": species_db.get(top["full_label"], {}).get("family", "-"),
                "Group": species_db.get(top["full_label"], {}).get("group", "-"),
            })
        st.dataframe(pd.DataFrame(rows))

        st.markdown("---")
        st.markdown("#### 🖼️ Individual results")
        for row_start in range(0, len(batch_results), 3):
            cols = st.columns(3)
            for col, br in zip(cols, batch_results[row_start:row_start+3]):
                with col:
                    st.image(br["image"], caption=br["file"].name, width="stretch")
                    top = br["results"][0]
                    css = confidence_color(top["confidence"])
                    emoji = confidence_emoji(top["confidence"])
                    st.markdown(
                        f'<div style="text-align:center;padding:0.3rem;">'
                        f'<div style="font-size:1.05rem;font-weight:600;">'
                        f'🌿 <i>{top["species"]}</i></div>'
                        f'<span class="{css}" style="font-size:1.2rem;">'
                        f'{emoji} {top["confidence"]*100:.1f}%</span></div>',
                        unsafe_allow_html=True)
                    for r in br["results"][:3]:
                        pct = r["confidence"] * 100
                        bc = "#4CAF50" if pct >= 50 else "#FFC107" if pct >= 20 else "#F44336"
                        st.markdown(
                            f'<div style="margin:3px 0;font-size:0.78rem;">'
                            f'<span style="color:#bbb;">{r["species"][:28]}</span>'
                            f'<span style="color:#888;float:right;">{pct:.1f}%</span>'
                            f'<div class="mini-bar-bg"><div class="mini-bar-fill" '
                            f'style="background:{bc};width:{min(pct,100)}%;"></div></div></div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Species distribution")
        sp_counts = pd.Series([br["results"][0]["species"]
                              for br in batch_results]).value_counts()
        fig = px.bar(x=sp_counts.index, y=sp_counts.values,
                     labels={"x": "Species", "y": "Count"},
                     color=sp_counts.values, color_continuous_scale="Greens")
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ddd"),
                          showlegend=False, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
