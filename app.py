"""
🌲 Wood Species Identifier — Streamlit Cloud Edition
FSD-Trained ResNet50 · 112 Species · 97.32% Accuracy
Grad-CAM · IAWA Features · OOD Detection · Confidence Calibration
TÜBİTAK 1002 Project
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
# ║  CONFIGURATION — UPDATE THE GOOGLE DRIVE FILE ID BELOW  ║
# ╚══════════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────

# To get the file ID:
# 1. Open Google Drive → find fsd_BEST_resnet50_full.pth
# 2. Right-click → Share → "Anyone with the link"
# 3. Copy the link — it looks like:
#    https://drive.google.com/file/d/XXXXXXXXXXXXXXX/view?usp=sharing
# 4. The XXXXXXXXXXXXXXX part is your FILE_ID — paste it below

GOOGLE_DRIVE_FILE_ID = "1EA-sG1ue4inaIvcDP8yFRLLKAzrhPbE_"  # <── UPDATE THIS

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_FILENAME = "fsd_BEST_resnet50_full.pth"
OOD_ENTROPY_THRESHOLD = 3.0
OOD_CONFIDENCE_THRESHOLD = 0.30   # legacy single-threshold (still referenced)
# Tiered confidence cut-offs (used by check_ood)
TIER_OOD_CONF_HARD       = 0.15   # below this => almost certainly out-of-distribution
TIER_OOD_CONF_SOFT       = 0.30   # below this AND high entropy/low margin => OOD
TIER_OOD_ENTROPY         = 0.85   # normalized entropy above this => model has no preference
TIER_OOD_MARGIN          = 1.5    # top-1 / top-2 ratio below this => no clear winner
TIER_UNCERTAIN_CONF      = 0.50   # below this (and not OOD) => "low confidence" yellow tier
CALIBRATION_TEMPERATURE = 1.1

# ─────────────────────────────────────────────────────────────
# Page Config & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wood Species Identifier",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stat-card {
        background: #16213e; border-radius: 10px; padding: 1rem; text-align: center;
    }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #4CAF50; }
    .stat-label { font-size: 0.8rem; color: #aaa; }
    .ood-warning {
        background: #3e1a1a; border-radius: 14px; padding: 1.5rem;
        border-left: 5px solid #F44336; margin-bottom: 1rem;
    }
    .uncertain-warning {
        background: #3e3a1a; border-radius: 14px; padding: 1.5rem;
        border-left: 5px solid #FFC107; margin-bottom: 1rem;
    }
    .mini-bar-bg { background: #333; border-radius: 4px; height: 8px; margin-top: 2px; }
    .mini-bar-fill { border-radius: 4px; height: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model Download & Loading
# ─────────────────────────────────────────────────────────────
def download_model_from_drive():
    """Download model from Google Drive if not already present."""
    model_path = MODEL_DIR / MODEL_FILENAME
    if model_path.exists():
        return model_path

    if GOOGLE_DRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE":
        return None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        with st.spinner("📥 Downloading the wood-identification model (~98 MB) — "
                        "this only needs to happen the first time you open the app."):
            gdown.download(url, str(model_path), quiet=True)
        if model_path.exists():
            return model_path
    except Exception as e:
        st.error(f"Download failed: {e}")

    return None


@st.cache_data(show_spinner="📚 Loading the species reference database…")
def load_species_db():
    db_path = DATA_DIR / "species_database.json"
    if db_path.exists():
        with open(db_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner="🔬 Loading IAWA wood-anatomy reference data…")
def load_iawa_templates():
    """Load family-level IAWA templates and species-level overrides."""
    tpl_path = DATA_DIR / "iawa_templates.json"
    if tpl_path.exists():
        with open(tpl_path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("family_templates", {}),
            data.get("species_overrides", {}),
            data.get("schema_hardwood", []),
            data.get("schema_softwood", []),
        )
    return {}, {}, [], []


def merge_iawa(label, info, templates, overrides):
    """Merge family template + species iawa + species override into a full feature dict."""
    family = info.get("family", "")
    base = dict(templates.get(family, {}))
    species_iawa = info.get("iawa", {}) or {}
    # Per-species values from species_database.json take precedence over family defaults
    base.update({k: v for k, v in species_iawa.items() if v not in (None, "")})
    # Per-species overrides from iawa_templates.json have the highest precedence
    base.update(overrides.get(label, {}))
    return base


@st.cache_resource(show_spinner="🌲 Preparing the wood-identification model…")
def load_model():
    model_path = download_model_from_drive()
    if model_path is None or not model_path.exists():
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint["fsd_classes"]
    num_classes = checkpoint["num_classes_fsd"]

    model = models.resnet50(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, classes, device


def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def overlay_gradcam(image, cam, alpha=0.5):
    import matplotlib.cm as cm
    img_np = np.array(image.resize((224, 224)).convert("RGB")).astype(float) / 255.0
    heatmap = cm.jet(cam)[:, :, :3]
    overlay = np.clip((1 - alpha) * img_np + alpha * heatmap, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────
# OOD Detection & Calibration
# ─────────────────────────────────────────────────────────────
def calibrate_probs(logits, temperature=CALIBRATION_TEMPERATURE):
    return F.softmax(logits / temperature, dim=1)


def check_ood(probs):
    """Three-tier confidence assessment:
       - "ood":       image likely outside the model's training distribution
       - "uncertain": in-distribution but model is hesitant (low confidence, but
                      a clear-ish top candidate or moderate entropy)
       - "ok":        confident prediction, no warning shown
    """
    sorted_probs = np.sort(probs)[::-1]
    max_conf = float(sorted_probs[0])
    second_conf = float(sorted_probs[1]) if len(sorted_probs) > 1 else 1e-10
    margin = max_conf / max(second_conf, 1e-10)

    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped))
    norm_entropy = float(entropy / np.log(len(probs)))

    # OOD: very low absolute confidence, OR low confidence + flat distribution + no clear winner
    if (max_conf < TIER_OOD_CONF_HARD
        or (max_conf < TIER_OOD_CONF_SOFT
            and norm_entropy > TIER_OOD_ENTROPY
            and margin < TIER_OOD_MARGIN)):
        tier = "ood"
        reason = ("This image doesn't look like the wood micrographs the model was "
                  "trained on (very flat probability distribution). The prediction "
                  "below is unlikely to be correct.")
    elif max_conf < TIER_UNCERTAIN_CONF:
        tier = "uncertain"
        if margin >= 2.0:
            sub = (f"the top guess is about {margin:.1f}× more likely than the next, "
                   "so the model has a clear preference but isn't fully certain")
        elif norm_entropy > 0.75:
            sub = "the model is splitting probability across several similar-looking candidates"
        else:
            sub = "the model is hesitating between a few similar candidates"
        reason = (f"The model is not very confident — {sub}. "
                  "The image looks like a normal wood micrograph; please cross-check "
                  "the top-3 list with another method or an expert.")
    else:
        tier = "ok"
        reason = "OK"

    return {
        "tier": tier,
        "is_ood": tier == "ood",
        "is_uncertain": tier == "uncertain",
        "max_confidence": max_conf,
        "second_confidence": second_conf,
        "margin": margin,
        "normalized_entropy": norm_entropy,
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, image, classes, device, transform, top_k=5, use_calibration=True):
    img_np = np.array(image.convert("RGB"))
    tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
    output = model(tensor)
    probs = (calibrate_probs(output) if use_calibration
             else torch.softmax(output, dim=1)).cpu().numpy()[0]
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


def predict_with_gradcam(model, image, classes, device, transform, top_k=5):
    img_np = np.array(image.convert("RGB"))
    tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
    tensor.requires_grad = True
    gradcam = GradCAM(model, model.layer4[-1])
    output = model(tensor)
    class_idx = output.argmax(dim=1).item()
    cam = gradcam.generate(tensor, class_idx)
    cam_overlay = overlay_gradcam(image, cam)
    probs = calibrate_probs(output.detach()).cpu().numpy()[0]
    top_idx = np.argsort(probs)[-top_k:][::-1]
    ood = check_ood(probs)
    results = []
    for idx in top_idx:
        raw = classes[idx]
        parts = raw.split(" ", 1)
        clean = parts[1] if len(parts) > 1 and parts[0].isdigit() else raw
        results.append({"species": clean, "full_label": raw,
                        "confidence": float(probs[idx]), "class_idx": int(idx)})
    return results, ood, cam_overlay, cam


# ─────────────────────────────────────────────────────────────
# Visualization Helpers
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


# ─────────────────────────────────────────────────────────────
# IAWA tab rendering
# ─────────────────────────────────────────────────────────────
# Pretty labels for the merged IAWA dict
_IAWA_LABELS = {
    # General / growth rings
    "growth_rings": "Growth rings",
    # Vessels / porosity (hardwoods + Ephedra)
    "porosity": "Porosity",
    "vessel_arrangement": "Vessel arrangement",
    "vessel_diameter_um": "Vessel Ø (μm)",
    "vessel_frequency_per_mm2": "Vessel frequency (per mm²)",
    "perforation_plates": "Perforation plates",
    "intervessel_pits": "Intervessel pits",
    "vessel_ray_pits": "Vessel-ray pits",
    "helical_thickenings_vessels": "Helical thickenings (vessels)",
    "tyloses": "Tyloses",
    "vessel_deposits": "Vessel deposits",
    # Tracheids (softwoods)
    "tracheid_diameter_um": "Tracheid Ø (μm)",
    "tracheid_pitting": "Tracheid pitting",
    "earlywood_latewood_transition": "EW→LW transition",
    "torus_margo_pits": "Torus-margo pits",
    # Fibres / parenchyma
    "fibres": "Fibres",
    "fibre_wall_thickness": "Fibre wall thickness",
    "parenchyma": "Axial parenchyma",
    "axial_parenchyma": "Axial parenchyma",
    # Rays
    "ray_width": "Ray width",
    "ray_height_cells": "Ray height (cells)",
    "ray_composition": "Ray composition",
    "ray_tracheids": "Ray tracheids",
    "cross_field_pitting": "Cross-field pitting",
    "aggregate_rays": "Aggregate rays",
    # Special
    "helical_thickenings_tracheids": "Helical thickenings (tracheids)",
    "storied_structure": "Storied structure",
    "crystals": "Crystals",
    "silica": "Silica bodies",
    "resin_canals": "Resin canals",
    "notes": "Diagnostic notes",
}

_HARDWOOD_VESSEL_KEYS = ["porosity", "vessel_arrangement", "vessel_diameter_um",
                        "vessel_frequency_per_mm2", "perforation_plates",
                        "intervessel_pits", "vessel_ray_pits",
                        "helical_thickenings_vessels", "tyloses", "vessel_deposits"]

_HARDWOOD_PARENCHYMA_FIBRE_KEYS = ["fibres", "fibre_wall_thickness", "parenchyma"]

_HARDWOOD_RAY_KEYS = ["ray_width", "ray_height_cells", "ray_composition", "aggregate_rays"]

_HARDWOOD_SPECIAL_KEYS = ["storied_structure", "crystals", "silica", "resin_canals"]

_SOFTWOOD_TRACHEID_KEYS = ["porosity", "vessel_arrangement", "tracheid_diameter_um",
                          "tracheid_pitting", "earlywood_latewood_transition",
                          "torus_margo_pits"]

_SOFTWOOD_RAY_KEYS = ["ray_width", "ray_height_cells", "ray_composition",
                     "ray_tracheids", "cross_field_pitting"]

_SOFTWOOD_RESIN_KEYS = ["axial_parenchyma", "resin_canals",
                       "helical_thickenings_tracheids", "crystals"]


def _iawa_table(iawa, keys):
    """Render a Markdown table for the given keys, skipping missing/empty values."""
    rows = []
    for k in keys:
        val = iawa.get(k)
        if val in (None, "", "N/A"):
            continue
        rows.append(f"| **{_IAWA_LABELS.get(k, k)}** | {val} |")
    if not rows:
        return None
    return "| Feature | Value |\n|---|---|\n" + "\n".join(rows)


def render_species_info(label, db, templates, overrides):
    info = db.get(label, {})
    if not info:
        return
    iawa = merge_iawa(label, info, templates, overrides)
    is_softwood = info.get("group", "") in ("Softwood", "Gymnosperm")

    st.markdown("#### 📖 Species information")

    if is_softwood:
        tab_general, tab_tracheids, tab_rays, tab_resin, tab_notes = st.tabs([
            "📋 General",
            "🔬 Tracheids & growth",
            "📐 Rays & cross-field",
            "🌲 Parenchyma, resin & crystals",
            "📝 Notes",
        ])
    else:
        tab_general, tab_vessels, tab_pf, tab_rays, tab_special, tab_notes = st.tabs([
            "📋 General",
            "🔬 Vessels & porosity",
            "🪵 Parenchyma & fibres",
            "📐 Rays",
            "💎 Inclusions & special",
            "📝 Notes",
        ])

    # General tab — two columns to avoid the empty right side we used to have
    with tab_general:
        gc_left, gc_right = st.columns(2)
        with gc_left:
            st.markdown("**Identification**")
            st.markdown(f"""
| | |
|---|---|
| **Common name** | {info.get('common', 'N/A')} |
| **Family** | {info.get('family', 'N/A')} |
| **Group** | {info.get('group', 'N/A')} |
| **Region** | {info.get('region', 'N/A')} |
| **Uses** | {info.get('uses', 'N/A')} |
""")
        with gc_right:
            st.markdown("**Physical & macro features**")
            # Pick the most useful "at-a-glance" anatomical features for the right column
            if is_softwood:
                key_pairs = [
                    ("Density",          f"{info.get('density_gcm3', 'N/A')} g/cm³"),
                    ("Durability",       info.get('durability', 'N/A')),
                    ("Growth rings",     iawa.get('growth_rings', 'N/A')),
                    ("Porosity",         iawa.get('porosity', 'N/A')),
                    ("Resin canals",     iawa.get('resin_canals', 'N/A')),
                    ("Cross-field pits", iawa.get('cross_field_pitting', 'N/A')),
                ]
            else:
                key_pairs = [
                    ("Density",            f"{info.get('density_gcm3', 'N/A')} g/cm³"),
                    ("Durability",         info.get('durability', 'N/A')),
                    ("Growth rings",       iawa.get('growth_rings', 'N/A')),
                    ("Porosity",           iawa.get('porosity', 'N/A')),
                    ("Vessel arrangement", iawa.get('vessel_arrangement', 'N/A')),
                    ("Parenchyma",         iawa.get('parenchyma', 'N/A')),
                ]
            tbl = "| | |\n|---|---|\n" + "\n".join(
                f"| **{k}** | {v} |" for k, v in key_pairs if v not in (None, "", "N/A"))
            st.markdown(tbl)
        # Diagnostic note inline (full text also appears under "Notes" tab)
        if iawa.get("notes"):
            st.caption("📌 " + iawa["notes"])

    if is_softwood:
        with tab_tracheids:
            tbl = _iawa_table(iawa, _SOFTWOOD_TRACHEID_KEYS)
            st.markdown(tbl or "_No tracheid data available._")
        with tab_rays:
            tbl = _iawa_table(iawa, _SOFTWOOD_RAY_KEYS)
            st.markdown(tbl or "_No ray data available._")
        with tab_resin:
            tbl = _iawa_table(iawa, _SOFTWOOD_RESIN_KEYS)
            st.markdown(tbl or "_No data available._")
        with tab_notes:
            note = iawa.get("notes")
            if note:
                st.info(note)
            else:
                st.caption("_No diagnostic notes recorded._")
    else:
        with tab_vessels:
            tbl = _iawa_table(iawa, _HARDWOOD_VESSEL_KEYS)
            st.markdown(tbl or "_No vessel data available._")
        with tab_pf:
            tbl = _iawa_table(iawa, _HARDWOOD_PARENCHYMA_FIBRE_KEYS)
            st.markdown(tbl or "_No parenchyma/fibre data available._")
        with tab_rays:
            tbl = _iawa_table(iawa, _HARDWOOD_RAY_KEYS)
            st.markdown(tbl or "_No ray data available._")
        with tab_special:
            tbl = _iawa_table(iawa, _HARDWOOD_SPECIAL_KEYS)
            st.markdown(tbl or "_No inclusions / special features recorded._")
        with tab_notes:
            note = iawa.get("notes")
            if note:
                st.info(note)
            else:
                st.caption("_No diagnostic notes recorded._")


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">🔬 Wood Species Identifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">'
                'FSD Microscopic · ResNet50 · 112 Species · 97.32% Accuracy · '
                'Grad-CAM · IAWA · OOD Detection'
                '</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Number of predictions to show", 1, 10, 5,
            help="How many candidate species to list, ranked from most to least likely. "
                 "5 is a good default — the correct species is in the top 5 about 99.7% of the time.")
        show_gradcam = st.checkbox(
            "Show heatmap of focus regions", value=True,
            help="Highlights the parts of the micrograph the model paid most attention to "
                 "when making its prediction. Red/yellow = strong focus. "
                 "(Technical name: Grad-CAM.)")
        show_species_info = st.checkbox(
            "Show species info & wood anatomy (IAWA)", value=True,
            help="After identification, display the species' common name, family, density, "
                 "regions, uses, and IAWA wood-anatomy features (porosity, vessels, rays, "
                 "parenchyma, etc.) drawn from standard references.")
        use_calibration = st.checkbox(
            "Use calibrated confidence", value=True,
            help="Smooths the confidence scores so a 90% really means about 90% of the time "
                 "the model is right. Without this, neural networks tend to be overconfident.")
        show_ood = st.checkbox(
            "Warn on unfamiliar images", value=True,
            help="Shows a warning if the uploaded image doesn't look like the wood micrographs "
                 "the model was trained on (e.g. a non-wood photo, a poor-quality image, or a "
                 "species not in the 112 the model knows). "
                 "(Technical name: out-of-distribution / OOD detection.)")
        st.divider()
        st.header("ℹ️ Model")
        st.markdown("""
        | | |
        |---|---|
        | **Architecture** | ResNet50 (deep learning) |
        | **Dataset** | FSD Micro (2,240 micrographs) |
        | **Species** | 112 |
        | **Top-1 accuracy** | 97.32% |
        | **Top-3 accuracy** | 99.11% |
        | **Top-5 accuracy** | 99.70% |
        """)
        with st.expander("What does Top-K accuracy mean?"):
            st.markdown(
                "**Top-1** = how often the model's #1 guess is correct.  \n"
                "**Top-3** = how often the correct species is among its top 3 guesses.  \n"
                "**Top-5** = how often it's among the top 5.  \n\n"
                "Higher numbers mean the model is more reliable. For research-grade "
                "wood identification, look at the top-3 or top-5 list rather than just "
                "the single best guess."
            )
        st.divider()
        st.caption("TÜBİTAK 1002 · Streamlit + PyTorch")

    # Load model and reference data
    model, classes, device_used = load_model()
    species_db = load_species_db()
    iawa_templates, iawa_overrides, _, _ = load_iawa_templates()

    if model is None:
        if GOOGLE_DRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE":
            st.error(
                "### ⚠️ Google Drive File ID not configured!\n\n"
                "Open `app.py` and set `GOOGLE_DRIVE_FILE_ID` to your model's "
                "Google Drive share link ID.\n\n"
                "**Steps:**\n"
                "1. Go to Google Drive → `Wood_Project/FSD_Pretrained/models/`\n"
                "2. Right-click `fsd_BEST_resnet50_full.pth` → Share → Anyone with link\n"
                "3. Copy the link → extract the file ID\n"
                "4. Paste it in `app.py` line 30")
        else:
            st.error("### ⚠️ Model download failed. Check the file ID and sharing permissions.")
        return

    transform = get_transform()
    device_label = "🟢 GPU" if device_used.type == "cuda" else "🔵 CPU"
    st.sidebar.success(f"Model loaded — {device_label}")

    # Upload
    st.markdown("### 📤 Upload micrograph(s)")
    uploaded_files = st.file_uploader(
        "Drop one or more wood micrograph images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True)

    if not uploaded_files:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        for col, val, lbl in [(c1, "112", "Species the model knows"),
                              (c2, "97.3%", "Correct on first guess"),
                              (c3, "99.7%", "Correct in top 5 guesses")]:
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{val}'
                            f'</div><div class="stat-label">{lbl}</div></div>',
                            unsafe_allow_html=True)

        st.info("👆 **Upload one or more wood micrographs above to get started.**\n\n"
                "Best results come from **microscopic cross-section images** "
                "(~100× magnification, ideally ~1024×768 px PNG/JPG).")

        with st.expander("ℹ️ What does this app actually do?"):
            st.markdown(
                "This app looks at a microscopic photo of wood (a *micrograph*) and tries "
                "to identify the **tree species** it came from.\n\n"
                "**How it works (in plain language):**\n"
                "1. You upload an image of a wood cross-section taken under a microscope.\n"
                "2. A trained AI model — a deep neural network called *ResNet-50* — has "
                "learned what the cellular structure of 112 tree species looks like.\n"
                "3. It returns its best guesses, ranked by confidence.\n"
                "4. The app also shows **where in your image** the AI looked (heatmap) "
                "and the **wood-anatomy features** of the predicted species "
                "(vessels, rays, parenchyma, etc., from the IAWA standard).\n\n"
                "**Important — please read:** The model can only recognise the 112 species "
                "it was trained on. If you upload an image from a species outside that "
                "list, it will still output its best guess, but that guess will be wrong. "
                "The *Warn on unfamiliar images* checkbox in the sidebar tries to flag "
                "these cases, but it isn't perfect. Always treat the output as a "
                "**suggestion to verify**, not a final answer."
            )

        with st.expander("📖 Quick glossary"):
            st.markdown(
                "- **Micrograph** — a photo taken through a microscope.\n"
                "- **Cross-section** — a slice cut perpendicular to the grain, "
                "showing the rings and cells end-on.\n"
                "- **IAWA** — *International Association of Wood Anatomists*. "
                "The standard list of features (porosity, vessels, rays, parenchyma, "
                "resin canals, etc.) used worldwide to identify wood under a microscope.\n"
                "- **Hardwood vs softwood** — botanical groups, not a measure of hardness. "
                "Hardwoods (broadleaf trees) have *vessels*; softwoods (conifers) don't.\n"
                "- **Confidence** — how sure the model is, from 0% to 100%. Higher is more reliable.\n"
                "- **Heatmap (Grad-CAM)** — an overlay showing which parts of the image "
                "drove the prediction. Useful for sanity-checking the model.\n"
                "- **Out-of-distribution (OOD)** — fancy way of saying \"this image looks "
                "unlike anything I was trained on, so don't trust my answer.\""
            )
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
            if show_gradcam:
                results, ood, cam_overlay, cam = predict_with_gradcam(
                    model, image, classes, device_used, transform, top_k)
            else:
                results, ood = predict(model, image, classes, device_used,
                                       transform, top_k, use_calibration)
            ms = (time.time() - t0) * 1000

            # Tiered confidence warnings
            if show_ood and ood["is_ood"]:
                st.markdown(f"""
                <div class="ood-warning">
                    <div style="color:#F44336;font-size:1.1rem;font-weight:600;">
                        🚫 Out-of-distribution — image likely not in training set</div>
                    <div style="color:#ccc;margin-top:0.3rem;">{ood['reason']}</div>
                    <div style="color:#888;font-size:0.8rem;margin-top:0.3rem;">
                        Top-1 conf: {ood['max_confidence']*100:.1f}% ·
                        Top-1/Top-2 margin: {ood['margin']:.2f}× ·
                        Entropy: {ood['normalized_entropy']:.3f}</div>
                </div>""", unsafe_allow_html=True)
            elif show_ood and ood.get("is_uncertain"):
                st.markdown(f"""
                <div class="uncertain-warning">
                    <div style="color:#FFC107;font-size:1.1rem;font-weight:600;">
                        ⚠️ Low confidence — verify with a second method</div>
                    <div style="color:#ccc;margin-top:0.3rem;">{ood['reason']}</div>
                    <div style="color:#888;font-size:0.8rem;margin-top:0.3rem;">
                        Top-1 conf: {ood['max_confidence']*100:.1f}% ·
                        Top-1/Top-2 margin: {ood['margin']:.2f}× ·
                        Entropy: {ood['normalized_entropy']:.3f}</div>
                </div>""", unsafe_allow_html=True)

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
                f'⚡ {ms:.0f} ms · {top["full_label"]}</div>'
                f'</div>',
                unsafe_allow_html=True)

            st.markdown(f"#### Top {top_k} candidate species (most to least likely)")
            st.plotly_chart(make_confidence_chart(results))

        # Grad-CAM
        if show_gradcam:
            st.markdown("---")
            st.markdown("#### 🔥 Where the model looked")
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(image.resize((224, 224)), caption="Your image", width="stretch")
            with gc2:
                st.image(cam_overlay, caption="Heatmap overlay", width="stretch")
            st.caption(
                "Red/yellow areas are the parts of the image the model paid most attention to. "
                "If the heatmap is on cellular features (vessels, rays, growth rings), the prediction is more trustworthy. "
                "If it's on dust, edges, or background, treat the result with caution. "
                "(Technical name: Grad-CAM.)"
            )

        # Species info
        if show_species_info:
            st.markdown("---")
            render_species_info(results[0]["full_label"], species_db,
                                iawa_templates, iawa_overrides)

    # ─── BATCH MODE ──────────────────────────────────────
    else:
        st.markdown(f"### 🔄 Batch — {len(uploaded_files)} images")

        # 1. Run inference once for every image (cheap predict, no Grad-CAM)
        progress = st.progress(0, text="Classifying...")
        batch_results = []
        total_time = 0.0
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            t0 = time.time()
            results, ood = predict(model, image, classes, device_used,
                                   transform, top_k=top_k, use_calibration=use_calibration)
            total_time += time.time() - t0
            batch_results.append({"file": file, "image": image,
                                  "results": results, "ood": ood})
            progress.progress((i+1)/len(uploaded_files),
                              text=f"Classified {i+1}/{len(uploaded_files)}")
        progress.empty()

        # 2. Stats row (always visible above the tabs)
        avg_conf = float(np.mean([br["results"][0]["confidence"] for br in batch_results]))
        ood_count = sum(1 for br in batch_results if br["ood"]["is_ood"])
        uncertain_count = sum(1 for br in batch_results if br["ood"].get("is_uncertain"))
        unique_sp = len(set(br["results"][0]["species"] for br in batch_results))
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, val, lbl in [(s1, str(len(batch_results)), "Images"),
                               (s2, f"{avg_conf*100:.1f}%", "Avg. confidence"),
                               (s3, str(unique_sp), "Unique species"),
                               (s4, str(uncertain_count), "Low-confidence"),
                               (s5, str(ood_count), "OOD warnings")]:
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{val}'
                            f'</div><div class="stat-label">{lbl}</div></div>',
                            unsafe_allow_html=True)
        st.caption(f"⏱️ Classified {len(batch_results)} images in "
                   f"{total_time:.1f} s (avg {total_time/len(batch_results)*1000:.0f} ms/image)")

        # 3. Top-level tabs — every piece of info now lives in exactly ONE place
        tab_summary, tab_gallery, tab_inspect, tab_reference, tab_distrib = st.tabs([
            "📋 Summary table",
            "🖼️ Gallery",
            "🔍 Inspect one image",
            "📚 Species reference",
            "📊 Distribution",
        ])

        # ── Tab 1: summary table ────────────────────────────────
        with tab_summary:
            rows = []
            for br in batch_results:
                top = br["results"][0]
                rows.append({
                    "Image":      br["file"].name,
                    "Status":    ("🔴 OOD" if br["ood"]["is_ood"]
                                  else "🟡 Low conf" if br["ood"].get("is_uncertain")
                                  else "🟢 OK"),
                    "Species":    top["species"],
                    "Confidence": f"{top['confidence']*100:.1f}%",
                    "Family":     species_db.get(top["full_label"], {}).get("family", "-"),
                    "Group":      species_db.get(top["full_label"], {}).get("group", "-"),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Status legend: 🟢 OK = high confidence · 🟡 Low conf = below 50%, "
                       "verify · 🔴 OOD = image looks unfamiliar to the model.")

        # ── Tab 2: compact gallery (4 per row, no per-card expanders) ──
        with tab_gallery:
            st.caption("Quick visual overview. For full anatomy info, use **Species reference**; "
                       "for the heatmap and full per-image breakdown, use **Inspect one image**.")
            per_row = 4
            for row_start in range(0, len(batch_results), per_row):
                cols = st.columns(per_row)
                for col, br in zip(cols, batch_results[row_start:row_start + per_row]):
                    with col:
                        st.image(br["image"], caption=br["file"].name, width="stretch")
                        top = br["results"][0]
                        css = confidence_color(top["confidence"])
                        emoji = confidence_emoji(top["confidence"])
                        status = ("🔴" if br["ood"]["is_ood"]
                                  else "🟡" if br["ood"].get("is_uncertain") else "🟢")
                        st.markdown(
                            f'<div style="text-align:center;padding:0.3rem;">'
                            f'<div style="font-size:0.95rem;font-weight:600;">'
                            f'{status} 🌿 <i>{top["species"]}</i></div>'
                            f'<span class="{css}" style="font-size:1.1rem;">'
                            f'{emoji} {top["confidence"]*100:.1f}%</span></div>',
                            unsafe_allow_html=True)
                        # Mini top-3 bars
                        for r in br["results"][:3]:
                            pct = r["confidence"] * 100
                            bc = "#4CAF50" if pct >= 50 else "#FFC107" if pct >= 20 else "#F44336"
                            sp_short = r["species"][:24]
                            bar_w = min(pct, 100)
                            st.markdown(
                                f'<div style="margin:2px 0;font-size:0.72rem;">'
                                f'<span style="color:#bbb;">{sp_short}</span>'
                                f'<span style="color:#888;float:right;">{pct:.1f}%</span>'
                                f'<div class="mini-bar-bg">'
                                f'<div class="mini-bar-fill" style="background:{bc};'
                                f'width:{bar_w}%;"></div></div></div>',
                                unsafe_allow_html=True)

        # ── Tab 3: deep-dive on ONE image (with Grad-CAM + species info) ──
        with tab_inspect:
            file_names = [br["file"].name for br in batch_results]
            chosen = st.selectbox("Choose an image", file_names, key="batch_inspect_select")
            chosen_br = next(br for br in batch_results if br["file"].name == chosen)
            chosen_image = chosen_br["image"]

            di_col_img, _, di_col_res = st.columns([1, 0.05, 1.2])
            with di_col_img:
                st.image(chosen_image, caption=chosen, width="stretch")
                st.caption(f"📐 {chosen_image.width}×{chosen_image.height} · {chosen_image.mode}")

            with di_col_res:
                if show_gradcam:
                    d_results, d_ood, d_cam_overlay, _cam = predict_with_gradcam(
                        model, chosen_image, classes, device_used, transform, top_k)
                else:
                    d_results, d_ood = predict(model, chosen_image, classes, device_used,
                                               transform, top_k, use_calibration)

                if show_ood and d_ood["is_ood"]:
                    st.markdown(f"""
                    <div class="ood-warning">
                        <div style="color:#F44336;font-size:1.1rem;font-weight:600;">
                            🚫 Out-of-distribution — image likely not in training set</div>
                        <div style="color:#ccc;margin-top:0.3rem;">{d_ood['reason']}</div>
                        <div style="color:#888;font-size:0.8rem;margin-top:0.3rem;">
                            Top-1 conf: {d_ood['max_confidence']*100:.1f}% ·
                            Top-1/Top-2 margin: {d_ood['margin']:.2f}× ·
                            Entropy: {d_ood['normalized_entropy']:.3f}</div>
                    </div>""", unsafe_allow_html=True)
                elif show_ood and d_ood.get("is_uncertain"):
                    st.markdown(f"""
                    <div class="uncertain-warning">
                        <div style="color:#FFC107;font-size:1.1rem;font-weight:600;">
                            ⚠️ Low confidence — verify with a second method</div>
                        <div style="color:#ccc;margin-top:0.3rem;">{d_ood['reason']}</div>
                        <div style="color:#888;font-size:0.8rem;margin-top:0.3rem;">
                            Top-1 conf: {d_ood['max_confidence']*100:.1f}% ·
                            Top-1/Top-2 margin: {d_ood['margin']:.2f}× ·
                            Entropy: {d_ood['normalized_entropy']:.3f}</div>
                    </div>""", unsafe_allow_html=True)

                d_top = d_results[0]
                d_css = confidence_color(d_top["confidence"])
                d_emoji = confidence_emoji(d_top["confidence"])
                cal_tag = (' <span style="color:#888;font-size:0.75rem;">(calibrated)</span>'
                           if use_calibration else '')
                st.markdown(
                    f'<div class="prediction-card">'
                    f'<div style="color:#aaa;font-size:0.85rem;">PREDICTED SPECIES</div>'
                    f'<div class="species-name">🌿 <i>{d_top["species"]}</i></div>'
                    f'<div style="margin-top:0.8rem;">Confidence: {d_emoji} '
                    f'<span class="{d_css}" style="font-size:1.4rem;">'
                    f'{d_top["confidence"]*100:.1f}%</span>{cal_tag}</div>'
                    f'<div style="color:#666;font-size:0.8rem;margin-top:0.3rem;">'
                    f'{d_top["full_label"]}</div></div>',
                    unsafe_allow_html=True)

                st.markdown(f"**Top {top_k} candidates (most → least likely)**")
                st.plotly_chart(make_confidence_chart(d_results))

            if show_gradcam:
                st.markdown("##### 🔥 Where the model looked")
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.image(chosen_image.resize((224, 224)),
                             caption="Your image", width="stretch")
                with gc2:
                    st.image(d_cam_overlay, caption="Heatmap overlay", width="stretch")
                st.caption(
                    "Red/yellow areas are the parts of the image the model paid most attention to. "
                    "If the heatmap is on cellular features (vessels, rays, growth rings), the prediction is more trustworthy. "
                    "If it's on dust, edges, or background, treat the result with caution. "
                    "(Technical name: Grad-CAM.)"
                )

            if show_species_info:
                st.markdown("---")
                render_species_info(d_top["full_label"], species_db,
                                    iawa_templates, iawa_overrides)

        # ── Tab 4: species reference — DEDUPLICATED, one entry per unique species ──
        with tab_reference:
            if not show_species_info:
                st.info("Turn on **Show species info & wood anatomy (IAWA)** in the sidebar "
                        "to see the species reference here.")
            else:
                # Build unique species list with image counts
                unique_map = {}
                for br in batch_results:
                    lbl = br["results"][0]["full_label"]
                    unique_map.setdefault(lbl, []).append(br["file"].name)

                st.caption(f"{len(unique_map)} unique species predicted across "
                           f"{len(batch_results)} images. Each species' wood-anatomy "
                           "details are shown once below.")

                # Selectbox if many; otherwise just stack them
                ordered = sorted(unique_map.items(),
                                 key=lambda kv: -len(kv[1]))  # most frequent first
                if len(ordered) > 4:
                    options = [f"{lbl}  ({len(files)} image{'s' if len(files)!=1 else ''})"
                               for lbl, files in ordered]
                    pick = st.selectbox("Pick a predicted species", options,
                                        key="species_reference_pick")
                    pick_idx = options.index(pick)
                    lbl, files = ordered[pick_idx]
                    st.caption(f"**Image{'s' if len(files)!=1 else ''} predicted as this species:** "
                               + ", ".join(files))
                    render_species_info(lbl, species_db, iawa_templates, iawa_overrides)
                else:
                    for lbl, files in ordered:
                        info = species_db.get(lbl, {})
                        common = info.get("common", "")
                        species_only = lbl.split(" ", 1)[-1]
                        with st.expander(
                            f"🌿  {species_only}"
                            + (f" — {common}" if common else "")
                            + f"  · {len(files)} image{'s' if len(files)!=1 else ''}",
                            expanded=(len(ordered) == 1),
                        ):
                            st.caption(f"**Image{'s' if len(files)!=1 else ''}:** "
                                       + ", ".join(files))
                            render_species_info(lbl, species_db,
                                                iawa_templates, iawa_overrides)

        # ── Tab 5: distribution chart ────────────────────────────
        with tab_distrib:
            sp_counts = pd.Series([br["results"][0]["species"]
                                  for br in batch_results]).value_counts()
            fig = px.bar(x=sp_counts.index, y=sp_counts.values,
                         labels={"x": "Species", "y": "Number of images"},
                         color=sp_counts.values, color_continuous_scale="Greens")
            fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ddd"),
                              showlegend=False, xaxis=dict(tickangle=-30),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            if len(sp_counts) > 1:
                st.caption(f"Most predicted species: **{sp_counts.index[0]}** "
                           f"({sp_counts.iloc[0]} images). "
                           f"Total unique species: {len(sp_counts)}.")



if __name__ == "__main__":
    main()
