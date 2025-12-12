import os
import json
import pickle
import numpy as np
import torch
import streamlit as st
from typing import Dict, Tuple, List
from input_to_image_embedd import LatentNN
from decoder import Decoder

# ------------------------------- constants -----------------------------------
FEATURE_NAMES: List[str] = [
    'LAT',      # Latitude of the fault or subfault patch
    'LON',      # Longitude of the fault or subfault patch
    'DEP',      # Depth of the fault or subfault patch
    'STRK',     # Strike angle (orientation of the fault relative to North)
    'DIP',      # Dip angle (steepness of the fault plane)
    'RAKE',     # Rake angle (direction of slip)
    'LEN_f',    # Fault length (if known before the event)
    'WID',      # Fault width (if known before the event)
    'Htop',     # Depth to the top of the fault
    'HypX',     # Hypocenter location along the fault's length
    'HypZ',     # Hypocenter location along the fault's width
    'Nx',       # Number of subfaults along strike
    'Nz',       # Number of subfaults along dip
    'Dx',       # Length of each subfault patch
    'Dz',       # Width of each subfault patch
    'Mw'        # Moment Magnitude
]
TEXT_VEC_PATH = r"Dataset/text_vec.npy"
SCALER_X_PATH = r"scaler_x.pkl"
LATENT_WEIGHTS_PATH = r"models/latent_model.pth"
DECODER_WEIGHTS_PATH = r"models/decoder_model.pth"
HYPERPARAMS_PATH = r"models/best_hyperparams.json"
DZ_JSON_PATH = os.path.join("assets", "dz.json")
OUTPUT_DIM = 2704  # 16 x 13 x 13


# ------------------------------- helpers -------------------------------------
def _infer_input_dim(npy_path: str) -> int:
    """
    Infer input dimension from the .npy dict (authoritative training inputs).
    """
    data = np.load(npy_path, allow_pickle=True).item()
    try:
        first_key = next(iter(data))
    except StopIteration:
        raise ValueError(f"No entries found in {npy_path}")
    arr = np.asarray(data[first_key]).reshape(-1)
    return int(arr.shape[0])


@st.cache_resource(show_spinner=False)
def load_dataset_and_ranges(npy_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load the key->vector dict and compute per-feature min/max across the dataset.

    Returns:
        text_dict: mapping of event key -> 9-dim feature vector
        mins:      shape (9,) array of feature-wise minimums
        maxs:      shape (9,) array of feature-wise maximums
    """
    text_dict = np.load(npy_path, allow_pickle=True).item()

    # Stack to (N, D); vectors can be object arrays; convert safely to float.
    vectors: List[np.ndarray] = []
    for key, vec in text_dict.items():
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size != len(FEATURE_NAMES):
            # Skip malformed entries quietly
            continue
        vectors.append(arr)

    if not vectors:
        raise ValueError("No valid vectors found in Dataset/text_vec.npy")

    mat = np.vstack(vectors)  # (N, 9)
    mins = mat.min(axis=0)
    maxs = mat.max(axis=0)
    return text_dict, mins, maxs


@st.cache_resource(show_spinner=False)
def load_models_and_scaler() -> Tuple[LatentNN, Decoder, object, torch.device]:
    """
    Build LatentNN and Decoder, load weights, load StandardScaler.
    All objects are cached across reruns.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for the latent model
    with open(HYPERPARAMS_PATH, "r") as f:
        hp = json.load(f)
    dropout_prob = hp["dropout_prob"]
    hidden_dims = [hp[f"hidden_layer_{i}"] for i in range(1, 1 + 1)]  # matches training

    # Infer input dimension (should be 9 for this project, but we infer robustly)
    input_dim = _infer_input_dim(TEXT_VEC_PATH)

    latent = LatentNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=OUTPUT_DIM,
        dropout_prob=dropout_prob,
    )
    latent.load_state_dict(torch.load(LATENT_WEIGHTS_PATH, map_location=device))
    latent.to(device).eval()

    # Decoder: use the trained decoder weights saved during pipeline training
    decoder = Decoder(model_weights_path=DECODER_WEIGHTS_PATH, device=str(device))
    decoder.to(device).eval()

    # Load feature scaler (used during training on X)
    with open(SCALER_X_PATH, "rb") as f:
        scaler_x = pickle.load(f)

    return latent, decoder, scaler_x, device


@st.cache_resource(show_spinner=False)
def load_dz_json(path: str) -> Dict[str, float]:
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def run_inference(
    feature_vec: np.ndarray,
    *,
    latent: LatentNN,
    decoder: Decoder,
    scaler_x,
    device: torch.device,
) -> np.ndarray:
    """
    End-to-end forward pass from raw 9-dim features to a (50, 50) image array.
    Returns an array in [0, 1].
    """
    # Scale like training
    x_scaled = scaler_x.transform(feature_vec.reshape(1, -1))
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        latent_img = latent(x_tensor)                        # [1, 2704]
        pred = decoder(latent_img)                           # [1, 1, 50, 50]
        img = pred[0, 0].detach().cpu().numpy()

    # Clamp to [0, 1] just in case
    if img.max() > 1.0 or img.min() < 0.0:
        img = np.clip(img, 0.0, 1.0)
    return img

# --------------------------------- UI ----------------------------------------
st.set_page_config(page_title="Interactive Slip Map Generator", layout="wide")
st.title("Interactive Slip Map Generator")
st.caption(
    "Adjust the input parameters using the sliders below. The ranges are "
    "derived from your dataset (min/max per feature)."
)

with st.sidebar:
    st.subheader("Model & Data")
    st.text(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    st.text("Latent: models/latent_model.pth")
    st.text("Decoder: models/decoder_model.pth")
    st.divider()
    st.markdown("Optional presets (prefill sliders from a dataset event):")

try:
    text_dict, mins, maxs = load_dataset_and_ranges(TEXT_VEC_PATH)
    latent, decoder, scaler_x, device = load_models_and_scaler()
    dz_by_key = load_dz_json(DZ_JSON_PATH)
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()


# Optional preset selector in the sidebar
with st.sidebar:
    preset_key = st.selectbox(
        "Choose an event to prefill (optional)",
        options=["<none>"] + sorted(list(text_dict.keys())),
        index=0,
    )


def _default_values() -> List[float]:
    # Use the midpoint of each feature range as a sensible default
    return [float((lo + hi) / 2.0) for lo, hi in zip(mins, maxs)]


# Maintain slider state
if "feature_values" not in st.session_state:
    st.session_state.feature_values = _default_values()

if preset_key != "<none>":
    # Prefill from the dataset entry
    preset_vec = np.asarray(text_dict[preset_key], dtype=float).reshape(-1)
    if preset_vec.size == len(FEATURE_NAMES):
        st.session_state.feature_values = preset_vec.tolist()


# Build sliders in two columns for compact layout
cols = st.columns(2)
for i, name in enumerate(FEATURE_NAMES):
    col = cols[i % 2]
    lo = float(mins[i])
    hi = float(maxs[i])
    val = float(st.session_state.feature_values[i])
    # Guard against degenerate ranges
    if lo == hi:
        hi = lo + 1e-6
    st.session_state.feature_values[i] = col.slider(
        label=name,
        min_value=lo,
        max_value=hi,
        value=min(max(val, lo), hi),
        step=(hi - lo) / 200.0 if hi > lo else 1e-6,
        help=f"Range in dataset: [{lo:.4g}, {hi:.4g}]",
    )


with st.expander("Advanced options", expanded=False):
    apply_dz = st.checkbox("Apply Dz scaling (convert to slip units)", value=True)
    manual_dz = None
    if apply_dz:
        # If a preset is selected and we have Dz for it, use as default
        default_dz = float(dz_by_key.get(preset_key, 10.0)) if preset_key != "<none>" else 10.0
        # Estimate Dz range from available values if present
        if dz_by_key:
            dz_vals = np.array(list(dz_by_key.values()), dtype=float)
            dz_min = float(np.nanmin(dz_vals))
            dz_max = float(np.nanmax(dz_vals))
            manual_dz = st.slider("Dz", min_value=dz_min, max_value=dz_max, value=default_dz)
        else:
            manual_dz = st.number_input("Dz", value=default_dz)


# Auto-generate on every change (Streamlit reruns the script on interaction)
# Assemble the 9-dim vector in the exact training order
x = np.asarray(st.session_state.feature_values, dtype=float).reshape(-1)
if x.size != len(FEATURE_NAMES):
    st.error("Feature vector has incorrect size.")
    st.stop()

# Run the forward pass
with st.spinner("Generating slip map..."):
    img = run_inference(x, latent=latent, decoder=decoder, scaler_x=scaler_x, device=device)

# Optionally convert to slip using Dz
if apply_dz and manual_dz is not None:
    # Lazy import to avoid circular imports
    from assets.utils import pixels_to_slip
    slip_img = pixels_to_slip(img, manual_dz, image_name=preset_key if preset_key != "<none>" else None, plot=False)
    disp = slip_img
    cmap = "viridis"
    colorbar_label = "Slip"
else:
    disp = img
    cmap = "gray"
    colorbar_label = "Pixel Intensity (normalized)"

# Render with a consistent color scale
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(disp, origin="lower", cmap=cmap)
ax.axis("off")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(colorbar_label)

st.pyplot(fig, clear_figure=True)

plt.close(fig)


st.markdown(
    """
    How to run:

    1. Install dependencies: `pip install streamlit torch pillow numpy matplotlib`
    2. Run: `streamlit run interactive_slip_app.py`
    """
)