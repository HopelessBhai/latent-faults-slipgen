import os
import json
import pickle
import numpy as np
import torch
import streamlit as st
from typing import Dict, Tuple, List, Optional
from scipy.stats import truncnorm
from input_to_image_embedd import LatentNN
from decoder import Decoder

# ------------------------------- constants -----------------------------------
# NOTE: The last feature is actually Mo (seismic moment), not Mw (moment magnitude)
# The model was trained on Mw values (typically 1-10 scale)

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
    'Mw',       

]

TEXT_VEC_PATH = r"./Dataset/text_vec.npy"
SCALER_X_PATH = r"./scaler_x.pkl"
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
def load_dataset_and_ranges(npy_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the key->vector dict and compute per-feature min/max/mean/std across the dataset.
    This matches the training pipeline which uses StandardScaler.

    Returns:
        text_dict: mapping of event key -> 9-dim feature vector
        mins:      shape (9,) array of feature-wise minimums
        maxs:      shape (9,) array of feature-wise maximums
        means:     shape (9,) array of feature-wise means (for StandardScaler)
        stds:      shape (9,) array of feature-wise stds (for StandardScaler)
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
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)
    return text_dict, mins, maxs, means, stds


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


def compute_seismic_moment(mw: float) -> float:
    """
    Convert moment magnitude Mw to seismic moment mo (in N⋅m).
    
    Formula: Mw = (2/3) * log10(mo) - 10.7
    Rearranged: mo = 10^((Mw + 10.7) * 3/2)
    """
    return 10 ** ((mw + 10.7) * 3 / 2)


def compute_rupture_dimensions(mo: float) -> Tuple[float, float]:
    """
    Compute effective rupture dimensions from seismic moment.
    
    Formulas:
    log10(Leff) = -3.52 + 0.27 * log10(mo)
    log10(Weff) = -4.34 + 0.29 * log10(mo)
    
    Returns:
        Leff: Effective rupture length (km)
        Weff: Effective rupture width (km)
    """
    log10_mo = np.log10(mo)
    log10_leff = -3.52 + 0.27 * log10_mo
    log10_weff = -4.34 + 0.29 * log10_mo
    leff = 10 ** log10_leff
    weff = 10 ** log10_weff
    return leff, weff


def sample_truncated_normal(mean: float, std: float, lower: float, upper: float, 
                            random_state: Optional[np.random.Generator] = None) -> float:
    """
    Sample from a truncated normal distribution.
    
    Args:
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        lower: Lower bound
        upper: Upper bound
        random_state: Optional random number generator
        
    Returns:
        Sampled value within [lower, upper]
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Convert bounds to standard normal scale
    a = (lower - mean) / std
    b = (upper - mean) / std
    
    # Sample from truncated normal and convert back
    sample = truncnorm.rvs(a, b, loc=mean, scale=std, random_state=random_state)
    return float(sample)


def compute_parameters(mw: float, strk: float, dip: float, rake: float, 
                      lat: float, lon: float, dep: float, 
                      nx: float, nz: float, dx: float, dz: float,
                      random_seed: Optional[int] = None) -> np.ndarray:
    """
    Compute/assemble all 16 parameters.
    
    Args:
        mw: Moment magnitude
        strk, dip, rake: Angles
        lat, lon, dep: Location
        nx, nz, dx, dz: Subfault geometry
        random_seed: Optional random seed for sampling
        
    Returns:
        Array of 16 parameters matching FEATURE_NAMES order.
    """
    # Set random seed if provided
    rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
    
    # Seismic moment for dimensions (M0 in N.m)
    mo = compute_seismic_moment(mw)
    
    # Step 2: Compute effective rupture dimensions
    leff, weff = compute_rupture_dimensions(mo)
    len_f = leff
    wid = weff
    
    # Step 3: Sample hypocenter location (normalized)
    hypx_norm = sample_truncated_normal(mean=0.30, std=0.13, lower=0.0, upper=0.5, random_state=rng)
    hypz_norm = sample_truncated_normal(mean=0.52, std=0.23, lower=0.0, upper=1.0, random_state=rng)
    
    # Step 4: Convert to physical coordinates
    hypx = hypx_norm * len_f
    hypz = hypz_norm * wid
    
    # Step 5: Sample top-of-rupture depth
    htop = rng.uniform(5.0, 15.0)
    
    # Assemble in the exact FEATURE_NAMES order:
    # ['LAT', 'LON', 'DEP', 'STRK', 'DIP', 'RAKE', 'LEN_f', 'WID', 'Htop', 'HypX', 'HypZ', 'Nx', 'Nz', 'Dx', 'Dz', 'Mw']
    # If the model expects Mo instead of Mw at the last position, we would use 'mo' here.
    params = np.array([
        lat, lon, dep, strk, dip, rake, 
        len_f, wid, htop, hypx, hypz, 
        nx, nz, dx, dz, mw
    ], dtype=float)
    
    return params


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
    "Adjust Mw, STRK, DIP, and RAKE. Other parameters are computed automatically "
    "using scaling laws and statistical sampling."
)

with st.sidebar:
    st.subheader("Model & Data")
    st.text(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    st.text("Latent: models/latent_model.pth")
    st.text("Decoder: models/decoder_model.pth")
    st.divider()

try:
    text_dict, mins, maxs, means, stds = load_dataset_and_ranges(TEXT_VEC_PATH)
    latent, decoder, scaler_x, device = load_models_and_scaler()
    dz_by_key = load_dz_json(DZ_JSON_PATH)
    
    # Verify that the scaler matches the dataset statistics
    # The scaler should have been fit on the same data
    scaler_means = scaler_x.mean_
    scaler_stds = np.sqrt(scaler_x.var_)
    
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# Get ranges for the adjustable parameters
# Feature order: [LEN_f, WID, RAKE, HypX, HypZ, Htop, DIP, STRK, Mo]
# NOTE: The last feature is Mo (seismic moment), not Mw (magnitude)
# We'll convert Mw to Mo before using it
mo_idx = 8  # This is actually Mo in the dataset
strk_idx = 7
dip_idx = 6
rake_idx = 2

# Mw slider range: 1 to 10 (reasonable range for moment magnitude)
mw_min = 1.0
mw_max = 10.0

# Initialize session state for all adjustable parameters
if "mw" not in st.session_state:
    st.session_state.mw = 5.5
if "strk" not in st.session_state:
    st.session_state.strk = float((mins[FEATURE_NAMES.index('STRK')] + maxs[FEATURE_NAMES.index('STRK')]) / 2.0)
if "dip" not in st.session_state:
    st.session_state.dip = float((mins[FEATURE_NAMES.index('DIP')] + maxs[FEATURE_NAMES.index('DIP')]) / 2.0)
if "rake" not in st.session_state:
    st.session_state.rake = float((mins[FEATURE_NAMES.index('RAKE')] + maxs[FEATURE_NAMES.index('RAKE')]) / 2.0)

# New parameters
for p in ['LAT', 'LON', 'DEP', 'Nx', 'Nz', 'Dx', 'Dz']:
    if p not in st.session_state:
        idx = FEATURE_NAMES.index(p)
        st.session_state[p.lower()] = float((mins[idx] + maxs[idx]) / 2.0)

if "random_seed" not in st.session_state:
    st.session_state.random_seed = None

# Main parameter controls
st.subheader("Primary Parameters")
cols = st.columns(2)

# Moment magnitude
mw = cols[0].slider("Mw (Moment Magnitude)", 1.0, 10.0, st.session_state.mw, 0.1)
st.session_state.mw = mw

# Strike
idx_strk = FEATURE_NAMES.index('STRK')
strk = cols[1].slider("STRK (Strike, degrees)", float(mins[idx_strk]), float(maxs[idx_strk]), st.session_state.strk, 0.1)
st.session_state.strk = strk

# Dip
idx_dip = FEATURE_NAMES.index('DIP')
dip = cols[0].slider("DIP (Dip, degrees)", float(mins[idx_dip]), float(maxs[idx_dip]), st.session_state.dip, 0.1)
st.session_state.dip = dip

# Rake
idx_rake = FEATURE_NAMES.index('RAKE')
rake = cols[1].slider("RAKE (Rake, degrees)", float(mins[idx_rake]), float(maxs[idx_rake]), st.session_state.rake, 0.1)
st.session_state.rake = rake

st.subheader("Additional Parameters")
cols2 = st.columns(3)
lat = cols2[0].slider("LAT", float(mins[0]), float(maxs[0]), st.session_state.lat, 0.01)
lon = cols2[1].slider("LON", float(mins[1]), float(maxs[1]), st.session_state.lon, 0.01)
dep = cols2[2].slider("DEP", float(mins[2]), float(maxs[2]), st.session_state.dep, 0.1)

cols3 = st.columns(4)
nx = cols3[0].slider("Nx", float(mins[11]), float(maxs[11]), st.session_state.nx, 1.0)
nz = cols3[1].slider("Nz", float(mins[12]), float(maxs[12]), st.session_state.nz, 1.0)
dx = cols3[2].slider("Dx", float(mins[13]), float(maxs[13]), st.session_state.dx, 0.1)

st.session_state.lat, st.session_state.lon, st.session_state.dep = lat, lon, dep
st.session_state.nx, st.session_state.nz, st.session_state.dx = nx, nz, dx

# Random seed
st.divider()
use_seed = st.checkbox("Use random seed for sampling", value=False)
random_seed = None
if use_seed:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2**31 - 1,
        value=st.session_state.random_seed if st.session_state.random_seed is not None else 42,
    )
    st.session_state.random_seed = random_seed

with st.expander("Advanced options", expanded=False):
    apply_dz = st.checkbox("Apply Dz scaling (convert to slip units)", value=True)
    manual_dz = None
    if apply_dz:
        # Estimate Dz range from available values if present
        default_dz = 1.0
        if dz_by_key:
            dz_vals = np.array(list(dz_by_key.values()), dtype=float)
            dz_min = float(np.nanmin(dz_vals))
            dz_max = float(np.nanmax(dz_vals))
            manual_dz = st.slider("Dz", min_value=dz_min, max_value=dz_max, value=default_dz)
        else:
            manual_dz = st.number_input("Dz", value=default_dz)
    
    # If not applying Dz from slider/input, use default centered dataset value
    idx_dz = FEATURE_NAMES.index('Dz')
    actual_dz = manual_dz if manual_dz is not None else float((mins[idx_dz] + maxs[idx_dz]) / 2.0)

# Compute all parameters
computed_params = compute_parameters(mw, strk, dip, rake, lat, lon, dep, nx, nz, dx, actual_dz, random_seed)

# Display computed parameters
st.divider()
st.subheader("Computed Parameters")
with st.expander("View computed values", expanded=True):
    param_display = {
        "LEN_f (km)": f"{computed_params[6]:.3f}",
        "WID (km)": f"{computed_params[7]:.3f}",
        "HypX (km)": f"{computed_params[9]:.3f}",
        "HypZ (km)": f"{computed_params[10]:.3f}",
        "Htop (km)": f"{computed_params[8]:.3f}",
    }
    for key, value in param_display.items():
        st.text(f"{key}: {value}")
    
    # Show seismic moment
    mo_val = compute_seismic_moment(mw)
    st.text(f"Mo (seismic moment, N⋅m): {mo_val:.2e}")
    st.text(f"Note: Model uses Mo (seismic moment), not Mw (magnitude)")
    
    # Debug: Show parameter ranges vs computed values
    with st.expander("Debug: Parameter ranges and normalization", expanded=False):
        st.text("Raw parameter values (before StandardScaler):")
        
        for i, name in enumerate(FEATURE_NAMES):
            computed_val = computed_params[i]
            dataset_min = mins[i]
            dataset_max = maxs[i]
            dataset_mean = means[i]
            dataset_std = stds[i]
            in_range = "✓" if dataset_min <= computed_val <= dataset_max else "✗"
            st.text(f"{name}: {computed_val:.3f} (range: [{dataset_min:.3f}, {dataset_max:.3f}], "
                   f"mean: {dataset_mean:.3f}, std: {dataset_std:.3f}) {in_range}")
        
        # Show normalized values (what the model actually sees)
        normalized_params = scaler_x.transform(computed_params.reshape(1, -1))[0]
        st.text("\nNormalized parameter values (after StandardScaler, what model sees):")
        for i, name in enumerate(FEATURE_NAMES):
            st.text(f"{name}: {normalized_params[i]:.3f}")
        
        # Show scaler statistics
        st.text(f"\nScaler statistics (from saved scaler_x.pkl):")
        st.text(f"Means: {scaler_means}")
        st.text(f"Stds: {scaler_stds}")

# Run the forward pass with computed parameters
with st.spinner("Generating slip map..."):
    img = run_inference(computed_params, latent=latent, decoder=decoder, scaler_x=scaler_x, device=device)

# Optionally convert to slip using Dz
if apply_dz and manual_dz is not None:
    # Lazy import to avoid circular imports
    from assets.utils import pixels_to_slip
    slip_img = pixels_to_slip(img, manual_dz, image_name=None, plot=False)
    disp = slip_img  # Convert meters to centimeters for display
    cmap = "viridis"
    colorbar_label = "Slip (m)"  # Display in centimeters
    use_physical_coords = True
else:
    disp = img
    cmap = "gray"
    colorbar_label = "Pixel Intensity (normalized)"
    use_physical_coords = False

# Render with proper axis labels and physical coordinates
import matplotlib.pyplot as plt

# Get physical dimensions from computed parameters
len_f_km = computed_params[6]  # LEN_f in km
wid_km = computed_params[7]    # WID in km

# Create figure with appropriate size
fig, ax = plt.subplots(figsize=(10, 6))

if use_physical_coords:
    # Create coordinate arrays for physical dimensions
    # Image is 50x50 pixels, map to physical dimensions
    along_strike = np.linspace(0, len_f_km, disp.shape[1])
    down_dip = np.linspace(0, wid_km, disp.shape[0])
    
    # Create meshgrid for contour lines
    X, Y = np.meshgrid(along_strike, down_dip)
    
    # Create the heatmap
    im = ax.imshow(disp, origin="lower", cmap=cmap, 
                   extent=[0, len_f_km, 0, wid_km],
                   aspect='auto', interpolation='bilinear')
    
    # Add black contour lines
    # Determine contour levels based on data range
    vmin, vmax = disp.min(), disp.max()
    n_levels = 8  # Number of contour levels
    contour_levels = np.linspace(vmin, vmax, n_levels)
    
    contours = ax.contour(X, Y, disp, levels=contour_levels, 
                          colors='black', linewidths=2, alpha=0.6)
    
    # Optionally add contour labels
    
    ax.clabel(contours, inline=True, fontsize=15, fmt='%1.0f')
    
    # Set axis labels
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel("Along-strike direction (km)", fontsize=22)
    ax.set_ylabel("Down-dip direction (km)", fontsize=22)
    
    # Set axis ticks
    ax.set_xticks(np.linspace(0, len_f_km, 5))
    ax.set_yticks(np.linspace(0, wid_km, 5))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

else:
    # For normalized pixel values, just show the image
    im = ax.imshow(disp, origin="lower", cmap=cmap, aspect='auto')
    ax.axis("off")

# Add colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(colorbar_label, fontsize=22)
cbar.ax.tick_params(labelsize=22)


# # Set title
# if use_physical_coords:
#     ax.set_title("Slip Distribution", fontsize=14, fontweight='bold')
# else:
#     ax.set_title("Predicted Slip Map (Normalized)", fontsize=14, fontweight='bold')

# Tight layout for better appearance
plt.tight_layout()

st.pyplot(fig, clear_figure=True)

plt.close(fig)


st.markdown(
    """
    How to run:

    1. Install dependencies: `pip install streamlit torch pillow numpy matplotlib`
    2. Run: `streamlit run interactive_slip_app.py`
    """
)


