"""
01_feature_extraction.py
========================
Phase 3  —  Step 4.1: Feature Pool Construction

Extracts 4 feature families from every image using minicv as the sole
image-processing backend.  Produces:

    X_train_features.npy     (N_train  × 703)
    X_val_features.npy       (N_val    × 703)
    X_test_features.npy      (N_test   × 703)
    feature_index.json        naming / index scheme for all 703 features
    figures/feat_aug_panel.png  before-vs-after augmentation feature panel

Feature Index (703 total dimensions)
─────────────────────────────────────────────────────────────────
  [  0 –  95]  Family 1a │ color_histogram   — 32 bins × 3 channels (R,G,B)
  [ 96 – 119]  Family 1b │ pixel_statistics  — 8 stats × 3 channels
  [120 – 443]  Family 2a │ hog_descriptor    — cell=32 px, bins=9, block=2
  [444 – 446]  Family 2b │ sobel_stats       — magnitude mean, std, max
  [447 – 510]  Family 3  │ lbp_descriptor    — 64 bins (uniform LBP codes)
  [511 – 702]  Family 4  │ spatial_color     — 4 quadrants × 16 bins × 3 ch
─────────────────────────────────────────────────────────────────
"""

import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── minicv imports (adjust path if needed) ────────────────────────────────
sys.path.insert(0, os.path.abspath("."))
import minicv.io         as io
import minicv.features   as features
import minicv.filtering  as filt

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = r"..\data"           # folder containing X_train_augmented.npy etc.
OUT_DIR  = "features"
FIG_DIR  = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Feature hyper-parameters (edit here to change globally) ───────────────
HIST_BINS      = 32    # colour histogram bins per channel
STAT_KEYS      = [     # pixel_statistics keys extracted IN THIS ORDER (fixed)
    "mean_0","std_0","min_0","max_0","skewness_0","kurtosis_0","energy_0","entropy_0",
    "mean_1","std_1","min_1","max_1","skewness_1","kurtosis_1","energy_1","entropy_1",
    "mean_2","std_2","min_2","max_2","skewness_2","kurtosis_2","energy_2","entropy_2",
]
HOG_CELL       = 32    # pixels per HOG cell  (128 // 32 = 4 cells/side)
HOG_BINS       = 9     # orientation bins
HOG_BLOCK      = 2     # cells per block side  (3×3 blocks → 324 features)
LBP_BINS       = 64    # LBP histogram bins
SPATIAL_BINS   = 16    # colour histogram bins per channel per quadrant
SPATIAL_GRID   = 2     # grid size (2×2 = 4 quadrants)

# Derived feature counts
N_HIST   = HIST_BINS * 3                          # 96
N_STATS  = len(STAT_KEYS)                         # 24
# HOG length computed dynamically from first image; stored here after first call
N_HOG    = None
N_SOBEL  = 3                                      # mean, std, max of |G|
N_LBP    = LBP_BINS                               # 64
N_SPATIAL= SPATIAL_GRID**2 * SPATIAL_BINS * 3    # 192
TOTAL_FEATURES = N_HIST + N_STATS + -1 + N_SOBEL + N_LBP + N_SPATIAL
# HOG length filled in after first extraction


# ══════════════════════════════════════════════════════════════════════════
#  Core per-image extractor
# ══════════════════════════════════════════════════════════════════════════

def extract_features(img_float):
    """
    Extract all feature families from a single image.

    Parameters
    ----------
    img_float : np.ndarray
        Preprocessed image, shape (H, W, 3), dtype float64, range [0.0, 1.0].

    Returns
    -------
    np.ndarray
        1-D feature vector of shape (TOTAL_FEATURES,).
    """
    # ── Prepare colour variants ──────────────────────────────────────────
    # minicv functions that need uint8 (lbp, hog need gray uint8)
    img_u8   = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    gray_u8  = io.rgb_to_gray(img_u8)               # (H, W) uint8
    gray_f   = gray_u8.astype(np.float64) / 255.0   # (H, W) float64

    H, W = img_float.shape[:2]

    # ── Family 1a: Colour Histogram ──────────────────────────────────────
    # Uses float image; returns 32×3 = 96 values, L1-normalised per channel
    f1a = features.color_histogram(img_float, bins=HIST_BINS, normalize=True)

    # ── Family 1b: Pixel Statistics ─────────────────────────────────────
    # Returns dict; extract in fixed order defined by STAT_KEYS
    stats_dict = features.pixel_statistics(img_float)
    f1b = np.array([stats_dict.get(k, 0.0) for k in STAT_KEYS], dtype=np.float64)

    # ── Family 2a: HOG Descriptor ────────────────────────────────────────
    # Operates on grayscale float; L2-Hys normalised blocks
    f2a = features.hog_descriptor(
        gray_f, cell_size=HOG_CELL, n_bins=HOG_BINS, block_size=HOG_BLOCK
    )

    # ── Family 2b: Sobel Magnitude Statistics ────────────────────────────
    # mean, std, max of the gradient magnitude map
    grads  = filt.sobel_gradients(gray_f)
    mag    = grads["magnitude"]
    f2b    = np.array([mag.mean(), mag.std(), mag.max()], dtype=np.float64)

    # ── Family 3: LBP Texture ────────────────────────────────────────────
    # Needs uint8 grayscale; returns 64-bin normalised histogram
    f3 = features.lbp_descriptor(gray_u8, n_bins=LBP_BINS, normalize=True)

    # ── Family 4: Spatial Colour Histogram (2×2 grid) ────────────────────
    # Divide image into quadrants; compute per-region colour histogram
    # Captures WHERE colours appear, not just WHAT colours exist globally
    qH, qW = H // SPATIAL_GRID, W // SPATIAL_GRID
    spatial_parts = []
    for gr in range(SPATIAL_GRID):
        for gc in range(SPATIAL_GRID):
            quadrant = img_float[gr*qH:(gr+1)*qH, gc*qW:(gc+1)*qW]
            h_q = features.color_histogram(quadrant, bins=SPATIAL_BINS, normalize=True)
            spatial_parts.append(h_q)
    f4 = np.concatenate(spatial_parts)   # 4 × 16 × 3 = 192

    return np.concatenate([f1a, f1b, f2a, f2b, f3, f4])


# ══════════════════════════════════════════════════════════════════════════
#  Feature index documentation
# ══════════════════════════════════════════════════════════════════════════

def build_index(hog_len):
    """Build and save the feature naming/indexing scheme."""
    idx = {}
    cursor = 0

    # Family 1a — colour histogram
    channels = ["R", "G", "B"]
    for c, ch in enumerate(channels):
        for b in range(HIST_BINS):
            idx[cursor] = f"fam1a_colorhist_{ch}_bin{b:02d}"
            cursor += 1

    # Family 1b — pixel statistics
    for k in STAT_KEYS:
        idx[cursor] = f"fam1b_pixelstat_{k}"
        cursor += 1

    # Family 2a — HOG
    for i in range(hog_len):
        idx[cursor] = f"fam2a_hog_{i:04d}"
        cursor += 1

    # Family 2b — Sobel stats
    for name in ["mag_mean", "mag_std", "mag_max"]:
        idx[cursor] = f"fam2b_sobel_{name}"
        cursor += 1

    # Family 3 — LBP
    for b in range(LBP_BINS):
        idx[cursor] = f"fam3_lbp_bin{b:02d}"
        cursor += 1

    # Family 4 — spatial colour
    q = 0
    for gr in range(SPATIAL_GRID):
        for gc in range(SPATIAL_GRID):
            for ch in channels:
                for b in range(SPATIAL_BINS):
                    idx[cursor] = f"fam4_spatial_q{q}_{ch}_bin{b:02d}"
                    cursor += 1
            q += 1

    print(f"\nFeature index built: {cursor} total features")

    # Print summary table
    ranges = {
        "Family 1a — color_histogram   ": (0, N_HIST - 1),
        "Family 1b — pixel_statistics  ": (N_HIST, N_HIST + N_STATS - 1),
        "Family 2a — hog_descriptor    ": (N_HIST + N_STATS, N_HIST + N_STATS + hog_len - 1),
        "Family 2b — sobel_stats       ": (N_HIST + N_STATS + hog_len,
                                           N_HIST + N_STATS + hog_len + N_SOBEL - 1),
        "Family 3  — lbp_descriptor    ": (N_HIST + N_STATS + hog_len + N_SOBEL,
                                           N_HIST + N_STATS + hog_len + N_SOBEL + N_LBP - 1),
        "Family 4  — spatial_color     ": (N_HIST + N_STATS + hog_len + N_SOBEL + N_LBP,
                                           cursor - 1),
    }
    print("\n  Feature Index Summary")
    print("  " + "─" * 60)
    for name, (lo, hi) in ranges.items():
        print(f"  [{lo:4d} – {hi:4d}]  {name}  ({hi-lo+1} dims)")
    print("  " + "─" * 60)
    print(f"  {'Total':45s}  {cursor} dims\n")

    # Save as JSON (keys as strings for JSON compatibility)
    with open(os.path.join(OUT_DIR, "feature_index.json"), "w") as f:
        json.dump({"total": cursor, "ranges": {k: list(v) for k, v in ranges.items()},
                   "features": {str(k): v for k, v in idx.items()}}, f, indent=2)
    print(f"  Saved feature_index.json")
    return idx, ranges


# ══════════════════════════════════════════════════════════════════════════
#  Batch extraction
# ══════════════════════════════════════════════════════════════════════════

def extract_split(X, split_name):
    """Extract features for every image in a split. Returns (N, D) array."""
    print(f"\nExtracting features — {split_name}  ({len(X)} images)...")
    feats = []
    for i in tqdm(range(len(X)), desc=split_name, unit="img"):
        fv = extract_features(X[i])
        feats.append(fv)
    return np.array(feats, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
#  Before / After Augmentation Panel  (4.1 requirement)
# ══════════════════════════════════════════════════════════════════════════

def make_aug_panel(X_orig, X_aug, Y_orig, Y_aug, class_names, n_examples=4):
    """
    Show original vs augmented images side-by-side alongside their
    colour histogram and LBP histogram to visualise how features shift.

    X_orig : subset of un-augmented images, shape (N, H, W, 3) float64
    X_aug  : augmented counterparts, same layout
    """
    fig = plt.figure(figsize=(20, n_examples * 4.2))
    fig.patch.set_facecolor("#0f1117")
    col_labels = [
        "Original Image", "Augmented Image",
        "Colour Histogram (orig)", "Colour Histogram (aug)",
        "LBP Histogram (orig)",   "LBP Histogram (aug)",
    ]
    n_cols = 6

    for row in range(n_examples):
        img_o = X_orig[row]
        img_a = X_aug[row]

        # Images
        for col, img in [(0, img_o), (1, img_a)]:
            ax = fig.add_subplot(n_examples, n_cols, row * n_cols + col + 1)
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(col_labels[col] + f"\nclass: {class_names[Y_orig[row]]}",
                         color="white", fontsize=7)
            ax.axis("off")

        # Colour histograms
        for col, img in [(2, img_o), (3, img_a)]:
            ax = fig.add_subplot(n_examples, n_cols, row * n_cols + col + 1)
            h = features.color_histogram(img, bins=32, normalize=True)
            x = np.arange(32)
            for c, (colour, label) in enumerate(zip(["#e74c3c","#2ecc71","#3498db"], "RGB")):
                ax.bar(x + c * 0.28, h[c*32:(c+1)*32], width=0.28,
                       color=colour, alpha=0.75, label=label)
            ax.set_title(col_labels[col], color="white", fontsize=7)
            ax.set_facecolor("#1a1a2e"); ax.tick_params(colors="white", labelsize=5)
            for sp in ax.spines.values(): sp.set_edgecolor("#444")
            if row == 0: ax.legend(fontsize=5, facecolor="#1a1a2e", labelcolor="white")

        # LBP histograms
        img_o_u8 = np.clip(img_o * 255, 0, 255).astype(np.uint8)
        img_a_u8 = np.clip(img_a * 255, 0, 255).astype(np.uint8)
        g_o = io.rgb_to_gray(img_o_u8)
        g_a = io.rgb_to_gray(img_a_u8)
        for col, g in [(4, g_o), (5, g_a)]:
            ax = fig.add_subplot(n_examples, n_cols, row * n_cols + col + 1)
            lbp = features.lbp_descriptor(g, n_bins=64, normalize=True)
            ax.bar(range(64), lbp, width=1, color="#9b59b6", alpha=0.8)
            ax.set_title(col_labels[col], color="white", fontsize=7)
            ax.set_facecolor("#1a1a2e"); ax.tick_params(colors="white", labelsize=5)
            for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.suptitle("Feature Extraction — Before vs After Augmentation",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "feat_aug_panel.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    t0 = time.time()

    # ── Load preprocessed arrays ─────────────────────────────────────────
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_augmented.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train_augmented.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    # Also load the un-augmented train set for the aug panel
    X_train_orig = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train_orig = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

    X_train = X_train.astype(np.float64) / 255.0
    X_val   = X_val.astype(np.float64) / 255.0
    X_test  = X_test.astype(np.float64) / 255.0
    X_train_orig = X_train_orig.astype(np.float64) / 255.0


    print(f"  Train (aug): {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Class names — adjust to match your dataset's label encoding
    # (These should match whatever integer labels you stored in Y arrays)
    CLASS_NAMES = [
        "butterfly", "cat", "chicken", "cow",
        "dog", "elephant", "horse", "sheep", "spider", "squirrel"
    ]  # ← edit to match your 10 classes in label order

    # ── Dry run on one image to get HOG length ────────────────────────────
    print("\nDry run to determine HOG vector length...")
    sample_fv = extract_features(X_train[0])
    N_HOG_actual = len(sample_fv) - (N_HIST + N_STATS + N_SOBEL + N_LBP + N_SPATIAL)
    total_dims   = len(sample_fv)
    print(f"  HOG length:    {N_HOG_actual}")
    print(f"  Total dims:    {total_dims}")

    # ── Build and save feature index ─────────────────────────────────────
    feature_idx, feature_ranges = build_index(N_HOG_actual)

    # ── Extract features for all splits ──────────────────────────────────
    F_train = extract_split(X_train, "train (augmented)")
    F_val   = extract_split(X_val,   "val")
    F_test  = extract_split(X_test,  "test")

    np.save(os.path.join(OUT_DIR, "X_train_features.npy"), F_train)
    np.save(os.path.join(OUT_DIR, "X_val_features.npy"),   F_val)
    np.save(os.path.join(OUT_DIR, "X_test_features.npy"),  F_test)
    np.save(os.path.join(OUT_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(OUT_DIR, "Y_val.npy"),   Y_val)
    np.save(os.path.join(OUT_DIR, "Y_test.npy"),  Y_test)

    print(f"\n  Feature matrices saved:")
    print(f"    train: {F_train.shape}")
    print(f"    val:   {F_val.shape}")
    print(f"    test:  {F_test.shape}")

    # ── Check for NaNs / Infs ─────────────────────────────────────────────
    for name, arr in [("train", F_train), ("val", F_val), ("test", F_test)]:
        bad = np.isnan(arr).sum() + np.isinf(arr).sum()
        if bad > 0:
            print(f"  WARNING: {bad} NaN/Inf values in {name} — replacing with 0")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Before / After Augmentation Panel ────────────────────────────────
    print("\nGenerating augmentation feature panel...")
    
    # 1. Take the first N images from the pure original dataset
    N_PANEL = 4
    X_panel_orig = X_train_orig[:N_PANEL]
    Y_panel_orig = Y_train_orig[:N_PANEL]
    
    # 2. Generate guaranteed matching augmented versions on the fly
    # (Assuming you imported rotate or gaussian_filter from minicv)
    X_panel_aug = np.zeros_like(X_panel_orig)
    for i in range(N_PANEL):
        # Apply a simple rotation to visually prove the feature shift
        from minicv.transforms import rotate
        X_panel_aug[i] = rotate(X_panel_orig[i], angle=15)

    make_aug_panel(
        X_orig     = X_panel_orig,
        X_aug      = X_panel_aug,
        Y_orig     = Y_panel_orig,
        Y_aug      = Y_panel_orig, # Labels stay the same
        class_names= CLASS_NAMES,
        n_examples = N_PANEL,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  ({elapsed:.0f} s)")
    print("=" * 55)
    print(f"  Total features per image:  {total_dims}")
    print(f"  Train feature matrix:      {F_train.shape}")
    print("=" * 55)
