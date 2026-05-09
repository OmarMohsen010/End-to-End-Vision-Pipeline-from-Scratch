"""
02_mrmr_selection.py
====================
Phase 3  —  Step 4.2: MRMR Feature Selection

Loads the full feature matrices produced by 01_feature_extraction.py,
applies MRMR (Minimum Redundancy Maximum Relevance) to select the top K
features from the training set, applies the same selection to val and test,
then saves:

    features/X_train_selected.npy
    features/X_val_selected.npy
    features/X_test_selected.npy
    features/mrmr_selected_indices.npy   — integer indices into the 703-dim vector
    features/mrmr_selected_names.json    — human-readable feature names
    figures/mrmr_panel.png               — selection visualisation
    figures/mrmr_family_pie.png          — which families survived selection
    figures/feat_selected_aug_panel.png  — before/after aug panel on SELECTED features
"""

import sys, os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mrmr import mrmr_classif

sys.path.insert(0, os.path.abspath("."))
import minicv.features as features
import minicv.io       as io

# ── Paths ─────────────────────────────────────────────────────────────────
FEAT_DIR = "features"
FIG_DIR  = "figures"
DATA_DIR = r"..\data"
os.makedirs(FIG_DIR, exist_ok=True)

# ── MRMR hyper-parameters ─────────────────────────────────────────────────
K = 100   # number of features to select
          # rule of thumb: sqrt(703) ≈ 27 (minimum), 50–150 is practical.
          # Set K=100 as a starting point; tune based on validation accuracy.

# ══════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════

print("Loading feature matrices...")
F_train = np.load(os.path.join(FEAT_DIR, "X_train_features.npy"))
F_val   = np.load(os.path.join(FEAT_DIR, "X_val_features.npy"))
F_test  = np.load(os.path.join(FEAT_DIR, "X_test_features.npy"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

# Load the feature index so we can name selected features
with open(os.path.join(FEAT_DIR, "feature_index.json")) as f:
    feat_meta = json.load(f)

feature_names = [feat_meta["features"][str(i)] for i in range(F_train.shape[1])]
total_dims    = F_train.shape[1]

print(f"  Train:  {F_train.shape}")
print(f"  Val:    {F_val.shape}")
print(f"  Test:   {F_test.shape}")
print(f"  Selecting top K = {K} from {total_dims} features\n")

# ── Replace any NaN/Inf (safety) ──────────────────────────────────────────
F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
F_val   = np.nan_to_num(F_val,   nan=0.0, posinf=0.0, neginf=0.0)
F_test  = np.nan_to_num(F_test,  nan=0.0, posinf=0.0, neginf=0.0)

# ══════════════════════════════════════════════════════════════════════════
#  MRMR  (fit on TRAINING set only)
# ══════════════════════════════════════════════════════════════════════════

print("Running MRMR on training features...")
t0 = time.time()

# mrmr_classif expects a pandas DataFrame for X and a Series for y
X_df = pd.DataFrame(F_train, columns=feature_names)
y_sr = pd.Series(Y_train.astype(int))

# selected_names is a list of K feature names, ordered by MRMR score
selected_names = mrmr_classif(X=X_df, y=y_sr, K=K)

elapsed = time.time() - t0
print(f"  MRMR completed in {elapsed:.1f} s")

# Recover integer indices into the original 703-dim vector
name_to_idx  = {name: i for i, name in enumerate(feature_names)}
selected_idx = np.array([name_to_idx[n] for n in selected_names], dtype=np.int64)

print(f"\n  Selected {len(selected_idx)} features (indices {selected_idx.min()}–{selected_idx.max()})")

# ── Apply selection to all splits (same indices, no refitting) ────────────
F_train_sel = F_train[:, selected_idx]
F_val_sel   = F_val[:,   selected_idx]
F_test_sel  = F_test[:,  selected_idx]

# ── Save outputs ─────────────────────────────────────────────────────────
np.save(os.path.join(FEAT_DIR, "X_train_selected.npy"), F_train_sel)
np.save(os.path.join(FEAT_DIR, "X_val_selected.npy"),   F_val_sel)
np.save(os.path.join(FEAT_DIR, "X_test_selected.npy"),  F_test_sel)
np.save(os.path.join(FEAT_DIR, "mrmr_selected_indices.npy"), selected_idx)

with open(os.path.join(FEAT_DIR, "mrmr_selected_names.json"), "w") as f:
    json.dump({
        "K": K,
        "total_before": total_dims,
        "selected_indices": selected_idx.tolist(),
        "selected_names":   selected_names,
    }, f, indent=2)

print(f"\n  Saved selected feature matrices:")
print(f"    train: {F_train_sel.shape}")
print(f"    val:   {F_val_sel.shape}")
print(f"    test:  {F_test_sel.shape}")

# ══════════════════════════════════════════════════════════════════════════
#  Family breakdown of selected features
# ══════════════════════════════════════════════════════════════════════════

FAMILY_PREFIXES = {
    "Family 1a — Color Histogram":   "fam1a",
    "Family 1b — Pixel Statistics":  "fam1b",
    "Family 2a — HOG Descriptor":    "fam2a",
    "Family 2b — Sobel Statistics":  "fam2b",
    "Family 3  — LBP Texture":       "fam3",
    "Family 4  — Spatial Color":     "fam4",
}

family_counts = {k: 0 for k in FAMILY_PREFIXES}
for name in selected_names:
    for fam_label, prefix in FAMILY_PREFIXES.items():
        if name.startswith(prefix):
            family_counts[fam_label] += 1
            break

print("\n  Family breakdown of selected features:")
for fam, count in family_counts.items():
    bar = "█" * count
    print(f"    {fam:35s}  {count:3d}  {bar}")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 1 — MRMR Selection Overview Panel
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK = "#0f1117"; PANEL = "#1a1a2e"; ACCENT = "#3498db"
FAMILY_COLOURS = ["#e74c3c","#f39c12","#2ecc71","#1abc9c","#9b59b6","#3498db"]

# ── Plot 1: feature index scatter (which indices were selected) ────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(PANEL)
unselected = np.setdiff1d(np.arange(total_dims), selected_idx)
ax1.scatter(unselected,  np.zeros(len(unselected)),
            c="#444", s=3, alpha=0.4, label="Not selected")
ax1.scatter(selected_idx, np.ones(len(selected_idx)),
            c=ACCENT, s=8, alpha=0.9, label=f"Selected (K={K})")

# Draw family boundary lines
ranges_raw = feat_meta["ranges"]
boundaries = []
for fam_label, bounds in ranges_raw.items():
    boundaries.append((bounds[0], bounds[1], fam_label.strip()))
for lo, hi, label in boundaries:
    ax1.axvline(lo,  color="#555", linewidth=0.5, linestyle="--")
    ax1.text((lo+hi)/2, 1.35, label.split("—")[-1].strip()[:18],
             color="#aaa", fontsize=6, ha="center", rotation=0)

ax1.set_xlim(-5, total_dims + 5)
ax1.set_ylim(-0.5, 1.8)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["Not selected", "Selected"], color="white", fontsize=8)
ax1.set_xlabel("Feature Index (0 – {:d})".format(total_dims - 1),
               color="white", fontsize=9)
ax1.set_title(f"MRMR Selection Map  —  K={K} of {total_dims} features",
              color="white", fontsize=10, fontweight="bold")
ax1.tick_params(colors="white", labelsize=7)
for sp in ax1.spines.values(): sp.set_edgecolor("#333")
ax1.legend(fontsize=8, facecolor=PANEL, labelcolor="white", loc="upper right")

# ── Plot 2: family pie chart ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
pie_labels  = [k.split("—")[-1].strip() for k in family_counts.keys()]
pie_values  = list(family_counts.values())
pie_colours = FAMILY_COLOURS[:len(pie_values)]
wedges, texts, autotexts = ax2.pie(
    pie_values, labels=None, colors=pie_colours,
    autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
    startangle=140, pctdistance=0.75,
    wedgeprops={"edgecolor": DARK, "linewidth": 1.5}
)
for at in autotexts: at.set_color("white"); at.set_fontsize(7)
ax2.legend(wedges, [f"{l} ({v})" for l,v in zip(pie_labels, pie_values)],
           loc="lower center", fontsize=6, facecolor=PANEL, labelcolor="white",
           bbox_to_anchor=(0.5, -0.25))
ax2.set_title("Selected Features by Family", color="white",
              fontsize=9, fontweight="bold")

# ── Plot 3: per-index selection frequency bar (family view) ───────────────
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor(PANEL)

# Colour each selected index by its family
idx_colours = []
for idx_val in selected_idx:
    name = feature_names[idx_val]
    colour = "#888"
    for i, (_, prefix) in enumerate(FAMILY_PREFIXES.items()):
        if name.startswith(prefix):
            colour = FAMILY_COLOURS[i]
            break
    idx_colours.append(colour)

ax3.bar(range(K), selected_idx, color=idx_colours, alpha=0.85, width=0.8)
ax3.set_xlabel("MRMR Rank (0 = most relevant)", color="white", fontsize=9)
ax3.set_ylabel("Original Feature Index", color="white", fontsize=9)
ax3.set_title("Original Index of Each MRMR-Selected Feature  (coloured by family)",
              color="white", fontsize=10, fontweight="bold")
ax3.tick_params(colors="white", labelsize=7)
for sp in ax3.spines.values(): sp.set_edgecolor("#333")

# Legend for family colours
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=FAMILY_COLOURS[i], label=list(FAMILY_PREFIXES.keys())[i].split("—")[-1].strip())
              for i in range(len(FAMILY_PREFIXES))]
ax3.legend(handles=legend_els, fontsize=7, facecolor=PANEL, labelcolor="white",
           loc="upper right", ncol=3)

plt.suptitle("Step 4.2 — MRMR Feature Selection Results",
             color="white", fontsize=14, fontweight="bold", y=1.01)

out = os.path.join(FIG_DIR, "mrmr_panel.png")
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\n  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Before/After Augmentation on SELECTED Features
# ══════════════════════════════════════════════════════════════════════════

print("Generating selected-feature augmentation panel...")

# 1. Load ONLY the pure original dataset
X_train_orig = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train_orig = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

# 2. FIX: Convert from uint8 [0-255] to float64 [0.0-1.0] immediately
X_train_orig = X_train_orig.astype(np.float64) / 255.0

CLASS_NAMES = [
    "butterfly","cat","chicken","cow",
    "dog","elephant","horse","sheep","spider","squirrel"
]

N_PANEL = 4
orig_imgs = X_train_orig[:N_PANEL]

# 3. FIX: Generate guaranteed matching augmented versions dynamically
from minicv.transforms import rotate, translate
aug_imgs = np.zeros_like(orig_imgs)
for i in range(N_PANEL):
    aug_imgs[i] = translate(orig_imgs[i], tx=5,ty=10)

def get_sel_features(img):
    """Extract the full feature vector then slice selected indices."""
    img_u8  = np.clip(img * 255, 0, 255).astype(np.uint8)
    gray_u8 = io.rgb_to_gray(img_u8)
    # colour histogram of selected colour features only
    h_full = features.color_histogram(img, bins=32, normalize=True)
    lbp_full = features.lbp_descriptor(gray_u8, n_bins=64, normalize=True)
    return h_full, lbp_full

fig2 = plt.figure(figsize=(20, N_PANEL * 4))
fig2.patch.set_facecolor(DARK)
cols = ["Orig Image", "Aug Image",
        "Colour Hist ORIG (sel)", "Colour Hist AUG (sel)",
        "LBP ORIG (sel)", "LBP AUG (sel)"]

for row in range(N_PANEL):
    o_img  = orig_imgs[row]
    a_img  = aug_imgs[row]
    h_o, lbp_o = get_sel_features(o_img)
    h_a, lbp_a = get_sel_features(a_img)

    # Find which colour histogram / LBP indices survived MRMR
    sel_hist_local = [i for i, n in enumerate(selected_names) if "fam1a" in n and int(n.split("bin")[-1]) < 32]
    sel_lbp_local  = [i for i, n in enumerate(selected_names) if "fam3"  in n]

    for col_i, (title, content) in enumerate(zip(cols, [
        o_img, a_img,
        (h_o, sel_hist_local, "orig"), (h_a, sel_hist_local, "aug"),
        (lbp_o, sel_lbp_local, "orig"), (lbp_a, sel_lbp_local, "aug"),
    ])):
        ax = fig2.add_subplot(N_PANEL, 6, row * 6 + col_i + 1)
        ax.set_facecolor(PANEL)

        if col_i < 2:
            ax.imshow(np.clip(content, 0, 1))
            lbl = CLASS_NAMES[Y_train_orig[row]]
            ax.set_title(f"{title}\n{lbl}", color="white", fontsize=7)
            ax.axis("off")
        elif col_i < 4:
            hist_data, sel_local, tag = content
            x = np.arange(32)
            for c, colour in enumerate(["#e74c3c","#2ecc71","#3498db"]):
                h_slice = hist_data[c*32:(c+1)*32].copy()
                # Highlight selected bins
                alphas = np.where(
                    np.isin(np.arange(32) + c*32, [name_to_idx.get(n, -1) for n in selected_names if "fam1a" in n]),
                    0.9, 0.25
                )
                for b in range(32):
                    ax.bar(x[b] + c*0.28, h_slice[b], width=0.28,
                           color=colour, alpha=float(alphas[b]))
            ax.set_title(title + "\n(bright=MRMR selected)", color="white", fontsize=6)
            ax.tick_params(colors="white", labelsize=4)
            for sp in ax.spines.values(): sp.set_edgecolor("#333")
        else:
            lbp_data, sel_local, tag = content
            colours_lbp = np.where(
                np.isin(np.arange(64),
                        [name_to_idx.get(n, -1) % 64 for n in selected_names if "fam3" in n]),
                "#9b59b6", "#444"
            )
            for b in range(64):
                ax.bar(b, lbp_data[b], width=1, color=colours_lbp[b], alpha=0.85)
            ax.set_title(title + "\n(purple=MRMR selected)", color="white", fontsize=6)
            ax.tick_params(colors="white", labelsize=4)
            for sp in ax.spines.values(): sp.set_edgecolor("#333")

plt.suptitle("Selected Features — Before vs After Augmentation  (highlighted = MRMR-chosen)",
             color="white", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
out2 = os.path.join(FIG_DIR, "feat_selected_aug_panel.png")
plt.savefig(out2, dpi=120, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Saved {out2}")

# ══════════════════════════════════════════════════════════════════════════
#  Final summary
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  MRMR Selection Complete")
print("=" * 60)
print(f"  Features before MRMR : {total_dims}")
print(f"  Features after  MRMR : {K}   (dimensionality reduction: "
      f"{100*(1-K/total_dims):.1f}%)")
print(f"\n  Family breakdown of selected K={K}:")
for fam, count in family_counts.items():
    pct = 100 * count / K
    print(f"    {fam:35s}  {count:3d}  ({pct:.0f}%)")
print("\n  Ready for model training (Step 5).")
print("  Use:  features/X_train_selected.npy")
print("        features/X_val_selected.npy")
print("        features/X_test_selected.npy")
print("=" * 60)
