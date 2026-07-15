# ============================================================
# Notebook Initialization — Dependency Check & Conditional Install/Upgrade
# ============================================================

print("[INFO] ==== Environment initialized ====")
# ============================================================
# Imports 1
# ============================================================

import platform
import os
import sys
import glob
import math
import random
import warnings
import subprocess
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

# Standard library for installed package version lookup
from importlib.metadata import version as get_installed_version, PackageNotFoundError

# ------------------------------------------------------------
# Helper function: pip install / upgrade
# ------------------------------------------------------------
def pip_install_or_upgrade(package_spec):
    print(f"[INFO] Installing/upgrading {package_spec} ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", package_spec
    ])

# ------------------------------------------------------------
# Bootstrap packaging if missing
# Needed BEFORE using packaging.version.Version
# ------------------------------------------------------------
try:
    from packaging.version import Version
except ImportError:
    print("[WARNING] 'packaging' not found. Installing packaging...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "packaging"
    ])
    from packaging.version import Version

# ------------------------------------------------------------
# Conditional install/upgrade:
# only install if missing OR upgrade if below minimum version
# ------------------------------------------------------------
def ensure_minimum_version(pip_name, min_version):
    """
    Ensure that a package is installed and its version is >= min_version.

    Behavior:
      - if package is missing -> install
      - if installed version < min_version -> upgrade
      - if installed version >= min_version -> do nothing
    """
    try:
        installed_version = get_installed_version(pip_name)
        print(f"[INFO] {pip_name} installed version: {installed_version}")

        if Version(installed_version) < Version(min_version):
            print(f"[INFO] {pip_name} is below minimum required version ({min_version}) -> upgrading...")
            pip_install_or_upgrade(f"{pip_name}>={min_version}")
        else:
            print(f"[INFO] {pip_name} already satisfies minimum version ({min_version}). Skipping.")
    except PackageNotFoundError:
        print(f"[WARNING] {pip_name} is not installed -> installing...")
        pip_install_or_upgrade(f"{pip_name}>={min_version}")

# ------------------------------------------------------------
# Core tools with minimum versions
# ------------------------------------------------------------
CORE_TOOLS = {
    "pip": "26.0.1",
    "setuptools": "82.0.1",
    "wheel": "0.46.3",
    "packaging": "26.0",
}

print(f"[INFO] Python version: {platform.python_version()}")

print("\n[INFO] Checking core packaging tools...")
for tool_name, min_version in CORE_TOOLS.items():
    ensure_minimum_version(tool_name, min_version)

# ------------------------------------------------------------
# Required Python packages for this notebook
# module_name is informative only here
# pip = package name in pip
# min_version = minimum acceptable version
# ------------------------------------------------------------
REQUIRED_PACKAGES = {
    "numpy": {"pip": "numpy", "min_version": "1.24.0"},
    "pandas": {"pip": "pandas", "min_version": "2.0.0"},
    "matplotlib": {"pip": "matplotlib", "min_version": "3.7.0"},
    "scipy": {"pip": "scipy", "min_version": "1.10.0"},
    "skimage": {"pip": "scikit-image", "min_version": "0.21.0"},
    "sklearn": {"pip": "scikit-learn", "min_version": "1.3.0"},
    "pingouin": {"pip": "pingouin", "min_version": "0.5.3"},
    "statsmodels": {"pip": "statsmodels", "min_version": "0.14.0"},
    "networkx": {"pip": "networkx", "min_version": "3.1"},
    "seaborn": {"pip": "seaborn", "min_version": "0.12.2"},
    "tqdm": {"pip": "tqdm", "min_version": "4.66.0"},
    "scikit_posthocs": {"pip": "scikit-posthocs", "min_version": "0.7.0"},
    "umap": {"pip": "umap-learn", "min_version": "0.5.4"},
    "PIL": {"pip": "pillow", "min_version": "9.5.0"},
    "cv2": {"pip": "opencv-python", "min_version": "4.8.0.76"},
    "shap": {"pip": "shap", "min_version": "0.45.0"},
    "openpyxl": {"pip": "openpyxl", "min_version": "3.1.0"},
}

print("\n[INFO] Checking required packages against minimum versions...")
for module_name, pkg_info in REQUIRED_PACKAGES.items():
    ensure_minimum_version(
        pip_name=pkg_info["pip"],
        min_version=pkg_info["min_version"]
    )

# ------------------------------------------------------------
# Verify OpenCV  and Skimage validation
# ------------------------------------------------------------

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Verify scikit-image availability
try:
    import skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

print(f"[INFO] OpenCV available: {OPENCV_AVAILABLE}")
print(f"[INFO] scikit-image available: {SKIMAGE_AVAILABLE}")
# ============================================================
# Imports 2
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pingouin as pg
import umap.umap_ as umap
import shap

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from scipy.ndimage import convolve
from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    f_oneway,
    mannwhitneyu,
    kruskal
)

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from skimage import io, exposure
from skimage.color import rgb2gray
from skimage.draw import line
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, remove_small_objects
from skimage.transform import resize
from skimage.util import img_as_ubyte

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scikit_posthocs as sp

# ------------------------------------------------------------
# Environment info
# ------------------------------------------------------------
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("matplotlib version:", plt.matplotlib.__version__)
print("seaborn version:", sns.__version__)
print("scikit-image version:", skimage.__version__)
print("OpenCV version:", cv2.__version__)
print("Python executable:", sys.executable)

print("\n[INFO] ===== Starting pipeline =====")

# ------------------------------------------------------------
# Create exports folder
# ------------------------------------------------------------


# Get the directory where THIS script is located

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

SCRIPT_DIR = get_base_dir()

EXPORT_DIR = SCRIPT_DIR / "exports"
OUTPUT_DIR = SCRIPT_DIR / "synthetic_microglia_40x"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Script directory:", SCRIPT_DIR)
print("[INFO] EXPORT_DIR:", EXPORT_DIR)
print("[INFO] OUTPUT_DIR:", OUTPUT_DIR)

# ============================================================
# Universal table exporter
# ============================================================
# Every time the script writes a .csv file, it also writes a
# matching .xlsx file. The .xlsx file is recommended for Excel
# users because it avoids regional decimal/separator problems.
# ============================================================

def _safe_excel_sheet_name(name="Data"):
    """
    Excel sheet names cannot exceed 31 characters and cannot contain
    some special characters.
    """
    invalid_chars = ["\\", "/", "*", "[", "]", ":", "?"]
    name = str(name)

    for ch in invalid_chars:
        name = name.replace(ch, "_")

    if not name.strip():
        name = "Data"

    return name[:31]


def _format_excel_sheet(ws, df):
    """
    Basic formatting for Excel exports:
    - freeze header row
    - add autofilter
    - adjust column widths
    - use scientific notation for p-values
    - preserve numeric values as real numbers
    """
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    pvalue_like_columns = {
        "p_value",
        "p_adj",
        "p_adj_global",
        "p_adj_family",
        "levene_p"
    }

    for col_idx, col_name in enumerate(df.columns, start=1):
        col_letter = ws.cell(row=1, column=col_idx).column_letter
        col_name_str = str(col_name)
        col_name_lower = col_name_str.lower()

        # Estimate readable column width
        try:
            sample_values = df[col_name].astype(str).head(200).tolist()
            max_len = max([len(col_name_str)] + [len(v) for v in sample_values])
        except Exception:
            max_len = len(col_name_str)

        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 50)

        # Numeric formatting
        try:
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name])
            is_integer = pd.api.types.is_integer_dtype(df[col_name])
            is_float = pd.api.types.is_float_dtype(df[col_name])
        except Exception:
            is_numeric = False
            is_integer = False
            is_float = False

        if is_numeric:
            if (
                col_name_lower in pvalue_like_columns
                or col_name_lower.endswith("_p")
                or "p_value" in col_name_lower
                or "p_adj" in col_name_lower
            ):
                number_format = "0.000E+00"
            elif is_integer:
                number_format = "0"
            elif is_float:
                number_format = "0.000000"
            else:
                number_format = "General"

            for row_idx in range(2, ws.max_row + 1):
                ws.cell(row=row_idx, column=col_idx).number_format = number_format


def _write_xlsx_copy_from_dataframe(df, csv_path):
    """
    Create an .xlsx copy next to a .csv file.
    """
    csv_path = Path(csv_path)
    xlsx_path = csv_path.with_suffix(".xlsx")

    try:
        sheet_name = _safe_excel_sheet_name(csv_path.stem)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

            ws = writer.sheets[sheet_name]
            _format_excel_sheet(ws, df)

        print(f"[INFO] Excel copy saved: {xlsx_path}")

    except Exception as e:
        print(f"[WARNING] Could not create Excel copy for {csv_path}: {e}")


# ------------------------------------------------------------
# Patch pandas DataFrame.to_csv once
# ------------------------------------------------------------
# This keeps all existing df.to_csv(...) calls working normally,
# but automatically creates a matching .xlsx file.
# ------------------------------------------------------------

if not hasattr(pd.DataFrame, "_original_to_csv_for_dual_export"):
    pd.DataFrame._original_to_csv_for_dual_export = pd.DataFrame.to_csv

    def _to_csv_and_xlsx(self, path_or_buf=None, *args, **kwargs):
        # First, write the normal CSV exactly as before
        result = pd.DataFrame._original_to_csv_for_dual_export(
            self,
            path_or_buf,
            *args,
            **kwargs
        )

        # Then, if the output is a CSV file path, create an XLSX copy
        if path_or_buf is not None:
            try:
                output_path = Path(path_or_buf)

                if output_path.suffix.lower() == ".csv":
                    _write_xlsx_copy_from_dataframe(self, output_path)

            except Exception as e:
                print(f"[WARNING] Automatic XLSX export failed for {path_or_buf}: {e}")

        return result

    pd.DataFrame.to_csv = _to_csv_and_xlsx

# Automatically save every figure generated in the notebook
plt.rcParams["savefig.directory"] = os.getcwd()
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.format"] = "png"

print(f"[INFO] OpenCV available: {OPENCV_AVAILABLE}")
print(f"[INFO] scikit-image available: {SKIMAGE_AVAILABLE}")
print(f"[INFO] PIL available: {PIL_AVAILABLE}")

# ============================================================
# Plotting guide
# ============================================================

def lighten_color(color, amount=0.6):
    """
    Lighten a matplotlib color.
    amount=0 -> original color
    amount=1 -> white
    """
    import colorsys
    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except Exception:
        c = color

    r, g, b = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - amount * (1 - l)
    return colorsys.hls_to_rgb(h, l, s)


def should_use_boxplot(df, group_col="cell_state", image_threshold=5, group_threshold=5):
    """
    Switch from per-image bars to grouped boxplot + points when:
      - number of groups > group_threshold
      - OR any group has more than image_threshold images
    """
    if df.empty or group_col not in df.columns:
        return False

    n_groups = df[group_col].nunique(dropna=True)
    max_images_in_group = df[group_col].value_counts(dropna=True).max()

    return (n_groups > group_threshold) or (max_images_in_group > image_threshold)


def build_group_color_map(groups):
    return {
        state: plt.cm.tab10(i % 10)
        for i, state in enumerate(groups)
    }


def plot_bars_or_boxplot(
    df,
    value_col,
    group_col="cell_state",
    filename_col="sample_id",
    state_order=None,
    ylabel=None,
    xlabel=None,
    title=None,
    figsize_bar=(14, 6),
    figsize_box=(10, 6),
    save_path=None,
    image_threshold=5,
    group_threshold=5,
    show_legend=True
):
    """
    Universal plotting function:
      - small dataset -> bar plot per image
      - large dataset -> boxplot by group + individual points
    """
    df_plot = df.copy().dropna(subset=[value_col, group_col])

    if df_plot.empty:
        print(f"[WARNING] No valid data to plot for: {value_col}")
        return

    # Group ordering
    if state_order is None:
        groups = list(df_plot[group_col].dropna().unique())
    else:
        groups = [g for g in state_order if g in df_plot[group_col].dropna().unique()]
        groups += [g for g in df_plot[group_col].dropna().unique() if g not in groups]

    df_plot[group_col] = pd.Categorical(df_plot[group_col], categories=groups, ordered=True)

    sort_cols = [group_col]
    if filename_col in df_plot.columns:
        sort_cols.append(filename_col)

    df_plot = df_plot.sort_values(sort_cols).reset_index(drop=True)

    color_map = build_group_color_map(groups)
    use_box = should_use_boxplot(
        df_plot,
        group_col=group_col,
        image_threshold=image_threshold,
        group_threshold=group_threshold
    )

    # --------------------------------------------------------
    # CASE 1 — bar plot per image
    # --------------------------------------------------------
    if not use_box:
        df_plot["color"] = df_plot[group_col].astype(object).map(color_map)

        plt.figure(figsize=figsize_bar)

        plt.bar(
            x=np.arange(len(df_plot)),
            height=df_plot[value_col],
            color=df_plot["color"],
            edgecolor="black",
            linewidth=0.7
        )

        if filename_col in df_plot.columns:
            plt.xticks(
                ticks=np.arange(len(df_plot)),
                labels=df_plot[filename_col],
                rotation=75,
                ha="right",
                fontsize=8
            )
        else:
            plt.xticks(np.arange(len(df_plot)))

        plt.ylabel(ylabel if ylabel else value_col)
        plt.xlabel(xlabel if xlabel else "Image Filename")
        plt.title(title if title else value_col)

        if show_legend:
            handles = [
                plt.Line2D([0], [0], color=color_map[state], lw=8, label=state)
                for state in groups
            ]
            plt.legend(handles=handles, title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

        ax = plt.gca()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")

        plt.show()
        return

    # --------------------------------------------------------
    # CASE 2 — boxplot + individual points
    # --------------------------------------------------------
    plt.figure(figsize=figsize_box)

    light_palette = {g: lighten_color(color_map[g], 0.6) for g in groups}

    sns.boxplot(
        data=df_plot,
        x=group_col,
        y=value_col,
        hue=group_col,
        order=groups,
        palette=light_palette,
        dodge=False,
        width=0.55,
        showfliers=False,
        linewidth=1.2
    )

    sns.stripplot(
        data=df_plot,
        x=group_col,
        y=value_col,
        hue=group_col,
        order=groups,
        palette=color_map,
        dodge=False,
        jitter=0.12,
        alpha=0.9,
        size=5,
        edgecolor="black",
        linewidth=0.4
    )

    ax = plt.gca()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.ylabel(ylabel if ylabel else value_col)
    plt.xlabel(xlabel if xlabel else "Group")
    plt.title(title if title else f"{value_col} by group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()

print("[INFO] ===== Environment Created =====")
# ============================================================
# Synthetic Microglia Generator — 5 Functional States
# ============================================================
# Processes = GREEN channel
# Soma = GREEN
# Nucleus = BLUE
# ============================================================
print("[INFO] ===== Module 0 - Synthetic Microglia =====")
# ============================================================
# Global configuration
# ============================================================
N_IMAGES_TOTAL = 25
IMAGE_SIZE = (512, 512)
BACKGROUND_LEVEL = 0.0
MAX_INTENSITY = 1.0
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

CELL_STATES = [
    "resting_microglia",
    "primed_microglia",
    "activated_microglia",
    "ameboid_microglia",
    "resolving_microglia"
]

IMAGES_PER_STATE = N_IMAGES_TOTAL // len(CELL_STATES)

# ============================================================
# Utility functions
# ============================================================

def create_empty_image():
    """Return a pure black RGB image."""
    return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

def add_green(img, rr, cc, intensity):
    """Add intensity to GREEN channel."""
    img[rr, cc, 1] = np.clip(img[rr, cc, 1] + intensity, 0, MAX_INTENSITY)
    return img

def add_blue(img, rr, cc, intensity):
    """Add intensity to BLUE channel."""
    img[rr, cc, 2] = np.clip(img[rr, cc, 2] + intensity, 0, MAX_INTENSITY)
    return img

def draw_soma(img, center, radius, intensity=0.8):
    """Draw soma in GREEN, nucleus in BLUE."""
    yy, xx = np.ogrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
    mask = (yy - center[0])**2 + (xx - center[1])**2 <= radius**2

    # Soma (green)
    img[mask, 1] = np.clip(img[mask, 1] + intensity, 0, MAX_INTENSITY)

    # Nucleus (blue)
    nucleus_mask = (yy - center[0])**2 + (xx - center[1])**2 <= (radius * 0.45)**2
    img[nucleus_mask, 2] = np.clip(img[nucleus_mask, 2] + 0.9, 0, MAX_INTENSITY)

    return img

def draw_branch(img, start, length, n_segments, angle_spread, base_angle,
                thickness=1, intensity=0.7, arborization_prob=0.0):

    y, x = start
    seg_length = length / n_segments
    current_angle = base_angle

    for _ in range(n_segments):
        current_angle += np.random.uniform(-angle_spread, angle_spread)

        dy = seg_length * np.sin(current_angle)
        dx = seg_length * np.cos(current_angle)

        new_y = int(y + dy)
        new_x = int(x + dx)

        rr, cc = line(int(y), int(x), new_y, new_x)

        # Processes in GREEN
        for t in range(-thickness, thickness + 1):
            rr_shift = np.clip(rr + t, 0, IMAGE_SIZE[0] - 1)
            cc_shift = np.clip(cc + t, 0, IMAGE_SIZE[1] - 1)
            img = add_green(img, rr_shift, cc_shift, intensity)

        # Arborization
        if np.random.rand() < arborization_prob:
            side_angle = current_angle + np.random.uniform(-np.pi/3, np.pi/3)
            side_len = length * np.random.uniform(0.1, 0.3)

            img, _ = draw_branch(
                img,
                start=(new_y, new_x),
                length=side_len,
                n_segments=max(2, n_segments // 3),
                angle_spread=angle_spread,
                base_angle=side_angle,
                thickness=max(1, thickness - 1),
                intensity=intensity * 0.8,
                arborization_prob=arborization_prob * 0.7
            )

        y, x = new_y, new_x

    return img, (y, x)

def add_gaussian_noise(img, sigma=0.02):
    noise = np.random.normal(0, sigma, img.shape)
    img = img + noise
    return np.clip(img, 0, MAX_INTENSITY)

def apply_confocal_blur(img, sigma=1.0):
    """Blur all channels slightly for realism."""
    blurred = img.copy()
    for c in range(3):
        blurred[:, :, c] = gaussian(img[:, :, c], sigma=sigma, preserve_range=True)
    return blurred

# ============================================================
# Microglia Morphology Generators
# ============================================================

def generate_resting_microglia():
    img = create_empty_image()
    center = (256, 256)
    img = draw_soma(img, center, radius=10, intensity=0.8)

    n_branches = np.random.randint(8, 12)
    for _ in range(n_branches):
        angle = np.random.uniform(0, 2*np.pi)
        img, _ = draw_branch(
            img, center,
            length=np.random.uniform(120, 200),
            n_segments=12,
            angle_spread=np.deg2rad(8),
            base_angle=angle,
            thickness=1,
            intensity=0.6,
            arborization_prob=0.5
        )

    return add_gaussian_noise(apply_confocal_blur(img, sigma=1.0), sigma=0.03)

def generate_primed_microglia():
    img = create_empty_image()
    center = (256, 256)
    img = draw_soma(img, center, radius=12, intensity=0.85)

    n_branches = np.random.randint(6, 9)
    for _ in range(n_branches):
        angle = np.random.uniform(0, 2*np.pi)
        img, _ = draw_branch(
            img, center,
            length=np.random.uniform(100, 150),
            n_segments=10,
            angle_spread=np.deg2rad(12),
            base_angle=angle,
            thickness=1,
            intensity=0.7,
            arborization_prob=0.25
        )

    return add_gaussian_noise(apply_confocal_blur(img, sigma=1.1), sigma=0.035)

def generate_activated_microglia():
    img = create_empty_image()
    center = (256, 256)
    img = draw_soma(img, center, radius=14, intensity=0.9)

    n_branches = np.random.randint(3, 6)
    for _ in range(n_branches):
        angle = np.random.uniform(0, 2*np.pi)
        img, _ = draw_branch(
            img, center,
            length=np.random.uniform(40, 80),
            n_segments=5,
            angle_spread=np.deg2rad(20),
            base_angle=angle,
            thickness=2,
            intensity=0.8,
            arborization_prob=0.1
        )

    return add_gaussian_noise(apply_confocal_blur(img, sigma=1.2), sigma=0.04)

def generate_ameboid_microglia():
    img = create_empty_image()
    center = (256, 256)
    img = draw_soma(img, center, radius=18, intensity=1.0)

    for _ in range(np.random.randint(0, 3)):
        angle = np.random.uniform(0, 2*np.pi)
        img, _ = draw_branch(
            img, center,
            length=np.random.uniform(20, 40),
            n_segments=3,
            angle_spread=np.deg2rad(25),
            base_angle=angle,
            thickness=3,
            intensity=0.9,
            arborization_prob=0.0
        )

    return add_gaussian_noise(apply_confocal_blur(img, sigma=1.3), sigma=0.05)

def generate_resolving_microglia():
    img = create_empty_image()
    center = (256, 256)
    img = draw_soma(img, center, radius=12, intensity=0.85)

    n_branches = np.random.randint(4, 7)
    for _ in range(n_branches):
        angle = np.random.uniform(0, 2*np.pi)
        img, _ = draw_branch(
            img, center,
            length=np.random.uniform(80, 120),
            n_segments=8,
            angle_spread=np.deg2rad(15),
            base_angle=angle,
            thickness=1,
            intensity=0.7,
            arborization_prob=0.2
        )

    return add_gaussian_noise(apply_confocal_blur(img, sigma=1.1), sigma=0.035)

# ============================================================
# Main generation loop
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] OUTPUT_DIR:", OUTPUT_DIR)
    print("[INFO] OUTPUT_DIR abs:", OUTPUT_DIR.resolve())

    generators = {
        "resting_microglia": generate_resting_microglia,
        "primed_microglia": generate_primed_microglia,
        "activated_microglia": generate_activated_microglia,
        "ameboid_microglia": generate_ameboid_microglia,
        "resolving_microglia": generate_resolving_microglia
    }

    records = []
    counter = 0

    for cell_state in CELL_STATES:
        for _ in range(IMAGES_PER_STATE):
            img = generators[cell_state]()
            fname = f"{cell_state}_{counter:03d}.png"
            save_path = OUTPUT_DIR / fname

            plt.imsave(save_path, img)
            
            records.append({"filename": fname, "cell_state": cell_state})
            counter += 1

    labels_path = OUTPUT_DIR / "labels.csv"
    pd.DataFrame(records).to_csv(labels_path, index=False)

# ============================================================
# Visualization panel
# ============================================================

def visualize_panel():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".png")])[:25]

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    for ax, fname in zip(axes.flatten(), files):
        img = plt.imread(OUTPUT_DIR / fname)
        ax.imshow(img)
        ax.set_title(fname, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(EXPORT_DIR / "Module 0 - synthetic_microglia_panel.png", dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
    visualize_panel()

print("\n[INFO] ===== Module 0 Completed =====")

# ============================================================
# Universal Multiclass Dataset Loader
# ============================================================

print("\n[INFO] ===== Module 1 - Dataset Loader =====")

# ------------------------------------------------------------
# Folder selector
# ------------------------------------------------------------
def select_dataset_directory():
    root = tk.Tk()
    root.withdraw()                     
    root.attributes("-topmost", True) 
    root.update()

    selected_dir = filedialog.askdirectory(
        title="Select microglia dataset folder",
        mustexist=True
    )

    root.destroy()

    if not selected_dir:
        raise ValueError("No folder selected.")

    return selected_dir

# ------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------
def load_dataset_multiclass(base_dir, extensions=("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")):
    base_dir = os.path.abspath(base_dir)

    image_paths = []
    cell_states = []
    display_names = []
    unique_ids = []

    # Recursive search: supports images in subfolders and sub‑subfolders
    for ext in extensions:
        pattern = os.path.join(base_dir, "**", ext)
        for img in sorted(glob.glob(pattern, recursive=True)):
            img_abs = os.path.abspath(img)
            rel_path = os.path.relpath(img_abs, base_dir).replace("\\", "/")
            parts = Path(rel_path).parts

            # If folders are present, the group will be the first folder
            if len(parts) >= 2:
                state_name = parts[0]
            else:
                fname = os.path.basename(img_abs)
                state_name = os.path.splitext(fname)[0].rsplit("_", 1)[0]

            image_paths.append(img_abs)
            cell_states.append(state_name)
            display_names.append(os.path.basename(img_abs))  # Show Short name
            unique_ids.append(rel_path)                      # Unique identifier

    if len(image_paths) == 0:
        print("[WARNING] No images found.")
        return [], [], [], [], []

    classes = list(dict.fromkeys(cell_states))

    print(f"[INFO] Detected groups: {classes}")
    print(f"[INFO] Loaded {len(image_paths)} images from: {base_dir}")

    return image_paths, cell_states, classes, display_names, unique_ids
# ============================================================
# Select folder and load dataset
# ============================================================

try:
    DATASET_DIR = select_dataset_directory()
except Exception:
    DATASET_DIR = r"your/path/here/dataset"

#
all_paths, all_cell_states, class_names, all_display_names, all_unique_ids = load_dataset_multiclass(DATASET_DIR)
state_order = class_names

print("\n[INFO] ===== Module 1 Completed =====")

# ============================================================
# Core Image Processing Functions
# ============================================================

print("\n[INFO] ===== Module 2 - Core Image Processing Functions =====")

# ------------------------------------------------------------
# Skeleton for fractal dimension
# ------------------------------------------------------------
def compute_skeleton_for_fd(binary):
    """
    Simple and permissive FD skeleton:
    - keep most foreground signal
    - mild smoothing
    - no largest-component filtering
    - no spur pruning
    """
    binary = binary.astype(bool)

    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)

    binary_clean = remove_small_objects(binary, max_size=29)
    binary_smooth = gaussian(binary_clean.astype(float), sigma=1.0) > 0.2
    skel = skeletonize(binary_smooth)
    return skel

print("\n[INFO] ===== Module 2 Completed =====")

# ============================================================
# Microglia preprocessing + Visual Inspection Panel
# ============================================================

print("\n[INFO] ===== Module 3 - Microglia preprocessing =====")

# ============================================================
# Module 3 configuration
# ============================================================

# Recommended for high-resolution microglia images.
# Use (512, 512) if processing is too slow.
PROCESS_TARGET_SIZE = (1024, 1024)

# Adaptive threshold is better for fluorescence images with
# bright soma + faint processes + uneven background.
USE_ADAPTIVE_THRESHOLD = True

# En skimage threshold_local usa: threshold = local_mean - offset
# Por eso offset positivo puede hacer que el fondo negro entre como objeto.
# Usamos offset negativo para hacerlo más estricto.
ADAPTIVE_OFFSET = -5

# Evita que el fondo muy oscuro se segmente aunque el threshold local falle.
MIN_SIGNAL_INTENSITY = 10

# Object size must increase when image size increases.
# Suggested:
#   512  -> 80-150
#   1024 -> 200-400
MIN_OBJECT_SIZE = 250

# Fill only small holes inside the segmented cell.
MIN_HOLE_SIZE = 500

# Mild blur to reduce pixel noise before thresholding.
GAUSSIAN_BLUR = True

# Keep conservative fluorescence enhancement.
AUTO_ENHANCE = True

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_image(path):
    """
    Load image and convert RGB/RGBA to grayscale.

    This keeps your preferred behavior:
    RGB image -> grayscale image.

    The output is always uint8 in range 0-255.
    """

    # --- OpenCV backend ---
    if OPENCV_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            # RGB/BGR/RGBA/BGRA -> grayscale
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = exposure.rescale_intensity(
                img.astype(np.float32),
                out_range=(0, 255)
            ).astype(np.uint8)

            return img

    # --- scikit-image backend ---
    try:
        img = io.imread(path)

        # RGB/RGBA -> grayscale
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = rgb2gray(img)

        img = img.astype(np.float32)

        if img.max() > 0:
            img = exposure.rescale_intensity(
                img,
                out_range=(0, 255)
            )

        return img.astype(np.uint8)

    except Exception:
        pass

    # --- PIL backend ---
    if PIL_AVAILABLE:
        try:
            img = Image.open(path).convert("L")
            img = np.array(img).astype(np.float32)

            if img.max() > 0:
                img = exposure.rescale_intensity(
                    img,
                    out_range=(0, 255)
                )

            return img.astype(np.uint8)

        except Exception:
            pass

    raise ValueError(f"[ERROR] Could not load image: {path}")

# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------
def _remove_small_components(binary, min_size):
    labeled_img = label(binary)
    cleaned = np.zeros_like(binary, dtype=bool)

    for region in regionprops(labeled_img):
        if region.area >= min_size:
            cleaned[labeled_img == region.label] = True

    return cleaned

def auto_enhance_fluorescence(
    img,
    p_low=1,
    p_high=99.5,
    min_p95_intensity=80,
    max_gain=2.0
):
    """
    Conservative rescue of weak fluorescence.
    Only boosts clearly dim images and caps the gain.
    Avoids homogenizing biological differences.
    """

    img = img.astype(np.float32)

    if img.max() <= 1.0:
        img *= 255.0

    if img.max() == 0:
        return img.astype(np.uint8)

    p95 = np.percentile(img, 95)

    # Only rescue underexposed images
    if p95 < min_p95_intensity and p95 > 0:
        gain = min(min_p95_intensity / p95, max_gain)
        img = img * gain

    # Mild percentile clipping only
    low, high = np.percentile(img, (p_low, p_high))

    if high > low:
        img = exposure.rescale_intensity(
            img,
            in_range=(low, high),
            out_range=(0, 255)
        )

    return np.clip(img, 0, 255).astype(np.uint8)

def _fill_small_holes(binary, min_hole_size):
    inv = ~binary
    labeled_img = label(inv)
    filled = binary.copy()

    for region in regionprops(labeled_img):
        if region.area < min_hole_size:
            coords = region.coords
            filled[coords[:, 0], coords[:, 1]] = True

    return filled

def preprocess_image(
    img,
    target_size=PROCESS_TARGET_SIZE,
    use_adaptive_threshold=USE_ADAPTIVE_THRESHOLD,
    adaptive_offset=ADAPTIVE_OFFSET,
    min_signal_intensity=MIN_SIGNAL_INTENSITY,
    gaussian_blur=GAUSSIAN_BLUR,
    min_object_size=MIN_OBJECT_SIZE,
    min_hole_size=MIN_HOLE_SIZE,
    clean_holes=True,
    auto_enhance=AUTO_ENHANCE
):
    """
    Preprocess grayscale fluorescence image:
      1. Resize to common size for comparable metrics
      2. Enhance fluorescence conservatively
      3. Mild blur
      4. Adaptive or Otsu threshold
      5. Remove small objects
      6. Fill small holes
      7. Skeletonize
    """

    # --------------------------------------------------------
    # Resize to common size
    # --------------------------------------------------------
    if target_size is None:
        img_resized = img.copy()
    else:
        if OPENCV_AVAILABLE:
            # INTER_AREA is good for downsampling.
            # INTER_CUBIC is better if an image is smaller than target_size.
            if img.shape[0] > target_size[0] or img.shape[1] > target_size[1]:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC

            img_resized = cv2.resize(
                img,
                target_size,
                interpolation=interpolation
            )
        else:
            img_resized = resize(
                img,
                target_size,
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.uint8)

    # --------------------------------------------------------
    # Automatic fluorescence enhancement
    # --------------------------------------------------------
    if auto_enhance:
        img_resized = auto_enhance_fluorescence(
            img_resized,
            p_low=1,
            p_high=99.8,
            min_p95_intensity=70,
            max_gain=1.5
        )

    # --------------------------------------------------------
    # Optional blur
    # --------------------------------------------------------
    if gaussian_blur:
        if OPENCV_AVAILABLE:
            img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)
        else:
            img_blur = gaussian(
                img_resized,
                sigma=0.5,
                preserve_range=True
            ).astype(np.uint8)
    else:
        img_blur = img_resized

    # --------------------------------------------------------
    # Thresholding
    # --------------------------------------------------------
    if use_adaptive_threshold:
        # Window size adapted to image size.
        # For 1024 px this gives ~71 px.
        # For 512 px this gives ~51 px.
        block_size = int(min(img_blur.shape) * 0.07)

        # threshold_local requires odd block size
        if block_size % 2 == 0:
            block_size += 1

        block_size = max(block_size, 51)

        adaptive_thresh = threshold_local(
            img_blur,
            block_size=block_size,
            method="gaussian",
            offset=adaptive_offset
        )

        # Two conditions:
        # 1. Pixel must be above local threshold.
        # 2. Pixel must have minimum real fluorescence signal.
        binary = (img_blur > adaptive_thresh) & (img_blur > min_signal_intensity)

    else:
        thresh = threshold_otsu(img_blur)

        # Also apply minimum signal filter for safety.
        binary = (img_blur > thresh) & (img_blur > min_signal_intensity)

    # --------------------------------------------------------
    # Remove small objects
    # --------------------------------------------------------
    if min_object_size > 0:
        binary = _remove_small_components(binary, min_object_size)

    # --------------------------------------------------------
    # Fill small holes
    # --------------------------------------------------------
    if clean_holes and min_hole_size > 0:
        binary = _fill_small_holes(binary, min_hole_size)

    # --------------------------------------------------------
    # Skeletons
    # --------------------------------------------------------
    skeleton_visual = skeletonize(binary)
    skeleton_fd = compute_skeleton_for_fd(binary)

    # --------------------------------------------------------
    # Convert to uint8
    # --------------------------------------------------------
    binary_uint8 = binary.astype(np.uint8) * 255
    skeleton_visual_uint8 = skeleton_visual.astype(np.uint8) * 255
    skeleton_fd_uint8 = skeleton_fd.astype(np.uint8) * 255

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------
    coords = np.column_stack(np.where(binary))

    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = (y_min, x_min, y_max, x_max)
    else:
        bbox = (0, 0, img_resized.shape[0], img_resized.shape[1])

    return (
        img_resized,
        binary_uint8,
        skeleton_visual_uint8,
        skeleton_fd_uint8,
        bbox
    )

# ------------------------------------------------------------
# Visual inspection panel
# ------------------------------------------------------------
def visualize_preprocessing_panel_ordered(
    image_paths,
    cell_states,
    preprocessed_data,
    max_classes=5,
    max_images_per_class=10,
    preferred_order=None,
    save_path=None
):
    if save_path is None:
        save_path = os.path.join(EXPORT_DIR, "Module 3 - processing.png")

    class_to_images = defaultdict(list)
    for p, s in zip(image_paths, cell_states):
        class_to_images[s].append(p)

    classes = list(class_to_images.keys())

    if preferred_order is not None:
        ordered = [c for c in preferred_order if c in classes]
        ordered += [c for c in classes if c not in ordered]
    else:
        ordered = classes

    selected = []
    for cls in ordered[:max_classes]:
        for p in class_to_images[cls][:max_images_per_class]:
            selected.append((p, cls))

    fig, axes = plt.subplots(
        len(selected), 4,
        figsize=(12, 3 * len(selected)),
        squeeze=False
    )

    for i, (path, state) in enumerate(selected):
        try:
            data = preprocessed_data[path]

            # original en gris
            img_original = load_image(path)

            r = data["img_resized"]
            b = data["binary_uint8"]
            s = data["skeleton_visual_uint8"]

            axes[i, 0].imshow(img_as_ubyte(img_original), cmap="gray")
            axes[i, 0].set_title(f"{state}\n{data['sample_id']}", fontsize=8)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(r, cmap="gray")
            axes[i, 1].set_title(f"Resized {PROCESS_TARGET_SIZE[0]}x{PROCESS_TARGET_SIZE[1]}")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(b, cmap="gray")
            axes[i, 2].set_title("Binary")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(s, cmap="gray")
            axes[i, 3].set_title("Skeleton")
            axes[i, 3].axis("off")

        except Exception as e:
            for j in range(4):
                axes[i, j].text(0.5, 0.5, str(e), ha="center", va="center")
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    
# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
# ============================================================
# Preprocessing cache
# ============================================================

path_to_display = dict(zip(all_paths, all_display_names))
path_to_uid = dict(zip(all_paths, all_unique_ids))

preprocessed_data = {}

for path, cell_state in zip(all_paths, all_cell_states):
    try:
        img = load_image(path)

        img_resized, binary_uint8, skeleton_visual_uint8, skeleton_fd_uint8, bbox = preprocess_image(
            img,
            target_size=PROCESS_TARGET_SIZE,
            use_adaptive_threshold=USE_ADAPTIVE_THRESHOLD,
            adaptive_offset=ADAPTIVE_OFFSET,
            min_signal_intensity=MIN_SIGNAL_INTENSITY,
            gaussian_blur=GAUSSIAN_BLUR,
            min_object_size=MIN_OBJECT_SIZE,
            min_hole_size=MIN_HOLE_SIZE,
            clean_holes=True,
            auto_enhance=AUTO_ENHANCE
        )

        preprocessed_data[path] = {
            "filename": path_to_display[path],
            "sample_id": path_to_uid[path],
            "cell_state": cell_state,
            "img_resized": img_resized,
            "binary_uint8": binary_uint8,
            "skeleton_visual_uint8": skeleton_visual_uint8,
            "skeleton_fd_uint8": skeleton_fd_uint8,
            "bbox": bbox
        }

    except Exception as e:
        print(f"[PREPROCESS ERROR] {path}: {e}")

visualize_preprocessing_panel_ordered(
    all_paths,
    all_cell_states,
    preprocessed_data=preprocessed_data,
    max_classes=5,
    max_images_per_class=10,
    preferred_order=state_order,
    save_path=os.path.join(EXPORT_DIR, "Module 3 - processing.png")
)

# ============================================================
# Fractal Dimension (Box-Counting Method)
# ============================================================
# Universal, reproducible, and robust implementation.
#
# Key design principles:
#   - Works with ANY microglial activation state
#   - Uses boolean masks (binary or skeleton) safely
#   - Avoids superfluous outputs
#   - Handles empty or degenerate masks gracefully
#   - Uses powers of two for reproducibility across datasets
#
# Returned values:
#   fd       → estimated fractal dimension
#   sizes    → box sizes used
#   counts   → number of non-empty boxes at each scale
#   coeffs   → (slope, intercept) from log–log regression
#
# Researchers may plug ANY microglial binary structure here:
#   - binary mask
#   - FD skeleton
#   - thresholded channels
# ============================================================

def fractal_dimension(binary_img):
    """
    Classical and permissive box-counting FD.
    No foreground cropping, no extra filtering.
    """
    binary = binary_img.astype(bool)

    if not np.any(binary):
        return 0.0, [], [], (0.0, 0.0)

    h, w = binary.shape
    max_size = min(h, w)

    sizes = []
    size = max_size

    while size >= 2:
        sizes.append(size)
        size //= 2

    counts = []

    for size in sizes:
        n_h = h // size
        n_w = w // size

        count = 0
        for i in range(n_h):
            for j in range(n_w):
                block = binary[i * size:(i + 1) * size, j * size:(j + 1) * size]
                if np.any(block):
                    count += 1

        counts.append(count)

    sizes_log = np.log(1 / np.array(sizes))
    counts_log = np.log(np.array(counts))

    coeffs = np.polyfit(sizes_log, counts_log, 1)
    fd = coeffs[0]

    return fd, sizes, counts, coeffs

# ============================================================
# Fractal Dimension Summary Table
# ============================================================

preferred_order = state_order
fd_rows = []

for path, cell_state in zip(all_paths, all_cell_states):
    try:
        data = preprocessed_data[path]

        binary_uint8 = data["binary_uint8"]
        binary_bool = binary_uint8.astype(bool)

        if np.any(binary_bool):
            fd, _, _, _ = fractal_dimension(binary_bool)
        else:
            fd = None

        fd_rows.append({
            "filename": data["filename"],
            "sample_id": data["sample_id"],
            "cell_state": data["cell_state"],
            "fractal_dimension": fd
        })

    except Exception as e:
        fd_rows.append({
            "filename": os.path.basename(path),
            "sample_id": path,
            "cell_state": cell_state,
            "fractal_dimension": None,
            "error": str(e)
        })

df_fd = pd.DataFrame(fd_rows)

if preferred_order is not None:
    df_fd["order_key"] = df_fd["cell_state"].apply(
        lambda x: state_order.index(x) if x in state_order else len(state_order)
    )
    df_fd = df_fd.sort_values(by=["order_key", "filename"]).drop(columns=["order_key"])
else:
    df_fd = df_fd.sort_values(by=["cell_state", "filename"])

df_fd = df_fd[["filename", "sample_id", "cell_state", "fractal_dimension"]]

# Uses dot decimal and comma separator.
df_fd.to_csv(os.path.join(EXPORT_DIR, "Module 3 - fractal_dimension_summary_table.csv"), index=False)

df_fd

# ------------------------------------------------------------
# Apply ordering if provided
# ------------------------------------------------------------
if preferred_order is not None:
    df_fd["order_key"] = df_fd["cell_state"].apply(
        lambda x: state_order.index(x) if x in state_order else len(state_order)
    )

    df_fd = df_fd.sort_values(by=["order_key", "filename"]).drop(columns=["order_key"])

else:
    df_fd = df_fd.sort_values(by=["cell_state", "filename"])

# ------------------------------------------------------------
# Final clean table
# ------------------------------------------------------------
df_fd = df_fd[["filename", "sample_id", "cell_state", "fractal_dimension"]]

# ------------------------------------------------------------
# EXPORT TABLE (Fractal Dimension Summary Table)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_fd.to_csv(os.path.join(EXPORT_DIR, "Module 3 - fractal_dimension_summary_table.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_fd

# ============================================================
# Plot of Fractal Dimension
# ============================================================

plot_bars_or_boxplot(
    df=df_fd,
    value_col="fractal_dimension",
    group_col="cell_state",
    filename_col="sample_id",
    state_order=state_order,
    ylabel="Fractal Dimension (FD)",
    xlabel="Image Filename / Group",
    title="Fractal Dimension Summary",
    save_path=os.path.join(EXPORT_DIR, "Module 3 - bar_plot_of_fractal_dimension.png")
)

print("\n[INFO] ===== Module 3 Completed =====")

# ============================================================
# Lacunarity Analysis (Gliding-Box Method)
# ============================================================
#
# Key design principles:
#   - Works with any microglial activation state
#   - Accepts any binary structure (mask or skeleton)
#   - Avoids unnecessary outputs
#   - Handles empty or degenerate masks safely
#   - Uses deterministic box sizes for reproducibility
#
# Returned values:
#   lac_values  → lacunarity values for each valid box size
#   valid_sizes → box sizes actually used in the computation
#
# Researchers may freely modify `box_sizes` depending on resolution, magnification, or biological scale of interest.
# ============================================================

print("\n[INFO] ===== Module 4 - Lacunarity Curves =====")

def compute_lacunarity(binary_img, box_sizes=[4, 8, 16, 32, 64, 128, 256]):
    """
    Fast lacunarity using an integral image.

    This avoids the very slow nested Python loops over every window.
    Recommended when images are resized to 1024x1024.
    """

    binary = binary_img.astype(np.uint8)
    h, w = binary.shape

    if not np.any(binary):
        return [np.nan] * len(box_sizes), []

    lac_values = []
    valid_sizes = []

    # Integral image:
    # integral[y, x] contains the sum of pixels above and left of (y, x)
    integral = np.pad(binary, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    for r in box_sizes:
        if r > h or r > w:
            continue

        # Fast sum of every r x r window
        sums = (
            integral[r:, r:]
            - integral[:-r, r:]
            - integral[r:, :-r]
            + integral[:-r, :-r]
        ).astype(np.float64)

        sums = sums.ravel()

        mean = sums.mean()
        var = sums.var()

        if mean == 0:
            lac = np.nan
        else:
            lac = var / (mean ** 2) + 1

        lac_values.append(lac)
        valid_sizes.append(r)

    return lac_values, valid_sizes

# ============================================================
# Lacunarity Curves for All Images
# ============================================================
# This block computes and plots lacunarity curves (Λ vs. box size) for each microglial image in the dataset.
#
# Design principles:
#   - Universal: works with any microglial activation state
#   - Reproducible: deterministic ordering and processing
#   - Minimalistic: only essential information is shown
#   - Robust: handles empty masks and processing errors safely
#
# Each subplot shows:
#   - box sizes used
#   - corresponding lacunarity values
#   - filename for identification
#
# Lacunarity helps distinguish microglial states:
#   - Resting microglia → higher Λ (heterogeneous, ramified)
#   - Activated / ameboid → lower Λ (compact, homogeneous)
#   - Primed / resolving → intermediate Λ
# ============================================================

lac_rows = []

n = len(all_paths)
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, (path, cell_state) in enumerate(zip(all_paths, all_cell_states)):

    try:
        data = preprocessed_data[path]

        binary_uint8 = data["binary_uint8"]
        binary_bool = binary_uint8.astype(bool)

        lac_values, sizes = compute_lacunarity(binary_bool)

        lac_rows.append({
            "filename": data["filename"],
            "sample_id": data["sample_id"],
            "cell_state": data["cell_state"],
            "lacunarity_mean": np.nanmean(lac_values)
        })

        ax = axes[i]
        ax.plot(sizes, lac_values, marker="o")
        ax.set_title(data["sample_id"], fontsize=9)
        ax.set_xlabel("Box size (pixels)")
        ax.set_ylabel("Lacunarity (Λ)")

    except Exception as e:
        ax = axes[i]
        ax.text(0.5, 0.5, f"Error:\n{os.path.basename(path)}", ha="center")
        ax.axis("off")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE (Lacunarity Curves for All Images)
# ------------------------------------------------------------

plt.savefig(os.path.join(EXPORT_DIR, "Module 4 - lacunarity_curves_for_all_images.png"), dpi=600, bbox_inches="tight")
# IMPORTANT: This saves the figure above. Change the filename ONLY if needed.

plt.show() 

# ============================================================
# Lacunarity Summary Table
# ============================================================
# This table summarizes lacunarity for each microglial image using:
#   - filename
#   - group / microglial activation state
#   - mean lacunarity across valid box sizes
#
# The table is ordered according to a user-defined biological progression (state_order).
# ============================================================

df_lac = pd.DataFrame(lac_rows)

# ------------------------------------------------------------
# Apply ordering
# ------------------------------------------------------------
df_lac["order_key"] = df_lac["cell_state"].apply(
    lambda x: state_order.index(x) if x in state_order else len(state_order)
)

df_lac = df_lac.sort_values(by=["order_key", "filename"]).drop(columns=["order_key"])

# ------------------------------------------------------------
# Final clean table
# ------------------------------------------------------------
df_lac = df_lac[["filename", "sample_id", "cell_state", "lacunarity_mean"]]

# ------------------------------------------------------------
# EXPORT TABLE (Lacunarity Summary Table)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_lac.to_csv(os.path.join(EXPORT_DIR, "Module 4 - lacunarity_summary_table.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_lac  

# ============================================================
# Plot of Mean Lacunarity Values
# ============================================================
plot_bars_or_boxplot(
    df=df_lac,
    value_col="lacunarity_mean",
    group_col="cell_state",
    filename_col="sample_id",
    state_order=state_order,
    ylabel="Mean Lacunarity (Λ)",
    xlabel="Image Filename / Group",
    title="Mean Lacunarity Across Groups",
    save_path=os.path.join(EXPORT_DIR, "Module 4 - bar_plot_of_mean_lacunarity_values.png")
)

print("\n[INFO] ===== Module 4 Completed =====")

# ============================================================
# GLCM Feature Extraction
# ============================================================
# This function computes GLCM-based texture metrics for a
# grayscale microglial image. Features are averaged across
# multiple directions and distances to ensure rotational
# invariance and robustness across activation states.
#
# Design principles:
#   - Universal: works with any microglial activation state
#   - Reproducible: fixed distances and angles
#   - Minimalistic: only essential features are returned
#   - Robust: handles empty or low-contrast images safely
#
# Researchers may adjust distances or angles depending on
# resolution or biological scale of interest.
# ============================================================

print("\n[INFO] ===== Module 5 - GLCM - Based Texture Metrics =====")

def compute_glcm_features(
    gray_img,
    mask=None,
    distances=[1, 2, 4],
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
    levels=32
):
    """
    Compute GLCM features on a cropped grayscale ROI.
    Optionally uses a binary mask to suppress background influence.
    """
    gray = gray_img.astype(np.float32)

    if gray.size == 0:
        return {
            "contrast": np.nan,
            "homogeneity": np.nan,
            "energy": np.nan,
            "correlation": np.nan
        }

    # Normalize to [0, 1]
    gray = gray - gray.min()
    if gray.max() > 0:
        gray = gray / gray.max()

    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool)
        if np.any(mask):
            vals = gray[mask]
            vmin, vmax = vals.min(), vals.max()

            if vmax > vmin:
                gray = (gray - vmin) / (vmax - vmin)
                gray = np.clip(gray, 0, 1)
            else:
                gray = np.zeros_like(gray)

            gray = gray * mask.astype(np.float32)
        else:
            return {
                "contrast": np.nan,
                "homogeneity": np.nan,
                "energy": np.nan,
                "correlation": np.nan
            }

    # Quantize
    gray_q = np.floor(gray * (levels - 1)).astype(np.uint8)

    if np.std(gray_q) == 0:
        return {
            "contrast": 0.0,
            "homogeneity": 1.0,
            "energy": 1.0,
            "correlation": 0.0
        }

    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    return {
        "contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "energy": float(np.mean(graycoprops(glcm, "energy"))),
        "correlation": float(np.mean(graycoprops(glcm, "correlation")))
    }

# ============================================================
# GLCM Summary Table
# ============================================================
# This block computes GLCM texture metrics for all microglial images and stores them in a summary table.
#
# Columns:
#   filename | cell_state | contrast | homogeneity | energy | correlation
# ============================================================

glcm_rows = []

for path, cell_state in zip(all_paths, all_cell_states):
    try:
        data = preprocessed_data[path]

        img_resized = data["img_resized"]
        binary_uint8 = data["binary_uint8"]
        bbox = data["bbox"]

        r0, c0, r1, c1 = bbox
        gray = img_resized[r0:r1, c0:c1]
        binary_roi = binary_uint8[r0:r1, c0:c1].astype(bool)

        feats = compute_glcm_features(gray, mask=binary_roi)

        glcm_rows.append({
            "filename": data["filename"],
            "sample_id": data["sample_id"],
            "cell_state": data["cell_state"],
            **feats
        })

    except Exception as e:
        glcm_rows.append({
            "filename": os.path.basename(path),
            "sample_id": path,
            "cell_state": cell_state,
            "contrast": None,
            "homogeneity": None,
            "energy": None,
            "correlation": None,
            "error": str(e)
        })

df_glcm = pd.DataFrame(glcm_rows)

# ------------------------------------------------------------
# Apply ordering
# ------------------------------------------------------------

df_glcm["order_key"] = df_glcm["cell_state"].apply(
    lambda x: state_order.index(x) if x in state_order else len(state_order)
)

df_glcm = df_glcm.sort_values(by=["order_key", "filename"]).drop(columns=["order_key"])

# ------------------------------------------------------------
# Final clean table
# ------------------------------------------------------------
df_glcm = df_glcm[[
    "filename",
    "sample_id",
    "cell_state",
    "contrast",
    "homogeneity",
    "energy",
    "correlation"
]]

# ------------------------------------------------------------
# EXPORT TABLE (GLCM Summary Table)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_glcm.to_csv(os.path.join(EXPORT_DIR, "Module 5 - glcm_summary_table.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_glcm  

# ============================================================
# Diagnostic Plots for GLCM Metrics
# ============================================================
# This block visualizes GLCM texture metrics across all microglial samples in the dataset.
#
# Each subplot shows:
#   - the metric values for each image
#   - filenames for identification
#
# These plots help reveal texture differences between
# microglial activation states:
#   - Resting microglia → higher contrast, lower homogeneity
#   - Activated / ameboid → high homogeneity, high energy
#   - Primed / resolving → intermediate profiles
#
# The goal is to provide a quick, interpretable overview of fine-scale intensity patterns across groups.
# ============================================================

metrics = ["contrast", "homogeneity", "energy", "correlation"]

df_diag = df_glcm.copy().replace({None: np.nan})

# Añadir sample_id único desde preprocessed_data
filename_to_sample = {}
for p in all_paths:
    d = preprocessed_data[p]
    filename_to_sample[(d["filename"], d["cell_state"])] = d["sample_id"]

df_diag["sample_id"] = [
    filename_to_sample.get((row["filename"], row["cell_state"]), row["filename"])
    for _, row in df_diag.iterrows()
]

# Orden consistente
df_diag["order_key"] = df_diag["cell_state"].apply(
    lambda x: state_order.index(x) if x in state_order else len(state_order)
)
df_diag = df_diag.sort_values(["order_key", "cell_state", "sample_id"]).reset_index(drop=True)
df_diag["xpos"] = np.arange(len(df_diag))

color_map = build_group_color_map(state_order)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    for state in state_order:
        sub = df_diag[df_diag["cell_state"] == state].dropna(subset=[metric])
        if sub.empty:
            continue

        ax.plot(
            sub["xpos"],
            sub[metric],
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=4.5,
            color=color_map[state],
            label=state,
            alpha=0.9
        )

    ax.set_xticks(df_diag["xpos"])
    ax.set_xticklabels(df_diag["sample_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} across samples")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

handles = [
    plt.Line2D([0], [0], color=color_map[state], marker="o", lw=2, label=state)
    for state in state_order if state in df_diag["cell_state"].unique()
]
fig.legend(handles=handles, title="Group", loc="upper right")

plt.tight_layout()
# ------------------------------------------------------------
# EXPORT FIGURE (Diagnostic Plots for GLCM Metrics)
# ------------------------------------------------------------
plt.savefig(os.path.join(EXPORT_DIR, "Module 5 - diagnostic_plots_for_glcm_metrics.png"), dpi=600, bbox_inches="tight")
# IMPORTANT: This saves the figure above. Change the filename ONLY if needed.

plt.show()

# ============================================================
# Ordered Bar Plots for GLCM Metrics
# ============================================================
# This version ensures full robustness by:
#   - replacing None values with NaN
#   - filtering invalid rows before plotting
#   - preserving microglial biological ordering
#   - keeping the visualization minimalistic and reproducible
#
# These plots help compare fine‑scale texture differences across:
#   - resting_microglia
#   - primed_microglia
#   - activated_microglia
#   - ameboid_microglia
#   - resolving_microglia
#
# GLCM metrics capture subtle intensity‑based signatures that complement fractal dimension and lacunarity.
# ============================================================

# Replace None with NaN for safe numeric plotting
df_plot = df_glcm.replace({None: np.nan})

# Metrics to visualize
metrics = ["contrast", "homogeneity", "energy", "correlation"]

# Deterministic color map per group
unique_states = df_plot["cell_state"].unique()
color_map = {
    state: plt.cm.tab10(i % 10)
    for i, state in enumerate(unique_states)
}

df_plot["color"] = df_plot["cell_state"].map(color_map)

metrics = ["contrast", "homogeneity", "energy", "correlation"]

for metric in metrics:
    plot_bars_or_boxplot(
        df=df_plot,
        value_col=metric,
        group_col="cell_state",
        filename_col="sample_id",
        state_order=state_order,
        ylabel=metric.capitalize(),
        xlabel="Image Filename / Group",
        title=f"{metric.capitalize()} Across Groups",
        save_path=os.path.join(EXPORT_DIR, f"Module 5 - ordered_bar_plot_glcm_{metric}.png")
    )

# ============================================================
# Final Figure Panel — FD, Lacunarity, and GLCM Metrics
# ============================================================
# This block generates a 2×2 panel summarizing the key morphometric descriptors for microglial morphology:
#
#   A — Fractal Dimension (ramification / complexity)
#   B — Mean Lacunarity (heterogeneity / gap distribution)
#   C — GLCM Contrast (fine‑scale intensity variation)
#   D — GLCM Homogeneity (texture smoothness / compactness)
#
# Design principles:
#   - Microglia‑specific: reflects activation‑dependent remodeling
#   - Reproducible: deterministic ordering and colors
#   - Minimalistic: only essential visual elements
#   - Publication‑ready: suitable for high‑impact journals
#
# This panel provides a compact, multidimensional summary of microglial structural organization across groups.
# ============================================================

def plot_on_axis_bars_or_boxplot(
    ax,
    df,
    value_col,
    group_col="cell_state",
    filename_col="sample_id",
    state_order=None,
    ylabel=None,
    title=None,
    image_threshold=5,
    group_threshold=5
):
    """
    Same logic as plot_bars_or_boxplot, but draws inside an existing axis.
    """
    df_plot = df.copy().dropna(subset=[value_col, group_col])

    if df_plot.empty:
        ax.set_title(title if title else value_col)
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return

    # Group ordering
    if state_order is None:
        groups = list(df_plot[group_col].dropna().unique())
    else:
        groups = [g for g in state_order if g in df_plot[group_col].dropna().unique()]
        groups += [g for g in df_plot[group_col].dropna().unique() if g not in groups]

    df_plot[group_col] = pd.Categorical(df_plot[group_col], categories=groups, ordered=True)

    sort_cols = [group_col]
    if filename_col in df_plot.columns:
        sort_cols.append(filename_col)

    df_plot = df_plot.sort_values(sort_cols).reset_index(drop=True)

    color_map = build_group_color_map(groups)
    use_box = should_use_boxplot(
        df_plot,
        group_col=group_col,
        image_threshold=image_threshold,
        group_threshold=group_threshold
    )

    # --------------------------------------------------------
    # CASE 1 — bar plot per image
    # --------------------------------------------------------
    if not use_box:
        df_plot["color"] = df_plot[group_col].astype(object).map(color_map)

        ax.bar(
            x=np.arange(len(df_plot)),
            height=df_plot[value_col],
            color=df_plot["color"],
            edgecolor="black",
            linewidth=0.7
        )

        ax.set_xticks(np.arange(len(df_plot)))
        if filename_col in df_plot.columns:
            ax.set_xticklabels(df_plot[filename_col], rotation=75, ha="right", fontsize=7)

        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # --------------------------------------------------------
    # CASE 2 — boxplot + individual points
    # --------------------------------------------------------
    else:
        light_palette = {g: lighten_color(color_map[g], 0.6) for g in groups}

        sns.boxplot(
            data=df_plot,
            x=group_col,
            y=value_col,
            hue=group_col,
            order=groups,
            palette=light_palette,
            dodge=False,
            width=0.55,
            showfliers=False,
            linewidth=1.2,
            ax=ax
        )

        sns.stripplot(
            data=df_plot,
            x=group_col,
            y=value_col,
            hue=group_col,
            order=groups,
            palette=color_map,
            dodge=False,
            jitter=0.12,
            alpha=0.9,
            size=4.5,
            edgecolor="black",
            linewidth=0.4,
            ax=ax
        )

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_title(title if title else value_col, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else value_col)

# ------------------------------------------------------------
# Prepare ordered dataframes
# ------------------------------------------------------------
df_fd_plot = df_fd.copy().dropna(subset=["fractal_dimension"])
df_lac_plot = df_lac.copy().dropna(subset=["lacunarity_mean"])
df_glcm_plot = df_glcm.copy().dropna(subset=["contrast", "homogeneity"])

# ------------------------------------------------------------
# Create the 2×2 panel
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# ------------------------------------------------------------
# Panel A — Fractal Dimension
# ------------------------------------------------------------
plot_on_axis_bars_or_boxplot(
    ax=axes[0],
    df=df_fd_plot,
    value_col="fractal_dimension",
    group_col="cell_state",
    filename_col="sample_id",
    state_order=state_order,
    ylabel="FD",
    title="A — Fractal Dimension"
)

# ------------------------------------------------------------
# Panel B — Mean Lacunarity
# ------------------------------------------------------------
plot_on_axis_bars_or_boxplot(
    ax=axes[1],
    df=df_lac_plot,
    value_col="lacunarity_mean",
    group_col="cell_state",
    filename_col="sample_id",
    state_order=state_order,
    ylabel="Λ",
    title="B — Mean Lacunarity"
)

# ------------------------------------------------------------
# Panel C — GLCM Contrast
# ------------------------------------------------------------
plot_on_axis_bars_or_boxplot(
    ax=axes[2],
    df=df_glcm_plot,
    value_col="contrast",
    group_col="cell_state",
    filename_col="sample_id",
    state_order=state_order,
    ylabel="Contrast",
    title="C — GLCM Contrast"
)

# ------------------------------------------------------------
# Panel D — GLCM Homogeneity
# ------------------------------------------------------------
plot_on_axis_bars_or_boxplot(
    ax=axes[3],
    df=df_glcm_plot,
    value_col="homogeneity",
    group_col="cell_state",
    filename_col="sample_id",
    state_order=state_order,
    ylabel="Homogeneity",
    title="D — GLCM Homogeneity"
)

# ------------------------------------------------------------
# Legend (shared across all panels)
# ------------------------------------------------------------
present_states = []
for df_tmp in [df_fd_plot, df_lac_plot, df_glcm_plot]:
    present_states.extend(df_tmp["cell_state"].dropna().unique().tolist())

unique_states = []
for s in state_order:
    if s in present_states and s not in unique_states:
        unique_states.append(s)

color_map = {state: plt.cm.tab10(i % 10) for i, state in enumerate(unique_states)}

handles = [
    plt.Line2D([0], [0], color=color_map[state], lw=8, label=state)
    for state in unique_states
]
fig.legend(handles=handles, title="Group", loc="upper right")

plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE
# ------------------------------------------------------------
plt.savefig(
    os.path.join(EXPORT_DIR, "Module 5 - final_figure_panel_fd_lacunarity_glcm_metrics.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print("\n[INFO] ===== Module 5 Completed =====")

# ============================================================
#  Multiscale Fractal Spectrum (MFS)
# ============================================================
# This version is robust, minimalistic, and works with any skeletonized microglial morphology.
#
# Microglial relevance:
#   - Resting microglia → broad multiscale complexity
#   - Primed microglia → reduced fine-scale complexity
#   - Activated microglia → flatter spectra (compact morphology)
#   - Ameboid microglia → minimal multiscale variation
#   - Resolving microglia → re-expansion of multiscale structure
#
# Researchers may adjust `scales` depending on image resolution.
# ============================================================

print("\n[INFO] ===== Module 6 - Multiscale Fractal Spectrum (MFS) =====")

def multiscale_fractal_spectrum(binary_img, scales=[2, 4, 8, 16, 32, 64, 128, 256]):
    """
    Classical simple multiscale fractal spectrum.
    """
    binary = binary_img.astype(bool)
    h, w = binary.shape

    spectrum = []
    valid_scales = []

    for s in scales:
        if s > h or s > w:
            continue

        H = h // s
        W = w // s

        Z = binary[:H * s, :W * s]
        Z = Z.reshape(H, s, W, s).max(axis=(1, 3))

        count = np.sum(Z)

        if count > 0:
            fd_s = np.log(count) / np.log(1 / s)
        else:
            fd_s = np.nan

        spectrum.append(fd_s)
        valid_scales.append(s)

    return spectrum, valid_scales

# ============================================================
# Multiscale Fractal Spectrum (MFS) Table
# ============================================================
# Computes the MFS for each microglial image using the skeleton produced by preprocess_image().
#  Skeletons ensure:
#   - non‑empty structure
#   - stable multiscale behavior
#   - reproducibility across activation states
#
# Output:
#   filename | cell_state | mfs_scale_2 | mfs_scale_4 | ...
#
# This table provides a scale‑resolved signature of microglial complexity, capturing differences between:
#   - resting_microglia (broad spectra)
#   - primed_microglia (reduced fine‑scale complexity)
#   - activated_microglia (flattened spectra)
#   - ameboid_microglia (minimal multiscale variation)
#   - resolving_microglia (re‑emerging multiscale structure)
# ============================================================

mfs_rows = []

for path, cell_state in zip(all_paths, all_cell_states):
    try:
        data = preprocessed_data[path]

        skel = data["skeleton_fd_uint8"].astype(bool)
        spectrum, scales = multiscale_fractal_spectrum(skel)

        mfs_rows.append({
            "filename": data["filename"],
            "cell_state": data["cell_state"],
            **{f"mfs_scale_{s}": spectrum[j] for j, s in enumerate(scales)}
        })

    except Exception as e:
        print("Error processing:", path, e)
        continue

df_mfs = pd.DataFrame(mfs_rows)

# ------------------------------------------------------------
# EXPORT TABLE (MFS Table)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_mfs.to_csv(os.path.join(EXPORT_DIR, "Module 6 - mfs_table.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_mfs 

# ============================================================
# MFS Curves per Image
# ============================================================
# This block visualizes the Multiscale Fractal Spectrum for each microglial image.
#
# Each subplot shows:
#   - the scales used (box sizes)
#   - the corresponding multiscale fractal descriptors
#   - the filename for identification
#
# Biological interpretation:
#   - Resting microglia → broad, high‑variance spectra
#   - Primed microglia → reduced fine‑scale complexity
#   - Activated microglia → flatter spectra (compact morphology)
#   - Ameboid microglia → minimal multiscale variation
#   - Resolving microglia → re‑emergence of multiscale structure
#
# These curves provide a scale-resolved view of microglial complexity that complements FD, lacunarity, and GLCM metrics.
# ============================================================

n = len(df_mfs)
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, row in df_mfs.iterrows():
    ax = axes[i]

    scales = [int(col.split("_")[-1]) for col in df_mfs.columns if col.startswith("mfs_scale_")]
    values = [row[f"mfs_scale_{s}"] for s in scales]

    ax.plot(scales, values, marker="o")
    ax.set_title(row["filename"], fontsize=9)
    ax.set_xlabel("Scale (box size)")
    ax.set_ylabel("MFS value")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(EXPORT_DIR, "Module 6 - mfs_curves_per_image.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Comparative Multiscale Fractal Spectrum by Group
# ============================================================
# This block compares the Multiscale Fractal Spectrum across groups.
#
# Interpretation:
#   - Resting microglia → broad, high‑variance spectra
#   - Primed microglia → reduced fine‑scale complexity
#   - Activated microglia → flatter spectra (compact morphology)
#   - Ameboid microglia → minimal multiscale variation
#   - Resolving microglia → re‑emergence of multiscale structure
#
# This visualization highlights how microglial complexity
# evolves across spatial scales and differs between states.
# ============================================================

# Melt table into long format
df_long = df_mfs.melt(
    id_vars=["filename", "cell_state"],
    var_name="scale_label",
    value_name="MFS_s"
)

# Extract numeric scale
df_long["scale"] = df_long["scale_label"].str.extract(r"(\d+)").astype(int)

# Compute mean per cell_state × scale
df_grouped = df_long.groupby(["cell_state", "scale"], as_index=False)["MFS_s"].mean()

# Deterministic colors per mgroup
unique_states = df_grouped["cell_state"].unique()
color_map = {state: plt.cm.tab10(i % 10) for i, state in enumerate(unique_states)}

plt.figure(figsize=(8, 6))

for state in unique_states:
    sub = df_grouped[df_grouped["cell_state"] == state]
    plt.plot(sub["scale"], sub["MFS_s"], marker="o", color=color_map[state], label=state)

plt.xlabel("Scale (box size)")
plt.ylabel("Mean MFS value")
plt.title("Multiscale Fractal Spectrum by Group")
plt.legend(title="Group")
plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE (Comparative MFS by group)
# ------------------------------------------------------------
plt.savefig(os.path.join(EXPORT_DIR, "Module 6 - comparative_mfs_by_group.png"), dpi=600, bbox_inches="tight")
# IMPORTANT: This saves the figure above. Change the filename ONLY if needed.

plt.show()

# ============================================================
# Heatmap of the Multiscale Fractal Spectrum
# ============================================================
# This heatmap provides a compact visualization of the MFS across all microglial images. 
# Each row corresponds to one image, and each column corresponds to a multiscale descriptor (FD_s at a given box size).
#
# Biological interpretation:
#   - Resting microglia → high values at fine scales, broad profile
#   - Primed microglia → reduced fine-scale complexity
#   - Activated microglia → flatter, lower MFS across scales
#   - Ameboid microglia → minimal multiscale variation
#   - Resolving microglia → re-expansion of multiscale structure
#
# Design principles:
#   - Microglia-specific but universally applicable
#   - Reproducible: deterministic ordering and color mapping
#   - Minimalistic: only essential information is shown
#   - Customizable: colormap, ordering, normalization
# ============================================================

# ------------------------------------------------------------
# Select only numeric MFS columns
# (all columns starting with "mfs_scale_")
# ------------------------------------------------------------
mfs_cols = [col for col in df_mfs.columns if col.startswith("mfs_scale_")]
numeric_data = df_mfs[mfs_cols].values

# ------------------------------------------------------------
# Create the heatmap
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.imshow(numeric_data, cmap="viridis", aspect="auto")
plt.colorbar(label="MFS value")

# Row labels: filenames
plt.yticks(
    ticks=np.arange(len(df_mfs)),
    labels=df_mfs["filename"],
    fontsize=7
)

# Column labels: scale descriptors
plt.xticks(
    ticks=np.arange(len(mfs_cols)),
    labels=mfs_cols,
    rotation=45,
    ha="right",
    fontsize=8
)

plt.title("MFS — Heatmap")
plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE (Heatmap of the MFS)
# ------------------------------------------------------------
plt.savefig(os.path.join(EXPORT_DIR, "Module 6 - heatmap_of_the_MFS.png"), dpi=600, bbox_inches="tight")
# IMPORTANT: This saves the figure above. Change the filename ONLY if needed.

plt.show()  

print("\n[INFO] ===== Module 6 Completed =====")

# ============================================================
# Graph-Based Morphometrics
# ============================================================
# This module extracts graph-theoretic descriptors from the skeleton produced in the preprocessing pipeline.
#
# Microglial relevance:
#   - Resting microglia → many endpoints, many bifurcations,
#     long diameters, high connectivity
#   - Primed microglia → reduced distal branching
#   - Activated microglia → simplified arbor, short diameter
#   - Ameboid microglia → minimal graph structure
#   - Resolving microglia → re-expansion of branching
#
# Design principles:
#   - Universal: works with any microglial activation state
#   - Reproducible: deterministic 8-connectivity rules
#   - Minimalistic: only essential morphometric descriptors
#   - Accessible: clear English annotations for researchers
#
# Researchers may extend this module by adding:
#   - subtree size distributions
#   - cycle detection
#   - geodesic path statistics
# ============================================================

print("\n[INFO] ===== Module 7 - Graph-Based Morphometrics =====")

# ------------------------------------------------------------
# 8-connectivity kernel for neighbor counting
# ------------------------------------------------------------
kernel = np.array([
    [1, 1, 1],
    [1,10, 1],
    [1, 1, 1]
])


def neighbor_count(skel):
    conv = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return conv - 10


def skeleton_to_graph(skel):
    G = nx.Graph()
    skel = skel.astype(bool)

    ys, xs = np.where(skel)
    nodes = list(zip(ys, xs))
    G.add_nodes_from(nodes)

    node_set = set(nodes)

    # Only 4 duplicates "to the front" to avoid duplicate edges
    forward_neighbors = [
        (-1,  1),
        ( 0,  1),
        ( 1,  1),
        ( 1,  0),
    ]

    for y, x in nodes:
        for dy, dx in forward_neighbors:
            nb = (y + dy, x + dx)
            if nb in node_set:
                G.add_edge((y, x), nb)

    return G


def fast_largest_component_diameter(G):
    """
    Fast approximate graph diameter using two BFS passes.

    For skeleton-like graphs this is much faster than nx.diameter().
    It avoids returning NaN when the largest component has >2000 nodes.
    """

    if G.number_of_nodes() == 0:
        return np.nan

    try:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()

        if subG.number_of_nodes() == 0:
            return np.nan

        if subG.number_of_nodes() == 1:
            return 0

        # First BFS: arbitrary node -> farthest node
        start = next(iter(subG.nodes))
        lengths = nx.single_source_shortest_path_length(subG, start)
        farthest = max(lengths, key=lengths.get)

        # Second BFS: farthest node -> farthest distance
        lengths = nx.single_source_shortest_path_length(subG, farthest)
        diameter_approx = max(lengths.values())

        return int(diameter_approx)

    except Exception:
        return np.nan


def graph_morphometrics(skel):
    skel = skel.astype(bool)

    if not np.any(skel):
        return dict(
            endpoints=0,
            bifurcations=0,
            total_nodes=0,
            total_edges=0,
            total_length=0,
            diameter=np.nan,
            avg_degree=np.nan,
            avg_junction_degree=np.nan,
            endpoint_density=np.nan,
            bifurcation_density=np.nan,
            branching_index=np.nan,
            largest_component_nodes=0,
            n_components=0
        )

    neigh = neighbor_count(skel)

    endpoints = int(np.sum(skel & (neigh == 1)))
    bifurcations = int(np.sum(skel & (neigh >= 3)))

    G = skeleton_to_graph(skel)

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    total_length = total_edges

    # Connected components
    try:
        components = list(nx.connected_components(G))
        n_components = len(components)
        largest_component_nodes = len(max(components, key=len)) if components else 0
    except Exception:
        n_components = np.nan
        largest_component_nodes = np.nan

    # Fast diameter instead of nx.diameter cutoff at 2000 nodes
    diameter = fast_largest_component_diameter(G)

    # Degree metrics
    if total_nodes > 0:
        degrees = np.array([deg for _, deg in G.degree()], dtype=float)

        # This will usually be close to 2 in skeleton graphs.
        avg_degree = float(np.mean(degrees))

        # More biologically useful: average degree only at branching nodes.
        junction_degrees = degrees[degrees >= 3]
        if len(junction_degrees) > 0:
            avg_junction_degree = float(np.mean(junction_degrees))
        else:
            avg_junction_degree = 0.0

        endpoint_density = endpoints / total_nodes
        bifurcation_density = bifurcations / total_nodes

        if endpoints > 0:
            branching_index = bifurcations / endpoints
        else:
            branching_index = np.nan

    else:
        avg_degree = np.nan
        avg_junction_degree = np.nan
        endpoint_density = np.nan
        bifurcation_density = np.nan
        branching_index = np.nan

    return dict(
        endpoints=endpoints,
        bifurcations=bifurcations,
        total_nodes=total_nodes,
        total_edges=total_edges,
        total_length=total_length,
        diameter=diameter,
        avg_degree=avg_degree,
        avg_junction_degree=avg_junction_degree,
        endpoint_density=endpoint_density,
        bifurcation_density=bifurcation_density,
        branching_index=branching_index,
        largest_component_nodes=largest_component_nodes,
        n_components=n_components
    )

# ============================================================
# Apply graph metrics to all group images
# ============================================================

graph_rows = []

for path, cell_state in zip(all_paths, all_cell_states):
    try:
        data = preprocessed_data[path]

        skel = data["skeleton_visual_uint8"].astype(bool)

        metrics = graph_morphometrics(skel)

        graph_rows.append({
            "filename": data["filename"],
            "cell_state": data["cell_state"],
            **metrics
        })

    except Exception as e:
        print("Error processing:", path, e)
        continue

df_graph = pd.DataFrame(graph_rows)

# ------------------------------------------------------------
# EXPORT TABLE (Graph-Based Morphometrics)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_graph.to_csv(os.path.join(EXPORT_DIR, "Module 7 - graph_based_morphometrics_table.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_graph

# ============================================================
# Graph Overlay on Skeletonized Microglial Structures
# ============================================================
# This block visualizes the graph representation of each microglial skeleton by overlaying graph nodes on top of the skeleton.
#
# Biological interpretation:
#   - Resting microglia → dense arbor, many nodes, rich topology
#   - Primed microglia → reduced distal branching
#   - Activated microglia → compact, simplified graph
#   - Ameboid microglia → minimal or near-zero graph structure
#   - Resolving microglia → re-expansion of graph connectivity
#
# Design principles:
#   - Microglia-specific but universally applicable
#   - Reproducible: deterministic skeleton + graph extraction
#   - Minimalistic: nodes only (edges optional for clarity)
# ============================================================

# Number of images
n = len(all_paths)

# Dynamic grid: 4 columns, as many rows as needed
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, (path, cell_state) in enumerate(zip(all_paths, all_cell_states)):

    try:
        data = preprocessed_data[path]

        skel = data["skeleton_visual_uint8"].astype(bool)
        G = skeleton_to_graph(skel)

        ax = axes[i]
        ax.imshow(skel, cmap="gray")
        ax.set_title(f"{data['filename']} ({data['cell_state']})", fontsize=9)
        ax.axis("off")

        ys = [node[0] for node in G.nodes()]
        xs = [node[1] for node in G.nodes()]

        ax.scatter(xs, ys, s=1, c="red")

    except Exception as e:
        print("Error processing:", path, e)
        continue

# Hide unused subplots (if any)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE (Graph Overlay on Skeletonized Microglial Structures)
# ------------------------------------------------------------
plt.savefig(os.path.join(EXPORT_DIR, "Module 7 - graph_overlay_on_skeletonized_microglial_structures.png"), dpi=600, bbox_inches="tight")
# IMPORTANT: This saves the figure above. Change the filename ONLY if needed.

plt.show()  

# ============================================================
# Build Master DataFrame (df_master_basic)
# ============================================================
# This merges all morphometric tables into a single unified
# dataframe indexed by filename + cell_state.
# ============================================================

# Start with fractal dimension table
df_master_basic = df_fd.copy()

# Merge lacunarity
df_master_basic = df_master_basic.merge(df_lac, on=["filename", "cell_state"], how="left")

# Merge GLCM metrics
df_master_basic = df_master_basic.merge(df_glcm, on=["filename", "cell_state"], how="left")

# Merge Multiscale Fractal Spectrum
df_master_basic = df_master_basic.merge(df_mfs, on=["filename", "cell_state"], how="left")

# Merge graph-based morphometrics
df_master_basic = df_master_basic.merge(df_graph, on=["filename", "cell_state"], how="left")

# ------------------------------------------------------------
# EXPORT TABLE (Build Master DataFrame — df_master_basic)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_master_basic.to_csv(os.path.join(EXPORT_DIR, "Module 7 - build_master_dataframe_df_master_basic.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_master_basic 

# Use the master dataframe for graph-based barplots
df_graph_diag = df_master_basic.copy()

# Ensure a valid cell_id exists
if "cell_id" not in df_graph_diag.columns:
    if "filename" in df_graph_diag.columns:
        df_graph_diag["cell_id"] = df_graph_diag["filename"]
    else:
        df_graph_diag["cell_id"] = df_graph_diag.index.astype(str)

# ============================================================
# Comparative Plots of Graph Morphometrics
# ============================================================

metrics = [
    "endpoints",
    "bifurcations",
    "endpoint_density",
    "bifurcation_density",
    "branching_index",
    "avg_junction_degree",
    "diameter",
    "total_nodes",
    "total_length"
]

# Ensure consistent naming
if "stage" in df_graph_diag.columns and "cell_state" not in df_graph_diag.columns:
    df_graph_diag["cell_state"] = df_graph_diag["stage"]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    plot_on_axis_bars_or_boxplot(
        ax=axes[i],
        df=df_graph_diag,
        value_col=metric,
        group_col="cell_state",
        filename_col="cell_id",
        state_order=state_order,
        ylabel="Value",
        title=metric.replace("_", " ").capitalize()
    )

# Hide unused subplots
for j in range(len(metrics), len(axes)):
    axes[j].axis("off")

plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE (Comparative Plots of Graph Morphometrics)
# ------------------------------------------------------------
plt.savefig(
    os.path.join(EXPORT_DIR, "Module 7 - comparative_plots_of_graph_morphometrics_microglia.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# Use the master dataframe for group-level graph morphometrics
df_graph_diag = df_master_basic.copy()

# ============================================================
# Group-Level Comparison of Graph Morphometrics
# Universal version: works with individual images or groups
# ============================================================

metrics = [
    "endpoints",
    "bifurcations",
    "total_nodes",
    "total_edges",
    "total_length",
    "diameter",
    "avg_degree"
]

# Ensure consistent naming
if "stage" in df_graph_diag.columns and "cell_state" not in df_graph_diag.columns:
    df_graph_diag["cell_state"] = df_graph_diag["stage"]

# Keep only metrics that exist in the dataframe
metrics_existing = [m for m in metrics if m in df_graph_diag.columns]

# Remove metrics that are completely empty
metrics_existing = [
    m for m in metrics_existing
    if not df_graph_diag[m].dropna().empty
]

# Identify groups in the same order as the dataset loader
groups = [g for g in state_order if g in df_graph_diag["cell_state"].unique()]
groups += [g for g in df_graph_diag["cell_state"].unique() if g not in groups]

# Deterministic color map per group
color_map = {state: plt.cm.tab10(i % 10) for i, state in enumerate(groups)}

plt.figure(figsize=(12, 6))

for state in groups:
    subset = df_graph_diag[df_graph_diag["cell_state"] == state]

    if subset.empty:
        continue

    mean_vals = subset[metrics_existing].mean(numeric_only=True)

    plt.plot(
        metrics_existing,
        mean_vals[metrics_existing],
        marker="o",
        label=state,
        color=color_map[state]
    )

# Decide automatically whether log scale is useful
all_values = df_graph_diag[metrics_existing].values.flatten()
all_values = all_values[np.isfinite(all_values)]
all_values = all_values[all_values > 0]

if len(all_values) > 0:
    value_ratio = all_values.max() / all_values.min()

    if value_ratio > 100:
        plt.yscale("log")
        ylabel = "Mean value (log scale)"
    else:
        ylabel = "Mean value"
else:
    ylabel = "Mean value"

plt.xticks(rotation=45, ha="right")
plt.ylabel(ylabel)
plt.title("Graph Morphometrics — Group-Level Comparison")
plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(EXPORT_DIR, "Module 7 - group_level_comparison_of_graph_morphometrics.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print("\n[INFO] ===== Module 7 Completed =====")

# ============================================================
# Skeleton Centroid
# ============================================================
# Computes the centroid of a skeletonized morphology.
# Used as the universal anchor for Sholl analysis.
# ============================================================

print("\n[INFO] ===== Module 8 - Batch Sholl Analysis =====")

def skeleton_centroid(skel):
    skel = skel.astype(bool)

    ys, xs = np.where(skel)

    if len(xs) == 0:
        return None, None

    cx = xs.mean()
    cy = ys.mean()

    return cx, cy

# ============================================================
# Sholl Analysis
# ============================================================

def sholl_analysis(skel, step=10, max_radius=None):
    """
    Perform Sholl analysis on a skeletonized microglial morphology.
    """
    skel = skel.astype(bool)
    h, w = skel.shape

    cx, cy = skeleton_centroid(skel)

    if cx is None or cy is None:
        return {
            "radii": np.array([]),
            "intersections": np.array([]),
            "max_intersections": 0,
            "radius_of_max": np.nan,
            "auc": 0.0,
            "decay_rate": np.nan,
            "center_x": np.nan,
            "center_y": np.nan
        }

    ys, xs = np.where(skel)

    if max_radius is None:
        distances_to_skeleton = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        max_radius = int(np.ceil(distances_to_skeleton.max()))

    radii = np.arange(step, max_radius, step)
    intersections = []

    yy, xx = np.indices(skel.shape)

    for r in radii:
        circle = np.abs(np.sqrt((xx - cx)**2 + (yy - cy)**2) - r) < 0.5
        intersec = np.logical_and(circle, skel).sum()
        intersections.append(intersec)

    intersections = np.array(intersections)

    if len(intersections) == 0:
        return {
            "radii": radii,
            "intersections": intersections,
            "max_intersections": 0,
            "radius_of_max": np.nan,
            "auc": 0.0,
            "decay_rate": np.nan,
            "center_x": cx,
            "center_y": cy
        }

    max_intersections = intersections.max()
    radius_of_max = radii[intersections.argmax()]
    auc = float(np.sum((intersections[1:] + intersections[:-1]) * np.diff(radii) / 2))

    if len(radii) >= 2:
        coeffs = np.polyfit(radii, intersections, 1)
        decay_rate = coeffs[0]
    else:
        decay_rate = np.nan

    return dict(
        radii=radii,
        intersections=intersections,
        max_intersections=max_intersections,
        radius_of_max=radius_of_max,
        auc=auc,
        decay_rate=decay_rate,
        center_x=cx,
        center_y=cy
    )

# ============================================================
# Batch Sholl Analysis Across All Microglial Samples
# ============================================================

sholl_rows = []

for path, cell_state in zip(all_paths, all_cell_states):
    try:
        data = preprocessed_data[path]

        skel = data["skeleton_visual_uint8"].astype(bool)

        sh = sholl_analysis(skel)

        sholl_rows.append({
            "filename": data["filename"],
            "sample_id": data["sample_id"],
            "cell_state": data["cell_state"],
            "max_intersections": sh["max_intersections"],
            "radius_of_max": sh["radius_of_max"],
            "auc": sh["auc"],
            "decay_rate": sh["decay_rate"],
            "radii": sh["radii"],
            "intersections": sh["intersections"]
        })

    except Exception as e:
        print("Error processing:", path, e)
        continue

df_sholl = pd.DataFrame(sholl_rows)

# ============================================================
# Sholl Wide Table (one column per radius)
# Required for Mixed ANOVA in Module 9
# ============================================================

# Collect all radii found across all cells
all_sholl_radii = sorted({
    int(r)
    for radii in df_sholl["radii"]
    for r in radii
})

sholl_wide_rows = []

for _, row in df_sholl.iterrows():
    entry = {
        "filename": row["filename"],
        "cell_state": row["cell_state"]
    }

    # Create a dictionary radius -> intersections
    radius_to_value = {
        int(r): float(v)
        for r, v in zip(row["radii"], row["intersections"])
    }

    # Use the same radius columns for every cell
    # Missing radii are filled with 0 intersections.
    for r in all_sholl_radii:
        entry[f"sholl_{r}"] = radius_to_value.get(r, 0.0)

    sholl_wide_rows.append(entry)

df_sholl_wide = pd.DataFrame(sholl_wide_rows)

# Uses dot decimal and comma separator.
df_sholl_wide.to_csv(
    os.path.join(EXPORT_DIR, "Module 8 - sholl_wide_table.csv"),
    index=False
)

df_sholl_wide

# ------------------------------------------------------------
# EXPORT TABLE (Batch Sholl Analysis Across All Microglial Samples)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_sholl.to_csv(os.path.join(EXPORT_DIR, "Module 8 - batch_sholl_analysis_across_all_microglial_samples.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_sholl

# ============================================================
# Sholl Summary Table
# ============================================================

df_sholl_summary = df_sholl.drop(columns=["radii", "intersections"])

# ------------------------------------------------------------
# EXPORT TABLE (Sholl Summary Table)
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_sholl_summary.to_csv(os.path.join(EXPORT_DIR, "Module 8 - sholl_summary_table.csv"), index=False)
# IMPORTANT: This saves the table above. Change the filename ONLY if needed.

df_sholl_summary

metrics = ["max_intersections", "radius_of_max", "auc", "decay_rate"]

for metric in metrics:
    plot_bars_or_boxplot(
        df=df_sholl_summary,
        value_col=metric,
        group_col="cell_state",
        filename_col="sample_id",
        state_order=state_order,
        ylabel=metric.replace("_", " ").capitalize(),
        xlabel="Image Filename / Group",
        title=f"Sholl Metric — {metric.replace('_', ' ').capitalize()}",
        save_path=os.path.join(EXPORT_DIR, f"Module 8 - sholl_{metric}.png")
    )

# ============================================================
# Sholl Overlay for All Microglial Samples
# ============================================================

# Number of samples
n = len(all_paths)

# Dynamic grid layout
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, (path, cell_state) in enumerate(zip(all_paths, all_cell_states)):

    try:
        data = preprocessed_data[path]

        skel = data["skeleton_visual_uint8"].astype(bool)

        sh = sholl_analysis(skel)

        radii = sh["radii"]

        cx, cy = sh["center_x"], sh["center_y"]

        if np.isnan(cx) or np.isnan(cy):
            continue

        yy, xx = np.indices(skel.shape)
        intersections_mask = np.zeros_like(skel, dtype=bool)

        radii_to_plot = radii[::5]

        for r in radii_to_plot:
            circle = np.abs(np.sqrt((xx - cx)**2 + (yy - cy)**2) - r) < 0.5
            intersections_mask |= np.logical_and(circle, skel)

        ax = axes[i]
        ax.imshow(skel, cmap="gray")

        # Zoom around the skeleton so the cell is visible
        ys_skel, xs_skel = np.where(skel)

        if len(xs_skel) > 0:
            margin = 80

            x_min = max(xs_skel.min() - margin, 0)
            x_max = min(xs_skel.max() + margin, skel.shape[1])

            y_min = max(ys_skel.min() - margin, 0)
            y_max = min(ys_skel.max() + margin, skel.shape[0])

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)

        ax.scatter([cx], [cy], c="yellow", s=40)

        theta = np.linspace(0, 2 * np.pi, 400)
        for r in radii_to_plot:
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            ax.plot(x, y, color="cyan", alpha=0.15, linewidth=0.6)

        ys, xs = np.where(intersections_mask)
        ax.scatter(xs, ys, c="red", s=8)

        ax.set_title(f"{data['filename']} ({data['cell_state']})", fontsize=9)
        ax.axis("off")

    except Exception as e:
        print("Error processing:", path, e)
        continue

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()

# ------------------------------------------------------------
# EXPORT FIGURE (Sholl Overlay for All Microglial Samples)
# ------------------------------------------------------------
plt.savefig(os.path.join(EXPORT_DIR, "Module 8 - sholl_overlay_for_all_microglial_samples.png"), dpi=600, bbox_inches="tight")
# IMPORTANT: This saves the figure above. Change the filename ONLY if needed.

plt.show() 

# ============================================================
# Build Master Morphometric DataFrame
# ============================================================

# ============================================================
# Pandas display settings
# ============================================================
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# ============================================================
# Identify all feature dataframes available in the workspace
# ============================================================

feature_dfs = []

feature_candidates = [
    ("df_sholl_summary", globals().get("df_sholl_summary")),
    ("df_sholl_wide", globals().get("df_sholl_wide")),
    ("df_fd", globals().get("df_fd")),
    ("df_lac", globals().get("df_lac")),
    ("df_glcm", globals().get("df_glcm")),
    ("df_mfs", globals().get("df_mfs")),
    ("df_graph", globals().get("df_graph")),
]

for name, df in feature_candidates:
    if df is not None and not df.empty:
        feature_dfs.append(df.copy())

if len(feature_dfs) == 0:
    raise ValueError("No feature dataframes found. Please run Modules 1–8 first.")

# ============================================================
# Utility: detect ID and state columns
# ============================================================

def detect_id_column(df):
    # Prefer filename because it is shared by all feature tables.
    # sample_id is useful for display, but causes duplicate columns during merges.
    candidates = ["cell_id", "filename", "image_id", "image_name", "id", "sample_id"]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError("No ID column found in dataframe.")


def detect_state_column(df):
    candidates = ["cell_state", "state", "group", "label", "phenotype"]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError("No cell_state column found in dataframe.")

# ============================================================
# Standardize column names and remove auxiliary duplicate IDs
# ============================================================

clean_dfs = []
sample_id_maps = []

for df in feature_dfs:
    df = df.copy()

    id_col = detect_id_column(df)
    state_col = detect_state_column(df)

    df.rename(columns={id_col: "cell_id", state_col: "cell_state"}, inplace=True)

    # Save sample_id once, only for final display
    if "sample_id" in df.columns:
        tmp_sample = df[["cell_id", "cell_state", "sample_id"]].copy()
        tmp_sample = tmp_sample.dropna(subset=["cell_id", "cell_state"])
        tmp_sample = tmp_sample.drop_duplicates(subset=["cell_id", "cell_state"])
        sample_id_maps.append(tmp_sample)

    # Remove auxiliary ID/display columns before merging
    # They cause sample_id_x / sample_id_y conflicts.
    auxiliary_id_cols = ["sample_id", "filename", "image_id", "image_name", "id"]

    for col in auxiliary_id_cols:
        if col in df.columns and col not in ["cell_id", "cell_state"]:
            df.drop(columns=[col], inplace=True)

    # Remove duplicated columns inside the same dataframe
    df = df.loc[:, ~df.columns.duplicated()]

    clean_dfs.append(df)

# ============================================================
# Merge all feature tables on cell_id + cell_state
# ============================================================

df_master = clean_dfs[0]

for df in clean_dfs[1:]:

    # Safety: remove repeated non-key columns before merging
    overlapping_cols = [
        c for c in df.columns
        if c in df_master.columns and c not in ["cell_id", "cell_state"]
    ]

    if overlapping_cols:
        print(f"[Module 8] Dropping duplicated columns before merge: {overlapping_cols}")
        df = df.drop(columns=overlapping_cols)

    df_master = df_master.merge(
        df,
        on=["cell_id", "cell_state"],
        how="outer"
    )

# ============================================================
# Add one clean sample_id column back, if available
# ============================================================

if len(sample_id_maps) > 0:
    df_sample_id = pd.concat(sample_id_maps, ignore_index=True)
    df_sample_id = df_sample_id.drop_duplicates(subset=["cell_id", "cell_state"])

    df_master = df_master.merge(
        df_sample_id,
        on=["cell_id", "cell_state"],
        how="left"
    )

# Remove duplicate columns as final safety
df_master = df_master.loc[:, ~df_master.columns.duplicated()]

# Sort columns
priority_cols = ["cell_id", "sample_id", "cell_state"]

cols = [c for c in priority_cols if c in df_master.columns] + \
       [c for c in df_master.columns if c not in priority_cols]

df_master = df_master[cols]

# ============================================================
# EXPORT
# ============================================================

# Uses dot decimal and comma separator.
df_master.to_csv(os.path.join(EXPORT_DIR, "Module 8 - build_master_morphometric_dataframe_universal.csv"), index=False)

df_master

print("\n[INFO] ===== Module 8 Completed =====")

# ============================================================
# Module 9 — Statistical Testing
# ============================================================
# This module performs:
#   - Between-group ANOVA / Welch / Kruskal–Wallis
#   - Mixed ANOVA (between × within) for Sholl metrics
#   - Mauchly’s test of sphericity
#   - Greenhouse–Geisser / Huynh–Feldt corrections
#
# SPECIALIZED FOR MICROGLIA:
#   - Uses cell_state as grouping factor
#   - Optimized for small datasets
#   - Compatible with df_master (universal feature table)
# ============================================================
# BETWEEN-GROUP PIPELINE
#   Normality:
#       - n < 3     -> insufficient_n -> treated as non-parametric
#       - 3 <= n < 5000  -> Shapiro-Wilk
#       - n >= 5000      -> Lilliefors
#       - zero variance  -> zero_variance -> treated as non-parametric
#
#   If normal:
#       - Levene for variance homogeneity
#       - 2 groups:
#           * equal variances    -> Student t-test
#           * unequal variances  -> Welch t-test
#       - >2 groups:
#           * equal variances    -> ANOVA
#           * unequal variances  -> Welch ANOVA
#
#   If non-normal:
#       - 2 groups  -> Mann–Whitney U
#       - >2 groups -> Kruskal–Wallis
#
# EFFECT SIZES
#   - Student/Welch t-test -> Cohen's d
#   - Mann–Whitney U       -> rank-biserial correlation (RBC)
#   - ANOVA                -> eta squared
#   - Welch ANOVA          -> np2 if available from pingouin
#   - Kruskal–Wallis       -> epsilon squared
#
# MULTIPLE TESTING
#   - Global p-values across metrics -> FDR Benjamini-Hochberg
#
# REPEATED MEASURES
#   - Sholl metrics handled separately with mixed ANOVA
# ============================================================

print("\n[INFO] ===== Module 9 - Statistical Testing =====")

warnings.filterwarnings("default")

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

if "df_master" not in globals():
    raise NameError("df_master not found. Please run the Master Builder module first.")

df_stats = df_master.copy()

GROUP_COL = "cell_state"
if GROUP_COL not in df_stats.columns:
    raise KeyError("cell_state column not found in df_master.")

if "cell_id" not in df_stats.columns:
    raise KeyError("cell_id column not found in df_master.")

# ------------------------------------------------------------
# Numeric metrics
# ------------------------------------------------------------
numeric_cols = [
    c for c in df_stats.select_dtypes(include=[np.number]).columns
    if c not in ["cell_id"]
]

# Repeated-measures patterns excluded from between-group tests
rm_patterns = ["sholl_", "mfs_scale_"]

def is_repeated_measures_metric(metric):
    return any(p in metric.lower() for p in rm_patterns)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clean_group_data(df, metric, group_col):
    """
    Returns:
        valid_groups: list of group labels with at least 1 non-NA value
        data_by_group: dict[group] -> 1D numpy array
        n_groups: number of valid groups
        n_total: total non-missing observations
    """
    sub = df[[group_col, metric]].dropna().copy()
    valid_groups = []
    data_by_group = {}

    for g in sub[group_col].dropna().unique():
        vals = sub.loc[sub[group_col] == g, metric].dropna().values.astype(float)
        if len(vals) > 0:
            valid_groups.append(g)
            data_by_group[g] = vals

    return valid_groups, data_by_group, len(valid_groups), len(sub)

def group_normality_test(x):
    """
    Normality per group:
      - n < 3: cannot reliably test
      - zero variance: not evaluable
      - 3 <= n < 5000: Shapiro-Wilk
      - n >= 5000: Lilliefors
    """
    n = len(x)

    if n < 3:
        return np.nan, "insufficient_n"

    if np.nanstd(x) == 0:
        return np.nan, "zero_variance"

    if n < 5000:
        stat, p = shapiro(x)
        return p, "Shapiro-Wilk"
    else:
        stat, p = lilliefors(x, dist="norm")
        return p, "Lilliefors"

def evaluate_normality(data_by_group, alpha=0.05):
    """
    A metric is considered normal only if ALL groups:
      - have testable sample size (>= 3)
      - do not have zero variance
      - pass the relevant normality test
    """
    pvals = {}
    methods = {}
    normal = True

    for g, x in data_by_group.items():
        p, method = group_normality_test(x)
        pvals[g] = p
        methods[g] = method

        if np.isnan(p):
            normal = False
        elif p <= alpha:
            normal = False

    return normal, pvals, methods

def safe_eta_squared_from_anova(F_value, k, n_total):
    """
    eta^2 from one-way ANOVA F statistic
    """
    try:
        df_between = k - 1
        df_within = n_total - k
        if df_within <= 0:
            return np.nan
        return (F_value * df_between) / ((F_value * df_between) + df_within)
    except Exception:
        return np.nan

def safe_epsilon_squared_from_kruskal(H_value, k, n_total):
    """
    epsilon^2 for Kruskal-Wallis
    """
    try:
        denom = n_total - k
        if denom <= 0:
            return np.nan
        return (H_value - k + 1) / denom
    except Exception:
        return np.nan

# ------------------------------------------------------------
# 1. BETWEEN-GROUP TESTS
# ------------------------------------------------------------
results = []

for metric in numeric_cols:

    if is_repeated_measures_metric(metric):
        continue

    valid_groups, data_by_group, k, n_total = clean_group_data(df_stats, metric, GROUP_COL)

    # Need at least 2 groups with data
    if k < 2:
        warnings.warn(f"[{metric}] skipped: fewer than 2 groups with data.")
        continue

    # Avoid zero-variance total metrics that cannot be tested meaningfully
    try:
        all_values = np.concatenate([data_by_group[g] for g in valid_groups])
    except Exception as e:
        warnings.warn(f"[{metric}] skipped: could not concatenate values ({e}).")
        continue

    if len(all_values) == 0 or np.nanstd(all_values) == 0:
        warnings.warn(f"[{metric}] skipped: zero global variance or no valid values.")
        continue

    # ----------------------------
    # Normality
    # ----------------------------
    normal, norm_pvals, norm_methods = evaluate_normality(data_by_group, alpha=0.05)

    normality_method_summary = "; ".join(
        [f"{g}:{norm_methods[g]}" for g in valid_groups]
    )
    normality_p_summary = "; ".join(
        [f"{g}:{'nan' if pd.isna(norm_pvals[g]) else round(norm_pvals[g], 6)}" for g in valid_groups]
    )

    # ----------------------------
    # If normal -> Levene
    # ----------------------------
    lev_p = np.nan
    equal_var = np.nan

    if normal:
        try:
            lev_stat, lev_p = levene(*[data_by_group[g] for g in valid_groups], center="median")
            equal_var = bool(lev_p > 0.05)
        except Exception as e:
            warnings.warn(f"[{metric}] Levene test failed: {e}")
            lev_p = np.nan
            equal_var = False

    # ----------------------------
    # Choose test based on k
    # ----------------------------
    test_name = None
    stat = np.nan
    p = np.nan
    effect_size = np.nan
    effect_size_name = np.nan

    try:
        # ====================================================
        # TWO GROUPS
        # ====================================================
        if k == 2:
            g1, g2 = valid_groups
            x1 = data_by_group[g1]
            x2 = data_by_group[g2]

            if normal:
                if equal_var:
                    test_name = "Student t-test"
                    stat, p = ttest_ind(x1, x2, equal_var=True, nan_policy="omit")
                else:
                    test_name = "Welch t-test"
                    stat, p = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")

                try:
                    effect_size = pg.compute_effsize(x1, x2, eftype="cohen")
                    effect_size_name = "cohen_d"
                except Exception as e:
                    warnings.warn(f"[{metric}] effect size computation failed: {e}")
                    effect_size = np.nan
                    effect_size_name = np.nan

            else:
                test_name = "Mann–Whitney U"
                stat, p = mannwhitneyu(x1, x2, alternative="two-sided")

                try:
                    mw = pg.mwu(x1, x2, alternative="two-sided")
                    if "RBC" in mw.columns:
                        effect_size = mw["RBC"].iloc[0]
                        effect_size_name = "rank_biserial_correlation"
                    else:
                        warnings.warn(
                            f"[{metric}] Mann–Whitney effect size column RBC not found. "
                            f"Available columns: {list(mw.columns)}"
                        )
                        effect_size = np.nan
                        effect_size_name = np.nan
                except Exception as e:
                    warnings.warn(f"[{metric}] Mann–Whitney effect size failed: {e}")
                    effect_size = np.nan
                    effect_size_name = np.nan

        # ====================================================
        # MORE THAN TWO GROUPS
        # ====================================================
        elif k > 2:
            arrays = [data_by_group[g] for g in valid_groups]

            if normal:
                if equal_var:
                    test_name = "ANOVA"
                    stat, p = f_oneway(*arrays)

                    try:
                        effect_size = safe_eta_squared_from_anova(stat, k, n_total)
                        effect_size_name = "eta_squared"
                    except Exception as e:
                        warnings.warn(f"[{metric}] ANOVA effect size failed: {e}")
                        effect_size = np.nan
                        effect_size_name = np.nan

                else:
                    test_name = "Welch ANOVA"
                    tmp = df_stats[[GROUP_COL, metric]].dropna().copy()
                    welch = pg.welch_anova(dv=metric, between=GROUP_COL, data=tmp)

                    # Normalize column names across pingouin versions
                    welch_cols_map = {c: c.lower().replace("-", "_") for c in welch.columns}

                    # Statistic
                    stat_found = False
                    for original_col, norm_col in welch_cols_map.items():
                        if norm_col in ["f", "f_val"]:
                            stat = welch[original_col].iloc[0]
                            stat_found = True
                            break

                    if not stat_found:
                        warnings.warn(
                            f"[{metric}] Welch ANOVA statistic extraction failed. "
                            f"Available columns: {list(welch.columns)}"
                        )

                    # p-value
                    p_found = False
                    for original_col, norm_col in welch_cols_map.items():
                        if norm_col in ["p_unc", "p", "pval", "p_value"]:
                            p = welch[original_col].iloc[0]
                            p_found = True
                            break

                    if not p_found or pd.isna(p):
                        warnings.warn(
                            f"[{metric}] Welch ANOVA p-value extraction failed. "
                            f"Available columns: {list(welch.columns)}"
                        )

                    # effect size
                    es_found = False
                    for original_col, norm_col in welch_cols_map.items():
                        if norm_col in ["np2", "eta2", "eta_squared"]:
                            effect_size = welch[original_col].iloc[0]
                            effect_size_name = norm_col
                            es_found = True
                            break

                    if not es_found:
                        warnings.warn(
                            f"[{metric}] Welch ANOVA effect size not found. "
                            f"Available columns: {list(welch.columns)}"
                        )
                        effect_size = np.nan
                        effect_size_name = np.nan

            else:
                test_name = "Kruskal–Wallis"
                stat, p = kruskal(*arrays)

                try:
                    effect_size = safe_epsilon_squared_from_kruskal(stat, k, n_total)
                    effect_size_name = "epsilon_squared"
                except Exception as e:
                    warnings.warn(f"[{metric}] Kruskal effect size failed: {e}")
                    effect_size = np.nan
                    effect_size_name = np.nan

        # General warning if any test ran but p is still NaN
        if test_name is not None and pd.isna(p):
            warnings.warn(
                f"[{metric}] {test_name} returned p_value = NaN. "
                f"Check input data, group sizes, zero variance within groups, or library output format."
            )

    except Exception as e:
        test_name = "FAILED"
        stat = np.nan
        p = np.nan
        effect_size = np.nan
        effect_size_name = np.nan
        warnings.warn(f"[{metric}] test failed: {e}")

    results.append({
        "metric": metric,
        "n_groups": k,
        "n_total": n_total,
        "groups": " | ".join(map(str, valid_groups)),
        "normal": normal,
        "normality_method": normality_method_summary,
        "normality_p_by_group": normality_p_summary,
        "levene_p": lev_p,
        "equal_variance": equal_var,
        "test": test_name,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect_size,
        "effect_size_name": effect_size_name
    })

# ------------------------------------------------------------
# 2. MIXED ANOVA FOR SHOLL
# ------------------------------------------------------------
sholl_cols = [c for c in numeric_cols if "sholl_" in c.lower()]

if len(sholl_cols) >= 3:
    try:
        df_sholl_rm = df_stats[["cell_id", GROUP_COL] + sholl_cols].copy()

        df_long = df_sholl_rm.melt(
            id_vars=["cell_id", GROUP_COL],
            value_vars=sholl_cols,
            var_name="radius",
            value_name="value"
        ).dropna()

        df_long["radius_num"] = df_long["radius"].str.extract(r"(\d+)").astype(float)

        mix = pg.mixed_anova(
            dv="value",
            within="radius_num",
            between=GROUP_COL,
            subject="cell_id",
            data=df_long
        )

        if "Source" in mix.columns:
            interaction_rows = mix[mix["Source"].astype(str).str.contains(r"\*", regex=True, na=False)]
            if not interaction_rows.empty:
                mix_row = interaction_rows.iloc[0]
            else:
                mix_row = mix.iloc[0]
        else:
            mix_row = mix.iloc[0]

        try:
            mauchly = pg.sphericity(df_long, dv="value", subject="cell_id", within="radius_num")
            sphericity_p = mauchly[1] if isinstance(mauchly, (tuple, list)) and len(mauchly) > 1 else np.nan
        except Exception as e:
            warnings.warn(f"[Sholl] Mauchly sphericity test failed: {e}")
            sphericity_p = np.nan

        try:
            eps_raw = pg.epsilon(df_long, dv="value", subject="cell_id", within="radius_num")
            gg = np.nan
            hf = np.nan

            if isinstance(eps_raw, dict):
                gg = eps_raw.get("gg", np.nan)
                hf = eps_raw.get("hf", np.nan)
            elif isinstance(eps_raw, (list, tuple)) and len(eps_raw) >= 2:
                gg = eps_raw[0]
                hf = eps_raw[1]
        except Exception as e:
            warnings.warn(f"[Sholl] epsilon calculation failed: {e}")
            gg = np.nan
            hf = np.nan

        p_mix = (
            mix_row["p-unc"] if "p-unc" in mix_row.index else
            mix_row["p_unc"] if "p_unc" in mix_row.index else
            mix_row["p"] if "p" in mix_row.index else
            mix_row["pval"] if "pval" in mix_row.index else
            np.nan
        )

        np2_mix = mix_row["np2"] if "np2" in mix_row.index else np.nan
        F_mix = mix_row["F"] if "F" in mix_row.index else np.nan

        if pd.isna(p_mix):
            warnings.warn("[Sholl] Mixed ANOVA returned p_value = NaN.")

        results.append({
            "metric": "Sholl",
            "n_groups": df_long[GROUP_COL].nunique(),
            "n_total": len(df_long),
            "groups": " | ".join(map(str, df_long[GROUP_COL].dropna().unique())),
            "normal": np.nan,
            "normality_method": "repeated-measures module",
            "normality_p_by_group": np.nan,
            "levene_p": np.nan,
            "equal_variance": np.nan,
            "test": "Mixed ANOVA (between × within)",
            "statistic": F_mix,
            "p_value": p_mix,
            "effect_size": np2_mix,
            "effect_size_name": "np2",
            "sphericity_p": sphericity_p,
            "GG_epsilon": gg,
            "HF_epsilon": hf
        })

    except Exception as e:
        warnings.warn(f"[Sholl] Mixed ANOVA failed: {e}")
        results.append({
            "metric": "Sholl",
            "n_groups": np.nan,
            "n_total": np.nan,
            "groups": np.nan,
            "normal": np.nan,
            "normality_method": "repeated-measures module",
            "normality_p_by_group": np.nan,
            "levene_p": np.nan,
            "equal_variance": np.nan,
            "test": f"Mixed ANOVA FAILED: {str(e)}",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": np.nan,
            "sphericity_p": np.nan,
            "GG_epsilon": np.nan,
            "HF_epsilon": np.nan
        })

# ------------------------------------------------------------
# 3. MULTIPLE TESTING CORRECTION BY METRIC FAMILY (BH/FDR)
# ------------------------------------------------------------
df_stats_results = pd.DataFrame(results)

def assign_metric_family(metric):
    """
    Assign each metric to a biologically interpretable family.
    FDR correction will be performed separately within each family.
    """
    m = str(metric).lower()

    # Sholl summary metrics
    if (
        m == "sholl"
        or "sholl" in m
        or m in ["max_intersections", "radius_of_max", "auc", "decay_rate"]
    ):
        return "sholl"

    # Fractal / lacunarity morphology
    if m in ["fractal_dimension", "lacunarity_mean"]:
        return "fractal_lacunarity"

    # Texture features
    if m in ["contrast", "homogeneity", "energy", "correlation"]:
        return "glcm_texture"

    # Skeleton / graph morphology
    if m in [
        "endpoints",
        "bifurcations",
        "total_nodes",
        "total_edges",
        "total_length",
        "diameter",
        "avg_degree",
        "avg_junction_degree",
        "endpoint_density",
        "bifurcation_density",
        "branching_index",
        "largest_component_nodes",
        "n_components"
    ]:
        return "graph_morphology"

    # Multiscale fractal spectrum
    if m.startswith("mfs_scale_"):
        return "mfs"

    # Fallback
    return "other"


if df_stats_results.empty:
    warnings.warn("No statistical results were generated.")
    df_stats_results["metric_family"] = np.nan
    df_stats_results["p_adj_global"] = np.nan
    df_stats_results["p_adj_family"] = np.nan
    df_stats_results["significant_global_fdr"] = False
    df_stats_results["significant_family_fdr"] = False

else:
    # Assign metric family
    df_stats_results["metric_family"] = df_stats_results["metric"].apply(assign_metric_family)

    valid_mask = df_stats_results["p_value"].notna()

    # --------------------------------------------------------
    # Optional: keep global FDR for reference
    # --------------------------------------------------------
    df_stats_results["p_adj_global"] = np.nan

    if valid_mask.any():
        df_stats_results.loc[valid_mask, "p_adj_global"] = multipletests(
            df_stats_results.loc[valid_mask, "p_value"],
            method="fdr_bh"
        )[1]
    else:
        warnings.warn("No valid p-values available for global BH/FDR correction.")

    # --------------------------------------------------------
    # Main correction: FDR within each metric family
    # --------------------------------------------------------
    df_stats_results["p_adj_family"] = np.nan

    for family, idx in df_stats_results[valid_mask].groupby("metric_family").groups.items():
        idx = list(idx)

        df_stats_results.loc[idx, "p_adj_family"] = multipletests(
            df_stats_results.loc[idx, "p_value"],
            method="fdr_bh"
        )[1]

    # Keep p_adj as the main column used downstream
    # Now p_adj means family-wise FDR-adjusted p-value
    df_stats_results["p_adj"] = df_stats_results["p_adj_family"]

    # Significance flags
    df_stats_results["significant_raw"] = df_stats_results["p_value"] < 0.05
    df_stats_results["significant_global_fdr"] = df_stats_results["p_adj_global"] < 0.05
    df_stats_results["significant_family_fdr"] = df_stats_results["p_adj_family"] < 0.05

    n_nan = (~valid_mask).sum()
    if n_nan > 0:
        warnings.warn(
            f"{n_nan} metrics had NaN p-values and were excluded from BH/FDR correction."
        )

# ------------------------------------------------------------
# Export
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_stats_results.to_csv(os.path.join(EXPORT_DIR, "Module 9 - microglia_statistical_testing_results.csv"), index=False)

pd.set_option("display.max_rows", None)
df_stats_results

# ============================================================
# Post-hoc Pairwise Comparisons
# ============================================================

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if "df_master" not in globals():
    raise NameError("df_master not found. Please run Module 8 or the Master Builder first.")

if "df_stats_results" not in globals():
    raise NameError("df_stats_results not found. Please run the previous statistics module first.")

df_stats = df_master.copy()

GROUP_COL = "cell_state"
if GROUP_COL not in df_stats.columns:
    raise KeyError("cell_state column not found in df_master.")

def is_repeated_measures_metric(metric):
    return any(p in str(metric).lower() for p in ["sholl_", "mfs_scale_"]) or str(metric) == "Sholl"

def normalize_test_name(x):
    return str(x).strip().lower().replace("–", "-").replace("—", "-")

pairwise_rows = []

for _, row in df_stats_results.iterrows():

    metric = row["metric"]
    global_test = row["test"]
    p_adj_family = row.get("p_adj_family", row.get("p_adj", np.nan))
    metric_family = row.get("metric_family", "unknown")
    n_groups = row.get("n_groups", np.nan)

    # Skip repeated-measures metrics
    if is_repeated_measures_metric(metric):
        continue

    # Metric must exist in df_master
    if metric not in df_stats.columns:
        continue

    # Post-hoc only makes sense for >2 groups
    if pd.isna(n_groups) or int(n_groups) <= 2:
        continue

    # Only metrics significant after BH/FDR correction within their metric family
    if pd.isna(p_adj_family) or p_adj_family >= 0.05:
        continue

    sub = df_stats[[GROUP_COL, metric]].dropna().copy()
    if sub.empty:
        continue

    if sub[GROUP_COL].nunique() < 2:
        continue

    values = sub[metric].values
    labels = sub[GROUP_COL].values
    test_clean = normalize_test_name(global_test)

    # --------------------------------------------------------
    # ANOVA -> Tukey HSD
    # --------------------------------------------------------
    if test_clean == "anova":
        try:
            tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)

            for res in tukey.summary().data[1:]:
                g1, g2, meandiff, p_adj, lower, upper, reject = res

                pairwise_rows.append({
                    "metric": metric,
                    "metric_family": metric_family,
                    "global_test": global_test,
                    "group1": g1,
                    "group2": g2,
                    "mean_difference": meandiff,
                    "test": "Tukey HSD",
                    "p_unc": np.nan,
                    "p_adj": p_adj,
                    "significant": bool(reject)
                })

        except Exception as e:
            print(f"[POSTHOC ERROR] metric={metric} | test={global_test} | Tukey failed: {e}")

    # --------------------------------------------------------
    # Welch ANOVA -> Games-Howell
    # --------------------------------------------------------
    elif test_clean == "welch anova":
        try:
            gh = pg.pairwise_gameshowell(data=sub, dv=metric, between=GROUP_COL)

            pcol = "pval" if "pval" in gh.columns else ("p-unc" if "p-unc" in gh.columns else None)
            diffcol = "diff" if "diff" in gh.columns else None

            for _, r in gh.iterrows():
                pval = r[pcol] if pcol is not None else np.nan
                mdiff = r[diffcol] if diffcol is not None else np.nan

                pairwise_rows.append({
                    "metric": metric,
                    "metric_family": metric_family,
                    "global_test": global_test,
                    "group1": r["A"],
                    "group2": r["B"],
                    "mean_difference": mdiff,
                    "test": "Games-Howell",
                    "p_unc": np.nan,
                    "p_adj": pval,
                    "significant": bool(pval < 0.05) if pd.notna(pval) else False
                })

        except Exception as e:
            print(f"[POSTHOC ERROR] metric={metric} | test={global_test} | Games-Howell failed: {e}")

    # --------------------------------------------------------
    # Kruskal-Wallis -> Dunn + BH
    # --------------------------------------------------------
    elif test_clean == "kruskal-wallis":
        try:
            # Unadjusted Dunn
            dunn_unc = sp.posthoc_dunn(
                sub,
                val_col=metric,
                group_col=GROUP_COL,
                p_adjust=None
            )

            # BH-adjusted Dunn
            dunn_adj = sp.posthoc_dunn(
                sub,
                val_col=metric,
                group_col=GROUP_COL,
                p_adjust="fdr_bh"
            )

            groups = list(dunn_adj.index)

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    g1 = groups[i]
                    g2 = groups[j]

                    p_unc = dunn_unc.loc[g1, g2]
                    p_adj = dunn_adj.loc[g1, g2]

                    mean_diff = (
                        sub.loc[sub[GROUP_COL] == g1, metric].mean()
                        - sub.loc[sub[GROUP_COL] == g2, metric].mean()
                    )

                    pairwise_rows.append({
                        "metric": metric,
                        "metric_family": metric_family,
                        "global_test": global_test,
                        "group1": g1,
                        "group2": g2,
                        "mean_difference": mean_diff,
                        "test": "Dunn (BH)",
                        "p_unc": p_unc,
                        "p_adj": p_adj,
                        "significant": bool(p_adj < 0.05) if pd.notna(p_adj) else False
                    })

        except Exception as e:
            print(f"[POSTHOC ERROR] metric={metric} | test={global_test} | Dunn failed: {e}")

df_posthoc = pd.DataFrame(pairwise_rows)

# Uses dot decimal and comma separator.
df_posthoc.to_csv(os.path.join(EXPORT_DIR, "Module 9 - posthoc_pairwise_comparisons.csv"), index=False)

pd.set_option("display.max_rows", None)

df_posthoc

print("\n[INFO] ===== Module 9 Completed =====")

# ============================================================
# Dimensionality Reduction & Feature Importance
# ============================================================

print("\n[INFO] ===== Module 10 - Dimensionality Reduction & Feature Importance =====")

# ------------------------------------------------------------
# 1. Auto-detect master dataframe
# ------------------------------------------------------------
possible_master_names = ["df_master", "df_all", "df_features", "df_ordered"]

df_ml = None
for name in possible_master_names:
    if name in globals():
        df_ml = globals()[name].copy()
        break

if df_ml is None:
    raise NameError("No master dataframe found. Please create df_master.")

# ------------------------------------------------------------
# 2. Auto-detect grouping column
# ------------------------------------------------------------
possible_group_cols = ["cell_state", "condition", "group", "stage", "label"]

GROUP_COL = None
for col in possible_group_cols:
    if col in df_ml.columns:
        GROUP_COL = col
        break

if GROUP_COL is None:
    raise KeyError("No grouping column found.")

# ------------------------------------------------------------
# 3. Select numeric morphometric metrics
# ------------------------------------------------------------
EXCLUDE_COLS = ["cell_id", "image_id"]

numeric_cols = [
    c for c in df_ml.select_dtypes(include=[np.number]).columns
    if c not in EXCLUDE_COLS
]

X_df = df_ml[numeric_cols].copy()
medians = X_df.median(numeric_only=True)
X_df = X_df.fillna(medians)
X = X_df.values
y = df_ml[GROUP_COL].values

# ------------------------------------------------------------
# 4. Standardize features
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 10.1 PCA
# ============================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0], y=X_pca[:,1],
    hue=y, palette="viridis", s=60, alpha=0.9
)
plt.title("PCA — PC1 vs PC2")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(title=GROUP_COL)
plt.tight_layout()

plt.savefig(os.path.join(EXPORT_DIR, "Module 10 - pca.png"), dpi=600, bbox_inches="tight")
plt.show()

# PCA loadings
pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=numeric_cols,
    columns=["PC1", "PC2"]
).sort_values("PC1", ascending=False)

# ============================================================
# 10.2 UMAP
# ============================================================

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_umap[:,0], y=X_umap[:,1],
    hue=y, palette="viridis", s=60, alpha=0.9
)
plt.title("UMAP Embedding")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title=GROUP_COL)
plt.tight_layout()

plt.savefig(os.path.join(EXPORT_DIR, "Module 10 - umap.png"), dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# 10.3 Random Forest Feature Importance
# ============================================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_scaled, y)

importances = pd.DataFrame({
    "metric": numeric_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,10))
top_importances = importances.head(20)

sns.barplot(
    data=top_importances,
    x="importance",
    y="metric",
    hue="metric",
    palette="viridis",
    dodge=False,
    legend=False
)

plt.title("Top 20 Most Important Features (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Metric")
plt.tight_layout()

plt.savefig(os.path.join(EXPORT_DIR, "Module 10 - rf_top20.png"), dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# SHAP Interpretability
# ============================================================

# SHAP warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Use original feature names
feature_names = X_df.columns.tolist()

# SHAP explainer for tree-based models
explainer = shap.TreeExplainer(rf)

# Compute SHAP values
shap_values = explainer.shap_values(X_scaled)

# ------------------------------------------------------------
# Handle multiclass outputs robustly
# ------------------------------------------------------------

if isinstance(shap_values, list):
    # Convert to array: [n_classes, n_samples, n_features]
    shap_array = np.array(shap_values)

    # Mean absolute SHAP across classes and samples
    mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))

    # For summary plot: concatenate classes
    shap_for_plot = np.mean(np.abs(shap_array), axis=0)

else:
    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        # [n_samples, n_features, n_classes]
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
        shap_for_plot = np.mean(np.abs(shap_array), axis=2)
    else:
        # binary/single-output fallback
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
        shap_for_plot = shap_array

# ------------------------------------------------------------
# Create SHAP table
# ------------------------------------------------------------
df_shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

# ------------------------------------------------------------
# SHAP bar plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 10))
top_shap = df_shap_importance.head(20).sort_values("mean_abs_shap", ascending=True)

plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("Top 20 Features by SHAP Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(EXPORT_DIR, "Module 10 - shap_top20_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ------------------------------------------------------------
# SHAP summary plot
# ------------------------------------------------------------
shap.summary_plot(
    shap_for_plot,
    X_df,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(EXPORT_DIR, "Module 10 - shap_summary_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ------------------------------------------------------------
# SHAP importance table
# ------------------------------------------------------------

# Uses dot decimal and comma separator.
df_shap_importance.to_csv(
    os.path.join(EXPORT_DIR, "Module 10 - shap_importance_table.csv"),
    index=False
)

print("\n[INFO] ===== Module 10 Completed =====")

print("\n[INFO] ===== Finished =====")