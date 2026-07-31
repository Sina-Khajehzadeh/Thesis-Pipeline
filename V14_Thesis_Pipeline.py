"""Reproducible binary-classification experiment pipeline used in the thesis.

The pipeline is configured through JSON and requires the companion
``dataset_loader.py`` module. See ``V14_Thesis_Pipeline_Reader_Guide.ipynb``
for configuration examples, execution commands, and the artifact structure.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid

try:
    import psutil
except Exception:
    psutil = None

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

PIPELINE_VERSION = "V14"
ARTIFACT_SCHEMA_VERSION = "1.0"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================================
# 1. Repository configuration and output paths
# =============================================================================

def safe_path_part(value, default="run"):
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    chars = []
    for ch in text:
        chars.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    cleaned = "".join(chars).strip("._-")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or default


def scenario_folder_name(scenario_name):
    return safe_path_part(scenario_name, default="scenario")


def resolve_results_dir(config, total_n):
    layout_defaults = {
        "include_output_folder": True,
        "include_sample_size": True,
        "include_file_prefix": True,
        "timestamp_run_dir": False,
        "timestamp_format": "%Y%m%d_%H%M%S",
        "run_id": None,
        "results_dir": None,
    }
    layout = {**layout_defaults, **config.get("output_layout", {})}

    explicit_dir = config.get("results_dir", None) or layout.get("results_dir", None)
    if explicit_dir:
        return explicit_dir, layout

    parts = []
    if layout.get("include_output_folder", True):
        parts.append(safe_path_part(config.get("output_folder", "results")))
    if layout.get("include_sample_size", True):
        parts.append(f"N_{int(total_n)}")
    if layout.get("include_file_prefix", True):
        parts.append(safe_path_part(config.get("file_prefix", "run")))

    run_id = config.get("run_id", None) or layout.get("run_id", None)
    if run_id:
        parts.append(safe_path_part(run_id))

    if layout.get("timestamp_run_dir", False):
        parts.append(datetime.now().strftime(layout.get("timestamp_format", "%Y%m%d_%H%M%S")))

    if not parts:
        parts.append(f"results_N_{int(total_n)}")

    return os.path.join(config["output_root"], *parts), layout


RUN_CONFIG = None
MODEL_CONFIGS = None
PLOT_CONFIG = None
SCENARIO_CONFIGS = None

TOTAL_N = None
TRAIN_FRAC = None
ITERATIONS = None
RESULTS_DIR = None
FILE_PREFIX = None

def load_config(config):
    global RUN_CONFIG, MODEL_CONFIGS, PLOT_CONFIG, SCENARIO_CONFIGS
    global TOTAL_N, TRAIN_FRAC, ITERATIONS, RESULTS_DIR, FILE_PREFIX

    required_keys = [
        "total_n", "train_frac", "inner_validation_frac", "iterations",
        "output_root", "output_folder", "file_prefix", "scenarios",
        "budget_reference_model", "plots", "models",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"CONFIG is missing required key(s): {missing}")

    plot_defaults = {
        "plot_subfolder_name": "Plotly_PowerPoint_Slides",
        "save_html": True,
        "save_png": True,
        "save_svg": False,
        "save_jpeg": False,
        "show_figures": True,
        "save_wide_roc_calibration_version": False,
        "ppt_width": 1600,
        "ppt_height": 900,
        "roc_cal_width": 2000,
        "roc_cal_height": 800,
        "calibration_bins": 10,
        "image_scale": 2,
        "renderer": "colab",
    }
    plots = {**plot_defaults, **config.get("plots", {})}
    config = {**config, "plots": plots}

    budgeting_defaults = {
        "enabled": True,
        "mode": "reference_runtime",
        "reference_model": "TabPFN",
        "applies_to": "optuna_tuning_only",
        "non_tuned_reference_runtime_basis": (
            "reference_execution_runtime"
        ),
        "budgeted_selection_rule": "best_auc_within_budget",
        "no_budget_selection_rule": "best_auc_all_completed_trials",
    }
    budgeting = {**budgeting_defaults, **config.get("budgeting", {})}
    config = {**config, "budgeting": budgeting}

    splitting_defaults = {
        "strategy": "auto",
        "group_aware": True,
        "strict": True,
        "candidate_splits": 256,
        "min_class_count_per_partition": 2,
        "low_class_count_warning": 10,
        "max_prevalence_deviation": None,
        "max_row_fraction_deviation": None,
        "shuffle_rows_within_partitions": True,
        "require_groups": False,
        "temporal_window": "latest",
        "temporal_gap_rows": 0,
        "temporal_enforce_group_disjoint": False,
    }
    splitting = {**splitting_defaults, **config.get("splitting", {})}
    config = {**config, "splitting": splitting}

    preprocessing_defaults = {
        "mode": "auto",
        "recommended_raw_dataframe_mode": True,
        "numeric_imputation": "median",
        "categorical_imputation_value": "__MISSING__",
        "categorical_encoding": "onehot",
        "onehot_min_frequency": None,
        "onehot_max_categories": None,
        "max_output_features": None,
    }
    preprocessing = {**preprocessing_defaults, **config.get("preprocessing", {})}
    config = {**config, "preprocessing": preprocessing}

    RUN_CONFIG = config
    MODEL_CONFIGS = config["models"]
    PLOT_CONFIG = config["plots"]
    SCENARIO_CONFIGS = config["scenarios"]

    TOTAL_N = int(config["total_n"])
    TRAIN_FRAC = float(config["train_frac"])
    ITERATIONS = int(config["iterations"])

    if not 0 < TRAIN_FRAC < 1:
        raise ValueError("CONFIG['train_frac'] must be between 0 and 1.")
    if not 0 < float(config["inner_validation_frac"]) < 1:
        raise ValueError("CONFIG['inner_validation_frac'] must be between 0 and 1.")
    if TOTAL_N <= 0 or ITERATIONS <= 0:
        raise ValueError("CONFIG['total_n'] and CONFIG['iterations'] must be positive.")
    if int(splitting["candidate_splits"]) <= 0:
        raise ValueError("CONFIG['splitting']['candidate_splits'] must be positive.")
    if int(splitting["min_class_count_per_partition"]) < 1:
        raise ValueError(
            "CONFIG['splitting']['min_class_count_per_partition'] must be at least 1."
        )
    if int(splitting["temporal_gap_rows"]) < 0:
        raise ValueError("CONFIG['splitting']['temporal_gap_rows'] cannot be negative.")
    if str(splitting["temporal_window"]).lower() not in {"latest", "earliest"}:
        raise ValueError(
            "CONFIG['splitting']['temporal_window'] must be 'latest' or 'earliest'."
        )
    for key in ("max_prevalence_deviation", "max_row_fraction_deviation"):
        value = splitting.get(key)
        if value is not None and float(value) < 0:
            raise ValueError(f"CONFIG['splitting']['{key}'] cannot be negative.")

    RESULTS_DIR, output_layout = resolve_results_dir(config, TOTAL_N)
    config = {
        **config,
        "output_layout": output_layout,
        "resolved_results_dir": RESULTS_DIR,
    }
    RUN_CONFIG = config

    FILE_PREFIX = safe_path_part(f"{config['file_prefix']}_{TOTAL_N}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    refresh_plot_settings()

    print("Configuration loaded inside pipeline module")
    print(f"Results folder: {RESULTS_DIR}")





# =============================================================================
# 2. Scientific-computing dependencies and shared utilities
# =============================================================================

import logging
import sys
import os
import contextlib
from contextlib import contextmanager
import warnings
import time
import pickle
import json
import io
import hashlib

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

try:
    import optuna
except Exception:
    optuna = None

try:
    import numpy as np
except Exception as exc:
    raise ImportError("numpy is required for this pipeline. Install it with: pip install numpy") from exc

try:
    import pandas as pd
except Exception as exc:
    raise ImportError("pandas is required for this pipeline. Install it with: pip install pandas") from exc

try:
    import scipy.stats as st
    from scipy.stats import multivariate_normal, multivariate_t
    from scipy.linalg import block_diag
except Exception as exc:
    raise ImportError("scipy is required for this pipeline. Install it with: pip install scipy") from exc

try:
    import torch
except Exception:
    torch = None


def cuda_available():
    """Return True only if torch is installed and reports a usable CUDA device.

    Safe to call when torch is not installed (returns False instead of raising).
    """
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


try:
    import matplotlib
    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    SMOTE = None
    RandomUnderSampler = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        GridSearchCV,
        GroupShuffleSplit,
        StratifiedGroupKFold,
        train_test_split,
    )
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.calibration import calibration_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        OneHotEncoder,
        PolynomialFeatures,
        StandardScaler,
    )
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
        roc_curve,
        recall_score,
        precision_score,
        brier_score_loss,
    )
except Exception as exc:
    raise ImportError("scikit-learn is required for this pipeline. Install it with: pip install scikit-learn") from exc

try:
    from xgboost import XGBClassifier, callback
except Exception:
    XGBClassifier = None
    callback = None

try:
    from catboost import CatBoostClassifier, Pool
except Exception:
    CatBoostClassifier = None
    Pool = None

try:
    from tabpfn_client import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
        AutoTabPFNClassifier,
    )
except Exception:
    AutoTabPFNClassifier = None

try:
    from tabpfn import TabPFNClassifier as LocalTabPFNClassifier
except Exception:
    LocalTabPFNClassifier = None

try:
    from tabpfn.model_loading import ModelVersion as AutoTabPFNModelVersion
except Exception:
    try:
        from tabpfn.constants import ModelVersion as AutoTabPFNModelVersion
    except Exception:
        AutoTabPFNModelVersion = None

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None

warnings.filterwarnings("ignore")
if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


optuna_trials_memory = {}
budget_timing_memory = []
selection_memory = []
final_fit_timing_memory = []
tabpfn_budget_evaluation_memory = []

# =============================================================================
# 3. Metrics, sampling, splitting, and train-only preprocessing
# =============================================================================

def scenario_file_prefix(scenario_name, file_prefix):
    return f"{safe_path_part(file_prefix)}_{safe_path_part(scenario_name, default='scenario')}"



def get_ci(data):
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values

    if len(data) < 2 or np.std(data) == 0:
        return 0.0

    return st.t.interval(
        0.95,
        len(data) - 1,
        loc=np.mean(data),
        scale=st.sem(data)
    )[1] - np.mean(data)


def safe_mean(data):
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values

    if len(data) == 0:
        return np.nan

    return float(np.mean(data))



def safe_sd(data):
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values

    if len(data) < 2:
        return 0.0

    return float(np.std(data, ddof=1))


def safe_sum(data):
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values

    if len(data) == 0:
        return 0.0

    return float(np.sum(data))


def predict_binary_argmax(model, X):
    """Run one probability inference and derive labels by class-wise argmax."""
    if not hasattr(model, "classes_"):
        raise ValueError(
            f"{type(model).__name__} must expose classes_ after fit; expected labels 0 and 1."
        )

    classes = np.asarray(model.classes_)
    labels = classes.tolist() if classes.ndim == 1 else []
    if len(labels) != 2 or labels.count(0) != 1 or labels.count(1) != 1:
        raise ValueError(
            f"{type(model).__name__}.classes_ must contain exactly labels 0 and 1; "
            f"received {classes!r}."
        )

    proba_all = np.asarray(model.predict_proba(X))
    if proba_all.ndim != 2 or proba_all.shape[1] != len(classes):
        raise ValueError(
            f"{type(model).__name__}.predict_proba() must return one column per class; "
            f"received shape {proba_all.shape} for classes {classes!r}."
        )

    positive_class_index = labels.index(1)
    proba = proba_all[:, positive_class_index]
    pred = classes[np.argmax(proba_all, axis=1)]
    return pred, proba


def evaluate_binary_model(model, X_test, y_test):
    pred, proba = predict_binary_argmax(model, X_test)

    fpr, tpr, _ = roc_curve(y_test, proba)
    sensitivity = recall_score(y_test, pred, zero_division=0)
    precision = precision_score(y_test, pred, zero_division=0)

    return (
        balanced_accuracy_score(y_test, pred),
        roc_auc_score(y_test, proba),
        brier_score_loss(y_test, proba, pos_label=1),
        fpr,
        tpr,
        sensitivity,
        precision,
        proba,
        y_test
    )


def stop_tracker_get_energy(tracker):
    _ = tracker.stop()
    emissions_data = getattr(tracker, "final_emissions_data", None)
    return getattr(emissions_data, "energy_consumed", np.nan)



def _v11_legacy_group_majority_label(y_array, groups):
    """Return arrays of (unique_group_id, majority_label) for stratifying groups.

    A group's label is its majority row-label; used only to keep the group-level
    train/test split approximately class-balanced. Ground-truth row labels are
    untouched.
    """
    uniq = np.unique(groups)
    g_label = np.empty(len(uniq), dtype=int)
    for i, g in enumerate(uniq):
        yl = y_array[groups == g]
        g_label[i] = int(round(float(np.mean(yl)) >= 0.5))
    return uniq, g_label


def _v11_legacy_grouped_sample_split(
    X_np,
    y_array,
    groups,
    total_n,
    train_frac=0.8,
    seed=42,
    sampling_config=None,
):
    """Patient/group-aware analogue of exact_sample_split_from_config.

    Draws WHOLE groups until about `total_n` rows are collected, then assigns
    whole groups to train vs test by `train_frac`. No group appears on both
    sides. Class prevalence is approximated at the group level (exact row-level
    prevalence control is impossible once whole groups are kept together).

    Returns the same tuple shape as the row-level splitters, plus group arrays:
      X_train, X_test, y_train, y_test, split_info, groups_train, groups_test
    """
    if sampling_config is None:
        sampling_config = {"strategy": "original_prevalence"}
    strategy = sampling_config.get("strategy", "original_prevalence")

    rng = np.random.default_rng(seed)
    original_prev1 = float(np.mean(y_array == 1))

    if strategy == "original_prevalence":
        target_prev1 = original_prev1
    elif strategy == "balanced":
        target_prev1 = 0.50
    elif strategy == "custom_prevalence":
        target_prev1 = float(sampling_config["class1_prevalence"])
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    uniq, g_label = _v11_legacy_group_majority_label(y_array, groups)

    sizes = {g: int(np.sum(groups == g)) for g in uniq}

    g0 = uniq[g_label == 0]
    g1 = uniq[g_label == 1]
    rng.shuffle(g0)
    rng.shuffle(g1)

    chosen = []
    rows_total = 0
    rows_pos = 0
    i0 = i1 = 0
    while rows_total < total_n and (i0 < len(g0) or i1 < len(g1)):
        want_pos = (rows_pos / rows_total) < target_prev1 if rows_total > 0 else target_prev1 >= 0.5
        if want_pos and i1 < len(g1):
            g = g1[i1]; i1 += 1
        elif (not want_pos) and i0 < len(g0):
            g = g0[i0]; i0 += 1
        elif i1 < len(g1):
            g = g1[i1]; i1 += 1
        elif i0 < len(g0):
            g = g0[i0]; i0 += 1
        else:
            break
        chosen.append(g)
        rows_total += sizes[g]
        rows_pos += int(np.sum(y_array[groups == g] == 1))

    chosen = np.array(chosen)
    if len(chosen) < 2:
        raise ValueError(
            "Group-aware split needs at least 2 groups in the subsample; "
            "got fewer. Reduce total_n or check the groups array."
        )

    chosen_set = set(chosen.tolist())
    c_label = {g: int(round(float(np.mean(y_array[groups == g])) >= 0.5)) for g in chosen}
    c0 = np.array([g for g in chosen if c_label[g] == 0])
    c1 = np.array([g for g in chosen if c_label[g] == 1])
    rng.shuffle(c0)
    rng.shuffle(c1)

    def take(arr):
        k = int(round(len(arr) * train_frac))
        k = min(max(k, 1), max(len(arr) - 1, 1)) if len(arr) >= 2 else len(arr)
        return arr[:k], arr[k:]

    tr0, te0 = take(c0)
    tr1, te1 = take(c1)
    train_groups = np.concatenate([tr0, tr1])
    test_groups = np.concatenate([te0, te1])

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = X_np[train_idx]
    X_test = X_np[test_idx]
    y_train = y_array[train_idx]
    y_test = y_array[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    y_sample = np.concatenate([y_train, y_test])
    split_info = {
        "Sampling_Strategy": strategy,
        "Split_Level": "group",
        "Original_Class1_Prevalence": original_prev1,
        "Target_Class1_Prevalence": target_prev1,
        "Sample_Class1_Prevalence": float(np.mean(y_sample == 1)),
        "Train_Class1_Prevalence": float(np.mean(y_train == 1)) if len(y_train) else float("nan"),
        "Test_Class1_Prevalence": float(np.mean(y_test == 1)) if len(y_test) else float("nan"),
        "Train_N": len(y_train),
        "Test_N": len(y_test),
        "Train_Class0_N": int(np.sum(y_train == 0)),
        "Train_Class1_N": int(np.sum(y_train == 1)),
        "Test_Class0_N": int(np.sum(y_test == 0)),
        "Test_Class1_N": int(np.sum(y_test == 1)),
        "N_Groups_Total": int(len(chosen)),
        "N_Groups_Train": int(len(train_groups)),
        "N_Groups_Test": int(len(test_groups)),
    }
    return X_train, X_test, y_train, y_test, split_info, groups_train, groups_test


def _v11_legacy_grouped_inner_split(
    X_train, y_train, groups_train, val_frac=0.2, seed=42
):
    """Group-aware inner train/validation split for Optuna tuning.

    Assigns whole groups to train_sub vs validation, so no group straddles the
    tuning/validation boundary. Falls back gracefully if there are very few
    groups. Returns: X_train_sub, X_val, y_train_sub, y_val.
    """
    from sklearn.model_selection import GroupShuffleSplit

    uniq = np.unique(groups_train)
    if len(uniq) < 2:
        from sklearn.model_selection import train_test_split
        return train_test_split(
            X_train, y_train, test_size=val_frac, stratify=y_train, random_state=seed
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(gss.split(X_train, y_train, groups=groups_train))
    return X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]


def _v11_legacy_exact_stratified_sample_split(
    X_np, y_array, total_n, train_frac=0.8, seed=42
):
    rng = np.random.default_rng(seed)

    idx0 = np.where(y_array == 0)[0]
    idx1 = np.where(y_array == 1)[0]

    prev1 = float(np.mean(y_array == 1))

    n_train = int(total_n * train_frac)
    n_test = total_n - n_train

    n1_train = int(round(n_train * prev1))
    n0_train = n_train - n1_train

    n1_test = int(round(n_test * prev1))
    n0_test = n_test - n1_test

    assert len(idx0) >= n0_train + n0_test, "Not enough class-0 samples"
    assert len(idx1) >= n1_train + n1_test, "Not enough class-1 samples"

    train0 = rng.choice(idx0, size=n0_train, replace=False)
    train1 = rng.choice(idx1, size=n1_train, replace=False)

    rem0 = np.setdiff1d(idx0, train0)
    rem1 = np.setdiff1d(idx1, train1)

    test0 = rng.choice(rem0, size=n0_test, replace=False)
    test1 = rng.choice(rem1, size=n1_test, replace=False)

    train_idx = np.concatenate([train0, train1])
    test_idx = np.concatenate([test0, test1])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train_clean = X_np[train_idx]
    X_test_clean = X_np[test_idx]
    y_train = y_array[train_idx]
    y_test = y_array[test_idx]

    y_sample = np.concatenate([y_train, y_test])

    split_info = {
        "Original_Class1_Prevalence": float(prev1),
        "Sample_Class1_Prevalence": float(np.mean(y_sample == 1)),
        "Train_Class1_Prevalence": float(np.mean(y_train == 1)),
        "Test_Class1_Prevalence": float(np.mean(y_test == 1)),
        "Train_N": len(y_train),
        "Test_N": len(y_test),
        "Train_Class0_N": int(np.sum(y_train == 0)),
        "Train_Class1_N": int(np.sum(y_train == 1)),
        "Test_Class0_N": int(np.sum(y_test == 0)),
        "Test_Class1_N": int(np.sum(y_test == 1)),
    }

    return X_train_clean, X_test_clean, y_train, y_test, split_info


def _v11_legacy_exact_sample_split_from_config(
    X_np,
    y_array,
    total_n,
    train_frac=0.8,
    seed=42,
    sampling_config=None
):
    if sampling_config is None:
        sampling_config = {"strategy": "original_prevalence"}

    strategy = sampling_config.get("strategy", "original_prevalence")

    original_prev1 = float(np.mean(y_array == 1))

    if strategy == "original_prevalence":
        target_prev1 = original_prev1

    elif strategy == "balanced":
        target_prev1 = 0.50

    elif strategy == "custom_prevalence":
        target_prev1 = float(sampling_config["class1_prevalence"])

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    rng = np.random.default_rng(seed)

    idx0 = np.where(y_array == 0)[0]
    idx1 = np.where(y_array == 1)[0]

    n_train = int(total_n * train_frac)
    n_test = total_n - n_train

    n1_train = int(round(n_train * target_prev1))
    n0_train = n_train - n1_train

    n1_test = int(round(n_test * target_prev1))
    n0_test = n_test - n1_test

    assert len(idx0) >= n0_train + n0_test, "Not enough class-0 samples"
    assert len(idx1) >= n1_train + n1_test, "Not enough class-1 samples"

    train0 = rng.choice(idx0, size=n0_train, replace=False)
    train1 = rng.choice(idx1, size=n1_train, replace=False)

    rem0 = np.setdiff1d(idx0, train0)
    rem1 = np.setdiff1d(idx1, train1)

    test0 = rng.choice(rem0, size=n0_test, replace=False)
    test1 = rng.choice(rem1, size=n1_test, replace=False)

    train_idx = np.concatenate([train0, train1])
    test_idx = np.concatenate([test0, test1])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train_clean = X_np[train_idx]
    X_test_clean = X_np[test_idx]
    y_train = y_array[train_idx]
    y_test = y_array[test_idx]

    y_sample = np.concatenate([y_train, y_test])

    split_info = {
        "Sampling_Strategy": strategy,
        "Original_Class1_Prevalence": original_prev1,
        "Target_Class1_Prevalence": target_prev1,
        "Sample_Class1_Prevalence": float(np.mean(y_sample == 1)),
        "Train_Class1_Prevalence": float(np.mean(y_train == 1)),
        "Test_Class1_Prevalence": float(np.mean(y_test == 1)),
        "Train_N": len(y_train),
        "Test_N": len(y_test),
        "Train_Class0_N": int(np.sum(y_train == 0)),
        "Train_Class1_N": int(np.sum(y_train == 1)),
        "Test_Class0_N": int(np.sum(y_test == 0)),
        "Test_Class1_N": int(np.sum(y_test == 1)),
    }

    return X_train_clean, X_test_clean, y_train, y_test, split_info


def _take_rows(values, indices):
    """Positionally select rows without discarding pandas dtypes or labels."""
    indices = np.asarray(indices, dtype=int)
    if hasattr(values, "iloc"):
        return values.iloc[indices].copy()
    return np.asarray(values)[indices]


def _binary_counts(y_values):
    y_values = np.asarray(y_values, dtype=int).ravel()
    return int(np.sum(y_values == 0)), int(np.sum(y_values == 1))


def _validate_binary_partition(y_values, partition_name, min_class_count=1):
    y_values = np.asarray(y_values).ravel()
    labels = set(np.unique(y_values).tolist())
    if labels != {0, 1}:
        raise ValueError(
            f"{partition_name} must contain both binary labels 0 and 1; "
            f"received labels {sorted(labels)}. Increase the sample size, reduce "
            "the number of partitions, or use a feasible split strategy."
        )
    n0, n1 = _binary_counts(y_values)
    if min(n0, n1) < int(min_class_count):
        raise ValueError(
            f"{partition_name} has class counts class-0={n0}, class-1={n1}; "
            f"each class requires at least {int(min_class_count)} rows."
        )
    return n0, n1


def _target_class1_prevalence(y_array, sampling_config):
    sampling_config = sampling_config or {"strategy": "original_prevalence"}
    strategy = str(sampling_config.get("strategy", "original_prevalence")).lower()
    original = float(np.mean(np.asarray(y_array) == 1))
    if strategy == "original_prevalence":
        target = original
    elif strategy == "balanced":
        target = 0.5
    elif strategy == "custom_prevalence":
        if "class1_prevalence" not in sampling_config:
            raise ValueError(
                "sampling.strategy='custom_prevalence' requires class1_prevalence."
            )
        target = float(sampling_config["class1_prevalence"])
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if not 0 < target < 1:
        raise ValueError("The target class-1 prevalence must be strictly between 0 and 1.")
    return strategy, original, target


def _split_fingerprint(train_indices, test_indices, seed, strategy):
    digest = hashlib.sha256()
    digest.update(str(strategy).encode("utf-8"))
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(test_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()[:20]


def _shuffle_partition_indices(indices, rng, enabled):
    indices = np.asarray(indices, dtype=int).copy()
    if enabled:
        rng.shuffle(indices)
    return indices


def _allocate_binary_partition_counts(
    available_n0,
    available_n1,
    total_n,
    train_frac,
    target_prev1,
    min_class_count,
):
    total_n = int(total_n)
    n_train = int(round(total_n * float(train_frac)))
    n_train = min(max(n_train, 1), total_n - 1)
    n_test = total_n - n_train
    minimum = int(min_class_count)
    if min(n_train, n_test) < 2 * minimum:
        raise ValueError(
            f"Requested split sizes train={n_train}, test={n_test} cannot place at "
            f"least {minimum} rows of each class in both partitions."
        )

    minimum_total_n1 = max(2 * minimum, total_n - int(available_n0))
    maximum_total_n1 = min(total_n - 2 * minimum, int(available_n1))
    if minimum_total_n1 > maximum_total_n1:
        raise ValueError(
            "The requested sample size, split fractions, class prevalence, and "
            "minimum class counts are infeasible for the available binary labels."
        )
    total_n1 = int(np.clip(
        round(total_n * float(target_prev1)), minimum_total_n1, maximum_total_n1
    ))
    train_n1_min = max(minimum, total_n1 - (n_test - minimum))
    train_n1_max = min(n_train - minimum, total_n1 - minimum)
    if train_n1_min > train_n1_max:
        raise ValueError("No feasible stratified train/test class allocation exists.")
    train_n1 = int(np.clip(
        round(n_train * float(target_prev1)), train_n1_min, train_n1_max
    ))
    test_n1 = total_n1 - train_n1
    train_n0 = n_train - train_n1
    test_n0 = n_test - test_n1
    return train_n0, train_n1, test_n0, test_n1


def _build_split_info(
    y_all,
    y_train,
    y_test,
    train_indices,
    test_indices,
    sampling_strategy,
    target_prev1,
    requested_total_n,
    requested_train_frac,
    seed,
    split_strategy,
    min_class_count,
    candidate_method,
    candidate_count,
    candidate_score,
    groups_all=None,
):
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if len(np.unique(train_indices)) != len(train_indices):
        raise RuntimeError("Duplicate row indices detected inside the training partition.")
    if len(np.unique(test_indices)) != len(test_indices):
        raise RuntimeError("Duplicate row indices detected inside the test partition.")
    row_overlap = np.intersect1d(train_indices, test_indices)
    if len(row_overlap):
        raise RuntimeError(
            f"Row leakage detected: {len(row_overlap)} rows occur in train and test."
        )
    train_n0, train_n1 = _validate_binary_partition(
        y_train, "training partition", min_class_count
    )
    test_n0, test_n1 = _validate_binary_partition(
        y_test, "test partition", min_class_count
    )
    sample_y = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    actual_total = int(len(sample_y))
    actual_train_frac = float(len(y_train) / actual_total)
    sample_prev = float(np.mean(sample_y == 1))
    train_prev = float(np.mean(np.asarray(y_train) == 1))
    test_prev = float(np.mean(np.asarray(y_test) == 1))

    info = {
        "Sampling_Strategy": sampling_strategy,
        "Split_Strategy": split_strategy,
        "Split_Level": "group" if groups_all is not None else "row",
        "Split_Seed": int(seed),
        "Split_Valid": True,
        "Split_Fingerprint": _split_fingerprint(
            train_indices, test_indices, seed, split_strategy
        ),
        "Train_Test_Row_Overlap_N": 0,
        "Original_Class1_Prevalence": float(np.mean(np.asarray(y_all) == 1)),
        "Target_Class1_Prevalence": float(target_prev1),
        "Sample_Class1_Prevalence": sample_prev,
        "Train_Class1_Prevalence": train_prev,
        "Test_Class1_Prevalence": test_prev,
        "Requested_Total_N": int(requested_total_n),
        "Actual_Total_N": actual_total,
        "Requested_Train_Fraction": float(requested_train_frac),
        "Actual_Train_Fraction": actual_train_frac,
        "Total_N_Deviation": int(actual_total - int(requested_total_n)),
        "Train_Fraction_Deviation": float(actual_train_frac - requested_train_frac),
        "Sample_Prevalence_Deviation": float(sample_prev - target_prev1),
        "Train_Prevalence_Deviation": float(train_prev - target_prev1),
        "Test_Prevalence_Deviation": float(test_prev - target_prev1),
        "Train_N": int(len(y_train)),
        "Test_N": int(len(y_test)),
        "Train_Class0_N": train_n0,
        "Train_Class1_N": train_n1,
        "Test_Class0_N": test_n0,
        "Test_Class1_N": test_n1,
        "Split_Candidate_Method": candidate_method,
        "Split_Candidates_Evaluated": int(candidate_count),
        "Split_Candidate_Score": float(candidate_score),
        "_train_indices": np.asarray(train_indices, dtype=int),
        "_test_indices": np.asarray(test_indices, dtype=int),
    }

    if groups_all is not None:
        groups_all = np.asarray(groups_all)
        train_groups = np.unique(groups_all[np.asarray(train_indices, dtype=int)])
        test_groups = np.unique(groups_all[np.asarray(test_indices, dtype=int)])
        overlap = np.intersect1d(train_groups, test_groups)
        if len(overlap):
            raise RuntimeError(
                f"Group leakage detected: {len(overlap)} groups occur in train and test."
            )
        info.update({
            "N_Groups_Total": int(len(np.union1d(train_groups, test_groups))),
            "N_Groups_Train": int(len(train_groups)),
            "N_Groups_Test": int(len(test_groups)),
            "Train_Test_Group_Overlap_N": 0,
        })
    else:
        info.update({
            "N_Groups_Total": np.nan,
            "N_Groups_Train": np.nan,
            "N_Groups_Test": np.nan,
            "Train_Test_Group_Overlap_N": np.nan,
        })
    return info


def exact_sample_split_from_config(
    X_np,
    y_array,
    total_n,
    train_frac=0.8,
    seed=42,
    sampling_config=None,
    split_config=None,
):
    """Exact row-level binary sampling with deterministic stratified partitions."""
    split_config = split_config or {}
    min_count = int(split_config.get("min_class_count_per_partition", 2))
    y_array = np.asarray(y_array, dtype=int).ravel()
    if int(total_n) > len(y_array):
        raise ValueError(
            f"total_n={int(total_n)} exceeds the {len(y_array)} available rows."
        )
    _validate_binary_partition(y_array, "complete dataset", 1)
    strategy, original_prev, target_prev = _target_class1_prevalence(
        y_array, sampling_config
    )
    idx0 = np.flatnonzero(y_array == 0)
    idx1 = np.flatnonzero(y_array == 1)
    tr0_n, tr1_n, te0_n, te1_n = _allocate_binary_partition_counts(
        len(idx0), len(idx1), total_n, train_frac, target_prev, min_count
    )
    rng = np.random.default_rng(seed)
    idx0 = rng.permutation(idx0)
    idx1 = rng.permutation(idx1)
    train_idx = np.concatenate([idx0[:tr0_n], idx1[:tr1_n]])
    test_idx = np.concatenate([
        idx0[tr0_n:tr0_n + te0_n],
        idx1[tr1_n:tr1_n + te1_n],
    ])
    shuffle = bool(split_config.get("shuffle_rows_within_partitions", True))
    train_idx = _shuffle_partition_indices(train_idx, rng, shuffle)
    test_idx = _shuffle_partition_indices(test_idx, rng, shuffle)
    y_train = y_array[train_idx]
    y_test = y_array[test_idx]
    score = (
        abs(float(np.mean(y_train == 1)) - target_prev)
        + abs(float(np.mean(y_test == 1)) - target_prev)
    )
    info = _build_split_info(
        y_array, y_train, y_test, train_idx, test_idx, strategy, target_prev,
        total_n, train_frac, seed, "stratified", min_count,
        "exact_binary_allocation", 1, score,
    )
    return (
        _take_rows(X_np, train_idx),
        _take_rows(X_np, test_idx),
        y_train,
        y_test,
        info,
    )


def exact_stratified_sample_split(
    X_np, y_array, total_n, train_frac=0.8, seed=42, split_config=None
):
    return exact_sample_split_from_config(
        X_np=X_np,
        y_array=y_array,
        total_n=total_n,
        train_frac=train_frac,
        seed=seed,
        sampling_config={"strategy": "original_prevalence"},
        split_config=split_config,
    )


def _select_group_subset_indices(
    y_array, groups, total_n, target_prev1, seed, candidate_splits, min_count
):
    y_array = np.asarray(y_array, dtype=int)
    group_codes, unique_groups = pd.factorize(np.asarray(groups), sort=False)
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("Group-aware splitting requires at least two unique groups.")
    if int(total_n) > len(y_array):
        raise ValueError(
            f"total_n={int(total_n)} exceeds the {len(y_array)} available rows."
        )
    if int(total_n) == len(y_array):
        return np.arange(len(y_array)), 1, 0.0

    group_size = np.bincount(group_codes, minlength=n_groups)
    group_pos = np.bincount(group_codes, weights=y_array, minlength=n_groups)
    base_rng = np.random.default_rng(seed)
    best = None
    evaluated = 0
    group_rate = group_pos / group_size
    original_prev = float(np.mean(y_array == 1))
    rate_scale = max(float(np.std(group_rate)), 0.05)
    base_strength = (float(target_prev1) - original_prev) / rate_scale
    strength_multipliers = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    for candidate_number in range(int(candidate_splits)):
        strength = base_strength * strength_multipliers[
            candidate_number % len(strength_multipliers)
        ]
        weights = np.exp(np.clip(
            strength * (group_rate - original_prev), -8.0, 8.0
        ))
        random_u = np.maximum(base_rng.random(n_groups), np.finfo(float).tiny)
        order = np.argsort(-np.log(random_u) / weights, kind="stable")
        cumulative_rows = np.cumsum(group_size[order])
        cumulative_pos = np.cumsum(group_pos[order])
        crossing = int(np.searchsorted(cumulative_rows, int(total_n), side="left"))

        for endpoint in sorted({crossing - 1, crossing, crossing + 1}):
            if endpoint < 0 or endpoint >= n_groups:
                continue
            chosen_codes = order[:endpoint + 1]
            sampled_n = int(cumulative_rows[endpoint])
            n1 = int(round(float(cumulative_pos[endpoint])))
            n0 = sampled_n - n1
            if min(n0, n1) < 2 * int(min_count):
                continue
            evaluated += 1
            prevalence = float(n1 / sampled_n)
            score = (
                abs(sampled_n - int(total_n)) / max(1, int(total_n))
                + abs(prevalence - target_prev1)
            )
            candidate = (
                score,
                abs(sampled_n - int(total_n)),
                candidate_number,
                chosen_codes.copy(),
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is None:
        raise ValueError(
            "No whole-group subsample can satisfy the requested sample size and "
            "minimum binary-class counts. Increase total_n or relax the class floor."
        )
    selected_mask = np.isin(group_codes, np.asarray(best[3], dtype=int))
    selected_indices = np.flatnonzero(selected_mask)
    return selected_indices, evaluated, float(best[0])


def _best_group_partition_indices(
    y_values, groups_values, train_frac, seed, split_config
):
    y_values = np.asarray(y_values, dtype=int).ravel()
    groups_values = np.asarray(groups_values).ravel()
    min_count = int(split_config.get("min_class_count_per_partition", 2))
    n_candidates = int(split_config.get("candidate_splits", 256))
    group_codes, unique_groups = pd.factorize(groups_values, sort=False)
    if len(unique_groups) < 2:
        raise ValueError("A group-aware partition requires at least two unique groups.")
    overall_prev = float(np.mean(y_values == 1))
    group_size = np.bincount(group_codes, minlength=len(unique_groups))
    group_pos = np.bincount(
        group_codes, weights=y_values, minlength=len(unique_groups)
    )
    total_pos = int(np.sum(y_values == 1))
    total_rows = len(y_values)
    candidates = []

    def consider_group_codes(train_group_codes, method):
        train_group_codes = np.asarray(train_group_codes, dtype=int)
        if len(train_group_codes) == 0 or len(train_group_codes) == len(unique_groups):
            return
        train_rows = int(np.sum(group_size[train_group_codes]))
        tr1 = int(round(float(np.sum(group_pos[train_group_codes]))))
        tr0 = train_rows - tr1
        test_rows = total_rows - train_rows
        te1 = total_pos - tr1
        te0 = test_rows - te1
        if min(tr0, tr1, te0, te1) < min_count:
            return
        actual_frac = train_rows / total_rows
        tr_prev = tr1 / train_rows
        te_prev = te1 / test_rows
        score = (
            2.0 * abs(actual_frac - train_frac)
            + abs(tr_prev - overall_prev)
            + abs(te_prev - overall_prev)
        )
        candidates.append((score, method, train_group_codes.copy()))

    requested_test_frac = 1.0 - float(train_frac)
    approx_folds = int(round(1.0 / requested_test_frac))
    approx_folds = min(max(2, approx_folds), len(unique_groups))
    try:
        sgkf = StratifiedGroupKFold(
            n_splits=approx_folds, shuffle=True, random_state=int(seed)
        )
        dummy = np.zeros((len(y_values), 1), dtype=np.uint8)
        for train_idx, test_idx in sgkf.split(dummy, y_values, groups_values):
            consider_group_codes(
                np.unique(group_codes[np.asarray(train_idx, dtype=int)]),
                "StratifiedGroupKFold",
            )
    except ValueError:
        pass

    rng = np.random.default_rng(seed)
    n_train_groups = int(round(len(unique_groups) * float(train_frac)))
    n_train_groups = min(max(1, n_train_groups), len(unique_groups) - 1)
    for _ in range(n_candidates):
        train_group_codes = rng.permutation(len(unique_groups))[:n_train_groups]
        consider_group_codes(
            train_group_codes, "deterministic_group_candidate_search"
        )

    if not candidates:
        raise ValueError(
            "No leakage-free group partition contains both labels with the configured "
            f"minimum of {min_count} rows per class. The class/group structure is "
            "incompatible with this split; V12 will not fall back to row splitting."
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    score, method, selected_train_groups = candidates[0]
    train_mask = np.isin(group_codes, selected_train_groups)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)

    max_prev = split_config.get("max_prevalence_deviation")
    if max_prev is not None:
        deviations = [
            abs(float(np.mean(y_values[train_idx] == 1)) - overall_prev),
            abs(float(np.mean(y_values[test_idx] == 1)) - overall_prev),
        ]
        if max(deviations) > float(max_prev):
            raise ValueError(
                "Best group split exceeds splitting.max_prevalence_deviation: "
                f"observed {max(deviations):.6f}, allowed {float(max_prev):.6f}."
            )
    max_fraction = split_config.get("max_row_fraction_deviation")
    actual_fraction = len(train_idx) / len(y_values)
    if max_fraction is not None and abs(actual_fraction - train_frac) > float(max_fraction):
        raise ValueError(
            "Best group split exceeds splitting.max_row_fraction_deviation: "
            f"observed {abs(actual_fraction - train_frac):.6f}, "
            f"allowed {float(max_fraction):.6f}."
        )
    return train_idx, test_idx, method, len(candidates), float(score)


def grouped_sample_split(
    X_np,
    y_array,
    groups,
    total_n,
    train_frac=0.8,
    seed=42,
    sampling_config=None,
    split_config=None,
):
    """Select whole groups, then find the best valid leakage-free outer split."""
    split_config = split_config or {}
    y_array = np.asarray(y_array, dtype=int).ravel()
    groups = np.asarray(groups).ravel()
    if len(groups) != len(y_array):
        raise ValueError("groups must have exactly one value per row.")
    if pd.isna(groups).any():
        raise ValueError("groups contains missing values; every row needs a group id.")
    _validate_binary_partition(y_array, "complete dataset", 1)
    strategy, original_prev, target_prev = _target_class1_prevalence(
        y_array, sampling_config
    )
    min_count = int(split_config.get("min_class_count_per_partition", 2))
    subset_idx, subset_candidates, subset_score = _select_group_subset_indices(
        y_array,
        groups,
        total_n,
        target_prev,
        seed,
        int(split_config.get("candidate_splits", 256)),
        min_count,
    )
    y_subset = y_array[subset_idx]
    groups_subset = groups[subset_idx]
    local_train, local_test, method, candidates, score = _best_group_partition_indices(
        y_subset, groups_subset, train_frac, seed, split_config
    )
    train_idx = subset_idx[local_train]
    test_idx = subset_idx[local_test]
    rng = np.random.default_rng(seed)
    shuffle = bool(split_config.get("shuffle_rows_within_partitions", True))
    train_idx = _shuffle_partition_indices(train_idx, rng, shuffle)
    test_idx = _shuffle_partition_indices(test_idx, rng, shuffle)
    y_train = y_array[train_idx]
    y_test = y_array[test_idx]
    info = _build_split_info(
        y_array, y_train, y_test, train_idx, test_idx, strategy, target_prev,
        total_n, train_frac, seed, "stratified_group", min_count,
        method, candidates, score, groups_all=groups,
    )
    info.update({
        "Group_Subset_Candidates_Evaluated": int(subset_candidates),
        "Group_Subset_Score": float(subset_score),
        "Group_Subset_N_Deviation": int(len(subset_idx) - int(total_n)),
    })
    return (
        _take_rows(X_np, train_idx),
        _take_rows(X_np, test_idx),
        y_train,
        y_test,
        info,
        groups[train_idx],
        groups[test_idx],
    )


def stratified_inner_split(
    X_train, y_train, val_frac=0.2, seed=42, split_config=None, return_info=False
):
    split_config = split_config or {}
    result = exact_sample_split_from_config(
        X_np=X_train,
        y_array=y_train,
        total_n=len(y_train),
        train_frac=1.0 - float(val_frac),
        seed=seed,
        sampling_config={"strategy": "original_prevalence"},
        split_config=split_config,
    )
    X_sub, X_val, y_sub, y_val, info = result
    info["Split_Strategy"] = "inner_stratified"
    return (X_sub, X_val, y_sub, y_val, info) if return_info else result[:4]


def grouped_inner_split(
    X_train,
    y_train,
    groups_train,
    val_frac=0.2,
    seed=42,
    split_config=None,
    return_info=False,
):
    split_config = split_config or {}
    y_train = np.asarray(y_train, dtype=int).ravel()
    groups_train = np.asarray(groups_train).ravel()
    local_sub, local_val, method, candidates, score = _best_group_partition_indices(
        y_train, groups_train, 1.0 - float(val_frac), seed, split_config
    )
    rng = np.random.default_rng(seed)
    shuffle = bool(split_config.get("shuffle_rows_within_partitions", True))
    local_sub = _shuffle_partition_indices(local_sub, rng, shuffle)
    local_val = _shuffle_partition_indices(local_val, rng, shuffle)
    y_sub = y_train[local_sub]
    y_val = y_train[local_val]
    min_count = int(split_config.get("min_class_count_per_partition", 2))
    info = _build_split_info(
        y_train, y_sub, y_val, local_sub, local_val,
        "original_prevalence", float(np.mean(y_train == 1)), len(y_train),
        1.0 - float(val_frac), seed, "inner_stratified_group", min_count,
        method, candidates, score, groups_all=groups_train,
    )
    output = (
        _take_rows(X_train, local_sub),
        _take_rows(X_train, local_val),
        y_sub,
        y_val,
        info,
    )
    return output if return_info else output[:4]


def temporal_sample_split(
    X_data,
    y_array,
    timestamps,
    total_n,
    train_frac,
    seed,
    sampling_config=None,
    split_config=None,
    groups=None,
):
    """Chronological holdout; enabled only when explicitly configured."""
    split_config = split_config or {}
    strategy, original_prev, target_prev = _target_class1_prevalence(
        y_array, sampling_config
    )
    if strategy != "original_prevalence":
        raise ValueError(
            "Temporal splitting supports sampling.strategy='original_prevalence' only; "
            "class rebalancing would alter the chronological sampling design."
        )
    y_array = np.asarray(y_array, dtype=int).ravel()
    timestamps = np.asarray(timestamps).ravel()
    if len(timestamps) != len(y_array) or pd.isna(timestamps).any():
        raise ValueError("timestamps must be complete and aligned one-to-one with X and y.")
    if int(total_n) > len(y_array):
        raise ValueError("total_n exceeds the number of available timestamped rows.")
    order = np.argsort(timestamps, kind="stable")
    window = str(split_config.get("temporal_window", "latest")).lower()
    selected = order[-int(total_n):] if window == "latest" else order[:int(total_n)]
    gap = int(split_config.get("temporal_gap_rows", 0))
    n_train = int(round(int(total_n) * float(train_frac)))
    test_start = n_train + gap
    if n_train <= 0 or test_start >= len(selected):
        raise ValueError("The temporal split and gap leave an empty train or test partition.")
    train_idx = selected[:n_train]
    test_idx = selected[test_start:]
    min_count = int(split_config.get("min_class_count_per_partition", 2))
    y_train = y_array[train_idx]
    y_test = y_array[test_idx]
    groups_for_info = np.asarray(groups) if groups is not None else None
    if groups_for_info is not None:
        overlap = np.intersect1d(
            np.unique(groups_for_info[train_idx]), np.unique(groups_for_info[test_idx])
        )
        if len(overlap) and bool(split_config.get("temporal_enforce_group_disjoint", False)):
            raise ValueError(
                f"Temporal holdout has {len(overlap)} overlapping groups and "
                "temporal_enforce_group_disjoint=True."
            )
    info = _build_split_info(
        y_array, y_train, y_test, train_idx, test_idx, strategy, target_prev,
        total_n, train_frac, seed, "temporal", min_count,
        "stable_chronological_holdout", 1, 0.0,
        groups_all=(groups_for_info if groups_for_info is not None and not len(overlap) else None),
    )
    info.update({
        "Temporal_Window": window,
        "Temporal_Gap_Rows": gap,
        "Temporal_Train_Max": str(np.max(timestamps[train_idx])),
        "Temporal_Test_Min": str(np.min(timestamps[test_idx])),
        "Temporal_Order_Valid": bool(
            np.max(timestamps[train_idx]) <= np.min(timestamps[test_idx])
        ),
        "Temporal_Group_Overlap_N": int(len(overlap)) if groups_for_info is not None else np.nan,
    })
    return (
        _take_rows(X_data, train_idx), _take_rows(X_data, test_idx),
        y_train, y_test, info,
        (groups_for_info[train_idx] if groups_for_info is not None else None),
        (groups_for_info[test_idx] if groups_for_info is not None else None),
    )


def temporal_inner_split(
    X_train,
    y_train,
    timestamps_train,
    val_frac,
    seed,
    split_config=None,
    return_info=False,
):
    split_config = split_config or {}
    y_train = np.asarray(y_train, dtype=int).ravel()
    timestamps_train = np.asarray(timestamps_train).ravel()
    order = np.argsort(timestamps_train, kind="stable")
    n_sub = int(round(len(order) * (1.0 - float(val_frac))))
    gap = int(split_config.get("temporal_gap_rows", 0))
    val_start = n_sub + gap
    if n_sub <= 0 or val_start >= len(order):
        raise ValueError("The inner temporal split and gap leave an empty partition.")
    sub_idx = order[:n_sub]
    val_idx = order[val_start:]
    min_count = int(split_config.get("min_class_count_per_partition", 2))
    y_sub, y_val = y_train[sub_idx], y_train[val_idx]
    info = _build_split_info(
        y_train, y_sub, y_val, sub_idx, val_idx, "original_prevalence",
        float(np.mean(y_train == 1)), len(y_train), 1.0 - float(val_frac),
        seed, "inner_temporal", min_count, "stable_chronological_holdout", 1, 0.0,
    )
    info.update({
        "Temporal_Gap_Rows": gap,
        "Temporal_Train_Max": str(np.max(timestamps_train[sub_idx])),
        "Temporal_Test_Min": str(np.min(timestamps_train[val_idx])),
        "Temporal_Order_Valid": bool(
            np.max(timestamps_train[sub_idx]) <= np.min(timestamps_train[val_idx])
        ),
    })
    output = (
        _take_rows(X_train, sub_idx), _take_rows(X_train, val_idx),
        y_sub, y_val, info,
    )
    return output if return_info else output[:4]


def _resolve_preprocessing_mode(X, preprocessing_config):
    mode = str(preprocessing_config.get("mode", "auto")).lower()
    allowed = {"auto", "preprocessed_numeric", "raw_dataframe"}
    if mode not in allowed:
        raise ValueError(f"preprocessing.mode must be one of {sorted(allowed)}.")
    if mode == "auto":
        if isinstance(X, pd.DataFrame):
            has_non_numeric = any(
                not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes
            )
            has_missing = bool(X.isna().any().any())
            mode = "raw_dataframe" if has_non_numeric or has_missing else "preprocessed_numeric"
        else:
            mode = "preprocessed_numeric"
    if mode == "raw_dataframe" and not isinstance(X, pd.DataFrame):
        raise ValueError(
            "preprocessing.mode='raw_dataframe' requires X to be a pandas DataFrame "
            "so numeric and categorical columns can be identified safely."
        )
    return mode


def _make_one_hot_encoder(preprocessing_config):
    kwargs = {"handle_unknown": "ignore"}
    if preprocessing_config.get("onehot_min_frequency") is not None:
        kwargs["min_frequency"] = preprocessing_config["onehot_min_frequency"]
    if preprocessing_config.get("onehot_max_categories") is not None:
        kwargs["max_categories"] = preprocessing_config["onehot_max_categories"]
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def _build_conventional_preprocessor(training_frame, preprocessing_config):
    numeric_columns = list(training_frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_columns = [c for c in training_frame.columns if c not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            make_pipeline(SimpleImputer(
                strategy=str(preprocessing_config.get("numeric_imputation", "median"))
            )),
            numeric_columns,
        ))
    if categorical_columns:
        if str(preprocessing_config.get("categorical_encoding", "onehot")).lower() != "onehot":
            raise ValueError("V12 currently supports categorical_encoding='onehot'.")
        transformers.append((
            "categorical",
            make_pipeline(
                SimpleImputer(
                    strategy="constant",
                    fill_value=str(preprocessing_config.get(
                        "categorical_imputation_value", "__MISSING__"
                    )),
                ),
                _make_one_hot_encoder(preprocessing_config),
            ),
            categorical_columns,
        ))
    if not transformers:
        raise ValueError("X contains no usable feature columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def _as_finite_float32(values, partition_name):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{partition_name} must be a non-empty two-dimensional matrix.")
    if not np.isfinite(array).all():
        raise ValueError(
            f"{partition_name} contains NaN or infinite values after preprocessing."
        )
    return array


def _prepare_feature_views(
    X_train_raw,
    X_test_raw,
    X_sub_raw,
    X_val_raw,
    preprocessing_mode,
    preprocessing_config,
):
    """Fit outer and inner preprocessing independently on their training rows."""
    started = time.perf_counter()
    if preprocessing_mode == "preprocessed_numeric":
        conventional = tuple(
            _as_finite_float32(value, name)
            for value, name in zip(
                (X_train_raw, X_test_raw, X_sub_raw, X_val_raw),
                ("training data", "test data", "inner-training data", "validation data"),
            )
        )
        tabpfn = conventional
        output_features = int(conventional[0].shape[1])
    else:
        outer = _build_conventional_preprocessor(X_train_raw, preprocessing_config)
        outer_train = _as_finite_float32(
            outer.fit_transform(X_train_raw), "preprocessed training data"
        )
        outer_test = _as_finite_float32(
            outer.transform(X_test_raw), "preprocessed test data"
        )
        inner = _build_conventional_preprocessor(X_sub_raw, preprocessing_config)
        inner_train = _as_finite_float32(
            inner.fit_transform(X_sub_raw), "preprocessed inner-training data"
        )
        inner_val = _as_finite_float32(
            inner.transform(X_val_raw), "preprocessed validation data"
        )
        conventional = (outer_train, outer_test, inner_train, inner_val)
        tabpfn = (X_train_raw, X_test_raw, X_sub_raw, X_val_raw)
        output_features = int(outer_train.shape[1])

    maximum = preprocessing_config.get("max_output_features")
    if maximum is not None and output_features > int(maximum):
        raise ValueError(
            f"Preprocessing produced {output_features} features, above configured "
            f"max_output_features={int(maximum)}."
        )
    metadata = {
        "Preprocessing_Mode": preprocessing_mode,
        "Preprocessing_Fit_Outside_Model_Timing": True,
        "Preprocessing_Time_Seconds": float(time.perf_counter() - started),
        "Preprocessing_Input_Features": int(X_train_raw.shape[1]),
        "Preprocessing_Output_Features": output_features,
        "Preprocessing_Outer_Fit_N": int(len(X_train_raw)),
        "Preprocessing_Inner_Fit_N": int(len(X_sub_raw)),
    }
    return conventional, tabpfn, metadata


def run_v12_data_preparation_self_tests():
    """Fast synthetic checks for split validity, determinism, and preprocessing."""
    split_cfg = {
        "candidate_splits": 64,
        "min_class_count_per_partition": 2,
        "shuffle_rows_within_partitions": True,
    }
    rng = np.random.default_rng(2025)
    X_numeric = rng.normal(size=(300, 5)).astype(np.float32)
    y_numeric = np.tile(np.array([0, 1], dtype=int), 150)

    row_a = exact_sample_split_from_config(
        X_numeric, y_numeric, 200, 0.8, 2025,
        {"strategy": "original_prevalence"}, split_cfg,
    )
    row_b = exact_sample_split_from_config(
        X_numeric, y_numeric, 200, 0.8, 2025,
        {"strategy": "original_prevalence"}, split_cfg,
    )
    assert row_a[4]["Split_Fingerprint"] == row_b[4]["Split_Fingerprint"]
    _validate_binary_partition(row_a[2], "self-test row train", 2)
    _validate_binary_partition(row_a[3], "self-test row test", 2)

    group_sizes = rng.integers(3, 9, size=70)
    groups = np.repeat(np.arange(len(group_sizes)), group_sizes)
    X_group = rng.normal(size=(len(groups), 4)).astype(np.float32)
    y_group = ((np.arange(len(groups)) + groups) % 3 != 0).astype(int)
    grouped = grouped_sample_split(
        X_group, y_group, groups, 260, 0.8, 2025,
        {"strategy": "original_prevalence"}, split_cfg,
    )
    assert len(np.intersect1d(np.unique(grouped[5]), np.unique(grouped[6]))) == 0
    inner = grouped_inner_split(
        grouped[0], grouped[2], grouped[5], 0.2, 2025, split_cfg, True
    )
    assert inner[4]["Train_Test_Group_Overlap_N"] == 0

    temporal = temporal_sample_split(
        X_numeric, y_numeric, np.arange(len(y_numeric)), 200, 0.8, 2025,
        {"strategy": "original_prevalence"}, split_cfg,
    )
    assert temporal[4]["Temporal_Order_Valid"]
    _validate_binary_partition(temporal[2], "self-test temporal train", 2)
    _validate_binary_partition(temporal[3], "self-test temporal test", 2)

    raw = pd.DataFrame({
        "numeric": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "category": ["a", "b", None, "a", "b", "test_only", "a", "b"],
    })
    conventional, tabpfn, metadata = _prepare_feature_views(
        raw.iloc[:5], raw.iloc[5:], raw.iloc[:4], raw.iloc[4:5],
        "raw_dataframe",
        {
            "numeric_imputation": "median",
            "categorical_imputation_value": "__MISSING__",
            "categorical_encoding": "onehot",
        },
    )
    assert all(np.isfinite(view).all() for view in conventional)
    assert conventional[0].shape[1] == conventional[1].shape[1]
    assert isinstance(tabpfn[0], pd.DataFrame)
    assert metadata["Preprocessing_Outer_Fit_N"] == 5

    impossible_groups = np.repeat([0, 1], 20)
    impossible_y = np.repeat([0, 1], 20)
    try:
        grouped_sample_split(
            np.zeros((40, 2)), impossible_y, impossible_groups, 40, 0.8, 2025,
            {"strategy": "original_prevalence"}, split_cfg,
        )
    except ValueError:
        impossible_rejected = True
    else:
        impossible_rejected = False
    assert impossible_rejected, "An impossible group split must fail explicitly."

    return {
        "row_stratification": "passed",
        "group_isolation": "passed",
        "inner_group_isolation": "passed",
        "temporal_order": "passed",
        "train_only_preprocessing": "passed",
        "impossible_split_rejection": "passed",
    }



# =============================================================================
# 4. Hyperparameter optimisation, runtime budgets, and model execution
# =============================================================================

def count_complete_trials(study):
    return sum(
        t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    )


def run_optuna_with_time_budget(
    study,
    objective,
    model_name,
    iteration_num,
    time_budget_seconds,
    max_trials=50
):
    """
    Runs Optuna under a wall-clock runtime budget (set by the reference model).

    Important:
    - The budget applies only to Optuna tuning.
    - A trial that finishes after the budget is NOT eligible for selection.
    - Final model fitting and test prediction happen later and are not time-limited.
    """

    budget = float(time_budget_seconds)
    tune_start = time.perf_counter()
    tune_start_utc = v14_utc_now()
    trials_before = len(study.trials)

    def timed_objective(trial):
        trial_start_elapsed = time.perf_counter() - tune_start
        trial.set_user_attr("trial_start_elapsed_seconds", trial_start_elapsed)

        value = objective(trial)

        trial_end_elapsed = time.perf_counter() - tune_start
        trial.set_user_attr("trial_end_elapsed_seconds", trial_end_elapsed)

        return value

    for _ in range(max_trials):
        elapsed = time.perf_counter() - tune_start

        if elapsed >= budget:
            break

        remaining = budget - elapsed

        study.optimize(
            timed_objective,
            n_trials=1,
            timeout=remaining,
            n_jobs=1
        )

        elapsed = time.perf_counter() - tune_start

        if elapsed >= budget:
            break

    tune_end = time.perf_counter()
    tune_end_utc = v14_utc_now()
    tune_elapsed = tune_end - tune_start

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    eligible_trials = [
        t for t in completed_trials
        if t.user_attrs.get("trial_end_elapsed_seconds", np.inf) <= budget
    ]

    budget_timing_memory.append({
        "Model": model_name,
        "Iteration": iteration_num,
        "Budgeting_Enabled": True,
        "TabPFN_Time_Budget_Seconds": budget,
        "Actual_Optuna_Tuning_Time_Seconds": tune_elapsed,
        "HPO_Start_UTC": tune_start_utc,
        "HPO_End_UTC": tune_end_utc,
        "HPO_Start_Perf_Counter_Seconds": tune_start,
        "HPO_End_Perf_Counter_Seconds": tune_end,
        "HPO_Timer": "time.perf_counter",
        "HPO_Timing_Boundary": "Optuna tuning loop only",
        "Optuna_Tuning_Over_Budget_Seconds": tune_elapsed - budget,
        "Optuna_Tuning_Over_Budget_Flag": tune_elapsed > budget,
        "Total_Trials_Started": len(study.trials),
        "Total_Completed_Trials": len(completed_trials),
        "Eligible_Trials_Within_Budget": len(eligible_trials),
        "Trials_Started_This_Run": len(study.trials) - trials_before,
        "Strict_Budget_Selection_Rule": "Only trials completed before budget are eligible"
    })

    return tune_elapsed


def run_optuna_without_budget(
    study,
    objective,
    model_name,
    iteration_num,
    max_trials=50
):
    """
    Runs Optuna without any runtime budget.
    All completed trials are eligible.
    Selection is based on best validation AUC.
    """

    tune_start = time.perf_counter()
    tune_start_utc = v14_utc_now()
    trials_before = len(study.trials)

    study.optimize(
        objective,
        n_trials=max_trials,
        n_jobs=1
    )

    tune_end = time.perf_counter()
    tune_end_utc = v14_utc_now()
    tune_elapsed = tune_end - tune_start

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    budget_timing_memory.append({
        "Model": model_name,
        "Iteration": iteration_num,
        "Budgeting_Enabled": False,
        "TabPFN_Time_Budget_Seconds": np.nan,
        "Actual_Optuna_Tuning_Time_Seconds": tune_elapsed,
        "HPO_Start_UTC": tune_start_utc,
        "HPO_End_UTC": tune_end_utc,
        "HPO_Start_Perf_Counter_Seconds": tune_start,
        "HPO_End_Perf_Counter_Seconds": tune_end,
        "HPO_Timer": "time.perf_counter",
        "HPO_Timing_Boundary": "Optuna tuning loop only",
        "Optuna_Tuning_Over_Budget_Seconds": np.nan,
        "Optuna_Tuning_Over_Budget_Flag": False,
        "Total_Trials_Started": len(study.trials),
        "Total_Completed_Trials": len(completed_trials),
        "Eligible_Trials_Within_Budget": len(completed_trials),
        "Trials_Started_This_Run": len(study.trials) - trials_before,
        "Strict_Budget_Selection_Rule": "No runtime budget; all completed trials are eligible"
    })

    return tune_elapsed


def get_best_trial_within_budget(study, model_name, iteration_num, budget_seconds):
    """
    Selects the best validation-AUC trial among trials that finished within
    the TabPFN runtime budget.
    """

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    eligible_trials = [
        t for t in completed_trials
        if t.user_attrs.get("trial_end_elapsed_seconds", np.inf) <= budget_seconds
    ]

    if len(eligible_trials) == 0:
        selection_memory.append({
            "Model": model_name,
            "Iteration": iteration_num,
            "Selected_Trial_Number": np.nan,
            "Selection_Method": "Default_fallback_no_trial_completed_within_budget",
            "Selected_AUC": np.nan,
            "Selected_Trial_End_Elapsed_Seconds": np.nan,
            "Total_Completed_Trials": len(completed_trials),
            "Eligible_Trials_Within_Budget": 0,
            "Budget_Seconds": float(budget_seconds)
        })

        return None

    selected_trial = max(eligible_trials, key=lambda t: t.value)

    selection_memory.append({
        "Model": model_name,
        "Iteration": iteration_num,
        "Selected_Trial_Number": selected_trial.number,
        "Selection_Method": "Best_AUC_within_TabPFN_budget",
        "Selected_AUC": float(selected_trial.value),
        "Selected_Trial_End_Elapsed_Seconds": float(
            selected_trial.user_attrs.get("trial_end_elapsed_seconds", np.nan)
        ),
        "Total_Completed_Trials": len(completed_trials),
        "Eligible_Trials_Within_Budget": len(eligible_trials),
        "Budget_Seconds": float(budget_seconds)
    })

    return selected_trial


def get_best_trial_without_budget(study, model_name, iteration_num):
    """
    Selects the best validation-AUC trial among all completed trials.
    Used when budgeting is disabled.
    """

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
    ]

    if len(completed_trials) == 0:
        selection_memory.append({
            "Model": model_name,
            "Iteration": iteration_num,
            "Selected_Trial_Number": np.nan,
            "Selection_Method": "Default_fallback_no_completed_trial",
            "Selected_AUC": np.nan,
            "Selected_Trial_End_Elapsed_Seconds": np.nan,
            "Total_Completed_Trials": 0,
            "Eligible_Trials_Within_Budget": 0,
            "Budget_Seconds": np.nan
        })

        return None

    selected_trial = max(completed_trials, key=lambda t: t.value)

    selection_memory.append({
        "Model": model_name,
        "Iteration": iteration_num,
        "Selected_Trial_Number": selected_trial.number,
        "Selection_Method": "Best_AUC_all_completed_trials_no_budget",
        "Selected_AUC": float(selected_trial.value),
        "Selected_Trial_End_Elapsed_Seconds": np.nan,
        "Total_Completed_Trials": len(completed_trials),
        "Eligible_Trials_Within_Budget": len(completed_trials),
        "Budget_Seconds": np.nan
    })

    return selected_trial


def get_latest_budget_info(model_name, iteration_num):
    matches = [
        row for row in budget_timing_memory
        if row.get("Model") == model_name
        and row.get("Iteration") == iteration_num
        and "Actual_Optuna_Tuning_Time_Seconds" in row
    ]

    if len(matches) == 0:
        return {
            "TabPFN_Time_Budget_Seconds": np.nan,
            "Actual_Optuna_Tuning_Time_Seconds": np.nan,
            "HPO_Start_UTC": None,
            "HPO_End_UTC": None,
            "HPO_Start_Perf_Counter_Seconds": np.nan,
            "HPO_End_Perf_Counter_Seconds": np.nan,
            "HPO_Timer": "time.perf_counter",
            "HPO_Timing_Boundary": "Optuna tuning loop only",
            "Optuna_Tuning_Over_Budget_Seconds": np.nan,
            "Optuna_Tuning_Over_Budget_Flag": np.nan,
            "Total_Trials_Started": np.nan,
            "Total_Completed_Trials": np.nan,
            "Eligible_Trials_Within_Budget": np.nan,
            "Trials_Started_This_Run": np.nan
        }

    return matches[-1]


def get_latest_selection_info(model_name, iteration_num):
    matches = [
        row for row in selection_memory
        if row.get("Model") == model_name
        and row.get("Iteration") == iteration_num
    ]

    if len(matches) == 0:
        return {
            "Selected_Trial_Number": np.nan,
            "Selection_Method": "",
            "Selected_AUC": np.nan,
            "Selected_Trial_End_Elapsed_Seconds": np.nan,
            "Eligible_Trials_Within_Budget": np.nan
        }

    return matches[-1]


def get_latest_final_fit_info(model_name, iteration_num):
    matches = [
        row for row in final_fit_timing_memory
        if row.get("Model") == model_name
        and row.get("Iteration") == iteration_num
    ]

    if len(matches) == 0:
        return {
            "Final_Fit_Predict_Time_Seconds": np.nan,
            "Final_Fit_Time_Seconds": np.nan,
            "Prediction_Time_Seconds": np.nan,
            "Final_Fit_Start_UTC": None,
            "Final_Fit_End_UTC": None,
            "Prediction_Start_UTC": None,
            "Prediction_End_UTC": None,
        }

    return matches[-1]


def v14_record_final_fit_prediction_timing(
    model_name,
    iteration_num,
    fit_start,
    fit_end,
    prediction_start,
    prediction_end,
    fit_start_utc,
    fit_end_utc,
    prediction_start_utc,
    prediction_end_utc,
    final_stage_start=None,
    final_stage_start_utc=None,
):
    stage_start = (
        fit_start if final_stage_start is None else final_stage_start
    )
    record = {
        "Model": model_name,
        "Iteration": iteration_num,
        "Final_Model_Preparation_Time_Seconds": float(
            fit_start - stage_start
        ),
        "Final_Fit_Time_Seconds": float(fit_end - fit_start),
        "Prediction_Time_Seconds": float(
            prediction_end - prediction_start
        ),
        "Final_Fit_Predict_Time_Seconds": float(
            prediction_end - stage_start
        ),
        "Final_Fit_Predict_Start_UTC": (
            final_stage_start_utc or fit_start_utc
        ),
        "Final_Fit_Predict_End_UTC": prediction_end_utc,
        "Final_Fit_Start_UTC": fit_start_utc,
        "Final_Fit_End_UTC": fit_end_utc,
        "Prediction_Start_UTC": prediction_start_utc,
        "Prediction_End_UTC": prediction_end_utc,
        "Final_Fit_Start_Perf_Counter_Seconds": float(fit_start),
        "Final_Fit_End_Perf_Counter_Seconds": float(fit_end),
        "Prediction_Start_Perf_Counter_Seconds": float(
            prediction_start
        ),
        "Prediction_End_Perf_Counter_Seconds": float(prediction_end),
        "Final_Fit_Predict_Start_Perf_Counter_Seconds": float(
            stage_start
        ),
        "Final_Fit_Predict_End_Perf_Counter_Seconds": float(
            prediction_end
        ),
        "Final_Fit_Predict_Timer": "time.perf_counter",
    }
    final_fit_timing_memory.append(record)
    return record


def v14_reference_budget_from_components(
    *,
    reference_is_tuned,
    actual_optuna_tuning_time_seconds=None,
    actual_reference_execution_runtime_seconds=None,
    non_tuned_basis="reference_execution_runtime",
):
    # Tuned references use the dedicated HPO timer; non-tuned references use
    # their configured execution-runtime basis.
    if reference_is_tuned:
        value = actual_optuna_tuning_time_seconds
        basis = "reference_tuning_runtime"
        source = "Actual_Optuna_Tuning_Time_Seconds"
    else:
        normalized_basis = str(
            non_tuned_basis or "reference_execution_runtime"
        ).strip().lower()
        aliases = {
            "reference_full_wall_clock": "reference_execution_runtime",
            "full_wall_clock": "reference_execution_runtime",
            "execution_runtime": "reference_execution_runtime",
        }
        basis = aliases.get(normalized_basis, normalized_basis)
        if basis != "reference_execution_runtime":
            raise ValueError(
                "The preserved non-tuned reference runtime basis must be "
                "'reference_execution_runtime'."
            )
        value = actual_reference_execution_runtime_seconds
        source = "Actual_Total_Runtime_Seconds"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = np.nan
    return {
        "Budget_Basis": basis,
        "Reference_Budget_Seconds": numeric,
        "Reference_Budget_Source_Field": source,
        "Wall_Clock_Definition": (
            "Elapsed real time between the start and end of the measured "
            "operation, analogous to timing it with a stopwatch."
        ),
    }


def save_optuna_trials(study, model_name, param_mapping, iteration_num, budget_seconds=None):
    df = study.trials_dataframe()

    if df.empty or "value" not in df.columns:
        return

    df["Model"] = model_name
    df["Iteration"] = iteration_num
    df["time_seconds"] = df["duration"].dt.total_seconds()
    df = df.rename(columns={"value": "AUC"})
    df = df.rename(columns=param_mapping)

    if budget_seconds is not None and "user_attrs_trial_end_elapsed_seconds" in df.columns:
        df["Finished_Within_TabPFN_Budget"] = (
            df["user_attrs_trial_end_elapsed_seconds"] <= float(budget_seconds)
        )

    user_attr_cols = [c for c in df.columns if c.startswith("user_attrs_")]

    cols_to_keep = (
        ["number", "Iteration", "Model", "AUC", "time_seconds"]
        + list(param_mapping.values())
        + user_attr_cols
        + ["Finished_Within_TabPFN_Budget"]
    )

    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]

    if model_name not in optuna_trials_memory:
        optuna_trials_memory[model_name] = []

    optuna_trials_memory[model_name].append(df)


def create_budgeted_summary(errors, model_names, scenario_name=""):
    rows = []

    for m in model_names:
        if len(errors.get(f"{m}_AUC", [])) == 0:
            continue

        rows.append({
            "Scenario": scenario_name,
            "Model": m,

            "BalancedAccuracy_Mean": safe_mean(errors[f"{m}_BalancedAccuracy"]),
            "BalancedAccuracy_SD": safe_sd(errors[f"{m}_BalancedAccuracy"]),
            "BalancedAccuracy_CI95_pm": get_ci(errors[f"{m}_BalancedAccuracy"]),

            "AUC_Mean": safe_mean(errors[f"{m}_AUC"]),
            "AUC_SD": safe_sd(errors[f"{m}_AUC"]),
            "AUC_CI95_pm": get_ci(errors[f"{m}_AUC"]),

            "Brier_Mean": safe_mean(errors[f"{m}_Brier"]),
            "Brier_SD": safe_sd(errors[f"{m}_Brier"]),
            "Brier_CI95_pm": get_ci(errors[f"{m}_Brier"]),

            "Sensitivity_Mean": safe_mean(errors[f"{m}_Sensitivity"]),
            "Sensitivity_SD": safe_sd(errors[f"{m}_Sensitivity"]),
            "Sensitivity_CI95_pm": get_ci(errors[f"{m}_Sensitivity"]),

            "Precision_Mean": safe_mean(errors[f"{m}_Precision"]),
            "Precision_SD": safe_sd(errors[f"{m}_Precision"]),
            "Precision_CI95_pm": get_ci(errors[f"{m}_Precision"]),

            "Total_Runtime_Mean": safe_mean(errors[f"{m}_TotalRuntime"]),
            "Total_Runtime_SD": safe_sd(errors[f"{m}_TotalRuntime"]),
            "Total_Runtime_CI95_pm": get_ci(errors[f"{m}_TotalRuntime"]),

            "Actual_Total_Runtime_Mean": safe_mean(errors[f"{m}_ActualTotalRuntime"]),
            "Actual_Total_Runtime_SD": safe_sd(errors[f"{m}_ActualTotalRuntime"]),
            "Actual_Total_Runtime_CI95_pm": get_ci(errors[f"{m}_ActualTotalRuntime"]),

            "Budgeted_Total_Runtime_Mean": safe_mean(errors[f"{m}_BudgetedTotalRuntime"]),
            "Budgeted_Total_Runtime_SD": safe_sd(errors[f"{m}_BudgetedTotalRuntime"]),
            "Budgeted_Total_Runtime_CI95_pm": get_ci(errors[f"{m}_BudgetedTotalRuntime"]),


            "Actual_Optuna_Tuning_Time_Mean": safe_mean(errors[f"{m}_ActualOptunaTuningTime"]),
            "Actual_Optuna_Tuning_Time_SD": safe_sd(errors[f"{m}_ActualOptunaTuningTime"]),

            "Optuna_Tuning_Time_Capped_Mean": safe_mean(errors[f"{m}_OptunaTuningTimeCapped"]),
            "Optuna_Tuning_Time_Capped_SD": safe_sd(errors[f"{m}_OptunaTuningTimeCapped"]),


            "Optuna_Completed_Trials_Mean": safe_mean(errors[f"{m}_CompletedTrials"]),
            "Optuna_Completed_Trials_SD": safe_sd(errors[f"{m}_CompletedTrials"]),

            "Eligible_Trials_Within_Budget_Mean": safe_mean(errors[f"{m}_EligibleTrials"]),
            "Eligible_Trials_Within_Budget_SD": safe_sd(errors[f"{m}_EligibleTrials"]),


            "Selected_AUC_Mean": safe_mean(errors[f"{m}_SelectedAUC"]),
            "Selected_AUC_SD": safe_sd(errors[f"{m}_SelectedAUC"]),

            "Final_Fit_Predict_Time_Mean": safe_mean(errors[f"{m}_FinalFitPredictTime"]),
            "Final_Fit_Predict_Time_SD": safe_sd(errors[f"{m}_FinalFitPredictTime"]),

            "Energy_Mean": safe_mean(errors[f"{m}_Energy"]),
            "Energy_SD": safe_sd(errors[f"{m}_Energy"]),
            "Energy_CI95_pm": get_ci(errors[f"{m}_Energy"]),

            "StrictBudget_BalancedAccuracy_Mean": safe_mean(errors[f"{m}_StrictBudgetBalancedAccuracy"]),
            "StrictBudget_BalancedAccuracy_SD": safe_sd(errors[f"{m}_StrictBudgetBalancedAccuracy"]),
            "StrictBudget_BalancedAccuracy_CI95_pm": get_ci(errors[f"{m}_StrictBudgetBalancedAccuracy"]),
            "StrictBudget_AUC_Mean": safe_mean(errors[f"{m}_StrictBudgetAUC"]),
            "StrictBudget_AUC_SD": safe_sd(errors[f"{m}_StrictBudgetAUC"]),
            "StrictBudget_AUC_CI95_pm": get_ci(errors[f"{m}_StrictBudgetAUC"]),
            "StrictBudget_Brier_Mean": safe_mean(errors[f"{m}_StrictBudgetBrier"]),
            "StrictBudget_Brier_SD": safe_sd(errors[f"{m}_StrictBudgetBrier"]),
            "StrictBudget_Brier_CI95_pm": get_ci(errors[f"{m}_StrictBudgetBrier"]),
            "StrictBudget_Sensitivity_Mean": safe_mean(errors[f"{m}_StrictBudgetSensitivity"]),
            "StrictBudget_Sensitivity_SD": safe_sd(errors[f"{m}_StrictBudgetSensitivity"]),
            "StrictBudget_Sensitivity_CI95_pm": get_ci(errors[f"{m}_StrictBudgetSensitivity"]),
            "StrictBudget_Precision_Mean": safe_mean(errors[f"{m}_StrictBudgetPrecision"]),
            "StrictBudget_Precision_SD": safe_sd(errors[f"{m}_StrictBudgetPrecision"]),
            "StrictBudget_Precision_CI95_pm": get_ci(errors[f"{m}_StrictBudgetPrecision"]),

            "PredictionBudget_BalancedAccuracy_Mean": safe_mean(errors[f"{m}_PredictionBudgetBalancedAccuracy"]),
            "PredictionBudget_BalancedAccuracy_SD": safe_sd(errors[f"{m}_PredictionBudgetBalancedAccuracy"]),
            "PredictionBudget_BalancedAccuracy_CI95_pm": get_ci(errors[f"{m}_PredictionBudgetBalancedAccuracy"]),
            "PredictionBudget_AUC_Mean": safe_mean(errors[f"{m}_PredictionBudgetAUC"]),
            "PredictionBudget_AUC_SD": safe_sd(errors[f"{m}_PredictionBudgetAUC"]),
            "PredictionBudget_AUC_CI95_pm": get_ci(errors[f"{m}_PredictionBudgetAUC"]),
            "PredictionBudget_Brier_Mean": safe_mean(errors[f"{m}_PredictionBudgetBrier"]),
            "PredictionBudget_Brier_SD": safe_sd(errors[f"{m}_PredictionBudgetBrier"]),
            "PredictionBudget_Brier_CI95_pm": get_ci(errors[f"{m}_PredictionBudgetBrier"]),
            "PredictionBudget_Sensitivity_Mean": safe_mean(errors[f"{m}_PredictionBudgetSensitivity"]),
            "PredictionBudget_Sensitivity_SD": safe_sd(errors[f"{m}_PredictionBudgetSensitivity"]),
            "PredictionBudget_Sensitivity_CI95_pm": get_ci(errors[f"{m}_PredictionBudgetSensitivity"]),
            "PredictionBudget_Precision_Mean": safe_mean(errors[f"{m}_PredictionBudgetPrecision"]),
            "PredictionBudget_Precision_SD": safe_sd(errors[f"{m}_PredictionBudgetPrecision"]),
            "PredictionBudget_Precision_CI95_pm": get_ci(errors[f"{m}_PredictionBudgetPrecision"]),

            "TabPFN_Fit_Time_Mean": safe_mean(errors[f"{m}_TabPFNFitTime"]),
            "TabPFN_Fit_Time_SD": safe_sd(errors[f"{m}_TabPFNFitTime"]),
            "TabPFN_PredictProba_Time_Mean": safe_mean(errors[f"{m}_TabPFNPredictProbaTime"]),
            "TabPFN_PredictProba_Time_SD": safe_sd(errors[f"{m}_TabPFNPredictProbaTime"]),
            "TabPFN_EndToEnd_FitPlusPredictProba_Time_Mean": safe_mean(errors[f"{m}_TabPFNEndToEndFitPredictProbaTime"]),
            "TabPFN_EndToEnd_FitPlusPredictProba_Time_SD": safe_sd(errors[f"{m}_TabPFNEndToEndFitPredictProbaTime"]),
            "TabPFN_Strict_EndToEnd_Feasible_Rate": safe_mean(errors[f"{m}_TabPFNStrictEndToEndBudgetPassed"]),
            "TabPFN_Strict_EndToEnd_N_Feasible": safe_sum(errors[f"{m}_TabPFNStrictEndToEndBudgetPassed"]),
            "TabPFN_PredictProba_Feasible_Rate": safe_mean(errors[f"{m}_TabPFNPredictProbaBudgetPassed"]),
            "TabPFN_PredictProba_N_Feasible": safe_sum(errors[f"{m}_TabPFNPredictProbaBudgetPassed"]),
            "TabPFN_Strict_EndToEnd_Overrun_Mean": safe_mean(errors[f"{m}_TabPFNStrictEndToEndOverrun"]),
            "TabPFN_PredictProba_Overrun_Mean": safe_mean(errors[f"{m}_TabPFNPredictProbaOverrun"]),
            "TabPFN_Full_Train_N_Mean": safe_mean(errors[f"{m}_TabPFNFullTrainN"]),
            "TabPFN_Budgeted_Context_Sample_Size_Mean": safe_mean(errors[f"{m}_TabPFNBudgetedContextSampleSize"]),
            "TabPFN_Budgeted_Context_Sample_Size_SD": safe_sd(errors[f"{m}_TabPFNBudgetedContextSampleSize"]),
            "TabPFN_Context_N_Used_Mean": safe_mean(errors[f"{m}_TabPFNContextNUsed"]),
            "TabPFN_Context_N_Used_SD": safe_sd(errors[f"{m}_TabPFNContextNUsed"]),
            "TabPFN_Context_Fraction_Used_Mean": safe_mean(errors[f"{m}_TabPFNContextFractionUsed"]),
            "TabPFN_Context_Search_Attempts_Mean": safe_mean(errors[f"{m}_TabPFNContextSearchAttempts"]),
            "TabPFN_Full_Prediction_N_Mean": safe_mean(errors[f"{m}_TabPFNFullPredictionN"]),
            "TabPFN_Budgeted_Prediction_Sample_Size_Mean": safe_mean(errors[f"{m}_TabPFNBudgetedPredictionSampleSize"]),
            "TabPFN_Budgeted_Prediction_Sample_Size_SD": safe_sd(errors[f"{m}_TabPFNBudgetedPredictionSampleSize"]),
            "TabPFN_Prediction_Fraction_Used_Mean": safe_mean(errors[f"{m}_TabPFNPredictionFractionUsed"]),
            "TabPFN_Prediction_Search_Attempts_Mean": safe_mean(errors[f"{m}_TabPFNPredictionSearchAttempts"]),
            "TabPFN_PredictProba_Budget_Delta_Mean": safe_mean(errors[f"{m}_TabPFNPredictProbaBudgetDelta"]),
            "TabPFN_PredictProba_Budget_Remaining_Mean": safe_mean(errors[f"{m}_TabPFNPredictProbaBudgetRemaining"]),
            "TabPFN_PredictProba_Budget_Use_Ratio_Mean": safe_mean(errors[f"{m}_TabPFNPredictProbaBudgetUseRatio"]),
            "TabPFN_Total_Runtime_Budget_Delta_Mean": safe_mean(errors[f"{m}_TabPFNTotalRuntimeBudgetDelta"]),
            "TabPFN_Total_Runtime_Budget_Remaining_Mean": safe_mean(errors[f"{m}_TabPFNTotalRuntimeBudgetRemaining"]),
            "TabPFN_ML_Reference_Budget_Mean": safe_mean(errors[f"{m}_TabPFNMLReferenceBudget"]),
            "TabPFN_Effective_Time_Budget_Mean": safe_mean(errors[f"{m}_TabPFNEffectiveTimeBudget"]),
            "TabPFN_Effective_Budget_Multiplier_Mean": safe_mean(errors[f"{m}_TabPFNEffectiveBudgetMultiplier"]),
            "TabPFN_Min_Context_Requested_Mean": safe_mean(errors[f"{m}_TabPFNMinContextRequested"]),
            "TabPFN_Min_Context_Target_Mean": safe_mean(errors[f"{m}_TabPFNMinContextTarget"]),
            "TabPFN_Min_Context_Runtime_Mean": safe_mean(errors[f"{m}_TabPFNMinContextRuntime"]),
            "TabPFN_Min_Context_Budget_Multiplier_Applied_Rate": safe_mean(errors[f"{m}_TabPFNMinContextBudgetMultiplierApplied"]),
            "TabPFN_Min_Context_Requirement_Met_Rate": safe_mean(errors[f"{m}_TabPFNMinContextRequirementMet"]),
            "TabPFN_Original_ML_Budget_Total_Runtime_Delta_Mean": safe_mean(errors[f"{m}_TabPFNOriginalMLBudgetTotalRuntimeDelta"]),
            "TabPFN_Original_ML_Budget_Total_Runtime_Remaining_Mean": safe_mean(errors[f"{m}_TabPFNOriginalMLBudgetTotalRuntimeRemaining"]),

            "N_Successful_Iterations": len(errors[f"{m}_AUC"])
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("AUC_Mean", ascending=False)


def reset_runtime_memory():
    global optuna_trials_memory
    global budget_timing_memory
    global selection_memory
    global final_fit_timing_memory
    global tabpfn_budget_evaluation_memory

    optuna_trials_memory = {}
    budget_timing_memory = []
    selection_memory = []
    final_fit_timing_memory = []
    tabpfn_budget_evaluation_memory = []



import copy

ACTIVE_MODEL_CONFIGS = None
ACTIVE_SCENARIO_NAME = None

ACTIVE_BUDGET_REFERENCE_MODEL = None


def set_active_budget_reference(model_name):
    """Set the budget-reference model for the current scenario.

    The reference model establishes the runtime budget; every other tuned model
    is capped to it. See run_budgeted_study() for how this flag is consumed.
    """
    global ACTIVE_BUDGET_REFERENCE_MODEL
    ACTIVE_BUDGET_REFERENCE_MODEL = model_name


def deep_update(base, updates):
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_model_override(model_cfg, override):
    override = copy.deepcopy(override)

    replace_search_space = override.pop("replace_search_space", False)

    remove_search_params = override.pop("remove_search_params", [])

    replace_fixed_params = override.pop("replace_fixed_params", False)

    if replace_search_space:
        model_cfg["search_space"] = copy.deepcopy(
            override.pop("search_space", {})
        )

    if replace_fixed_params:
        model_cfg["fixed_params"] = copy.deepcopy(
            override.pop("fixed_params", {})
        )

    deep_update(model_cfg, override)

    for p in remove_search_params:
        model_cfg.get("search_space", {}).pop(p, None)
        model_cfg.get("default_params", {}).pop(p, None)

    return model_cfg


def set_active_scenario(scenario_name):
    global ACTIVE_MODEL_CONFIGS
    global ACTIVE_SCENARIO_NAME

    ACTIVE_SCENARIO_NAME = scenario_name
    ACTIVE_MODEL_CONFIGS = copy.deepcopy(MODEL_CONFIGS)

    scenario_cfg = SCENARIO_CONFIGS.get(scenario_name, {})

    if "enabled_models" in scenario_cfg:
        enabled_models = set(scenario_cfg["enabled_models"])

        for model_name in ACTIVE_MODEL_CONFIGS:
            ACTIVE_MODEL_CONFIGS[model_name]["enabled"] = model_name in enabled_models

    model_overrides = scenario_cfg.get("model_overrides", {})

    for model_name, overrides in model_overrides.items():
        if model_name not in ACTIVE_MODEL_CONFIGS:
            raise ValueError(f"{model_name} is not defined in MODEL_CONFIGS.")

        ACTIVE_MODEL_CONFIGS[model_name] = apply_model_override(
            ACTIVE_MODEL_CONFIGS[model_name],
            overrides
        )

    validate_model_configs(ACTIVE_MODEL_CONFIGS)

    print(f"Active scenario loaded: {scenario_name}")



def validate_model_configs(model_configs):
    for model_name, cfg in model_configs.items():

        if not cfg.get("enabled", False):
            continue

        fixed_keys = set(cfg.get("fixed_params", {}).keys())
        search_keys = set(cfg.get("search_space", {}).keys())
        default_keys = set(cfg.get("default_params", {}).keys())

        overlap = fixed_keys.intersection(search_keys)
        assert len(overlap) == 0, (
            f"{model_name}: parameters cannot be in both fixed_params and search_space: {overlap}"
        )

        missing_defaults = search_keys.difference(default_keys)
        assert len(missing_defaults) == 0, (
            f"{model_name}: tuned parameters missing from default_params: {missing_defaults}"
        )

    print("Model configuration check passed")



def get_model_cfg(model_name):
    if ACTIVE_MODEL_CONFIGS is not None:
        return ACTIVE_MODEL_CONFIGS[model_name]

    return MODEL_CONFIGS[model_name]


def get_seed(iteration_num):
    return RUN_CONFIG.get("base_seed", 2025) + iteration_num


def suggest_params_from_config(trial, model_name):
    search_space = get_model_cfg(model_name).get("search_space", {})
    params = {}

    for param_name, spec in search_space.items():
        ptype = spec["type"]

        if ptype == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )

        elif ptype == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )

        elif ptype == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name,
                spec["choices"]
            )

        else:
            raise ValueError(f"Unknown hyperparameter type for {param_name}: {ptype}")

    return params


def make_param_mapping(model_name):
    search_space = get_model_cfg(model_name).get("search_space", {})
    return {
        f"params_{p}": p.replace("_", " ").title()
        for p in search_space.keys()
    }


def make_study(iteration_num):
    return optuna.create_study(
        direction=RUN_CONFIG.get("optuna_direction", "maximize"),
        sampler=optuna.samplers.TPESampler(seed=get_seed(iteration_num)),
        pruner=optuna.pruners.NopPruner()
    )


def run_budgeted_study(
    model_name,
    objective,
    iteration_num,
    time_budget_seconds
):
    cfg = get_model_cfg(model_name)

    study = make_study(iteration_num)

    max_trials = cfg.get(
        "max_trials",
        RUN_CONFIG.get("default_max_trials", 50)
    )

    budgeting_cfg = RUN_CONFIG.get("budgeting", {})
    scenario_cfg = (
        SCENARIO_CONFIGS.get(ACTIVE_SCENARIO_NAME, {})
        if isinstance(SCENARIO_CONFIGS, dict)
        else {}
    )
    budgeting_enabled = bool(
        scenario_cfg.get("budgeting_enabled", budgeting_cfg.get("enabled", True))
    )

    is_reference = (model_name == ACTIVE_BUDGET_REFERENCE_MODEL)
    run_capped = budgeting_enabled and (not is_reference)

    if run_capped:
        run_optuna_with_time_budget(
            study=study,
            objective=objective,
            model_name=model_name,
            iteration_num=iteration_num,
            time_budget_seconds=time_budget_seconds,
            max_trials=max_trials
        )

        selected_trial = get_best_trial_within_budget(
            study,
            model_name,
            iteration_num,
            time_budget_seconds
        )

        budget_seconds_for_saving = time_budget_seconds

    else:
        run_optuna_without_budget(
            study=study,
            objective=objective,
            model_name=model_name,
            iteration_num=iteration_num,
            max_trials=max_trials
        )

        selected_trial = get_best_trial_without_budget(
            study,
            model_name,
            iteration_num
        )

        budget_seconds_for_saving = None

    optimal_params = (
        selected_trial.params
        if selected_trial is not None
        else cfg.get("default_params", {})
    )

    save_optuna_trials(
        study,
        model_name,
        make_param_mapping(model_name),
        iteration_num,
        budget_seconds=budget_seconds_for_saving
    )

    return optimal_params, selected_trial


def start_energy_tracker(project_name):
    if not RUN_CONFIG.get("track_energy", True):
        return None

    tracker = EmissionsTracker(
        project_name=project_name,
        save_to_file=False,
        log_level="error",
        tracking_mode=RUN_CONFIG.get("energy_tracking_mode", "process")
    )
    tracker.start()
    return tracker


def stop_energy_tracker(tracker):
    if tracker is None:
        return np.nan
    return stop_tracker_get_energy(tracker)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _merge_dicts(*values):
    merged = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _resolve_auto_tabpfn_model_version(version_name):
    """Resolve optional AutoTabPFN model version names.

    AutoTabPFNClassifier currently supports the extension model versions V2 and
    V2_5. It does not expose a V3 max_time path. Leaving this unset uses the
    library default, which is V2_5 in current tabpfn-extensions releases.
    """
    if version_name in (None, "", "default", "auto"):
        return None
    if AutoTabPFNModelVersion is None:
        raise ImportError(
            "Could not import TabPFN ModelVersion; remove auto_tabpfn.model_version "
            "or install compatible tabpfn/tabpfn-extensions packages."
        )
    try:
        if isinstance(version_name, AutoTabPFNModelVersion):
            return version_name
    except TypeError:
        pass

    normalized = str(version_name).strip().upper().replace(".", "_").replace("-", "_")
    aliases = {
        "V2": "V2",
        "V25": "V2_5",
        "V2_5": "V2_5",
        "V2_50": "V2_5",
    }
    member_name = aliases.get(normalized, normalized)
    if hasattr(AutoTabPFNModelVersion, member_name):
        return getattr(AutoTabPFNModelVersion, member_name)

    for item in AutoTabPFNModelVersion:
        item_value = str(getattr(item, "value", item)).upper().replace(".", "_").replace("-", "_")
        if item_value == normalized:
            return item

    raise ValueError(
        f"Unsupported AutoTabPFN model_version={version_name!r}. "
        "Use None/default, 'V2', or 'V2_5'. AutoTabPFN does not provide a V3 max_time path."
    )


def _resolve_local_tabpfn_model_version(version_name):
    """Resolve an optional model version for plain local TabPFN only."""
    if version_name in (None, "", "default", "auto"):
        return None
    if AutoTabPFNModelVersion is None:
        raise ImportError(
            "Could not import tabpfn ModelVersion. Use local model_version='auto' "
            "or install a compatible tabpfn package."
        )

    try:
        if isinstance(version_name, AutoTabPFNModelVersion):
            return version_name
    except TypeError:
        pass

    normalized = str(version_name).strip().upper().replace(".", "_").replace("-", "_")
    if normalized.startswith("MODELVERSION_"):
        normalized = normalized.removeprefix("MODELVERSION_")
    aliases = {
        "V2": "V2",
        "V25": "V2_5",
        "V2_5": "V2_5",
        "V26": "V2_6",
        "V2_6": "V2_6",
        "V3": "V3",
        "3": "V3",
    }
    member_name = aliases.get(normalized, normalized)
    if hasattr(AutoTabPFNModelVersion, member_name):
        return getattr(AutoTabPFNModelVersion, member_name)

    available = [getattr(item, "name", str(item)) for item in AutoTabPFNModelVersion]
    raise ValueError(
        f"Unsupported local TabPFN model_version={version_name!r}. "
        f"Use 'auto' or one of {available}."
    )


def _tabpfn_model_version_label(model_version):
    if model_version is None:
        return "auto"
    return str(getattr(model_version, "value", getattr(model_version, "name", model_version)))


def _get_budgeted_tabpfn_device(cfg):
    execution_cfg = _as_dict(cfg.get("execution", {}))
    dev_pref = str(
        execution_cfg.get("local_device", cfg.get("tabpfn_device", "auto"))
    ).lower()
    if dev_pref not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            "models.TabPFN.execution.local_device must be 'cpu', 'cuda', or 'auto'."
        )
    if dev_pref == "cpu":
        return "cpu"
    if dev_pref == "cuda":
        if (
            bool(execution_cfg.get("require_requested_device", False))
            and not cuda_available()
        ):
            raise RuntimeError(
                "Local TabPFN requested CUDA with require_requested_device=true, "
                "but torch.cuda.is_available() is false."
            )
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def _build_auto_tabpfn_kwargs(cfg, iteration_num, max_time, device):
    auto_cfg = _as_dict(cfg.get("auto_tabpfn", {}))

    debug_root = auto_cfg.get(
        "debug_root",
        RUN_CONFIG.get("auto_tabpfn_debug_root", None) if RUN_CONFIG is not None else None
    )
    if debug_root is None:
        debug_root = os.path.join(
            RUN_CONFIG.get("output_root", ".") if RUN_CONFIG is not None else ".",
            "AutoTabPFN_Debug"
        )
    default_path = os.path.join(debug_root, f"iter_{iteration_num}")

    default_init_args = {
        "verbosity": auto_cfg.get("verbosity", 3),
        "path": auto_cfg.get("path", default_path),
    }
    default_fit_args = {
        "num_bag_folds": 0,
        "fit_weighted_ensemble": False,
    }

    phe_init_args = _merge_dicts(default_init_args, cfg.get("phe_init_args"), auto_cfg.get("phe_init_args"))
    phe_fit_args = _merge_dicts(default_fit_args, cfg.get("phe_fit_args"), auto_cfg.get("phe_fit_args"))

    kwargs = {
        "max_time": max_time,
        "device": device,
        "random_state": get_seed(iteration_num),
        "ignore_pretraining_limits": cfg.get("ignore_pretraining_limits", True),
        "n_ensemble_models": int(auto_cfg.get("n_ensemble_models", cfg.get("n_ensemble_models", 1))),
        "n_estimators": int(auto_cfg.get("n_estimators", cfg.get("n_estimators", 1))),
        "eval_metric": auto_cfg.get("eval_metric", cfg.get("eval_metric", "roc_auc")),
        "presets": auto_cfg.get("presets", cfg.get("presets", None)),
        "phe_init_args": phe_init_args,
        "phe_fit_args": phe_fit_args,
    }

    model_version = _resolve_auto_tabpfn_model_version(auto_cfg.get("model_version", cfg.get("auto_tabpfn_model_version", None)))
    if model_version is not None:
        kwargs["model_version"] = model_version

    return kwargs


def _run_plain_local_tabpfn_fallback(X_train, X_test, y_train, cfg, iteration_num, device):
    if LocalTabPFNClassifier is None:
        raise ImportError(
            "Plain local tabpfn.TabPFNClassifier is unavailable; install tabpfn "
            "or disable auto_tabpfn.fallback_to_plain_local."
        )

    local_model = LocalTabPFNClassifier(
        model_path=cfg.get("local_model_path", "auto"),
        device=device,
        ignore_pretraining_limits=cfg.get("ignore_pretraining_limits", True),
        n_estimators=int(cfg.get("local_n_estimators", 8)),
        random_state=get_seed(iteration_num),
        show_progress_bar=bool(cfg.get("local_show_progress_bar", False)),
    )
    local_model.fit(X_train, y_train)
    pred, proba = predict_binary_argmax(local_model, X_test)
    return pred, proba


def _tabpfn_context_candidates(full_n, local_cfg):
    """Return optional user-provided context sizes.

    By default this returns an empty list. The normal third-scenario TabPFN path
    now uses measured runtime feedback instead of a pre-baked context grid.
    """
    full_n = int(full_n)
    if full_n <= 0:
        return []

    explicit = local_cfg.get("context_candidates", None)
    if explicit is None:
        explicit = local_cfg.get("adaptive_context_sizes", None)

    if explicit is None:
        return []

    candidates = []
    for item in explicit:
        if item is None:
            continue
        if isinstance(item, str) and item.strip().lower() in {"all", "full"}:
            candidates.append(full_n)
            continue
        value = float(item)
        if 0 < value < 1:
            candidates.append(int(np.ceil(value * full_n)))
        elif value > 0:
            candidates.append(int(value))

    candidates.append(full_n)
    candidates = sorted({min(int(x), full_n) for x in candidates if int(x) > 0})
    return candidates


def _tabpfn_data_driven_start_context(X_train, y_train):
    """Smallest practical context derived from the current data shape."""
    y_arr = np.asarray(y_train)
    full_n = len(y_arr)
    if full_n <= 0:
        return 0
    n_classes = max(1, len(np.unique(y_arr)))
    n_features = int(X_train.shape[1]) if hasattr(X_train, "shape") and len(X_train.shape) > 1 else 1
    return int(min(full_n, max(n_classes, n_features + n_classes)))


def _stratified_context_subset(X_train, y_train, context_n, seed):
    """Choose a deterministic stratified row subset for local TabPFN context."""
    y_arr = np.asarray(y_train)
    full_n = len(y_arr)
    context_n = int(min(max(1, context_n), full_n))
    if context_n >= full_n:
        return X_train, y_train, np.arange(full_n)

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_arr, return_counts=True)
    n_classes = len(classes)
    if context_n < n_classes:
        context_n = n_classes

    allocations = {}
    remaining = context_n
    for cls, count in zip(classes, counts):
        alloc = max(1, int(round(context_n * (count / full_n))))
        alloc = min(alloc, int(count))
        allocations[cls] = alloc
        remaining -= alloc

    while remaining != 0:
        if remaining > 0:
            room = [
                cls for cls, count in zip(classes, counts)
                if allocations[cls] < int(count)
            ]
            if not room:
                break
            cls = max(room, key=lambda c: int(counts[np.where(classes == c)[0][0]]) - allocations[c])
            allocations[cls] += 1
            remaining -= 1
        else:
            removable = [cls for cls in classes if allocations[cls] > 1]
            if not removable:
                break
            cls = max(removable, key=lambda c: allocations[c])
            allocations[cls] -= 1
            remaining += 1

    selected = []
    for cls in classes:
        idx = np.flatnonzero(y_arr == cls)
        take = min(allocations[cls], len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())

    selected = np.array(sorted(selected), dtype=int)
    return _take_rows(X_train, selected), y_arr[selected], selected


def _make_local_tabpfn(local_kwargs, model_version=None):
    local_kwargs = dict(local_kwargs)

    if model_version is None:
        make_model = lambda kwargs: LocalTabPFNClassifier(**kwargs)
    else:
        create_for_version = getattr(
            LocalTabPFNClassifier,
            "create_default_for_version",
            None,
        )
        if not callable(create_for_version):
            raise RuntimeError(
                "This tabpfn package cannot force a local model version. "
                "Upgrade tabpfn or set local model_version='auto'."
            )
        local_kwargs.pop("model_path", None)
        make_model = lambda kwargs: create_for_version(model_version, **kwargs)

    try:
        return make_model(local_kwargs)
    except TypeError as exc:
        message = str(exc)
        if (
            "auto_scale_n_estimators" not in message
            and "show_progress_bar" not in message
        ):
            raise
        local_kwargs = dict(local_kwargs)
        local_kwargs.pop("auto_scale_n_estimators", None)
        local_kwargs.pop("show_progress_bar", None)
        return make_model(local_kwargs)



def run_tabpfn(X_train, X_test, y_train, y_test, iteration_num,
               time_budget_seconds=None):
    model_name = "TabPFN"
    cfg = get_model_cfg(model_name)
    execution_cfg = _as_dict(cfg.get("execution", {}))
    requested_execution_path = str(
        execution_cfg.get("path", execution_cfg.get("execution_path", "auto"))
    ).lower()
    requested_execution_path = {
        "client": "cloud",
        "cloud_client": "cloud",
        "cloud/client": "cloud",
        "local_gpu": "local",
        "local_cpu": "local",
    }.get(requested_execution_path, requested_execution_path)
    if requested_execution_path not in {"auto", "local", "cloud"}:
        raise ValueError(
            "models.TabPFN.execution.path must be 'auto', 'local', or 'cloud'."
        )

    finite_time_budget = (
        time_budget_seconds is not None
        and np.isfinite(time_budget_seconds)
        and time_budget_seconds > 0
    )
    use_time_budget = finite_time_budget and requested_execution_path != "cloud"
    use_unbudgeted_local = (
        requested_execution_path == "local" and not finite_time_budget
    )

    tabpfn_budget_meta = {}
    tracker = start_energy_tracker(f"{model_name}_iter{iteration_num}")

    try:
        if use_unbudgeted_local:
            if LocalTabPFNClassifier is None:
                raise ImportError(
                    "Local tabpfn.TabPFNClassifier is unavailable; install 'tabpfn' "
                    "or configure models.TabPFN.execution.path='cloud'."
                )
            default_tabpfn_cfg = _as_dict(
                _as_dict(globals().get("CONFIG", {}).get("models", {})).get(
                    "TabPFN", {}
                )
            )
            default_local_cfg = _as_dict(
                default_tabpfn_cfg.get("local_tabpfn_budget", {})
            )
            requested_local_cfg = _as_dict(
                cfg.get("local_tabpfn_budget", cfg.get("local_tabpfn", {}))
            )
            local_cfg = _merge_dicts(default_local_cfg, requested_local_cfg)
            device = _get_budgeted_tabpfn_device(cfg)
            local_model_version = _resolve_local_tabpfn_model_version(
                local_cfg.get("model_version", "auto")
            )
            local_model_version_label = _tabpfn_model_version_label(
                local_model_version
            )
            local_kwargs = {
                "model_path": local_cfg.get(
                    "model_path", cfg.get("local_model_path", "auto")
                ),
                "device": device,
                "ignore_pretraining_limits": local_cfg.get(
                    "ignore_pretraining_limits",
                    cfg.get("ignore_pretraining_limits", True),
                ),
                "n_estimators": int(
                    local_cfg.get(
                        "n_estimators", cfg.get("local_n_estimators", 1)
                    )
                ),
                "auto_scale_n_estimators": bool(
                    local_cfg.get("auto_scale_n_estimators", False)
                ),
                "fit_mode": local_cfg.get(
                    "fit_mode", cfg.get("local_fit_mode", "fit_preprocessors")
                ),
                "random_state": get_seed(iteration_num),
                "show_progress_bar": bool(
                    local_cfg.get("show_progress_bar", False)
                ),
            }
            if (
                "context_strategy" in local_cfg
                or "context" in local_cfg
            ):
                unbudgeted_context_plan = v14_resolve_context_plan(
                    local_cfg,
                    full_train_n=len(y_train),
                    total_sample_n=RUN_CONFIG.get(
                        "total_n", len(y_train) + len(y_test)
                    ),
                )
            else:
                unbudgeted_context_plan = {
                    "strategy": "full",
                    "strategy_source": "backward_compatible_unbudgeted_default",
                    "configured_rule": "full_outer_training_partition",
                    "target_context_n": int(len(y_train)),
                    "total_sample_n": int(
                        RUN_CONFIG.get(
                            "total_n", len(y_train) + len(y_test)
                        )
                    ),
                }
            if unbudgeted_context_plan["strategy"] == "adaptive":
                raise ValueError(
                    "An explicit adaptive local TabPFN context requires a "
                    "finite runtime budget; set budgeting.apply_runtime_budget=true."
                )
            context_target = int(
                unbudgeted_context_plan["target_context_n"]
            )
            if context_target < len(y_train):
                X_fit, y_fit, _ = _stratified_context_subset(
                    X_train,
                    y_train,
                    context_target,
                    seed=get_seed(iteration_num) + context_target,
                )
            else:
                X_fit, y_fit = X_train, y_train
            local_model = _make_local_tabpfn(
                dict(local_kwargs), model_version=local_model_version
            )
            fit_start = time.perf_counter()
            local_model.fit(X_fit, y_fit)
            fit_time = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            pred, proba = predict_binary_argmax(local_model, X_test)
            predict_proba_time = time.perf_counter() - predict_start
            metric_y_true = y_test
            end_to_end_time = fit_time + predict_proba_time
            tabpfn_budget_meta = {
                "TabPFN_Local_Budget_Mode": (
                    f"unbudgeted_{unbudgeted_context_plan['strategy']}_context"
                ),
                "TabPFN_Local_Device": device,
                "TabPFN_Local_Model_Version": local_model_version_label,
                "TabPFN_Local_Model_Path": local_kwargs.get("model_path", "auto"),
                "TabPFN_Local_Fit_Mode": local_kwargs.get("fit_mode", ""),
                "TabPFN_Local_N_Estimators": local_kwargs.get(
                    "n_estimators", np.nan
                ),
                "TabPFN_Full_Train_N": int(len(y_train)),
                "TabPFN_Budgeted_Context_Sample_Size": int(len(y_fit)),
                "TabPFN_Context_N_Used": int(len(y_fit)),
                "TabPFN_Context_Fraction_Used": float(
                    len(y_fit) / len(y_train)
                ),
                "TabPFN_Context_Strategy": unbudgeted_context_plan[
                    "strategy"
                ],
                "TabPFN_Context_Strategy_Source": unbudgeted_context_plan[
                    "strategy_source"
                ],
                "TabPFN_Context_Configured_Rule": (
                    unbudgeted_context_plan["configured_rule"]
                ),
                "TabPFN_Context_Fraction_Of_Outer_Train": float(
                    len(y_fit) / len(y_train)
                ),
                "TabPFN_Context_Fraction_Of_Total_Sampled_Records": (
                    float(
                        len(y_fit)
                        / unbudgeted_context_plan["total_sample_n"]
                    )
                    if RUN_CONFIG.get("total_n") else np.nan
                ),
                "TabPFN_Context_Selection_Rule": (
                    "exact_single_"
                    f"{unbudgeted_context_plan['strategy']}_"
                    "context_unbudgeted_full_test"
                ),
                "TabPFN_Context_Search_Attempts": 1,
                "TabPFN_Context_Search_Total_Runtime_Seconds": end_to_end_time,
                "TabPFN_Context_Candidates": [int(len(y_fit))],
                "TabPFN_Context_Attempt_Log": [{
                    "context_n": int(len(y_fit)),
                    "full_train_n": int(len(y_train)),
                    "full_prediction_n": int(len(y_test)),
                    "fit_time": float(fit_time),
                    "predict_proba_time": float(predict_proba_time),
                    "end_to_end_time": float(end_to_end_time),
                    "strict_budget_passed": None,
                    "predict_budget_passed": None,
                }],
                "TabPFN_Adaptive_Context_Enabled": False,
                "TabPFN_Full_Prediction_N": int(len(y_test)),
                "TabPFN_Budgeted_Prediction_Sample_Size": int(len(y_test)),
                "TabPFN_Prediction_Fraction_Used": 1.0,
                "TabPFN_Prediction_Selection_Rule": "full_test_prediction",
                "TabPFN_Prediction_Search_Attempts": 0,
                "TabPFN_Prediction_Candidates": [int(len(y_test))],
                "TabPFN_Prediction_Attempt_Log": [],
                "TabPFN_Fit_Time_Seconds": float(fit_time),
                "TabPFN_PredictProba_Time_Seconds": float(
                    predict_proba_time
                ),
                "TabPFN_EndToEnd_FitPlusPredictProba_Seconds": float(
                    end_to_end_time
                ),
                "TabPFN_Primary_Budget_Interpretation": "NoBudgeting",
                "TabPFN_Secondary_Budget_Interpretation": "NoBudgeting",
            }

        elif use_time_budget:
            if LocalTabPFNClassifier is None:
                raise ImportError(
                    "Local tabpfn.TabPFNClassifier is unavailable; install 'tabpfn' "
                    "to run TabPFN under a tuned-ML-model CPU budget."
                )

            default_tabpfn_cfg = _as_dict(
                _as_dict(globals().get("CONFIG", {}).get("models", {})).get("TabPFN", {})
            )
            default_local_cfg = _as_dict(
                default_tabpfn_cfg.get("local_tabpfn_budget", {})
            )
            requested_local_cfg = _as_dict(
                cfg.get("local_tabpfn_budget", cfg.get("local_tabpfn", {}))
            )
            local_cfg = _merge_dicts(default_local_cfg, requested_local_cfg)
            device = _get_budgeted_tabpfn_device(cfg)
            ml_reference_budget_seconds = float(time_budget_seconds)
            budget_seconds = float(ml_reference_budget_seconds)
            sample_size_for_context = RUN_CONFIG.get("total_n", None)
            context_plan = v14_resolve_context_plan(
                local_cfg,
                full_train_n=len(y_train),
                total_sample_n=sample_size_for_context or (
                    len(y_train) + len(y_test)
                ),
            )
            context_strategy = context_plan["strategy"]
            min_context_fraction = context_plan["fraction"]
            min_context_requested = context_plan["requested_context_n"]
            has_min_context_floor = min_context_requested > 0
            print(
                "   TabPFN context plan -> "
                f"strategy={context_strategy}, "
                f"rule={context_plan['configured_rule']}, "
                f"requested={min_context_requested or 'data-driven'}"
            )
            min_context_multiplier_enabled = bool(
                local_cfg.get("minimum_context_budget_multiplier_enabled", True)
            )
            fixed_minimum_context_only = context_strategy == "fixed"
            local_model_version = _resolve_local_tabpfn_model_version(
                local_cfg.get("model_version", "auto")
            )
            local_model_version_label = _tabpfn_model_version_label(local_model_version)
            local_model_path_label = (
                f"create_default_for_version({local_model_version_label})"
                if local_model_version is not None
                else local_cfg.get("model_path", cfg.get("local_model_path", "auto"))
            )
            local_kwargs = {
                "model_path": local_cfg.get("model_path", cfg.get("local_model_path", "auto")),
                "device": device,
                "ignore_pretraining_limits": local_cfg.get(
                    "ignore_pretraining_limits",
                    cfg.get("ignore_pretraining_limits", True),
                ),
                "n_estimators": int(local_cfg.get("n_estimators", cfg.get("local_n_estimators", 1))),
                "auto_scale_n_estimators": bool(local_cfg.get("auto_scale_n_estimators", False)),
                "fit_mode": local_cfg.get("fit_mode", cfg.get("local_fit_mode", "fit_preprocessors")),
                "random_state": get_seed(iteration_num),
                "show_progress_bar": bool(local_cfg.get("show_progress_bar", False)),
            }

            print(
                f"   Iteration {iteration_num}: Running local TabPFN CPU budget evaluation "
                f"(ML budget={ml_reference_budget_seconds:.3f}s, device={device}, "
                f"model_version={local_model_version_label}, "
                f"fit_mode={local_kwargs['fit_mode']}, "
                f"n_estimators={local_kwargs['n_estimators']})..."
            )

            adaptive_enabled = context_strategy == "adaptive"
            full_context_n = int(len(y_train))
            full_prediction_n = int(len(y_test))
            class_floor = max(1, len(np.unique(np.asarray(y_train))))
            min_context_target = int(context_plan["target_context_n"])
            user_candidates = _tabpfn_context_candidates(full_context_n, local_cfg)

            best_pass = None
            smallest_attempt = None
            attempts = []
            attempted_context_sizes = set()
            min_context_probe = None
            min_context_budget_multiplier_applied = False
            min_context_runtime_seconds = np.nan
            context_search_start = time.perf_counter()

            def budget_multiplier():
                return (
                    float(budget_seconds / ml_reference_budget_seconds)
                    if ml_reference_budget_seconds > 0 else np.nan
                )

            def refresh_budget_flags(attempt_result=None):
                for attempt in attempts:
                    attempt["ml_reference_budget_seconds"] = float(ml_reference_budget_seconds)
                    attempt["effective_budget_seconds"] = float(budget_seconds)
                    attempt["budget_multiplier"] = float(budget_multiplier())
                    attempt["predict_budget_passed"] = bool(
                        float(attempt["predict_proba_time"]) <= budget_seconds
                    )
                    attempt["strict_budget_passed"] = bool(
                        float(attempt["end_to_end_time"]) <= budget_seconds
                    )

                if attempt_result is not None:
                    attempt_result["predict_budget_passed"] = bool(
                        float(attempt_result["predict_proba_time"]) <= budget_seconds
                    )
                    attempt_result["strict_budget_passed"] = bool(
                        float(attempt_result["end_to_end_time"]) <= budget_seconds
                    )
                return attempt_result

            def evaluate_context(context_n):
                context_n = int(max(1, min(int(context_n), full_context_n)))
                if context_n in attempted_context_sizes:
                    return None
                attempted_context_sizes.add(context_n)

                X_ctx, y_ctx, selected_idx = _stratified_context_subset(
                    X_train,
                    y_train,
                    context_n,
                    seed=get_seed(iteration_num) + int(context_n),
                )

                tabpfn = _make_local_tabpfn(
                    dict(local_kwargs),
                    model_version=local_model_version,
                )
                fit_start = time.perf_counter()
                tabpfn.fit(X_ctx, y_ctx)
                fit_time_i = time.perf_counter() - fit_start

                predict_start = time.perf_counter()
                pred_i, proba_i = predict_binary_argmax(tabpfn, X_test)
                predict_time_i = time.perf_counter() - predict_start

                end_to_end_i = fit_time_i + predict_time_i
                predict_pass_i = bool(predict_time_i <= budget_seconds)
                strict_pass_i = bool(end_to_end_i <= budget_seconds)

                attempt = {
                    "context_n": int(len(y_ctx)),
                    "full_train_n": int(full_context_n),
                    "full_prediction_n": int(full_prediction_n),
                    "ml_reference_budget_seconds": float(ml_reference_budget_seconds),
                    "effective_budget_seconds": float(budget_seconds),
                    "budget_multiplier": float(budget_multiplier()),
                    "fit_time": float(fit_time_i),
                    "predict_proba_time": float(predict_time_i),
                    "end_to_end_time": float(end_to_end_i),
                    "predict_budget_passed": predict_pass_i,
                    "strict_budget_passed": strict_pass_i,
                }
                attempts.append(attempt)

                print(
                    "   TabPFN context attempt -> "
                    f"N_context={len(y_ctx)}/{full_context_n}, "
                    f"N_test={full_prediction_n}, fit={fit_time_i:.3f}s, "
                    f"predict_proba_full_test={predict_time_i:.3f}s, "
                    f"fit+predict={end_to_end_i:.3f}s, "
                    f"effective_budget={budget_seconds:.3f}s "
                    f"(x{budget_multiplier():.3f} ML budget), "
                    f"total_budget={'PASS' if strict_pass_i else 'TIMEOUT'}"
                )

                return {
                    "context_n": int(len(y_ctx)),
                    "full_train_n": int(full_context_n),
                    "full_prediction_n": int(full_prediction_n),
                    "fit_time": fit_time_i,
                    "predict_proba_time": predict_time_i,
                    "end_to_end_time": end_to_end_i,
                    "pred": pred_i,
                    "proba": proba_i,
                    "y_true": y_test,
                    "strict_budget_passed": strict_pass_i,
                    "predict_budget_passed": predict_pass_i,
                }

            def remember_attempt(attempt_result):
                nonlocal best_pass, smallest_attempt
                attempt_result = refresh_budget_flags(attempt_result)
                if attempt_result is None:
                    return
                if (
                    smallest_attempt is None
                    or int(attempt_result["context_n"]) < int(smallest_attempt["context_n"])
                ):
                    smallest_attempt = attempt_result
                if bool(attempt_result["strict_budget_passed"]):
                    if (
                        best_pass is None
                        or int(attempt_result["context_n"]) > int(best_pass["context_n"])
                    ):
                        best_pass = attempt_result

            def refine_between(pass_result, timeout_result):
                low_context = int(pass_result["context_n"])
                high_context = int(timeout_result["context_n"])

                while high_context - low_context > 1:
                    midpoint = int((low_context + high_context) // 2)
                    midpoint_result = evaluate_context(midpoint)
                    if midpoint_result is None:
                        break

                    remember_attempt(midpoint_result)
                    if bool(midpoint_result["strict_budget_passed"]):
                        low_context = int(midpoint_result["context_n"])
                    else:
                        high_context = int(midpoint_result["context_n"])

            def evaluate_min_context_target():
                nonlocal min_context_probe
                nonlocal min_context_runtime_seconds
                nonlocal budget_seconds
                nonlocal min_context_budget_multiplier_applied

                min_context_probe = evaluate_context(min_context_target)
                if min_context_probe is not None:
                    min_context_runtime_seconds = float(min_context_probe["end_to_end_time"])
                    if min_context_multiplier_enabled and min_context_runtime_seconds > budget_seconds:
                        budget_seconds = min_context_runtime_seconds
                        min_context_budget_multiplier_applied = True
                        refresh_budget_flags(min_context_probe)
                        print(
                            "   TabPFN effective budget expanded for fixed context target -> "
                            f"ML_budget={ml_reference_budget_seconds:.3f}s, "
                            f"target_context={min_context_target}, "
                            f"measured_fit+predict={min_context_runtime_seconds:.3f}s, "
                            f"effective_budget={budget_seconds:.3f}s, "
                            f"multiplier={budget_multiplier():.3f}"
                        )
                    remember_attempt(min_context_probe)

            def run_adaptive_context_search():
                start_context = None
                if best_pass is not None:
                    start_result = best_pass
                else:
                    start_context = _tabpfn_data_driven_start_context(X_train, y_train)
                    start_result = evaluate_context(start_context)
                    remember_attempt(start_result)

                if start_result is not None and bool(start_result["strict_budget_passed"]):
                    low_result = start_result
                    timeout_result = None

                    while int(low_result["context_n"]) < full_context_n:
                        current_n = int(low_result["context_n"])
                        total_time = float(low_result["end_to_end_time"])

                        if total_time <= 0:
                            next_context = full_context_n
                        else:
                            time_scaled_context = int(
                                np.floor(current_n * budget_seconds / total_time)
                            )
                            data_step = int(np.ceil(np.sqrt(full_context_n)))
                            next_context = max(
                                current_n + 1,
                                current_n + data_step,
                                time_scaled_context,
                            )
                            next_context = min(full_context_n, next_context)

                        if next_context <= current_n:
                            break

                        next_result = evaluate_context(next_context)
                        remember_attempt(next_result)
                        if next_result is None:
                            break

                        if bool(next_result["strict_budget_passed"]):
                            low_result = next_result
                            if int(low_result["context_n"]) >= full_context_n:
                                break
                        else:
                            timeout_result = next_result
                            break

                    if timeout_result is not None:
                        refine_between(low_result, timeout_result)

                else:
                    if start_context is not None and start_context > class_floor:
                        floor_result = evaluate_context(class_floor)
                        remember_attempt(floor_result)
                        if (
                            floor_result is not None
                            and bool(floor_result["strict_budget_passed"])
                            and start_result is not None
                        ):
                            refine_between(floor_result, start_result)

            if context_strategy == "fixed":
                evaluate_min_context_target()
            elif context_strategy == "full":
                remember_attempt(evaluate_context(full_context_n))
            else:
                if min_context_target > 0:
                    evaluate_min_context_target()
                if user_candidates:
                    first_timeout = None
                    previous_pass = best_pass
                    for context_n in user_candidates:
                        attempt_result = evaluate_context(context_n)
                        remember_attempt(attempt_result)
                        if attempt_result is None:
                            continue
                        if bool(attempt_result["strict_budget_passed"]):
                            previous_pass = attempt_result
                        else:
                            first_timeout = attempt_result
                            break
                    if previous_pass is not None and first_timeout is not None:
                        refine_between(previous_pass, first_timeout)
                else:
                    run_adaptive_context_search()

            context_search_total_runtime = time.perf_counter() - context_search_start
            chosen = best_pass if best_pass is not None else smallest_attempt
            if chosen is None:
                raise RuntimeError(
                    f"TabPFN {context_strategy} context execution produced no attempts."
                )

            fit_time = float(chosen["fit_time"])
            predict_proba_time = float(chosen["predict_proba_time"])
            end_to_end_time = float(chosen["end_to_end_time"])
            pred = chosen["pred"]
            proba = chosen["proba"]
            metric_y_true = chosen["y_true"]

            strict_end_to_end_pass = bool(chosen["strict_budget_passed"])
            predict_proba_pass = bool(chosen["predict_budget_passed"])
            context_n_used = int(chosen["context_n"])
            prediction_n_used = full_prediction_n
            context_fraction = float(context_n_used / full_context_n) if full_context_n else np.nan
            prediction_fraction = float(prediction_n_used / full_prediction_n) if full_prediction_n else np.nan
            predict_budget_delta = float(predict_proba_time - budget_seconds)
            predict_budget_remaining = float(budget_seconds - predict_proba_time)
            predict_budget_use_ratio = (
                float(predict_proba_time / budget_seconds)
                if budget_seconds > 0 else np.nan
            )
            total_budget_delta = float(end_to_end_time - budget_seconds)
            total_budget_remaining = float(budget_seconds - end_to_end_time)
            original_total_budget_delta = float(end_to_end_time - ml_reference_budget_seconds)
            original_total_budget_remaining = float(ml_reference_budget_seconds - end_to_end_time)
            effective_budget_multiplier = float(budget_multiplier())
            min_context_requirement_met = bool(
                not has_min_context_floor or context_n_used >= min_context_target
            )
            if context_strategy == "fixed":
                context_selection_rule = (
                    "exact_single_fixed_context_candidate_full_test"
                )
            elif context_strategy == "full":
                context_selection_rule = (
                    "exact_single_full_outer_train_context_full_test"
                )
            elif not has_min_context_floor:
                context_selection_rule = (
                    "largest_data_driven_context_within_ml_budget_plus_full_test_predict"
                    if strict_end_to_end_pass
                    else "smallest_data_driven_context_observed_after_budget_timeout"
                )
            elif min_context_budget_multiplier_applied:
                context_selection_rule = (
                    "fixed_minimum_context_with_multiplied_budget_plus_full_test_predict"
                    if strict_end_to_end_pass
                    else "fixed_minimum_context_observed_but_effective_budget_timeout"
                )
            else:
                context_selection_rule = (
                    "largest_context_at_or_above_minimum_within_ml_budget_plus_full_test_predict"
                    if strict_end_to_end_pass
                    else "minimum_context_floor_within_ml_budget_plus_full_test_predict"
                )
            if has_min_context_floor:
                context_budget_calculation = (
                    f"effective_budget_seconds=max(ML_budget_seconds, "
                    f"runtime_at_fraction_floor_{min_context_target})="
                    f"max({ml_reference_budget_seconds:.6f}, "
                    f"{float(min_context_runtime_seconds):.6f})="
                    f"{budget_seconds:.6f}; multiplier="
                    f"{effective_budget_multiplier:.6f}; final_context_N={context_n_used}"
                )
            else:
                context_budget_calculation = (
                    f"effective_budget_seconds=ML_budget_seconds="
                    f"{budget_seconds:.6f}; no_fixed_context_floor; "
                    f"final_context_N={context_n_used}"
                )

            tabpfn_budget_meta = {
                "TabPFN_Local_Budget_Mode": (
                    f"{context_strategy}_context_effective_total_runtime_budget_full_test"
                ),
                "TabPFN_Local_Device": device,
                "TabPFN_Local_Model_Version": local_model_version_label,
                "TabPFN_Local_Model_Path": local_model_path_label,
                "TabPFN_Local_Fit_Mode": local_kwargs.get("fit_mode", ""),
                "TabPFN_Local_N_Estimators": local_kwargs.get("n_estimators", np.nan),
                "TabPFN_Budget_Reference_Seconds": ml_reference_budget_seconds,
                "TabPFN_ML_Reference_Budget_Seconds": ml_reference_budget_seconds,
                "TabPFN_Effective_Time_Budget_Seconds": budget_seconds,
                "TabPFN_Effective_Budget_Multiplier": effective_budget_multiplier,
                "TabPFN_Min_Context_Requested": min_context_requested,
                "TabPFN_Min_Context_Target": min_context_target,
                "TabPFN_Min_Context_Runtime_Seconds": min_context_runtime_seconds,
                "TabPFN_Min_Context_Budget_Multiplier_Applied": min_context_budget_multiplier_applied,
                "TabPFN_Min_Context_Requirement_Met": min_context_requirement_met,
                "TabPFN_Fixed_Minimum_Context_Only": fixed_minimum_context_only,
                "TabPFN_Context_Budget_Calculation": context_budget_calculation,
                "TabPFN_Full_Train_N": full_context_n,
                "TabPFN_Budgeted_Context_Sample_Size": context_n_used,
                "TabPFN_Context_N_Used": context_n_used,
                "TabPFN_Context_Fraction_Used": context_fraction,
                "TabPFN_Context_Strategy": context_strategy,
                "TabPFN_Context_Strategy_Source": context_plan[
                    "strategy_source"
                ],
                "TabPFN_Context_Configured_Rule": context_plan[
                    "configured_rule"
                ],
                "TabPFN_Context_Fraction_Denominator": context_plan[
                    "fraction_denominator"
                ],
                "TabPFN_Context_Fraction_Of_Outer_Train": (
                    float(context_n_used / full_context_n)
                    if full_context_n else np.nan
                ),
                "TabPFN_Context_Fraction_Of_Total_Sampled_Records": (
                    float(context_n_used / context_plan["total_sample_n"])
                    if context_plan["total_sample_n"] else np.nan
                ),
                "TabPFN_Context_Selection_Rule": context_selection_rule,
                "TabPFN_Context_Search_Attempts": len(attempts),
                "TabPFN_Context_Search_Total_Runtime_Seconds": context_search_total_runtime,
                "TabPFN_Context_Search_Effective_Budget_Passed": bool(
                    context_search_total_runtime <= budget_seconds
                ),
                "TabPFN_Context_Search_Over_Effective_Budget_Seconds": max(
                    0.0, context_search_total_runtime - budget_seconds
                ),
                "TabPFN_Context_Candidates": [a["context_n"] for a in attempts],
                "TabPFN_Context_Attempt_Log": attempts,
                "TabPFN_Full_Prediction_N": full_prediction_n,
                "TabPFN_Budgeted_Prediction_Sample_Size": full_prediction_n,
                "TabPFN_Prediction_Fraction_Used": prediction_fraction,
                "TabPFN_Prediction_Selection_Rule": "full_test_prediction_no_budget_reduction",
                "TabPFN_Prediction_Search_Attempts": 0,
                "TabPFN_Prediction_Candidates": [full_prediction_n],
                "TabPFN_Prediction_Attempt_Log": [],
                "TabPFN_Adaptive_Context_Enabled": adaptive_enabled,
                "TabPFN_Fit_Time_Seconds": fit_time,
                "TabPFN_PredictProba_Time_Seconds": predict_proba_time,
                "TabPFN_PredictProba_Budget_Delta_Seconds": predict_budget_delta,
                "TabPFN_PredictProba_Budget_Remaining_Seconds": predict_budget_remaining,
                "TabPFN_PredictProba_Budget_Use_Ratio": predict_budget_use_ratio,
                "TabPFN_EndToEnd_FitPlusPredictProba_Seconds": end_to_end_time,
                "TabPFN_Total_Runtime_Budget_Delta_Seconds": total_budget_delta,
                "TabPFN_Total_Runtime_Budget_Remaining_Seconds": total_budget_remaining,
                "TabPFN_Original_ML_Budget_Total_Runtime_Delta_Seconds": original_total_budget_delta,
                "TabPFN_Original_ML_Budget_Total_Runtime_Remaining_Seconds": original_total_budget_remaining,
                "TabPFN_Strict_EndToEnd_Budget_Passed": strict_end_to_end_pass,
                "TabPFN_PredictProba_Budget_Passed": predict_proba_pass,
                "TabPFN_Strict_EndToEnd_Overrun_Seconds": max(0.0, end_to_end_time - budget_seconds),
                "TabPFN_PredictProba_Overrun_Seconds": max(0.0, predict_proba_time - budget_seconds),
                "TabPFN_Strict_EndToEnd_Status": (
                    "WithinBudget" if strict_end_to_end_pass else "Timeout"
                ),
                "TabPFN_PredictProba_Status": (
                    "WithinBudget" if predict_proba_pass else "Timeout"
                ),
                "TabPFN_Primary_Budget_Interpretation": (
                    "SelectedCandidateFitPlusFullTestPredict_EffectiveBudget"
                ),
                "TabPFN_Secondary_Budget_Interpretation": (
                    "CumulativeAdaptiveSearchRuntime_ReportedSeparately"
                ),
            }

            print(
                "   TabPFN timing -> "
                f"context_N={context_n_used}/{full_context_n}, test_N={full_prediction_n}, "
                f"ML_budget={ml_reference_budget_seconds:.3f}s, "
                f"effective_budget={budget_seconds:.3f}s "
                f"(x{effective_budget_multiplier:.3f}), "
                f"fit={fit_time:.3f}s, predict_proba={predict_proba_time:.3f}s, "
                f"total_budget_gap={total_budget_remaining:.3f}s, "
                f"fit+predict_proba={end_to_end_time:.3f}s, "
                f"adaptive_search_total={context_search_total_runtime:.3f}s | "
                f"strict={'PASS' if strict_end_to_end_pass else 'TIMEOUT'}, "
                f"predict_proba={'PASS' if predict_proba_pass else 'TIMEOUT'}"
            )

        else:
            print(
                f"   Iteration {iteration_num}: Running TabPFN cloud/client..."
            )

            tabpfn = TabPFNClassifier(model_path=cfg["model_path"])
            print(f"   TabPFN model path: {tabpfn.model_path}")

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                tabpfn.fit(X_train, y_train)
                pred, proba = predict_binary_argmax(tabpfn, X_test)
                metric_y_true = y_test

        fpr, tpr, _ = roc_curve(metric_y_true, proba)

        result = (
            balanced_accuracy_score(metric_y_true, pred),
            roc_auc_score(metric_y_true, proba),
            brier_score_loss(metric_y_true, proba, pos_label=1),
            fpr,
            tpr,
            recall_score(metric_y_true, pred, zero_division=0),
            precision_score(metric_y_true, pred, zero_division=0),
            proba,
            metric_y_true
        )

    finally:
        energy_kwh = stop_energy_tracker(tracker)

    if tabpfn_budget_meta:
        return (*result, energy_kwh, tabpfn_budget_meta)

    return (*result, energy_kwh)



def run_linear_sparse_logistic_regression_budgeted(
    X_train, X_test, y_train, y_test,
    X_train_sub, y_train_sub, X_val, y_val,
    iteration_num,
    time_budget_seconds
):
    model_name = "L-SLR"
    cfg = get_model_cfg(model_name)
    fixed_params = v14_apply_cpu_model_params(
        model_name, cfg.get("fixed_params", {})
    )

    tracker = start_energy_tracker(f"{model_name}_budgeted_iter{iteration_num}")

    try:
        def objective(trial):
            params = suggest_params_from_config(trial, model_name)

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    random_state=get_seed(iteration_num),
                    **fixed_params,
                    **params
                )
            )

            model.fit(X_train_sub, y_train_sub)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        optimal_params, _ = run_budgeted_study(
            model_name,
            objective,
            iteration_num,
            time_budget_seconds
        )

        final_stage_start = time.perf_counter()
        final_stage_start_utc = v14_utc_now()

        final_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=get_seed(iteration_num),
                **fixed_params,
                **optimal_params
            )
        )

        fit_start = time.perf_counter()
        fit_start_utc = v14_utc_now()
        final_model.fit(X_train, y_train)
        fit_end = time.perf_counter()
        fit_end_utc = v14_utc_now()
        prediction_start = time.perf_counter()
        prediction_start_utc = v14_utc_now()
        result = evaluate_binary_model(final_model, X_test, y_test)
        prediction_end = time.perf_counter()
        prediction_end_utc = v14_utc_now()
        v14_record_final_fit_prediction_timing(
            model_name,
            iteration_num,
            fit_start,
            fit_end,
            prediction_start,
            prediction_end,
            fit_start_utc,
            fit_end_utc,
            prediction_start_utc,
            prediction_end_utc,
            final_stage_start=final_stage_start,
            final_stage_start_utc=final_stage_start_utc,
        )

    finally:
        energy_kwh = stop_energy_tracker(tracker)

    return (*result, energy_kwh)



def run_sparse_logistic_regression_budgeted(
    X_train, X_test, y_train, y_test,
    X_train_sub, y_train_sub, X_val, y_val,
    iteration_num,
    time_budget_seconds
):
    model_name = "Augmented_SLR"
    cfg = get_model_cfg(model_name)
    fixed_params = v14_apply_cpu_model_params(
        model_name, cfg.get("fixed_params", {})
    )
    aug_cfg = cfg.get("feature_augmentation", {})

    tracker = start_energy_tracker(f"{model_name}_budgeted_iter{iteration_num}")

    try:
        def make_augmented_pipeline(params):
            return make_pipeline(
                PolynomialFeatures(
                    degree=aug_cfg.get("degree", 2),
                    include_bias=aug_cfg.get("include_bias", False)
                ),
                StandardScaler(),
                LogisticRegression(
                    random_state=get_seed(iteration_num),
                    **fixed_params,
                    **params
                )
            )

        def objective(trial):
            params = suggest_params_from_config(trial, model_name)
            model = make_augmented_pipeline(params)
            model.fit(X_train_sub, y_train_sub)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        optimal_params, _ = run_budgeted_study(
            model_name,
            objective,
            iteration_num,
            time_budget_seconds
        )

        final_stage_start = time.perf_counter()
        final_stage_start_utc = v14_utc_now()

        final_model = make_augmented_pipeline(optimal_params)
        fit_start = time.perf_counter()
        fit_start_utc = v14_utc_now()
        final_model.fit(X_train, y_train)
        fit_end = time.perf_counter()
        fit_end_utc = v14_utc_now()
        prediction_start = time.perf_counter()
        prediction_start_utc = v14_utc_now()
        result = evaluate_binary_model(final_model, X_test, y_test)
        prediction_end = time.perf_counter()
        prediction_end_utc = v14_utc_now()
        v14_record_final_fit_prediction_timing(
            model_name,
            iteration_num,
            fit_start,
            fit_end,
            prediction_start,
            prediction_end,
            fit_start_utc,
            fit_end_utc,
            prediction_start_utc,
            prediction_end_utc,
            final_stage_start=final_stage_start,
            final_stage_start_utc=final_stage_start_utc,
        )

    finally:
        energy_kwh = stop_energy_tracker(tracker)

    return (*result, energy_kwh)



def run_random_forest_budgeted(
    X_train, X_test, y_train, y_test,
    X_train_sub, y_train_sub, X_val, y_val,
    iteration_num,
    time_budget_seconds
):
    model_name = "RandomForest"
    cfg = get_model_cfg(model_name)
    fixed_params = v14_apply_cpu_model_params(
        model_name, cfg.get("fixed_params", {})
    )

    tracker = start_energy_tracker(f"{model_name}_budgeted_iter{iteration_num}")

    try:
        def objective(trial):
            params = suggest_params_from_config(trial, model_name)

            model = RandomForestClassifier(
                random_state=get_seed(iteration_num),
                **fixed_params,
                **params
            )

            model.fit(X_train_sub, y_train_sub)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        optimal_params, _ = run_budgeted_study(
            model_name,
            objective,
            iteration_num,
            time_budget_seconds
        )

        final_stage_start = time.perf_counter()
        final_stage_start_utc = v14_utc_now()

        final_model = RandomForestClassifier(
            random_state=get_seed(iteration_num),
            **fixed_params,
            **optimal_params
        )

        fit_start = time.perf_counter()
        fit_start_utc = v14_utc_now()
        final_model.fit(X_train, y_train)
        fit_end = time.perf_counter()
        fit_end_utc = v14_utc_now()
        prediction_start = time.perf_counter()
        prediction_start_utc = v14_utc_now()
        result = evaluate_binary_model(final_model, X_test, y_test)
        prediction_end = time.perf_counter()
        prediction_end_utc = v14_utc_now()
        v14_record_final_fit_prediction_timing(
            model_name,
            iteration_num,
            fit_start,
            fit_end,
            prediction_start,
            prediction_end,
            fit_start_utc,
            fit_end_utc,
            prediction_start_utc,
            prediction_end_utc,
            final_stage_start=final_stage_start,
            final_stage_start_utc=final_stage_start_utc,
        )

    finally:
        energy_kwh = stop_energy_tracker(tracker)

    return (*result, energy_kwh)



def run_xgboost_budgeted(
    X_train, X_test, y_train, y_test,
    X_train_sub, y_train_sub, X_val, y_val,
    iteration_num,
    time_budget_seconds
):
    model_name = "XGBoost"
    cfg = get_model_cfg(model_name)
    fixed_params = v14_apply_cpu_model_params(
        model_name, cfg.get("fixed_params", {})
    )

    tracker = start_energy_tracker(f"{model_name}_budgeted_iter{iteration_num}")

    try:
        device_type = v14_resolve_requested_device(
            cfg.get("execution", {}).get("device", "auto"),
            require_requested=cfg.get("execution", {}).get(
                "require_requested_device", False
            ),
            model_name=model_name,
        )

        def objective(trial):
            params = suggest_params_from_config(trial, model_name)

            model = XGBClassifier(
                random_state=get_seed(iteration_num),
                device=device_type,
                **fixed_params,
                **params
            )

            model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            best_iter = getattr(model, "best_iteration", None)
            best_n_estimators = best_iter + 1 if best_iter is not None else fixed_params.get("n_estimators", 500)
            trial.set_user_attr("best_n_estimators", best_n_estimators)

            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        optimal_params, selected_trial = run_budgeted_study(
            model_name,
            objective,
            iteration_num,
            time_budget_seconds
        )

        best_n_estimators = (
            selected_trial.user_attrs.get("best_n_estimators", fixed_params.get("n_estimators", 500))
            if selected_trial is not None
            else fixed_params.get("n_estimators", 500)
        )

        final_stage_start = time.perf_counter()
        final_stage_start_utc = v14_utc_now()

        final_params = fixed_params.copy()
        final_params.pop("early_stopping_rounds", None)
        final_params["n_estimators"] = best_n_estimators

        final_model = XGBClassifier(
            random_state=get_seed(iteration_num),
            device=device_type,
            **final_params,
            **optimal_params
        )

        fit_start = time.perf_counter()
        fit_start_utc = v14_utc_now()
        final_model.fit(X_train, y_train)
        fit_end = time.perf_counter()
        fit_end_utc = v14_utc_now()
        prediction_start = time.perf_counter()
        prediction_start_utc = v14_utc_now()
        result = evaluate_binary_model(final_model, X_test, y_test)
        prediction_end = time.perf_counter()
        prediction_end_utc = v14_utc_now()
        v14_record_final_fit_prediction_timing(
            model_name,
            iteration_num,
            fit_start,
            fit_end,
            prediction_start,
            prediction_end,
            fit_start_utc,
            fit_end_utc,
            prediction_start_utc,
            prediction_end_utc,
            final_stage_start=final_stage_start,
            final_stage_start_utc=final_stage_start_utc,
        )

    finally:
        energy_kwh = stop_energy_tracker(tracker)

    return (*result, energy_kwh)



def run_catboost_budgeted(
    X_train, X_test, y_train, y_test,
    X_train_sub, y_train_sub, X_val, y_val,
    iteration_num,
    time_budget_seconds
):
    model_name = "CatBoost"
    cfg = get_model_cfg(model_name)
    fixed_params = v14_apply_cpu_model_params(
        model_name, cfg.get("fixed_params", {})
    )

    tracker = start_energy_tracker(f"{model_name}_budgeted_iter{iteration_num}")

    try:
        resolved_device = v14_resolve_requested_device(
            cfg.get("execution", {}).get("device", "auto"),
            require_requested=cfg.get("execution", {}).get(
                "require_requested_device", False
            ),
            model_name=model_name,
        )
        task_type = "GPU" if resolved_device == "cuda" else "CPU"

        def add_catboost_device_params(params):
            params["task_type"] = task_type

            if task_type == "GPU":
                gpu_params = cfg.get("gpu_params", {})
                params.update(gpu_params)

            return params

        def objective(trial):
            params = suggest_params_from_config(trial, model_name)

            cb_params = {
                **fixed_params,
                **params,
                "random_seed": get_seed(iteration_num)
            }

            cb_params = add_catboost_device_params(cb_params)

            model = CatBoostClassifier(**cb_params)

            model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=(X_val, y_val),
                verbose=False
            )

            best_it = getattr(model, "best_iteration_", None)

            if best_it is not None and best_it >= 0:
                best_iterations = int(best_it) + 1
            else:
                best_iterations = int(getattr(model, "tree_count_", fixed_params.get("iterations", 100)))

            trial.set_user_attr("best_iterations", best_iterations)

            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        optimal_params, selected_trial = run_budgeted_study(
            model_name,
            objective,
            iteration_num,
            time_budget_seconds
        )

        best_iterations = (
            selected_trial.user_attrs.get("best_iterations", fixed_params.get("iterations", 100))
            if selected_trial is not None
            else fixed_params.get("iterations", 100)
        )

        final_stage_start = time.perf_counter()
        final_stage_start_utc = v14_utc_now()

        final_params = fixed_params.copy()
        final_params.pop("early_stopping_rounds", None)
        final_params.pop("use_best_model", None)
        final_params["iterations"] = best_iterations

        final_params = {
            **final_params,
            **optimal_params,
            "random_seed": get_seed(iteration_num)
        }

        final_params = add_catboost_device_params(final_params)

        final_model = CatBoostClassifier(**final_params)
        fit_start = time.perf_counter()
        fit_start_utc = v14_utc_now()
        final_model.fit(X_train, y_train, verbose=False)
        fit_end = time.perf_counter()
        fit_end_utc = v14_utc_now()
        prediction_start = time.perf_counter()
        prediction_start_utc = v14_utc_now()
        result = evaluate_binary_model(final_model, X_test, y_test)
        prediction_end = time.perf_counter()
        prediction_end_utc = v14_utc_now()
        v14_record_final_fit_prediction_timing(
            model_name,
            iteration_num,
            fit_start,
            fit_end,
            prediction_start,
            prediction_end,
            fit_start_utc,
            fit_end_utc,
            prediction_start_utc,
            prediction_end_utc,
            final_stage_start=final_stage_start,
            final_stage_start_utc=final_stage_start_utc,
        )

    finally:
        energy_kwh = stop_energy_tracker(tracker)

    return (*result, energy_kwh)



MODEL_RUNNERS = {
    "TabPFN": run_tabpfn,
    "L-SLR": run_linear_sparse_logistic_regression_budgeted,
    "Augmented_SLR": run_sparse_logistic_regression_budgeted,
    "RandomForest": run_random_forest_budgeted,
    "XGBoost": run_xgboost_budgeted,
    "CatBoost": run_catboost_budgeted,
}

def get_enabled_model_names():
    cfgs = ACTIVE_MODEL_CONFIGS if ACTIVE_MODEL_CONFIGS is not None else MODEL_CONFIGS

    return [
        model_name
        for model_name, cfg in cfgs.items()
        if cfg.get("enabled", False)
    ]


def refresh_plot_settings():
    global PLOT_SUBFOLDER_NAME, SAVE_HTML, SHOW_FIGURES
    global SAVE_WIDE_ROC_CALIBRATION_VERSION
    global SAVE_PNG, SAVE_SVG, SAVE_JPEG, IMAGE_SCALE, PLOTLY_RENDERER
    global PPT_WIDTH, PPT_HEIGHT, ROC_CAL_WIDTH, ROC_CAL_HEIGHT
    global MODEL_ORDER

    if PLOT_CONFIG is None:
        raise ValueError("PLOT_CONFIG is not loaded. Call load_config(config) first.")

    PLOT_SUBFOLDER_NAME = PLOT_CONFIG.get("plot_subfolder_name", "Plotly_PowerPoint_Slides")
    SAVE_HTML = bool(PLOT_CONFIG.get("save_html", True))
    SHOW_FIGURES = bool(PLOT_CONFIG.get("show_figures", True))
    SAVE_WIDE_ROC_CALIBRATION_VERSION = bool(
        PLOT_CONFIG.get("save_wide_roc_calibration_version", False)
    )

    SAVE_PNG = bool(PLOT_CONFIG.get("save_png", True))
    SAVE_SVG = bool(PLOT_CONFIG.get("save_svg", False))
    SAVE_JPEG = bool(PLOT_CONFIG.get("save_jpeg", False))
    IMAGE_SCALE = PLOT_CONFIG.get("image_scale", 2)
    PLOTLY_RENDERER = PLOT_CONFIG.get("renderer", "colab")

    PPT_WIDTH = PLOT_CONFIG.get("ppt_width", 1600)
    PPT_HEIGHT = PLOT_CONFIG.get("ppt_height", 900)
    ROC_CAL_WIDTH = PLOT_CONFIG.get("roc_cal_width", 2000)
    ROC_CAL_HEIGHT = PLOT_CONFIG.get("roc_cal_height", 800)

    MODEL_ORDER = [
        model_name
        for model_name, cfg in MODEL_CONFIGS.items()
        if cfg.get("enabled", False)
    ]

def discover_scenarios_to_plot(results_dir, file_prefix):
    """
    Dynamically discovers scenario folders that contain RawResults.pkl.
    If a `scenarios_to_plot` dict/list exists in the notebook globals, it
    uses that order. Otherwise, it scans the RESULTS_DIR folder.
    """
    if "scenarios_to_plot" in globals() and isinstance(globals()["scenarios_to_plot"], dict):
        return list(globals()["scenarios_to_plot"].keys())

    scenario_names = []

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Main results folder not found:\n{results_dir}")

    for item in os.listdir(results_dir):
        scenario_dir = os.path.join(results_dir, item)

        if not os.path.isdir(scenario_dir):
            continue

        scenario_prefix = scenario_file_prefix(
            scenario_name=item,
            file_prefix=file_prefix,
        )
        raw_path = os.path.join(scenario_dir, f"{scenario_prefix}_RawResults.pkl")

        if os.path.exists(raw_path):
            scenario_names.append(item)

    if len(scenario_names) == 0:
        raise FileNotFoundError(
            "No scenario folders with RawResults.pkl were found inside:\n"
            f"{results_dir}"
        )

    return scenario_names



CONFIG = {
    "experiment_name": "TabPFN_Budgeted",

    "total_n": 1000,
    "train_frac": 0.80,
    "inner_validation_frac": 0.20,
    "iterations": 10,
    "base_seed": 2025,

    "output_root": "/content/drive/MyDrive",
    "output_folder": "LMU-Thesis TabPFN-Budgeted",
    "file_prefix": "TabPFN Budgeted V12",
    "output_layout": {
        "include_output_folder": True,
        "include_sample_size": True,
        "include_file_prefix": True,
        "run_id": None,
        "timestamp_run_dir": False,
        "timestamp_format": "%Y%m%d_%H%M%S",
        "results_dir": None,
    },

    "sampling": {
        "strategy": "original_prevalence"
    },

    "splitting": {
        "strategy": "auto",
        "group_aware": True,
        "strict": True,
        "candidate_splits": 256,
        "min_class_count_per_partition": 2,
        "low_class_count_warning": 10,
        "max_prevalence_deviation": None,
        "max_row_fraction_deviation": None,
        "shuffle_rows_within_partitions": True,
        "require_groups": False,
        "temporal_window": "latest",
        "temporal_gap_rows": 0,
        "temporal_enforce_group_disjoint": False,
    },

    "preprocessing": {
        "mode": "auto",
        "numeric_imputation": "median",
        "categorical_imputation_value": "__MISSING__",
        "categorical_encoding": "onehot",
        "onehot_min_frequency": None,
        "onehot_max_categories": None,
        "max_output_features": None,
    },

    "scenarios": {},

    "budget_reference_model": "TabPFN",

    "budgeting": {
        "enabled": True,
        "mode": "reference_runtime",
        "reference_model": "TabPFN",
        "applies_to": "optuna_tuning_only",
        "budgeted_selection_rule": "best_auc_within_budget",
        "no_budget_selection_rule": "best_auc_all_completed_trials",

        "cap_untuned_competitors": True,
    },

    "optuna_direction": "maximize",
    "default_max_trials": 50,

    "track_energy": True,
    "energy_tracking_mode": "process",

    "plots": {
        "plot_subfolder_name": "Plotly_PowerPoint_Slides",
        "save_html": True,
        "save_png": True,
        "save_svg": False,
        "save_jpeg": False,
        "show_figures": True,
        "save_wide_roc_calibration_version": False,
        "ppt_width": 1600,
        "ppt_height": 900,
        "roc_cal_width": 2000,
        "roc_cal_height": 800,
        "calibration_bins": 10,
        "image_scale": 2,
        "renderer": "colab",
    },

    "models": {
        "TabPFN": {
            "enabled": True,
            "is_budget_reference": True,
            "input_mode": "Cloud API / Native",
            "tuned_by_optuna": False,
            "model_path": "v3_default",

            "ignore_pretraining_limits": True,
            "phe_init_args": None,
            "tabpfn_device": "cpu",
            "local_tabpfn_budget": {
                "model_version": "V3",
                "model_path": "auto",
                "fit_mode": "fit_preprocessors",
                "n_estimators": 1,
                "auto_scale_n_estimators": False,
                "show_progress_bar": False,
                "adaptive_context_enabled": True,
                "context_strategy": {
                    "strategy": "adaptive"
                },
                "minimum_context_budget_multiplier_enabled": True,
                "fixed_minimum_context_only": False,
            },
        },

        "L-SLR": {
            "enabled": True,
            "is_budget_reference": False,
            "input_mode": "StandardScaler",
            "tuned_by_optuna": True,
            "fixed_params": {
                "penalty": "l1",
                "solver": "saga",
                "max_iter": 5000,
            },
            "search_space": {
                "C": {
                    "type": "float",
                    "low": 0.001,
                    "high": 100.0,
                    "log": True,
                }
            },
            "default_params": {
                "C": 1.0
            }
        },

        "Augmented_SLR": {
            "enabled": True,
            "is_budget_reference": False,
            "input_mode": "PolynomialFeatures(degree=2) + StandardScaler",
            "tuned_by_optuna": True,
            "tuned_by_optuna": True,
            "feature_augmentation": {
                "use_polynomial_features": True,
                "degree": 2,
                "include_bias": False,
            },
            "fixed_params": {
                "penalty": "l1",
                "solver": "saga",
                "max_iter": 5000,
            },
            "search_space": {
                "C": {
                    "type": "float",
                    "low": 0.001,
                    "high": 100.0,
                    "log": True,
                }
            },
            "default_params": {
                "C": 1.0
            }
        },

        "RandomForest": {
            "enabled": True,
            "is_budget_reference": False,
            "input_mode": "Native",
            "tuned_by_optuna": True,
            "fixed_params": {
                "n_jobs": -1,
            },
            "search_space": {
                "n_estimators": {
                    "type": "int",
                    "low": 50,
                    "high": 1000,
                },
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 12,
                },
                "max_features": {
                    "type": "float",
                    "low": 0.3,
                    "high": 1.0,
                    "log": False,
                },
                "min_samples_leaf": {
                    "type": "int",
                    "low": 1,
                    "high": 20,
                },
            },
            "default_params": {
                "n_estimators": 200,
                "max_depth": 6,
                "max_features": 1.0,
                "min_samples_leaf": 1,
            }
        },

        "XGBoost": {
            "enabled": True,
            "is_budget_reference": False,
            "input_mode": "Native",
            "tuned_by_optuna": True,
            "fixed_params": {
                "n_estimators": 500,
                "eval_metric": "auc",
                "n_jobs": -1,
                "early_stopping_rounds": 20,
            },
            "search_space": {
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 8,
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
                "min_child_weight": {
                    "type": "float",
                    "low": 1.0,
                    "high": 20.0,
                    "log": True,
                },
                "subsample": {
                    "type": "float",
                    "low": 0.6,
                    "high": 1.0,
                    "log": False,
                },
            },
            "default_params": {
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 1.0,
                "subsample": 1.0,
            }
        },

        "CatBoost": {
            "enabled": True,
            "is_budget_reference": False,
            "input_mode": "Native",
            "tuned_by_optuna": True,
            "fixed_params": {
                "iterations": 1000,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": 0,
                "early_stopping_rounds": 20,
                "use_best_model": True,
                "allow_writing_files": False,
            },
            "search_space": {
                "depth": {
                    "type": "int",
                    "low": 3,
                    "high": 8,
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
                "l2_leaf_reg": {
                    "type": "float",
                    "low": 1.0,
                    "high": 20.0,
                    "log": True,
                },
                "random_strength": {
                    "type": "float",
                    "low": 0.1,
                    "high": 10.0,
                    "log": True,
                },
            },
            "default_params": {
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
            }
        },
    },
}



MODEL_ORDER = []

MODEL_COLORS = {
    "TabPFN": "#1f77b4",
    "L-SLR": "#ff7f0e",
    "Augmented_SLR": "#2ca02c",
    "RandomForest": "#d62728",
    "XGBoost": "#9467bd",
    "CatBoost": "#8c564b",
}

PREDICTIVE_METRICS = [
    {
        "key": "AUC",
        "title": "AUC",
        "ylabel": "AUC",
        "higher_is_better": True,
        "row": 1,
        "col": 1,
    },
    {
        "key": "Brier",
        "title": "Brier Score",
        "ylabel": "Brier score",
        "higher_is_better": False,
        "row": 1,
        "col": 2,
    },
    {
        "key": "BalancedAccuracy",
        "title": "Balanced Accuracy",
        "ylabel": "Balanced accuracy",
        "higher_is_better": True,
        "row": 1,
        "col": 3,
    },
    {
        "key": "Sensitivity",
        "title": "Sensitivity",
        "ylabel": "Sensitivity",
        "higher_is_better": True,
        "row": 2,
        "col": 1,
    },
    {
        "key": "Precision",
        "title": "Precision",
        "ylabel": "Precision",
        "higher_is_better": True,
        "row": 2,
        "col": 2,
    },
]

EFFICIENCY_METRICS = [
    {
        "key": "ActualTotalRuntime",
        "title": "Actual Total Runtime",
        "ylabel": "Seconds",
        "higher_is_better": False,
        "row": 1,
        "col": 1,
    },
    {
        "key": "BudgetedTotalRuntime",
        "title": "Budgeted Total Runtime",
        "ylabel": "Seconds",
        "higher_is_better": False,
        "row": 1,
        "col": 2,
    },
    {
        "key": "Energy",
        "title": "Energy Consumption",
        "ylabel": "kWh",
        "higher_is_better": False,
        "row": 2,
        "col": 1,
    },
    {
        "key": "EligibleTrials",
        "title": "Eligible Trials Within TabPFN Budget",
        "ylabel": "Number of trials",
        "higher_is_better": True,
        "row": 2,
        "col": 2,
    },
]

PREDICTIVE_TABLE_METRICS = [
    ("AUC", "AUC", True),
    ("Brier", "Brier", False),
    ("BalancedAccuracy", "Balanced Accuracy", True),
    ("Sensitivity", "Sensitivity", True),
    ("Precision", "Precision", True),
]

OVERALL_TABLE_METRICS = [
    ("AUC", "AUC", True),
    ("Brier", "Brier", False),
    ("BalancedAccuracy", "Balanced Accuracy", True),
    ("Sensitivity", "Sensitivity", True),
    ("Precision", "Precision", True),
    ("BudgetedTotalRuntime", "Budgeted Runtime", False),
    ("ActualTotalRuntime", "Actual Runtime", False),
    ("Energy", "Energy", False),
]



# =============================================================================
# 5. Reporting figures, dashboard exports, and Monte Carlo orchestration
# =============================================================================

def clean_scenario_subtitle(scenario_name, sample_size):
    """Render every scenario label uniformly; names never control behavior."""
    return f"{scenario_name} | Sample size: {sample_size}"


def get_model_values(errors, model, metric_key):
    key = f"{model}_{metric_key}"

    if key not in errors:
        return np.array([], dtype=float)

    vals = pd.to_numeric(pd.Series(errors[key]), errors="coerce").dropna().values
    return vals.astype(float)


def metric_mean(errors, model, metric_key):
    vals = get_model_values(errors, model, metric_key)

    if len(vals) == 0:
        return np.nan

    return float(np.mean(vals))

def get_available_models(errors, model_order=None):
    if model_order is None:
        model_order = MODEL_ORDER

    available = []

    for model in model_order:
        vals = get_model_values(errors, model, "AUC")
        if len(vals) > 0:
            available.append(model)

    return available


def get_top_models(errors, metric_key, higher_is_better=True, top_n=3):
    rows = []

    for model in get_available_models(errors):
        mean_val = metric_mean(errors, model, metric_key)

        if np.isfinite(mean_val):
            rows.append((model, mean_val))

    if len(rows) == 0:
        return []

    rows = sorted(rows, key=lambda x: x[1], reverse=higher_is_better)
    return rows[:top_n]


def format_metric_value(metric_key, value):
    """
    Prevent tiny energy values from appearing as 0.0000.
    """
    if not np.isfinite(value):
        return ""

    if metric_key == "Energy":
        return f"{value:.2e}"

    if metric_key in ["ActualTotalRuntime", "BudgetedTotalRuntime"]:
        return f"{value:.3f}"

    if metric_key == "EligibleTrials":
        return f"{value:.1f}"

    return f"{value:.4f}"


def add_box_metric(
    fig,
    errors,
    metric_spec,
    available_models,
    showlegend=False
):
    metric_key = metric_spec["key"]
    row = metric_spec["row"]
    col = metric_spec["col"]

    for model in available_models:
        vals = get_model_values(errors, model, metric_key)

        if len(vals) == 0:
            continue

        fig.add_trace(
            go.Box(
                y=vals,
                name=model,
                marker_color=MODEL_COLORS.get(model, "#444444"),
                boxmean=True,
                boxpoints="all",
                jitter=0.20,
                pointpos=0,
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    f"{metric_spec['title']}: %{{y:.5f}}"
                    "<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    fig.update_yaxes(
        title_text=metric_spec["ylabel"],
        automargin=True,
        row=row,
        col=col
    )

    fig.update_xaxes(
        tickangle=25,
        automargin=True,
        row=row,
        col=col
    )


def build_top_summary_table(errors, table_metrics, title="Top-3 Model Summary"):
    metric_names = []
    first_models = []
    second_models = []
    third_models = []

    for metric_key, label, higher_is_better in table_metrics:
        top = get_top_models(
            errors,
            metric_key,
            higher_is_better=higher_is_better,
            top_n=3
        )

        formatted = [
            f"{model}<br>({format_metric_value(metric_key, val)})"
            for model, val in top
        ]

        while len(formatted) < 3:
            formatted.append("")

        metric_names.append(label)
        first_models.append(formatted[0])
        second_models.append(formatted[1])
        third_models.append(formatted[2])

    return go.Table(
        header=dict(
            values=["Metric", "1st", "2nd", "3rd"],
            fill_color="#E5ECF6",
            align="center",
            font=dict(size=13, color="black")
        ),
        cells=dict(
            values=[metric_names, first_models, second_models, third_models],
            fill_color="white",
            align="center",
            font=dict(size=12, color="black"),
            height=34
        )
    )


def add_mean_roc_curve(fig, errors, available_models, row=1, col=1):
    base_fpr = np.linspace(0, 1, 250)

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="black", width=2),
            name="Chance",
            hovertemplate="Chance line<extra></extra>",
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    for model in available_models:
        fpr_key = f"{model}_FPR"
        tpr_key = f"{model}_TPR"

        if fpr_key not in errors or tpr_key not in errors:
            continue

        fprs = errors[fpr_key]
        tprs = errors[tpr_key]

        if len(fprs) == 0 or len(tprs) == 0:
            continue

        interp_tprs = []

        for fpr, tpr in zip(fprs, tprs):
            try:
                fpr = np.asarray(fpr, dtype=float)
                tpr = np.asarray(tpr, dtype=float)

                interp_tpr = np.interp(base_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tpr[-1] = 1.0
                interp_tprs.append(interp_tpr)

            except Exception:
                pass

        if len(interp_tprs) == 0:
            continue

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = metric_mean(errors, model, "AUC")

        fig.add_trace(
            go.Scatter(
                x=base_fpr,
                y=mean_tpr,
                mode="lines",
                name=f"{model}, AUC={mean_auc:.3f}",
                line=dict(color=MODEL_COLORS.get(model, "#444444"), width=2.5),
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "FPR: %{x:.3f}<br>"
                    "Mean TPR: %{y:.3f}<br>"
                    f"Mean AUC: {mean_auc:.3f}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="False positive rate", range=[0, 1], row=row, col=col)
    fig.update_yaxes(title_text="True positive rate", range=[0, 1], row=row, col=col)


def add_calibration_curve(fig, errors, available_models, row=1, col=2, n_bins=10):
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="black", width=2),
            name="Perfect calibration",
            hovertemplate="Perfect calibration<extra></extra>",
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    for model in available_models:
        proba_key = f"{model}_Proba"
        label_key = f"{model}_TrueLabels"

        if proba_key not in errors or label_key not in errors:
            continue

        if len(errors[proba_key]) == 0 or len(errors[label_key]) == 0:
            continue

        try:
            y_true = np.concatenate(errors[label_key]).astype(int)
            proba = np.concatenate(errors[proba_key]).astype(float)

            valid = np.isfinite(proba)
            y_true = y_true[valid]
            proba = proba[valid]

            prob_true, prob_pred = calibration_curve(
                y_true,
                proba,
                n_bins=n_bins,
                strategy="uniform"
            )

            brier_mean = metric_mean(errors, model, "Brier")

            fig.add_trace(
                go.Scatter(
                    x=prob_pred,
                    y=prob_true,
                    mode="lines+markers",
                    name=f"{model}, Brier={brier_mean:.3f}",
                    line=dict(color=MODEL_COLORS.get(model, "#444444"), width=2.5),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{model}</b><br>"
                        "Predicted probability: %{x:.3f}<br>"
                        "Observed probability: %{y:.3f}<br>"
                        f"Mean Brier: {brier_mean:.3f}"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

        except Exception as e:
            print(f"Calibration skipped for {model}: {e}")

    fig.update_xaxes(title_text="Predicted probability", range=[0, 1], row=row, col=col)
    fig.update_yaxes(title_text="Observed probability", range=[0, 1], row=row, col=col)


def apply_ppt_layout(
    fig,
    title,
    subtitle="",
    show_legend=True,
    width=None,
    height=None,
    bottom_margin=90
):
    if width is None:
        width = PPT_WIDTH

    if height is None:
        height = PPT_HEIGHT

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            x=0.5,
            xanchor="center",
            font=dict(size=24)
        ),
        width=width,
        height=height,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=70, r=70, t=115, b=bottom_margin),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.13,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ) if show_legend else None,
    )

    fig.update_annotations(font=dict(size=14))

    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="rgba(0,0,0,0.12)",
        automargin=True,
        tickfont=dict(size=10),
        title_font=dict(size=12)
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="rgba(0,0,0,0.12)",
        automargin=True,
        tickfont=dict(size=10),
        title_font=dict(size=12)
    )

    return fig



def make_predictive_performance_slide(
    errors,
    scenario_name="Scenario",
    sample_size=3000,
    output_dir=None,
    file_prefix="Model_Performance",
    save_html=True
):
    available_models = get_available_models(errors)
    subtitle = clean_scenario_subtitle(scenario_name, sample_size)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "AUC",
            "Brier Score",
            "Balanced Accuracy",
            "Sensitivity",
            "Precision",
            "Top-3 Model Summary"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.20,
        horizontal_spacing=0.10,
    )

    for metric_spec in PREDICTIVE_METRICS:
        add_box_metric(
            fig,
            errors,
            metric_spec,
            available_models,
            showlegend=False
        )

    fig.add_trace(
        build_top_summary_table(
            errors,
            table_metrics=PREDICTIVE_TABLE_METRICS
        ),
        row=2,
        col=3
    )

    apply_ppt_layout(
        fig,
        title="Predictive Performance Across Models",
        subtitle=subtitle,
        show_legend=False,
        width=PPT_WIDTH,
        height=PPT_HEIGHT,
        bottom_margin=80
    )

    if save_html:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(
            output_dir,
            f"{file_prefix}_{scenario_name}_Slide1_Predictive_Performance_Clean.html"
        )
        fig.write_html(html_path)
        print(f"Saved clean predictive-performance slide:\n{html_path}")

    return fig


def make_runtime_efficiency_slide(
    errors,
    scenario_name="Scenario",
    sample_size=3000,
    output_dir=None,
    file_prefix="Model_Performance",
    save_html=True
):
    available_models = get_available_models(errors)
    subtitle = clean_scenario_subtitle(scenario_name, sample_size)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Actual Total Runtime",
            "Budgeted Total Runtime",
            "Energy Consumption",
            "Eligible Trials Within TabPFN Budget"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.20,
        horizontal_spacing=0.14,
    )

    for metric_spec in EFFICIENCY_METRICS:
        add_box_metric(
            fig,
            errors,
            metric_spec,
            available_models,
            showlegend=False
        )

    apply_ppt_layout(
        fig,
        title="Runtime and Efficiency Across Models",
        subtitle=subtitle,
        show_legend=False,
        width=PPT_WIDTH,
        height=PPT_HEIGHT,
        bottom_margin=80
    )

    if save_html:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(
            output_dir,
            f"{file_prefix}_{scenario_name}_Slide2_Runtime_Efficiency_Clean.html"
        )
        fig.write_html(html_path)
        print(f"Saved runtime-efficiency slide:\n{html_path}")

    return fig



def make_roc_calibration_slide_wide(
    errors,
    scenario_name="Scenario",
    sample_size=3000,
    output_dir=None,
    file_prefix="Model_Performance",
    save_html=True
):
    available_models = get_available_models(errors)
    subtitle = clean_scenario_subtitle(scenario_name, sample_size)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Mean ROC Curves",
            "Calibration Curves",
            "Top-3 Model Summary"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "table"}],
        ],
        horizontal_spacing=0.075,
        column_widths=[0.37, 0.37, 0.26],
    )

    add_mean_roc_curve(
        fig,
        errors,
        available_models,
        row=1,
        col=1
    )

    add_calibration_curve(
        fig,
        errors,
        available_models,
        row=1,
        col=2,
        n_bins=PLOT_CONFIG["calibration_bins"]
    )

    fig.add_trace(
        build_top_summary_table(
            errors,
            table_metrics=OVERALL_TABLE_METRICS
        ),
        row=1,
        col=3
    )

    apply_ppt_layout(
        fig,
        title="ROC, Calibration, and Overall Ranking",
        subtitle=subtitle,
        show_legend=True,
        width=ROC_CAL_WIDTH,
        height=ROC_CAL_HEIGHT,
        bottom_margin=155
    )

    fig.update_xaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=1)

    fig.update_xaxes(range=[0, 1], row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=1, col=2)

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.19,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )

    if save_html:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(
            output_dir,
            f"{file_prefix}_{scenario_name}_Slide3_ROC_Calibration_Summary_Wide.html"
        )
        fig.write_html(html_path)
        print(f"Saved wider ROC-calibration-summary slide:\n{html_path}")

    return fig



def ranking_line_text(errors, metric_key, label, higher_is_better=True):
    """
    Creates one compact ranking line from best to worst.
    Example: ROC/AUC: TabPFN=0.841 > CatBoost=0.833 > ...
    """
    rows = []

    for model in get_available_models(errors):
        val = metric_mean(errors, model, metric_key)
        if np.isfinite(val):
            rows.append((model, val))

    rows = sorted(rows, key=lambda x: x[1], reverse=higher_is_better)

    parts = [
        f"{model}={format_metric_value(metric_key, val)}"
        for model, val in rows
    ]

    return f"<b>{label}</b>: " + " &gt; ".join(parts)


def build_top_summary_table_roc_slide(errors):
    """
    Compact Top-3 summary table for the third slide.
    """
    table_metrics = [
        ("AUC", "AUC", True),
        ("Brier", "Brier", False),
        ("BalancedAccuracy", "Balanced Accuracy", True),
        ("Sensitivity", "Sensitivity", True),
        ("Precision", "Precision", True),
        ("BudgetedTotalRuntime", "Budgeted Runtime", False),
        ("ActualTotalRuntime", "Actual Runtime", False),
        ("Energy", "Energy", False),
    ]

    metric_names = []
    first_models = []
    second_models = []
    third_models = []

    for metric_key, label, higher_is_better in table_metrics:
        top = get_top_models(
            errors,
            metric_key,
            higher_is_better=higher_is_better,
            top_n=3
        )

        formatted = [
            f"{model}<br>({format_metric_value(metric_key, val)})"
            for model, val in top
        ]

        while len(formatted) < 3:
            formatted.append("")

        metric_names.append(label)
        first_models.append(formatted[0])
        second_models.append(formatted[1])
        third_models.append(formatted[2])

    return go.Table(
        columnwidth=[1.35, 1.25, 1.25, 1.25],
        header=dict(
            values=["Metric", "1st", "2nd", "3rd"],
            fill_color="#E5ECF6",
            align="center",
            font=dict(size=12, color="black")
        ),
        cells=dict(
            values=[metric_names, first_models, second_models, third_models],
            fill_color="white",
            align="center",
            font=dict(size=11, color="black"),
            height=25
        )
    )


def build_overall_top3_matrix_table(errors):
    """
    Overall Top-3 matrix table.
    Runtime column is removed.
    Metrics included:
    AUC, Brier, Balanced Accuracy, Sensitivity, Precision, Energy.
    """

    available_models = get_available_models(errors)

    preferred_order = MODEL_ORDER

    model_rows = [m for m in preferred_order if m in available_models]

    matrix_metrics = [
        ("AUC", "AUC", True),
        ("Brier", "Brier", False),
        ("BalancedAccuracy", "BalAcc", True),
        ("Sensitivity", "Sens", True),
        ("Precision", "Prec", True),
        ("Energy", "Energy", False),
    ]

    top3_sets = {}

    for metric_key, label, higher_is_better in matrix_metrics:
        top_models = get_top_models(
            errors,
            metric_key,
            higher_is_better=higher_is_better,
            top_n=3
        )
        top3_sets[metric_key] = set([m for m, v in top_models])

    model_col = []
    metric_cols = {label: [] for _, label, _ in matrix_metrics}
    overall_col = []

    font_colors = {"Model": [], "Overall": []}
    for _, label, _ in matrix_metrics:
        font_colors[label] = []

    for model in model_rows:
        model_col.append(f"<b>{model}</b>")
        overall_count = 0

        for metric_key, label, _ in matrix_metrics:
            if model in top3_sets[metric_key]:
                metric_cols[label].append("Top 3")
                font_colors[label].append("blue")
                overall_count += 1
            else:
                metric_cols[label].append("Not Top 3")
                font_colors[label].append("red")

        overall_col.append(f"<b>{overall_count}</b>")
        font_colors["Model"].append("black")
        font_colors["Overall"].append("black")

    values = [
        model_col,
        metric_cols["AUC"],
        metric_cols["Brier"],
        metric_cols["BalAcc"],
        metric_cols["Sens"],
        metric_cols["Prec"],
        metric_cols["Energy"],
        overall_col
    ]

    colors = [
        font_colors["Model"],
        font_colors["AUC"],
        font_colors["Brier"],
        font_colors["BalAcc"],
        font_colors["Sens"],
        font_colors["Prec"],
        font_colors["Energy"],
        font_colors["Overall"]
    ]

    return go.Table(
        columnwidth=[2.7, 0.62, 0.72, 0.72, 0.62, 0.62, 0.82, 0.82],
        header=dict(
            values=[
                "<b>Model</b>",
                "<b>AUC</b>",
                "<b>Brier</b>",
                "<b>BalAcc</b>",
                "<b>Sens</b>",
                "<b>Prec</b>",
                "<b>Energy</b>",
                "<b>Overall</b>"
            ],
            fill_color="#E5E5E5",
            align="center",
            font=dict(size=12, color="black"),
            line_color="black",
            line_width=1.4
        ),
        cells=dict(
            values=values,
            fill_color=[
                ["#F7F7F7"] * len(model_rows),
                ["white"] * len(model_rows),
                ["white"] * len(model_rows),
                ["white"] * len(model_rows),
                ["white"] * len(model_rows),
                ["white"] * len(model_rows),
                ["white"] * len(model_rows),
                ["#FFF7D6"] * len(model_rows),
            ],
            align="center",
            font=dict(
                size=13,
                color=colors
            ),
            line_color="black",
            line_width=1.0,
            height=26
        )
    )


def make_roc_calibration_slide(
    errors,
    scenario_name="Scenario",
    sample_size=3000,
    output_dir=None,
    file_prefix="Model_Performance",
    save_html=True
):
    available_models = get_available_models(errors)
    subtitle = clean_scenario_subtitle(scenario_name, sample_size)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Mean ROC Curves",
            "Calibration Curves",
            "Top-3 Model Summary",
            "",
            "",
            "Overall Top-3 Summary<br><sup>blue = Top 3 | red = Not Top 3</sup>"
        ],
        specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "xy", "rowspan": 2}, {"type": "table"}],
            [None, None, {"type": "table"}],
        ],
        horizontal_spacing=0.045,
        vertical_spacing=0.06,
        column_widths=[0.335, 0.335, 0.33],
        row_heights=[0.47, 0.53]
    )

    add_mean_roc_curve(
        fig,
        errors,
        available_models,
        row=1,
        col=1
    )

    add_calibration_curve(
        fig,
        errors,
        available_models,
        row=1,
        col=2,
        n_bins=PLOT_CONFIG["calibration_bins"]
    )

    fig.add_trace(
        build_top_summary_table_roc_slide(errors),
        row=1,
        col=3
    )

    fig.add_trace(
        build_overall_top3_matrix_table(errors),
        row=2,
        col=3
    )

    apply_ppt_layout(
        fig,
        title="ROC, Calibration, and Overall Ranking",
        subtitle=subtitle,
        show_legend=True,
        width=ROC_CAL_WIDTH,
        height=ROC_CAL_HEIGHT,
        bottom_margin=120
    )

    fig.update_xaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=1)

    fig.update_xaxes(range=[0, 1], row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=1, col=2)

    roc_line = ranking_line_text(
        errors,
        metric_key="AUC",
        label="ROC/AUC ranking",
        higher_is_better=True
    )

    calibration_line = ranking_line_text(
        errors,
        metric_key="Brier",
        label="Calibration/Brier ranking",
        higher_is_better=False
    )

    fig.add_annotation(
        text=roc_line,
        xref="paper",
        yref="paper",
        x=0.335,
        y=-0.105,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="center",
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.90)"
    )

    fig.add_annotation(
        text=calibration_line,
        xref="paper",
        yref="paper",
        x=0.335,
        y=-0.155,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="center",
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.90)"
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.235,
            xanchor="center",
            x=0.5,
            font=dict(size=8)
        ),
        margin=dict(l=45, r=45, t=85, b=150)
    )

    fig.update_annotations(font=dict(size=12))

    if save_html:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(
            output_dir,
            f"{file_prefix}_{scenario_name}_Slide3_ROC_Calibration_Summary_Compact_Fitted.html"
        )
        fig.write_html(html_path)
        print(f"Saved compact fitted ROC-calibration-summary slide:\n{html_path}")

    return fig



def load_errors_from_scenario_folder(RESULTS_DIR, FILE_PREFIX, scenario_name):
    scenario_dir = os.path.join(RESULTS_DIR, scenario_folder_name(scenario_name))
    scenario_prefix = scenario_file_prefix(scenario_name, FILE_PREFIX)
    raw_path = os.path.join(scenario_dir, f"{scenario_prefix}_RawResults.pkl")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw results not found:\n{raw_path}")

    with open(raw_path, "rb") as f:
        errors = pickle.load(f)

    return errors, scenario_dir



def _show_plotly_figure(fig, renderer=None):
    """Show a Plotly figure in notebooks/Colab without breaking batch runs."""
    if not SHOW_FIGURES:
        return

    renderer = renderer or PLOTLY_RENDERER

    try:
        if renderer:
            fig.show(renderer=renderer)
        else:
            fig.show()
    except Exception:
        try:
            fig.show()
        except Exception as exc:
            print(f"WARNING: Could not display figure inline: {exc}")


def save_plotly_figure_for_ppt(
    fig,
    output_dir,
    base_filename,
    save_html=None,
    save_png=None,
    save_svg=None,
    save_jpeg=None,
    show_figures=None,
    image_scale=None,
):
    """
    Save a Plotly figure as HTML and static PowerPoint-ready images.

    PNG/JPEG/SVG export uses Plotly's Kaleido backend. If Kaleido is missing,
    this function warns and continues instead of failing the whole pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    if save_html is None:
        save_html = SAVE_HTML
    if save_png is None:
        save_png = SAVE_PNG
    if save_svg is None:
        save_svg = SAVE_SVG
    if save_jpeg is None:
        save_jpeg = SAVE_JPEG
    if show_figures is None:
        show_figures = SHOW_FIGURES
    if image_scale is None:
        image_scale = IMAGE_SCALE

    saved_paths = {}
    width = fig.layout.width or PPT_WIDTH
    height = fig.layout.height or PPT_HEIGHT

    if save_html:
        html_path = os.path.join(output_dir, f"{base_filename}.html")
        try:
            fig.write_html(html_path)
            saved_paths["html"] = html_path
            print(f"Saved HTML:\n{html_path}")
        except Exception as exc:
            saved_paths["html_error"] = str(exc)
            print(f"WARNING: HTML export failed for {base_filename}: {exc}")

    def _write_static(fmt, extension):
        path = os.path.join(output_dir, f"{base_filename}.{extension}")
        try:
            fig.write_image(
                path,
                format=fmt,
                width=width,
                height=height,
                scale=image_scale if fmt in {"png", "jpeg"} else 1,
            )
            saved_paths[extension] = path
            print(f"Saved {extension.upper()}:\n{path}")
        except Exception as exc:
            saved_paths[f"{extension}_error"] = str(exc)
            print(
                f"WARNING: {extension.upper()} export failed for "
                f"{base_filename}."
            )
            print("   Static export needs Kaleido. In Colab run: !pip install -U kaleido")
            print(f"   Error: {exc}")

    if save_png:
        _write_static("png", "png")

    if save_jpeg:
        _write_static("jpeg", "jpeg")

    if save_svg:
        _write_static("svg", "svg")

    if show_figures:
        _show_plotly_figure(fig)

    return saved_paths


def create_all_plotly_slides_for_scenario(
    scenario_name,
    results_dir=None,
    file_prefix=None,
    sample_size=None,
    plot_subfolder_name=None,
    save_html=None,
    show_figures=None,
    save_wide_roc_version=None,
):
    """Load one scenario's RawResults.pkl, build 3 figures, show them, and save HTML/static images."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    if file_prefix is None:
        file_prefix = FILE_PREFIX
    if sample_size is None:
        sample_size = TOTAL_N
    if plot_subfolder_name is None:
        plot_subfolder_name = PLOT_SUBFOLDER_NAME
    if save_html is None:
        save_html = SAVE_HTML
    if show_figures is None:
        show_figures = SHOW_FIGURES
    if save_wide_roc_version is None:
        save_wide_roc_version = SAVE_WIDE_ROC_CALIBRATION_VERSION

    errors_loaded, scenario_dir_loaded = load_errors_from_scenario_folder(
        RESULTS_DIR=results_dir,
        FILE_PREFIX=file_prefix,
        scenario_name=scenario_name,
    )

    output_dir = os.path.join(scenario_dir_loaded, plot_subfolder_name)
    os.makedirs(output_dir, exist_ok=True)

    fig1 = make_predictive_performance_slide(
        errors=errors_loaded,
        scenario_name=scenario_name,
        sample_size=sample_size,
        output_dir=output_dir,
        file_prefix=file_prefix,
        save_html=False,
    )

    fig2 = make_runtime_efficiency_slide(
        errors=errors_loaded,
        scenario_name=scenario_name,
        sample_size=sample_size,
        output_dir=output_dir,
        file_prefix=file_prefix,
        save_html=False,
    )

    fig3 = make_roc_calibration_slide(
        errors=errors_loaded,
        scenario_name=scenario_name,
        sample_size=sample_size,
        output_dir=output_dir,
        file_prefix=file_prefix,
        save_html=False,
    )

    fig3_wide = None
    if save_wide_roc_version:
        fig3_wide = make_roc_calibration_slide_wide(
            errors=errors_loaded,
            scenario_name=scenario_name,
            sample_size=sample_size,
            output_dir=output_dir,
            file_prefix=file_prefix,
            save_html=False,
        )

    slide1_paths = save_plotly_figure_for_ppt(
        fig1,
        output_dir,
        f"{file_prefix}_{scenario_name}_Slide1_Predictive_Performance_Clean",
        save_html=save_html,
        show_figures=show_figures,
    )

    slide2_paths = save_plotly_figure_for_ppt(
        fig2,
        output_dir,
        f"{file_prefix}_{scenario_name}_Slide2_Runtime_Efficiency_Clean",
        save_html=save_html,
        show_figures=show_figures,
    )

    slide3_paths = save_plotly_figure_for_ppt(
        fig3,
        output_dir,
        f"{file_prefix}_{scenario_name}_Slide3_ROC_Calibration_Summary_Compact_Fitted",
        save_html=save_html,
        show_figures=show_figures,
    )

    fig3_wide_paths = None
    if fig3_wide is not None:
        fig3_wide_paths = save_plotly_figure_for_ppt(
            fig3_wide,
            output_dir,
            f"{file_prefix}_{scenario_name}_Slide3_ROC_Calibration_Summary_Wide",
            save_html=save_html,
            show_figures=show_figures,
        )

    return {
        "scenario_name": scenario_name,
        "output_dir": output_dir,
        "fig1_predictive": fig1,
        "fig2_runtime_efficiency": fig2,
        "fig3_roc_calibration_compact": fig3,
        "fig3_roc_calibration_wide": fig3_wide,
        "slide1_paths": slide1_paths,
        "slide2_paths": slide2_paths,
        "slide3_paths": slide3_paths,
        "slide3_wide_paths": fig3_wide_paths,
    }


RESULT_METRICS = [
    "BalancedAccuracy",
    "Brier",
    "AUC",
    "TotalRuntime",
    "ActualTotalRuntime",
    "BudgetedTotalRuntime",
    "OptunaTuningTimeCapped",
    "FPR",
    "TPR",
    "Sensitivity",
    "Precision",
    "Proba",
    "TrueLabels",
    "Energy",
    "TabPFNBudget",
    "ActualOptunaTuningTime",
    "CompletedTrials",
    "EligibleTrials",
    "SelectedAUC",
    "FinalFitPredictTime",

    "StrictBudgetBalancedAccuracy",
    "StrictBudgetAUC",
    "StrictBudgetBrier",
    "StrictBudgetSensitivity",
    "StrictBudgetPrecision",
    "PredictionBudgetBalancedAccuracy",
    "PredictionBudgetAUC",
    "PredictionBudgetBrier",
    "PredictionBudgetSensitivity",
    "PredictionBudgetPrecision",
    "TabPFNFitTime",
    "TabPFNPredictProbaTime",
    "TabPFNEndToEndFitPredictProbaTime",
    "TabPFNStrictEndToEndBudgetPassed",
    "TabPFNPredictProbaBudgetPassed",
    "TabPFNStrictEndToEndOverrun",
    "TabPFNPredictProbaOverrun",
    "TabPFNFullTrainN",
    "TabPFNBudgetedContextSampleSize",
    "TabPFNContextNUsed",
    "TabPFNContextFractionUsed",
    "TabPFNContextSearchAttempts",
    "TabPFNFullPredictionN",
    "TabPFNBudgetedPredictionSampleSize",
    "TabPFNPredictionFractionUsed",
    "TabPFNPredictionSearchAttempts",
    "TabPFNPredictProbaBudgetDelta",
    "TabPFNPredictProbaBudgetRemaining",
    "TabPFNPredictProbaBudgetUseRatio",
    "TabPFNTotalRuntimeBudgetDelta",
    "TabPFNTotalRuntimeBudgetRemaining",
    "TabPFNMLReferenceBudget",
    "TabPFNEffectiveTimeBudget",
    "TabPFNEffectiveBudgetMultiplier",
    "TabPFNMinContextRequested",
    "TabPFNMinContextTarget",
    "TabPFNMinContextRuntime",
    "TabPFNMinContextBudgetMultiplierApplied",
    "TabPFNMinContextRequirementMet",
    "TabPFNOriginalMLBudgetTotalRuntimeDelta",
    "TabPFNOriginalMLBudgetTotalRuntimeRemaining",
]


def initialize_errors(model_names):
    return {
        f"{model}_{metric}": []
        for model in model_names
        for metric in RESULT_METRICS
    }


split_cols = [
    "Scenario",
    "MonteCarlo_Iteration",
    "Pipeline_Version",
    "Sampling_Strategy",
    "Split_Strategy",
    "Split_Level",
    "Split_Seed",
    "Split_Valid",
    "Split_Fingerprint",
    "Split_Candidate_Method",
    "Split_Candidates_Evaluated",
    "Split_Candidate_Score",
    "Original_Class1_Prevalence",
    "Target_Class1_Prevalence",
    "Sample_Class1_Prevalence",
    "Train_Class1_Prevalence",
    "Test_Class1_Prevalence",
    "InnerTrain_Class1_Prevalence",
    "Validation_Class1_Prevalence",
    "Requested_Total_N",
    "Actual_Total_N",
    "Total_N_Deviation",
    "Requested_Train_Fraction",
    "Actual_Train_Fraction",
    "Train_Fraction_Deviation",
    "Sample_Prevalence_Deviation",
    "Train_Prevalence_Deviation",
    "Test_Prevalence_Deviation",
    "Train_N",
    "Test_N",
    "Train_Class0_N",
    "Train_Class1_N",
    "Test_Class0_N",
    "Test_Class1_N",
    "N_Groups_Total",
    "N_Groups_Train",
    "N_Groups_Test",
    "Train_Test_Row_Overlap_N",
    "Train_Test_Group_Overlap_N",
    "Group_Subset_Candidates_Evaluated",
    "Group_Subset_Score",
    "Group_Subset_N_Deviation",
    "Inner_Split_Strategy",
    "Inner_Split_Valid",
    "Inner_Split_Fingerprint",
    "Inner_Split_Candidate_Method",
    "Inner_Split_Candidates_Evaluated",
    "InnerTrain_N",
    "Validation_N",
    "InnerTrain_Class0_N",
    "InnerTrain_Class1_N",
    "Validation_Class0_N",
    "Validation_Class1_N",
    "Inner_Train_Validation_Row_Overlap_N",
    "Inner_Train_Validation_Group_Overlap_N",
    "Low_Class_Count_Warning",
    "Temporal_Window",
    "Temporal_Gap_Rows",
    "Temporal_Train_Max",
    "Temporal_Test_Min",
    "Temporal_Order_Valid",
    "Temporal_Group_Overlap_N",
    "Preprocessing_Mode",
    "Preprocessing_Fit_Outside_Model_Timing",
    "Preprocessing_Time_Seconds",
    "Preprocessing_Input_Features",
    "Preprocessing_Output_Features",
    "Preprocessing_Outer_Fit_N",
    "Preprocessing_Inner_Fit_N",
]



def export_dashboard_bundle(
    scenario_name,
    scenario_dir,
    scenario_prefix,
    errors,
    iter_df,
    summary_df,
    active_model_names,
    sample_size,
    calibration_bins=10,
):
    """Write everything the three HTML dashboards need to regenerate, for ONE
    scenario at ONE sample size, in a stable schema.

    Outputs (in scenario_dir):
      {prefix}_dashboard.json   - human-readable, what the dashboards load
      {prefix}_dashboard.pkl    - same payload, lossless (numpy-friendly)

    The JSON payload has four arrays, matching the keys the dashboards use:
      - "iteration_level": one row per (model, iteration) with the raw metrics
            (AUC, BalancedAccuracy, Brier, Sensitivity, Precision, Time_Seconds,
             Energy_kWh, Kneedle_Selection_Time, Time_Without_Kneedle, iteration)
      - "summary": one row per (model) with *_Mean / *_SD / *_CI95_pm and
            N_Iterations (drives every "with sample size" + trade-off + table view)
      - "roc_points": mean-ROC polyline per model (FalsePositiveRate,
            MeanTruePositiveRate, Curve_Role="ROC")
      - "calibration_points": reliability polyline per model
            (PredictedProbability, ObservedProbability, MeanBrier, Curve_Role="Calibration")

    Each row carries Scenario and Sample_Size so multiple scenarios / sample
    sizes can be concatenated into one long table (exactly how the dashboards
    ingest several runs).
    """
    from sklearn.calibration import calibration_curve

    metric_cols = [
        "AUC", "BalancedAccuracy", "Brier", "Sensitivity", "Precision",
        "Time_Seconds", "Energy_kWh", "Kneedle_Selection_Time",
        "Time_Without_Kneedle",
    ]

    it = iter_df.copy()
    it["Scenario"] = scenario_name
    it["Sample_Size"] = int(sample_size)
    iteration_records = it.to_dict(orient="records")

    sm = summary_df.copy()
    if not sm.empty:
        sm["Scenario"] = scenario_name
        sm["Sample_Size"] = int(sample_size)
    summary_records = sm.to_dict(orient="records")

    base_fpr = np.linspace(0, 1, 250)
    roc_records = []
    for model in active_model_names:
        fpr_key, tpr_key = f"{model}_FPR", f"{model}_TPR"
        if fpr_key not in errors or tpr_key not in errors:
            continue
        interp = []
        for fpr, tpr in zip(errors[fpr_key], errors[tpr_key]):
            try:
                fpr = np.asarray(fpr, float); tpr = np.asarray(tpr, float)
                t = np.interp(base_fpr, fpr, tpr); t[0] = 0.0; t[-1] = 1.0
                interp.append(t)
            except Exception:
                pass
        if not interp:
            continue
        mean_tpr = np.mean(interp, axis=0)
        for x, yv in zip(base_fpr, mean_tpr):
            roc_records.append({
                "Scenario": scenario_name, "Sample_Size": int(sample_size),
                "Model": model, "Curve_Role": "ROC",
                "FalsePositiveRate": float(x), "MeanTruePositiveRate": float(yv),
            })

    cal_records = []
    for model in active_model_names:
        y_key, p_key = f"{model}_TrueLabels", f"{model}_Proba"
        if y_key not in errors or p_key not in errors:
            continue
        try:
            y_true = np.concatenate(errors[y_key]).astype(int)
            proba = np.concatenate(errors[p_key]).astype(float)
            valid = np.isfinite(proba)
            y_true, proba = y_true[valid], proba[valid]
            if len(np.unique(y_true)) < 2:
                continue
            prob_true, prob_pred = calibration_curve(
                y_true, proba, n_bins=calibration_bins, strategy="uniform"
            )
            bmean = float(np.mean(errors.get(f"{model}_Brier", [np.nan])))
            for pp, pt in zip(prob_pred, prob_true):
                cal_records.append({
                    "Scenario": scenario_name, "Sample_Size": int(sample_size),
                    "Model": model, "Curve_Role": "Calibration",
                    "PredictedProbability": float(pp),
                    "ObservedProbability": float(pt),
                    "MeanBrier": bmean,
                })
        except Exception:
            pass

    payload = {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario_name,
        "sample_size": int(sample_size),
        "models": list(active_model_names),
        "metrics": metric_cols,
        "iteration_level": iteration_records,
        "summary": summary_records,
        "roc_points": roc_records,
        "calibration_points": cal_records,
    }

    json_path = os.path.join(scenario_dir, f"{scenario_prefix}_dashboard.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)

    pkl_path = os.path.join(scenario_dir, f"{scenario_prefix}_dashboard.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"   Dashboard bundle saved:\n      {json_path}\n      {pkl_path}")
    return json_path, pkl_path


def _json_safe(o):
    """Fallback serializer so numpy types land cleanly in JSON."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_safe)


def write_df_json(df, path):
    write_json(path, df.to_dict(orient="records"))


def build_dashboard_datasets(output_root, out_dir=None, budget_scenarios=None, nobudget_scenarios=None):
    """Walk the pipeline output tree, collect every per-(scenario, sample_size)
    `*_dashboard.json`, and assemble the COMBINED datasets the three HTML
    dashboards need (all sample sizes concatenated).

    Writes into out_dir (default: output_root/Combined_Dashboard_Datasets):
      combined_iteration_level.json / .csv
      combined_summary.json / .csv
      combined_roc_points.json / .csv
      combined_calibration_points.json / .csv
      combined_budget_vs_nobudget.json / .csv   (paired deltas + significance)

    budget_scenarios / nobudget_scenarios: optional lists of scenario names that
    belong to each side of the comparison. If omitted, scenarios whose name
    contains "nobudget"/"no_budget"/"no-budget" (case-insensitive) are treated as
    No-Budget and the rest as Budget. The paired layer is computed only for
    (model, sample_size) pairs present on BOTH sides.
    """
    import glob

    if out_dir is None:
        out_dir = os.path.join(output_root, "Combined_Dashboard_Datasets")
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(output_root, "**", "*_dashboard.json"), recursive=True)
    if not files:
        print(f"WARNING: No *_dashboard.json files found under {output_root}.")
        return {}

    iter_all, summ_all, roc_all, cal_all = [], [], [], []
    for fp in files:
        try:
            with open(fp) as f:
                p = json.load(f)
        except Exception as e:
            print(f"   WARNING: skipping {fp}: {e}")
            continue
        iter_all.extend(p.get("iteration_level", []))
        summ_all.extend(p.get("summary", []))
        roc_all.extend(p.get("roc_points", []))
        cal_all.extend(p.get("calibration_points", []))

    iter_df = pd.DataFrame(iter_all)
    summ_df = pd.DataFrame(summ_all)
    roc_df = pd.DataFrame(roc_all)
    cal_df = pd.DataFrame(cal_all)

    def _dump(df, stem):
        df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
        write_df_json(df, os.path.join(out_dir, f"{stem}.json"))

    _dump(iter_df, "combined_iteration_level")
    _dump(summ_df, "combined_summary")
    _dump(roc_df, "combined_roc_points")
    _dump(cal_df, "combined_calibration_points")

    cmp_df = pd.DataFrame()
    if not summ_df.empty and "Scenario" in summ_df.columns:
        scen = summ_df["Scenario"].unique().tolist()

        def is_nobudget(name):
            n = str(name).lower().replace(" ", "")
            return ("nobudget" in n) or ("no_budget" in n) or ("no-budget" in n)

        if nobudget_scenarios is None:
            nobudget_scenarios = [s for s in scen if is_nobudget(s)]
        if budget_scenarios is None:
            budget_scenarios = [s for s in scen if s not in nobudget_scenarios]

        metric_bases = ["AUC", "BalancedAccuracy", "Brier", "Sensitivity",
                        "Precision", "Time_Seconds", "Energy_kWh"]

        b = summ_df[summ_df["Scenario"].isin(budget_scenarios)]
        nb = summ_df[summ_df["Scenario"].isin(nobudget_scenarios)]

        rows = []
        if not b.empty and not nb.empty:
            keys = ["Model", "Sample_Size"]
            merged = pd.merge(b, nb, on=keys, suffixes=("_B", "_NB"))
            for _, r in merged.iterrows():
                row = {"Model": r["Model"], "Sample_Size": int(r["Sample_Size"])}
                for m in metric_bases:
                    bm, bsd, bci = f"{m}_Mean_B", f"{m}_SD_B", f"{m}_CI95_pm_B"
                    nm, nsd, nci = f"{m}_Mean_NB", f"{m}_SD_NB", f"{m}_CI95_pm_NB"
                    if bm in r and nm in r:
                        b_mean, nb_mean = r.get(bm), r.get(nm)
                        b_ci, nb_ci = r.get(bci, np.nan), r.get(nci, np.nan)
                        row[f"{m}_B_mean"] = b_mean
                        row[f"{m}_NB_mean"] = nb_mean
                        row[f"{m}_B_sd"] = r.get(bsd, np.nan)
                        row[f"{m}_NB_sd"] = r.get(nsd, np.nan)
                        row[f"{m}_B_ci"] = b_ci
                        row[f"{m}_NB_ci"] = nb_ci
                        try:
                            row[f"{m}_delta"] = float(nb_mean) - float(b_mean)
                        except Exception:
                            row[f"{m}_delta"] = np.nan
                        try:
                            lo_b, hi_b = float(b_mean) - float(b_ci), float(b_mean) + float(b_ci)
                            lo_n, hi_n = float(nb_mean) - float(nb_ci), float(nb_mean) + float(nb_ci)
                            row[f"{m}_CI95_Nonoverlap_Flag"] = bool(hi_b < lo_n or hi_n < lo_b)
                        except Exception:
                            row[f"{m}_CI95_Nonoverlap_Flag"] = False
                        lower_is_better = m in ("Brier", "Time_Seconds", "Energy_kWh")
                        try:
                            if lower_is_better:
                                row[f"{m}_budget_advantage"] = bool(float(b_mean) < float(nb_mean))
                            else:
                                row[f"{m}_budget_advantage"] = bool(float(b_mean) > float(nb_mean))
                        except Exception:
                            row[f"{m}_budget_advantage"] = None
                rows.append(row)
        cmp_df = pd.DataFrame(rows)
        _dump(cmp_df, "combined_budget_vs_nobudget")

    print(f"Combined dashboard datasets written to {out_dir}:")
    print(f"   iteration rows={len(iter_df)}, summary rows={len(summ_df)}, "
          f"roc rows={len(roc_df)}, calibration rows={len(cal_df)}, "
          f"comparison rows={len(cmp_df)}")
    return {
        "iteration_level": iter_df, "summary": summ_df,
        "roc_points": roc_df, "calibration_points": cal_df,
        "budget_vs_nobudget": cmp_df,
        "paths": {
            "out_dir": out_dir,
            "combined_iteration_level_csv": os.path.join(out_dir, "combined_iteration_level.csv"),
            "combined_iteration_level_json": os.path.join(out_dir, "combined_iteration_level.json"),
            "combined_summary_csv": os.path.join(out_dir, "combined_summary.csv"),
            "combined_summary_json": os.path.join(out_dir, "combined_summary.json"),
            "combined_roc_points_csv": os.path.join(out_dir, "combined_roc_points.csv"),
            "combined_roc_points_json": os.path.join(out_dir, "combined_roc_points.json"),
            "combined_calibration_points_csv": os.path.join(out_dir, "combined_calibration_points.csv"),
            "combined_calibration_points_json": os.path.join(out_dir, "combined_calibration_points.json"),
            "combined_budget_vs_nobudget_csv": os.path.join(out_dir, "combined_budget_vs_nobudget.csv"),
            "combined_budget_vs_nobudget_json": os.path.join(out_dir, "combined_budget_vs_nobudget.json"),
        },
    }


def run_monte_carlo(X, y, config, groups=None, timestamps=None):
    """Run the full TabPFN-budgeted Monte Carlo simulation.

    X and y are passed explicitly, so this module never depends on global
    notebook variables named X or y.

    groups : optional 1-D entity identifiers aligned to X. In auto mode their
        presence selects strict group-aware stratification for outer and inner
        partitions. No grouped split silently falls back to row-level splitting.
    timestamps : optional 1-D ordering values aligned to X. They are used only
        when config["splitting"]["strategy"] == "temporal".

    The pipeline itself is dataset-agnostic: all dataset-specific preparation
    (pairing, leakage-column removal, feature engineering, building `groups`)
    must be done OUTSIDE this function, which only receives the final X, y, and
    optional groups.
    """
    load_config(config)

    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if int(X.shape[0]) == 0 or int(X.shape[1]) == 0:
        raise ValueError("X must contain at least one row and one feature column.")
    if isinstance(X, pd.DataFrame) and not X.columns.is_unique:
        raise ValueError("X DataFrame column names must be unique.")

    y_raw = np.asarray(y).ravel()
    if len(y_raw) != int(X.shape[0]):
        raise ValueError("X and y must contain the same number of rows.")
    if pd.isna(y_raw).any():
        raise ValueError("y contains missing labels.")
    labels = set(np.unique(y_raw).tolist())
    if labels != {0, 1}:
        raise ValueError(
            "V12 requires exactly two binary labels coded as 0 and 1; "
            f"received {sorted(labels)}."
        )
    y_array = y_raw.astype(int, copy=False)

    split_config = RUN_CONFIG.get("splitting", {})
    preprocessing_config = RUN_CONFIG.get("preprocessing", {})
    preprocessing_mode = _resolve_preprocessing_mode(X, preprocessing_config)
    if preprocessing_mode == "raw_dataframe":
        X_data = X.copy()
    else:
        X_data = _as_finite_float32(X, "X")

    groups_array = None
    if groups is not None:
        groups_raw = np.asarray(groups).ravel()
        if len(groups_raw) != len(y_array):
            raise ValueError("groups must contain exactly one identifier per row.")
        if pd.isna(groups_raw).any():
            raise ValueError("groups contains missing identifiers.")
        groups_array, _ = pd.factorize(groups_raw, sort=False)
        groups_array = groups_array.astype(np.int64, copy=False)

    timestamps_array = None
    if timestamps is not None:
        timestamps_array = np.asarray(timestamps).ravel()
        if len(timestamps_array) != len(y_array):
            raise ValueError("timestamps must contain exactly one value per row.")
        if pd.isna(timestamps_array).any():
            raise ValueError("timestamps contains missing values.")

    requested_strategy = str(split_config.get("strategy", "auto")).lower()
    strategy_aliases = {
        "group": "stratified_group",
        "grouped": "stratified_group",
        "group_stratified": "stratified_group",
        "row": "stratified",
    }
    requested_strategy = strategy_aliases.get(requested_strategy, requested_strategy)
    if requested_strategy == "auto":
        use_groups = bool(split_config.get("group_aware", True)) and groups_array is not None
        split_strategy = "stratified_group" if use_groups else "stratified"
    else:
        split_strategy = requested_strategy
    if split_strategy not in {"stratified", "stratified_group", "temporal"}:
        raise ValueError(
            "splitting.strategy must be 'auto', 'stratified', "
            "'stratified_group', or 'temporal'."
        )
    if bool(split_config.get("require_groups", False)) and groups_array is None:
        raise ValueError("splitting.require_groups=True but no groups were supplied.")
    if split_strategy == "stratified_group" and groups_array is None:
        raise ValueError("splitting.strategy='stratified_group' requires groups.")
    if split_strategy == "temporal" and timestamps_array is None:
        raise ValueError("splitting.strategy='temporal' requires timestamps.")
    use_group_split = split_strategy == "stratified_group"

    print(f"V12 split strategy: {split_strategy} | preprocessing: {preprocessing_mode}")
    if use_group_split:
        print(
            f"Strict group isolation enabled: {len(np.unique(groups_array))} groups "
            f"across {len(y_array)} rows."
        )

    print("=" * 100)
    print("DATASET CHECK")
    print("=" * 100)
    print(f"Total available samples: {len(y_array):,}")
    print(f"Original class-0 prevalence: {np.mean(y_array == 0) * 100:.2f}%")
    print(f"Original class-1 prevalence: {np.mean(y_array == 1) * 100:.2f}%")
    print(f"Configured sample size per iteration: {RUN_CONFIG['total_n']:,}")
    print(f"Configured train fraction: {RUN_CONFIG['train_frac']:.2f}")
    print(f"Configured iterations: {RUN_CONFIG['iterations']}")
    print(f"Sampling strategy: {RUN_CONFIG.get('sampling', {}).get('strategy', 'original_prevalence')}")
    print(f"Split strategy: {split_strategy}")
    print(f"Preprocessing mode: {preprocessing_mode}")
    print("=" * 100)

    all_summaries = []
    all_iteration_metrics = []
    all_budget_timing = []
    all_selection_records = []
    all_final_fit_timing = []
    all_optuna_trials = []
    all_tabpfn_budget_evaluations = []



    for scenario_name, scenario_cfg in SCENARIO_CONFIGS.items():

        set_active_scenario(scenario_name)

        active_model_names = get_enabled_model_names()

        budgeting_enabled = scenario_cfg.get(
            "budgeting_enabled",
            RUN_CONFIG.get("budgeting", {}).get("enabled", True)
        )
        external_runtime_budget = scenario_cfg.get(
            "external_runtime_budget_seconds"
        )
        if external_runtime_budget is not None:
            external_runtime_budget = float(external_runtime_budget)
            if (
                not np.isfinite(external_runtime_budget)
                or external_runtime_budget <= 0
            ):
                raise ValueError(
                    "external_runtime_budget_seconds must be a finite positive "
                    "number when supplied."
                )
            budgeting_enabled = True

        if budgeting_enabled and external_runtime_budget is None:
            budget_model = scenario_cfg.get(
                "budget_reference_model",
                RUN_CONFIG["budget_reference_model"]
            )
        elif external_runtime_budget is not None:
            budget_model = "__external_runtime_budget__"
        else:
            budget_model = None
        set_active_budget_reference(budget_model)

        if budgeting_enabled and external_runtime_budget is None:
            assert budget_model in active_model_names, \
                f"{budget_model} must be enabled in scenario {scenario_name}."

        for model_name in active_model_names:
            assert model_name in MODEL_RUNNERS, \
                f"{model_name} is enabled but has no function in MODEL_RUNNERS."

        reference_is_tuned = bool(
            get_model_cfg(budget_model).get("tuned_by_optuna", False)
        ) if (
            budgeting_enabled and external_runtime_budget is None
        ) else False
        if not budgeting_enabled:
            resolved_budget_basis = "no_budgeting"
            reference_budget_source_field = None
        elif external_runtime_budget is not None:
            resolved_budget_basis = "external_persisted_reference_budget"
            reference_budget_source_field = (
                "external_runtime_budget_seconds"
            )
        elif reference_is_tuned:
            resolved_budget_basis = "reference_tuning_runtime"
            reference_budget_source_field = (
                "Actual_Optuna_Tuning_Time_Seconds"
            )
        else:
            non_tuned_reference_basis = RUN_CONFIG.get(
                "budgeting", {}
            ).get(
                "non_tuned_reference_runtime_basis",
                "reference_execution_runtime",
            )
            resolved = v14_reference_budget_from_components(
                reference_is_tuned=False,
                actual_reference_execution_runtime_seconds=np.nan,
                non_tuned_basis=non_tuned_reference_basis,
            )
            resolved_budget_basis = resolved["Budget_Basis"]
            reference_budget_source_field = resolved[
                "Reference_Budget_Source_Field"
            ]

        if budgeting_enabled and reference_is_tuned:
            assert budget_model in MODEL_RUNNERS, \
                f"Reference model {budget_model} has no runner."

        cap_untuned_competitors = RUN_CONFIG.get("budgeting", {}).get(
            "cap_untuned_competitors", True
        )

        sampling_config = scenario_cfg.get(
            "sampling",
            RUN_CONFIG.get("sampling", {"strategy": "original_prevalence"})
        )

        scenario_dir = os.path.join(RESULTS_DIR, scenario_folder_name(scenario_name))
        os.makedirs(scenario_dir, exist_ok=True)

        scenario_prefix = scenario_file_prefix(
            scenario_name=scenario_name,
            file_prefix=FILE_PREFIX
        )

        reset_runtime_memory()

        errors = initialize_errors(active_model_names)
        iteration_rows = []

        print("=" * 100)
        print(f"STARTING SCENARIO: {scenario_name}")
        print("=" * 100)
        print(f"Active models: {active_model_names}")
        print(f"Sampling config: {sampling_config}")
        print(f"Scenario folder: {scenario_dir}")
        print("=" * 100)


        def base_row_for_model(
            model_name,
            iteration_num,
            split_info,
            inner_train_class1_prev,
            val_class1_prev,
        ):
            cfg = get_model_cfg(model_name)

            base_row = {
                "Pipeline_Version": PIPELINE_VERSION,
                "Scenario": scenario_name,

                "Budget_Reference_Model": budget_model if budgeting_enabled else "None",
                "Budget_Basis": resolved_budget_basis,
                "Reference_Budget_Source_Field": (
                    reference_budget_source_field
                ),
                "Wall_Clock_Definition": (
                    "Elapsed real time between the start and end of the "
                    "measured operation, analogous to timing it with a "
                    "stopwatch."
                ),

                "Sampling_Strategy": split_info.get("Sampling_Strategy", ""),
                "Original_Class1_Prevalence": split_info.get("Original_Class1_Prevalence", np.nan),
                "Target_Class1_Prevalence": split_info.get("Target_Class1_Prevalence", np.nan),
                "Sample_Class1_Prevalence": split_info.get("Sample_Class1_Prevalence", np.nan),
                "Train_Class1_Prevalence": split_info.get("Train_Class1_Prevalence", np.nan),
                "Test_Class1_Prevalence": split_info.get("Test_Class1_Prevalence", np.nan),
                "InnerTrain_Class1_Prevalence": inner_train_class1_prev,
                "Validation_Class1_Prevalence": val_class1_prev,

                "Train_N": split_info.get("Train_N", np.nan),
                "Test_N": split_info.get("Test_N", np.nan),
                "Train_Class0_N": split_info.get("Train_Class0_N", np.nan),
                "Train_Class1_N": split_info.get("Train_Class1_N", np.nan),
                "Test_Class0_N": split_info.get("Test_Class0_N", np.nan),
                "Test_Class1_N": split_info.get("Test_Class1_N", np.nan),

                "MonteCarlo_Iteration": iteration_num,
                "Model": model_name,
                "Input_Mode": cfg.get("input_mode", ""),
            }
            base_row.update({
                key: value
                for key, value in split_info.items()
                if not str(key).startswith("_")
            })
            return base_row


        def record_success(model_name, base_row, outputs, total_runtime, tabpfn_budget=np.nan):
            output_meta = {}
            if (
                isinstance(outputs, tuple)
                and len(outputs) == 11
                and isinstance(outputs[-1], dict)
            ):
                acc, auc, brier, fpr, tpr, sens, prec, proba, y_true, energy, output_meta = outputs
            else:
                acc, auc, brier, fpr, tpr, sens, prec, proba, y_true, energy = outputs

            iteration_num = base_row["MonteCarlo_Iteration"]

            budget_info = get_latest_budget_info(model_name, iteration_num)
            selection_info = get_latest_selection_info(model_name, iteration_num)
            final_fit_info = get_latest_final_fit_info(model_name, iteration_num)

            actual_optuna_time = budget_info.get("Actual_Optuna_Tuning_Time_Seconds", np.nan)
            final_fit_time = final_fit_info.get("Final_Fit_Predict_Time_Seconds", np.nan)

            tabpfn_meta = output_meta if model_name == "TabPFN" else {}
            tabpfn_strict_pass = tabpfn_meta.get("TabPFN_Strict_EndToEnd_Budget_Passed", np.nan)
            tabpfn_predict_pass = tabpfn_meta.get("TabPFN_PredictProba_Budget_Passed", np.nan)
            tabpfn_strict_pass_bool = bool(tabpfn_strict_pass) if isinstance(tabpfn_strict_pass, (bool, np.bool_)) else False
            tabpfn_predict_pass_bool = bool(tabpfn_predict_pass) if isinstance(tabpfn_predict_pass, (bool, np.bool_)) else False
            tabpfn_fit_time = tabpfn_meta.get("TabPFN_Fit_Time_Seconds", np.nan)
            tabpfn_predict_time = tabpfn_meta.get("TabPFN_PredictProba_Time_Seconds", np.nan)
            tabpfn_end_to_end_time = tabpfn_meta.get("TabPFN_EndToEnd_FitPlusPredictProba_Seconds", np.nan)
            tabpfn_strict_overrun = tabpfn_meta.get("TabPFN_Strict_EndToEnd_Overrun_Seconds", np.nan)
            tabpfn_predict_overrun = tabpfn_meta.get("TabPFN_PredictProba_Overrun_Seconds", np.nan)
            tabpfn_full_train_n = tabpfn_meta.get("TabPFN_Full_Train_N", np.nan)
            tabpfn_budgeted_context_sample_size = tabpfn_meta.get("TabPFN_Budgeted_Context_Sample_Size", np.nan)
            tabpfn_context_n_used = tabpfn_meta.get("TabPFN_Context_N_Used", np.nan)
            tabpfn_context_fraction = tabpfn_meta.get("TabPFN_Context_Fraction_Used", np.nan)
            tabpfn_context_attempts = tabpfn_meta.get("TabPFN_Context_Search_Attempts", np.nan)
            tabpfn_context_search_runtime = tabpfn_meta.get(
                "TabPFN_Context_Search_Total_Runtime_Seconds", np.nan
            )
            tabpfn_context_search_budget_passed = tabpfn_meta.get(
                "TabPFN_Context_Search_Effective_Budget_Passed", np.nan
            )
            tabpfn_context_search_over_budget = tabpfn_meta.get(
                "TabPFN_Context_Search_Over_Effective_Budget_Seconds", np.nan
            )
            tabpfn_full_prediction_n = tabpfn_meta.get("TabPFN_Full_Prediction_N", np.nan)
            tabpfn_budgeted_prediction_sample_size = tabpfn_meta.get("TabPFN_Budgeted_Prediction_Sample_Size", np.nan)
            tabpfn_prediction_fraction = tabpfn_meta.get("TabPFN_Prediction_Fraction_Used", np.nan)
            tabpfn_prediction_attempts = tabpfn_meta.get("TabPFN_Prediction_Search_Attempts", np.nan)
            tabpfn_predict_budget_delta = tabpfn_meta.get("TabPFN_PredictProba_Budget_Delta_Seconds", np.nan)
            tabpfn_predict_budget_remaining = tabpfn_meta.get("TabPFN_PredictProba_Budget_Remaining_Seconds", np.nan)
            tabpfn_predict_budget_use_ratio = tabpfn_meta.get("TabPFN_PredictProba_Budget_Use_Ratio", np.nan)
            tabpfn_total_budget_delta = tabpfn_meta.get("TabPFN_Total_Runtime_Budget_Delta_Seconds", np.nan)
            tabpfn_total_budget_remaining = tabpfn_meta.get("TabPFN_Total_Runtime_Budget_Remaining_Seconds", np.nan)
            tabpfn_ml_reference_budget = tabpfn_meta.get("TabPFN_ML_Reference_Budget_Seconds", np.nan)
            tabpfn_effective_time_budget = tabpfn_meta.get("TabPFN_Effective_Time_Budget_Seconds", np.nan)
            tabpfn_effective_budget_multiplier = tabpfn_meta.get("TabPFN_Effective_Budget_Multiplier", np.nan)
            tabpfn_min_context_requested = tabpfn_meta.get("TabPFN_Min_Context_Requested", np.nan)
            tabpfn_min_context_target = tabpfn_meta.get("TabPFN_Min_Context_Target", np.nan)
            tabpfn_min_context_runtime = tabpfn_meta.get("TabPFN_Min_Context_Runtime_Seconds", np.nan)
            tabpfn_min_context_multiplier_applied = tabpfn_meta.get(
                "TabPFN_Min_Context_Budget_Multiplier_Applied", np.nan
            )
            tabpfn_min_context_requirement_met = tabpfn_meta.get(
                "TabPFN_Min_Context_Requirement_Met", np.nan
            )
            tabpfn_original_ml_budget_total_delta = tabpfn_meta.get(
                "TabPFN_Original_ML_Budget_Total_Runtime_Delta_Seconds", np.nan
            )
            tabpfn_original_ml_budget_total_remaining = tabpfn_meta.get(
                "TabPFN_Original_ML_Budget_Total_Runtime_Remaining_Seconds", np.nan
            )

            strict_budget_acc = acc if tabpfn_strict_pass_bool else np.nan
            strict_budget_auc = auc if tabpfn_strict_pass_bool else np.nan
            strict_budget_brier = brier if tabpfn_strict_pass_bool else np.nan
            strict_budget_sens = sens if tabpfn_strict_pass_bool else np.nan
            strict_budget_prec = prec if tabpfn_strict_pass_bool else np.nan

            prediction_budget_acc = acc if tabpfn_predict_pass_bool else np.nan
            prediction_budget_auc = auc if tabpfn_predict_pass_bool else np.nan
            prediction_budget_brier = brier if tabpfn_predict_pass_bool else np.nan
            prediction_budget_sens = sens if tabpfn_predict_pass_bool else np.nan
            prediction_budget_prec = prec if tabpfn_predict_pass_bool else np.nan

            budgeting_enabled = bool(scenario_cfg.get(
                "budgeting_enabled",
                RUN_CONFIG.get("budgeting", {}).get("enabled", True),
            ))

            if model_name == budget_model:
                optuna_time_capped = np.nan
                budgeted_total_runtime = total_runtime
                actual_total_runtime = total_runtime

            else:
                actual_total_runtime = total_runtime

                if budgeting_enabled:
                    if np.isfinite(actual_optuna_time) and np.isfinite(tabpfn_budget):
                        optuna_time_capped = min(actual_optuna_time, tabpfn_budget)
                    else:
                        optuna_time_capped = np.nan

                    if np.isfinite(optuna_time_capped) and np.isfinite(final_fit_time):
                        budgeted_total_runtime = optuna_time_capped + final_fit_time
                    else:
                        budgeted_total_runtime = np.nan

                else:
                    optuna_time_capped = actual_optuna_time
                    budgeted_total_runtime = actual_total_runtime

            if model_name == "TabPFN" and tabpfn_meta:
                optuna_time_capped = np.nan
                if (
                    tabpfn_meta.get("TabPFN_Local_Budget_Mode")
                    == "unbudgeted_full_context"
                ):
                    budgeted_total_runtime = actual_total_runtime
                else:
                    budgeted_total_runtime = (
                        tabpfn_end_to_end_time
                        if tabpfn_strict_pass_bool else np.nan
                    )

            errors[f"{model_name}_BalancedAccuracy"].append(acc)
            errors[f"{model_name}_Brier"].append(brier)
            errors[f"{model_name}_AUC"].append(auc)
            errors[f"{model_name}_TotalRuntime"].append(total_runtime)
            errors[f"{model_name}_ActualTotalRuntime"].append(actual_total_runtime)
            errors[f"{model_name}_BudgetedTotalRuntime"].append(budgeted_total_runtime)
            errors[f"{model_name}_OptunaTuningTimeCapped"].append(optuna_time_capped)

            errors[f"{model_name}_FPR"].append(fpr)
            errors[f"{model_name}_TPR"].append(tpr)
            errors[f"{model_name}_Sensitivity"].append(sens)
            errors[f"{model_name}_Precision"].append(prec)
            errors[f"{model_name}_Proba"].append(proba)
            errors[f"{model_name}_TrueLabels"].append(y_true)
            errors[f"{model_name}_Energy"].append(energy)

            errors[f"{model_name}_TabPFNBudget"].append(tabpfn_budget)
            errors[f"{model_name}_ActualOptunaTuningTime"].append(actual_optuna_time)
            errors[f"{model_name}_CompletedTrials"].append(
                budget_info.get("Total_Completed_Trials", np.nan)
            )
            errors[f"{model_name}_EligibleTrials"].append(
                selection_info.get("Eligible_Trials_Within_Budget", np.nan)
            )
            errors[f"{model_name}_SelectedAUC"].append(
                selection_info.get("Selected_AUC", np.nan)
            )
            errors[f"{model_name}_FinalFitPredictTime"].append(final_fit_time)

            errors[f"{model_name}_StrictBudgetBalancedAccuracy"].append(strict_budget_acc)
            errors[f"{model_name}_StrictBudgetAUC"].append(strict_budget_auc)
            errors[f"{model_name}_StrictBudgetBrier"].append(strict_budget_brier)
            errors[f"{model_name}_StrictBudgetSensitivity"].append(strict_budget_sens)
            errors[f"{model_name}_StrictBudgetPrecision"].append(strict_budget_prec)
            errors[f"{model_name}_PredictionBudgetBalancedAccuracy"].append(prediction_budget_acc)
            errors[f"{model_name}_PredictionBudgetAUC"].append(prediction_budget_auc)
            errors[f"{model_name}_PredictionBudgetBrier"].append(prediction_budget_brier)
            errors[f"{model_name}_PredictionBudgetSensitivity"].append(prediction_budget_sens)
            errors[f"{model_name}_PredictionBudgetPrecision"].append(prediction_budget_prec)
            errors[f"{model_name}_TabPFNFitTime"].append(tabpfn_fit_time)
            errors[f"{model_name}_TabPFNPredictProbaTime"].append(tabpfn_predict_time)
            errors[f"{model_name}_TabPFNEndToEndFitPredictProbaTime"].append(tabpfn_end_to_end_time)
            errors[f"{model_name}_TabPFNStrictEndToEndBudgetPassed"].append(
                int(tabpfn_strict_pass_bool) if tabpfn_meta else np.nan
            )
            errors[f"{model_name}_TabPFNPredictProbaBudgetPassed"].append(
                int(tabpfn_predict_pass_bool) if tabpfn_meta else np.nan
            )
            errors[f"{model_name}_TabPFNStrictEndToEndOverrun"].append(tabpfn_strict_overrun)
            errors[f"{model_name}_TabPFNPredictProbaOverrun"].append(tabpfn_predict_overrun)
            errors[f"{model_name}_TabPFNFullTrainN"].append(tabpfn_full_train_n)
            errors[f"{model_name}_TabPFNBudgetedContextSampleSize"].append(tabpfn_budgeted_context_sample_size)
            errors[f"{model_name}_TabPFNContextNUsed"].append(tabpfn_context_n_used)
            errors[f"{model_name}_TabPFNContextFractionUsed"].append(tabpfn_context_fraction)
            errors[f"{model_name}_TabPFNContextSearchAttempts"].append(tabpfn_context_attempts)
            errors[f"{model_name}_TabPFNFullPredictionN"].append(tabpfn_full_prediction_n)
            errors[f"{model_name}_TabPFNBudgetedPredictionSampleSize"].append(tabpfn_budgeted_prediction_sample_size)
            errors[f"{model_name}_TabPFNPredictionFractionUsed"].append(tabpfn_prediction_fraction)
            errors[f"{model_name}_TabPFNPredictionSearchAttempts"].append(tabpfn_prediction_attempts)
            errors[f"{model_name}_TabPFNPredictProbaBudgetDelta"].append(tabpfn_predict_budget_delta)
            errors[f"{model_name}_TabPFNPredictProbaBudgetRemaining"].append(tabpfn_predict_budget_remaining)
            errors[f"{model_name}_TabPFNPredictProbaBudgetUseRatio"].append(tabpfn_predict_budget_use_ratio)
            errors[f"{model_name}_TabPFNTotalRuntimeBudgetDelta"].append(tabpfn_total_budget_delta)
            errors[f"{model_name}_TabPFNTotalRuntimeBudgetRemaining"].append(tabpfn_total_budget_remaining)
            errors[f"{model_name}_TabPFNMLReferenceBudget"].append(tabpfn_ml_reference_budget)
            errors[f"{model_name}_TabPFNEffectiveTimeBudget"].append(tabpfn_effective_time_budget)
            errors[f"{model_name}_TabPFNEffectiveBudgetMultiplier"].append(tabpfn_effective_budget_multiplier)
            errors[f"{model_name}_TabPFNMinContextRequested"].append(tabpfn_min_context_requested)
            errors[f"{model_name}_TabPFNMinContextTarget"].append(tabpfn_min_context_target)
            errors[f"{model_name}_TabPFNMinContextRuntime"].append(tabpfn_min_context_runtime)
            errors[f"{model_name}_TabPFNMinContextBudgetMultiplierApplied"].append(
                int(bool(tabpfn_min_context_multiplier_applied))
                if isinstance(tabpfn_min_context_multiplier_applied, (bool, np.bool_))
                else tabpfn_min_context_multiplier_applied
            )
            errors[f"{model_name}_TabPFNMinContextRequirementMet"].append(
                int(bool(tabpfn_min_context_requirement_met))
                if isinstance(tabpfn_min_context_requirement_met, (bool, np.bool_))
                else tabpfn_min_context_requirement_met
            )
            errors[f"{model_name}_TabPFNOriginalMLBudgetTotalRuntimeDelta"].append(
                tabpfn_original_ml_budget_total_delta
            )
            errors[f"{model_name}_TabPFNOriginalMLBudgetTotalRuntimeRemaining"].append(
                tabpfn_original_ml_budget_total_remaining
            )

            success_status = "Success"
            if tabpfn_meta:
                if (
                    tabpfn_meta.get("TabPFN_Local_Budget_Mode")
                    == "unbudgeted_full_context"
                ):
                    success_status = "Success_NoBudget_LocalFullContext"
                else:
                    success_status = (
                        "Success_StrictEffectiveBudget_AdaptiveContext"
                        if tabpfn_strict_pass_bool
                        else "ObservedButStrictEffectiveBudgetTimeout_AdaptiveContext"
                    )

            row = {
                **base_row,
                "Status": success_status,
                "BalancedAccuracy": acc,
                "AUC": auc,
                "Brier": brier,
                "Sensitivity": sens,
                "Precision": prec,

                "Actual_Total_Runtime_Seconds": actual_total_runtime,
                "Budgeted_Total_Runtime_Seconds": budgeted_total_runtime,
                "Budget_Accounted_Runtime_Seconds": budgeted_total_runtime,
                "Total_Runtime_Seconds": total_runtime,

                "TabPFN_Time_Budget_Seconds": tabpfn_budget,
                "Reference_Budget_Seconds": tabpfn_budget,
                "Actual_Optuna_Tuning_Time_Seconds": actual_optuna_time,
                "HPO_Start_UTC": budget_info.get("HPO_Start_UTC"),
                "HPO_End_UTC": budget_info.get("HPO_End_UTC"),
                "HPO_Start_Perf_Counter_Seconds": budget_info.get(
                    "HPO_Start_Perf_Counter_Seconds", np.nan
                ),
                "HPO_End_Perf_Counter_Seconds": budget_info.get(
                    "HPO_End_Perf_Counter_Seconds", np.nan
                ),
                "HPO_Timer": budget_info.get(
                    "HPO_Timer", "time.perf_counter"
                ),
                "HPO_Timing_Boundary": budget_info.get(
                    "HPO_Timing_Boundary", "Optuna tuning loop only"
                ),
                "Optuna_Tuning_Time_Capped_Seconds": optuna_time_capped,
                "Optuna_Tuning_Over_Budget_Seconds": budget_info.get(
                    "Optuna_Tuning_Over_Budget_Seconds", np.nan
                ),
                "Optuna_Tuning_Over_Budget_Flag": budget_info.get(
                    "Optuna_Tuning_Over_Budget_Flag", np.nan
                ),
                "Optuna_Completed_Trials": budget_info.get("Total_Completed_Trials", np.nan),
                "Optuna_Trials_Started": budget_info.get("Total_Trials_Started", np.nan),
                "Eligible_Trials_Within_Budget": selection_info.get(
                    "Eligible_Trials_Within_Budget", np.nan
                ),

                "Selection_Method": selection_info.get("Selection_Method", ""),
                "Selected_Trial_Number": selection_info.get("Selected_Trial_Number", np.nan),
                "Selected_AUC": selection_info.get("Selected_AUC", np.nan),
                "Selected_Trial_End_Elapsed_Seconds": selection_info.get(
                    "Selected_Trial_End_Elapsed_Seconds", np.nan
                ),

                "Final_Fit_Predict_Time_Seconds": final_fit_time,
                "Final_Model_Preparation_Time_Seconds": (
                    final_fit_info.get(
                        "Final_Model_Preparation_Time_Seconds", np.nan
                    )
                ),
                "Final_Fit_Time_Seconds": final_fit_info.get(
                    "Final_Fit_Time_Seconds", np.nan
                ),
                "Prediction_Time_Seconds": final_fit_info.get(
                    "Prediction_Time_Seconds", np.nan
                ),
                "Final_Fit_Predict_Start_UTC": final_fit_info.get(
                    "Final_Fit_Predict_Start_UTC"
                ),
                "Final_Fit_Predict_End_UTC": final_fit_info.get(
                    "Final_Fit_Predict_End_UTC"
                ),
                "Final_Fit_Start_UTC": final_fit_info.get(
                    "Final_Fit_Start_UTC"
                ),
                "Final_Fit_End_UTC": final_fit_info.get(
                    "Final_Fit_End_UTC"
                ),
                "Prediction_Start_UTC": final_fit_info.get(
                    "Prediction_Start_UTC"
                ),
                "Prediction_End_UTC": final_fit_info.get(
                    "Prediction_End_UTC"
                ),
                "Final_Fit_Predict_Timer": final_fit_info.get(
                    "Final_Fit_Predict_Timer"
                ),

                "StrictBudget_BalancedAccuracy": strict_budget_acc,
                "StrictBudget_AUC": strict_budget_auc,
                "StrictBudget_Brier": strict_budget_brier,
                "StrictBudget_Sensitivity": strict_budget_sens,
                "StrictBudget_Precision": strict_budget_prec,
                "PredictionBudget_BalancedAccuracy": prediction_budget_acc,
                "PredictionBudget_AUC": prediction_budget_auc,
                "PredictionBudget_Brier": prediction_budget_brier,
                "PredictionBudget_Sensitivity": prediction_budget_sens,
                "PredictionBudget_Precision": prediction_budget_prec,

                "TabPFN_Local_Budget_Mode": tabpfn_meta.get("TabPFN_Local_Budget_Mode", ""),
                "TabPFN_Local_Device": tabpfn_meta.get("TabPFN_Local_Device", ""),
                "TabPFN_Local_Model_Version": tabpfn_meta.get("TabPFN_Local_Model_Version", ""),
                "TabPFN_Local_Model_Path": tabpfn_meta.get("TabPFN_Local_Model_Path", ""),
                "TabPFN_Local_Fit_Mode": tabpfn_meta.get("TabPFN_Local_Fit_Mode", ""),
                "TabPFN_Local_N_Estimators": tabpfn_meta.get("TabPFN_Local_N_Estimators", np.nan),
                "TabPFN_Budget_Reference_Seconds": tabpfn_meta.get("TabPFN_Budget_Reference_Seconds", np.nan),
                "TabPFN_ML_Reference_Budget_Seconds": tabpfn_ml_reference_budget,
                "TabPFN_Effective_Time_Budget_Seconds": tabpfn_effective_time_budget,
                "TabPFN_Effective_Budget_Multiplier": tabpfn_effective_budget_multiplier,
                "TabPFN_Min_Context_Requested": tabpfn_min_context_requested,
                "TabPFN_Min_Context_Target": tabpfn_min_context_target,
                "TabPFN_Min_Context_Runtime_Seconds": tabpfn_min_context_runtime,
                "TabPFN_Min_Context_Budget_Multiplier_Applied": tabpfn_min_context_multiplier_applied,
                "TabPFN_Min_Context_Requirement_Met": tabpfn_min_context_requirement_met,
                "TabPFN_Fixed_Minimum_Context_Only": tabpfn_meta.get("TabPFN_Fixed_Minimum_Context_Only", ""),
                "TabPFN_Context_Budget_Calculation": tabpfn_meta.get("TabPFN_Context_Budget_Calculation", ""),
                "TabPFN_Full_Train_N": tabpfn_full_train_n,
                "TabPFN_Budgeted_Context_Sample_Size": tabpfn_budgeted_context_sample_size,
                "TabPFN_Context_N_Used": tabpfn_context_n_used,
                "TabPFN_Context_Fraction_Used": tabpfn_context_fraction,
                "TabPFN_Context_Selection_Rule": tabpfn_meta.get("TabPFN_Context_Selection_Rule", ""),
                "TabPFN_Context_Search_Attempts": tabpfn_context_attempts,
                "TabPFN_Context_Search_Total_Runtime_Seconds": tabpfn_context_search_runtime,
                "TabPFN_Context_Search_Effective_Budget_Passed": tabpfn_context_search_budget_passed,
                "TabPFN_Context_Search_Over_Effective_Budget_Seconds": tabpfn_context_search_over_budget,
                "TabPFN_Context_Candidates": tabpfn_meta.get("TabPFN_Context_Candidates", ""),
                "TabPFN_Context_Attempt_Log": tabpfn_meta.get("TabPFN_Context_Attempt_Log", ""),
                "TabPFN_Adaptive_Context_Enabled": tabpfn_meta.get("TabPFN_Adaptive_Context_Enabled", np.nan),
                "TabPFN_Full_Prediction_N": tabpfn_full_prediction_n,
                "TabPFN_Budgeted_Prediction_Sample_Size": tabpfn_budgeted_prediction_sample_size,
                "TabPFN_Prediction_Fraction_Used": tabpfn_prediction_fraction,
                "TabPFN_Prediction_Selection_Rule": tabpfn_meta.get("TabPFN_Prediction_Selection_Rule", ""),
                "TabPFN_Prediction_Search_Attempts": tabpfn_prediction_attempts,
                "TabPFN_Prediction_Candidates": tabpfn_meta.get("TabPFN_Prediction_Candidates", ""),
                "TabPFN_Prediction_Attempt_Log": tabpfn_meta.get("TabPFN_Prediction_Attempt_Log", ""),
                "TabPFN_Fit_Time_Seconds": tabpfn_fit_time,
                "TabPFN_PredictProba_Time_Seconds": tabpfn_predict_time,
                "TabPFN_PredictProba_Budget_Delta_Seconds": tabpfn_predict_budget_delta,
                "TabPFN_PredictProba_Budget_Remaining_Seconds": tabpfn_predict_budget_remaining,
                "TabPFN_PredictProba_Budget_Use_Ratio": tabpfn_predict_budget_use_ratio,
                "TabPFN_EndToEnd_FitPlusPredictProba_Seconds": tabpfn_end_to_end_time,
                "TabPFN_Total_Runtime_Budget_Delta_Seconds": tabpfn_total_budget_delta,
                "TabPFN_Total_Runtime_Budget_Remaining_Seconds": tabpfn_total_budget_remaining,
                "TabPFN_Original_ML_Budget_Total_Runtime_Delta_Seconds": tabpfn_original_ml_budget_total_delta,
                "TabPFN_Original_ML_Budget_Total_Runtime_Remaining_Seconds": tabpfn_original_ml_budget_total_remaining,
                "TabPFN_Strict_EndToEnd_Budget_Passed": (
                    tabpfn_strict_pass_bool if tabpfn_meta else np.nan
                ),
                "TabPFN_PredictProba_Budget_Passed": (
                    tabpfn_predict_pass_bool if tabpfn_meta else np.nan
                ),
                "TabPFN_Strict_EndToEnd_Overrun_Seconds": tabpfn_strict_overrun,
                "TabPFN_PredictProba_Overrun_Seconds": tabpfn_predict_overrun,
                "TabPFN_Strict_EndToEnd_Status": tabpfn_meta.get("TabPFN_Strict_EndToEnd_Status", ""),
                "TabPFN_PredictProba_Status": tabpfn_meta.get("TabPFN_PredictProba_Status", ""),
                "TabPFN_Primary_Budget_Interpretation": tabpfn_meta.get(
                    "TabPFN_Primary_Budget_Interpretation", ""
                ),
                "TabPFN_Secondary_Budget_Interpretation": tabpfn_meta.get(
                    "TabPFN_Secondary_Budget_Interpretation", ""
                ),

                "Energy_kWh": energy,
                "Error": "",
            }

            iteration_rows.append(row)

            if tabpfn_meta:
                tabpfn_budget_evaluation_memory.append({
                    "Scenario": row.get("Scenario", scenario_name),
                    "MonteCarlo_Iteration": row.get("MonteCarlo_Iteration", iteration_num),
                    "Model": model_name,
                    "Budget_Reference_Model": row.get("Budget_Reference_Model", ""),
                    "Budget_Basis": row.get("Budget_Basis", ""),
                    "TabPFN_Time_Budget_Seconds": tabpfn_budget,
                    "TabPFN_Budget_Reference_Seconds": row.get("TabPFN_Budget_Reference_Seconds", np.nan),
                    "TabPFN_ML_Reference_Budget_Seconds": tabpfn_ml_reference_budget,
                    "TabPFN_Effective_Time_Budget_Seconds": tabpfn_effective_time_budget,
                    "TabPFN_Effective_Budget_Multiplier": tabpfn_effective_budget_multiplier,
                    "TabPFN_Min_Context_Requested": tabpfn_min_context_requested,
                    "TabPFN_Min_Context_Target": tabpfn_min_context_target,
                    "TabPFN_Min_Context_Runtime_Seconds": tabpfn_min_context_runtime,
                    "TabPFN_Min_Context_Budget_Multiplier_Applied": tabpfn_min_context_multiplier_applied,
                    "TabPFN_Min_Context_Requirement_Met": tabpfn_min_context_requirement_met,
                    "TabPFN_Fixed_Minimum_Context_Only": row.get("TabPFN_Fixed_Minimum_Context_Only", ""),
                    "TabPFN_Context_Budget_Calculation": row.get("TabPFN_Context_Budget_Calculation", ""),
                    "TabPFN_Full_Train_N": tabpfn_full_train_n,
                    "TabPFN_Budgeted_Context_Sample_Size": tabpfn_budgeted_context_sample_size,
                    "TabPFN_Context_N_Used": tabpfn_context_n_used,
                    "TabPFN_Context_Fraction_Used": tabpfn_context_fraction,
                    "TabPFN_Context_Selection_Rule": row.get("TabPFN_Context_Selection_Rule", ""),
                    "TabPFN_Context_Search_Attempts": tabpfn_context_attempts,
                    "TabPFN_Context_Search_Total_Runtime_Seconds": tabpfn_context_search_runtime,
                    "TabPFN_Context_Search_Effective_Budget_Passed": tabpfn_context_search_budget_passed,
                    "TabPFN_Context_Search_Over_Effective_Budget_Seconds": tabpfn_context_search_over_budget,
                    "TabPFN_Context_Candidates": tabpfn_meta.get("TabPFN_Context_Candidates", ""),
                    "TabPFN_Context_Attempt_Log": tabpfn_meta.get("TabPFN_Context_Attempt_Log", ""),
                    "TabPFN_Adaptive_Context_Enabled": tabpfn_meta.get("TabPFN_Adaptive_Context_Enabled", np.nan),
                    "TabPFN_Full_Prediction_N": tabpfn_full_prediction_n,
                    "TabPFN_Budgeted_Prediction_Sample_Size": tabpfn_budgeted_prediction_sample_size,
                    "TabPFN_Prediction_Fraction_Used": tabpfn_prediction_fraction,
                    "TabPFN_Prediction_Selection_Rule": row.get("TabPFN_Prediction_Selection_Rule", ""),
                    "TabPFN_Prediction_Search_Attempts": tabpfn_prediction_attempts,
                    "TabPFN_Prediction_Candidates": row.get("TabPFN_Prediction_Candidates", ""),
                    "TabPFN_Prediction_Attempt_Log": row.get("TabPFN_Prediction_Attempt_Log", ""),
                    "TabPFN_Fit_Time_Seconds": tabpfn_fit_time,
                    "TabPFN_PredictProba_Time_Seconds": tabpfn_predict_time,
                    "TabPFN_PredictProba_Budget_Delta_Seconds": tabpfn_predict_budget_delta,
                    "TabPFN_PredictProba_Budget_Remaining_Seconds": tabpfn_predict_budget_remaining,
                    "TabPFN_PredictProba_Budget_Use_Ratio": tabpfn_predict_budget_use_ratio,
                    "TabPFN_EndToEnd_FitPlusPredictProba_Seconds": tabpfn_end_to_end_time,
                    "TabPFN_Total_Runtime_Budget_Delta_Seconds": tabpfn_total_budget_delta,
                    "TabPFN_Total_Runtime_Budget_Remaining_Seconds": tabpfn_total_budget_remaining,
                    "TabPFN_Original_ML_Budget_Total_Runtime_Delta_Seconds": tabpfn_original_ml_budget_total_delta,
                    "TabPFN_Original_ML_Budget_Total_Runtime_Remaining_Seconds": tabpfn_original_ml_budget_total_remaining,
                    "TabPFN_Strict_EndToEnd_Budget_Passed": tabpfn_strict_pass_bool,
                    "TabPFN_PredictProba_Budget_Passed": tabpfn_predict_pass_bool,
                    "TabPFN_Strict_EndToEnd_Overrun_Seconds": tabpfn_strict_overrun,
                    "TabPFN_PredictProba_Overrun_Seconds": tabpfn_predict_overrun,
                    "TabPFN_Strict_EndToEnd_Status": row.get("TabPFN_Strict_EndToEnd_Status", ""),
                    "TabPFN_PredictProba_Status": row.get("TabPFN_PredictProba_Status", ""),
                    "Observed_BalancedAccuracy": acc,
                    "Observed_AUC": auc,
                    "Observed_Brier": brier,
                    "Observed_Sensitivity": sens,
                    "Observed_Precision": prec,
                    "StrictBudget_BalancedAccuracy": strict_budget_acc,
                    "StrictBudget_AUC": strict_budget_auc,
                    "StrictBudget_Brier": strict_budget_brier,
                    "StrictBudget_Sensitivity": strict_budget_sens,
                    "StrictBudget_Precision": strict_budget_prec,
                    "PredictionBudget_BalancedAccuracy": prediction_budget_acc,
                    "PredictionBudget_AUC": prediction_budget_auc,
                    "PredictionBudget_Brier": prediction_budget_brier,
                    "PredictionBudget_Sensitivity": prediction_budget_sens,
                    "PredictionBudget_Precision": prediction_budget_prec,
                    "TabPFN_Local_Budget_Mode": row.get("TabPFN_Local_Budget_Mode", ""),
                    "TabPFN_Local_Device": row.get("TabPFN_Local_Device", ""),
                    "TabPFN_Local_Model_Version": row.get("TabPFN_Local_Model_Version", ""),
                    "TabPFN_Local_Model_Path": row.get("TabPFN_Local_Model_Path", ""),
                    "TabPFN_Local_Fit_Mode": row.get("TabPFN_Local_Fit_Mode", ""),
                    "TabPFN_Local_N_Estimators": row.get("TabPFN_Local_N_Estimators", np.nan),
                })


        def record_failure(model_name, base_row, error_message, total_runtime, tabpfn_budget=np.nan):
            row = {
                **base_row,
                "Status": "Failed",
                "BalancedAccuracy": np.nan,
                "AUC": np.nan,
                "Brier": np.nan,
                "Sensitivity": np.nan,
                "Precision": np.nan,

                "Actual_Total_Runtime_Seconds": total_runtime,
                "Budgeted_Total_Runtime_Seconds": np.nan,
                "Total_Runtime_Seconds": total_runtime,

                "TabPFN_Time_Budget_Seconds": tabpfn_budget,
                "Actual_Optuna_Tuning_Time_Seconds": np.nan,
                "Optuna_Tuning_Time_Capped_Seconds": np.nan,
                "Optuna_Tuning_Over_Budget_Seconds": np.nan,
                "Optuna_Tuning_Over_Budget_Flag": np.nan,
                "Optuna_Completed_Trials": np.nan,
                "Optuna_Trials_Started": np.nan,
                "Eligible_Trials_Within_Budget": np.nan,

                "Selection_Method": "",
                "Selected_Trial_Number": np.nan,
                "Selected_AUC": np.nan,
                "Selected_Trial_End_Elapsed_Seconds": np.nan,

                "Final_Fit_Predict_Time_Seconds": np.nan,
                "Energy_kWh": np.nan,
                "Error": str(error_message),
            }

            iteration_rows.append(row)
            print(
                f"WARNING: {model_name} failed in iteration "
                f"{base_row['MonteCarlo_Iteration']}: "
                f"{str(error_message)[:200]}"
            )


        def checkpoint():
            checkpoint_iter_df = pd.DataFrame(iteration_rows)
            checkpoint_iter_csv = os.path.join(
                scenario_dir,
                f"{scenario_prefix}_IterationLevel_Metrics_CHECKPOINT.csv"
            )
            checkpoint_iter_json = os.path.join(
                scenario_dir,
                f"{scenario_prefix}_IterationLevel_Metrics_CHECKPOINT.json"
            )
            checkpoint_iter_df.to_csv(checkpoint_iter_csv, index=False)
            write_df_json(checkpoint_iter_df, checkpoint_iter_json)

            raw_checkpoint_pkl = os.path.join(
                scenario_dir,
                f"{scenario_prefix}_RawResults_CHECKPOINT.pkl"
            )
            raw_checkpoint_json = os.path.join(
                scenario_dir,
                f"{scenario_prefix}_RawResults_CHECKPOINT.json"
            )

            with open(raw_checkpoint_pkl, "wb") as f:
                pickle.dump(errors, f)
            write_json(raw_checkpoint_json, errors)



        for i in tqdm(range(RUN_CONFIG["iterations"]), desc=scenario_name):

            seed = get_seed(i)

            if split_strategy == "stratified_group":
                (X_train_raw, X_test_raw, y_train, y_test, split_info,
                 groups_train, groups_test) = grouped_sample_split(
                    X_np=X_data,
                    y_array=y_array,
                    groups=groups_array,
                    total_n=RUN_CONFIG["total_n"],
                    train_frac=RUN_CONFIG["train_frac"],
                    seed=seed,
                    sampling_config=sampling_config,
                    split_config=split_config,
                )
            elif split_strategy == "temporal":
                (X_train_raw, X_test_raw, y_train, y_test, split_info,
                 groups_train, groups_test) = temporal_sample_split(
                    X_data=X_data,
                    y_array=y_array,
                    timestamps=timestamps_array,
                    total_n=RUN_CONFIG["total_n"],
                    train_frac=RUN_CONFIG["train_frac"],
                    seed=seed,
                    sampling_config=sampling_config,
                    split_config=split_config,
                    groups=groups_array,
                )
            else:
                X_train_raw, X_test_raw, y_train, y_test, split_info = exact_sample_split_from_config(
                    X_np=X_data,
                    y_array=y_array,
                    total_n=RUN_CONFIG["total_n"],
                    train_frac=RUN_CONFIG["train_frac"],
                    seed=seed,
                    sampling_config=sampling_config,
                    split_config=split_config,
                )
                groups_train = None
                groups_test = None

            print(
                f"Iter {i} | "
                f"sampling={split_info['Sampling_Strategy']} | "
                f"target class-1={split_info['Target_Class1_Prevalence'] * 100:.2f}% | "
                f"sample class-1={split_info['Sample_Class1_Prevalence'] * 100:.2f}% | "
                f"train class-1={split_info['Train_Class1_Prevalence'] * 100:.2f}% | "
                f"test class-1={split_info['Test_Class1_Prevalence'] * 100:.2f}%"
            )


            if split_strategy == "stratified_group":
                X_train_sub_raw, X_val_raw, y_train_sub, y_val, inner_info = grouped_inner_split(
                    X_train_raw,
                    y_train,
                    groups_train,
                    val_frac=RUN_CONFIG["inner_validation_frac"],
                    seed=seed,
                    split_config=split_config,
                    return_info=True,
                )
            elif split_strategy == "temporal":
                timestamps_train = timestamps_array[split_info["_train_indices"]]
                X_train_sub_raw, X_val_raw, y_train_sub, y_val, inner_info = temporal_inner_split(
                    X_train_raw,
                    y_train,
                    timestamps_train,
                    val_frac=RUN_CONFIG["inner_validation_frac"],
                    seed=seed,
                    split_config=split_config,
                    return_info=True,
                )
            else:
                X_train_sub_raw, X_val_raw, y_train_sub, y_val, inner_info = stratified_inner_split(
                    X_train_raw,
                    y_train,
                    val_frac=RUN_CONFIG["inner_validation_frac"],
                    seed=seed,
                    split_config=split_config,
                    return_info=True,
                )

            inner_train_class1_prev = float(np.mean(y_train_sub == 1))
            val_class1_prev = float(np.mean(y_val == 1))

            conventional_views, tabpfn_views, preprocessing_info = _prepare_feature_views(
                X_train_raw,
                X_test_raw,
                X_train_sub_raw,
                X_val_raw,
                preprocessing_mode,
                preprocessing_config,
            )
            (
                X_train_clean,
                X_test_clean,
                X_train_sub_clean,
                X_val_clean,
            ) = conventional_views
            (
                X_train_tabpfn,
                X_test_tabpfn,
                X_train_sub_tabpfn,
                X_val_tabpfn,
            ) = tabpfn_views

            inner_train_n0, inner_train_n1 = _binary_counts(y_train_sub)
            val_n0, val_n1 = _binary_counts(y_val)
            split_info.update(preprocessing_info)
            split_info.update({
                "Inner_Split_Strategy": inner_info.get("Split_Strategy", ""),
                "Inner_Split_Valid": inner_info.get("Split_Valid", False),
                "Inner_Split_Fingerprint": inner_info.get("Split_Fingerprint", ""),
                "Inner_Split_Candidate_Method": inner_info.get("Split_Candidate_Method", ""),
                "Inner_Split_Candidates_Evaluated": inner_info.get("Split_Candidates_Evaluated", np.nan),
                "InnerTrain_N": int(len(y_train_sub)),
                "Validation_N": int(len(y_val)),
                "InnerTrain_Class0_N": inner_train_n0,
                "InnerTrain_Class1_N": inner_train_n1,
                "Validation_Class0_N": val_n0,
                "Validation_Class1_N": val_n1,
                "InnerTrain_Class1_Prevalence": inner_train_class1_prev,
                "Validation_Class1_Prevalence": val_class1_prev,
                "Inner_Train_Validation_Group_Overlap_N": inner_info.get(
                    "Train_Test_Group_Overlap_N", np.nan
                ),
                "Inner_Train_Validation_Row_Overlap_N": inner_info.get(
                    "Train_Test_Row_Overlap_N", np.nan
                ),
            })

            low_class_warning = int(split_config.get("low_class_count_warning", 10))
            smallest_class_count = min(
                split_info["Train_Class0_N"], split_info["Train_Class1_N"],
                split_info["Test_Class0_N"], split_info["Test_Class1_N"],
                inner_train_n0, inner_train_n1, val_n0, val_n1,
            )
            split_info["Low_Class_Count_Warning"] = bool(
                smallest_class_count < low_class_warning
            )
            if split_info["Low_Class_Count_Warning"]:
                print(
                    f"   Split warning: smallest partition/class cell has "
                    f"{smallest_class_count} rows (< {low_class_warning})."
                )


            if external_runtime_budget is not None:
                budget_runtime = float(external_runtime_budget)
                print(
                    f"   Iter {i}: using externally persisted paired runtime "
                    f"budget={budget_runtime:.6f}s; no reference model is rerun."
                )
                checkpoint()
            elif not budgeting_enabled:
                budget_runtime = None
                print(
                    f"   Iter {i}: NO-BUDGETING scenario; all models run on their "
                    f"own settings (Optuna up to {RUN_CONFIG.get('default_max_trials', 50)} trials)."
                )
                checkpoint()
            else:
                if budget_model == "TabPFN":
                    X_train_reference = X_train_tabpfn
                    X_test_reference = X_test_tabpfn
                    X_train_sub_reference = X_train_sub_tabpfn
                    X_val_reference = X_val_tabpfn
                else:
                    X_train_reference = X_train_clean
                    X_test_reference = X_test_clean
                    X_train_sub_reference = X_train_sub_clean
                    X_val_reference = X_val_clean

                ref_base = base_row_for_model(
                    model_name=budget_model,
                    iteration_num=i,
                    split_info=split_info,
                    inner_train_class1_prev=inner_train_class1_prev,
                    val_class1_prev=val_class1_prev,
                )

                try:
                    if reference_is_tuned:
                        ref_outputs, ref_total_runtime = v14_execute_model_call(
                            budget_model,
                            i,
                            scenario_name,
                            TOTAL_N,
                            lambda: MODEL_RUNNERS[budget_model](
                                X_train_reference,
                                X_test_reference,
                                y_train,
                                y_test,
                                X_train_sub_reference,
                                y_train_sub,
                                X_val_reference,
                                y_val,
                                i,
                                np.inf
                            ),
                            time_budget_seconds=np.inf,
                        )

                        ref_budget_info = get_latest_budget_info(budget_model, i)
                        reference_budget = v14_reference_budget_from_components(
                            reference_is_tuned=True,
                            actual_optuna_tuning_time_seconds=(
                                ref_budget_info.get(
                                    "Actual_Optuna_Tuning_Time_Seconds",
                                    np.nan,
                                )
                            ),
                            actual_reference_execution_runtime_seconds=(
                                ref_total_runtime
                            ),
                        )
                        budget_runtime = reference_budget[
                            "Reference_Budget_Seconds"
                        ]
                        ref_base.update(reference_budget)
                        budget_basis_msg = "tuning-loop time"

                    else:
                        ref_outputs, ref_total_runtime = v14_execute_model_call(
                            budget_model,
                            i,
                            scenario_name,
                            TOTAL_N,
                            lambda: MODEL_RUNNERS[budget_model](
                                X_train_reference,
                                X_test_reference,
                                y_train,
                                y_test,
                                i
                            ),
                            time_budget_seconds=None,
                        )
                        reference_budget = v14_reference_budget_from_components(
                            reference_is_tuned=False,
                            actual_reference_execution_runtime_seconds=(
                                ref_total_runtime
                            ),
                            non_tuned_basis=RUN_CONFIG.get(
                                "budgeting", {}
                            ).get(
                                "non_tuned_reference_runtime_basis",
                                "reference_execution_runtime",
                            ),
                        )
                        budget_runtime = reference_budget[
                            "Reference_Budget_Seconds"
                        ]
                        ref_base.update(reference_budget)
                        budget_basis_msg = "full wall-clock"

                    record_success(
                        model_name=budget_model,
                        base_row=ref_base,
                        outputs=ref_outputs,
                        total_runtime=ref_total_runtime,
                        tabpfn_budget=budget_runtime
                    )

                except Exception as e:
                    ref_total_runtime = getattr(
                        e, "v14_scientific_runtime_seconds", None
                    )
                    if ref_total_runtime is None:
                        ref_total_runtime = np.nan

                    record_failure(
                        model_name=budget_model,
                        base_row=ref_base,
                        error_message=e,
                        total_runtime=ref_total_runtime,
                        tabpfn_budget=np.nan
                    )

                    checkpoint()
                    continue

                if not np.isfinite(budget_runtime) or budget_runtime <= 0:
                    print(
                        f"   WARNING: Iter {i}: could not derive a valid budget from reference "
                        f"'{budget_model}'; skipping iteration."
                    )
                    checkpoint()
                    continue

                print(
                    f"   Iter {i}: budget reference = {budget_model} "
                    f"({budget_basis_msg}); budget = {budget_runtime:.3f} seconds"
                )

                checkpoint()


            for model_name in active_model_names:

                if model_name == budget_model:
                    continue

                if model_name == "TabPFN":
                    X_train_model = X_train_tabpfn
                    X_test_model = X_test_tabpfn
                    X_train_sub_model = X_train_sub_tabpfn
                    X_val_model = X_val_tabpfn
                else:
                    X_train_model = X_train_clean
                    X_test_model = X_test_clean
                    X_train_sub_model = X_train_sub_clean
                    X_val_model = X_val_clean

                base_row = base_row_for_model(
                    model_name=model_name,
                    iteration_num=i,
                    split_info=split_info,
                    inner_train_class1_prev=inner_train_class1_prev,
                    val_class1_prev=val_class1_prev,
                )

                model_is_tuned = bool(
                    get_model_cfg(model_name).get("tuned_by_optuna", False)
                )

                try:
                    if model_is_tuned:
                        outputs, total_runtime = v14_execute_model_call(
                            model_name,
                            i,
                            scenario_name,
                            TOTAL_N,
                            lambda: MODEL_RUNNERS[model_name](
                                X_train_model,
                                X_test_model,
                                y_train,
                                y_test,
                                X_train_sub_model,
                                y_train_sub,
                                X_val_model,
                                y_val,
                                i,
                                budget_runtime
                            ),
                            time_budget_seconds=budget_runtime,
                        )
                    else:
                        if (
                            cap_untuned_competitors
                            and budget_runtime is not None
                            and np.isfinite(budget_runtime)
                        ):
                            outputs, total_runtime = v14_execute_model_call(
                                model_name,
                                i,
                                scenario_name,
                                TOTAL_N,
                                lambda: MODEL_RUNNERS[model_name](
                                    X_train_model,
                                    X_test_model,
                                    y_train,
                                    y_test,
                                    i,
                                    budget_runtime
                                ),
                                time_budget_seconds=budget_runtime,
                            )
                        else:
                            outputs, total_runtime = v14_execute_model_call(
                                model_name,
                                i,
                                scenario_name,
                                TOTAL_N,
                                lambda: MODEL_RUNNERS[model_name](
                                    X_train_model,
                                    X_test_model,
                                    y_train,
                                    y_test,
                                    i
                                ),
                                time_budget_seconds=None,
                            )

                    record_success(
                        model_name=model_name,
                        base_row=base_row,
                        outputs=outputs,
                        total_runtime=total_runtime,
                        tabpfn_budget=budget_runtime
                    )

                except Exception as e:
                    total_runtime = getattr(
                        e, "v14_scientific_runtime_seconds", None
                    )
                    if total_runtime is None:
                        total_runtime = np.nan

                    record_failure(
                        model_name=model_name,
                        base_row=base_row,
                        error_message=e,
                        total_runtime=total_runtime,
                        tabpfn_budget=budget_runtime
                    )

                checkpoint()


        raw_path = os.path.join(scenario_dir, f"{scenario_prefix}_RawResults.pkl")

        with open(raw_path, "wb") as f:
            pickle.dump(errors, f)
        raw_json_path = os.path.join(scenario_dir, f"{scenario_prefix}_RawResults.json")
        write_json(raw_json_path, errors)

        iter_df = pd.DataFrame(iteration_rows)
        iter_path = os.path.join(scenario_dir, f"{scenario_prefix}_IterationLevel_Metrics.csv")
        iter_df.to_csv(iter_path, index=False)
        iter_json_path = os.path.join(scenario_dir, f"{scenario_prefix}_IterationLevel_Metrics.json")
        write_df_json(iter_df, iter_json_path)

        existing_split_cols = [c for c in split_cols if c in iter_df.columns]
        split_check_path = os.path.join(scenario_dir, f"{scenario_prefix}_SplitDistribution_Check.csv")

        if existing_split_cols:
            split_check_df = iter_df[existing_split_cols].drop_duplicates()
            split_check_df.to_csv(split_check_path, index=False)
            write_df_json(
                split_check_df,
                os.path.join(scenario_dir, f"{scenario_prefix}_SplitDistribution_Check.json")
            )
            split_audit_csv = os.path.join(
                scenario_dir, f"{scenario_prefix}_SplitAudit.csv"
            )
            split_audit_json = os.path.join(
                scenario_dir, f"{scenario_prefix}_SplitAudit.json"
            )
            split_check_df.to_csv(split_audit_csv, index=False)
            write_df_json(split_check_df, split_audit_json)

        summary_df = create_budgeted_summary(
            errors=errors,
            model_names=active_model_names,
            scenario_name=scenario_name
        )

        if not summary_df.empty:
            summary_df["Sampling_Strategy"] = sampling_config.get("strategy", "")

        summary_path = os.path.join(scenario_dir, f"{scenario_prefix}_Summary.csv")
        summary_df.to_csv(summary_path, index=False)
        write_df_json(
            summary_df,
            os.path.join(scenario_dir, f"{scenario_prefix}_Summary.json")
        )

        try:
            export_dashboard_bundle(
                scenario_name=scenario_name,
                scenario_dir=scenario_dir,
                scenario_prefix=scenario_prefix,
                errors=errors,
                iter_df=iter_df,
                summary_df=summary_df,
                active_model_names=active_model_names,
                sample_size=TOTAL_N,
                calibration_bins=PLOT_CONFIG.get("calibration_bins", 10),
            )
        except Exception as _exc:
            print(
                f"   WARNING: Could not write dashboard bundle for "
                f"{scenario_name}: {_exc}"
            )

        display(summary_df)

        budget_timing_df = pd.DataFrame(budget_timing_memory)
        selection_df = pd.DataFrame(selection_memory)
        final_fit_df = pd.DataFrame(final_fit_timing_memory)
        tabpfn_budget_eval_df = pd.DataFrame(tabpfn_budget_evaluation_memory)

        for df in [budget_timing_df, selection_df, final_fit_df, tabpfn_budget_eval_df]:
            if not df.empty:
                df["Scenario"] = scenario_name
                df["Sampling_Strategy"] = sampling_config.get("strategy", "")

        budget_timing_path = os.path.join(
            scenario_dir,
            f"{scenario_prefix}_TabPFN_Budget_Tuning_Time.csv"
        )

        selection_path = os.path.join(
            scenario_dir,
            f"{scenario_prefix}_Budgeted_Hyperparameter_Selection.csv"
        )

        final_fit_path = os.path.join(
            scenario_dir,
            f"{scenario_prefix}_Final_Fit_Predict_Time.csv"
        )

        tabpfn_budget_eval_path = os.path.join(
            scenario_dir,
            f"{scenario_prefix}_TabPFN_Local_Budget_Evaluation.csv"
        )

        budget_timing_df.to_csv(budget_timing_path, index=False)
        selection_df.to_csv(selection_path, index=False)
        final_fit_df.to_csv(final_fit_path, index=False)
        tabpfn_budget_eval_df.to_csv(tabpfn_budget_eval_path, index=False)
        write_df_json(budget_timing_df, budget_timing_path.replace(".csv", ".json"))
        write_df_json(selection_df, selection_path.replace(".csv", ".json"))
        write_df_json(final_fit_df, final_fit_path.replace(".csv", ".json"))
        write_df_json(tabpfn_budget_eval_df, tabpfn_budget_eval_path.replace(".csv", ".json"))

        optuna_trial_dfs = []

        for model_name, obj in optuna_trials_memory.items():
            if isinstance(obj, list) and len(obj) > 0:
                df = pd.concat(obj, ignore_index=True)
                df["Scenario"] = scenario_name
                df["Sampling_Strategy"] = sampling_config.get("strategy", "")
                optuna_trial_dfs.append(df)

        if optuna_trial_dfs:
            optuna_all_df = pd.concat(optuna_trial_dfs, ignore_index=True)
        else:
            optuna_all_df = pd.DataFrame()

        optuna_all_path = os.path.join(
            scenario_dir,
            f"{scenario_prefix}_Optuna_AllTrials.csv"
        )

        optuna_all_df.to_csv(optuna_all_path, index=False)
        write_df_json(optuna_all_df, optuna_all_path.replace(".csv", ".json"))

        all_summaries.append(summary_df)
        all_iteration_metrics.append(iter_df)

        if not budget_timing_df.empty:
            all_budget_timing.append(budget_timing_df)

        if not selection_df.empty:
            all_selection_records.append(selection_df)

        if not final_fit_df.empty:
            all_final_fit_timing.append(final_fit_df)

        if not optuna_all_df.empty:
            all_optuna_trials.append(optuna_all_df)

        if not tabpfn_budget_eval_df.empty:
            all_tabpfn_budget_evaluations.append(tabpfn_budget_eval_df)

        print(f"Finished and saved all outputs for {scenario_name}")
        print(f"Scenario folder:\n{scenario_dir}")



    cross_summary_df = (
        pd.concat(all_summaries, ignore_index=True)
        if all_summaries else pd.DataFrame()
    )

    all_iter_df = (
        pd.concat(all_iteration_metrics, ignore_index=True)
        if all_iteration_metrics else pd.DataFrame()
    )

    all_budget_timing_df = (
        pd.concat(all_budget_timing, ignore_index=True)
        if all_budget_timing else pd.DataFrame()
    )

    all_selection_df = (
        pd.concat(all_selection_records, ignore_index=True)
        if all_selection_records else pd.DataFrame()
    )

    all_final_fit_df = (
        pd.concat(all_final_fit_timing, ignore_index=True)
        if all_final_fit_timing else pd.DataFrame()
    )

    all_optuna_trials_df = (
        pd.concat(all_optuna_trials, ignore_index=True)
        if all_optuna_trials else pd.DataFrame()
    )

    all_tabpfn_budget_eval_df = (
        pd.concat(all_tabpfn_budget_evaluations, ignore_index=True)
        if all_tabpfn_budget_evaluations else pd.DataFrame()
    )


    cross_summary_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_CrossScenario_Summary.csv"
    )
    cross_summary_json_path = cross_summary_path.replace(".csv", ".json")

    all_iter_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_AllIteration_Metrics.csv"
    )
    all_iter_json_path = all_iter_path.replace(".csv", ".json")

    all_budget_timing_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_AllBudget_Tuning_Time.csv"
    )
    all_budget_timing_json_path = all_budget_timing_path.replace(".csv", ".json")

    all_selection_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_AllBudgeted_Hyperparameter_Selection.csv"
    )
    all_selection_json_path = all_selection_path.replace(".csv", ".json")

    all_final_fit_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_AllFinal_Fit_Predict_Time.csv"
    )
    all_final_fit_json_path = all_final_fit_path.replace(".csv", ".json")

    all_optuna_trials_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_AllOptuna_Trials.csv"
    )
    all_optuna_trials_json_path = all_optuna_trials_path.replace(".csv", ".json")

    all_tabpfn_budget_eval_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_AllTabPFN_Local_Budget_Evaluation.csv"
    )
    all_tabpfn_budget_eval_json_path = all_tabpfn_budget_eval_path.replace(".csv", ".json")

    config_snapshot_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_ConfigSnapshot.pkl"
    )
    config_snapshot_json_path = config_snapshot_path.replace(".pkl", ".json")
    run_manifest_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_RunManifest.json"
    )


    cross_summary_df.to_csv(cross_summary_path, index=False)
    all_iter_df.to_csv(all_iter_path, index=False)
    all_budget_timing_df.to_csv(all_budget_timing_path, index=False)
    all_selection_df.to_csv(all_selection_path, index=False)
    all_final_fit_df.to_csv(all_final_fit_path, index=False)
    all_optuna_trials_df.to_csv(all_optuna_trials_path, index=False)
    all_tabpfn_budget_eval_df.to_csv(all_tabpfn_budget_eval_path, index=False)
    write_df_json(cross_summary_df, cross_summary_json_path)
    write_df_json(all_iter_df, all_iter_json_path)
    write_df_json(all_budget_timing_df, all_budget_timing_json_path)
    write_df_json(all_selection_df, all_selection_json_path)
    write_df_json(all_final_fit_df, all_final_fit_json_path)
    write_df_json(all_optuna_trials_df, all_optuna_trials_json_path)
    write_df_json(all_tabpfn_budget_eval_df, all_tabpfn_budget_eval_json_path)

    with open(config_snapshot_path, "wb") as f:
        pickle.dump(
            {
                "PIPELINE_VERSION": PIPELINE_VERSION,
                "RUN_CONFIG": RUN_CONFIG,
                "MODEL_CONFIGS": MODEL_CONFIGS,
                "SCENARIO_CONFIGS": SCENARIO_CONFIGS,
                "PLOT_CONFIG": PLOT_CONFIG,
            },
            f
        )
    write_json(
        config_snapshot_json_path,
        {
            "PIPELINE_VERSION": PIPELINE_VERSION,
            "RUN_CONFIG": RUN_CONFIG,
            "MODEL_CONFIGS": MODEL_CONFIGS,
            "SCENARIO_CONFIGS": SCENARIO_CONFIGS,
            "PLOT_CONFIG": PLOT_CONFIG,
        },
    )
    write_json(
        run_manifest_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "results_dir": RESULTS_DIR,
            "file_prefix": FILE_PREFIX,
            "total_n": TOTAL_N,
            "iterations": ITERATIONS,
            "output_layout": RUN_CONFIG.get("output_layout", {}),
            "scenarios": [
                {
                    "scenario": name,
                    "folder": os.path.join(RESULTS_DIR, scenario_folder_name(name)),
                    "file_prefix": scenario_file_prefix(name, FILE_PREFIX),
                }
                for name in SCENARIO_CONFIGS.keys()
            ],
            "cross_scenario_paths": {
                "summary_csv": cross_summary_path,
                "summary_json": cross_summary_json_path,
                "all_iteration_metrics_csv": all_iter_path,
                "all_iteration_metrics_json": all_iter_json_path,
                "all_tabpfn_budget_evaluation_csv": all_tabpfn_budget_eval_path,
                "all_tabpfn_budget_evaluation_json": all_tabpfn_budget_eval_json_path,
            },
        },
    )


    print("=" * 100)
    print("TABPFN-RUNTIME-BUDGETED MONTE-CARLO PIPELINE COMPLETE")
    print("=" * 100)
    print("Saved cross-scenario files:")
    print("0.", RESULTS_DIR)
    print("1.", cross_summary_path)
    print("2.", all_iter_path)
    print("3.", all_budget_timing_path)
    print("4.", all_selection_path)
    print("5.", all_final_fit_path)
    print("6.", all_optuna_trials_path)
    print("7.", all_tabpfn_budget_eval_path)
    print("8.", config_snapshot_path)
    print("9.", config_snapshot_json_path)
    print("10.", run_manifest_path)
    print("=" * 100)

    display(cross_summary_df)
    display(all_iter_df.head())



    return {
        "cross_summary_df": cross_summary_df,
        "all_iteration_metrics_df": all_iter_df,
        "all_budget_timing_df": all_budget_timing_df,
        "all_selection_df": all_selection_df,
        "all_final_fit_df": all_final_fit_df,
        "all_optuna_trials_df": all_optuna_trials_df,
        "all_tabpfn_budget_evaluation_df": all_tabpfn_budget_eval_df,
        "paths": {
            "results_dir": RESULTS_DIR,
            "cross_summary": cross_summary_path,
            "cross_summary_json": cross_summary_json_path,
            "all_iteration_metrics": all_iter_path,
            "all_iteration_metrics_json": all_iter_json_path,
            "all_budget_timing": all_budget_timing_path,
            "all_budget_timing_json": all_budget_timing_json_path,
            "all_selection": all_selection_path,
            "all_selection_json": all_selection_json_path,
            "all_final_fit": all_final_fit_path,
            "all_final_fit_json": all_final_fit_json_path,
            "all_optuna_trials": all_optuna_trials_path,
            "all_optuna_trials_json": all_optuna_trials_json_path,
            "all_tabpfn_budget_evaluation": all_tabpfn_budget_eval_path,
            "all_tabpfn_budget_evaluation_json": all_tabpfn_budget_eval_json_path,
            "config_snapshot": config_snapshot_path,
            "config_snapshot_json": config_snapshot_json_path,
            "run_manifest": run_manifest_path,
        }
    }


def generate_plots(config=None, scenarios_to_plot=None):
    """Generate Plotly/PPT-ready plots from saved RawResults.pkl files."""
    if config is not None:
        load_config(config)

    if RESULTS_DIR is None or FILE_PREFIX is None:
        raise ValueError(
            "Plot settings are not loaded. Call generate_plots(config=CONFIG) "
            "or run run_monte_carlo/run_all first."
        )

    print("=" * 100)
    print("GENERATING POWERPOINT-READY PLOTS")
    print("=" * 100)

    if scenarios_to_plot is None:
        scenarios_to_plot = discover_scenarios_to_plot(
            results_dir=RESULTS_DIR,
            file_prefix=FILE_PREFIX,
        )

    print("Scenarios selected for plotting:")
    for scenario_name in scenarios_to_plot:
        print(" -", scenario_name)

    created_plot_outputs = []
    plot_output_rows = []

    for scenario_name in scenarios_to_plot:
        try:
            result = create_all_plotly_slides_for_scenario(
                scenario_name=scenario_name,
                results_dir=RESULTS_DIR,
                file_prefix=FILE_PREFIX,
                sample_size=TOTAL_N,
                plot_subfolder_name=PLOT_SUBFOLDER_NAME,
                save_html=SAVE_HTML,
                show_figures=SHOW_FIGURES,
                save_wide_roc_version=SAVE_WIDE_ROC_CALIBRATION_VERSION,
            )

            created_plot_outputs.append(result)

            plot_output_rows.append({
                "Scenario": scenario_name,
                "Output_Dir": result["output_dir"],
                "Slide1_HTML": result["slide1_paths"].get("html", ""),
                "Slide1_PNG": result["slide1_paths"].get("png", ""),
                "Slide1_JPEG": result["slide1_paths"].get("jpeg", ""),
                "Slide1_SVG": result["slide1_paths"].get("svg", ""),
                "Slide2_HTML": result["slide2_paths"].get("html", ""),
                "Slide2_PNG": result["slide2_paths"].get("png", ""),
                "Slide2_JPEG": result["slide2_paths"].get("jpeg", ""),
                "Slide2_SVG": result["slide2_paths"].get("svg", ""),
                "Slide3_HTML": result["slide3_paths"].get("html", ""),
                "Slide3_PNG": result["slide3_paths"].get("png", ""),
                "Slide3_JPEG": result["slide3_paths"].get("jpeg", ""),
                "Slide3_SVG": result["slide3_paths"].get("svg", ""),
                "Status": "Success",
                "Error": "",
            })

            print(
                f"Finished plots for {scenario_name}\n"
                f"Saved in:\n{result['output_dir']}"
            )

        except Exception as exc:
            print(
                f"WARNING: Could not create plots for {scenario_name}: {exc}"
            )
            plot_output_rows.append({
                "Scenario": scenario_name,
                "Output_Dir": "",
                "Slide1_HTML": "",
                "Slide1_PNG": "",
                "Slide1_JPEG": "",
                "Slide1_SVG": "",
                "Slide2_HTML": "",
                "Slide2_PNG": "",
                "Slide2_JPEG": "",
                "Slide2_SVG": "",
                "Slide3_HTML": "",
                "Slide3_PNG": "",
                "Slide3_JPEG": "",
                "Slide3_SVG": "",
                "Status": "Failed",
                "Error": str(exc),
            })

    plot_outputs_df = pd.DataFrame(plot_output_rows)
    plot_outputs_path = os.path.join(
        RESULTS_DIR,
        f"{FILE_PREFIX}_PlotlySlideOutputs.csv",
    )
    plot_outputs_df.to_csv(plot_outputs_path, index=False)

    try:
        display(plot_outputs_df)
    except Exception:
        pass

    print("=" * 100)
    print("ALL AVAILABLE SCENARIO PLOTS COMPLETE")
    print("=" * 100)
    print("Plot output index saved to:")
    print(plot_outputs_path)

    return {
        "created_plot_outputs": created_plot_outputs,
        "plot_outputs_df": plot_outputs_df,
        "plot_outputs_path": plot_outputs_path,
    }


def generate_all_plots(config=None, scenarios_to_plot=None):
    return generate_plots(config=config, scenarios_to_plot=scenarios_to_plot)


def run_all(X, y, config, groups=None, timestamps=None):
    """Run simulation first, then generate and save/show plots."""
    monte_carlo_outputs = run_monte_carlo(
        X, y, config, groups=groups, timestamps=timestamps
    )
    plot_outputs = generate_plots()
    return {
        "monte_carlo": monte_carlo_outputs,
        "plots": plot_outputs,
    }



# =============================================================================
# 6. Resource control, monitoring, and reproducible artifact I/O
# =============================================================================

V14_EVIDENCE_LEVELS = {
    "1": [
        "split membership",
        "predictions",
        "model configuration",
        "Optuna trials",
        "runtime",
        "measured energy",
        "TabPFN context evidence",
        "device metadata",
    ],
    "2": ["iteration metrics"],
    "3": ["aggregated statistics"],
    "4": ["plot-data datasets"],
    "5": ["figures"],
}

V14_REQUIRED_MODEL_FILES = (
    "metrics.json",
    "runtime_breakdown.json",
    "energy_breakdown.json",
    "final_model_config.json",
)

V14_PREDICTION_METRICS = (
    "AUROC",
    "Balanced_Accuracy",
    "Sensitivity",
    "Precision",
    "Brier_Score",
)


def v14_utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


V14_CPU_MONITORING_MEMORY = []


def v14_detect_cpu_capacity():
    """Return host and process-available logical CPU counts without guessing."""
    detected = os.cpu_count()
    candidates = []
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        try:
            value = process_cpu_count()
            if value:
                candidates.append(("os.process_cpu_count", int(value)))
        except Exception:
            pass
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            affinity_n = len(get_affinity(0))
            if affinity_n:
                candidates.append(("os.sched_getaffinity", int(affinity_n)))
        except Exception:
            pass
    if psutil is not None:
        try:
            affinity = psutil.Process(os.getpid()).cpu_affinity()
            if affinity:
                candidates.append(("psutil.Process.cpu_affinity", len(affinity)))
        except Exception:
            pass
    if detected:
        candidates.append(("os.cpu_count", int(detected)))
    if not candidates:
        return {
            "CPU_Count_Detected": None,
            "CPU_Count_Available_To_Process": None,
            "CPU_Availability_Source": None,
            "CPU_Availability_Unavailable_Reason": (
                "No supported CPU-count or affinity API returned a value."
            ),
        }
    available = min(value for _, value in candidates if value > 0)
    sources = [source for source, value in candidates if value == available]
    return {
        "CPU_Count_Detected": int(detected) if detected else None,
        "CPU_Count_Available_To_Process": int(available),
        "CPU_Availability_Source": "+".join(sorted(set(sources))),
        "CPU_Availability_Unavailable_Reason": None,
    }


def v14_resolve_cpu_parallelism(config, model_cfg=None):
    global_cfg = copy.deepcopy(config.get("cpu_parallelism", {}))
    model_override = copy.deepcopy((model_cfg or {}).get("cpu_parallelism", {}))
    resolved_cfg = v14_deep_merge(global_cfg, model_override)
    capacity = v14_detect_cpu_capacity()
    available = capacity["CPU_Count_Available_To_Process"]
    policy = str(resolved_cfg.get("policy", "max_available")).lower()
    threads_setting = resolved_cfg.get("threads", "auto")
    if policy not in {"max_available", "explicit"}:
        raise ValueError(
            "cpu_parallelism.policy must be 'max_available' or 'explicit'."
        )
    if isinstance(threads_setting, str):
        normalized = threads_setting.strip().lower()
        if normalized not in {"auto", "max_available"}:
            raise ValueError(
                "cpu_parallelism.threads must be 'auto', 'max_available', "
                "or a positive integer."
            )
        if not available:
            raise RuntimeError(
                "Automatic CPU parallelism was requested but process CPU "
                "capacity could not be detected."
            )
        threads = int(available)
        source = f"{normalized}:{capacity['CPU_Availability_Source']}"
    else:
        threads = int(threads_setting)
        if threads <= 0:
            raise ValueError("cpu_parallelism.threads must be positive.")
        source = "explicit_json"
    allow_oversubscription = bool(
        resolved_cfg.get("allow_oversubscription", False)
    )
    if (
        available
        and threads > available
        and not allow_oversubscription
    ):
        raise ValueError(
            f"Configured CPU threads ({threads}) exceed CPUs available to the "
            f"process ({available}); set allow_oversubscription=true only with "
            "an explicit library justification."
        )
    return {
        **capacity,
        "CPU_Parallelism_Policy": policy,
        "CPU_Threads_Resolved": int(threads),
        "CPU_Thread_Setting_Source": source,
        "CPU_Allow_Oversubscription": allow_oversubscription,
        "CPU_Prevent_Nested_Oversubscription": bool(
            resolved_cfg.get("prevent_nested_oversubscription", True)
        ),
        "Optuna_Trial_N_Jobs": 1,
        "Monte_Carlo_Execution": "sequential",
    }


def v14_apply_global_cpu_environment(config):
    """Set conservative process-wide defaults; runtime limits remain authoritative."""
    cpu = v14_resolve_cpu_parallelism(config)
    threads = str(cpu["CPU_Threads_Resolved"])
    explicit_environment = config.get("cpu_parallelism", {}).get(
        "environment", {}
    )
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for variable in variables:
        requested = explicit_environment.get(variable)
        if requested is not None:
            os.environ[variable] = str(requested)
        else:
            os.environ.setdefault(variable, threads)
    cpu.update({variable: os.environ.get(variable) for variable in variables})
    return cpu


def v14_cpu_model_thread_parameter(model_name):
    return {
        "RandomForest": "n_jobs",
        "XGBoost": "n_jobs",
        "CatBoost": "thread_count",
        "L-SLR": "n_jobs_and_native_threadpools",
        "Augmented_SLR": "n_jobs_and_native_threadpools",
        "TabPFN": "torch_and_native_threadpools",
    }.get(model_name)


def v14_apply_cpu_model_params(model_name, fixed_params):
    """Map the common policy to supported estimator parameters only."""
    params = copy.deepcopy(fixed_params)
    cfg = get_model_cfg(model_name)
    resolved_device = "cpu"
    if model_name in {"XGBoost", "CatBoost"}:
        execution = cfg.get("execution", {})
        resolved_device = v14_resolve_requested_device(
            execution.get("device", "auto"),
            execution.get("require_requested_device", False),
            model_name,
        )
    if resolved_device != "cpu":
        return params
    cpu = v14_resolve_cpu_parallelism(RUN_CONFIG, cfg)
    threads = cpu["CPU_Threads_Resolved"]
    if model_name in {"RandomForest", "XGBoost", "L-SLR", "Augmented_SLR"}:
        params["n_jobs"] = threads
    elif model_name == "CatBoost":
        params["thread_count"] = threads
    return params


def v14_execution_resource_spec(model_name, time_budget_seconds=None):
    cfg = get_model_cfg(model_name)
    execution_path = "local"
    resolved_device = "cpu"
    if model_name == "TabPFN":
        execution = cfg.get("execution", {})
        requested_path = str(
            execution.get("path", execution.get("execution_path", "auto"))
        ).lower()
        requested_path = {
            "client": "cloud",
            "cloud_client": "cloud",
            "cloud/client": "cloud",
            "local_cpu": "local",
            "local_gpu": "local",
        }.get(requested_path, requested_path)
        finite_budget = (
            time_budget_seconds is not None
            and np.isfinite(time_budget_seconds)
            and float(time_budget_seconds) > 0
        )
        execution_path = (
            ("local" if finite_budget else "cloud")
            if requested_path == "auto" else requested_path
        )
        if execution_path == "cloud":
            return {
                "Execution_Path": "cloud",
                "Resolved_Device": "remote_unknown",
                "CPU_Threads_Configured": None,
                "CPU_Thread_Parameter": None,
                "CPU_Thread_Limitation": (
                    "Remote TabPFN compute is outside the local runtime."
                ),
                **v14_detect_cpu_capacity(),
            }
        requested_device = execution.get(
            "local_device", cfg.get("tabpfn_device", "auto")
        )
        resolved_device = v14_resolve_requested_device(
            requested_device,
            execution.get("require_requested_device", False),
            model_name,
        )
    elif model_name in {"XGBoost", "CatBoost"}:
        execution = cfg.get("execution", {})
        resolved_device = v14_resolve_requested_device(
            execution.get("device", "auto"),
            execution.get("require_requested_device", False),
            model_name,
        )
    if resolved_device != "cpu":
        return {
            "Execution_Path": execution_path,
            "Resolved_Device": resolved_device,
            "CPU_Threads_Configured": None,
            "CPU_Thread_Parameter": None,
            "CPU_Thread_Limitation": (
                "Local CPU parallelism does not control a CUDA execution."
            ),
            **v14_detect_cpu_capacity(),
        }
    cpu = v14_resolve_cpu_parallelism(RUN_CONFIG, cfg)
    limitation = None
    if model_name in {"L-SLR", "Augmented_SLR"}:
        limitation = (
            "The configured solver contains serial regions; n_jobs/native "
            "thread pools are configured without changing the solver."
        )
    return {
        "Execution_Path": execution_path,
        "Resolved_Device": resolved_device,
        "CPU_Threads_Configured": cpu["CPU_Threads_Resolved"],
        "CPU_Thread_Parameter": v14_cpu_model_thread_parameter(model_name),
        "CPU_Thread_Limitation": limitation,
        **cpu,
    }


@contextmanager
def v14_cpu_thread_context(model_name, resource_spec):
    if resource_spec.get("Resolved_Device") != "cpu":
        yield
        return
    threads = int(resource_spec["CPU_Threads_Configured"])
    prevent_nested = bool(
        resource_spec.get("CPU_Prevent_Nested_Oversubscription", True)
    )
    internally_parallel = model_name in {
        "RandomForest", "XGBoost", "CatBoost"
    }
    native_limit = 1 if prevent_nested and internally_parallel else threads
    pool_context = (
        threadpool_limits(limits=native_limit)
        if threadpool_limits is not None else contextlib.nullcontext()
    )
    prior_torch_threads = None
    if model_name == "TabPFN" and torch is not None:
        try:
            prior_torch_threads = int(torch.get_num_threads())
            torch.set_num_threads(threads)
            resource_spec["Torch_Num_Threads"] = int(torch.get_num_threads())
        except Exception as exc:
            resource_spec["Torch_Num_Threads"] = None
            resource_spec["Torch_Thread_Setting_Unavailable_Reason"] = str(exc)
    resource_spec["Native_Threadpool_Limit"] = native_limit
    resource_spec["Threadpoolctl_Available"] = threadpool_limits is not None
    try:
        with pool_context:
            yield
    finally:
        if prior_torch_threads is not None:
            try:
                torch.set_num_threads(prior_torch_threads)
            except Exception:
                pass


# CPU monitoring is observational. It does not define HPO, reference-budget,
# final-fit, prediction, or TabPFN context-budget timer boundaries.
class V14CPUResourceMonitor:
    def __init__(
        self,
        model_name,
        execution_variant,
        scenario_name,
        sample_size,
        iteration,
        total_iterations,
        resource_spec,
        monitoring_config,
    ):
        self.model_name = model_name
        self.execution_variant = execution_variant
        self.scenario_name = scenario_name
        self.sample_size = sample_size
        self.iteration = int(iteration)
        self.total_iterations = int(total_iterations)
        self.resource_spec = resource_spec
        self.config = monitoring_config
        self.interval = float(
            monitoring_config.get("sampling_interval_seconds", 5.0)
        )
        if self.interval <= 0:
            raise ValueError(
                "cpu_monitoring.sampling_interval_seconds must be positive."
            )
        self.samples = []
        self.started_perf = None
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.unavailable_reason = None

    def start(self):
        if self.resource_spec.get("Execution_Path") == "cloud":
            if self.config.get("show_console", True):
                print(
                    f"[REMOTE EXECUTION] Model={self.execution_variant} | "
                    "Path=cloud/client | Local CPU cores do not represent "
                    "remote server compute."
                )
            self.unavailable_reason = (
                "Remote execution: local monitoring cannot measure server compute."
            )
            return
        if self.resource_spec.get("Resolved_Device") != "cpu":
            if self.config.get("show_console", True):
                print(
                    f"[LOCAL GPU EXECUTION] Model={self.execution_variant} | "
                    f"Device={self.resource_spec.get('Resolved_Device')} | "
                    "CPU monitoring is not a measure of GPU utilization."
                )
            self.unavailable_reason = "Execution is not local CPU."
            return
        available = self.resource_spec.get("CPU_Count_Available_To_Process")
        configured = self.resource_spec.get("CPU_Threads_Configured")
        policy = self.resource_spec.get("CPU_Parallelism_Policy")
        if self.config.get("show_console", True):
            print(
                f"[CPU CONFIG] Model={self.execution_variant} | "
                f"Available CPUs={available} | Configured Threads={configured} | "
                f"Policy={policy}"
            )
        if not self.config.get("enabled", True):
            self.unavailable_reason = "CPU monitoring disabled by JSON."
            return
        if psutil is None:
            self.unavailable_reason = "psutil is unavailable."
            return
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        self.started_perf = time.perf_counter()
        self.thread = threading.Thread(
            target=self._loop,
            name=f"v14-cpu-monitor-{safe_path_part(self.execution_variant)}",
            daemon=True,
        )
        self.thread.start()

    def _sample(self, show=False):
        if self.process is None or self.started_perf is None:
            return
        try:
            process_percent = float(self.process.cpu_percent(interval=None))
            system_percent = float(psutil.cpu_percent(interval=None))
            sample = {
                "timestamp_utc": v14_utc_now(),
                "elapsed_seconds": float(
                    time.perf_counter() - self.started_perf
                ),
                "process_cpu_percent": process_percent,
                "approx_active_logical_cores": process_percent / 100.0,
                "system_cpu_percent": system_percent,
            }
            self.samples.append(sample)
            if show and self.config.get("show_console", True):
                print(
                    f"[CPU] {self.execution_variant} | "
                    f"Iter {self.iteration}/{self.total_iterations} | "
                    f"available={self.resource_spec.get('CPU_Count_Available_To_Process')} | "
                    f"configured={self.resource_spec.get('CPU_Threads_Configured')} | "
                    f"active~{sample['approx_active_logical_cores']:.2f} cores | "
                    f"process={process_percent:.1f}%"
                )
        except Exception as exc:
            self.unavailable_reason = f"CPU sample failed: {exc}"

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample(show=True)

    def stop(self):
        if self.thread is not None:
            self._sample(show=False)
            self.stop_event.set()
            self.thread.join(timeout=min(self.interval, 1.0))
        values = [
            sample["process_cpu_percent"] for sample in self.samples
            if np.isfinite(sample["process_cpu_percent"])
        ]
        enough = len(values) > 0
        monitoring_enabled = bool(self.config.get("enabled", True))
        monitoring_active = bool(
            monitoring_enabled
            and self.process is not None
            and self.started_perf is not None
        )
        summary = {
            "Scenario": self.scenario_name,
            "Sample_Size_Actual": self.sample_size,
            "Iteration": self.iteration,
            "Model": self.model_name,
            "Execution_Variant": self.execution_variant,
            "Execution_Path": self.resource_spec.get("Execution_Path"),
            "Resolved_Device": self.resource_spec.get("Resolved_Device"),
            "CPU_Cores_Available": self.resource_spec.get(
                "CPU_Count_Available_To_Process"
            ),
            "CPU_Threads_Configured": self.resource_spec.get(
                "CPU_Threads_Configured"
            ),
            "CPU_Parallelism_Policy": self.resource_spec.get(
                "CPU_Parallelism_Policy"
            ),
            "CPU_Thread_Setting_Source": self.resource_spec.get(
                "CPU_Thread_Setting_Source"
            ),
            "Model_CPU_Thread_Parameter": self.resource_spec.get(
                "CPU_Thread_Parameter"
            ),
            "Model_CPU_Thread_Value": self.resource_spec.get(
                "CPU_Threads_Configured"
            ),
            "CPU_Process_Percent_Mean": (
                float(np.mean(values)) if enough else None
            ),
            "CPU_Process_Percent_Peak": (
                float(np.max(values)) if enough else None
            ),
            "CPU_Approx_Active_Cores_Mean": (
                float(np.mean(values) / 100.0) if enough else None
            ),
            "CPU_Approx_Active_Cores_Peak": (
                float(np.max(values) / 100.0) if enough else None
            ),
            "CPU_Monitoring_Samples_N": int(len(values)),
            "CPU_Monitoring_Interval_Seconds": self.interval,
            "CPU_Monitoring_Enabled": monitoring_enabled,
            "CPU_Monitoring_Active_During_Model_Call": monitoring_active,
            "CPU_Monitoring_Evidence_Available": enough,
            "CPU_Monitoring_Unavailable_Reason": (
                None if enough else (
                    self.unavailable_reason
                    or "No utilization sample was collected."
                )
            ),
            "CPU_Monitoring_Overhead_Handling": (
                "Lightweight monitoring runs concurrently; negligible "
                "monitoring overhead may be included in observed wall-clock "
                "runtime. No estimated overhead is subtracted."
                if monitoring_active else
                "CPU monitoring did not run concurrently with this execution."
            ),
            "CPU_Monitoring_Excluded_From_Runtime": (
                False if monitoring_active else None
            ),
            "CPU_Monitoring_Budget_Boundary_Role": (
                "none; monitoring does not define HPO, final-fit, prediction, "
                "reference-budget, or TabPFN context-budget timer boundaries"
            ),
            "Torch_Num_Threads": self.resource_spec.get("Torch_Num_Threads"),
            "Native_Threadpool_Limit": self.resource_spec.get(
                "Native_Threadpool_Limit"
            ),
            "Optuna_Trial_N_Jobs": 1,
            "Monte_Carlo_Execution": "sequential",
            "samples": copy.deepcopy(self.samples),
        }
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            summary[variable] = os.environ.get(variable)
        if self.config.get("show_console", True):
            if enough:
                print(
                    f"[CPU SUMMARY] Model={self.execution_variant} | "
                    f"available={summary['CPU_Cores_Available']} | "
                    f"configured={summary['CPU_Threads_Configured']} | "
                    f"mean={summary['CPU_Process_Percent_Mean']:.1f}% | "
                    f"peak={summary['CPU_Process_Percent_Peak']:.1f}% | "
                    f"samples={summary['CPU_Monitoring_Samples_N']}"
                )
            else:
                print(
                    f"[CPU SUMMARY] Model={self.execution_variant} | "
                    f"utilization unavailable: "
                    f"{summary['CPU_Monitoring_Unavailable_Reason']}"
                )
        return summary


def v14_execute_model_call(
    model_name,
    iteration_num,
    scenario_name,
    sample_size,
    call,
    time_budget_seconds=None,
    execution_variant=None,
):
    """
    Execute one model call using an observed stopwatch-style wall-clock timer.

    Monitor creation/start/stop is outside this broad timer. Lightweight CPU
    sampling may run concurrently during the call, so negligible monitor
    overhead can be included in Actual_Total_Runtime_Seconds. Dedicated Optuna,
    final-fit/prediction, and TabPFN context timers remain authoritative for
    their respective scientific budget boundaries.
    """
    variant = execution_variant or str(
        (RUN_CONFIG or {}).get("v14_execution_variant") or model_name
    )
    display_scenario = str(
        (RUN_CONFIG or {}).get("v14_display_scenario_name", scenario_name)
    )
    resource_spec = v14_execution_resource_spec(
        model_name, time_budget_seconds=time_budget_seconds
    )
    monitoring = copy.deepcopy(
        (RUN_CONFIG or {}).get("cpu_monitoring", {})
    )
    monitor = V14CPUResourceMonitor(
        model_name=model_name,
        execution_variant=variant,
        scenario_name=display_scenario,
        sample_size=sample_size,
        iteration=int(
            (RUN_CONFIG or {}).get(
                "v14_external_iteration", int(iteration_num) + 1
            )
        ),
        total_iterations=int(
            (RUN_CONFIG or {}).get(
                "v14_total_iterations",
                (RUN_CONFIG or {}).get("iterations", 1),
            )
        ),
        resource_spec=resource_spec,
        monitoring_config=monitoring,
    )
    monitor.start()
    scientific_runtime = None
    try:
        with v14_cpu_thread_context(model_name, resource_spec):
            started = time.perf_counter()
            try:
                output = call()
            finally:
                scientific_runtime = time.perf_counter() - started
        return output, scientific_runtime
    except Exception as exc:
        try:
            setattr(exc, "v14_scientific_runtime_seconds", scientific_runtime)
        except Exception:
            pass
        raise
    finally:
        summary = monitor.stop()
        summary["Scientific_Runtime_Seconds"] = scientific_runtime
        V14_CPU_MONITORING_MEMORY.append(summary)


def v14_latest_cpu_monitor_record(
    scenario_name, sample_size, iteration, execution_variant
):
    for record in reversed(V14_CPU_MONITORING_MEMORY):
        if (
            str(record.get("Scenario")) == str(scenario_name)
            and int(record.get("Sample_Size_Actual")) == int(sample_size)
            and int(record.get("Iteration")) == int(iteration)
            and str(record.get("Execution_Variant")) == str(execution_variant)
        ):
            return copy.deepcopy(record)
    return None


# Atomic writes and checksums support evidence validation and safe resume.
def v14_json_safe(value):
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [v14_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): v14_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [v14_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def v14_canonical_json(value):
    return json.dumps(
        v14_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def v14_sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def v14_sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v14_config_hash(config):
    excluded = {
        "_config_source",
        "_resolved_at",
        "_run_dir",
        "_resume",
        "_force_rerun",
    }
    stable = {
        key: value for key, value in config.items()
        if key not in excluded
    }
    return v14_sha256_bytes(v14_canonical_json(stable).encode("utf-8"))


def v14_atomic_write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def v14_atomic_write_text(path, content):
    v14_atomic_write_bytes(path, str(content).encode("utf-8"))


def v14_atomic_write_json(path, value):
    payload = json.dumps(
        v14_json_safe(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    v14_atomic_write_text(path, payload + "\n")


def v14_atomic_write_csv(path, frame):
    text_buffer = io.StringIO()
    frame.to_csv(text_buffer, index=False)
    v14_atomic_write_text(path, text_buffer.getvalue())


def v14_atomic_write_npy(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def v14_atomic_write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def v14_write_table_bundle(frame, base_path):
    base_path = Path(base_path)
    frame = frame.copy()
    if "artifact_schema_version" not in frame.columns:
        frame.insert(0, "artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
    v14_atomic_write_csv(base_path.with_suffix(".csv"), frame)
    v14_atomic_write_json(
        base_path.with_suffix(".json"), frame.to_dict(orient="records")
    )
    parquet_path = base_path.with_suffix(".parquet")
    parquet_status = {
        "available": False,
        "path": str(parquet_path),
        "reason": None,
    }
    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{parquet_path.name}.",
            suffix=".parquet",
            dir=str(parquet_path.parent),
        )
        os.close(fd)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, parquet_path)
        parquet_status["available"] = True
    except Exception as exc:
        parquet_status["reason"] = str(exc)
        try:
            if "temporary" in locals() and os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass
        v14_atomic_write_json(
            str(parquet_path) + ".unavailable.json",
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "format": "parquet",
                "available": False,
                "reason": str(exc),
                "authoritative_fallbacks": [
                    str(base_path.with_suffix(".csv")),
                    str(base_path.with_suffix(".json")),
                ],
            },
        )
    v14_atomic_write_json(
        base_path.with_suffix(".metadata.json"),
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": v14_utc_now(),
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "formats": {
                "csv": str(base_path.with_suffix(".csv")),
                "json": str(base_path.with_suffix(".json")),
                "parquet": (
                    str(parquet_path)
                    if parquet_status["available"] else None
                ),
            },
            "parquet_status": parquet_status,
        },
    )
    return parquet_status


def v14_read_table_bundle(base_path):
    base_path = Path(base_path)
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as handle:
            return pd.DataFrame(json.load(handle))
    raise FileNotFoundError(f"No persisted table bundle found for {base_path}.")


class V14RunLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, **fields):
        record = {
            "timestamp_utc": v14_utc_now(),
            "event": event,
            **v14_json_safe(fields),
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


# =============================================================================
# 7. JSON resolution, devices, environment capture, and dataset loading
# =============================================================================

def v14_deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = v14_deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def v14_builtin_base_config():
    base = copy.deepcopy(CONFIG)
    base.update({
        "experiment_name": "V14_Experiment",
        "base_seed": 2025,
        "output_root": "./V14_Experiments",
        "output_folder": "",
        "file_prefix": "V14",
    })
    base["outputs"] = {
        "root": "./V14_Experiments",
        "save_fitted_models": False,
        "save_predictions_parquet": True,
        "save_predictions_npz": True,
        "save_group_ids": True,
        "artifact_index": True,
    }
    base["energy"] = {
        "enabled": False,
        "required": False,
        "tracking_mode": "process",
        "save_client_side_cloud_energy": True,
        "save_codecarbon_metadata": True,
        "allow_unavailable": True,
    }
    base["budgeting"] = v14_deep_merge(
        base.get("budgeting", {}),
        {
            "non_tuned_reference_runtime_basis": (
                "reference_execution_runtime"
            )
        },
    )
    base["execution"] = {
        "on_model_error": "continue",
        "on_iteration_error": "continue",
        "deterministic_torch": False,
    }
    base["cpu_parallelism"] = {
        "policy": "max_available",
        "threads": "auto",
        "prevent_nested_oversubscription": True,
        "allow_oversubscription": False,
        "environment": {},
    }
    base["cpu_monitoring"] = {
        "enabled": True,
        "show_console": True,
        "sampling_interval_seconds": 5.0,
        "save_timeseries": True,
    }
    base["auxiliary_comparators"] = []
    base["plots"] = v14_deep_merge(base.get("plots", {}), {
        "enabled": True,
        "save_html": True,
        "save_png": True,
        "save_svg": True,
        "show_figures": False,
        "dpi": 180,
        "calibration_bins": 10,
        "runtime_axis_scale": "linear",
        "scenario_roles": {},
        "plot_specs": [],
    })
    base["data"] = {
        "sklearn_dataset": "breast_cancer",
        "target": "target",
        "group_column": None,
        "drop_columns": [],
        "keep_dataframe": True,
    }
    base["dataset_loader"] = {
        "module": "dataset_loader",
        "module_path": None,
        "required_functions": [
            "prepare_dataset",
            "_load_dataframe",
        ],
    }
    base["optuna"] = {
        "direction": base.get("optuna_direction", "maximize"),
        "max_trials": base.get("default_max_trials", 50),
    }
    for model_name, model_cfg in base["models"].items():
        model_cfg.setdefault("execution", {})
        if model_name == "TabPFN":
            model_cfg["execution"].setdefault("path", "auto")
            model_cfg["execution"].setdefault("local_device", "auto")
            model_cfg["execution"].setdefault(
                "require_requested_device", False
            )
        elif model_name in {"XGBoost", "CatBoost"}:
            model_cfg["execution"].setdefault("device", "auto")
            model_cfg["execution"].setdefault(
                "require_requested_device", False
            )
        else:
            model_cfg["execution"].setdefault("device", "cpu")
    return base


def v14_load_json_file(path):
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a JSON object: {path}")
    return value


def v14_load_config_with_extends(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle detected at {path}.")
    seen.add(path)
    current = v14_load_json_file(path)
    parent_value = current.pop("extends", None)
    resolved = {}
    if parent_value:
        parents = parent_value if isinstance(parent_value, list) else [parent_value]
        for parent in parents:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            resolved = v14_deep_merge(
                resolved,
                v14_load_config_with_extends(parent_path, seen=seen),
            )
    resolved = v14_deep_merge(resolved, current)
    resolved["_config_source"] = str(path)
    return resolved


def v14_normalize_auxiliary_comparators(value):
    if value is None:
        return []
    if isinstance(value, dict):
        normalized_input = []
        for comparator_id, comparator_cfg in value.items():
            current = copy.deepcopy(comparator_cfg)
            current.setdefault("id", str(comparator_id))
            normalized_input.append(current)
    elif isinstance(value, list):
        normalized_input = copy.deepcopy(value)
    else:
        raise ValueError(
            "auxiliary_comparators must be a JSON array or object keyed by ID."
        )
    result = []
    seen = set()
    for raw in normalized_input:
        if not isinstance(raw, dict):
            raise ValueError("Each auxiliary comparator must be a JSON object.")
        comparator_id = str(raw.get("id", "")).strip()
        if not comparator_id:
            raise ValueError("Every auxiliary comparator requires a stable id.")
        if comparator_id in seen:
            raise ValueError(
                f"Duplicate auxiliary comparator id: {comparator_id!r}."
            )
        seen.add(comparator_id)
        base_model = str(raw.get("base_model", "")).strip()
        if not base_model:
            raise ValueError(
                f"Auxiliary comparator {comparator_id!r} requires base_model."
            )
        execution = v14_deep_merge(
            {"path": "local", "local_device": "cpu"},
            raw.get("execution", {}),
        )
        budgeting = v14_deep_merge(
            {"apply_runtime_budget": False}, raw.get("budgeting", {})
        )
        context = raw.get("context", {"strategy": "full"})
        if isinstance(context, str):
            context = {"strategy": context}
        if not isinstance(context, dict):
            raise ValueError(
                f"Comparator {comparator_id!r} context must be an object/string."
            )
        strategy = str(context.get("strategy", "full")).lower()
        if strategy not in {"fixed", "adaptive", "full"}:
            raise ValueError(
                f"Comparator {comparator_id!r} context.strategy must be "
                "'fixed', 'adaptive', or 'full'."
            )
        result.append({
            **raw,
            "id": comparator_id,
            "enabled": bool(raw.get("enabled", True)),
            "base_model": base_model,
            "analysis_role": "auxiliary_comparator",
            "execution": execution,
            "context": {**context, "strategy": strategy},
            "budgeting": budgeting,
            "pair_with": raw.get("pair_with", base_model),
            "scenario_applicability": raw.get(
                "scenario_applicability", raw.get("scenarios")
            ),
            "sample_size_applicability": raw.get(
                "sample_size_applicability", raw.get("sample_sizes")
            ),
            "overrides": copy.deepcopy(raw.get("overrides", {})),
        })
    return result


def v14_resolve_context_plan(local_cfg, full_train_n, total_sample_n):
    """Normalize canonical fixed/adaptive/full semantics, including V12 aliases."""
    raw_context = local_cfg.get(
        "context_strategy", local_cfg.get("context", {})
    )
    if isinstance(raw_context, str):
        context_cfg = {"strategy": raw_context}
    elif isinstance(raw_context, dict):
        context_cfg = copy.deepcopy(raw_context)
    else:
        raise ValueError("TabPFN context_strategy must be a string or object.")
    explicit_strategy = context_cfg.get("strategy", context_cfg.get("mode"))
    if explicit_strategy is None:
        if bool(local_cfg.get("fixed_minimum_context_only", False)):
            strategy = "fixed"
            strategy_source = "legacy_fixed_minimum_context_only"
        elif not bool(local_cfg.get("adaptive_context_enabled", True)):
            strategy = "full"
            strategy_source = "legacy_adaptive_context_enabled_false"
        else:
            strategy = "adaptive"
            strategy_source = "legacy_adaptive_context_enabled_true"
    else:
        strategy = str(explicit_strategy).lower()
        strategy_source = "context_strategy"
    if strategy not in {"fixed", "adaptive", "full"}:
        raise ValueError(
            "TabPFN context strategy must be 'fixed', 'adaptive', or 'full'."
        )
    full_train_n = int(full_train_n)
    total_sample_n = int(total_sample_n)
    absolute = context_cfg.get(
        "rows",
        context_cfg.get(
            "context_rows",
            local_cfg.get(
                "fixed_context_rows",
                local_cfg.get("minimum_context_rows"),
            ),
        ),
    )
    fraction = context_cfg.get(
        "fraction",
        context_cfg.get(
            "context_fraction",
            local_cfg.get(
                "minimum_context_fraction",
                local_cfg.get("min_context_fraction"),
            ),
        ),
    )
    denominator_name = str(
        context_cfg.get(
            "fraction_denominator",
            local_cfg.get("context_fraction_denominator", "total_sample"),
        )
    ).lower()
    if denominator_name not in {"total_sample", "outer_train"}:
        raise ValueError(
            "context fraction_denominator must be 'total_sample' or 'outer_train'."
        )
    configured_rule = None
    requested = 0
    if absolute is not None:
        requested = int(absolute)
        if requested <= 0:
            raise ValueError("Configured TabPFN context rows must be positive.")
        configured_rule = f"absolute_rows:{requested}"
    elif fraction is not None:
        fraction = float(fraction)
        if not 0 < fraction <= 1:
            raise ValueError(
                "Configured TabPFN context fraction must be in (0, 1]."
            )
        denominator = (
            total_sample_n
            if denominator_name == "total_sample" else full_train_n
        )
        requested = int(np.ceil(fraction * denominator))
        configured_rule = (
            f"fraction:{fraction:.12g}:denominator={denominator_name}:"
            f"denominator_n={denominator}"
        )
    if strategy == "fixed" and requested <= 0:
        raise ValueError(
            "Fixed TabPFN context requires context rows or a context fraction."
        )
    if strategy == "full":
        requested = full_train_n
        configured_rule = "full_outer_training_partition"
    class_floor = 2
    target = (
        int(min(full_train_n, max(class_floor, requested)))
        if requested > 0 else 0
    )
    return {
        "strategy": strategy,
        "strategy_source": strategy_source,
        "configured_rule": configured_rule,
        "requested_context_n": int(requested),
        "target_context_n": int(target),
        "fraction": fraction,
        "fraction_denominator": denominator_name if fraction is not None else None,
        "full_train_n": full_train_n,
        "total_sample_n": total_sample_n,
    }


def v14_normalize_config(raw_config):
    missing = [
        key for key in ("experiment_name", "iterations", "data")
        if key not in raw_config
    ]
    if "sample_sizes" not in raw_config and "total_n" not in raw_config:
        missing.append("sample_sizes")
    if "scenarios" not in raw_config and "scenario_name" not in raw_config:
        missing.append("scenarios")
    if missing:
        raise ValueError(
            "The resolved JSON experiment definition is missing required "
            f"field(s): {sorted(set(missing))}."
        )
    config = v14_deep_merge(v14_builtin_base_config(), raw_config)
    if raw_config.get("replace_scenarios", False):
        config["scenarios"] = copy.deepcopy(
            raw_config.get("scenarios", {})
        )
    if raw_config.get("replace_models", False):
        config["models"] = copy.deepcopy(raw_config.get("models", {}))
    if "total_n" in raw_config and "sample_sizes" not in raw_config:
        sample = raw_config["total_n"]
        config["sample_sizes"] = [
            "full" if str(sample).lower() in {"all", "full"} else sample
        ]
    if "scenario_name" in raw_config and "scenarios" not in raw_config:
        scenario_name = str(raw_config["scenario_name"])
        scenario = copy.deepcopy(raw_config.get("scenario", {}))
        if "budget_reference_model" in raw_config:
            scenario["budget_reference_model"] = raw_config[
                "budget_reference_model"
            ]
        if "budgeting" in raw_config:
            scenario["budgeting"] = copy.deepcopy(raw_config["budgeting"])
        config["scenarios"] = {scenario_name: scenario}
    if isinstance(config.get("scenarios"), list):
        config["scenarios"] = {
            str(name): {} for name in config["scenarios"]
        }
    if "optuna" in config:
        config["optuna_direction"] = config["optuna"].get(
            "direction", config.get("optuna_direction", "maximize")
        )
        config["default_max_trials"] = int(config["optuna"].get(
            "max_trials", config.get("default_max_trials", 50)
        ))
    energy = config.get("energy", {})
    config["track_energy"] = bool(
        energy.get("enabled", config.get("track_energy", False))
    )
    config["energy_tracking_mode"] = energy.get(
        "tracking_mode", config.get("energy_tracking_mode", "process")
    )
    outputs = config.get("outputs", {})
    config["output_root"] = str(
        outputs.get("root", config.get("output_root", "./V14_Experiments"))
    )
    config["iterations"] = int(config["iterations"])
    config["base_seed"] = int(config.get("base_seed", 2025))
    config["cpu_parallelism"] = v14_deep_merge(
        {
            "policy": "max_available",
            "threads": "auto",
            "prevent_nested_oversubscription": True,
            "allow_oversubscription": False,
            "environment": {},
        },
        config.get("cpu_parallelism", {}),
    )
    config["cpu_monitoring"] = v14_deep_merge(
        {
            "enabled": True,
            "show_console": True,
            "sampling_interval_seconds": 5.0,
            "save_timeseries": True,
        },
        config.get("cpu_monitoring", {}),
    )
    config["dataset_loader"] = v14_deep_merge(
        {
            "module": "dataset_loader",
            "module_path": None,
            "required_functions": [
                "prepare_dataset",
                "_load_dataframe",
            ],
        },
        config.get("dataset_loader", {}),
    )
    config["auxiliary_comparators"] = v14_normalize_auxiliary_comparators(
        config.get("auxiliary_comparators", [])
    )
    config["resolved_cpu_parallelism"] = v14_resolve_cpu_parallelism(config)
    config["_resolved_at"] = v14_utc_now()
    return config


def v14_resolve_requested_device(
    requested, require_requested=False, model_name="model"
):
    requested = str(requested or "auto").lower()
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            f"{model_name} execution device must be 'cpu', 'cuda', or 'auto'."
        )
    available = cuda_available()
    if requested == "cuda":
        if not available and bool(require_requested):
            raise RuntimeError(
                f"{model_name} requested CUDA with require_requested_device=true, "
                "but torch.cuda.is_available() is false."
            )
        return "cuda" if available else "cpu"
    if requested == "auto":
        return "cuda" if available else "cpu"
    return "cpu"


def v14_resolve_scenario(master_config, scenario_name):
    raw = copy.deepcopy(master_config["scenarios"].get(scenario_name, {}))
    budgeting = v14_deep_merge(
        master_config.get("budgeting", {}), raw.get("budgeting", {})
    )
    if "budgeting_enabled" in raw:
        budgeting["enabled"] = bool(raw["budgeting_enabled"])
    reference = raw.get(
        "budget_reference_model",
        master_config.get("budget_reference_model"),
    )
    if not budgeting.get("enabled", True):
        reference = None
    models = copy.deepcopy(master_config["models"])
    if "enabled_models" in raw:
        enabled = set(raw["enabled_models"])
        for model_name in models:
            models[model_name]["enabled"] = model_name in enabled
    for model_name, override in raw.get("model_overrides", {}).items():
        if model_name not in models:
            raise ValueError(
                f"Scenario {scenario_name!r} overrides unknown model {model_name!r}."
            )
        models[model_name] = apply_model_override(models[model_name], override)
    resolved = copy.deepcopy(raw)
    resolved["scenario_name"] = scenario_name
    resolved["semantic_role"] = raw.get(
        "semantic_role",
        master_config.get("plots", {}).get(
            "scenario_roles", {}
        ).get(scenario_name),
    )
    resolved["budgeting"] = budgeting
    resolved["budgeting_enabled"] = bool(budgeting.get("enabled", True))
    resolved["budget_reference_model"] = reference
    resolved["resolved_models"] = models
    resolved["enabled_models"] = [
        name for name, value in models.items() if value.get("enabled", False)
    ]
    return resolved


def v14_resolve_tabpfn_execution(scenario):
    cfg = scenario["resolved_models"].get("TabPFN", {})
    execution = cfg.get("execution", {})
    requested_path = str(
        execution.get("path", execution.get("execution_path", "auto"))
    ).lower()
    requested_path = {
        "client": "cloud",
        "cloud_client": "cloud",
        "cloud/client": "cloud",
        "local_gpu": "local",
        "local_cpu": "local",
    }.get(requested_path, requested_path)
    if requested_path not in {"auto", "local", "cloud"}:
        raise ValueError(
            "TabPFN execution.path must be 'auto', 'local', or 'cloud'."
        )
    if requested_path == "auto":
        reference = scenario.get("budget_reference_model")
        reference_is_tuned = bool(
            scenario["resolved_models"].get(reference, {}).get(
                "tuned_by_optuna", False
            )
        )
        resolved_path = (
            "local"
            if (
                scenario["budgeting"].get("enabled", True)
                and reference_is_tuned
                and scenario["budgeting"].get(
                    "cap_untuned_competitors", True
                )
            )
            else "cloud"
        )
    else:
        resolved_path = requested_path
    requested_device = str(execution.get("local_device", "auto")).lower()
    resolved_device = (
        v14_resolve_requested_device(
            requested_device,
            execution.get("require_requested_device", False),
            "TabPFN",
        )
        if resolved_path == "local"
        else "remote_unknown"
    )
    return {
        "requested_path": requested_path,
        "resolved_path": resolved_path,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "require_requested_device": bool(
            execution.get("require_requested_device", False)
        ),
        "remote_hardware_known": False if resolved_path == "cloud" else None,
    }


def v14_expected_model_device(model_name, model_cfg, scenario):
    if model_name == "TabPFN":
        return v14_resolve_tabpfn_execution(scenario)["resolved_device"]
    if model_name in {"XGBoost", "CatBoost"}:
        execution = model_cfg.get("execution", {})
        return v14_resolve_requested_device(
            execution.get("device", "auto"),
            execution.get("require_requested_device", False),
            model_name,
        )
    return "cpu"


def v14_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def v14_capture_environment():
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "optuna",
        "xgboost",
        "catboost",
        "tabpfn",
        "tabpfn-client",
        "codecarbon",
        "torch",
        "matplotlib",
        "plotly",
        "pyarrow",
    ]
    gpu_names = []
    gpu_count = 0
    pytorch_cuda = None
    cuda_is_available = cuda_available()
    if torch is not None:
        pytorch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_is_available:
            try:
                gpu_count = int(torch.cuda.device_count())
                gpu_names = [
                    torch.cuda.get_device_name(index)
                    for index in range(gpu_count)
                ]
            except Exception:
                gpu_names = []
    nvidia = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            nvidia["query"] = [
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
        else:
            nvidia["unavailable_reason"] = completed.stderr.strip()
    except Exception as exc:
        nvidia["unavailable_reason"] = str(exc)
    memory_gb = None
    cpu = platform.processor() or platform.machine()
    try:
        import psutil
        memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 3)
        cpu = platform.processor() or cpu
    except Exception:
        pass
    cpu_parallelism = (
        v14_resolve_cpu_parallelism(RUN_CONFIG)
        if isinstance(RUN_CONFIG, dict)
        else v14_detect_cpu_capacity()
    )
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "captured_at_utc": v14_utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "cpu_parallelism": cpu_parallelism,
        "thread_environment": {
            variable: os.environ.get(variable)
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "ram_gb": memory_gb,
        "cuda_available": cuda_is_available,
        "gpu_count": gpu_count,
        "gpu_models": gpu_names,
        "cuda_runtime": os.environ.get("CUDA_VERSION"),
        "pytorch_cuda_version": pytorch_cuda,
        "nvidia": nvidia,
        "packages": {
            package: v14_package_version(package) for package in packages
        },
        "determinism_statement": (
            "Seeded execution is requested. Bitwise GPU determinism is not "
            "claimed unless deterministic_torch is enabled and verified."
        ),
    }


def v14_capture_pip_freeze():
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout
        return f"pip freeze unavailable: {completed.stderr}\n"
    except Exception as exc:
        return f"pip freeze unavailable: {exc}\n"


def v14_hash_array(values):
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    if array.dtype.kind in {"O", "U", "S"}:
        digest.update(v14_canonical_json(array.tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def v14_hash_features(X):
    if isinstance(X, pd.DataFrame):
        digest = hashlib.sha256()
        digest.update(v14_canonical_json(list(X.columns)).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(X, index=True).values.tobytes()
        )
        return digest.hexdigest()
    return v14_hash_array(X)


V14_REQUIRED_DATASET_LOADER_FUNCTIONS = (
    "prepare_dataset",
    "_load_dataframe",
)


# The loader remains a separate, checksummed repository dependency.
def v14_validate_dataset_loader_module(
    module, resolved_path=None, required_functions=None
):
    required = list(V14_REQUIRED_DATASET_LOADER_FUNCTIONS)
    for name in required_functions or []:
        if name not in required:
            required.append(str(name))
    missing = [
        name for name in required
        if not callable(getattr(module, name, None))
    ]
    if missing:
        location = resolved_path or getattr(module, "__file__", None)
        raise ImportError(
            "ERROR: The resolved dataset loader is incompatible.\n\n"
            f"Loader: {location or getattr(module, '__name__', 'unknown')}\n"
            f"Missing required callable function(s): {missing}\n\n"
            "V14 requires prepare_dataset() and _load_dataframe()."
        )
    return required


def v14_resolve_dataset_loader(config):
    loader_config = copy.deepcopy(config.get("dataset_loader", {}))
    module_name = str(
        loader_config.get("module", "dataset_loader")
    ).strip() or "dataset_loader"
    configured_path = loader_config.get("module_path")
    required = loader_config.get(
        "required_functions",
        list(V14_REQUIRED_DATASET_LOADER_FUNCTIONS),
    )
    if configured_path:
        resolved_path = Path(configured_path).expanduser()
        if not resolved_path.is_absolute():
            config_source = config.get("_config_source")
            base_dir = (
                Path(config_source).resolve().parent
                if config_source else Path.cwd()
            )
            resolved_path = base_dir / resolved_path
        resolved_path = resolved_path.resolve()
        if not resolved_path.is_file():
            raise ImportError(
                "ERROR: Required companion module dataset_loader.py was not "
                "found.\n\n"
                f"Configured module path: {resolved_path}\n\n"
                "Place the compatible dataset_loader.py beside "
                "V14_REPRODUCIBLE_EXPERIMENT_PIPELINE.py or provide its "
                "configured module path."
            )
        import_name = (
            "_v14_dataset_loader_"
            + hashlib.sha256(
                str(resolved_path).encode("utf-8")
            ).hexdigest()[:12]
        )
        spec = importlib.util.spec_from_file_location(
            import_name, resolved_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not construct an import specification for {resolved_path}."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif module_name == "dataset_loader":
        resolved_path = (
            Path(__file__).resolve().parent / "dataset_loader.py"
        )
        if not resolved_path.is_file():
            raise ImportError(
                "ERROR: Required companion module dataset_loader.py was not "
                "found.\n\n"
                "Place the compatible dataset_loader.py beside\n"
                "V14_REPRODUCIBLE_EXPERIMENT_PIPELINE.py\n"
                "or provide its configured module path."
            )
        spec = importlib.util.spec_from_file_location(
            "_v14_dataset_loader_companion", resolved_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not construct an import specification for {resolved_path}."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ImportError(
                "ERROR: Required configured dataset-loader package could not "
                f"be imported: {module_name!r}."
            ) from exc
        module_file = getattr(module, "__file__", None)
        resolved_path = (
            Path(module_file).resolve() if module_file else None
        )
    required = v14_validate_dataset_loader_module(
        module,
        resolved_path=resolved_path,
        required_functions=required,
    )
    file_path = (
        Path(resolved_path).resolve()
        if resolved_path is not None else None
    )
    provenance = {
        "module": module_name,
        "filename": file_path.name if file_path else None,
        "resolved_path": str(file_path) if file_path else None,
        "sha256": (
            v14_sha256_file(file_path)
            if file_path is not None and file_path.is_file()
            else None
        ),
        "version": (
            getattr(module, "DATASET_LOADER_VERSION", None)
            or getattr(module, "__version__", None)
        ),
        "required_functions": required,
        "required_functions_present": True,
        "import_validated": True,
    }
    return module, provenance


def v14_load_dataset(config):
    loader_module, loader_provenance = v14_resolve_dataset_loader(config)
    prepare_dataset = loader_module.prepare_dataset
    load_dataframe = loader_module._load_dataframe
    data_config = copy.deepcopy(config.get("data", {}))
    X, y, groups, info = prepare_dataset(data_config, verbose=False)
    timestamps = None
    timestamp_column = data_config.get("timestamp_column")
    if timestamp_column:
        raw = load_dataframe(data_config)
        if timestamp_column not in raw.columns:
            raise ValueError(
                f"timestamp_column {timestamp_column!r} is not present."
            )
        if data_config.get("target"):
            keep = raw[data_config["target"]].notna()
        elif data_config.get("target_fn"):
            target_values = pd.Series(data_config["target_fn"](raw))
            keep = target_values.notna()
        else:
            keep = pd.Series(True, index=raw.index)
        timestamps = pd.to_datetime(
            raw.loc[keep, timestamp_column], errors="raise"
        ).reset_index(drop=True).to_numpy()
        if len(timestamps) != len(y):
            raise ValueError(
                "Timestamp extraction did not remain aligned with prepared data."
            )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset_identifier": data_config.get(
            "dataset_identifier",
            data_config.get(
                "source", data_config.get("sklearn_dataset", "in_memory")
            ),
        ),
        **v14_json_safe(info),
        "row_count": int(len(y)),
        "predictor_count": int(X.shape[1]),
        "feature_names": (
            [str(value) for value in X.columns]
            if isinstance(X, pd.DataFrame)
            else None
        ),
        "labels": [0, 1],
        "prevalence": float(np.mean(np.asarray(y) == 1)),
        "group_count": (
            int(len(pd.unique(groups))) if groups is not None else None
        ),
        "input_fingerprints": {
            "X_sha256": v14_hash_features(X),
            "y_sha256": v14_hash_array(y),
            "groups_sha256": (
                v14_hash_array(groups) if groups is not None else None
            ),
            "timestamps_sha256": (
                v14_hash_array(timestamps)
                if timestamps is not None else None
            ),
        },
        "preprocessing_mode": config.get("preprocessing", {}).get(
            "mode", "auto"
        ),
        "dataset_loader": loader_provenance,
        "dataset_loader_filename": loader_provenance["filename"],
        "dataset_loader_sha256": loader_provenance["sha256"],
        "dataset_loader_version": loader_provenance["version"],
        "dataset_loader_resolved_path": loader_provenance[
            "resolved_path"
        ],
        "confidentiality_note": (
            "The manifest stores fingerprints and aggregate metadata, not a "
            "duplicate of the analytical dataset."
        ),
    }
    return X, np.asarray(y, dtype=int), groups, timestamps, manifest


def v14_dependency_preflight(config, scenarios):
    errors = []
    warnings_out = []
    if optuna is None and any(
        model.get("enabled", False) and model.get("tuned_by_optuna", False)
        for scenario in scenarios.values()
        for model in scenario["resolved_models"].values()
    ):
        errors.append("Optuna is required by at least one enabled tuned model.")
    energy = config.get("energy", {})
    if energy.get("enabled", False) and EmissionsTracker is None:
        message = "CodeCarbon is requested but the codecarbon package is unavailable."
        if energy.get("required", False) or not energy.get(
            "allow_unavailable", False
        ):
            errors.append(message)
        else:
            warnings_out.append(message)
            config["track_energy"] = False
    for scenario_name, scenario in scenarios.items():
        enabled = scenario["enabled_models"]
        if not enabled:
            errors.append(f"Scenario {scenario_name!r} has no enabled models.")
        if scenario["budgeting"].get("enabled", True):
            reference = scenario.get("budget_reference_model")
            if reference not in enabled:
                errors.append(
                    f"Scenario {scenario_name!r} budget reference {reference!r} "
                    "is not enabled."
                )
        for model_name in enabled:
            if model_name not in MODEL_RUNNERS:
                errors.append(f"Unknown enabled model: {model_name}")
            if model_name == "XGBoost" and XGBClassifier is None:
                errors.append("XGBoost is enabled but xgboost is unavailable.")
            if model_name == "CatBoost" and CatBoostClassifier is None:
                errors.append("CatBoost is enabled but catboost is unavailable.")
            if model_name == "TabPFN":
                execution = v14_resolve_tabpfn_execution(scenario)
                if (
                    execution["resolved_path"] == "local"
                    and LocalTabPFNClassifier is None
                ):
                    errors.append(
                        f"Scenario {scenario_name!r} requires local TabPFN, "
                        "but tabpfn.TabPFNClassifier is unavailable."
                    )
                if (
                    execution["resolved_path"] == "cloud"
                    and TabPFNClassifier is None
                ):
                    errors.append(
                        f"Scenario {scenario_name!r} requires cloud/client TabPFN, "
                        "but tabpfn_client.TabPFNClassifier is unavailable."
                    )
        comparator_candidates = copy.deepcopy(
            config.get("auxiliary_comparators", [])
        )
        comparator_candidates.extend(
            v14_normalize_auxiliary_comparators(
                scenario.get("auxiliary_comparators", [])
            )
        )
        for comparator in comparator_candidates:
            if not comparator.get("enabled", True):
                continue
            base_model = comparator["base_model"]
            if base_model not in MODEL_RUNNERS:
                errors.append(
                    f"Comparator {comparator['id']!r} names unknown base "
                    f"model {base_model!r}."
                )
                continue
            if base_model == "XGBoost" and XGBClassifier is None:
                errors.append(
                    f"Comparator {comparator['id']!r} requires xgboost."
                )
            if base_model == "CatBoost" and CatBoostClassifier is None:
                errors.append(
                    f"Comparator {comparator['id']!r} requires catboost."
                )
            if base_model == "TabPFN":
                path = str(
                    comparator.get("execution", {}).get("path", "local")
                ).lower()
                if path == "cloud" and TabPFNClassifier is None:
                    errors.append(
                        f"Comparator {comparator['id']!r} requires "
                        "tabpfn-client."
                    )
                if path != "cloud" and LocalTabPFNClassifier is None:
                    errors.append(
                        f"Comparator {comparator['id']!r} requires local "
                        "tabpfn."
                    )
    if errors:
        raise ValueError("Configuration preflight failed:\n- " + "\n- ".join(errors))
    return warnings_out


def v14_validate_resolved_config(config, X, y, groups, timestamps):
    errors = []
    if int(config.get("iterations", 0)) <= 0:
        errors.append("iterations must be a positive integer.")
    sample_sizes = config.get("sample_sizes")
    if not isinstance(sample_sizes, list) or not sample_sizes:
        errors.append("sample_sizes must be a non-empty JSON array.")
    else:
        for sample in sample_sizes:
            if isinstance(sample, str):
                if sample.lower() not in {"full", "all"}:
                    errors.append(
                        f"Unsupported sample size string {sample!r}; use 'full'."
                    )
            else:
                try:
                    size = int(sample)
                except Exception:
                    errors.append(f"Invalid sample size {sample!r}.")
                    continue
                if size <= 0 or size > len(y):
                    errors.append(
                        f"Sample size {size} must be between 1 and {len(y)}."
                    )
    if len(y) != int(X.shape[0]):
        errors.append("X and y are not aligned.")
    if set(np.unique(y).tolist()) != {0, 1}:
        errors.append("The target must contain exactly binary labels 0 and 1.")
    split = config.get("splitting", {})
    strategy = str(split.get("strategy", "auto")).lower()
    if strategy in {"group", "grouped", "stratified_group"} and groups is None:
        errors.append("Grouped splitting was requested but groups are unavailable.")
    if split.get("require_groups", False) and groups is None:
        errors.append("splitting.require_groups=true but groups are unavailable.")
    if strategy == "temporal" and timestamps is None:
        errors.append("Temporal splitting was requested but timestamps are unavailable.")
    if not isinstance(config.get("scenarios"), dict) or not config["scenarios"]:
        errors.append("At least one scenario must be configured.")
    if config.get("execution", {}).get("on_model_error") not in {
        "continue", "raise"
    }:
        errors.append("execution.on_model_error must be 'continue' or 'raise'.")
    if config.get("execution", {}).get("on_iteration_error") not in {
        "continue", "raise"
    }:
        errors.append("execution.on_iteration_error must be 'continue' or 'raise'.")
    if config.get("plots", {}).get(
        "runtime_axis_scale", "linear"
    ) not in {"linear", "log"}:
        errors.append("plots.runtime_axis_scale must be 'linear' or 'log'.")
    try:
        v14_resolve_cpu_parallelism(config)
    except Exception as exc:
        errors.append(f"Invalid cpu_parallelism configuration: {exc}")
    try:
        interval = float(
            config.get("cpu_monitoring", {}).get(
                "sampling_interval_seconds", 5.0
            )
        )
        if interval <= 0:
            raise ValueError("must be positive")
    except Exception as exc:
        errors.append(
            "cpu_monitoring.sampling_interval_seconds is invalid: "
            f"{exc}"
        )
    comparator_ids = [
        comparator["id"]
        for comparator in config.get("auxiliary_comparators", [])
    ]
    if len(comparator_ids) != len(set(comparator_ids)):
        errors.append("Auxiliary comparator IDs must be unique.")
    output_root = Path(config["output_root"]).expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / f".v14_write_probe_{uuid.uuid4().hex}"
        v14_atomic_write_text(probe, "ok")
        probe.unlink()
    except Exception as exc:
        errors.append(f"Output root is not writable: {output_root}: {exc}")
    if errors:
        raise ValueError("Configuration validation failed:\n- " + "\n- ".join(errors))
    return True


def v14_resolve_sample_sizes(config, available_n):
    resolved = []
    for requested in config["sample_sizes"]:
        if isinstance(requested, str) and requested.lower() in {"full", "all"}:
            actual = int(available_n)
            label = "FULL"
            requested_value = "full"
        else:
            actual = int(requested)
            label = str(actual)
            requested_value = int(requested)
        resolved.append({
            "requested": requested_value,
            "actual": actual,
            "folder": f"N_{label}",
        })
    return resolved


# =============================================================================
# 8. Experiment matrix, split evidence, model artifacts, and comparators
# =============================================================================

def v14_make_run_id(config, config_hash_value):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        f"{safe_path_part(config['experiment_name'], 'Experiment')}_"
        f"{timestamp}_{config_hash_value[:6]}"
    )


def v14_find_resume_dir(config, config_hash_value):
    root = Path(config["output_root"]).expanduser().resolve()
    suffix = f"_{config_hash_value[:6]}"
    candidates = [
        path for path in root.glob(f"*{suffix}")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Resume requested but no run directory with the resolved config hash "
            f"{config_hash_value[:6]} exists under {root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def v14_build_experiment_matrix(
    experiment_id, config_hash_value, config, scenarios, samples
):
    rows = []
    for scenario_name, scenario in scenarios.items():
        enabled = scenario["enabled_models"]
        for sample in samples:
            comparators = v14_resolve_auxiliary_comparators(
                config,
                scenario_name,
                scenario,
                sample,
            )
            executions = [
                {
                    "Analysis_Role": "core_model",
                    "Execution_Variant": model_name,
                    "Base_Model": model_name,
                    "Comparator_ID": None,
                    "Paired_With": None,
                    "Expected_Device": v14_expected_model_device(
                        model_name,
                        scenario["resolved_models"][model_name],
                        scenario,
                    ),
                }
                for model_name in enabled
            ]
            for comparator in comparators:
                execution = comparator.get("execution", {})
                expected_device = (
                    (
                        execution.get("local_device", "cpu")
                        if execution.get("path", "local") != "cloud"
                        else "remote_unknown"
                    )
                    if comparator["base_model"] == "TabPFN"
                    else execution.get("device", "cpu")
                )
                executions.append({
                    "Analysis_Role": "auxiliary_comparator",
                    "Execution_Variant": comparator["id"],
                    "Base_Model": comparator["base_model"],
                    "Comparator_ID": comparator["id"],
                    "Paired_With": comparator.get("pair_with"),
                    "Expected_Device": expected_device,
                })
            for iteration in range(1, config["iterations"] + 1):
                for execution in executions:
                    rows.append({
                        "Experiment_ID": experiment_id,
                        "Config_Hash": config_hash_value,
                        "Scenario": scenario_name,
                        "Scenario_Role": scenario.get(
                            "semantic_role"
                        ) or scenario_name,
                        "Sample_Size_Requested": sample["requested"],
                        "Sample_Size_Actual": sample["actual"],
                        "Iteration": iteration,
                        "Seed": config["base_seed"] + iteration,
                        **execution,
                        "CPU_Parallelism_Policy": config[
                            "resolved_cpu_parallelism"
                        ]["CPU_Parallelism_Policy"],
                        "CPU_Count_Available_To_Process": config[
                            "resolved_cpu_parallelism"
                        ]["CPU_Count_Available_To_Process"],
                        "CPU_Threads_Resolved": config[
                            "resolved_cpu_parallelism"
                        ]["CPU_Threads_Resolved"],
                        "Status": "pending",
                    })
    return pd.DataFrame(rows)


def v14_public_audit(value):
    return {
        str(key): v14_json_safe(item)
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def v14_compute_split_evidence(
    X, y, groups, timestamps, config, scenario, sample_size, seed
):
    split_config = config.get("splitting", {})
    sampling_config = scenario.get(
        "sampling", config.get("sampling", {"strategy": "original_prevalence"})
    )
    requested = str(split_config.get("strategy", "auto")).lower()
    requested = {
        "group": "stratified_group",
        "grouped": "stratified_group",
        "group_stratified": "stratified_group",
        "row": "stratified",
    }.get(requested, requested)
    groups_array = None
    if groups is not None:
        groups_array, _ = pd.factorize(np.asarray(groups).ravel(), sort=False)
        groups_array = groups_array.astype(np.int64, copy=False)
    if requested == "auto":
        strategy = (
            "stratified_group"
            if (
                split_config.get("group_aware", True)
                and groups_array is not None
            )
            else "stratified"
        )
    else:
        strategy = requested
    if strategy == "stratified_group":
        (
            X_train,
            X_test,
            y_train,
            y_test,
            outer_info,
            groups_train,
            groups_test,
        ) = grouped_sample_split(
            X,
            y,
            groups_array,
            sample_size,
            config["train_frac"],
            seed,
            sampling_config,
            split_config,
        )
        (
            _,
            _,
            y_inner_train,
            y_validation,
            inner_info,
        ) = grouped_inner_split(
            X_train,
            y_train,
            groups_train,
            val_frac=config["inner_validation_frac"],
            seed=seed,
            split_config=split_config,
            return_info=True,
        )
    elif strategy == "temporal":
        (
            X_train,
            X_test,
            y_train,
            y_test,
            outer_info,
            groups_train,
            groups_test,
        ) = temporal_sample_split(
            X,
            y,
            timestamps,
            sample_size,
            config["train_frac"],
            seed,
            sampling_config,
            split_config,
            groups_array,
        )
        train_timestamps = np.asarray(timestamps)[outer_info["_train_indices"]]
        (
            _,
            _,
            y_inner_train,
            y_validation,
            inner_info,
        ) = temporal_inner_split(
            X_train,
            y_train,
            train_timestamps,
            val_frac=config["inner_validation_frac"],
            seed=seed,
            split_config=split_config,
            return_info=True,
        )
    else:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            outer_info,
        ) = exact_sample_split_from_config(
            X,
            y,
            sample_size,
            config["train_frac"],
            seed,
            sampling_config,
            split_config,
        )
        groups_train = None
        groups_test = None
        (
            _,
            _,
            y_inner_train,
            y_validation,
            inner_info,
        ) = stratified_inner_split(
            X_train,
            y_train,
            val_frac=config["inner_validation_frac"],
            seed=seed,
            split_config=split_config,
            return_info=True,
        )
    train_indices = np.asarray(outer_info["_train_indices"], dtype=np.int64)
    test_indices = np.asarray(outer_info["_test_indices"], dtype=np.int64)
    inner_local = np.asarray(inner_info["_train_indices"], dtype=np.int64)
    validation_local = np.asarray(inner_info["_test_indices"], dtype=np.int64)
    inner_train_indices = train_indices[inner_local]
    validation_indices = train_indices[validation_local]
    return {
        "strategy": strategy,
        "outer_info": outer_info,
        "inner_info": inner_info,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "inner_train_indices": inner_train_indices,
        "validation_indices": validation_indices,
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_inner_train": np.asarray(y_inner_train, dtype=int),
        "y_validation": np.asarray(y_validation, dtype=int),
        "groups_train": groups_train,
        "groups_test": groups_test,
    }


def v14_save_split_evidence(
    split_dir,
    split_evidence,
    raw_groups,
    config,
    scenario_name,
    requested_sample,
    actual_sample,
    iteration,
    seed,
):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario_name,
        "sample_size_requested": requested_sample,
        "sample_size_actual": actual_sample,
        "iteration": iteration,
        "seed": seed,
    }
    outer = {
        **common,
        **v14_public_audit(split_evidence["outer_info"]),
    }
    inner = {
        **common,
        **v14_public_audit(split_evidence["inner_info"]),
    }
    v14_atomic_write_json(split_dir / "outer_split_audit.json", outer)
    v14_atomic_write_json(split_dir / "inner_split_audit.json", inner)
    v14_atomic_write_csv(
        split_dir / "outer_split_audit.csv", pd.DataFrame([outer])
    )
    v14_atomic_write_csv(
        split_dir / "inner_split_audit.csv", pd.DataFrame([inner])
    )
    for name in (
        "train_indices",
        "test_indices",
        "inner_train_indices",
        "validation_indices",
    ):
        v14_atomic_write_npy(split_dir / f"{name}.npy", split_evidence[name])
    if (
        raw_groups is not None
        and config.get("outputs", {}).get("save_group_ids", True)
    ):
        raw_groups = np.asarray(raw_groups)
        v14_atomic_write_json(
            split_dir / "train_group_ids.json",
            sorted({
                str(value)
                for value in raw_groups[split_evidence["train_indices"]]
            }),
        )
        v14_atomic_write_json(
            split_dir / "test_group_ids.json",
            sorted({
                str(value)
                for value in raw_groups[split_evidence["test_indices"]]
            }),
        )
    return {
        path.name: v14_sha256_file(path)
        for path in split_dir.iterdir()
        if path.is_file()
    }


def v14_atomic_engine_config(
    master_config,
    scenario_name,
    scenario,
    sample_size,
    seed,
    engine_dir,
    external_iteration=None,
    execution_variant=None,
    display_scenario_name=None,
):
    atomic = copy.deepcopy(master_config)
    atomic["v14_total_iterations"] = int(master_config["iterations"])
    atomic["total_n"] = int(sample_size)
    atomic["iterations"] = 1
    atomic["v14_external_iteration"] = int(
        external_iteration if external_iteration is not None else 1
    )
    atomic["v14_execution_variant"] = execution_variant
    atomic["v14_display_scenario_name"] = (
        display_scenario_name or scenario_name
    )
    atomic["base_seed"] = int(seed)
    atomic["models"] = copy.deepcopy(scenario["resolved_models"])
    atomic["budgeting"] = copy.deepcopy(scenario["budgeting"])
    atomic["budget_reference_model"] = (
        scenario.get("budget_reference_model") or "None"
    )
    atomic["scenarios"] = {
        scenario_name: {
            "budgeting_enabled": bool(
                scenario["budgeting"].get("enabled", True)
            ),
            "budget_reference_model": scenario.get(
                "budget_reference_model"
            ),
            "sampling": copy.deepcopy(
                scenario.get("sampling", atomic.get("sampling", {}))
            ),
            "enabled_models": list(scenario["enabled_models"]),
            "model_overrides": {},
            "external_runtime_budget_seconds": scenario.get(
                "external_runtime_budget_seconds"
            ),
        }
    }
    atomic["results_dir"] = str(engine_dir)
    atomic["output_layout"] = {
        "results_dir": str(engine_dir),
        "include_output_folder": False,
        "include_sample_size": False,
        "include_file_prefix": False,
        "timestamp_run_dir": False,
    }
    atomic["output_root"] = str(engine_dir)
    atomic["output_folder"] = ""
    atomic["file_prefix"] = "V14_Atomic"
    atomic["plots"] = v14_deep_merge(atomic.get("plots", {}), {
        "save_html": False,
        "save_png": False,
        "save_svg": False,
        "save_jpeg": False,
        "show_figures": False,
    })
    atomic["track_energy"] = bool(
        master_config.get("energy", {}).get("enabled", False)
        and EmissionsTracker is not None
    )
    atomic["energy_tracking_mode"] = master_config.get("energy", {}).get(
        "tracking_mode", "process"
    )
    return atomic


def v14_load_atomic_raw_results(engine_dir, scenario_name):
    scenario_dir = Path(engine_dir) / scenario_folder_name(scenario_name)
    candidates = sorted(scenario_dir.glob("*_RawResults.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"V12 scientific engine did not create RawResults.pkl in {scenario_dir}."
        )
    with open(candidates[-1], "rb") as handle:
        return pickle.load(handle), candidates[-1]


def v14_scalar(value, default=None):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return v14_json_safe(value)


def v14_model_execution_metadata(
    model_name, model_cfg, scenario, environment
):
    if model_name == "TabPFN":
        tabpfn = v14_resolve_tabpfn_execution(scenario)
        return {
            "Execution_Path": tabpfn["resolved_path"],
            "Requested_Device": tabpfn["requested_device"],
            "Resolved_Device": tabpfn["resolved_device"],
            "Execution_Device": tabpfn["resolved_device"],
            "Remote_Hardware_Known": tabpfn["remote_hardware_known"],
            "CUDA_Available": environment["cuda_available"],
            "GPU_Model": environment["gpu_models"],
            "GPU_Count": environment["gpu_count"],
            "CUDA_Version": environment["cuda_runtime"],
            "PyTorch_CUDA_Version": environment["pytorch_cuda_version"],
            "TabPFN_Version": environment["packages"].get("tabpfn"),
            "TabPFN_Client_Version": environment["packages"].get(
                "tabpfn-client"
            ),
        }
    requested = model_cfg.get("execution", {}).get("device", "cpu")
    resolved = v14_expected_model_device(
        model_name, model_cfg, scenario
    )
    return {
        "Execution_Path": "local",
        "Requested_Device": requested,
        "Resolved_Device": resolved,
        "Execution_Device": resolved,
        "Remote_Hardware_Known": None,
        "CUDA_Available": environment["cuda_available"],
        "GPU_Model": environment["gpu_models"],
        "GPU_Count": environment["gpu_count"],
        "CUDA_Version": environment["cuda_runtime"],
        "PyTorch_CUDA_Version": environment["pytorch_cuda_version"],
    }


def v14_energy_record(
    model_name,
    measured_value,
    execution_metadata,
    energy_config,
    started_at,
    ended_at,
):
    value = v14_scalar(measured_value)
    enabled = bool(energy_config.get("enabled", False))
    cloud = (
        model_name == "TabPFN"
        and execution_metadata["Execution_Path"] == "cloud"
    )
    if cloud:
        measured_local = None
        measured_client = value
        scope = "local_client_process_only"
        comparable = False
        reason = (
            "Cloud/client measurement covers only the local process while "
            "communicating with or waiting for the remote service."
        )
    else:
        measured_local = value
        measured_client = None
        scope = "local_model_execution"
        comparable = value is not None
        reason = (
            "Comparable only with records having the same local execution scope, "
            "device class, and measurement availability."
        )
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "Energy_Tracking_Enabled": enabled,
        "Energy_Measurement_Available": value is not None,
        "Measured_Energy_kWh": measured_local,
        "Measured_ClientSide_Energy_kWh": measured_client,
        "Estimated_Energy_kWh": None,
        "Energy_Value_Type": "measured" if value is not None else "unavailable",
        "Energy_Scope": scope,
        "Execution_Path": execution_metadata["Execution_Path"],
        "Execution_Device": execution_metadata["Execution_Device"],
        "Energy_Comparable_Flag": comparable,
        "Energy_Comparability_Reason": reason,
        "Remote_Server_Energy_Measured": False if cloud else None,
        "Remote_Server_Energy_kWh": None,
        "Remote_Server_Energy_Unavailable_Reason": (
            "Remote server energy telemetry unavailable" if cloud else None
        ),
        "Estimated_Remote_Cloud_Energy_kWh": None,
        "tracking_mode": energy_config.get("tracking_mode", "process"),
        "measurement_start_utc": started_at,
        "measurement_end_utc": ended_at,
        "codecarbon_version": v14_package_version("codecarbon"),
        "measurement_limitations": (
            "Client-side energy represents only energy observed on the local "
            "machine while communicating with/waiting for the cloud TabPFN "
            "service. It does not include remote server/GPU computation."
            if cloud else (
                None if value is not None else
                "CodeCarbon was disabled, unavailable, or returned no measurement."
            )
        ),
    }


def v14_recompute_metrics(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability > 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    return {
        "AUROC": float(roc_auc_score(y_true, probability)),
        "Balanced_Accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "Sensitivity": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "Precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "Brier_Score": float(
            brier_score_loss(y_true, probability, pos_label=1)
        ),
        "predicted_class": predicted,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def v14_extract_selected_hyperparameters(
    model_name, model_cfg, selected_row, trials_frame
):
    if selected_row is None or trials_frame.empty:
        return copy.deepcopy(model_cfg.get("default_params", {}))
    trial_number = v14_scalar(selected_row.get("Selected_Trial_Number"))
    if trial_number is None:
        return copy.deepcopy(model_cfg.get("default_params", {}))
    matching = trials_frame[
        pd.to_numeric(trials_frame.get("number"), errors="coerce")
        == float(trial_number)
    ]
    if matching.empty:
        return copy.deepcopy(model_cfg.get("default_params", {}))
    trial = matching.iloc[0]
    selected = {}
    mapping = make_param_mapping(model_name)
    for original_name in model_cfg.get("search_space", {}):
        stored_name = mapping.get(f"params_{original_name}")
        if stored_name in trial.index:
            selected[original_name] = v14_scalar(trial[stored_name])
    return selected or copy.deepcopy(model_cfg.get("default_params", {}))


def v14_write_model_failure(
    model_dir,
    model_name,
    error_message,
    execution_metadata,
    common,
    started_at,
    ended_at,
    identity=None,
    cpu_monitor_record=None,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    identity = copy.deepcopy(identity or {})
    display_model = identity.get("Execution_Variant", model_name)
    identity_fields = {
        "Analysis_Role": identity.get("Analysis_Role", "core_model"),
        "Execution_Variant": display_model,
        "Base_Model": identity.get("Base_Model", model_name),
        "Comparator_ID": identity.get("Comparator_ID"),
        "Paired_With": identity.get("Paired_With"),
    }
    cpu_fields = {
        key: value
        for key, value in (cpu_monitor_record or {}).items()
        if key != "samples"
    }
    error = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **common,
        **identity_fields,
        **cpu_fields,
        "Model": display_model,
        "success": False,
        "error_type": "ModelExecutionError",
        "message": str(error_message),
        **execution_metadata,
    }
    v14_atomic_write_json(model_dir / "error.json", error)
    v14_atomic_write_text(
        model_dir / "traceback.txt",
        "The V12 engine caught this model exception internally:\n"
        + str(error_message)
        + "\n",
    )
    v14_atomic_write_json(
        model_dir / "metrics.json",
        {**error, **{metric: None for metric in V14_PREDICTION_METRICS}},
    )
    v14_atomic_write_json(
        model_dir / "runtime_breakdown.json",
        {
            **error,
            "measurement_start_utc": started_at,
            "measurement_end_utc": ended_at,
        },
    )
    v14_atomic_write_json(
        model_dir / "energy_breakdown.json",
        {
            **error,
            "Measured_Energy_kWh": None,
            "Measured_ClientSide_Energy_kWh": None,
            "Estimated_Energy_kWh": None,
            "Energy_Value_Type": "unavailable",
            "Energy_Scope": (
                "local_client_process_only"
                if (
                    model_name == "TabPFN"
                    and execution_metadata["Execution_Path"] == "cloud"
                )
                else "local_model_execution"
            ),
        },
    )
    v14_atomic_write_json(
        model_dir / "final_model_config.json",
        {
            **error,
            "fixed_parameters": None,
            "selected_hyperparameters": None,
        },
    )
    return error


def v14_save_model_artifacts(
    model_dir,
    model_name,
    model_cfg,
    scenario,
    row,
    raw_results,
    engine_outputs,
    split_evidence,
    environment,
    energy_config,
    common,
    started_at,
    ended_at,
    identity=None,
    cpu_monitor_record=None,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    identity = copy.deepcopy(identity or {})
    display_model = identity.get("Execution_Variant", model_name)
    identity_fields = {
        "Analysis_Role": identity.get("Analysis_Role", "core_model"),
        "Execution_Variant": display_model,
        "Base_Model": identity.get("Base_Model", model_name),
        "Comparator_ID": identity.get("Comparator_ID"),
        "Paired_With": identity.get("Paired_With"),
    }
    cpu_fields = {
        key: value
        for key, value in (cpu_monitor_record or {}).items()
        if key != "samples"
    }
    artifact_common = {**common, **identity_fields}
    execution_metadata = v14_model_execution_metadata(
        model_name, model_cfg, scenario, environment
    )
    if row is None or str(row.get("Error", "")).strip():
        message = (
            "No model result row was emitted."
            if row is None else row.get("Error", "Unknown model error")
        )
        return v14_write_model_failure(
            model_dir,
            model_name,
            message,
            execution_metadata,
            common,
            started_at,
            ended_at,
            identity=identity,
            cpu_monitor_record=cpu_monitor_record,
        )
    probabilities = raw_results.get(f"{model_name}_Proba", [])
    truths = raw_results.get(f"{model_name}_TrueLabels", [])
    if not probabilities or not truths:
        return v14_write_model_failure(
            model_dir,
            model_name,
            "Successful metric row lacks persisted probabilities or labels.",
            execution_metadata,
            common,
            started_at,
            ended_at,
            identity=identity,
            cpu_monitor_record=cpu_monitor_record,
        )
    probability = np.asarray(probabilities[-1], dtype=float)
    y_true = np.asarray(truths[-1], dtype=int)
    if len(probability) != len(split_evidence["test_indices"]):
        raise ValueError(
            f"{model_name} prediction length {len(probability)} does not match "
            f"test membership {len(split_evidence['test_indices'])}."
        )
    if not np.array_equal(y_true, np.asarray(split_evidence["y_test"], dtype=int)):
        raise ValueError(
            f"{display_model} test labels do not exactly match the persisted "
            "outer-test split."
        )
    recomputed = v14_recompute_metrics(y_true, probability)
    probability_0 = 1.0 - probability
    predictions = pd.DataFrame({
        "test_row_index": split_evidence["test_indices"],
        "y_true": y_true,
        "predicted_class": recomputed["predicted_class"],
        "probability_class_0": probability_0,
        "probability_class_1": probability,
    })
    v14_write_table_bundle(predictions, model_dir / "predictions")
    v14_atomic_write_npz(
        model_dir / "predictions.npz",
        test_row_index=split_evidence["test_indices"],
        y_true=y_true,
        predicted_class=recomputed["predicted_class"],
        probability_class_0=probability_0,
        probability_class_1=probability,
    )
    v14_atomic_write_npz(
        model_dir / "roc_curve.npz",
        fpr=recomputed["fpr"],
        tpr=recomputed["tpr"],
        thresholds=recomputed["thresholds"],
    )
    v14_atomic_write_json(
        model_dir / "roc_curve_manifest.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **artifact_common,
            "Model": display_model,
            "derived_from": "predictions.npz",
            "predictions_sha256": v14_sha256_file(
                model_dir / "predictions.npz"
            ),
            "arrays": ["fpr", "tpr", "thresholds"],
        },
    )
    metrics = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **artifact_common,
        **cpu_fields,
        "Model": display_model,
        "success": True,
        **{
            key: recomputed[key] for key in V14_PREDICTION_METRICS
        },
        "engine_reported": {
            "AUROC": v14_scalar(row.get("AUC")),
            "Balanced_Accuracy": v14_scalar(row.get("BalancedAccuracy")),
            "Sensitivity": v14_scalar(row.get("Sensitivity")),
            "Precision": v14_scalar(row.get("Precision")),
            "Brier_Score": v14_scalar(row.get("Brier")),
        },
        "metric_source": "predictions.npz",
        **execution_metadata,
    }
    v14_atomic_write_json(model_dir / "metrics.json", metrics)
    runtime = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **artifact_common,
        **cpu_fields,
        "Model": display_model,
        "Preprocessing_Time_Seconds": v14_scalar(
            row.get("Preprocessing_Time_Seconds")
        ),
        "Actual_Optuna_Tuning_Time_Seconds": v14_scalar(
            row.get("Actual_Optuna_Tuning_Time_Seconds")
        ),
        "Optuna_Tuning_Time_Capped_Seconds": v14_scalar(
            row.get("Optuna_Tuning_Time_Capped_Seconds")
        ),
        "Final_Fit_Predict_Time_Seconds": v14_scalar(
            row.get("Final_Fit_Predict_Time_Seconds")
        ),
        "Final_Model_Preparation_Time_Seconds": v14_scalar(
            row.get("Final_Model_Preparation_Time_Seconds")
        ),
        "Final_Fit_Time_Seconds": v14_scalar(
            row.get(
                "Final_Fit_Time_Seconds",
                row.get("TabPFN_Fit_Time_Seconds"),
            )
        ),
        "Prediction_Time_Seconds": v14_scalar(
            row.get(
                "Prediction_Time_Seconds",
                row.get("TabPFN_PredictProba_Time_Seconds"),
            )
        ),
        "Fit_Plus_Prediction_Time_Seconds": v14_scalar(
            row.get("TabPFN_EndToEnd_FitPlusPredictProba_Seconds")
        ),
        "Actual_Total_Runtime_Seconds": v14_scalar(
            row.get("Actual_Total_Runtime_Seconds")
        ),
        "Budget_Accounted_Runtime_Seconds": v14_scalar(
            row.get("Budgeted_Total_Runtime_Seconds")
        ),
        "Reference_Budget_Seconds": v14_scalar(
            row.get(
                "Reference_Budget_Seconds",
                row.get("TabPFN_Time_Budget_Seconds"),
            )
        ),
        "Budget_Basis": v14_scalar(row.get("Budget_Basis")),
        "Reference_Budget_Source_Field": v14_scalar(
            row.get("Reference_Budget_Source_Field")
        ),
        "Wall_Clock_Definition": v14_scalar(
            row.get("Wall_Clock_Definition")
        ),
        "HPO_Start_UTC": v14_scalar(row.get("HPO_Start_UTC")),
        "HPO_End_UTC": v14_scalar(row.get("HPO_End_UTC")),
        "HPO_Start_Perf_Counter_Seconds": v14_scalar(
            row.get("HPO_Start_Perf_Counter_Seconds")
        ),
        "HPO_End_Perf_Counter_Seconds": v14_scalar(
            row.get("HPO_End_Perf_Counter_Seconds")
        ),
        "HPO_Timer": v14_scalar(row.get("HPO_Timer")),
        "HPO_Timing_Boundary": v14_scalar(
            row.get("HPO_Timing_Boundary")
        ),
        "Final_Fit_Predict_Start_UTC": v14_scalar(
            row.get("Final_Fit_Predict_Start_UTC")
        ),
        "Final_Fit_Predict_End_UTC": v14_scalar(
            row.get("Final_Fit_Predict_End_UTC")
        ),
        "Final_Fit_Start_UTC": v14_scalar(
            row.get("Final_Fit_Start_UTC")
        ),
        "Final_Fit_End_UTC": v14_scalar(
            row.get("Final_Fit_End_UTC")
        ),
        "Prediction_Start_UTC": v14_scalar(
            row.get("Prediction_Start_UTC")
        ),
        "Prediction_End_UTC": v14_scalar(
            row.get("Prediction_End_UTC")
        ),
        "Final_Fit_Predict_Timer": v14_scalar(
            row.get("Final_Fit_Predict_Timer")
        ),
        "Effective_Budget_Seconds": v14_scalar(
            row.get("TabPFN_Effective_Time_Budget_Seconds")
        ),
        "Budget_Overrun_Seconds": v14_scalar(
            row.get("TabPFN_Strict_EndToEnd_Overrun_Seconds")
        ),
        "TabPFN_Context_Search_Runtime_Seconds": v14_scalar(
            row.get("TabPFN_Context_Search_Total_Runtime_Seconds")
        ),
        "Budgeting_Enabled": bool(
            scenario["budgeting"].get("enabled", True)
        ),
        "Budget_Reference_Model": scenario.get(
            "budget_reference_model"
        ),
        **execution_metadata,
    }
    v14_atomic_write_json(model_dir / "runtime_breakdown.json", runtime)
    energy_value = v14_scalar(row.get("Energy_kWh"))
    energy = v14_energy_record(
        model_name,
        energy_value,
        execution_metadata,
        energy_config,
        started_at,
        ended_at,
    )
    energy.update(artifact_common)
    energy.update(cpu_fields)
    energy["Model"] = display_model
    v14_atomic_write_json(model_dir / "energy_breakdown.json", energy)
    trials = engine_outputs.get("all_optuna_trials_df", pd.DataFrame()).copy()
    if not trials.empty and "Model" in trials.columns:
        trials = trials[trials["Model"] == model_name].copy()
    if not trials.empty:
        trials["Iteration"] = common["Iteration"]
        trials["Analysis_Role"] = identity_fields["Analysis_Role"]
        trials["Execution_Variant"] = display_model
        selected_number = v14_scalar(row.get("Selected_Trial_Number"))
        trials["selected"] = (
            pd.to_numeric(trials.get("number"), errors="coerce")
            == float(selected_number)
            if selected_number is not None else False
        )
        trials["budget_eligible"] = trials.get(
            "Finished_Within_TabPFN_Budget", True
        )
    v14_atomic_write_csv(model_dir / "optuna_trials.csv", trials)
    v14_atomic_write_json(
        model_dir / "optuna_trials.json",
        trials.to_dict(orient="records"),
    )
    selected_trial = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **artifact_common,
        "Model": display_model,
        "selected_trial_number": v14_scalar(
            row.get("Selected_Trial_Number")
        ),
        "selection_method": v14_scalar(row.get("Selection_Method")),
        "validation_AUROC": v14_scalar(row.get("Selected_AUC")),
        "budget_eligible_trials": v14_scalar(
            row.get("Eligible_Trials_Within_Budget")
        ),
        "completed_trials": v14_scalar(
            row.get("Optuna_Completed_Trials")
        ),
        "configured_max_trials": model_cfg.get(
            "max_trials", RUN_CONFIG.get("default_max_trials", 50)
        ),
        "fallback_default": (
            v14_scalar(row.get("Selected_Trial_Number")) is None
        ),
    }
    v14_atomic_write_json(model_dir / "selected_trial.json", selected_trial)
    selected_hyperparameters = v14_extract_selected_hyperparameters(
        model_name, model_cfg, row, trials
    )
    final_config = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **artifact_common,
        **cpu_fields,
        "Model": display_model,
        "fixed_parameters": (
            v14_apply_cpu_model_params(
                model_name, model_cfg.get("fixed_params", {})
            )
            if execution_metadata["Resolved_Device"] == "cpu"
            else copy.deepcopy(model_cfg.get("fixed_params", {}))
        ),
        "selected_hyperparameters": selected_hyperparameters,
        "final_estimator_count": (
            v14_scalar(row.get("TabPFN_Local_N_Estimators"))
            or selected_hyperparameters.get("n_estimators")
            or selected_hyperparameters.get("iterations")
        ),
        "seed": common["Seed"],
        "requested_device": execution_metadata["Requested_Device"],
        "resolved_device": execution_metadata["Resolved_Device"],
        "model_version": (
            v14_scalar(row.get("TabPFN_Local_Model_Version"))
            if model_name == "TabPFN" else None
        ),
        "package_version": environment["packages"].get({
            "RandomForest": "scikit-learn",
            "L-SLR": "scikit-learn",
            "Augmented_SLR": "scikit-learn",
            "XGBoost": "xgboost",
            "CatBoost": "catboost",
            "TabPFN": "tabpfn",
        }.get(model_name, "")),
        "preprocessing_mode": v14_scalar(
            row.get("Preprocessing_Mode")
        ),
        "execution_path": execution_metadata["Execution_Path"],
        "fit_mode": v14_scalar(row.get("TabPFN_Local_Fit_Mode")),
        "save_fitted_models": False,
        "fitted_model_binary": None,
    }
    v14_atomic_write_json(
        model_dir / "final_model_config.json", final_config
    )
    if model_name == "TabPFN":
        attempts = row.get("TabPFN_Context_Attempt_Log", [])
        if not isinstance(attempts, list):
            attempts = []
        context_n = v14_scalar(row.get("TabPFN_Context_N_Used"))
        context_indices = None
        if context_n is not None:
            if int(context_n) >= len(split_evidence["y_train"]):
                relative = np.arange(
                    len(split_evidence["y_train"]), dtype=np.int64
                )
            else:
                dummy = np.zeros(
                    (len(split_evidence["y_train"]), 1), dtype=np.float32
                )
                _, _, relative = _stratified_context_subset(
                    dummy,
                    split_evidence["y_train"],
                    int(context_n),
                    seed=common["Seed"] + int(context_n),
                )
            context_indices = split_evidence["train_indices"][
                np.asarray(relative, dtype=np.int64)
            ]
            v14_atomic_write_npy(
                model_dir / "tabpfn_context_indices.npy",
                context_indices,
            )
        context_search = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **artifact_common,
            "Model": display_model,
            "full_training_n": v14_scalar(
                row.get("TabPFN_Full_Train_N")
            ),
            "minimum_context_requested": v14_scalar(
                row.get("TabPFN_Min_Context_Requested")
            ),
            "minimum_context_target": v14_scalar(
                row.get("TabPFN_Min_Context_Target")
            ),
            "actual_context_n": context_n,
            "context_strategy": v14_scalar(
                row.get("TabPFN_Context_Strategy")
            ),
            "context_strategy_source": v14_scalar(
                row.get("TabPFN_Context_Strategy_Source")
            ),
            "context_configured_rule": v14_scalar(
                row.get("TabPFN_Context_Configured_Rule")
            ),
            "context_fraction": v14_scalar(
                row.get("TabPFN_Context_Fraction_Used")
            ),
            "context_fraction_of_outer_train": v14_scalar(
                row.get("TabPFN_Context_Fraction_Of_Outer_Train")
            ),
            "context_fraction_of_total_sampled_records": v14_scalar(
                row.get(
                    "TabPFN_Context_Fraction_Of_Total_Sampled_Records"
                )
            ),
            "candidate_attempts": attempts,
            "candidate_count": v14_scalar(
                row.get("TabPFN_Context_Search_Attempts")
            ),
            "cumulative_search_runtime_seconds": v14_scalar(
                row.get("TabPFN_Context_Search_Total_Runtime_Seconds")
            ),
            "reference_budget_seconds": v14_scalar(
                row.get("TabPFN_ML_Reference_Budget_Seconds")
            ),
            "effective_budget_seconds": v14_scalar(
                row.get("TabPFN_Effective_Time_Budget_Seconds")
            ),
            "effective_budget_multiplier": v14_scalar(
                row.get("TabPFN_Effective_Budget_Multiplier")
            ),
            "budget_expansion_occurred": v14_scalar(
                row.get(
                    "TabPFN_Min_Context_Budget_Multiplier_Applied"
                )
            ),
            "full_test_n": v14_scalar(
                row.get("TabPFN_Full_Prediction_N")
            ),
            "selection_mode": v14_scalar(
                row.get("TabPFN_Context_Selection_Rule")
            ),
            "requested_device": execution_metadata["Requested_Device"],
            "resolved_device": execution_metadata["Resolved_Device"],
            "selected_context_indices_file": (
                "tabpfn_context_indices.npy"
                if context_indices is not None else None
            ),
        }
        v14_atomic_write_json(
            model_dir / "tabpfn_context_search.json", context_search
        )
    if cpu_monitor_record is not None:
        cpu_summary = {
            key: value
            for key, value in cpu_monitor_record.items()
            if key != "samples"
        }
        v14_atomic_write_json(
            model_dir / "cpu_utilization_summary.json", cpu_summary
        )
        samples = cpu_monitor_record.get("samples", [])
        if samples:
            v14_atomic_write_csv(
                model_dir / "cpu_utilization_timeseries.csv",
                pd.DataFrame(samples),
            )
    required = [
        model_dir / name for name in V14_REQUIRED_MODEL_FILES
    ] + [model_dir / "predictions.npz", model_dir / "roc_curve.npz"]
    checksums = {
        path.name: v14_sha256_file(path)
        for path in required if path.exists()
    }
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **artifact_common,
        **cpu_fields,
        "Model": display_model,
        "success": len(checksums) == len(required),
        "required_file_checksums": checksums,
        "completed_at_utc": ended_at,
    }
    v14_atomic_write_json(model_dir / "_SUCCESS.json", marker)
    return {
        **metrics,
        **{
            key: value
            for key, value in runtime.items()
            if key.endswith("_Seconds")
        },
        **{
            key: value
            for key, value in energy.items()
            if key in {
                "Measured_Energy_kWh",
                "Measured_ClientSide_Energy_kWh",
                "Estimated_Energy_kWh",
                "Energy_Value_Type",
                "Energy_Scope",
                "Energy_Comparable_Flag",
                "Energy_Comparability_Reason",
            }
        },
        "TabPFN_Context_N": v14_scalar(
            row.get("TabPFN_Context_N_Used")
        ),
        "TabPFN_Context_Fraction": v14_scalar(
            row.get("TabPFN_Context_Fraction_Used")
        ),
        "Context_Strategy": v14_scalar(
            row.get("TabPFN_Context_Strategy")
        ),
        "Full_Train_N": v14_scalar(row.get("TabPFN_Full_Train_N")),
        "Context_N_Used": v14_scalar(
            row.get("TabPFN_Context_N_Used")
        ),
        "Context_Fraction_Of_Outer_Train": v14_scalar(
            row.get("TabPFN_Context_Fraction_Of_Outer_Train")
        ),
        "Context_Fraction_Of_Total_Sampled_Records": v14_scalar(
            row.get(
                "TabPFN_Context_Fraction_Of_Total_Sampled_Records"
            )
        ),
        "TabPFN_Reference_Budget_Seconds": v14_scalar(
            row.get("TabPFN_ML_Reference_Budget_Seconds")
        ),
        "TabPFN_Effective_Budget_Seconds": v14_scalar(
            row.get("TabPFN_Effective_Time_Budget_Seconds")
        ),
        "success": marker["success"],
    }


# Auxiliary comparators use the same test evidence as their paired core model
# but never count toward core-model completeness.
def v14_comparator_applies(comparator, scenario_name, scenario, sample_value):
    if not comparator.get("enabled", True):
        return False
    scenario_values = comparator.get("scenario_applicability")
    if scenario_values is not None:
        scenario_values = (
            [scenario_values]
            if isinstance(scenario_values, str) else list(scenario_values)
        )
        semantic_role = scenario.get(
            "semantic_role", scenario.get("analysis_role")
        )
        allowed = {str(value) for value in scenario_values}
        if (
            str(scenario_name) not in allowed
            and str(semantic_role) not in allowed
        ):
            return False
    sample_values = comparator.get("sample_size_applicability")
    if sample_values is not None:
        sample_actual = int(
            sample_value["actual"]
            if isinstance(sample_value, dict) else sample_value
        )
        sample_requested = (
            sample_value.get("requested")
            if isinstance(sample_value, dict) else sample_actual
        )
        sample_values = (
            [sample_values]
            if not isinstance(sample_values, list) else sample_values
        )
        normalized = {
            int(value)
            for value in sample_values
            if str(value).lower() not in {"full", "all"}
        }
        includes_full = any(
            str(value).lower() in {"full", "all"} for value in sample_values
        )
        requested_is_full = str(sample_requested).lower() in {
            "full", "all"
        }
        if (
            sample_actual not in normalized
            and not (includes_full and requested_is_full)
        ):
            return False
    return True


def v14_resolve_auxiliary_comparators(
    master_config, scenario_name, scenario, sample_value
):
    combined = copy.deepcopy(master_config.get("auxiliary_comparators", []))
    scenario_specific = scenario.get("auxiliary_comparators", [])
    if scenario_specific:
        combined.extend(v14_normalize_auxiliary_comparators(scenario_specific))
    result = []
    for comparator in combined:
        if not v14_comparator_applies(
            comparator, scenario_name, scenario, sample_value
        ):
            continue
        base_model = comparator["base_model"]
        if base_model not in MODEL_RUNNERS:
            raise ValueError(
                f"Comparator {comparator['id']!r} names unknown base model "
                f"{base_model!r}."
            )
        pair_with = comparator.get("pair_with")
        if pair_with not in scenario["enabled_models"]:
            raise ValueError(
                f"Comparator {comparator['id']!r} pair_with={pair_with!r} "
                "must name an enabled core model in the same scenario."
            )
        result.append(copy.deepcopy(comparator))
    return result


def v14_build_comparator_scenario(
    master_config,
    core_scenario,
    comparator,
    paired_record,
):
    base_model = comparator["base_model"]
    model_cfg = copy.deepcopy(core_scenario["resolved_models"][base_model])
    model_cfg = v14_deep_merge(model_cfg, comparator.get("overrides", {}))
    execution = copy.deepcopy(comparator.get("execution", {}))
    if base_model == "TabPFN":
        model_cfg["execution"] = v14_deep_merge(
            model_cfg.get("execution", {}),
            {
                "path": execution.get("path", "local"),
                "local_device": execution.get("local_device", "cpu"),
                "require_requested_device": execution.get(
                    "require_requested_device", False
                ),
            },
        )
        local_cfg = copy.deepcopy(
            model_cfg.get("local_tabpfn_budget", {})
        )
        local_cfg["context_strategy"] = copy.deepcopy(
            comparator.get("context", {"strategy": "full"})
        )
        model_cfg["local_tabpfn_budget"] = local_cfg
    elif "device" in execution:
        model_cfg["execution"] = v14_deep_merge(
            model_cfg.get("execution", {}),
            {
                "device": execution["device"],
                "require_requested_device": execution.get(
                    "require_requested_device", False
                ),
            },
        )
    resolved_models = copy.deepcopy(core_scenario["resolved_models"])
    for current_name in resolved_models:
        resolved_models[current_name]["enabled"] = current_name == base_model
    resolved_models[base_model] = model_cfg
    resolved_models[base_model]["enabled"] = True
    apply_budget = bool(
        comparator.get("budgeting", {}).get(
            "apply_runtime_budget", False
        )
    )
    external_budget = None
    if apply_budget:
        candidates = [
            paired_record.get("Reference_Budget_Seconds"),
            paired_record.get("TabPFN_Reference_Budget_Seconds"),
            paired_record.get("Actual_Total_Runtime_Seconds"),
        ]
        external_budget = next(
            (
                float(value) for value in candidates
                if value is not None
                and np.isfinite(float(value))
                and float(value) > 0
            ),
            None,
        )
        if external_budget is None:
            raise ValueError(
                f"Comparator {comparator['id']!r} requested a runtime budget, "
                "but its paired core record has no finite positive budget/runtime."
            )
    budgeting = copy.deepcopy(core_scenario.get("budgeting", {}))
    budgeting["enabled"] = apply_budget
    return {
        "scenario_name": core_scenario["scenario_name"],
        "semantic_role": core_scenario.get("semantic_role"),
        "sampling": copy.deepcopy(
            core_scenario.get(
                "sampling", master_config.get("sampling", {})
            )
        ),
        "budgeting": budgeting,
        "budgeting_enabled": apply_budget,
        "budget_reference_model": (
            comparator.get("pair_with") if apply_budget else None
        ),
        "external_runtime_budget_seconds": external_budget,
        "resolved_models": resolved_models,
        "enabled_models": [base_model],
    }


def v14_model_completion_valid(model_dir):
    model_dir = Path(model_dir)
    marker_path = model_dir / "_SUCCESS.json"
    if not marker_path.exists():
        return False
    marker = v14_load_json_file(marker_path)
    if not marker.get("success", False):
        return False
    for filename, expected in marker.get(
        "required_file_checksums", {}
    ).items():
        path = model_dir / filename
        if not path.exists() or v14_sha256_file(path) != expected:
            return False
    return True


def v14_run_auxiliary_comparator(
    X,
    y,
    groups,
    timestamps,
    master_config,
    scenario_name,
    core_scenario,
    sample,
    iteration,
    seed,
    comparator,
    paired_record,
    split_evidence,
    comparator_root,
    experiment_id,
    config_hash_value,
    environment,
    common,
    force_rerun=False,
):
    comparator_id = comparator["id"]
    comparator_dir = Path(comparator_root) / safe_path_part(
        comparator_id, "Comparator"
    )
    if (
        not force_rerun
        and v14_model_completion_valid(comparator_dir)
    ):
        metrics = v14_load_json_file(comparator_dir / "metrics.json")
        runtime = v14_load_json_file(
            comparator_dir / "runtime_breakdown.json"
        )
        energy = v14_load_json_file(
            comparator_dir / "energy_breakdown.json"
        )
        return {
            **metrics,
            **{
                key: value
                for key, value in runtime.items()
                if key.endswith("_Seconds")
            },
            **{
                key: energy.get(key)
                for key in (
                    "Measured_Energy_kWh",
                    "Measured_ClientSide_Energy_kWh",
                    "Estimated_Energy_kWh",
                    "Energy_Value_Type",
                    "Energy_Scope",
                    "Energy_Comparable_Flag",
                    "Energy_Comparability_Reason",
                )
            },
            "success": True,
        }
    comparator_dir.mkdir(parents=True, exist_ok=True)
    v14_atomic_write_json(
        comparator_dir / "comparator_definition.json", comparator
    )
    comparator_scenario = v14_build_comparator_scenario(
        master_config, core_scenario, comparator, paired_record
    )
    internal_scenario_name = (
        f"{scenario_name}__auxiliary__{comparator_id}"
    )
    engine_dir = comparator_dir / "_Engine"
    comparator_config = v14_atomic_engine_config(
        master_config,
        internal_scenario_name,
        comparator_scenario,
        sample["actual"],
        seed,
        engine_dir,
        external_iteration=iteration,
        execution_variant=comparator_id,
        display_scenario_name=scenario_name,
    )
    started_at = v14_utc_now()
    outputs = run_monte_carlo(
        X,
        y,
        comparator_config,
        groups=groups,
        timestamps=timestamps,
    )
    raw_results, _ = v14_load_atomic_raw_results(
        engine_dir, internal_scenario_name
    )
    frame = outputs["all_iteration_metrics_df"].copy()
    base_model = comparator["base_model"]
    matches = (
        frame[frame["Model"] == base_model]
        if not frame.empty and "Model" in frame.columns
        else pd.DataFrame()
    )
    row = matches.iloc[-1] if not matches.empty else None
    if row is not None:
        actual_outer = str(row.get("Split_Fingerprint", ""))
        expected_outer = str(
            split_evidence["outer_info"]["Split_Fingerprint"]
        )
        if actual_outer and actual_outer != expected_outer:
            raise RuntimeError(
                f"Comparator {comparator_id!r} used split {actual_outer}, "
                f"not paired split {expected_outer}."
            )
    ended_at = v14_utc_now()
    record = v14_save_model_artifacts(
        comparator_dir,
        base_model,
        comparator_scenario["resolved_models"][base_model],
        comparator_scenario,
        row,
        raw_results,
        outputs,
        split_evidence,
        environment,
        master_config.get("energy", {}),
        common,
        started_at,
        ended_at,
        identity={
            "Analysis_Role": "auxiliary_comparator",
            "Execution_Variant": comparator_id,
            "Base_Model": base_model,
            "Comparator_ID": comparator_id,
            "Paired_With": comparator.get("pair_with"),
        },
        cpu_monitor_record=v14_latest_cpu_monitor_record(
            scenario_name,
            sample["actual"],
            iteration,
            comparator_id,
        ),
    )
    record["Comparator_Apply_Runtime_Budget"] = bool(
        comparator.get("budgeting", {}).get(
            "apply_runtime_budget", False
        )
    )
    record["Comparator_Config_Hash"] = v14_config_hash(comparator)
    return record


def v14_build_paired_comparison(model_records):
    if not model_records:
        return pd.DataFrame()
    frame = pd.DataFrame(model_records)
    if "Analysis_Role" not in frame.columns:
        return pd.DataFrame()
    core = frame[frame["Analysis_Role"] == "core_model"]
    comparators = frame[
        frame["Analysis_Role"] == "auxiliary_comparator"
    ]
    rows = []
    metrics = [
        "AUROC",
        "Balanced_Accuracy",
        "Sensitivity",
        "Precision",
        "Brier_Score",
    ]
    for _, comparator in comparators.iterrows():
        paired_with = comparator.get("Paired_With")
        primary_match = core[
            core["Execution_Variant"] == paired_with
        ]
        if primary_match.empty:
            continue
        primary = primary_match.iloc[-1]
        exact_pair = all(
            str(primary.get(field)) == str(comparator.get(field))
            for field in (
                "Scenario",
                "Sample_Size_Actual",
                "Iteration",
                "Seed",
                "Split_Fingerprint",
                "Test_Indices_SHA256",
                "Test_Labels_SHA256",
            )
        )
        row = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "Experiment_ID": primary.get("Experiment_ID"),
            "Config_Hash": primary.get("Config_Hash"),
            "Scenario": primary.get("Scenario"),
            "Sample_Size_Requested": primary.get(
                "Sample_Size_Requested"
            ),
            "Sample_Size_Actual": primary.get("Sample_Size_Actual"),
            "Iteration": primary.get("Iteration"),
            "Seed": primary.get("Seed"),
            "Primary_Model": primary.get("Execution_Variant"),
            "Comparator_ID": comparator.get("Comparator_ID"),
            "Comparator_Base_Model": comparator.get("Base_Model"),
            "Split_Fingerprint": primary.get("Split_Fingerprint"),
            "Test_Indices_SHA256": primary.get("Test_Indices_SHA256"),
            "Test_Labels_SHA256": primary.get("Test_Labels_SHA256"),
            "Exact_Pair_Valid": bool(exact_pair),
            "Primary_Runtime_Seconds": primary.get(
                "Actual_Total_Runtime_Seconds"
            ),
            "Comparator_Runtime_Seconds": comparator.get(
                "Actual_Total_Runtime_Seconds"
            ),
            "Primary_Reference_Budget_Seconds": primary.get(
                "Reference_Budget_Seconds"
            ),
            "Primary_Effective_Budget_Seconds": primary.get(
                "Effective_Budget_Seconds"
            ),
            "Primary_Context_N": primary.get("Context_N_Used"),
            "Comparator_Context_N": comparator.get("Context_N_Used"),
            "Primary_Context_Strategy": primary.get("Context_Strategy"),
            "Comparator_Context_Strategy": comparator.get(
                "Context_Strategy"
            ),
        }
        for metric in metrics:
            primary_value = v14_scalar(primary.get(metric))
            comparator_value = v14_scalar(comparator.get(metric))
            row[f"Primary_{metric}"] = primary_value
            row[f"Comparator_{metric}"] = comparator_value
            row[f"Paired_Difference_{metric}_Comparator_Minus_Primary"] = (
                float(comparator_value - primary_value)
                if primary_value is not None and comparator_value is not None
                else None
            )
            row[f"Paired_Difference_{metric}_Primary_Minus_Comparator"] = (
                float(primary_value - comparator_value)
                if primary_value is not None and comparator_value is not None
                else None
            )
        primary_runtime = v14_scalar(
            row["Primary_Runtime_Seconds"]
        )
        comparator_runtime = v14_scalar(
            row["Comparator_Runtime_Seconds"]
        )
        row["Paired_Runtime_Difference_Comparator_Minus_Primary"] = (
            float(comparator_runtime - primary_runtime)
            if primary_runtime is not None and comparator_runtime is not None
            else None
        )
        row["Paired_Difference_Runtime_Comparator_Minus_Primary"] = (
            row["Paired_Runtime_Difference_Comparator_Minus_Primary"]
        )
        row["Paired_Difference_Runtime_Primary_Minus_Comparator"] = (
            float(primary_runtime - comparator_runtime)
            if primary_runtime is not None and comparator_runtime is not None
            else None
        )
        row["Comparator_Runtime_Divided_By_Primary_Runtime"] = (
            float(comparator_runtime / primary_runtime)
            if (
                primary_runtime is not None
                and comparator_runtime is not None
                and primary_runtime > 0
            )
            else None
        )
        row["Primary_Runtime_Divided_By_Comparator_Runtime"] = (
            float(primary_runtime / comparator_runtime)
            if (
                primary_runtime is not None
                and comparator_runtime is not None
                and comparator_runtime > 0
            )
            else None
        )
        row["Runtime_Ratio_Comparator_Over_Primary"] = row[
            "Comparator_Runtime_Divided_By_Primary_Runtime"
        ]
        rows.append(row)
    return pd.DataFrame(rows)


def v14_load_saved_split_evidence(iteration_dir, X, y, groups):
    split_dir = Path(iteration_dir) / "Splits"
    train_indices = np.load(
        split_dir / "train_indices.npy", allow_pickle=False
    ).astype(np.int64)
    test_indices = np.load(
        split_dir / "test_indices.npy", allow_pickle=False
    ).astype(np.int64)
    inner_train_indices = np.load(
        split_dir / "inner_train_indices.npy", allow_pickle=False
    ).astype(np.int64)
    validation_indices = np.load(
        split_dir / "validation_indices.npy", allow_pickle=False
    ).astype(np.int64)
    y_array = np.asarray(y, dtype=int)
    groups_array = None if groups is None else np.asarray(groups)
    return {
        "strategy": v14_load_json_file(
            split_dir / "outer_split_audit.json"
        ).get("Split_Strategy"),
        "outer_info": v14_load_json_file(
            split_dir / "outer_split_audit.json"
        ),
        "inner_info": v14_load_json_file(
            split_dir / "inner_split_audit.json"
        ),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "inner_train_indices": inner_train_indices,
        "validation_indices": validation_indices,
        "y_train": y_array[train_indices],
        "y_test": y_array[test_indices],
        "y_inner_train": y_array[inner_train_indices],
        "y_validation": y_array[validation_indices],
        "groups_train": (
            groups_array[train_indices]
            if groups_array is not None else None
        ),
        "groups_test": (
            groups_array[test_indices]
            if groups_array is not None else None
        ),
    }


def v14_core_iteration_artifacts_valid(
    iteration_dir, enabled_models
):
    iteration_dir = Path(iteration_dir)
    required_split = [
        iteration_dir / "Splits" / name
        for name in (
            "outer_split_audit.json",
            "inner_split_audit.json",
            "train_indices.npy",
            "test_indices.npy",
            "inner_train_indices.npy",
            "validation_indices.npy",
        )
    ]
    if not all(path.exists() for path in required_split):
        return False
    if not (
        iteration_dir / "iteration_model_results.csv"
    ).exists():
        return False
    return all(
        v14_model_completion_valid(
            iteration_dir / "Models" / safe_path_part(model_name, "Model")
        )
        for model_name in enabled_models
    )


def v14_resume_auxiliary_only(
    X,
    y,
    groups,
    timestamps,
    master_config,
    scenario_name,
    scenario,
    sample,
    iteration,
    iteration_dir,
    experiment_id,
    config_hash_value,
    environment,
    logger,
):
    iteration_dir = Path(iteration_dir)
    seed = int(master_config["base_seed"]) + int(iteration)
    split_evidence = v14_load_saved_split_evidence(
        iteration_dir, X, y, groups
    )
    common = {
        "Experiment_ID": experiment_id,
        "Pipeline_Version": PIPELINE_VERSION,
        "Config_Hash": config_hash_value,
        "Scenario": scenario_name,
        "Scenario_Role": scenario.get(
            "semantic_role"
        ) or scenario_name,
        "Sample_Size_Requested": sample["requested"],
        "Sample_Size_Actual": sample["actual"],
        "Iteration": iteration,
        "Seed": seed,
        "Split_Fingerprint": split_evidence["outer_info"].get(
            "Split_Fingerprint"
        ),
        "Inner_Split_Fingerprint": split_evidence["inner_info"].get(
            "Split_Fingerprint"
        ),
        "Test_Indices_SHA256": v14_hash_array(
            split_evidence["test_indices"]
        ),
        "Test_Labels_SHA256": v14_hash_array(split_evidence["y_test"]),
    }
    existing = pd.read_csv(
        iteration_dir / "iteration_model_results.csv"
    )
    if "Analysis_Role" not in existing.columns:
        existing["Analysis_Role"] = "core_model"
    core_records = existing[
        existing["Analysis_Role"] == "core_model"
    ].to_dict(orient="records")
    model_records = list(core_records)
    model_statuses = {
        str(record["Execution_Variant"]): bool(
            str(record.get("success", "")).lower()
            in {"true", "1", "yes"}
        )
        for record in core_records
    }
    comparators = v14_resolve_auxiliary_comparators(
        master_config,
        scenario_name,
        scenario,
        sample,
    )
    comparator_statuses = {}
    for comparator in comparators:
        paired_record = next(
            record for record in core_records
            if record.get("Execution_Variant") == comparator.get("pair_with")
        )
        record = v14_run_auxiliary_comparator(
            X,
            y,
            groups,
            timestamps,
            master_config,
            scenario_name,
            scenario,
            sample,
            iteration,
            seed,
            comparator,
            paired_record,
            split_evidence,
            iteration_dir / "Comparators",
            experiment_id,
            config_hash_value,
            environment,
            common,
            force_rerun=False,
        )
        model_records.append(record)
        comparator_statuses[comparator["id"]] = bool(
            record.get("success", False)
        )
        model_statuses[f"auxiliary:{comparator['id']}"] = (
            comparator_statuses[comparator["id"]]
        )
    paired = v14_build_paired_comparison(model_records)
    if not paired.empty and not bool(paired["Exact_Pair_Valid"].all()):
        raise RuntimeError(
            "Comparator-only resume failed exact-pair validation."
        )
    v14_write_table_bundle(
        paired, iteration_dir / "paired_comparison_iteration_level"
    )
    v14_write_table_bundle(
        pd.DataFrame(model_records),
        iteration_dir / "iteration_model_results",
    )
    manifest_path = iteration_dir / "iteration_manifest.json"
    manifest = (
        v14_load_json_file(manifest_path)
        if manifest_path.exists() else {}
    )
    manifest.update({
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "configuration_hash": config_hash_value,
        "scenario": scenario_name,
        "requested_sample_size": sample["requested"],
        "actual_sample_size": sample["actual"],
        "iteration": iteration,
        "seed": seed,
        "success": all(model_statuses.values()),
        "model_statuses": model_statuses,
        "comparator_statuses": comparator_statuses,
        "core_models_reused_without_rerun": True,
        "resume_updated_at_utc": v14_utc_now(),
    })
    v14_atomic_write_json(manifest_path, manifest)
    required_paths = [
        manifest_path,
        iteration_dir / "resolved_config.json",
        iteration_dir / "Splits" / "outer_split_audit.json",
        iteration_dir / "Splits" / "inner_split_audit.json",
        iteration_dir / "Splits" / "train_indices.npy",
        iteration_dir / "Splits" / "test_indices.npy",
        iteration_dir / "iteration_model_results.csv",
        iteration_dir / "paired_comparison_iteration_level.csv",
    ]
    required_paths.extend(
        iteration_dir
        / "Comparators"
        / safe_path_part(comparator["id"], "Comparator")
        / "_SUCCESS.json"
        for comparator in comparators
    )
    checksums = {
        str(path.relative_to(iteration_dir)).replace("\\", "/"):
        v14_sha256_file(path)
        for path in required_paths if path.exists()
    }
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "config_hash": config_hash_value,
        "scenario": scenario_name,
        "sample_size_actual": sample["actual"],
        "iteration": iteration,
        "seed": seed,
        "complete": (
            len(checksums) == len(required_paths)
            and all(model_statuses.values())
        ),
        "scientific_success": all(model_statuses.values()),
        "model_statuses": model_statuses,
        "comparator_statuses": comparator_statuses,
        "core_models_reused_without_rerun": True,
        "required_file_checksums": checksums,
        "completed_at_utc": v14_utc_now(),
    }
    v14_atomic_write_json(iteration_dir / "_SUCCESS.json", marker)
    logger.write(
        "comparator_resume_completed",
        **common,
        comparator_statuses=comparator_statuses,
        core_models_reused_without_rerun=True,
    )
    return "resumed/comparators-only"


# =============================================================================
# 9. Atomic scenario x sample-size x iteration execution
# =============================================================================

def v14_atomic_iteration(
    X,
    y,
    groups,
    timestamps,
    master_config,
    scenario_name,
    scenario,
    sample,
    iteration,
    run_dir,
    experiment_id,
    config_hash_value,
    environment,
    logger,
    force_rerun=False,
):
    seed = int(master_config["base_seed"]) + int(iteration)
    sample_dir = (
        Path(run_dir)
        / scenario_folder_name(scenario_name)
        / sample["folder"]
    )
    iteration_dir = sample_dir / f"Iteration_{iteration:03d}"
    success_marker = iteration_dir / "_SUCCESS.json"
    if success_marker.exists() and not force_rerun:
        marker = v14_load_json_file(success_marker)
        if (
            marker.get("config_hash") == config_hash_value
            and v14_validate_completion_marker(iteration_dir, marker)
        ):
            logger.write(
                "iteration_resumed_skipped",
                scenario=scenario_name,
                sample_size=sample["actual"],
                iteration=iteration,
                seed=seed,
            )
            return "resumed/skipped"
        if v14_core_iteration_artifacts_valid(
            iteration_dir, scenario["enabled_models"]
        ):
            return v14_resume_auxiliary_only(
                X,
                y,
                groups,
                timestamps,
                master_config,
                scenario_name,
                scenario,
                sample,
                iteration,
                iteration_dir,
                experiment_id,
                config_hash_value,
                environment,
                logger,
            )
    if (
        not success_marker.exists()
        and not force_rerun
        and v14_core_iteration_artifacts_valid(
            iteration_dir, scenario["enabled_models"]
        )
    ):
        return v14_resume_auxiliary_only(
            X,
            y,
            groups,
            timestamps,
            master_config,
            scenario_name,
            scenario,
            sample,
            iteration,
            iteration_dir,
            experiment_id,
            config_hash_value,
            environment,
            logger,
        )
    if success_marker.exists() and force_rerun:
        success_marker.unlink()
    iteration_dir.mkdir(parents=True, exist_ok=True)
    split_dir = iteration_dir / "Splits"
    model_root = iteration_dir / "Models"
    comparator_root = iteration_dir / "Comparators"
    engine_dir = iteration_dir / "_V12_Engine"
    started_at = v14_utc_now()
    monotonic_start = time.perf_counter()
    common = {
        "Experiment_ID": experiment_id,
        "Pipeline_Version": PIPELINE_VERSION,
        "Config_Hash": config_hash_value,
        "Scenario": scenario_name,
        "Scenario_Role": scenario.get(
            "semantic_role"
        ) or scenario_name,
        "Sample_Size_Requested": sample["requested"],
        "Sample_Size_Actual": sample["actual"],
        "Iteration": iteration,
        "Seed": seed,
    }
    logger.write(
        "iteration_started",
        **common,
        models=scenario["enabled_models"],
    )
    split_evidence = v14_compute_split_evidence(
        X,
        y,
        groups,
        timestamps,
        master_config,
        scenario,
        sample["actual"],
        seed,
    )
    common.update({
        "Split_Fingerprint": split_evidence["outer_info"].get(
            "Split_Fingerprint"
        ),
        "Inner_Split_Fingerprint": split_evidence["inner_info"].get(
            "Split_Fingerprint"
        ),
        "Test_Indices_SHA256": v14_hash_array(
            split_evidence["test_indices"]
        ),
        "Test_Labels_SHA256": v14_hash_array(split_evidence["y_test"]),
    })
    split_checksums = v14_save_split_evidence(
        split_dir,
        split_evidence,
        groups,
        master_config,
        scenario_name,
        sample["requested"],
        sample["actual"],
        iteration,
        seed,
    )
    atomic_config = v14_atomic_engine_config(
        master_config,
        scenario_name,
        scenario,
        sample["actual"],
        seed,
        engine_dir,
        external_iteration=iteration,
        display_scenario_name=scenario_name,
    )
    v14_atomic_write_json(iteration_dir / "resolved_config.json", atomic_config)
    engine_outputs = run_monte_carlo(
        X,
        y,
        atomic_config,
        groups=groups,
        timestamps=timestamps,
    )
    raw_results, raw_path = v14_load_atomic_raw_results(
        engine_dir, scenario_name
    )
    iter_frame = engine_outputs["all_iteration_metrics_df"].copy()
    if not iter_frame.empty:
        engine_outer = {
            str(value)
            for value in iter_frame.get(
                "Split_Fingerprint", pd.Series(dtype=str)
            ).dropna().unique()
        }
        engine_inner = {
            str(value)
            for value in iter_frame.get(
                "Inner_Split_Fingerprint", pd.Series(dtype=str)
            ).dropna().unique()
        }
        expected_outer = str(
            split_evidence["outer_info"]["Split_Fingerprint"]
        )
        expected_inner = str(
            split_evidence["inner_info"]["Split_Fingerprint"]
        )
        if engine_outer and engine_outer != {expected_outer}:
            raise RuntimeError(
                "Persisted outer split evidence does not match the split used "
                f"by the scientific engine: {expected_outer} versus {engine_outer}."
            )
        if engine_inner and engine_inner != {expected_inner}:
            raise RuntimeError(
                "Persisted inner split evidence does not match the split used "
                f"by the scientific engine: {expected_inner} versus {engine_inner}."
            )
    ended_at = v14_utc_now()
    model_records = []
    model_statuses = {}
    for model_name in scenario["enabled_models"]:
        matches = (
            iter_frame[iter_frame["Model"] == model_name]
            if not iter_frame.empty and "Model" in iter_frame.columns
            else pd.DataFrame()
        )
        row = matches.iloc[-1] if not matches.empty else None
        record = v14_save_model_artifacts(
            model_root / safe_path_part(model_name, "Model"),
            model_name,
            scenario["resolved_models"][model_name],
            scenario,
            row,
            raw_results,
            engine_outputs,
            split_evidence,
            environment,
            master_config.get("energy", {}),
            common,
            started_at,
            ended_at,
            identity={
                "Analysis_Role": "core_model",
                "Execution_Variant": model_name,
                "Base_Model": model_name,
                "Comparator_ID": None,
                "Paired_With": None,
            },
            cpu_monitor_record=v14_latest_cpu_monitor_record(
                scenario_name,
                sample["actual"],
                iteration,
                model_name,
            ),
        )
        model_records.append(record)
        model_statuses[model_name] = bool(record.get("success", False))
        logger.write(
            "model_artifacts_written",
            **common,
            model=model_name,
            success=model_statuses[model_name],
            execution_path=record.get("Execution_Path"),
            resolved_device=record.get("Resolved_Device"),
            energy_scope=record.get("Energy_Scope"),
        )
    core_records_hash_before_comparators = v14_sha256_bytes(
        v14_canonical_json(model_records).encode("utf-8")
    )
    comparator_statuses = {}
    comparators = v14_resolve_auxiliary_comparators(
        master_config,
        scenario_name,
        scenario,
        sample,
    )
    for comparator in comparators:
        paired_record = next(
            (
                item for item in model_records
                if (
                    item.get("Analysis_Role") == "core_model"
                    and item.get("Execution_Variant")
                    == comparator.get("pair_with")
                )
            ),
            None,
        )
        if paired_record is None:
            raise RuntimeError(
                f"Comparator {comparator['id']!r} has no persisted paired "
                f"core record {comparator.get('pair_with')!r}."
            )
        comparator_record = v14_run_auxiliary_comparator(
            X,
            y,
            groups,
            timestamps,
            master_config,
            scenario_name,
            scenario,
            sample,
            iteration,
            seed,
            comparator,
            paired_record,
            split_evidence,
            comparator_root,
            experiment_id,
            config_hash_value,
            environment,
            common,
            force_rerun=force_rerun,
        )
        model_records.append(comparator_record)
        comparator_statuses[comparator["id"]] = bool(
            comparator_record.get("success", False)
        )
        model_statuses[f"auxiliary:{comparator['id']}"] = (
            comparator_statuses[comparator["id"]]
        )
        logger.write(
            "comparator_artifacts_written",
            **common,
            comparator_id=comparator["id"],
            base_model=comparator["base_model"],
            paired_with=comparator.get("pair_with"),
            success=comparator_statuses[comparator["id"]],
        )
    core_records_after = [
        item for item in model_records
        if item.get("Analysis_Role") == "core_model"
    ]
    core_records_hash_after_comparators = v14_sha256_bytes(
        v14_canonical_json(core_records_after).encode("utf-8")
    )
    comparator_budget_isolation_valid = (
        core_records_hash_before_comparators
        == core_records_hash_after_comparators
    )
    paired_comparison = v14_build_paired_comparison(model_records)
    if not paired_comparison.empty and not bool(
        paired_comparison["Exact_Pair_Valid"].all()
    ):
        raise RuntimeError(
            "An auxiliary comparator failed exact paired split/test evidence "
            "validation."
        )
    v14_write_table_bundle(
        paired_comparison,
        iteration_dir / "paired_comparison_iteration_level",
    )
    duration = time.perf_counter() - monotonic_start
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario_name,
        "requested_sample_size": sample["requested"],
        "actual_sample_size": sample["actual"],
        "iteration": iteration,
        "seed": seed,
        "configuration_hash": config_hash_value,
        "start_timestamp_utc": started_at,
        "end_timestamp_utc": ended_at,
        "total_duration_seconds": duration,
        "success": all(model_statuses.values()),
        "completed_with_model_failures": not all(model_statuses.values()),
        "model_statuses": model_statuses,
        "comparator_statuses": comparator_statuses,
        "comparator_budget_isolation_valid": (
            comparator_budget_isolation_valid
        ),
        "python_version": environment["python_version"],
        "platform": environment["os"],
        "dependency_versions": environment["packages"],
        "cuda_available": environment["cuda_available"],
        "cpu": environment["cpu"],
        "gpu": environment["gpu_models"],
        "execution_devices": {
            model_name: v14_expected_model_device(
                model_name,
                scenario["resolved_models"][model_name],
                scenario,
            )
            for model_name in scenario["enabled_models"]
        },
        "split_checksums": split_checksums,
        "legacy_engine_evidence": str(raw_path),
    }
    v14_atomic_write_json(
        iteration_dir / "iteration_manifest.json", manifest
    )
    consolidated = pd.DataFrame(model_records)
    v14_write_table_bundle(
        consolidated, iteration_dir / "iteration_model_results"
    )
    required_paths = [
        iteration_dir / "iteration_manifest.json",
        iteration_dir / "resolved_config.json",
        split_dir / "outer_split_audit.json",
        split_dir / "inner_split_audit.json",
        split_dir / "train_indices.npy",
        split_dir / "test_indices.npy",
        iteration_dir / "iteration_model_results.csv",
        iteration_dir / "paired_comparison_iteration_level.csv",
    ]
    required_paths.extend(
        comparator_root
        / safe_path_part(comparator["id"], "Comparator")
        / "_SUCCESS.json"
        for comparator in comparators
    )
    checksums = {
        str(path.relative_to(iteration_dir)).replace("\\", "/"):
        v14_sha256_file(path)
        for path in required_paths if path.exists()
    }
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "config_hash": config_hash_value,
        "scenario": scenario_name,
        "sample_size_actual": sample["actual"],
        "iteration": iteration,
        "seed": seed,
        "complete": (
            len(checksums) == len(required_paths)
            and all(model_statuses.values())
        ),
        "scientific_success": all(model_statuses.values()),
        "model_statuses": model_statuses,
        "comparator_statuses": comparator_statuses,
        "comparator_budget_isolation_valid": (
            comparator_budget_isolation_valid
        ),
        "required_file_checksums": checksums,
        "completed_at_utc": ended_at,
    }
    marker_path = (
        success_marker
        if marker["complete"] else iteration_dir / "_FAILED.json"
    )
    v14_atomic_write_json(marker_path, marker)
    logger.write(
        "iteration_completed",
        **common,
        duration_seconds=duration,
        model_statuses=model_statuses,
    )
    if (
        not all(model_statuses.values())
        and master_config.get("execution", {}).get(
            "on_model_error", "continue"
        ) == "raise"
    ):
        raise RuntimeError(
            "One or more models failed and execution.on_model_error='raise': "
            f"{model_statuses}"
        )
    return "complete" if all(model_statuses.values()) else "failed"


# =============================================================================
# 10. Aggregation and saved-data-only figure generation
# =============================================================================

def v14_validate_completion_marker(iteration_dir, marker):
    if not marker.get("complete", False):
        return False
    for relative, expected in marker.get(
        "required_file_checksums", {}
    ).items():
        path = Path(iteration_dir) / relative
        if not path.exists() or v14_sha256_file(path) != expected:
            return False
    return True


def v14_collect_iteration_results(directory):
    frames = []
    for path in sorted(
        Path(directory).glob("Iteration_*/iteration_model_results.csv")
    ):
        frame = pd.read_csv(path)
        frame["_source_artifact"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def v14_collect_paired_comparisons(directory):
    frames = []
    for path in sorted(
        Path(directory).glob(
            "Iteration_*/paired_comparison_iteration_level.csv"
        )
    ):
        frame = pd.read_csv(path)
        frame["_source_artifact"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def v14_paired_comparison_summary(paired_frame):
    if paired_frame.empty:
        return pd.DataFrame()
    rows = []
    group_columns = [
        column for column in (
            "Scenario",
            "Sample_Size_Actual",
            "Primary_Model",
            "Comparator_ID",
            "Comparator_Base_Model",
        )
        if column in paired_frame.columns
    ]
    metric_names = [
        "AUROC",
        "Balanced_Accuracy",
        "Sensitivity",
        "Precision",
        "Brier_Score",
        "Runtime_Seconds",
    ]
    for keys, group in paired_frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            **dict(zip(group_columns, keys)),
            "N_Pairs": int(len(group)),
            "Exact_Pairs_N": int(
                group.get(
                    "Exact_Pair_Valid",
                    pd.Series(False, index=group.index),
                ).astype(str).str.lower().isin(
                    {"true", "1", "yes"}
                ).sum()
            ),
        }
        for metric in metric_names:
            primary_column = (
                "Primary_Runtime_Seconds"
                if metric == "Runtime_Seconds"
                else f"Primary_{metric}"
            )
            comparator_column = (
                "Comparator_Runtime_Seconds"
                if metric == "Runtime_Seconds"
                else f"Comparator_{metric}"
            )
            difference_column = (
                "Paired_Difference_Runtime_Comparator_Minus_Primary"
                if metric == "Runtime_Seconds"
                else (
                    f"Paired_Difference_{metric}_"
                    "Comparator_Minus_Primary"
                )
            )
            primary = pd.to_numeric(
                group.get(primary_column), errors="coerce"
            )
            comparator = pd.to_numeric(
                group.get(comparator_column), errors="coerce"
            )
            differences = pd.to_numeric(
                group.get(difference_column), errors="coerce"
            ).dropna()
            primary_mean = (
                float(primary.mean()) if primary.notna().any() else None
            )
            comparator_mean = (
                float(comparator.mean())
                if comparator.notna().any() else None
            )
            count = int(len(differences))
            mean_difference = (
                float(differences.mean()) if count else None
            )
            sd_difference = (
                float(differences.std(ddof=1))
                if count > 1 else (0.0 if count == 1 else None)
            )
            half_width = (
                float(
                    st.t.ppf(0.975, count - 1)
                    * sd_difference / np.sqrt(count)
                )
                if count > 1 and sd_difference is not None else 0.0
            )
            prefix = metric
            row[f"{prefix}_Primary_Mean"] = primary_mean
            row[f"{prefix}_Comparator_Mean"] = comparator_mean
            row[
                f"{prefix}_Difference_Of_Means_Comparator_Minus_Primary"
            ] = (
                float(comparator_mean - primary_mean)
                if primary_mean is not None and comparator_mean is not None
                else None
            )
            row[
                f"{prefix}_Difference_Of_Means_Primary_Minus_Comparator"
            ] = (
                float(primary_mean - comparator_mean)
                if primary_mean is not None and comparator_mean is not None
                else None
            )
            row[
                f"{prefix}_Mean_Paired_Difference_Comparator_Minus_Primary"
            ] = mean_difference
            primary_minus_comparator = (
                -mean_difference
                if mean_difference is not None else None
            )
            row[
                f"{prefix}_Mean_Paired_Difference_Primary_Minus_Comparator"
            ] = primary_minus_comparator
            row[
                f"{prefix}_Paired_Difference_SD_Either_Direction"
            ] = sd_difference
            comparator_lower = (
                mean_difference - half_width
                if mean_difference is not None else None
            )
            comparator_upper = (
                mean_difference + half_width
                if mean_difference is not None else None
            )
            row[
                f"{prefix}_Paired_Difference_Comparator_Minus_Primary_CI95_Lower"
            ] = comparator_lower
            row[
                f"{prefix}_Paired_Difference_Comparator_Minus_Primary_CI95_Upper"
            ] = comparator_upper
            row[
                f"{prefix}_Paired_Difference_Primary_Minus_Comparator_CI95_Lower"
            ] = (
                -comparator_upper
                if comparator_upper is not None else None
            )
            row[
                f"{prefix}_Paired_Difference_Primary_Minus_Comparator_CI95_Upper"
            ] = (
                -comparator_lower
                if comparator_lower is not None else None
            )
            row[f"{prefix}_Paired_Difference_SD"] = sd_difference
            row[f"{prefix}_Paired_Difference_CI95_Lower"] = (
                comparator_lower
            )
            row[f"{prefix}_Paired_Difference_CI95_Upper"] = (
                comparator_upper
            )
            row[f"{prefix}_N_Pairs"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def v14_descriptive_summary(
    iteration_frame, analysis_roles=("core_model",)
):
    if iteration_frame.empty:
        return pd.DataFrame()
    iteration_frame = iteration_frame.copy()
    if "Analysis_Role" not in iteration_frame.columns:
        iteration_frame["Analysis_Role"] = "core_model"
    if analysis_roles is not None:
        iteration_frame = iteration_frame[
            iteration_frame["Analysis_Role"].isin(list(analysis_roles))
        ].copy()
    if iteration_frame.empty:
        return pd.DataFrame()
    numeric_candidates = [
        *V14_PREDICTION_METRICS,
        "Preprocessing_Time_Seconds",
        "Actual_Optuna_Tuning_Time_Seconds",
        "Optuna_Tuning_Time_Capped_Seconds",
        "Final_Fit_Predict_Time_Seconds",
        "Final_Fit_Time_Seconds",
        "Prediction_Time_Seconds",
        "Fit_Plus_Prediction_Time_Seconds",
        "Actual_Total_Runtime_Seconds",
        "Budget_Accounted_Runtime_Seconds",
        "Reference_Budget_Seconds",
        "Effective_Budget_Seconds",
        "Budget_Overrun_Seconds",
        "TabPFN_Context_Search_Runtime_Seconds",
        "Measured_Energy_kWh",
        "Measured_ClientSide_Energy_kWh",
        "Estimated_Energy_kWh",
        "TabPFN_Context_N",
        "TabPFN_Context_Fraction",
        "TabPFN_Reference_Budget_Seconds",
        "TabPFN_Effective_Budget_Seconds",
    ]
    rows = []
    for model_name, group in iteration_frame.groupby("Model", dropna=False):
        success_values = group.get(
            "success", pd.Series(False, index=group.index)
        )
        success_values = success_values.astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        row = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "Model": model_name,
            "Analysis_Role": str(group["Analysis_Role"].iloc[0]),
            "Base_Model": (
                str(group["Base_Model"].iloc[0])
                if "Base_Model" in group.columns else model_name
            ),
            "Requested_N_Iterations": int(len(group)),
            "Successful_N": int(success_values.sum()),
            "Failed_N": int((~success_values).sum()),
            "CI95_Quantity": "two-sided Student-t confidence interval for the mean",
        }
        for column in numeric_candidates:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                statistics = {
                    "N": 0,
                    "Mean": None,
                    "SD": None,
                    "Median": None,
                    "Min": None,
                    "Max": None,
                    "Q1": None,
                    "Q3": None,
                    "CI95_Lower": None,
                    "CI95_Upper": None,
                }
            else:
                count = int(len(values))
                mean_value = float(values.mean())
                sd_value = (
                    float(values.std(ddof=1)) if count > 1 else 0.0
                )
                if count > 1 and sd_value > 0:
                    half_width = float(
                        st.t.ppf(0.975, count - 1)
                        * sd_value / np.sqrt(count)
                    )
                else:
                    half_width = 0.0
                statistics = {
                    "N": count,
                    "Mean": mean_value,
                    "SD": sd_value,
                    "Median": float(values.median()),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                    "Q1": float(values.quantile(0.25)),
                    "Q3": float(values.quantile(0.75)),
                    "CI95_Lower": mean_value - half_width,
                    "CI95_Upper": mean_value + half_width,
                }
            for statistic_name, statistic_value in statistics.items():
                row[f"{column}_{statistic_name}"] = statistic_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)


def v14_load_prediction_artifacts(sample_dir):
    rows = []
    sample_dir = Path(sample_dir)
    paths = list(
        sample_dir.glob("Iteration_*/Models/*/predictions.npz")
    ) + list(
        sample_dir.glob("Iteration_*/Comparators/*/predictions.npz")
    )
    for path in sorted(paths):
        iteration_name = path.parents[2].name
        metrics_path = path.parent / "metrics.json"
        identity = (
            v14_load_json_file(metrics_path)
            if metrics_path.exists() else {}
        )
        model_name = identity.get("Model", path.parent.name)
        iteration = int(iteration_name.split("_")[-1])
        with np.load(path, allow_pickle=False) as payload:
            for index in range(len(payload["y_true"])):
                rows.append({
                    "Iteration": iteration,
                    "Model": model_name,
                    "Analysis_Role": identity.get(
                        "Analysis_Role", "core_model"
                    ),
                    "Execution_Variant": identity.get(
                        "Execution_Variant", model_name
                    ),
                    "Base_Model": identity.get(
                        "Base_Model", model_name
                    ),
                    "Comparator_ID": identity.get("Comparator_ID"),
                    "Paired_With": identity.get("Paired_With"),
                    "test_row_index": int(payload["test_row_index"][index]),
                    "y_true": int(payload["y_true"][index]),
                    "predicted_class": int(
                        payload["predicted_class"][index]
                    ),
                    "probability_class_0": float(
                        payload["probability_class_0"][index]
                    ),
                    "probability_class_1": float(
                        payload["probability_class_1"][index]
                    ),
                    "_source_artifact": str(path),
                })
    return pd.DataFrame(rows)


def v14_build_plot_data(sample_dir, iteration_frame, plot_config):
    sample_dir = Path(sample_dir)
    plot_data_dir = sample_dir / "Aggregated" / "Plot_Data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for metric in V14_PREDICTION_METRICS:
        if metric not in iteration_frame.columns:
            continue
        for _, row in iteration_frame.iterrows():
            value = v14_scalar(row.get(metric))
            if value is not None:
                metric_rows.append({
                    "Iteration": int(row["Iteration"]),
                    "Model": row["Model"],
                    "Analysis_Role": row.get(
                        "Analysis_Role", "core_model"
                    ),
                    "Execution_Variant": row.get(
                        "Execution_Variant", row["Model"]
                    ),
                    "Base_Model": row.get("Base_Model", row["Model"]),
                    "Comparator_ID": row.get("Comparator_ID"),
                    "Paired_With": row.get("Paired_With"),
                    "Metric": metric,
                    "Value": value,
                    "Source_Artifact": row.get("_source_artifact"),
                })
    predictive = pd.DataFrame(metric_rows)
    v14_write_table_bundle(
        predictive, plot_data_dir / "predictive_distributions"
    )
    runtime_columns = [
        "Actual_Total_Runtime_Seconds",
        "Actual_Optuna_Tuning_Time_Seconds",
        "Final_Fit_Predict_Time_Seconds",
        "Reference_Budget_Seconds",
        "Effective_Budget_Seconds",
    ]
    runtime_rows = []
    for _, row in iteration_frame.iterrows():
        for metric in runtime_columns:
            value = v14_scalar(row.get(metric))
            if value is not None:
                runtime_rows.append({
                    "Iteration": int(row["Iteration"]),
                    "Model": row["Model"],
                    "Analysis_Role": row.get(
                        "Analysis_Role", "core_model"
                    ),
                    "Execution_Variant": row.get(
                        "Execution_Variant", row["Model"]
                    ),
                    "Base_Model": row.get("Base_Model", row["Model"]),
                    "Comparator_ID": row.get("Comparator_ID"),
                    "Paired_With": row.get("Paired_With"),
                    "Runtime_Component": metric,
                    "Seconds": value,
                    "Source_Artifact": row.get("_source_artifact"),
                })
    runtime = pd.DataFrame(runtime_rows)
    v14_write_table_bundle(runtime, plot_data_dir / "runtime")
    local_energy = iteration_frame[
        iteration_frame.get(
            "Energy_Scope", pd.Series(index=iteration_frame.index, dtype=str)
        ) == "local_model_execution"
    ].copy()
    client_energy = iteration_frame[
        iteration_frame.get(
            "Energy_Scope", pd.Series(index=iteration_frame.index, dtype=str)
        ) == "local_client_process_only"
    ].copy()
    local_columns = [
        column for column in [
            "Iteration", "Model", "Measured_Energy_kWh",
            "Energy_Scope", "Energy_Comparable_Flag",
            "Energy_Comparability_Reason", "_source_artifact",
        ] if column in local_energy.columns
    ]
    client_columns = [
        column for column in [
            "Iteration", "Model", "Measured_ClientSide_Energy_kWh",
            "Energy_Scope", "Energy_Comparable_Flag",
            "Energy_Comparability_Reason", "_source_artifact",
        ] if column in client_energy.columns
    ]
    v14_write_table_bundle(
        local_energy[local_columns],
        plot_data_dir / "energy_local_execution",
    )
    v14_write_table_bundle(
        client_energy[client_columns],
        plot_data_dir / "energy_cloud_client_side",
    )
    prediction_frame = v14_load_prediction_artifacts(sample_dir)
    v14_write_table_bundle(
        prediction_frame, plot_data_dir / "prediction_rows"
    )
    roc_rows = []
    calibration_rows = []
    if not prediction_frame.empty:
        mean_fpr = np.linspace(0.0, 1.0, 101)
        bins = int(plot_config.get("calibration_bins", 10))
        for model_name, model_frame in prediction_frame.groupby("Model"):
            interpolated = []
            for iteration, iteration_predictions in model_frame.groupby(
                "Iteration"
            ):
                y_values = iteration_predictions["y_true"].to_numpy()
                probability = iteration_predictions[
                    "probability_class_1"
                ].to_numpy()
                fpr, tpr, _ = roc_curve(y_values, probability)
                current = np.interp(mean_fpr, fpr, tpr)
                current[0] = 0.0
                current[-1] = 1.0
                interpolated.append(current)
                for x_value, y_value in zip(fpr, tpr):
                    roc_rows.append({
                        "Model": model_name,
                        "Iteration": int(iteration),
                        "Curve_Type": "iteration",
                        "FPR": float(x_value),
                        "TPR": float(y_value),
                    })
                observed, predicted_values = calibration_curve(
                    y_values, probability, n_bins=bins, strategy="quantile"
                )
                for predicted_value, observed_value in zip(
                    predicted_values, observed
                ):
                    calibration_rows.append({
                        "Model": model_name,
                        "Iteration": int(iteration),
                        "Mean_Predicted_Probability": float(predicted_value),
                        "Observed_Fraction_Positive": float(observed_value),
                    })
            if interpolated:
                matrix = np.vstack(interpolated)
                for index, x_value in enumerate(mean_fpr):
                    roc_rows.append({
                        "Model": model_name,
                        "Iteration": None,
                        "Curve_Type": "mean_interpolated",
                        "FPR": float(x_value),
                        "TPR": float(matrix[:, index].mean()),
                        "TPR_SD": (
                            float(matrix[:, index].std(ddof=1))
                            if len(matrix) > 1 else 0.0
                        ),
                        "N_Iterations": int(len(matrix)),
                    })
    roc_frame = pd.DataFrame(roc_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    v14_write_table_bundle(roc_frame, plot_data_dir / "roc")
    v14_write_table_bundle(
        calibration_frame, plot_data_dir / "calibration"
    )
    tradeoff_columns = [
        column for column in [
            "Iteration", "Model", "AUROC",
            "Actual_Total_Runtime_Seconds", "Execution_Path",
            "Resolved_Device", "Analysis_Role", "Execution_Variant",
            "Base_Model", "Comparator_ID", "Paired_With",
            "_source_artifact",
        ] if column in iteration_frame.columns
    ]
    tradeoff = iteration_frame[tradeoff_columns].copy()
    v14_write_table_bundle(
        tradeoff, plot_data_dir / "auroc_runtime_tradeoff"
    )
    paired_source = (
        sample_dir
        / "Aggregated"
        / "Data"
        / "paired_comparison_iteration_level"
    )
    paired = (
        v14_read_table_bundle(paired_source)
        if paired_source.with_suffix(".csv").exists()
        else pd.DataFrame()
    )
    v14_write_table_bundle(
        paired, plot_data_dir / "paired_budgeted_vs_comparator"
    )
    return {
        "predictive": predictive,
        "runtime": runtime,
        "local_energy": local_energy,
        "client_energy": client_energy,
        "predictions": prediction_frame,
        "roc": roc_frame,
        "calibration": calibration_frame,
        "tradeoff": tradeoff,
        "paired": paired,
        "plot_data_dir": plot_data_dir,
    }


def v14_save_figure(fig, stem, plot_config, sources, description):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if plot_config.get("save_png", True):
        path = stem.with_suffix(".png")
        fig.savefig(
            path,
            dpi=int(plot_config.get("dpi", 180)),
            bbox_inches="tight",
        )
        written.append(path)
    if plot_config.get("save_svg", True):
        path = stem.with_suffix(".svg")
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    provenance = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": v14_utc_now(),
        "Figure_ID": stem.name,
        "Plot_Type": plot_config.get(
            "_plot_type", stem.name.split("_")[0]
        ),
        "figure_stem": str(stem),
        "description": description,
        "source_artifacts": [str(Path(source)) for source in sources],
        "source_checksums": {
            str(Path(source)): v14_sha256_file(source)
            for source in sources if Path(source).exists()
        },
        "data_flow": (
            "persisted iteration evidence -> persisted aggregate -> "
            "persisted plot data -> figure"
        ),
        "filters": plot_config.get("_filters", {}),
        "aggregation_method": plot_config.get(
            "_aggregation_method", "saved iteration values"
        ),
        "confidence_interval_definition": plot_config.get(
            "_confidence_interval_definition",
            "two-sided Student-t 95% CI where shown",
        ),
        "runtime_field": plot_config.get("_runtime_field"),
        "runtime_field_definition": (
            "Actual_Total_Runtime_Seconds is observed stopwatch-style model "
            "execution wall-clock and is not the tuned-reference HPO budget."
            if plot_config.get("_runtime_field")
            == "Actual_Total_Runtime_Seconds"
            else (
                "Reference_Budget_Seconds is the dedicated Optuna tuning-loop "
                "wall-clock for tuned references, or the configured execution "
                "runtime for non-tuned references."
                if plot_config.get("_runtime_field")
                == "Reference_Budget_Seconds"
                else None
            )
        ),
        "wall_clock_definition": (
            "Elapsed real time between the start and end of the measured "
            "operation, analogous to timing it with a stopwatch; it is not "
            "CPU time summed across cores."
        ),
        "runtime_quantity_distinction": (
            "Reference_Budget_Seconds, Actual_Optuna_Tuning_Time_Seconds, "
            "Optuna_Tuning_Time_Capped_Seconds, "
            "Final_Fit_Predict_Time_Seconds, "
            "Budget_Accounted_Runtime_Seconds, and "
            "Actual_Total_Runtime_Seconds are distinct quantities."
        ),
        "energy_field": plot_config.get("_energy_field"),
        "runtime_axis_scale": plot_config.get(
            "runtime_axis_scale", "linear"
        ),
        "config_hash": plot_config.get("_config_hash"),
    }
    v14_atomic_write_json(
        str(stem) + ".provenance.json", provenance
    )
    return written


def v14_plot_box(frame, category, value, title, ylabel, stem, config, sources):
    if plt is None or frame.empty or value not in frame.columns:
        return []
    clean = frame[[category, value]].copy()
    clean[value] = pd.to_numeric(clean[value], errors="coerce")
    clean = clean.dropna()
    is_runtime_log = (
        config.get("runtime_axis_scale", "linear") == "log"
        and (
            "Runtime" in value
            or "Seconds" in value
            or "runtime" in str(stem).lower()
        )
    )
    if is_runtime_log:
        clean = clean[clean[value] > 0]
    if clean.empty:
        return []
    labels = list(dict.fromkeys(clean[category].astype(str).tolist()))
    values = [
        clean.loc[clean[category].astype(str) == label, value].to_numpy()
        for label in labels
    ]
    fig, axis = plt.subplots(figsize=(max(7, len(labels) * 1.2), 5))
    try:
        axis.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        axis.boxplot(values, labels=labels, showmeans=True)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if is_runtime_log:
        axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    written = v14_save_figure(
        fig, stem, config, sources,
        f"Box plot of {value} using every persisted iteration value.",
    )
    plt.close(fig)
    return written


def v14_write_plotly_html(frame, path, kind, title, **fields):
    if frame.empty:
        return None
    try:
        import plotly.express as px
        if kind == "box":
            figure = px.box(
                frame,
                x=fields["x"],
                y=fields["y"],
                color=fields.get("color"),
                points="all",
                title=title,
            )
        elif kind == "scatter":
            figure = px.scatter(
                frame,
                x=fields["x"],
                y=fields["y"],
                color=fields.get("color"),
                hover_data=fields.get("hover_data"),
                title=title,
            )
        elif kind == "line":
            figure = px.line(
                frame,
                x=fields["x"],
                y=fields["y"],
                color=fields.get("color"),
                line_dash=fields.get("line_dash"),
                title=title,
            )
        else:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".html", dir=str(path.parent)
        )
        os.close(fd)
        figure.write_html(temporary, include_plotlyjs="cdn")
        os.replace(temporary, path)
        return path
    except Exception as exc:
        v14_atomic_write_json(
            str(path) + ".unavailable.json",
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "available": False,
                "reason": str(exc),
            },
        )
        return None


def v14_generate_sample_plots(sample_dir, plot_config):
    sample_dir = Path(sample_dir)
    aggregate_dir = sample_dir / "Aggregated"
    iteration_frame = v14_read_table_bundle(
        aggregate_dir / "Data" / "iteration_level_results"
    )
    plot_data = v14_build_plot_data(
        sample_dir, iteration_frame, plot_config
    )
    plot_dir = aggregate_dir / "Plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    predictive_source = (
        plot_data["plot_data_dir"] / "predictive_distributions.csv"
    )
    for metric in V14_PREDICTION_METRICS:
        subset = plot_data["predictive"][
            plot_data["predictive"]["Metric"] == metric
        ] if not plot_data["predictive"].empty else pd.DataFrame()
        written.extend(v14_plot_box(
            subset,
            "Model",
            "Value",
            f"{metric.replace('_', ' ')} distribution",
            metric.replace("_", " "),
            plot_dir / f"predictive_{metric.lower()}",
            plot_config,
            [predictive_source],
        ))
        if plot_config.get("save_html", True):
            html = v14_write_plotly_html(
                subset,
                plot_dir / f"predictive_{metric.lower()}.html",
                "box",
                f"{metric.replace('_', ' ')} distribution",
                x="Model",
                y="Value",
                color="Model",
            )
            if html:
                written.append(html)
    runtime_wide = iteration_frame.copy()
    runtime_source = plot_data["plot_data_dir"] / "runtime.csv"
    written.extend(v14_plot_box(
        runtime_wide,
        "Model",
        "Actual_Total_Runtime_Seconds",
        "Actual total runtime",
        "Seconds",
        plot_dir / "runtime_actual_total",
        plot_config,
        [runtime_source],
    ))
    paired = plot_data["paired"].copy()
    paired_source = (
        plot_data["plot_data_dir"]
        / "paired_budgeted_vs_comparator.csv"
    )
    if not paired.empty:
        paired_auroc_rows = []
        paired_runtime_rows = []
        for _, paired_row in paired.iterrows():
            for label, model_column, value_column in (
                (
                    "Primary",
                    "Primary_Model",
                    "Primary_AUROC",
                ),
                (
                    "Comparator",
                    "Comparator_ID",
                    "Comparator_AUROC",
                ),
            ):
                paired_auroc_rows.append({
                    "Iteration": paired_row.get("Iteration"),
                    "Pair": (
                        f"{paired_row.get('Primary_Model')} vs "
                        f"{paired_row.get('Comparator_ID')}"
                    ),
                    "Execution": label,
                    "Model": paired_row.get(model_column),
                    "AUROC": paired_row.get(value_column),
                })
            for label, model_column, value_column in (
                (
                    "Primary",
                    "Primary_Model",
                    "Primary_Runtime_Seconds",
                ),
                (
                    "Comparator",
                    "Comparator_ID",
                    "Comparator_Runtime_Seconds",
                ),
            ):
                paired_runtime_rows.append({
                    "Iteration": paired_row.get("Iteration"),
                    "Pair": (
                        f"{paired_row.get('Primary_Model')} vs "
                        f"{paired_row.get('Comparator_ID')}"
                    ),
                    "Execution": label,
                    "Model": paired_row.get(model_column),
                    "Runtime_Seconds": paired_row.get(value_column),
                })
        paired_auroc = pd.DataFrame(paired_auroc_rows)
        paired_runtime = pd.DataFrame(paired_runtime_rows)
        v14_write_table_bundle(
            paired_auroc,
            plot_data["plot_data_dir"] / "paired_auroc",
        )
        v14_write_table_bundle(
            paired_runtime,
            plot_data["plot_data_dir"] / "paired_runtime",
        )
        written.extend(v14_plot_box(
            paired_auroc,
            "Model",
            "AUROC",
            "Paired budgeted vs auxiliary comparator AUROC",
            "AUROC",
            plot_dir / "paired_budgeted_vs_comparator_auroc",
            plot_config,
            [paired_source],
        ))
        written.extend(v14_plot_box(
            paired_runtime,
            "Model",
            "Runtime_Seconds",
            "Paired budgeted vs auxiliary comparator runtime",
            "Runtime (seconds)",
            plot_dir / "paired_budgeted_vs_comparator_runtime",
            plot_config,
            [paired_source],
        ))
        ratio_field = (
            "Comparator_Runtime_Divided_By_Primary_Runtime"
        )
        if ratio_field in paired.columns:
            runtime_ratio = paired[[
                "Primary_Model",
                "Comparator_ID",
                ratio_field,
            ]].copy()
            runtime_ratio["Pair"] = (
                runtime_ratio["Primary_Model"].astype(str)
                + " vs "
                + runtime_ratio["Comparator_ID"].astype(str)
            )
            v14_write_table_bundle(
                runtime_ratio,
                plot_data["plot_data_dir"] / "paired_runtime_ratio",
            )
            written.extend(v14_plot_box(
                runtime_ratio,
                "Pair",
                ratio_field,
                "Auxiliary comparator runtime divided by primary runtime",
                "Comparator runtime / primary runtime",
                plot_dir / "paired_budgeted_vs_comparator_runtime_ratio",
                plot_config,
                [paired_source],
            ))
    local_source = (
        plot_data["plot_data_dir"] / "energy_local_execution.csv"
    )
    written.extend(v14_plot_box(
        plot_data["local_energy"],
        "Model",
        "Measured_Energy_kWh",
        "Measured local-model energy",
        "kWh",
        plot_dir / "energy_local_execution",
        plot_config,
        [local_source],
    ))
    client_source = (
        plot_data["plot_data_dir"] / "energy_cloud_client_side.csv"
    )
    written.extend(v14_plot_box(
        plot_data["client_energy"],
        "Model",
        "Measured_ClientSide_Energy_kWh",
        "Measured cloud/client local-process energy",
        "kWh (local client process only)",
        plot_dir / "energy_cloud_client_side",
        plot_config,
        [client_source],
    ))
    tradeoff = plot_data["tradeoff"].copy()
    if (
        plt is not None
        and not tradeoff.empty
        and {"AUROC", "Actual_Total_Runtime_Seconds"} <= set(tradeoff.columns)
    ):
        tradeoff["AUROC"] = pd.to_numeric(
            tradeoff["AUROC"], errors="coerce"
        )
        tradeoff["Actual_Total_Runtime_Seconds"] = pd.to_numeric(
            tradeoff["Actual_Total_Runtime_Seconds"], errors="coerce"
        )
        tradeoff = tradeoff.dropna(
            subset=["AUROC", "Actual_Total_Runtime_Seconds"]
        )
        if not tradeoff.empty:
            fig, axis = plt.subplots(figsize=(7, 5))
            for model_name, group in tradeoff.groupby("Model"):
                axis.scatter(
                    group["Actual_Total_Runtime_Seconds"],
                    group["AUROC"],
                    label=model_name,
                    alpha=0.8,
                )
            axis.set_xlabel("Actual total runtime (seconds)")
            axis.set_xscale(
                plot_config.get("runtime_axis_scale", "linear")
            )
            axis.set_ylabel("AUROC")
            axis.set_title("AUROC vs actual runtime")
            axis.grid(alpha=0.25)
            axis.legend()
            written.extend(v14_save_figure(
                fig,
                plot_dir / "tradeoff_auroc_runtime",
                plot_config,
                [plot_data["plot_data_dir"] / "auroc_runtime_tradeoff.csv"],
                "Every point is one persisted model iteration.",
            ))
            plt.close(fig)
            if plot_config.get("save_html", True):
                html = v14_write_plotly_html(
                    tradeoff,
                    plot_dir / "tradeoff_auroc_runtime.html",
                    "scatter",
                    "AUROC vs actual runtime",
                    x="Actual_Total_Runtime_Seconds",
                    y="AUROC",
                    color="Model",
                    hover_data=["Iteration"],
                )
                if html:
                    written.append(html)
    mean_roc = plot_data["roc"]
    if not mean_roc.empty:
        mean_roc = mean_roc[
            mean_roc["Curve_Type"] == "mean_interpolated"
        ]
    if plt is not None and not mean_roc.empty:
        fig, axis = plt.subplots(figsize=(6, 6))
        for model_name, group in mean_roc.groupby("Model"):
            axis.plot(group["FPR"], group["TPR"], label=model_name)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("Mean ROC from persisted predictions")
        axis.legend()
        axis.grid(alpha=0.25)
        written.extend(v14_save_figure(
            fig,
            plot_dir / "roc_mean",
            plot_config,
            [plot_data["plot_data_dir"] / "roc.csv"],
            "Mean ROC interpolated from each persisted iteration ROC.",
        ))
        plt.close(fig)
    calibration = plot_data["calibration"]
    if plt is not None and not calibration.empty:
        fig, axis = plt.subplots(figsize=(6, 6))
        for model_name, group in calibration.groupby("Model"):
            mean_points = group.groupby(
                "Mean_Predicted_Probability", as_index=False
            )["Observed_Fraction_Positive"].mean()
            axis.plot(
                mean_points["Mean_Predicted_Probability"],
                mean_points["Observed_Fraction_Positive"],
                marker="o",
                label=model_name,
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed fraction positive")
        axis.set_title("Calibration from persisted probabilities")
        axis.legend()
        axis.grid(alpha=0.25)
        written.extend(v14_save_figure(
            fig,
            plot_dir / "calibration",
            plot_config,
            [plot_data["plot_data_dir"] / "calibration.csv"],
            "Calibration points derived from persisted probabilities.",
        ))
        plt.close(fig)
    index_frame = pd.DataFrame([
        {
            "path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "sha256": v14_sha256_file(path),
        }
        for path in written if Path(path).exists()
    ])
    v14_atomic_write_csv(plot_dir / "plot_index.csv", index_frame)
    v14_atomic_write_json(
        plot_dir / "plot_index.json",
        index_frame.to_dict(orient="records"),
    )
    return index_frame


def v14_aggregate_sample(sample_dir, expected_iterations, plot_config):
    sample_dir = Path(sample_dir)
    aggregate_dir = sample_dir / "Aggregated"
    data_dir = aggregate_dir / "Data"
    json_dir = aggregate_dir / "JSON"
    table_dir = aggregate_dir / "Tables"
    for path in (data_dir, json_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)
    iteration_frame = v14_collect_iteration_results(sample_dir)
    summary = v14_descriptive_summary(
        iteration_frame, analysis_roles=("core_model",)
    )
    auxiliary_summary = v14_descriptive_summary(
        iteration_frame, analysis_roles=("auxiliary_comparator",)
    )
    paired_iteration = v14_collect_paired_comparisons(sample_dir)
    paired_summary = v14_paired_comparison_summary(paired_iteration)
    v14_write_table_bundle(
        iteration_frame, data_dir / "iteration_level_results"
    )
    v14_write_table_bundle(summary, data_dir / "summary_statistics")
    v14_write_table_bundle(
        auxiliary_summary, data_dir / "auxiliary_comparator_summary"
    )
    v14_write_table_bundle(
        paired_iteration, data_dir / "paired_comparison_iteration_level"
    )
    v14_write_table_bundle(
        paired_summary, data_dir / "paired_comparison_aggregated"
    )
    v14_atomic_write_json(
        json_dir / "aggregation_manifest.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": v14_utc_now(),
            "requested_iterations": int(expected_iterations),
            "iteration_rows": int(len(iteration_frame)),
            "models": sorted(
                iteration_frame["Model"].dropna().unique().tolist()
            ) if not iteration_frame.empty else [],
            "analysis_roles": sorted(
                iteration_frame["Analysis_Role"].dropna().unique().tolist()
            ) if (
                not iteration_frame.empty
                and "Analysis_Role" in iteration_frame.columns
            ) else ["core_model"],
            "paired_comparison_rows": int(len(paired_iteration)),
            "default_summary_filter": "Analysis_Role == 'core_model'",
            "authoritative_source": str(
                data_dir / "iteration_level_results.csv"
            ),
        },
    )
    v14_atomic_write_csv(
        table_dir / "summary_statistics.csv", summary
    )
    reloaded = v14_read_table_bundle(
        data_dir / "iteration_level_results"
    )
    if len(reloaded) != len(iteration_frame):
        raise RuntimeError(
            "Persisted sample aggregation row count changed on reload."
        )
    if plot_config.get("enabled", True):
        v14_generate_sample_plots(sample_dir, plot_config)
    return iteration_frame, summary


def v14_generate_trajectory_plots(
    aggregate_root, iteration_frame, plot_config, scope_label
):
    aggregate_root = Path(aggregate_root)
    plot_data_dir = aggregate_root / "Plot_Data"
    plot_dir = aggregate_root / "Plots"
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows = []
    if not iteration_frame.empty:
        for (
            scenario_name,
            sample_size,
            model_name,
        ), group in iteration_frame.groupby([
            "Scenario", "Sample_Size_Actual", "Model"
        ]):
            for metric in [
                "AUROC",
                "Balanced_Accuracy",
                "Sensitivity",
                "Precision",
                "Brier_Score",
                "Actual_Total_Runtime_Seconds",
                "Measured_Energy_kWh",
                "Measured_ClientSide_Energy_kWh",
                "TabPFN_Context_N",
                "TabPFN_Context_Fraction",
            ]:
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(
                    group[metric], errors="coerce"
                ).dropna()
                if values.empty:
                    continue
                trajectory_rows.append({
                    "Scenario": scenario_name,
                    "Scenario_Role": (
                        str(group["Scenario_Role"].iloc[0])
                        if "Scenario_Role" in group.columns
                        else scenario_name
                    ),
                    "Sample_Size_Actual": int(sample_size),
                    "Model": model_name,
                    "Analysis_Role": (
                        str(group["Analysis_Role"].iloc[0])
                        if "Analysis_Role" in group.columns
                        else "core_model"
                    ),
                    "Base_Model": (
                        str(group["Base_Model"].iloc[0])
                        if "Base_Model" in group.columns
                        else model_name
                    ),
                    "Metric": metric,
                    "Mean": float(values.mean()),
                    "SD": (
                        float(values.std(ddof=1))
                        if len(values) > 1 else 0.0
                    ),
                    "N": int(len(values)),
                })
    trajectory = pd.DataFrame(trajectory_rows)
    v14_write_table_bundle(
        trajectory, plot_data_dir / "sample_size_trajectories"
    )
    source = plot_data_dir / "sample_size_trajectories.csv"
    if plt is not None and not trajectory.empty:
        for metric in sorted(trajectory["Metric"].unique()):
            subset = trajectory[trajectory["Metric"] == metric]
            fig, axis = plt.subplots(figsize=(8, 5))
            for keys, group in subset.groupby(["Scenario", "Model"]):
                group = group.sort_values("Sample_Size_Actual")
                label = " / ".join(map(str, keys))
                axis.plot(
                    group["Sample_Size_Actual"],
                    group["Mean"],
                    marker="o",
                    label=label,
                )
            axis.set_xlabel("Actual sample size")
            axis.set_ylabel(metric.replace("_", " "))
            if (
                metric == "Actual_Total_Runtime_Seconds"
                and plot_config.get("runtime_axis_scale", "linear")
                == "log"
            ):
                axis.set_yscale("log")
            axis.set_title(
                f"{metric.replace('_', ' ')} across sample sizes"
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            v14_save_figure(
                fig,
                plot_dir / f"trajectory_{metric.lower()}",
                plot_config,
                [source],
                f"{scope_label} trajectory from persisted iteration values.",
            )
            plt.close(fig)
            if plot_config.get("save_html", True):
                v14_write_plotly_html(
                    subset,
                    plot_dir / f"trajectory_{metric.lower()}.html",
                    "line",
                    f"{metric.replace('_', ' ')} across sample sizes",
                    x="Sample_Size_Actual",
                    y="Mean",
                    color="Model",
                    line_dash=(
                        "Scenario"
                        if subset["Scenario"].nunique() > 1 else None
                    ),
                )
    selected_sample = plot_config.get("cross_scenario_runtime_sample")
    if (
        selected_sample is not None
        and not iteration_frame.empty
        and "Actual_Total_Runtime_Seconds" in iteration_frame.columns
    ):
        available_samples = sorted(
            pd.to_numeric(
                iteration_frame["Sample_Size_Actual"], errors="coerce"
            ).dropna().astype(int).unique().tolist()
        )
        if not available_samples:
            return trajectory
        if str(selected_sample).lower() in {"max", "largest"}:
            resolved_sample = max(available_samples)
        elif str(selected_sample).lower() in {"min", "smallest"}:
            resolved_sample = min(available_samples)
        else:
            resolved_sample = int(selected_sample)
        selected_runtime = iteration_frame[
            pd.to_numeric(
                iteration_frame["Sample_Size_Actual"], errors="coerce"
            ) == resolved_sample
        ].copy()
        v14_write_table_bundle(
            selected_runtime,
            plot_data_dir
            / "cross_scenario_runtime_selected_sample",
        )
        v14_plot_box(
            selected_runtime,
            "Model",
            "Actual_Total_Runtime_Seconds",
            (
                "Cross-scenario runtime at selected sample size "
                f"{resolved_sample}"
            ),
            "Actual total runtime (seconds)",
            plot_dir / "cross_scenario_runtime_selected_sample",
            {
                **plot_config,
                "_filters": {
                    "Sample_Size_Actual": resolved_sample,
                },
                "_runtime_field": "Actual_Total_Runtime_Seconds",
            },
            [
                plot_data_dir
                / "cross_scenario_runtime_selected_sample.csv"
            ],
        )
    return trajectory


def v14_aggregate_scenario(
    scenario_dir, plot_config, scenario_name
):
    scenario_dir = Path(scenario_dir)
    frames = []
    summaries = []
    paired_frames = []
    for sample_dir in sorted(scenario_dir.glob("N_*")):
        data_dir = sample_dir / "Aggregated" / "Data"
        if (data_dir / "iteration_level_results.csv").exists():
            frames.append(v14_read_table_bundle(
                data_dir / "iteration_level_results"
            ))
        if (data_dir / "summary_statistics.csv").exists():
            current = v14_read_table_bundle(
                data_dir / "summary_statistics"
            ).copy()
            sample_manifest = v14_load_json_file(
                sample_dir / "sample_manifest.json"
            )
            current = current.assign(
                Scenario=scenario_name,
                Sample_Size_Requested=sample_manifest[
                    "requested_sample_size"
                ],
                Sample_Size_Actual=sample_manifest[
                    "actual_sample_size"
                ],
            )
            summaries.append(current)
        if (
            data_dir / "paired_comparison_iteration_level.csv"
        ).exists():
            paired_frames.append(v14_read_table_bundle(
                data_dir / "paired_comparison_iteration_level"
            ))
    iteration_frame = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    summary_frame = (
        pd.concat(summaries, ignore_index=True)
        if summaries else pd.DataFrame()
    )
    paired_frame = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames else pd.DataFrame()
    )
    paired_summary = v14_paired_comparison_summary(paired_frame)
    aggregate = scenario_dir / "Scenario_Aggregated"
    for folder in ("Data", "JSON", "Tables", "Plots", "Plot_Data"):
        (aggregate / folder).mkdir(parents=True, exist_ok=True)
    v14_write_table_bundle(
        iteration_frame,
        aggregate / "Data" / "all_sample_sizes_iteration_level",
    )
    v14_write_table_bundle(
        summary_frame,
        aggregate / "Data" / "all_sample_sizes_summary",
    )
    v14_write_table_bundle(
        paired_frame,
        aggregate / "Data" / "paired_comparison_iteration_level",
    )
    v14_write_table_bundle(
        paired_summary,
        aggregate / "Data" / "paired_comparison_aggregated",
    )
    v14_atomic_write_json(
        aggregate / "JSON" / "scenario_aggregation_manifest.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "scenario": scenario_name,
            "created_at_utc": v14_utc_now(),
            "iteration_rows": int(len(iteration_frame)),
            "sample_sizes": sorted(
                pd.to_numeric(
                    iteration_frame.get(
                        "Sample_Size_Actual", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                ).dropna().astype(int).unique().tolist()
            ),
        },
    )
    if plot_config.get("enabled", True):
        v14_generate_trajectory_plots(
            aggregate,
            v14_read_table_bundle(
                aggregate
                / "Data"
                / "all_sample_sizes_iteration_level"
            ),
            plot_config,
            f"Scenario {scenario_name}",
        )
    return iteration_frame, summary_frame


def v14_aggregate_global(run_dir, plot_config):
    run_dir = Path(run_dir)
    iteration_frames = []
    summary_frames = []
    paired_frames = []
    for scenario_dir in sorted(run_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        data_dir = scenario_dir / "Scenario_Aggregated" / "Data"
        iteration_path = data_dir / "all_sample_sizes_iteration_level.csv"
        summary_path = data_dir / "all_sample_sizes_summary.csv"
        paired_path = data_dir / "paired_comparison_iteration_level.csv"
        if iteration_path.exists():
            iteration_frames.append(v14_read_table_bundle(
                data_dir / "all_sample_sizes_iteration_level"
            ))
        if summary_path.exists():
            summary_frames.append(v14_read_table_bundle(
                data_dir / "all_sample_sizes_summary"
            ))
        if paired_path.exists():
            paired_frames.append(v14_read_table_bundle(
                data_dir / "paired_comparison_iteration_level"
            ))
    iteration_frame = (
        pd.concat(iteration_frames, ignore_index=True)
        if iteration_frames else pd.DataFrame()
    )
    summary_frame = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames else pd.DataFrame()
    )
    paired_frame = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames else pd.DataFrame()
    )
    paired_summary = v14_paired_comparison_summary(paired_frame)
    aggregate = run_dir / "Global_Aggregated"
    for folder in ("Data", "JSON", "Tables", "Plots", "Plot_Data"):
        (aggregate / folder).mkdir(parents=True, exist_ok=True)
    v14_write_table_bundle(
        iteration_frame,
        aggregate / "Data" / "master_iteration_level_results",
    )
    v14_write_table_bundle(
        summary_frame,
        aggregate / "Data" / "master_summary",
    )
    v14_write_table_bundle(
        paired_frame,
        aggregate / "Data" / "paired_comparison_iteration_level",
    )
    v14_write_table_bundle(
        paired_summary,
        aggregate / "Data" / "paired_comparison_aggregated",
    )
    if plot_config.get("enabled", True):
        v14_generate_trajectory_plots(
            aggregate,
            v14_read_table_bundle(
                aggregate / "Data" / "master_iteration_level_results"
            ),
            plot_config,
            "Global",
        )
    return iteration_frame, summary_frame


def v14_artifact_type(path):
    name = path.name.lower()
    if "manifest" in name:
        return "manifest"
    if "config" in name:
        return "configuration"
    if "prediction" in name:
        return "prediction_evidence"
    if "split" in name or name.endswith("_indices.npy"):
        return "split_evidence"
    if "optuna" in name or "selected_trial" in name:
        return "hyperparameter_evidence"
    if "energy" in name:
        return "energy_evidence"
    if "runtime" in name:
        return "runtime_evidence"
    if path.suffix.lower() in {".png", ".svg", ".html"}:
        return "figure"
    if "plot_data" in {part.lower() for part in path.parts}:
        return "plot_data"
    if "summary" in name or "iteration_level" in name:
        return "aggregate"
    return "artifact"


def v14_parse_artifact_scope(relative_path):
    parts = relative_path.parts
    scenario = None
    sample = None
    iteration = None
    model = None
    for index, part in enumerate(parts):
        if part.startswith("N_"):
            sample = part
            if index > 0:
                scenario = parts[index - 1]
        if part.startswith("Iteration_"):
            iteration = part
        if part == "Models" and index + 1 < len(parts):
            model = parts[index + 1]
    return scenario, sample, iteration, model


def v14_build_artifact_index(run_dir):
    run_dir = Path(run_dir)
    rows = []
    excluded = {
        "artifact_index.json",
        "artifact_index.csv",
    }
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(run_dir)
        scenario, sample, iteration, model = v14_parse_artifact_scope(
            relative
        )
        provenance = []
        if path.suffix.lower() in {".png", ".svg", ".html"}:
            provenance_path = Path(str(path.with_suffix("")) + ".provenance.json")
            if provenance_path.exists():
                try:
                    provenance = v14_load_json_file(
                        provenance_path
                    ).get("source_artifacts", [])
                except Exception:
                    provenance = []
        rows.append({
            "path": str(relative).replace("\\", "/"),
            "type": v14_artifact_type(path),
            "scenario": scenario,
            "sample_size": sample,
            "iteration": iteration,
            "model": model,
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": int(path.stat().st_size),
            "SHA256": v14_sha256_file(path),
            "creation_time_utc": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "provenance_source_artifacts": json.dumps(provenance),
        })
    frame = pd.DataFrame(rows)
    v14_atomic_write_csv(run_dir / "artifact_index.csv", frame)
    v14_atomic_write_json(
        run_dir / "artifact_index.json",
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": v14_utc_now(),
            "artifacts": frame.to_dict(orient="records"),
        },
    )
    return frame


# =============================================================================
# 11. Run validation, plot regeneration, and resume support
# =============================================================================

def v14_validate_run(run_dir, write_report=True):
    run_dir = Path(run_dir).resolve()
    issues = []
    checks = []
    core_model_completeness = []

    def check(name, passed, detail=""):
        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })
        if not passed:
            issues.append(f"{name}: {detail}")

    manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "resolved_master_config.json"
    matrix_path = run_dir / "experiment_matrix.csv"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    check("run_manifest_exists", manifest_path.exists(), manifest_path)
    check("resolved_config_exists", config_path.exists(), config_path)
    check("experiment_matrix_exists", matrix_path.exists(), matrix_path)
    check(
        "dataset_manifest_exists",
        dataset_manifest_path.exists(),
        dataset_manifest_path,
    )
    if not (
        manifest_path.exists()
        and config_path.exists()
        and matrix_path.exists()
        and dataset_manifest_path.exists()
    ):
        report = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "status": "FAIL",
            "checks": checks,
            "issues": issues,
        }
        if write_report:
            v14_atomic_write_json(
                run_dir / "validation_report.json", report
            )
        return report
    manifest = v14_load_json_file(manifest_path)
    config = v14_load_json_file(config_path)
    dataset_manifest = v14_load_json_file(dataset_manifest_path)
    matrix = pd.read_csv(matrix_path)
    loader_evidence = dataset_manifest.get("dataset_loader", {})
    loader_path_value = loader_evidence.get("resolved_path")
    loader_path = (
        Path(loader_path_value)
        if loader_path_value else None
    )
    loader_exists = bool(
        loader_path is not None and loader_path.is_file()
    )
    check(
        "dataset_loader_file_exists",
        loader_exists,
        loader_path_value,
    )
    check(
        "dataset_loader_required_functions_recorded",
        (
            loader_evidence.get("required_functions_present") is True
            and set(loader_evidence.get("required_functions", []))
            >= set(V14_REQUIRED_DATASET_LOADER_FUNCTIONS)
        ),
        loader_evidence,
    )
    if loader_exists:
        check(
            "dataset_loader_checksum",
            (
                loader_evidence.get("sha256")
                == v14_sha256_file(loader_path)
                == manifest.get("dataset_loader_sha256")
            ),
            "dataset-loader SHA256 must match dataset and run manifests",
        )
        try:
            _, current_loader = v14_resolve_dataset_loader(config)
        except Exception as exc:
            check("dataset_loader_import", False, exc)
        else:
            check(
                "dataset_loader_import",
                current_loader.get("import_validated") is True,
                current_loader,
            )
            check(
                "dataset_loader_runtime_checksum",
                current_loader.get("sha256")
                == loader_evidence.get("sha256"),
                "resolved loader differs from the recorded loader",
            )
    matrix_cells = matrix.drop_duplicates(
        subset=[
            "Scenario",
            "Sample_Size_Actual",
            "Iteration",
        ]
    )
    for _, matrix_row in matrix_cells.iterrows():
        scenario = scenario_folder_name(matrix_row["Scenario"])
        sample_requested = matrix_row["Sample_Size_Requested"]
        if str(sample_requested).lower() == "full":
            sample_folder = "N_FULL"
        else:
            sample_folder = f"N_{int(matrix_row['Sample_Size_Actual'])}"
        iteration_dir = (
            run_dir
            / scenario
            / sample_folder
            / f"Iteration_{int(matrix_row['Iteration']):03d}"
        )
        marker_path = iteration_dir / "_SUCCESS.json"
        check(
            f"completion_marker:{scenario}:{sample_folder}:{int(matrix_row['Iteration'])}",
            marker_path.exists(),
            marker_path,
        )
        if not marker_path.exists():
            continue
        marker = v14_load_json_file(marker_path)
        scenario_resolved = v14_resolve_scenario(
            config, str(matrix_row["Scenario"])
        )
        check(
            f"completion_checksums:{iteration_dir}",
            v14_validate_completion_marker(iteration_dir, marker),
            "marker checksums must match",
        )
        train_path = iteration_dir / "Splits" / "train_indices.npy"
        test_path = iteration_dir / "Splits" / "test_indices.npy"
        if train_path.exists() and test_path.exists():
            train_values = np.load(train_path, allow_pickle=False)
            test_values = np.load(test_path, allow_pickle=False)
            check(
                f"split_row_overlap:{iteration_dir}",
                len(np.intersect1d(train_values, test_values)) == 0,
                "train/test row indices must be disjoint",
            )
        execution_dirs = list(
            sorted((iteration_dir / "Models").glob("*"))
        ) + list(
            sorted((iteration_dir / "Comparators").glob("*"))
        )
        for model_dir in execution_dirs:
            if not model_dir.is_dir():
                continue
            metrics_path = model_dir / "metrics.json"
            runtime_path = model_dir / "runtime_breakdown.json"
            energy_path = model_dir / "energy_breakdown.json"
            config_model_path = model_dir / "final_model_config.json"
            for required in V14_REQUIRED_MODEL_FILES:
                check(
                    f"model_required:{model_dir.name}:{required}:{iteration_dir}",
                    (model_dir / required).exists(),
                    model_dir / required,
                )
            if not metrics_path.exists():
                continue
            metrics = v14_load_json_file(metrics_path)
            runtime = (
                v14_load_json_file(runtime_path)
                if runtime_path.exists() else {}
            )
            if metrics.get("success", False):
                prediction_path = model_dir / "predictions.npz"
                check(
                    f"prediction_exists:{model_dir}",
                    prediction_path.exists(),
                    prediction_path,
                )
                if prediction_path.exists():
                    with np.load(
                        prediction_path, allow_pickle=False
                    ) as payload:
                        recomputed = v14_recompute_metrics(
                            payload["y_true"],
                            payload["probability_class_1"],
                        )
                        length_ok = (
                            not test_path.exists()
                            or len(payload["y_true"])
                            == len(np.load(test_path, allow_pickle=False))
                        )
                        check(
                            f"prediction_length:{model_dir}",
                            length_ok,
                            "prediction/test lengths",
                        )
                        for metric in V14_PREDICTION_METRICS:
                            stored = metrics.get(metric)
                            passed = (
                                stored is not None
                                and abs(
                                    float(stored)
                                    - float(recomputed[metric])
                                ) <= 1e-10
                            )
                            check(
                                f"metric_consistency:{metric}:{model_dir}",
                                passed,
                                f"stored={stored}, recomputed={recomputed[metric]}",
                            )
                if (
                    metrics.get("Execution_Path") == "local"
                    and metrics.get("Resolved_Device") == "cpu"
                ):
                    available_cpu = metrics.get("CPU_Cores_Available")
                    configured_threads = metrics.get(
                        "CPU_Threads_Configured"
                    )
                    check(
                        f"cpu_capacity_recorded:{model_dir}",
                        (
                            available_cpu is not None
                            and int(available_cpu) > 0
                            and configured_threads is not None
                            and int(configured_threads) > 0
                        ),
                        "local CPU executions require detected capacity and threads",
                    )
                    if (
                        available_cpu is not None
                        and configured_threads is not None
                    ):
                        check(
                            f"cpu_no_oversubscription:{model_dir}",
                            int(configured_threads) <= int(available_cpu),
                            "configured threads exceed process-available CPUs",
                        )
                    monitoring_enabled = bool(
                        config.get("cpu_monitoring", {}).get(
                            "enabled", True
                        )
                    )
                    check(
                        f"cpu_monitoring_evidence:{model_dir}",
                        (
                            not monitoring_enabled
                            or bool(
                                metrics.get(
                                    "CPU_Monitoring_Evidence_Available",
                                    False,
                                )
                            )
                        ),
                        metrics.get(
                            "CPU_Monitoring_Unavailable_Reason", ""
                        ),
                    )
                    check(
                        f"optuna_sequential:{model_dir}",
                        metrics.get("Optuna_Trial_N_Jobs") == 1,
                        "Optuna must remain n_jobs=1",
                    )
                    check(
                        f"monte_carlo_sequential:{model_dir}",
                        metrics.get("Monte_Carlo_Execution")
                        == "sequential",
                        "Monte Carlo iterations must remain sequential",
                    )
                    if monitoring_enabled:
                        overhead_text = str(
                            metrics.get(
                                "CPU_Monitoring_Overhead_Handling", ""
                            )
                        ).lower()
                        check(
                            f"cpu_monitoring_overhead_metadata:{model_dir}",
                            (
                                metrics.get(
                                    "CPU_Monitoring_Excluded_From_Runtime"
                                ) is False
                                and "concurrently" in overhead_text
                                and "may be included" in overhead_text
                            ),
                            "active monitoring must disclose possible "
                            "lightweight overhead in observed wall-clock time",
                        )
                        check(
                            f"cpu_monitoring_budget_boundary:{model_dir}",
                            str(
                                metrics.get(
                                    "CPU_Monitoring_Budget_Boundary_Role",
                                    "",
                                )
                            ).lower().startswith("none"),
                            "CPU monitoring must not define a scientific "
                            "runtime-budget boundary",
                        )
                base_model = str(
                    metrics.get("Base_Model", metrics.get("Model", ""))
                )
                execution_model_cfg = scenario_resolved[
                    "resolved_models"
                ].get(base_model, {})
                is_tuned_core = bool(
                    metrics.get("Analysis_Role") == "core_model"
                    and execution_model_cfg.get(
                        "tuned_by_optuna", False
                    )
                )
                if is_tuned_core:
                    required_runtime_fields = [
                        "Actual_Optuna_Tuning_Time_Seconds",
                        "Final_Fit_Predict_Time_Seconds",
                        "Actual_Total_Runtime_Seconds",
                        "Budget_Accounted_Runtime_Seconds",
                        "HPO_Start_Perf_Counter_Seconds",
                        "HPO_End_Perf_Counter_Seconds",
                        "Final_Fit_Start_UTC",
                        "Final_Fit_End_UTC",
                        "Prediction_Start_UTC",
                        "Prediction_End_UTC",
                    ]
                    if bool(
                        scenario_resolved.get("budgeting", {}).get(
                            "enabled", True
                        )
                    ):
                        required_runtime_fields.append(
                            "Reference_Budget_Seconds"
                        )
                    missing_runtime_fields = [
                        field for field in required_runtime_fields
                        if runtime.get(field) is None
                    ]
                    check(
                        f"tuned_runtime_quantities_separate:{model_dir}",
                        not missing_runtime_fields,
                        f"missing={missing_runtime_fields}",
                    )
                    hpo_start_value = v14_scalar(
                        runtime.get(
                            "HPO_Start_Perf_Counter_Seconds"
                        )
                    )
                    hpo_end_value = v14_scalar(
                        runtime.get(
                            "HPO_End_Perf_Counter_Seconds"
                        )
                    )
                    optuna_value = v14_scalar(
                        runtime.get(
                            "Actual_Optuna_Tuning_Time_Seconds"
                        )
                    )
                    check(
                        f"tuned_hpo_duration_auditable:{model_dir}",
                        (
                            hpo_start_value is not None
                            and hpo_end_value is not None
                            and optuna_value is not None
                            and bool(
                                np.isclose(
                                    float(hpo_end_value)
                                    - float(hpo_start_value),
                                    float(optuna_value),
                                    rtol=1e-9,
                                    atol=1e-9,
                                )
                            )
                        ),
                        "dedicated HPO timer boundaries must reproduce "
                        "Actual_Optuna_Tuning_Time_Seconds",
                    )
                is_core_reference = (
                    metrics.get("Analysis_Role") == "core_model"
                    and bool(
                        scenario_resolved.get("budgeting", {}).get(
                            "enabled", True
                        )
                    )
                    and base_model
                    == str(
                        scenario_resolved.get(
                            "budget_reference_model"
                        )
                    )
                )
                if is_core_reference:
                    tuned_reference = bool(
                        execution_model_cfg.get(
                            "tuned_by_optuna", False
                        )
                    )
                    reference_budget = v14_scalar(
                        runtime.get("Reference_Budget_Seconds")
                    )
                    actual_total = v14_scalar(
                        runtime.get("Actual_Total_Runtime_Seconds")
                    )
                    if tuned_reference:
                        actual_optuna = v14_scalar(
                            runtime.get(
                                "Actual_Optuna_Tuning_Time_Seconds"
                            )
                        )
                        check(
                            f"tuned_reference_budget_basis:{model_dir}",
                            (
                                runtime.get("Budget_Basis")
                                == "reference_tuning_runtime"
                                and runtime.get(
                                    "Reference_Budget_Source_Field"
                                )
                                == "Actual_Optuna_Tuning_Time_Seconds"
                            ),
                            runtime,
                        )
                        check(
                            f"tuned_reference_budget_equals_hpo:{model_dir}",
                            (
                                reference_budget is not None
                                and actual_optuna is not None
                                and bool(
                                    np.isclose(
                                        float(reference_budget),
                                        float(actual_optuna),
                                        rtol=1e-9,
                                        atol=1e-9,
                                    )
                                )
                            ),
                            (
                                f"reference={reference_budget}; "
                                f"Optuna={actual_optuna}; "
                                f"total={actual_total}"
                            ),
                        )
                        hpo_start = v14_scalar(
                            runtime.get(
                                "HPO_Start_Perf_Counter_Seconds"
                            )
                        )
                        hpo_end = v14_scalar(
                            runtime.get(
                                "HPO_End_Perf_Counter_Seconds"
                            )
                        )
                        check(
                            f"tuned_reference_hpo_timer_boundary:{model_dir}",
                            (
                                hpo_start is not None
                                and hpo_end is not None
                                and actual_optuna is not None
                                and bool(
                                    np.isclose(
                                        float(hpo_end)
                                        - float(hpo_start),
                                        float(actual_optuna),
                                        rtol=1e-9,
                                        atol=1e-9,
                                    )
                                )
                            ),
                            "HPO_End - HPO_Start must equal the dedicated "
                            "Optuna tuning duration",
                        )
                        check(
                            f"tuned_reference_final_fit_excluded:{model_dir}",
                            (
                                runtime.get(
                                    "Reference_Budget_Source_Field"
                                )
                                == "Actual_Optuna_Tuning_Time_Seconds"
                                and runtime.get(
                                    "Final_Fit_Predict_Time_Seconds"
                                )
                                is not None
                            ),
                            "final fit/prediction must be separately recorded "
                            "and must not source the reference HPO budget",
                        )
                    else:
                        check(
                            f"non_tuned_reference_budget_basis:{model_dir}",
                            (
                                runtime.get("Budget_Basis")
                                == "reference_execution_runtime"
                                and runtime.get(
                                    "Reference_Budget_Source_Field"
                                )
                                == "Actual_Total_Runtime_Seconds"
                            ),
                            runtime,
                        )
                        check(
                            f"non_tuned_reference_budget_equals_execution:{model_dir}",
                            (
                                reference_budget is not None
                                and actual_total is not None
                                and bool(
                                    np.isclose(
                                        float(reference_budget),
                                        float(actual_total),
                                        rtol=1e-9,
                                        atol=1e-9,
                                    )
                                )
                            ),
                            (
                                f"reference={reference_budget}; "
                                f"execution={actual_total}"
                            ),
                        )
            if (
                energy_path.exists()
                and metrics.get("Base_Model") == "TabPFN"
            ):
                energy = v14_load_json_file(energy_path)
                if energy.get("Execution_Path") == "cloud":
                    check(
                        f"cloud_energy_scope:{model_dir}",
                        (
                            energy.get("Energy_Scope")
                            == "local_client_process_only"
                            and energy.get(
                                "Remote_Server_Energy_Measured"
                            ) is False
                            and energy.get("Remote_Server_Energy_kWh")
                            is None
                        ),
                        "cloud energy must remain client-side only",
                    )
            if metrics_path.exists() and config_model_path.exists():
                model_config = v14_load_json_file(config_model_path)
                check(
                    f"device_consistency:{model_dir}",
                    metrics.get("Resolved_Device")
                    == model_config.get("resolved_device"),
                    "metrics and final model config devices differ",
                )
            context_path = model_dir / "tabpfn_context_search.json"
            if context_path.exists():
                context = v14_load_json_file(context_path)
                if context.get("context_strategy") == "fixed":
                    check(
                        f"fixed_context_single_candidate:{model_dir}",
                        int(context.get("candidate_count") or 0) == 1,
                        "fixed context must execute exactly one candidate",
                    )
                if context.get("context_strategy") == "full":
                    check(
                        f"full_context_uses_outer_train:{model_dir}",
                        int(context.get("actual_context_n") or -1)
                        == int(context.get("full_training_n") or -2),
                        "full context must use the complete outer train partition",
                    )
        paired_path = (
            iteration_dir / "paired_comparison_iteration_level.csv"
        )
        if paired_path.exists():
            paired = pd.read_csv(paired_path)
            if not paired.empty:
                exact = paired["Exact_Pair_Valid"].astype(
                    str
                ).str.lower().isin({"true", "1", "yes"})
                check(
                    f"comparator_exact_pairing:{iteration_dir}",
                    bool(exact.all()),
                    "all comparator pairs require identical split/test evidence",
                )
                direction_pairs = [
                    (
                        f"Paired_Difference_{metric}_Primary_Minus_Comparator",
                        f"Paired_Difference_{metric}_Comparator_Minus_Primary",
                    )
                    for metric in V14_PREDICTION_METRICS
                ] + [(
                    "Paired_Difference_Runtime_Primary_Minus_Comparator",
                    "Paired_Difference_Runtime_Comparator_Minus_Primary",
                )]
                for primary_minus, comparator_minus in direction_pairs:
                    columns_present = (
                        primary_minus in paired.columns
                        and comparator_minus in paired.columns
                    )
                    inverse_ok = False
                    if columns_present:
                        first = pd.to_numeric(
                            paired[primary_minus], errors="coerce"
                        )
                        second = pd.to_numeric(
                            paired[comparator_minus], errors="coerce"
                        )
                        comparable = first.notna() & second.notna()
                        inverse_ok = bool(
                            comparable.any()
                            and np.allclose(
                                first[comparable].to_numpy(),
                                -second[comparable].to_numpy(),
                                rtol=1e-12,
                                atol=1e-12,
                            )
                        )
                    check(
                        f"paired_difference_directions:{primary_minus}:{iteration_dir}",
                        columns_present and inverse_ok,
                        "both saved directions must be exact additive inverses",
                    )
                ratio_columns = {
                    "Comparator_Runtime_Divided_By_Primary_Runtime",
                    "Primary_Runtime_Divided_By_Comparator_Runtime",
                }
                check(
                    f"paired_runtime_ratio_directions:{iteration_dir}",
                    ratio_columns.issubset(set(paired.columns)),
                    sorted(ratio_columns),
                )
        iteration_results_path = (
            iteration_dir / "iteration_model_results.csv"
        )
        if iteration_results_path.exists():
            iteration_results = pd.read_csv(iteration_results_path)
            if "Analysis_Role" in iteration_results.columns:
                core_rows = iteration_results[
                    iteration_results["Analysis_Role"] == "core_model"
                ].copy()
                expected_rows = matrix[
                    (matrix["Scenario"].astype(str)
                     == str(matrix_row["Scenario"]))
                    & (
                        pd.to_numeric(
                            matrix["Sample_Size_Actual"],
                            errors="coerce",
                        )
                        == float(matrix_row["Sample_Size_Actual"])
                    )
                    & (
                        pd.to_numeric(
                            matrix["Iteration"], errors="coerce"
                        )
                        == int(matrix_row["Iteration"])
                    )
                    & (
                        matrix["Analysis_Role"].astype(str)
                        == "core_model"
                    )
                ]
                expected_core = set(
                    expected_rows["Base_Model"].dropna().astype(str)
                )
                actual_core = set(
                    core_rows["Base_Model"].dropna().astype(str)
                )
                success_mask = core_rows.get(
                    "success",
                    pd.Series(False, index=core_rows.index),
                ).astype(str).str.lower().isin(
                    {"true", "1", "yes"}
                )
                successful_core = set(
                    core_rows.loc[
                        success_mask, "Base_Model"
                    ].dropna().astype(str)
                )
                failed_core = actual_core - successful_core
                missing_core = expected_core - actual_core
                unexpected_core = actual_core - expected_core
                detail = {
                    "Scenario": str(matrix_row["Scenario"]),
                    "Sample_Size_Actual": int(
                        matrix_row["Sample_Size_Actual"]
                    ),
                    "Iteration": int(matrix_row["Iteration"]),
                    "Expected_Core_Model_Set": sorted(expected_core),
                    "Actual_Recorded_Core_Model_Set": sorted(actual_core),
                    "Successful_Core_Models": sorted(successful_core),
                    "Failed_Core_Models": sorted(failed_core),
                    "Missing_Core_Models": sorted(missing_core),
                    "Unexpected_Core_Models": sorted(unexpected_core),
                }
                core_model_completeness.append(detail)
                check(
                    f"core_model_set_exact:{iteration_dir}",
                    (
                        actual_core == expected_core
                        and not missing_core
                        and not unexpected_core
                    ),
                    json.dumps(detail, sort_keys=True),
                )
                failure_policy = str(
                    config.get("execution", {}).get(
                        "on_model_error", "continue"
                    )
                ).lower()
                check(
                    f"core_model_failure_policy:{iteration_dir}",
                    not failed_core or failure_policy == "continue",
                    (
                        f"failure_policy={failure_policy}; "
                        f"failed={sorted(failed_core)}"
                    ),
                )
            else:
                check(
                    f"core_analysis_role_column:{iteration_dir}",
                    False,
                    "iteration results lack Analysis_Role",
                )
        else:
            expected_core_count = int(
                (
                    matrix[
                        (matrix["Scenario"].astype(str)
                         == str(matrix_row["Scenario"]))
                        & (
                            pd.to_numeric(
                                matrix["Sample_Size_Actual"],
                                errors="coerce",
                            )
                            == float(matrix_row["Sample_Size_Actual"])
                        )
                        & (
                            pd.to_numeric(
                                matrix["Iteration"], errors="coerce"
                            )
                            == int(matrix_row["Iteration"])
                        )
                        & (
                            matrix["Analysis_Role"].astype(str)
                            == "core_model"
                        )
                    ]
                ).shape[0]
            )
            check(
                f"core_iteration_results_exists:{iteration_dir}",
                expected_core_count == 0,
                iteration_results_path,
            )
    index_path = run_dir / "artifact_index.json"
    if index_path.exists():
        artifact_index = v14_load_json_file(index_path).get(
            "artifacts", []
        )
        for entry in artifact_index:
            path = run_dir / entry["path"]
            if path.name in {
                "validation_report.json",
                "validation_report.txt",
            }:
                continue
            check(
                f"artifact_checksum:{entry['path']}",
                path.exists()
                and v14_sha256_file(path) == entry["SHA256"],
                "artifact missing or checksum mismatch",
            )
    for provenance_path in run_dir.rglob("*.provenance.json"):
        provenance = v14_load_json_file(provenance_path)
        for source, expected in provenance.get(
            "source_checksums", {}
        ).items():
            source_path = Path(source)
            check(
                f"plot_provenance:{provenance_path.name}:{source}",
                source_path.exists()
                and v14_sha256_file(source_path) == expected,
                "plot source missing or changed",
            )
    report = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "experiment_id": manifest.get("experiment_id"),
        "config_hash": manifest.get("config_hash"),
        "validated_at_utc": v14_utc_now(),
        "status": "PASS" if not issues else "FAIL",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_failed": sum(not item["passed"] for item in checks),
        "checks": checks,
        "core_model_completeness": core_model_completeness,
        "issues": issues,
    }
    if write_report:
        v14_atomic_write_json(
            run_dir / "validation_report.json", report
        )
        lines = [
            f"V14 RUN VALIDATION: {report['status']}",
            f"Run: {run_dir}",
            f"Passed checks: {report['checks_passed']}",
            f"Failed checks: {report['checks_failed']}",
        ]
        if issues:
            lines.append("")
            lines.append("Issues:")
            lines.extend(f"- {issue}" for issue in issues)
        v14_atomic_write_text(
            run_dir / "validation_report.txt",
            "\n".join(lines) + "\n",
        )
    return report


def v14_regenerate_plots(
    run_dir, scenario_filter=None, sample_filter=None
):
    run_dir = Path(run_dir).resolve()
    config = v14_load_json_file(
        run_dir / "resolved_master_config.json"
    )
    plots = copy.deepcopy(config.get("plots", {}))
    plots["_config_hash"] = config.get("config_hash") or v14_config_hash(
        config
    )
    selected_scenarios = []
    for scenario_dir in sorted(run_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if not (scenario_dir / "scenario_manifest.json").exists():
            continue
        manifest = v14_load_json_file(
            scenario_dir / "scenario_manifest.json"
        )
        scenario_name = manifest["scenario"]
        if scenario_filter and scenario_name != scenario_filter:
            continue
        selected_scenarios.append(scenario_dir)
        for sample_dir in sorted(scenario_dir.glob("N_*")):
            sample_manifest_path = sample_dir / "sample_manifest.json"
            if not sample_manifest_path.exists():
                continue
            sample_manifest = v14_load_json_file(sample_manifest_path)
            if sample_filter is not None:
                requested = str(
                    sample_manifest["requested_sample_size"]
                ).lower()
                actual = str(sample_manifest["actual_sample_size"])
                if str(sample_filter).lower() not in {requested, actual}:
                    continue
            v14_generate_sample_plots(sample_dir, plots)
        v14_aggregate_scenario(
            scenario_dir, plots, scenario_name
        )
    if not scenario_filter and sample_filter is None:
        v14_aggregate_global(run_dir, plots)
    v14_build_artifact_index(run_dir)
    return {
        "run_dir": str(run_dir),
        "scenarios": [path.name for path in selected_scenarios],
        "regenerated_at_utc": v14_utc_now(),
    }


def v14_prepare_run_directory(
    config, config_hash_value, resume=False, run_dir=None
):
    if resume:
        selected = (
            Path(run_dir).expanduser().resolve()
            if run_dir else v14_find_resume_dir(config, config_hash_value)
        )
        if not selected.exists():
            raise FileNotFoundError(f"Resume run directory not found: {selected}")
        manifest_path = selected / "run_manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                f"Resume directory lacks run_manifest.json: {selected}"
            )
        manifest = v14_load_json_file(manifest_path)
        if manifest.get("config_hash") != config_hash_value:
            raise ValueError(
                "Resume config hash mismatch: stored "
                f"{manifest.get('config_hash')} versus resolved {config_hash_value}."
            )
        return selected, manifest["experiment_id"], True
    root = Path(config["output_root"]).expanduser().resolve()
    run_id = v14_make_run_id(config, config_hash_value)
    selected = root / run_id
    if selected.exists():
        selected = root / f"{run_id}_{uuid.uuid4().hex[:6]}"
    selected.mkdir(parents=True, exist_ok=False)
    return selected, selected.name, False


def v14_write_root_manifests(
    run_dir,
    experiment_id,
    config_hash_value,
    config,
    environment,
    dataset_manifest,
    matrix,
    resumed,
    dry_run,
):
    run_dir = Path(run_dir)
    resolved_config = copy.deepcopy(config)
    resolved_config["config_hash"] = config_hash_value
    resolved_config["experiment_id"] = experiment_id
    resolved_config["run_dir"] = str(run_dir)
    v14_atomic_write_json(
        run_dir / "resolved_master_config.json", resolved_config
    )
    v14_atomic_write_json(
        run_dir / "environment_manifest.json", environment
    )
    v14_atomic_write_text(
        run_dir / "pip_freeze.txt", v14_capture_pip_freeze()
    )
    v14_atomic_write_json(
        run_dir / "dataset_manifest.json", dataset_manifest
    )
    v14_atomic_write_csv(run_dir / "experiment_matrix.csv", matrix)
    v14_atomic_write_json(
        run_dir / "experiment_matrix.json",
        matrix.to_dict(orient="records"),
    )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": config_hash_value,
        "experiment_name": config["experiment_name"],
        "run_dir": str(run_dir),
        "started_at_utc": v14_utc_now(),
        "resumed": bool(resumed),
        "dry_run": bool(dry_run),
        "status": "planned" if dry_run else "running",
        "evidence_hierarchy": V14_EVIDENCE_LEVELS,
        "cpu_parallelism": config.get("resolved_cpu_parallelism"),
        "cpu_monitoring": config.get("cpu_monitoring"),
        "dataset_loader": dataset_manifest.get("dataset_loader"),
        "dataset_loader_filename": dataset_manifest.get(
            "dataset_loader_filename"
        ),
        "dataset_loader_sha256": dataset_manifest.get(
            "dataset_loader_sha256"
        ),
        "dataset_loader_version": dataset_manifest.get(
            "dataset_loader_version"
        ),
        "dataset_loader_resolved_path": dataset_manifest.get(
            "dataset_loader_resolved_path"
        ),
        "optuna_trial_n_jobs": 1,
        "monte_carlo_execution": "sequential",
        "cloud_energy_statement": (
            "Cloud/client TabPFN CodeCarbon energy is local-client energy only "
            "and does not represent remote cloud/server GPU energy."
        ),
    }
    v14_atomic_write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def v14_update_matrix_status(
    matrix, scenario_name, sample_actual, iteration, status, run_dir
):
    mask = (
        (matrix["Scenario"] == scenario_name)
        & (
            pd.to_numeric(
                matrix["Sample_Size_Actual"], errors="coerce"
            ) == int(sample_actual)
        )
        & (
            pd.to_numeric(matrix["Iteration"], errors="coerce")
            == int(iteration)
        )
    )
    matrix.loc[mask, "Status"] = status
    v14_atomic_write_csv(Path(run_dir) / "experiment_matrix.csv", matrix)
    v14_atomic_write_json(
        Path(run_dir) / "experiment_matrix.json",
        matrix.to_dict(orient="records"),
    )


def v14_print_sample_report(
    scenario_name, sample, requested_iterations, sample_dir
):
    frame = v14_collect_iteration_results(sample_dir)
    if frame.empty:
        successful = 0
        failed = requested_iterations
        devices = {}
    else:
        success_values = frame.get(
            "success", pd.Series(False, index=frame.index)
        ).astype(str).str.lower().isin({"true", "1", "yes"})
        successful_iterations = set(
            pd.to_numeric(
                frame.loc[success_values, "Iteration"], errors="coerce"
            ).dropna().astype(int).tolist()
        )
        all_iterations = set(range(1, requested_iterations + 1))
        successful = len(successful_iterations)
        failed = len(all_iterations - successful_iterations)
        devices = (
            frame.groupby("Model")["Resolved_Device"]
            .apply(lambda values: sorted({
                str(value) for value in values.dropna()
            }))
            .to_dict()
            if "Resolved_Device" in frame.columns else {}
        )
    sample_dir = Path(sample_dir)
    predictions = sum(
        len(list(sample_dir.glob(pattern)))
        for pattern in (
            "Iteration_*/Models/*/predictions.npz",
            "Iteration_*/Comparators/*/predictions.npz",
        )
    )
    metrics = sum(
        len(list(sample_dir.glob(pattern)))
        for pattern in (
            "Iteration_*/Models/*/metrics.json",
            "Iteration_*/Comparators/*/metrics.json",
        )
    )
    energy = sum(
        len(list(sample_dir.glob(pattern)))
        for pattern in (
            "Iteration_*/Models/*/energy_breakdown.json",
            "Iteration_*/Comparators/*/energy_breakdown.json",
        )
    )
    plots = len(list((sample_dir / "Aggregated" / "Plots").glob("*")))
    print("")
    print(f"Scenario: {scenario_name}")
    print(f"Sample size: {sample['requested']} (actual {sample['actual']})")
    print(f"Requested iterations: {requested_iterations}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Execution devices: {devices}")
    print(
        "Iteration evidence: "
        f"{len(list(sample_dir.glob('Iteration_*/iteration_manifest.json')))}"
    )
    print(f"Predictions: {predictions}")
    print(f"Metrics: {metrics}")
    print(f"Energy: {energy}")
    print(
        "Aggregation: "
        f"{sample_dir / 'Aggregated' / 'Data' / 'iteration_level_results.csv'}"
    )
    print(
        "Plot data: "
        f"{sample_dir / 'Aggregated' / 'Plot_Data'}"
    )
    print(f"Plots: {plots}")


# =============================================================================
# 12. Top-level experiment orchestration, self-tests, and CLI
# =============================================================================

def v14_run_experiment(
    raw_config,
    dry_run=False,
    resume=False,
    force_rerun=False,
    run_dir=None,
):
    config = v14_normalize_config(raw_config)
    X, y, groups, timestamps, dataset_manifest = v14_load_dataset(config)
    scenarios = {
        name: v14_resolve_scenario(config, name)
        for name in config["scenarios"]
    }
    v14_validate_resolved_config(
        config, X, y, groups, timestamps
    )
    preflight_warnings = v14_dependency_preflight(config, scenarios)
    if config.get("outputs", {}).get("save_fitted_models", False):
        raise ValueError(
            "outputs.save_fitted_models=true is not supported by the preserved "
            "V12 runners because they intentionally return predictions/metrics, "
            "not fitted estimator objects. Set it to false."
        )
    cpu_environment = v14_apply_global_cpu_environment(config)
    globals()["RUN_CONFIG"] = config
    environment = v14_capture_environment()
    environment["cpu_parallelism"] = cpu_environment
    if config.get("execution", {}).get(
        "deterministic_torch", False
    ) and torch is not None:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            raise RuntimeError(
                "deterministic_torch was requested but could not be enabled."
            ) from exc
    samples = v14_resolve_sample_sizes(config, len(y))
    config_hash_value = v14_config_hash(config)
    runtime_plot_config = copy.deepcopy(config["plots"])
    runtime_plot_config["_config_hash"] = config_hash_value
    selected_run_dir, experiment_id, resumed = (
        v14_prepare_run_directory(
            config,
            config_hash_value,
            resume=resume,
            run_dir=run_dir,
        )
    )
    matrix = v14_build_experiment_matrix(
        experiment_id,
        config_hash_value,
        config,
        scenarios,
        samples,
    )
    if resumed and (selected_run_dir / "experiment_matrix.csv").exists():
        stored_matrix = pd.read_csv(
            selected_run_dir / "experiment_matrix.csv"
        )
        if len(stored_matrix) != len(matrix):
            raise ValueError(
                "Resume experiment matrix size differs from resolved configuration."
            )
        matrix = stored_matrix
    manifest = v14_write_root_manifests(
        selected_run_dir,
        experiment_id,
        config_hash_value,
        config,
        environment,
        dataset_manifest,
        matrix,
        resumed,
        dry_run,
    )
    logger = V14RunLogger(
        selected_run_dir / "execution_log.txt"
    )
    logger.write(
        "run_initialized",
        experiment_id=experiment_id,
        config_hash=config_hash_value,
        dry_run=dry_run,
        resumed=resumed,
        preflight_warnings=preflight_warnings,
    )
    execution_role_counts = {
        str(role): int(count)
        for role, count in matrix["Analysis_Role"].value_counts().items()
    }
    expected_core_executions = execution_role_counts.get(
        "core_model", 0
    )
    expected_comparator_executions = execution_role_counts.get(
        "auxiliary_comparator", 0
    )
    expected_model_executions = int(len(matrix))
    resolved_comparators_by_scenario_sample = {
        name: {
            str(sample["requested"]): v14_resolve_auxiliary_comparators(
                config,
                name,
                scenario,
                sample,
            )
            for sample in samples
        }
        for name, scenario in scenarios.items()
    }
    dry_run_summary = {
        "experiment_id": experiment_id,
        "config_hash": config_hash_value,
        "run_dir": str(selected_run_dir),
        "scenarios": {
            name: {
                "enabled_models": scenario["enabled_models"],
                "budgeting": scenario["budgeting"],
                "budget_reference_model": scenario.get(
                    "budget_reference_model"
                ),
                "TabPFN_execution": (
                    v14_resolve_tabpfn_execution(scenario)
                    if "TabPFN" in scenario["enabled_models"] else None
                ),
                "auxiliary_comparators": [
                    {
                        "id": comparator["id"],
                        "base_model": comparator["base_model"],
                        "analysis_role": comparator["analysis_role"],
                        "pair_with": comparator.get("pair_with"),
                        "execution": comparator.get("execution"),
                        "context": comparator.get("context"),
                        "budgeting": comparator.get("budgeting"),
                    }
                    for comparator in {
                        item["id"]: item
                        for resolved in (
                            resolved_comparators_by_scenario_sample[name]
                            .values()
                        )
                        for item in resolved
                    }.values()
                ],
                "auxiliary_comparators_by_sample": {
                    sample_label: [
                        comparator["id"] for comparator in resolved
                    ]
                    for sample_label, resolved in (
                        resolved_comparators_by_scenario_sample[name].items()
                    )
                },
            }
            for name, scenario in scenarios.items()
        },
        "sample_sizes": samples,
        "iterations": config["iterations"],
        "experiment_matrix_rows": int(len(matrix)),
        "expected_core_executions": expected_core_executions,
        "expected_comparator_executions": (
            expected_comparator_executions
        ),
        "expected_total_executions": expected_model_executions,
        "expected_model_executions": int(
            expected_model_executions
        ),
        "execution_roles": execution_role_counts,
        "energy_enabled": bool(
            config.get("energy", {}).get("enabled", False)
        ),
        "dataset_loader": dataset_manifest.get("dataset_loader"),
        "dataset_loader_import_check": (
            "PASS"
            if dataset_manifest.get(
                "dataset_loader", {}
            ).get("import_validated", False)
            else "FAIL"
        ),
        "cpu_parallelism": config["resolved_cpu_parallelism"],
        "cpu_monitoring": config["cpu_monitoring"],
        "optuna_trial_n_jobs": 1,
        "monte_carlo_execution": "sequential",
        "preflight_warnings": preflight_warnings,
    }
    if dry_run:
        v14_atomic_write_json(
            selected_run_dir / "dry_run_plan.json", dry_run_summary
        )
        manifest["status"] = "dry_run_complete"
        manifest["completed_at_utc"] = v14_utc_now()
        v14_atomic_write_json(
            selected_run_dir / "run_manifest.json", manifest
        )
        v14_build_artifact_index(selected_run_dir)
        print(json.dumps(
            v14_json_safe(dry_run_summary), indent=2, ensure_ascii=False
        ))
        return dry_run_summary
    for scenario_name, scenario in scenarios.items():
        scenario_dir = (
            selected_run_dir / scenario_folder_name(scenario_name)
        )
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "config_hash": config_hash_value,
            "scenario": scenario_name,
            "budgeting": scenario["budgeting"],
            "budget_reference_model": scenario.get(
                "budget_reference_model"
            ),
            "semantic_role": scenario.get("semantic_role"),
            "enabled_models": scenario["enabled_models"],
            "auxiliary_comparators": list(
                {
                    comparator["id"]: comparator
                    for resolved in (
                        resolved_comparators_by_scenario_sample[
                            scenario_name
                        ].values()
                    )
                    for comparator in resolved
                }.values()
            ),
            "auxiliary_comparators_by_sample": (
                resolved_comparators_by_scenario_sample[scenario_name]
            ),
            "cpu_parallelism": config.get(
                "resolved_cpu_parallelism"
            ),
            "cpu_monitoring": config.get("cpu_monitoring"),
            "resolved_devices": {
                model_name: v14_expected_model_device(
                    model_name,
                    scenario["resolved_models"][model_name],
                    scenario,
                )
                for model_name in scenario["enabled_models"]
            },
        }
        v14_atomic_write_json(
            scenario_dir / "scenario_manifest.json",
            scenario_manifest,
        )
        for sample in samples:
            sample_dir = scenario_dir / sample["folder"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_manifest = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "config_hash": config_hash_value,
                "scenario": scenario_name,
                "requested_sample_size": sample["requested"],
                "actual_sample_size": sample["actual"],
                "requested_iterations": config["iterations"],
                "cpu_parallelism": config.get(
                    "resolved_cpu_parallelism"
                ),
                "cpu_monitoring": config.get("cpu_monitoring"),
                "status": "running",
                "started_at_utc": v14_utc_now(),
            }
            v14_atomic_write_json(
                sample_dir / "sample_manifest.json",
                sample_manifest,
            )
            v14_atomic_write_json(
                sample_dir / "resolved_config.json",
                {
                    **copy.deepcopy(config),
                    "resolved_scenario": scenario,
                    "resolved_sample": sample,
                    "config_hash": config_hash_value,
                },
            )
            for iteration in range(1, config["iterations"] + 1):
                v14_update_matrix_status(
                    matrix,
                    scenario_name,
                    sample["actual"],
                    iteration,
                    "running",
                    selected_run_dir,
                )
                try:
                    status = v14_atomic_iteration(
                        X,
                        y,
                        groups,
                        timestamps,
                        config,
                        scenario_name,
                        scenario,
                        sample,
                        iteration,
                        selected_run_dir,
                        experiment_id,
                        config_hash_value,
                        environment,
                        logger,
                        force_rerun=force_rerun,
                    )
                    v14_update_matrix_status(
                        matrix,
                        scenario_name,
                        sample["actual"],
                        iteration,
                        status,
                        selected_run_dir,
                    )
                except Exception as exc:
                    iteration_dir = (
                        sample_dir / f"Iteration_{iteration:03d}"
                    )
                    iteration_dir.mkdir(
                        parents=True, exist_ok=True
                    )
                    error = {
                        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                        "Experiment_ID": experiment_id,
                        "Config_Hash": config_hash_value,
                        "Scenario": scenario_name,
                        "Sample_Size_Requested": sample["requested"],
                        "Sample_Size_Actual": sample["actual"],
                        "Iteration": iteration,
                        "Seed": config["base_seed"] + iteration,
                        "success": False,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "timestamp_utc": v14_utc_now(),
                    }
                    v14_atomic_write_json(
                        iteration_dir / "error.json", error
                    )
                    v14_atomic_write_text(
                        iteration_dir / "traceback.txt",
                        traceback.format_exc(),
                    )
                    logger.write(
                        "iteration_failed",
                        **error,
                    )
                    v14_update_matrix_status(
                        matrix,
                        scenario_name,
                        sample["actual"],
                        iteration,
                        "failed",
                        selected_run_dir,
                    )
                    if config.get("execution", {}).get(
                        "on_iteration_error", "continue"
                    ) == "raise":
                        raise
            v14_aggregate_sample(
                sample_dir, config["iterations"], runtime_plot_config
            )
            sample_manifest["status"] = "complete"
            sample_manifest["completed_at_utc"] = v14_utc_now()
            sample_manifest["iteration_status_counts"] = (
                matrix[
                    (matrix["Scenario"] == scenario_name)
                    & (
                        pd.to_numeric(
                            matrix["Sample_Size_Actual"],
                            errors="coerce",
                        ) == sample["actual"]
                    )
                ]["Status"].value_counts().to_dict()
            )
            v14_atomic_write_json(
                sample_dir / "sample_manifest.json",
                sample_manifest,
            )
            v14_print_sample_report(
                scenario_name,
                sample,
                config["iterations"],
                sample_dir,
            )
        v14_aggregate_scenario(
            scenario_dir, runtime_plot_config, scenario_name
        )
    v14_aggregate_global(selected_run_dir, runtime_plot_config)
    v14_build_artifact_index(selected_run_dir)
    validation = v14_validate_run(
        selected_run_dir, write_report=True
    )
    manifest["status"] = (
        "complete" if validation["status"] == "PASS"
        else "complete_with_validation_failures"
    )
    manifest["completed_at_utc"] = v14_utc_now()
    manifest["validation_status"] = validation["status"]
    v14_atomic_write_json(
        selected_run_dir / "run_manifest.json", manifest
    )
    logger.write(
        "run_completed",
        experiment_id=experiment_id,
        validation_status=validation["status"],
        run_dir=str(selected_run_dir),
    )
    v14_build_artifact_index(selected_run_dir)
    print(f"Validation: {validation['status']}")
    print(f"Run directory: {selected_run_dir}")
    return {
        "experiment_id": experiment_id,
        "config_hash": config_hash_value,
        "run_dir": str(selected_run_dir),
        "validation": validation,
    }


def v14_configs_from_directory(config_dir):
    paths = sorted(Path(config_dir).glob("*.json"))
    selected = []
    for path in paths:
        raw = v14_load_config_with_extends(path)
        if raw.get("template_only", False):
            continue
        if raw.get("enabled", True) is False:
            continue
        selected.append((path, raw))
    if not selected:
        raise ValueError(
            f"No runnable JSON configurations found in {config_dir}."
        )
    return selected


def run_v14_self_tests():
    """Run lightweight checks that do not execute the definitive study."""
    results = []

    def run_case(number, name, test):
        try:
            test()
        except Exception as exc:
            results.append({
                "test": number,
                "name": name,
                "status": "FAIL",
                "reason": f"{type(exc).__name__}: {exc}",
            })
        else:
            results.append({
                "test": number,
                "name": name,
                "status": "PASS",
                "reason": None,
            })

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    def paired_fixture():
        common = {
            "Experiment_ID": "self-test",
            "Config_Hash": "hash",
            "Scenario": "arbitrary-scenario",
            "Sample_Size_Actual": 40,
            "Iteration": 1,
            "Seed": 2026,
            "Split_Fingerprint": "same-split",
            "Test_Indices_SHA256": "same-test-indices",
            "Test_Labels_SHA256": "same-y-true",
            "Actual_Total_Runtime_Seconds": 2.0,
            "AUROC": 0.70,
            "Balanced_Accuracy": 0.60,
            "Sensitivity": 0.60,
            "Precision": 0.60,
            "Brier_Score": 0.20,
        }
        primary = {
            **common,
            "Model": "TabPFN",
            "Execution_Variant": "TabPFN",
            "Base_Model": "TabPFN",
            "Analysis_Role": "core_model",
            "Reference_Budget_Seconds": 1.0,
        }
        comparator = {
            **common,
            "Model": "Plain_Local",
            "Execution_Variant": "Plain_Local",
            "Base_Model": "TabPFN",
            "Analysis_Role": "auxiliary_comparator",
            "Comparator_ID": "Plain_Local",
            "Paired_With": "TabPFN",
            "Actual_Total_Runtime_Seconds": 4.0,
            "AUROC": 0.80,
        }
        return primary, comparator

    def test_fixed():
        plan = v14_resolve_context_plan(
            {
                "context_strategy": {
                    "strategy": "fixed",
                    "rows": 12,
                }
            },
            full_train_n=30,
            total_sample_n=40,
        )
        attempts = [plan["target_context_n"]]
        require(plan["strategy"] == "fixed", plan)
        require(len(attempts) == 1, attempts)
        require(plan["target_context_n"] == 12, plan)

    def test_adaptive():
        plan = v14_resolve_context_plan(
            {"context_strategy": {"strategy": "adaptive"}},
            full_train_n=30,
            total_sample_n=40,
        )
        require(plan["strategy"] == "adaptive", plan)
        require(plan["target_context_n"] == 0, plan)

    def test_full():
        plan = v14_resolve_context_plan(
            {"context_strategy": {"strategy": "full"}},
            full_train_n=30,
            total_sample_n=40,
        )
        require(plan["strategy"] == "full", plan)
        require(plan["target_context_n"] == 30, plan)

    def test_comparator():
        primary, comparator = paired_fixture()
        paired = v14_build_paired_comparison([primary, comparator])
        require(len(paired) == 1, paired)
        require(bool(paired.iloc[0]["Exact_Pair_Valid"]), paired.iloc[0])
        require(
            comparator["Analysis_Role"] == "auxiliary_comparator",
            comparator,
        )
        require(
            Path("Comparators") / comparator["Comparator_ID"]
            != Path("Models") / primary["Execution_Variant"],
            "comparator and core paths must differ",
        )

    def test_comparator_isolation():
        primary, comparator = paired_fixture()
        core_before = v14_canonical_json([primary])
        _ = v14_build_paired_comparison([primary, comparator])
        core_after = v14_canonical_json([primary])
        require(core_before == core_after, "core record was mutated")
        require(
            primary["Reference_Budget_Seconds"] == 1.0,
            "comparator altered reference budget",
        )

    def test_ranking_exclusion():
        primary, comparator = paired_fixture()
        summary = v14_descriptive_summary(
            pd.DataFrame([primary, comparator])
        )
        require(
            summary["Model"].tolist() == ["TabPFN"],
            summary.to_dict(orient="records"),
        )

    def test_paired_direction():
        primary, comparator = paired_fixture()
        row = v14_build_paired_comparison(
            [primary, comparator]
        ).iloc[0]
        for metric in ("AUROC", "Runtime"):
            forward = row[
                f"Paired_Difference_{metric}_Primary_Minus_Comparator"
            ]
            reverse = row[
                f"Paired_Difference_{metric}_Comparator_Minus_Primary"
            ]
            require(
                bool(np.isclose(forward, -reverse)),
                f"{metric}: {forward} != -({reverse})",
            )

    def test_tuned_reference_budget():
        record = v14_reference_budget_from_components(
            reference_is_tuned=True,
            actual_optuna_tuning_time_seconds=3.0,
            actual_reference_execution_runtime_seconds=9.0,
        )
        require(
            record["Budget_Basis"] == "reference_tuning_runtime",
            record,
        )
        require(record["Reference_Budget_Seconds"] == 3.0, record)
        require(
            record["Reference_Budget_Source_Field"]
            == "Actual_Optuna_Tuning_Time_Seconds",
            record,
        )

    def test_final_fit_exclusion():
        first = v14_reference_budget_from_components(
            reference_is_tuned=True,
            actual_optuna_tuning_time_seconds=3.0,
            actual_reference_execution_runtime_seconds=9.0,
        )
        slower_final_stage = v14_reference_budget_from_components(
            reference_is_tuned=True,
            actual_optuna_tuning_time_seconds=3.0,
            actual_reference_execution_runtime_seconds=109.0,
        )
        require(
            first["Reference_Budget_Seconds"]
            == slower_final_stage["Reference_Budget_Seconds"]
            == 3.0,
            (first, slower_final_stage),
        )

    def test_non_tuned_reference():
        record = v14_reference_budget_from_components(
            reference_is_tuned=False,
            actual_reference_execution_runtime_seconds=7.5,
        )
        require(
            record["Budget_Basis"] == "reference_execution_runtime",
            record,
        )
        require(record["Reference_Budget_Seconds"] == 7.5, record)

    def test_cpu_parallelism():
        resolved = v14_resolve_cpu_parallelism({
            "cpu_parallelism": {
                "policy": "max_available",
                "threads": "auto",
            }
        })
        require(
            resolved["CPU_Count_Available_To_Process"] > 0,
            resolved,
        )
        require(
            resolved["CPU_Threads_Resolved"]
            == resolved["CPU_Count_Available_To_Process"],
            resolved,
        )

    def test_optuna_sequential():
        source = (
            inspect.getsource(run_optuna_with_time_budget)
            + inspect.getsource(run_optuna_without_budget)
        ).replace(" ", "")
        require(
            source.count("n_jobs=1") >= 2,
            "both Optuna execution paths must specify n_jobs=1",
        )

    def test_monte_carlo_sequential():
        source = inspect.getsource(run_monte_carlo)
        require("ThreadPoolExecutor" not in source, "thread executor found")
        require("ProcessPoolExecutor" not in source, "process executor found")
        require(
            'for i in tqdm(range(RUN_CONFIG["iterations"])' in source,
            "sequential iteration loop not found",
        )

    def test_monitor_metadata():
        monitor = V14CPUResourceMonitor(
            model_name="RandomForest",
            execution_variant="RandomForest",
            scenario_name="self-test",
            sample_size=40,
            iteration=1,
            total_iterations=1,
            resource_spec={
                "Execution_Path": "local",
                "Resolved_Device": "cpu",
                "CPU_Count_Available_To_Process": 2,
                "CPU_Threads_Configured": 2,
                "CPU_Parallelism_Policy": "max_available",
                "CPU_Thread_Setting_Source": "self-test",
                "CPU_Thread_Parameter": "n_jobs",
            },
            monitoring_config={
                "enabled": True,
                "show_console": False,
                "sampling_interval_seconds": 5,
            },
        )
        monitor.process = object()
        monitor.started_perf = time.perf_counter()
        monitor.samples = [{
            "timestamp_utc": v14_utc_now(),
            "elapsed_seconds": 0.01,
            "process_cpu_percent": 150.0,
            "approx_active_logical_cores": 1.5,
            "system_cpu_percent": 50.0,
        }]
        summary = monitor.stop()
        require(summary["CPU_Cores_Available"] == 2, summary)
        require(summary["CPU_Threads_Configured"] == 2, summary)
        require(
            summary["CPU_Monitoring_Excluded_From_Runtime"] is False,
            summary,
        )
        require(
            "may be included"
            in summary["CPU_Monitoring_Overhead_Handling"],
            summary,
        )
        require(
            summary["CPU_Monitoring_Budget_Boundary_Role"].startswith(
                "none"
            ),
            summary,
        )

    def test_dataset_loader():
        valid_config = v14_builtin_base_config()
        _, provenance = v14_resolve_dataset_loader(valid_config)
        require(provenance["import_validated"] is True, provenance)
        require(
            set(provenance["required_functions"])
            >= set(V14_REQUIRED_DATASET_LOADER_FUNCTIONS),
            provenance,
        )
        missing_path = (
            Path(tempfile.gettempdir())
            / f"missing_dataset_loader_{uuid.uuid4().hex}.py"
        )
        missing_config = copy.deepcopy(valid_config)
        missing_config["dataset_loader"]["module_path"] = str(missing_path)
        try:
            v14_resolve_dataset_loader(missing_config)
        except ImportError:
            pass
        else:
            raise AssertionError("missing loader was not rejected")
        invalid_module = type(
            "InvalidLoader", (), {"prepare_dataset": lambda: None}
        )()
        try:
            v14_validate_dataset_loader_module(invalid_module)
        except ImportError:
            pass
        else:
            raise AssertionError("incompatible loader was not rejected")

    def test_resume_marker():
        with tempfile.TemporaryDirectory(
            prefix="v14-self-test-resume-"
        ) as temp_dir:
            root = Path(temp_dir)
            evidence = root / "evidence.txt"
            evidence.write_text("valid", encoding="utf-8")
            marker = {
                "complete": True,
                "required_file_checksums": {
                    "evidence.txt": v14_sha256_file(evidence)
                },
            }
            require(
                v14_validate_completion_marker(root, marker),
                "valid completion marker was not reusable",
            )
            evidence.write_text("changed", encoding="utf-8")
            require(
                not v14_validate_completion_marker(root, marker),
                "invalid completion marker was incorrectly reusable",
            )

    def test_plot_regeneration():
        with tempfile.TemporaryDirectory(
            prefix="v14-self-test-plots-"
        ) as temp_dir:
            root = Path(temp_dir)
            v14_atomic_write_json(
                root / "resolved_master_config.json",
                {
                    "plots": {
                        "enabled": True,
                        "show_figures": False,
                    },
                    "config_hash": "self-test",
                },
            )
            result = v14_regenerate_plots(root)
            require(result["scenarios"] == [], result)
            require(
                (root / "artifact_index.json").exists(),
                "saved-data regeneration did not create an artifact index",
            )

    cases = [
        ("Fixed context semantics", test_fixed),
        ("Adaptive context", test_adaptive),
        ("Full context", test_full),
        ("Auxiliary comparator", test_comparator),
        ("Comparator isolation", test_comparator_isolation),
        ("Core ranking exclusion", test_ranking_exclusion),
        ("Paired comparison direction", test_paired_direction),
        ("Tuned-reference budget definition", test_tuned_reference_budget),
        ("Final-fit exclusion from HPO budget", test_final_fit_exclusion),
        ("Non-tuned reference definition", test_non_tuned_reference),
        ("CPU maximum parallelism", test_cpu_parallelism),
        ("Optuna sequential", test_optuna_sequential),
        ("Monte Carlo sequential", test_monte_carlo_sequential),
        ("CPU monitoring metadata", test_monitor_metadata),
        ("dataset_loader dependency", test_dataset_loader),
        ("Resume validity", test_resume_marker),
        ("Plot regeneration", test_plot_regeneration),
    ]
    for number, (name, test) in enumerate(cases, start=1):
        run_case(number, name, test)
    overall = (
        "PASS"
        if all(item["status"] == "PASS" for item in results)
        else "FAIL"
    )
    print("")
    print("V14 SELF TEST SUMMARY")
    print("")
    for item in results:
        suffix = (
            ""
            if item["status"] == "PASS"
            else f" - {item['reason']}"
        )
        print(
            f"Test {item['test']:02d}: {item['status']} "
            f"- {item['name']}{suffix}"
        )
    print(f"Overall: {overall}")
    return {"status": overall, "tests": results}


def v14_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "V14 JSON-driven reproducible thesis experiment and artifact pipeline"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="Path to one experiment JSON file.")
    source.add_argument(
        "--config-dir",
        help="Directory of JSON configurations to execute sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve, validate, and save the matrix without fitting models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching existing run.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun completed atomic iterations instead of skipping them.",
    )
    parser.add_argument(
        "--run-dir",
        help="Existing run directory for resume, plot regeneration, or validation.",
    )
    parser.add_argument(
        "--regenerate-plots",
        action="store_true",
        help="Regenerate plots exclusively from persisted artifacts.",
    )
    parser.add_argument(
        "--validate-run",
        action="store_true",
        help="Validate an existing run and write validation reports.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the lightweight V14 targeted self-test suite.",
    )
    parser.add_argument(
        "--scenario",
        help="Optional scenario filter for --regenerate-plots.",
    )
    parser.add_argument(
        "--sample",
        help="Optional requested/actual sample filter for --regenerate-plots.",
    )
    return parser


def v14_main(argv=None):
    parser = v14_cli_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        result = run_v14_self_tests()
        return 0 if result["status"] == "PASS" else 2
    if args.regenerate_plots or args.validate_run:
        if not args.run_dir:
            parser.error(
                "--run-dir is required with --regenerate-plots or --validate-run."
            )
        result = {}
        if args.regenerate_plots:
            result["plot_regeneration"] = v14_regenerate_plots(
                args.run_dir,
                scenario_filter=args.scenario,
                sample_filter=args.sample,
            )
        if args.validate_run:
            result["validation"] = v14_validate_run(
                args.run_dir, write_report=True
            )
            v14_build_artifact_index(args.run_dir)
        print(json.dumps(v14_json_safe(result), indent=2))
        return 0 if (
            not args.validate_run
            or result["validation"]["status"] == "PASS"
        ) else 2
    if not args.config and not args.config_dir:
        parser.error(
            "Provide --config or --config-dir, or choose an existing-run action."
        )
    configs = (
        [(Path(args.config), v14_load_config_with_extends(args.config))]
        if args.config else v14_configs_from_directory(args.config_dir)
    )
    results = []
    for path, raw in configs:
        print(f"Resolving configuration: {path}")
        result = v14_run_experiment(
            raw,
            dry_run=args.dry_run,
            resume=args.resume,
            force_rerun=args.force_rerun,
            run_dir=args.run_dir,
        )
        results.append(result)
    print(json.dumps(v14_json_safe(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(v14_main())
