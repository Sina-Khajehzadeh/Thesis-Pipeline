"""
dataset_loader.py
=================
Dataset-agnostic front door for the V14 thesis experiment pipeline.

The pipeline ENGINE already works on generic (X, y, groups). This module turns
ANY tabular dataset into exactly those three objects, driven by a small config
dict, so you can point the same pipeline at a new dataset without touching the
engine.

    X       -> a pandas DataFrame of features (numeric and/or categorical, NaNs OK;
               the engine does split-safe, train-fitted preprocessing per fold)
    y       -> a 1-D int array of labels in {0, 1}  (this engine is BINARY)
    groups  -> a 1-D array of group ids (e.g. patient id) or None for row-level

Supported sources: an in-memory DataFrame, a CSV/TSV/Parquet/Pickle/Feather/JSON
file, or a built-in scikit-learn toy dataset (handy for a quick smoke test).

Scope: this engine is built for BINARY classification. The target must have
exactly two classes; they are mapped to {0, 1}. Multiclass / regression would
need engine-level changes (metrics + the binary check) — see the README.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

DATASET_LOADER_VERSION = "1.0"
__version__ = DATASET_LOADER_VERSION
__all__ = ["prepare_dataset", "_load_dataframe"]


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def _load_dataframe(cfg: dict) -> pd.DataFrame:
    # 1) in-memory DataFrame
    if cfg.get("dataframe") is not None:
        return cfg["dataframe"].copy()

    # 2) built-in scikit-learn toy dataset (for quick tests)
    sk = cfg.get("sklearn_dataset")
    if sk:
        from sklearn import datasets as _d
        loaders = {
            "breast_cancer": _d.load_breast_cancer,
            "iris_binary": None,  # handled specially below
            "wine_binary": None,
        }
        if sk == "breast_cancer":
            b = _d.load_breast_cancer(as_frame=True)
            df = b.frame.copy()
            df.rename(columns={"target": cfg.get("target", "target")}, inplace=True)
            return df
        if sk in ("iris_binary", "wine_binary"):
            raw = _d.load_iris(as_frame=True) if sk == "iris_binary" else _d.load_wine(as_frame=True)
            df = raw.frame.copy()
            df = df[df["target"].isin([0, 1])].reset_index(drop=True)  # keep 2 classes
            df.rename(columns={"target": cfg.get("target", "target")}, inplace=True)
            return df
        raise ValueError(f"Unknown sklearn_dataset '{sk}'. Options: {sorted(loaders)}")

    # 3) a file path, dispatched by extension
    path = cfg.get("source")
    if not path:
        raise ValueError("dataset config needs one of: 'dataframe', 'sklearn_dataset', or 'source' (a file path).")
    rk = cfg.get("read_kwargs", {})
    low = str(path).lower()
    if low.endswith((".csv", ".txt")):
        return pd.read_csv(path, **rk)
    if low.endswith((".tsv",)):
        return pd.read_csv(path, sep="\t", **rk)
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, **rk)
    if low.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path, **rk)
    if low.endswith((".feather",)):
        return pd.read_feather(path, **rk)
    if low.endswith((".json",)):
        return pd.read_json(path, **rk)
    raise ValueError(f"Unrecognised file type for source '{path}'. "
                     "Use .csv/.tsv/.parquet/.pkl/.feather/.json, or pass a DataFrame.")


# --------------------------------------------------------------------------- #
# Target encoding -> {0, 1}                                                    #
# --------------------------------------------------------------------------- #
def _encode_binary_target(y_raw: pd.Series, positive_label=None, target_map=None):
    """Map an arbitrary 2-class target to {0,1}. Returns (y_int, positive_value)."""
    if target_map is not None:
        y = y_raw.map(target_map)
        if y.isna().any():
            missing = sorted(set(y_raw[y.isna()].unique()))
            raise ValueError(f"target_map does not cover these values: {missing}")
        pos = [k for k, v in target_map.items() if int(v) == 1]
        return y.astype(int).to_numpy(), (pos[0] if pos else 1)

    vals = pd.unique(y_raw.dropna())
    if len(vals) != 2:
        raise ValueError(
            f"Target has {len(vals)} distinct values {sorted(map(str, vals))[:8]}; "
            "this engine is BINARY (exactly 2 classes required). "
            "For multiclass/regression see the README."
        )
    s = set(vals.tolist())

    # already 0/1
    if s <= {0, 1}:
        return y_raw.astype(int).to_numpy(), 1

    # explicit positive label
    if positive_label is not None:
        if positive_label not in s:
            raise ValueError(f"positive_label={positive_label!r} is not one of the target values {sorted(map(str, s))}.")
        return (y_raw == positive_label).astype(int).to_numpy(), positive_label

    # friendly auto-detection of the "positive" class
    def pick(cands):
        for c in cands:
            for v in vals:
                if str(v).strip().lower() == c:
                    return v
        return None
    auto = pick(["1", "true", "yes", "y", "pos", "positive", "malignant", "default", "churn"])
    if auto is None:
        # deterministic fallback: larger when numeric, else lexicographically last
        try:
            auto = max(vals)
        except TypeError:
            auto = sorted(map(str, vals))[-1]
            auto = next(v for v in vals if str(v) == auto)
    return (y_raw == auto).astype(int).to_numpy(), auto


# --------------------------------------------------------------------------- #
# Main entry                                                                   #
# --------------------------------------------------------------------------- #
def prepare_dataset(cfg: dict, verbose: bool = True):
    """
    Build (X, y, groups, info) from a dataset config.

    Config keys (all optional unless noted):
      source / dataframe / sklearn_dataset : where the data comes from (one required)
      read_kwargs        : dict passed to the pandas reader (e.g. {'sep': ';'})
      target             : name of the label column                (required unless target_fn)
      target_fn          : callable(df) -> Series, to DERIVE the label instead of a column
      positive_label     : which target value becomes class 1 (else auto-detected)
      target_map         : explicit {value: 0/1} mapping (overrides positive_label)
      group_column       : column holding the group id (e.g. patient) -> grouped split.
                           Omit / None for ordinary row-level splitting.
      feature_columns    : explicit feature list (else: all columns except target/group/dropped)
      drop_columns       : columns to exclude (ids, leakage, free text, timestamps)
      datetime_columns   : datetime columns to drop (use the engine's temporal mode separately)
      row_filter         : callable(df) -> df, to subset rows before anything else
      drop_constant      : drop single-value columns (default True)
      max_cardinality    : drop object/category columns with more uniques than this (default None)
      keep_dataframe     : pass a raw DataFrame to the engine (default True, recommended).
                           Set False to one-hot encode to a numeric matrix here instead.

    Returns: X (DataFrame or float32 ndarray), y (int8 ndarray in {0,1}),
             groups (ndarray or None), info (dict).
    """
    df = _load_dataframe(cfg)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if cfg.get("row_filter"):
        df = cfg["row_filter"](df).copy()
    df = df.reset_index(drop=True)

    # ---- target ----
    if cfg.get("target_fn"):
        y_raw = pd.Series(cfg["target_fn"](df)).reset_index(drop=True)
    else:
        target = cfg.get("target")
        if not target:
            raise ValueError("dataset config needs 'target' (label column) or 'target_fn'.")
        if target not in df.columns:
            raise ValueError(f"target column '{target}' not found. Available: {list(df.columns)[:20]}...")
        y_raw = df[target].reset_index(drop=True)

    # drop rows with missing target
    keep = y_raw.notna().to_numpy()
    if not keep.all():
        df = df.loc[keep].reset_index(drop=True)
        y_raw = y_raw.loc[keep].reset_index(drop=True)

    y, pos_value = _encode_binary_target(y_raw, cfg.get("positive_label"), cfg.get("target_map"))
    y = y.astype("int8")

    # ---- groups ----
    groups = None
    gcol = cfg.get("group_column")
    if gcol:
        if gcol not in df.columns:
            raise ValueError(f"group_column '{gcol}' not found in the data.")
        groups = df[gcol].to_numpy()
        if pd.isna(groups).any():
            raise ValueError(f"group_column '{gcol}' has missing values; every row needs a group id.")

    # ---- features ----
    drop = set(cfg.get("drop_columns", []) or [])
    if cfg.get("target"):
        drop.add(cfg["target"])
    if gcol:
        drop.add(gcol)
    drop |= set(cfg.get("datetime_columns", []) or [])

    if cfg.get("feature_columns"):
        feats = [c for c in cfg["feature_columns"] if c in df.columns]
    else:
        feats = [c for c in df.columns if c not in drop]
    if not feats:
        raise ValueError("No feature columns left after applying target/group/drop. Check your config.")
    X = df[feats].copy()

    # booleans -> int (so they read as numeric)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    dropped_const, dropped_highcard = [], []
    if cfg.get("drop_constant", True):
        dropped_const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if dropped_const:
            X = X.drop(columns=dropped_const)

    maxc = cfg.get("max_cardinality")
    if maxc:
        for c in X.select_dtypes(include=["object", "category"]).columns:
            if X[c].nunique(dropna=True) > int(maxc):
                dropped_highcard.append(c)
        if dropped_highcard:
            X = X.drop(columns=dropped_highcard)

    if not cfg.get("keep_dataframe", True):
        # One-hot encode here and return a complete numeric matrix. NaNs in numeric
        # columns are filled with 0.0 (a leakage-free constant). NOTE: the preferred
        # path is keep_dataframe=True, where the engine imputes per split (no leakage).
        X = pd.get_dummies(X, dummy_na=True).astype("float32").fillna(0.0)

    X = X.reset_index(drop=True)

    # ---- validate & summarise ----
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) are misaligned.")
    if groups is not None and len(groups) != len(y):
        raise ValueError(f"groups ({len(groups)}) and y ({len(y)}) are misaligned.")
    classes, counts = np.unique(y, return_counts=True)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(f"Encoded target is not binary {{0,1}} (found {classes.tolist()}).")

    n_num = int(X.select_dtypes(include=["number"]).shape[1]) if isinstance(X, pd.DataFrame) else X.shape[1]
    n_cat = int(X.select_dtypes(include=["object", "category"]).shape[1]) if isinstance(X, pd.DataFrame) else 0
    info = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric": n_num,
        "n_categorical": n_cat,
        "class1_prevalence": float(np.mean(y == 1)),
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        "positive_value": pos_value,
        "n_groups": (int(pd.unique(groups).size) if groups is not None else None),
        "grouped": groups is not None,
        "dropped_constant": dropped_const,
        "dropped_high_cardinality": dropped_highcard,
        "X_type": "DataFrame" if isinstance(X, pd.DataFrame) else "ndarray",
    }

    if verbose:
        print("=" * 64)
        print("DATASET PREPARED")
        print("=" * 64)
        print(f"  rows                : {info['n_rows']:,}")
        print(f"  features            : {info['n_features']}  ({n_num} numeric, {n_cat} categorical)")
        print(f"  target              : '{cfg.get('target', '(derived)')}'  -> positive class = {pos_value!r} -> 1")
        print(f"  class balance       : class1={info['class1_prevalence']:.4f}  counts={info['class_counts']}")
        if groups is not None:
            ev = info["n_rows"] / max(info["n_groups"], 1)
            print(f"  grouping            : '{gcol}'  ->  {info['n_groups']:,} groups  (~{ev:.1f} rows/group)")
        else:
            print(f"  grouping            : none (row-level splitting)")
        if dropped_const:
            print(f"  dropped (constant)  : {dropped_const}")
        if dropped_highcard:
            print(f"  dropped (high-card) : {dropped_highcard}")
        print(f"  X passed to engine  : {info['X_type']}  (engine preprocesses per split)")
        print("=" * 64)

    return X, y, groups, info
