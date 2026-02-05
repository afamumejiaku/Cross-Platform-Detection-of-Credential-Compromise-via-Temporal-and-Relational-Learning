#!/usr/bin/env python3
"""
Feature Engineering + Label Simulation + Temporal Splits
This module handles:
- Per-email timeline construction
- Feature computation (similarity, temporal, cross-platform)
- Label simulation with campaign drift
- Temporal train/val/test splitting
- Caching to parquet files

GPU Acceleration:
- Uses PyTorch for TF-IDF similarity computation on GPU when available
"""

from __future__ import annotations

import os
import json
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

import torch

# -------------------------
# Configuration
# -------------------------

@dataclass
class BuildConfig:
    n_honeywords: int = 50
    tfidf_ngram: int = 3
    max_features: int = 5000  # Limit vocabulary size to prevent OOM
    sim_threshold_noisy: float = 0.85
    early_frac: float = 0.40
    cross_platform_leak_days: int = 30

@dataclass
class SplitConfig:
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20


# -------------------------
# Utils
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)

def days_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return (a - b).total_seconds() / (3600 * 24)

def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# -------------------------
# GPU-Accelerated Similarity Index
# -------------------------

def _collect_all_strings(df: pd.DataFrame, n_hw: int) -> List[str]:
    """Collect all password and honeyword strings for TF-IDF fitting."""
    cols_hw = [f"Honeyword_{i}" for i in range(1, n_hw + 1) if f"Honeyword_{i}" in df.columns]
    all_strings = []
    all_strings.extend(df["Password"].astype(str).fillna("").tolist())
    for c in cols_hw:
        all_strings.extend(df[c].astype(str).fillna("").tolist())
    all_strings = [s if isinstance(s, str) else str(s) for s in all_strings]
    all_strings = [s for s in all_strings if s is not None]
    if "" not in all_strings:
        all_strings.append("")
    return all_strings


class GPUSimilarityIndex:
    """
    TF-IDF char n-gram vectorizer with GPU-accelerated similarity computation.
    
    Uses sparse matrices to avoid memory issues with large vocabularies.
    Only converts small batches to dense when computing similarities.
    """
    
    def __init__(
        self, 
        strings: List[str], 
        ngram: int = 3, 
        max_features: int = 5000,
        device: Optional[torch.device] = None
    ):
        self.device = device if device else get_device()
        
        # Limit vocabulary size with max_features
        # Use float32 instead of float64
        self.vectorizer = TfidfVectorizer(
            analyzer="char", 
            ngram_range=(ngram, ngram), 
            lowercase=False,
            max_features=max_features,  # Prevents huge vocabulary
            dtype=np.float32  # Half the memory of float64
        )
        
        # Get unique strings while preserving order
        self.strings = list(dict.fromkeys(strings))
        self.s2i = {s: i for i, s in enumerate(self.strings)}
        
        print(f"Fitting TF-IDF on {len(self.strings)} unique strings...")
        
        # Fit and transform to sparse matrix
        tfidf_sparse = self.vectorizer.fit_transform(self.strings)
        
        print(f"TF-IDF vocabulary size: {tfidf_sparse.shape[1]}")
        print(f"TF-IDF matrix shape: {tfidf_sparse.shape}")
        print(f"TF-IDF sparsity: {1 - tfidf_sparse.nnz / (tfidf_sparse.shape[0] * tfidf_sparse.shape[1]):.4f}")
        
        # Keep as sparse on CPU, only move small chunks to GPU
        # Store as CSR (Compressed Sparse Row) for efficient row access
        self.tfidf_sparse = tfidf_sparse.tocsr().astype(np.float32)
        
        # Pre-compute norms on CPU (cheap operation)
        # Using scipy's sparse operations
        squared_norms = np.array(self.tfidf_sparse.multiply(self.tfidf_sparse).sum(axis=1)).flatten()
        self.norms = np.sqrt(squared_norms).clip(min=1e-8)
        
        print(f"TF-IDF index ready (kept sparse to save memory)")

    def idx(self, s: str) -> int:
        """Get index for a string."""
        return self.s2i.get(s, self.s2i.get("", 0))

    def sims_to_history(self, current_idx: int, hist_indices: List[int]) -> Tuple[float, float]:
        """
        Compute max and mean cosine similarity between current string and history.
        
        Operates on sparse matrices, only converts to dense for small vectors.
        """
        if not hist_indices:
            return 0.0, 0.0
        
        # Get current vector (sparse row)
        cur_vec = self.tfidf_sparse[current_idx]  # CSR row
        cur_norm = self.norms[current_idx]
        
        # Get history vectors (sparse rows)
        hist_vecs = self.tfidf_sparse[hist_indices]  # CSR submatrix
        hist_norms = self.norms[hist_indices]
        
        # Compute dot products using sparse operations
        # cur_vec is (1, vocab_size), hist_vecs.T is (vocab_size, n_hist)
        dots = hist_vecs.dot(cur_vec.T).toarray().flatten()  # Only this is dense
        
        # Normalize to get cosine similarities
        sims = dots / (hist_norms * cur_norm + 1e-8)
        
        if len(sims) == 0:
            return 0.0, 0.0
        
        return float(np.max(sims)), float(np.mean(sims))

    def batch_sims_to_history(
        self, 
        current_indices: List[int], 
        hist_indices_list: List[List[int]]
    ) -> List[Tuple[float, float]]:
        """
        Batch computation of similarities for multiple queries.
        More efficient for large batches.
        """
        results = []
        for cur_idx, hist_indices in zip(current_indices, hist_indices_list):
            results.append(self.sims_to_history(cur_idx, hist_indices))
        return results


# -------------------------
# Feature Engineering
# -------------------------

def build_features_and_labels(
    df: pd.DataFrame,
    build_cfg: BuildConfig,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Build all features and simulated labels for the dataset.
    
    Features computed:
    - Password similarity features (current vs historical passwords/honeywords)
    - Temporal features (time since last leak/attack, sequence index)
    - Cross-platform features (platforms seen, overlap score, cross-platform reuse)
    
    Labels simulated based on credential-stuffing attack patterns with campaign drift.
    """
    required = {"Breach_Source", "Email", "Password", "Timestamp", "AttackTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if device is None:
        device = get_device()

    # Normalize types
    df = df.copy()
    df["Breach_Source"] = df["Breach_Source"].astype(str)
    df["Email"] = df["Email"].astype(str)
    df["Password"] = df["Password"].astype(str).fillna("")
    df["Timestamp"] = to_datetime_safe(df["Timestamp"])
    df["AttackTime"] = to_datetime_safe(df["AttackTime"])

    # Sort by per-email AttackTime for stateful features
    df = df.sort_values(["Email", "AttackTime"], kind="mergesort").reset_index(drop=True)

    n_hw = build_cfg.n_honeywords
    hw_cols = [f"Honeyword_{i}" for i in range(1, n_hw + 1) if f"Honeyword_{i}" in df.columns]
    for c in hw_cols:
        df[c] = df[c].astype(str).fillna("")

    # Build TF-IDF similarity index with memory optimizations
    print("Building TF-IDF similarity index...")
    all_strings = _collect_all_strings(df, n_hw=n_hw)
    sim_index = GPUSimilarityIndex(
        all_strings, 
        ngram=build_cfg.tfidf_ngram, 
        max_features=build_cfg.max_features,
        device=device
    )

    # Initialize feature columns
    feat_cols = [
        "sim_to_prev_passwords_max",
        "sim_to_prev_passwords_mean",
        "exact_password_reuse",
        "sim_to_prev_honeywords_max",
        "sim_to_prev_honeywords_mean",
        "honeyword_trigger",
        "current_hw_matches_prev_pw",
        "time_since_last_leak",
        "time_since_last_attack",
        "attack_sequence_index",
        "delta_similarity",
        "num_platforms_seen",
        "platform_overlap_score",
        "cross_platform_reuse",
    ]
    for col in feat_cols:
        df[col] = 0.0

    # Per-email state tracking
    state_prev_pw_set: Dict[str, set] = {}
    state_prev_pw_indices: Dict[str, List[int]] = {}
    state_prev_hw_set: Dict[str, set] = {}
    state_prev_hw_indices: Dict[str, List[int]] = {}
    state_last_leak: Dict[str, pd.Timestamp] = {}
    state_last_attack: Dict[str, pd.Timestamp] = {}
    state_last_simmax: Dict[str, float] = {}
    state_seq_idx: Dict[str, int] = {}
    state_platforms_seen: Dict[str, set] = {}
    state_pw_to_platforms: Dict[str, Dict[str, set]] = {}

    print("Computing features...")
    for i in range(len(df)):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i}/{len(df)} rows...")
        
        row = df.iloc[i]
        email = row["Email"]
        pw = row["Password"]
        platform = row["Breach_Source"]
        cur_idx = sim_index.idx(pw)

        cur_hw = [row[c] for c in hw_cols if c in row.index]
        cur_hw = [h for h in cur_hw if isinstance(h, str) and h != ""]

        prev_pw_set = state_prev_pw_set.get(email, set())
        prev_pw_indices = state_prev_pw_indices.get(email, [])
        prev_hw_set = state_prev_hw_set.get(email, set())
        prev_hw_indices = state_prev_hw_indices.get(email, [])

        exact_reuse = 1.0 if pw in prev_pw_set else 0.0
        sim_pw_max, sim_pw_mean = sim_index.sims_to_history(cur_idx, prev_pw_indices)
        sim_hw_max, sim_hw_mean = sim_index.sims_to_history(cur_idx, prev_hw_indices)

        hw_trigger = 1.0 if any(h in prev_pw_set for h in cur_hw) else 0.0
        cur_hw_matches_prev_pw = 1.0 if any(h in prev_pw_set for h in cur_hw) else 0.0

        last_leak = state_last_leak.get(email, pd.NaT)
        last_attack = state_last_attack.get(email, pd.NaT)
        tsl = days_between(row["AttackTime"], last_leak)
        tsa = days_between(row["AttackTime"], last_attack)

        seq = int(state_seq_idx.get(email, 0))
        last_simmax = state_last_simmax.get(email, 0.0)
        delta_sim = sim_pw_max - last_simmax

        platforms_seen = state_platforms_seen.get(email, set())
        pw_map = state_pw_to_platforms.get(email, {})
        platforms_that_used_pw = pw_map.get(pw, set())
        
        num_platforms_seen = float(len(platforms_seen) if platforms_seen else 0)
        overlap_score = float(len(platforms_that_used_pw) / len(platforms_seen)) if platforms_seen else 0.0
        cross_platform_reuse = 1.0 if (len(platforms_that_used_pw - {platform}) > 0) else 0.0

        # Assign features
        df.at[i, "sim_to_prev_passwords_max"] = sim_pw_max
        df.at[i, "sim_to_prev_passwords_mean"] = sim_pw_mean
        df.at[i, "exact_password_reuse"] = exact_reuse
        df.at[i, "sim_to_prev_honeywords_max"] = sim_hw_max
        df.at[i, "sim_to_prev_honeywords_mean"] = sim_hw_mean
        df.at[i, "honeyword_trigger"] = hw_trigger
        df.at[i, "current_hw_matches_prev_pw"] = cur_hw_matches_prev_pw
        df.at[i, "time_since_last_leak"] = 0.0 if math.isnan(tsl) else float(max(tsl, 0.0))
        df.at[i, "time_since_last_attack"] = 0.0 if math.isnan(tsa) else float(max(tsa, 0.0))
        df.at[i, "attack_sequence_index"] = seq
        df.at[i, "delta_similarity"] = delta_sim
        df.at[i, "num_platforms_seen"] = num_platforms_seen
        df.at[i, "platform_overlap_score"] = overlap_score
        df.at[i, "cross_platform_reuse"] = cross_platform_reuse

        # Update state AFTER computing features
        state_last_leak[email] = row["Timestamp"] if not pd.isna(row["Timestamp"]) else last_leak
        state_last_attack[email] = row["AttackTime"] if not pd.isna(row["AttackTime"]) else last_attack
        state_last_simmax[email] = sim_pw_max
        state_seq_idx[email] = int(seq) + 1

        state_prev_pw_set.setdefault(email, set()).add(pw)
        state_prev_pw_indices.setdefault(email, []).append(cur_idx)
        state_prev_hw_set.setdefault(email, set()).update([h for h in cur_hw if h != ""])
        state_prev_hw_indices.setdefault(email, []).extend([sim_index.idx(h) for h in cur_hw if h != ""])

        state_platforms_seen.setdefault(email, set()).add(platform)
        state_pw_to_platforms.setdefault(email, {}).setdefault(pw, set()).add(platform)

    # Label simulation with campaign drift
    print("Simulating labels with campaign drift...")
    df["y"] = 0

    df["_rank_in_email"] = df.groupby("Email").cumcount()
    df["_len_in_email"] = df.groupby("Email")["Email"].transform("size")
    df["_frac_in_email"] = (df["_rank_in_email"] / (df["_len_in_email"].clip(lower=1) - 1).replace(0, 1)).astype(float)

    sim_thr = build_cfg.sim_threshold_noisy
    early_frac = build_cfg.early_frac
    leak_days = build_cfg.cross_platform_leak_days

    base_exact = df["exact_password_reuse"] >= 1.0
    base_honey = df["honeyword_trigger"] >= 1.0
    base_cross = (df["cross_platform_reuse"] >= 1.0) & (df["time_since_last_leak"] < float(leak_days))

    late = df["_frac_in_email"] > early_frac
    noisy_sim = df["sim_to_prev_passwords_max"] > sim_thr

    y_late = (base_exact | base_honey | base_cross | noisy_sim).astype(int)
    y_early = (base_exact).astype(int)

    df["y"] = np.where(late, y_late, y_early).astype(int)

    df = df.drop(columns=["_rank_in_email", "_len_in_email", "_frac_in_email"])

    print(f"Label distribution: {df['y'].value_counts().to_dict()}")
    
    return df


# -------------------------
# Temporal Split
# -------------------------

def temporal_split(
    df: pd.DataFrame, 
    split_cfg: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally by AttackTime into train/val/test.
    """
    df = df.sort_values("AttackTime", kind="mergesort").reset_index(drop=True)
    n = len(df)
    n_train = int(n * split_cfg.train_frac)
    n_val = int(n * split_cfg.val_frac)
    
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    
    assert len(train) + len(val) + len(test) == n
    
    print(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test


# -------------------------
# Caching
# -------------------------

def cache_paths(out_dir: str) -> Dict[str, str]:
    return {
        "features": os.path.join(out_dir, "features.parquet"),
        "train": os.path.join(out_dir, "train.parquet"),
        "val": os.path.join(out_dir, "val.parquet"),
        "test": os.path.join(out_dir, "test.parquet"),
        "build_meta": os.path.join(out_dir, "build_meta.json"),
    }

def build_and_cache(
    csv_path: str, 
    out_dir: str, 
    build_cfg: BuildConfig, 
    split_cfg: SplitConfig,
    device: Optional[torch.device] = None
) -> None:
    """
    Build features, labels, and splits, then cache to disk.
    """
    ensure_dir(out_dir)
    
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    df2 = build_features_and_labels(df, build_cfg, device=device)
    train, val, test = temporal_split(df2, split_cfg)

    paths = cache_paths(out_dir)
    
    print(f"Saving cached data to {out_dir}...")
    df2.to_parquet(paths["features"], index=False)
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)

    meta = {
        "csv": csv_path,
        "rows": int(len(df2)),
        "platforms": int(df2["Breach_Source"].nunique()),
        "unique_emails": int(df2["Email"].nunique()),
        "positive_labels": int(df2["y"].sum()),
        "negative_labels": int((df2["y"] == 0).sum()),
        "build_cfg": asdict(build_cfg),
        "split_cfg": asdict(split_cfg),
    }
    save_json(meta, paths["build_meta"])
    
    print("Stage 1 complete!")


def load_cached(out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached train/val/test splits from disk."""
    paths = cache_paths(out_dir)
    
    for key in ["train", "val", "test"]:
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Missing cached {key} split in {out_dir}. Run stage 1 first.")
    
    train = pd.read_parquet(paths["train"])
    val = pd.read_parquet(paths["val"])
    test = pd.read_parquet(paths["test"])
    
    # Ensure datetime after parquet
    for df in [train, val, test]:
        df["AttackTime"] = pd.to_datetime(df["AttackTime"], errors="coerce")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    
    return train, val, test


# -------------------------
# Main
# -------------------------

def main():
    # =========================
    # USER CONFIGURATION
    # =========================
    CSV_PATH = "Merged_breaches.csv"
    OUT_DIR = "outputFolder"

    # Build config
    N_HONEYWORDS = 50
    TFIDF_NGRAM = 3
    MAX_FEATURES = 5000
    SIM_THRESHOLD_NOISY = 0.85
    EARLY_FRAC = 0.40
    CROSS_PLATFORM_LEAK_DAYS = 30

    # Split config
    TRAIN_FRAC = 0.60
    VAL_FRAC = 0.20
    TEST_FRAC = 0.20

    # Device (set to None for auto-detect)
    DEVICE_NAME = None  # "cuda", "cpu", "mps", or None


    # =========================
    # DEVICE SETUP
    # =========================
    if DEVICE_NAME:
        device = torch.device(DEVICE_NAME)
    else:
        device = get_device()


    # =========================
    # CONFIG OBJECTS
    # =========================
    build_cfg = BuildConfig(
        n_honeywords=N_HONEYWORDS,
        tfidf_ngram=TFIDF_NGRAM,
        max_features=MAX_FEATURES,
        sim_threshold_noisy=SIM_THRESHOLD_NOISY,
        early_frac=EARLY_FRAC,
        cross_platform_leak_days=CROSS_PLATFORM_LEAK_DAYS,
    )

    split_cfg = SplitConfig(
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )


    # =========================
    # RUN PIPELINE
    # =========================
    build_and_cache(
        csv_path=CSV_PATH,
        out_dir=OUT_DIR,
        build_cfg=build_cfg,
        split_cfg=split_cfg,
        device=device
    )


if __name__ == "__main__":
    main()
