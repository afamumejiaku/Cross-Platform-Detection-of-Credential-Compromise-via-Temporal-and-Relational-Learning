
#!/usr/bin/env python3
"""
Breach detection
This module handles:
- LSTM classifier (PyTorch with CUDA)
- GRU classifier (PyTorch with CUDA)
- FTA-GRU simplified classifier (PyTorch with CUDA)
- FTA-LSTM simplified classifier (PyTorch with CUDA)
- Temporal GNN classifier (timesteps-as-nodes; PyTorch with CUDA)
- IFCNN-TPP classifier (CNN + temporal encoder; PyTorch with CUDA)
- CEP3 classifier (dilated temporal convolution encoder; PyTorch with CUDA)
- GCN bipartite classifier (PyTorch with CUDA)
- GAT bipartite classifier (PyTorch with CUDA)

All models support:
- Baseline A: Per-platform models
- Model B: Cross-platform model 
GPU Acceleration:
- All PyTorch models use CUDA/MPS when available
- Automatic mixed precision (AMP) for faster training
"""


from __future__ import annotations

import os
import json
import time
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ---- Multiprocessing FD pressure fix (Linux) ----
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass


# -------------------------
# Configuration
# -------------------------

BASE_FEATURES = [
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
]

CROSS_FEATURES = [
    "num_platforms_seen",
    "platform_overlap_score",
    "cross_platform_reuse",
]

ID_COLS = ["Email", "Breach_Source", "AttackTime", "Timestamp"]


# -------------------------
# Utils
# -------------------------

def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def make_loader(dataset, batch_size, shuffle, device, setting: str):
    """
    SAFE DataLoader builder (prevents Errno 24 "Too many open files"):

    - We force num_workers=0 for ALL settings/models because you are running a grid
      (cross_platform + per_platform + multiple models) inside one long process.
    - persistent_workers is always False.
    """
    pin = (device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,              # IMPORTANT: avoid FD exhaustion
        pin_memory=pin,
        persistent_workers=False,   # IMPORTANT: don't keep workers around
    )


# -------------------------
# Data Loading
# -------------------------

def load_cached(out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached train/val/test splits from Stage 1."""
    paths = {
        "train": os.path.join(out_dir, "train.parquet"),
        "val": os.path.join(out_dir, "val.parquet"),
        "test": os.path.join(out_dir, "test.parquet"),
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run Stage 1 first.")

    train = pd.read_parquet(paths["train"])
    val = pd.read_parquet(paths["val"])
    test = pd.read_parquet(paths["test"])

    for df in [train, val, test]:
        df["AttackTime"] = pd.to_datetime(df["AttackTime"], errors="coerce")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return train, val, test


# -------------------------
# Design Matrices
# -------------------------

def build_design_matrices(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    setting: str,
    platform_encoder: Optional[OneHotEncoder] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[OneHotEncoder], Dict]:
    """
    Build feature matrices based on setting:
    - per_platform: base features only
    - cross_platform: base + cross features + one-hot platform
    """
    if setting not in {"per_platform", "pooled", "cross_platform"}:
        raise ValueError(f"Unknown setting={setting}")

    use_cross = (setting == "cross_platform")
    feat_cols = list(BASE_FEATURES) + (list(CROSS_FEATURES) if use_cross else [])

    def _num_df(d: pd.DataFrame) -> np.ndarray:
        return d[feat_cols].astype(float).values

    Xtr_num, Xva_num, Xte_num = _num_df(train), _num_df(val), _num_df(test)

    if setting == "cross_platform":
        if platform_encoder is None:
            platform_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            platform_encoder.fit(train[["Breach_Source"]].astype(str))
        Ptr = platform_encoder.transform(train[["Breach_Source"]].astype(str))
        Pva = platform_encoder.transform(val[["Breach_Source"]].astype(str))
        Pte = platform_encoder.transform(test[["Breach_Source"]].astype(str))
        Xtr = np.concatenate([Xtr_num, Ptr], axis=1)
        Xva = np.concatenate([Xva_num, Pva], axis=1)
        Xte = np.concatenate([Xte_num, Pte], axis=1)
    else:
        Xtr, Xva, Xte = Xtr_num, Xva_num, Xte_num

    ytr = train["y"].astype(int).values
    yva = val["y"].astype(int).values
    yte = test["y"].astype(int).values

    meta = {
        "feat_cols": feat_cols,
        "setting": setting,
        "platform_onehot_dim": int(Xtr.shape[1] - len(feat_cols)) if setting == "cross_platform" else 0,
        "X_dim": int(Xtr.shape[1]),
    }
    return Xtr, ytr, Xva, yva, Xte, yte, platform_encoder, meta


# -------------------------
# Metrics (Accuracy-based threshold)
# -------------------------

def choose_threshold_by_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold by ACCURACY on validation set."""
    best_thr, best_acc = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    """Compute evaluation metrics (accuracy, F1, ROC-AUC, etc.)."""
    y_pred = (y_prob >= thr).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    out = {
        "threshold": float(thr),
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = None
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["auprc"] = None

    return out


# -------------------------
# Sequence Dataset
# -------------------------

class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_sequences_by_email(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    include_current: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequences per email for RNN models."""
    df = df.sort_values(["Email", "AttackTime"], kind="mergesort")
    F = len(feature_cols)

    X_list, y_list, idx_list = [], [], []

    for email, g in df.groupby("Email", sort=False):
        g = g.sort_values("AttackTime", kind="mergesort")
        feats = g[feature_cols].astype(float).values
        ys = g["y"].astype(int).values
        inds = g.index.values

        for i in range(len(g)):
            end = i + 1 if include_current else i
            start = max(0, end - seq_len)
            seq = feats[start:end]

            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], F), dtype=np.float32)
                seq = np.vstack([pad, seq]).astype(np.float32)

            X_list.append(seq)
            y_list.append(ys[i])
            idx_list.append(inds[i])

    return (
        np.stack(X_list, axis=0).astype(np.float32),
        np.array(y_list, dtype=np.int64),
        np.array(idx_list, dtype=np.int64),
    )


# -------------------------
# PyTorch Models (Sequence)
# -------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class FTAGru(nn.Module):
    """
    - Feature attention per timestep
    - Temporal GRU encoder
    - Temporal attention pooling
    """
    def __init__(self, input_dim: int, hidden: int = 128, attn_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.feature_gate(x)
        xg = x * gates
        h, _ = self.gru(xg)
        a = self.temporal_attn(h).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class FTALstm(nn.Module):
    """
    - Feature attention per timestep
    - Temporal LSTM encoder
    - Temporal attention pooling
    """
    def __init__(self, input_dim: int, hidden: int = 128, attn_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.feature_gate(x)
        xg = x * gates
        h, _ = self.lstm(xg)
        a = self.temporal_attn(h).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.head(pooled).squeeze(-1)

# -------------------------
# IFCNN_TPP (CNN + GRU + attention pooling)
# -------------------------

class IFCNNTpp(nn.Module):
    """
    IFCNN_TPP-style binary classifier:
      - 1D conv over time (feature channels)
      - GRU encoder
      - Attention pooling
      - MLP head -> logits

    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        conv_channels: int = 128,
        kernel_size: int = 3,
        gru_layers: int = 1,
        attn_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2

        # Conv expects (B, C, T) where C=input_dim (treat features as channels)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Linear(conv_channels, hidden)

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)                 # (B, F, T)
        z = self.conv(x)                      # (B, C, T)
        z = z.transpose(1, 2)                 # (B, T, C)
        z = self.proj(z)                      # (B, T, H)

        h, _ = self.gru(z)                    # (B, T, H)

        a = self.temporal_attn(h).squeeze(-1) # (B, T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)           # (B, H)

        logits = self.head(pooled).squeeze(-1)
        return logits


# -------------------------
# CEP3 (Causal dilated conv stack + attention pooling)
# -------------------------

class CEP3(nn.Module):
    """
    CEP3-style temporal classifier:
      - linear projection to hidden
      - 3-layer causal-ish dilated conv over time
      - attention pooling
      - MLP head -> logits
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.2,
        attn_dim: int = 64,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden)

        # We'll do "same" padding then mask via slicing for a causal feel.
        # This is stable and avoids heavy TPP math.
        def conv_block(dilation: int):
            pad = (kernel_size - 1) * dilation
            return nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=kernel_size, dilation=dilation, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.conv1 = conv_block(dilation=1)
        self.conv2 = conv_block(dilation=2)
        self.conv3 = conv_block(dilation=4)

        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.in_proj(x)                   # (B, T, H)
        h = h.transpose(1, 2)                 # (B, H, T)

        # Convs with padding; slice to keep length T (causal-ish)
        T = h.size(-1)
        h1 = self.conv1(h)[..., :T]
        h2 = self.conv2(h1)[..., :T]
        h3 = self.conv3(h2)[..., :T]

        out = h3.transpose(1, 2)              # (B, T, H)

        a = self.temporal_attn(out).squeeze(-1)  # (B, T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (out * w).sum(dim=1)            # (B, H)

        logits = self.head(pooled).squeeze(-1)
        return logits


# -------------------------
# Temporal GNN (timesteps-as-nodes) for sequences
# -------------------------

class TemporalGraphAttentionLayer(nn.Module):
    """
    Dense (batch) graph-attention over timesteps.
    Treat each timestep as a node; computes attention between all timestep pairs.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)

        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, Fin)
        adj: optional (T, T) or (B, T, T) binary mask where 0 means "no edge"
        returns: (B, T, Fout)
        """
        B, T, _ = x.shape
        h = torch.matmul(x, self.W)  # (B, T, Fout)

        # Build pairwise concat (B, T, T, 2*Fout)
        h1 = h.unsqueeze(2).expand(B, T, T, h.size(-1))
        h2 = h.unsqueeze(1).expand(B, T, T, h.size(-1))
        hcat = torch.cat([h1, h2], dim=-1)

        e = self.leakyrelu(torch.matmul(hcat, self.a).squeeze(-1))  # (B, T, T)

        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)  # (1, T, T)
            e = e.masked_fill(adj == 0, -1e9)

        attn = torch.softmax(e, dim=-1)  # along neighbors
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = torch.matmul(attn, h)  # (B, T, Fout)
        return out


class TemporalGNNClassifier(nn.Module):
    """
    Temporal GNN classifier:
      - GRU feature extraction over time
      - Multi-head temporal graph attention (timesteps as nodes)
      - Temporal self-attention
      - Mean pooling -> MLP head

    Works with your SeqDataset and train_torch_binary().
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        gru_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden = hidden
        self.gat_heads = gat_heads

        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Multi-head temporal graph attention
        # Each head outputs hidden/gat_heads dims; concatenation returns hidden
        head_dim = max(1, hidden // gat_heads)
        self.gat_layers = nn.ModuleList([
            TemporalGraphAttentionLayer(hidden, head_dim, dropout=dropout)
            for _ in range(gat_heads)
        ])

        self.layer_norm = nn.LayerNorm(head_dim * gat_heads)

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=head_dim * gat_heads,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(head_dim * gat_heads, (head_dim * gat_heads) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((head_dim * gat_heads) // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, Fin)
        h, _ = self.gru(x)  # (B, T, hidden)

        # Dense temporal GAT over timesteps
        gat_outs = [gat(h) for gat in self.gat_layers]          # list of (B, T, head_dim)
        h_gat = torch.cat(gat_outs, dim=-1)                     # (B, T, head_dim*heads)

        # Residual-ish normalization (dims match by construction)
        h_gat = self.layer_norm(h_gat)

        # Temporal self-attention
        attn_out, _ = self.temporal_attention(h_gat, h_gat, h_gat)

        # Pool and classify
        pooled = attn_out.mean(dim=1)                           # (B, D)
        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)                  # (B,)
        return logits

# -------------------------
# Graph Components (Bipartite Email-Platform)
# -------------------------

def build_bipartite_graph(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], torch.Tensor]:
    """Build bipartite email-platform graph."""
    emails = sorted(train_df["Email"].astype(str).unique().tolist())
    plats = sorted(train_df["Breach_Source"].astype(str).unique().tolist())
    email2n = {e: i for i, e in enumerate(emails)}
    plat2n = {p: i for i, p in enumerate(plats)}

    E, P = len(emails), len(plats)
    edges = set()

    for _, r in train_df[["Email", "Breach_Source"]].drop_duplicates().iterrows():
        u = email2n[str(r["Email"])]
        v = E + plat2n[str(r["Breach_Source"])]
        edges.add((u, v))
        edges.add((v, u))

    if not edges:
        raise ValueError("No edges in bipartite graph")

    idx = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    N = E + P
    val = torch.ones(idx.shape[1], dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, (N, N))

    return email2n, plat2n, A.coalesce()


def normalize_sparse_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    N = A.shape[0]

    deg = torch.zeros(N, dtype=torch.float32, device=val.device)
    deg.index_add_(0, idx[0], val)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    norm_val = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]

    return torch.sparse_coo_tensor(idx, norm_val, (N, N)).coalesce()


# -------------------------
# GCN Components
# -------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, A_norm: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Sparse matmul doesn't support fp16 on many CUDA builds.
        # Force this operation to fp32 even if AMP/autocast is enabled.
        if X.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                AX = torch.sparse.mm(A_norm.float(), X.float())
        else:
            AX = torch.sparse.mm(A_norm, X)
        return self.lin(AX)


class BipartiteGCNClassifier(nn.Module):
    def __init__(self, node_in_dim: int, gcn_hidden: int, gcn_out: int, event_in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(node_in_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(event_in_dim + 2 * gcn_out, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        A_norm: torch.Tensor,
        X_nodes: torch.Tensor,
        email_nodes: torch.Tensor,
        plat_nodes: torch.Tensor,
        X_event: torch.Tensor,
    ) -> torch.Tensor:
        h = self.act(self.gcn1(A_norm, X_nodes))
        h = self.dropout(h)
        h = self.gcn2(A_norm, h)

        he = h[email_nodes]
        hp = h[plat_nodes]
        z = torch.cat([X_event, he, hp], dim=1)
        return self.mlp(z).squeeze(-1)


# -------------------------
# GAT Components (COO edges; AMP-safe softmax)
# -------------------------

class GATLayer(nn.Module):
    """
    Sparse GAT layer using COO edge list.
    Computes attention only over edges in edge_index.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Softmax over incoming edges per destination node.
        Runs in FP32 for numerical stability under AMP, then casts back.
        """
        orig_dtype = scores.dtype
        scores_f = scores.float()  # fp32

        max_per_dst = torch.full((num_nodes,), -1e9, device=scores.device, dtype=torch.float32)
        max_per_dst.scatter_reduce_(0, dst, scores_f, reduce="amax", include_self=True)

        scores_f = (scores_f - max_per_dst[dst]).clamp(min=-20.0, max=20.0)
        scores_exp = torch.exp(scores_f)

        denom = torch.zeros((num_nodes,), device=scores.device, dtype=torch.float32)
        denom.scatter_add_(0, dst, scores_exp)

        out = scores_exp / denom[dst].clamp_min(1e-12)
        return out.to(orig_dtype)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        H = self.W(X)  # [N, Fout]

        src, dst = edge_index[0], edge_index[1]  # [E], [E]
        h_src = H[src]
        h_dst = H[dst]

        e = self.attn(torch.cat([h_src, h_dst], dim=1)).squeeze(-1)
        e = self.leakyrelu(e)

        alpha = self._segment_softmax(e, dst, num_nodes)
        alpha = self.dropout(alpha)

        out = torch.zeros((num_nodes, H.shape[1]), device=H.device, dtype=H.dtype)
        out.index_add_(0, dst, h_src * alpha.unsqueeze(-1))
        return out


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([GATLayer(in_dim, out_dim, dropout=dropout) for _ in range(heads)])

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        outs = [h(X, edge_index, num_nodes) for h in self.heads]
        return torch.cat(outs, dim=1)


class BipartiteGATClassifier(nn.Module):
    """Bipartite GAT classifier over the email-platform graph."""
    def __init__(
        self,
        node_in_dim: int,
        gat_hidden: int,
        gat_out: int,
        heads1: int,
        heads2: int,
        event_in_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

        self.gat1 = MultiHeadGATLayer(node_in_dim, gat_hidden, heads=heads1, dropout=dropout)
        self.gat2 = MultiHeadGATLayer(gat_hidden * heads1, gat_out, heads=heads2, dropout=dropout)

        gnn_dim = gat_out * heads2
        self.mlp = nn.Sequential(
            nn.Linear(event_in_dim + 2 * gnn_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        X_nodes: torch.Tensor,
        email_nodes: torch.Tensor,
        plat_nodes: torch.Tensor,
        X_event: torch.Tensor,
    ) -> torch.Tensor:
        N = X_nodes.shape[0]
        h = self.gat1(X_nodes, edge_index, N)
        h = self.act(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index, N)
        h = self.act(h)

        he = h[email_nodes]
        hp = h[plat_nodes]
        z = torch.cat([X_event, he, hp], dim=1)
        return self.mlp(z).squeeze(-1)


class GraphDataset(Dataset):
    def __init__(self, X_event: np.ndarray, y: np.ndarray, email_nodes: np.ndarray, plat_nodes: np.ndarray):
        self.Xe = torch.tensor(X_event, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.en = torch.tensor(email_nodes, dtype=torch.long)
        self.pn = torch.tensor(plat_nodes, dtype=torch.long)

    def __len__(self) -> int:
        return self.Xe.shape[0]

    def __getitem__(self, idx: int):
        return self.Xe[idx], self.y[idx], self.en[idx], self.pn[idx]


# -------------------------
# Training Loops with AMP (Accuracy Early Stop)
# -------------------------

def train_torch_binary(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 30,
    patience: int = 8,
    class_weight_pos: Optional[float] = None,
    use_amp: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train sequence model with AMP support. Uses ACCURACY for early stopping."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    if class_weight_pos is None:
        bce = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor([class_weight_pos], dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_acc = -1.0
    bad_epochs = 0
    history = {"val_acc": [], "val_f1": [], "val_loss": [], "train_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()

            if use_amp:
                with autocast():
                    logits = model(xb)
                    loss = bce(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = bce(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            train_losses.append(float(loss.item()))

        model.eval()
        all_prob, all_y, val_losses = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        logits = model(xb)
                        loss = bce(logits, yb)
                else:
                    logits = model(xb)
                    loss = bce(logits, yb)

                val_losses.append(float(loss.item()))
                prob = torch.sigmoid(logits).cpu().numpy()
                all_prob.append(prob)
                all_y.append(yb.cpu().numpy())

        y_prob = np.concatenate(all_prob)
        y_true = np.concatenate(all_y).astype(int)

        thr = choose_threshold_by_accuracy(y_true, y_prob)
        met = compute_metrics(y_true, y_prob, thr)
        val_acc = met["accuracy"]
        val_f1 = met["f1"]

        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_loss"].append(float(np.mean(val_losses)))
        history["train_loss"].append(float(np.mean(train_losses)))

        print(
            f"  Epoch {ep}: "
            f"train_loss={np.mean(train_losses):.4f}, "
            f"val_loss={np.mean(val_losses):.4f}, "
            f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_acc": best_val_acc, "history": history}


def predict_torch_seq(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True) -> np.ndarray:
    """Predict with sequence model."""
    model.eval()
    use_amp = use_amp and device.type == "cuda"
    probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            if use_amp:
                with autocast():
                    logits = model(xb)
            else:
                logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(prob)

    return np.concatenate(probs).astype(float)


def train_graph_binary(
    model: nn.Module,
    graph_obj: torch.Tensor,        # A_norm (sparse) for GCN or edge_index (long) for GAT
    X_nodes: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 30,
    patience: int = 8,
    class_weight_pos: Optional[float] = None,
    use_amp: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train GCN/GAT classifier with AMP support. Uses ACCURACY for early stopping."""
    model = model.to(device)
    graph_obj = graph_obj.to(device)
    X_nodes = X_nodes.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    if class_weight_pos is None:
        bce = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor([class_weight_pos], dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_acc = -1.0
    bad_epochs = 0
    history = {"val_acc": [], "val_f1": []}

    for ep in range(1, epochs + 1):
        model.train()

        for Xe, yb, en, pn in train_loader:
            Xe, yb = Xe.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            en, pn = en.to(device, non_blocking=True), pn.to(device, non_blocking=True)

            opt.zero_grad()

            if use_amp:
                with autocast():
                    logits = model(graph_obj, X_nodes, en, pn, Xe)
                    loss = bce(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(graph_obj, X_nodes, en, pn, Xe)
                loss = bce(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        model.eval()
        all_prob, all_y = [], []

        with torch.no_grad():
            for Xe, yb, en, pn in val_loader:
                Xe, yb = Xe.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                en, pn = en.to(device, non_blocking=True), pn.to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        logits = model(graph_obj, X_nodes, en, pn, Xe)
                else:
                    logits = model(graph_obj, X_nodes, en, pn, Xe)

                prob = torch.sigmoid(logits).cpu().numpy()
                all_prob.append(prob)
                all_y.append(yb.cpu().numpy())

        y_prob = np.concatenate(all_prob)
        y_true = np.concatenate(all_y).astype(int)

        thr = choose_threshold_by_accuracy(y_true, y_prob)
        met = compute_metrics(y_true, y_prob, thr)
        val_acc = met["accuracy"]
        val_f1 = met["f1"]

        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {ep}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_acc": best_val_acc, "history": history}

 =============================================================================
# SUMMARY TABLE PRINTING (requested)
# =============================================================================
def predict_graph(
    model: nn.Module,
    graph_obj: torch.Tensor,
    X_nodes: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> np.ndarray:
    """Predict with GCN/GAT model."""
    model.eval()
    graph_obj = graph_obj.to(device)
    X_nodes = X_nodes.to(device)

    use_amp = use_amp and device.type == "cuda"
    probs = []

    with torch.no_grad():
        for Xe, _, en, pn in loader:
            Xe = Xe.to(device, non_blocking=True)
            en, pn = en.to(device, non_blocking=True), pn.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    logits = model(graph_obj, X_nodes, en, pn, Xe)
            else:
                logits = model(graph_obj, X_nodes, en, pn, Xe)

            prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(prob)

    return np.concatenate(probs).astype(float)


# -------------------------
# Main Training Orchestration
# -------------------------

def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str,
    setting: str,
    model_name: str,
    seq_len: int = 10,
    batch: int = 256,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    seed: int = 42,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    seed_everything(seed)

    if device is None:
        device = get_device()

    results = {
        "setting": setting,
        "model": model_name,
        "seq_len": seq_len,
        "seed": seed,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if setting == "per_platform":
        all_platforms = sorted(train["Breach_Source"].astype(str).unique().tolist())
        per_plat_metrics = {}
        per_plat_pred_files = {}

        for plat in all_platforms:
            trp = train[train["Breach_Source"] == plat].copy()
            vap = val[val["Breach_Source"] == plat].copy()
            tep = test[test["Breach_Source"] == plat].copy()

            if len(trp) < 50 or len(vap) < 20 or len(tep) < 20:
                print(f"Skipping platform {plat}: insufficient data")
                continue

            print(f"\nTraining for platform: {plat}")
            plat_out = os.path.join(out_dir, f"per_platform_{plat}")
            ensure_dir(plat_out)

            met, pred_path, thr = _train_eval_single(
                trp, vap, tep, plat_out, setting="per_platform", model_name=model_name,
                seq_len=seq_len, batch=batch, epochs=epochs, lr=lr,
                weight_decay=weight_decay, patience=patience, seed=seed,
                device=device, use_amp=use_amp
            )
            per_plat_metrics[plat] = met
            per_plat_pred_files[plat] = {"predictions_csv": pred_path, "threshold": thr}

            # Optional cleanup for long loops
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["per_platform_metrics"] = per_plat_metrics
        results["per_platform_predictions"] = per_plat_pred_files
        save_json(results, os.path.join(out_dir, f"results_{setting}_{model_name}.json"))
        return results

    met, pred_path, thr = _train_eval_single(
        train, val, test, out_dir, setting=setting, model_name=model_name,
        seq_len=seq_len, batch=batch, epochs=epochs, lr=lr,
        weight_decay=weight_decay, patience=patience, seed=seed,
        device=device, use_amp=use_amp
    )
    results["metrics"] = met
    results["predictions_csv"] = pred_path
    results["threshold"] = thr
    save_json(results, os.path.join(out_dir, f"results_{setting}_{model_name}.json"))

    return results


def _train_eval_single(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str,
    setting: str,
    model_name: str,
    seq_len: int,
    batch: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[Dict[str, Any], str, float]:
    ensure_dir(out_dir)

    Xtr, ytr, Xva, yva, Xte, yte, platform_encoder, meta = build_design_matrices(
        train, val, test, setting=setting, platform_encoder=None
    )

    pos = max(int((ytr == 1).sum()), 1)
    neg = max(int((ytr == 0).sum()), 1)
    class_weight_pos = float(neg / pos)
    print(f"  Class balance: {pos} positive, {neg} negative (weight={class_weight_pos:.2f})")

    # -------------------------
    # Sequence models
    # -------------------------
    if model_name in {"lstm", "gru", "fta_gru", "fta_lstm", "gnn", "ifcnn_tpp", "cep3"}:
        print(f"Training {model_name.upper()}...")

        use_cross = (setting == "cross_platform")
        feature_cols = list(BASE_FEATURES) + (list(CROSS_FEATURES) if use_cross else [])

        if setting == "cross_platform":
            platform_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            platform_encoder.fit(train[["Breach_Source"]].astype(str))

            def add_plat_onehot(d: pd.DataFrame):
                oh = platform_encoder.transform(d[["Breach_Source"]].astype(str))
                oh_cols = [f"plat_oh_{i}" for i in range(oh.shape[1])]
                dd = d.copy()
                for j, c in enumerate(oh_cols):
                    dd[c] = oh[:, j]
                return dd, oh_cols

            train2, oh_cols = add_plat_onehot(train)
            val2, _ = add_plat_onehot(val)
            test2, _ = add_plat_onehot(test)
            feature_cols = feature_cols + oh_cols
        else:
            train2, val2, test2 = train, val, test

        Xtr_seq, ytr_seq, idx_tr = make_sequences_by_email(train2, feature_cols, seq_len=seq_len)
        Xva_seq, yva_seq, idx_va = make_sequences_by_email(val2, feature_cols, seq_len=seq_len)
        Xte_seq, yte_seq, idx_te = make_sequences_by_email(test2, feature_cols, seq_len=seq_len)

        tr_loader = make_loader(SeqDataset(Xtr_seq, ytr_seq), batch, True,  device, setting)
        va_loader = make_loader(SeqDataset(Xva_seq, yva_seq), batch, False, device, setting)
        te_loader = make_loader(SeqDataset(Xte_seq, yte_seq), batch, False, device, setting)

        input_dim = Xtr_seq.shape[-1]

        if model_name == "lstm":
            net = LSTMClassifier(input_dim=input_dim, hidden=128, layers=2, dropout=0.2)
        elif model_name == "gru":
            net = GRUClassifier(input_dim=input_dim, hidden=128, layers=2, dropout=0.2)
        elif model_name == "fta_gru":
            net = FTAGru(input_dim=input_dim, hidden=128, attn_dim=64, dropout=0.2)
        elif model_name == "fta_lstm":
            net = FTALstm(input_dim=input_dim, hidden=128, attn_dim=64, dropout=0.2)
        elif model_name == "gnn":
            net = TemporalGNNClassifier(input_dim=input_dim, hidden=128, gru_layers=2, gat_heads=4, dropout=0.2)
        elif model_name == "ifcnn_tpp":
            net = IFCNNTpp(input_dim=input_dim, hidden=128, conv_channels=128, kernel_size=3, gru_layers=1, dropout=0.2)
        elif model_name == "cep3":
            net = CEP3(input_dim=input_dim, hidden=128, kernel_size=3, dropout=0.2)
        else:
            raise ValueError(f"Unhandled model_name: {model_name}")


        net, train_info = train_torch_binary(
            net, tr_loader, va_loader, device=device,
            lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience,
            class_weight_pos=class_weight_pos, use_amp=use_amp
        )

        va_prob = predict_torch_seq(net, va_loader, device=device, use_amp=use_amp)
        best_thr = choose_threshold_by_accuracy(yva_seq, va_prob)
        te_prob = predict_torch_seq(net, te_loader, device=device, use_amp=use_amp)

        metrics = compute_metrics(yte_seq, te_prob, best_thr)
        metrics["train_info"] = train_info

        torch.save(net.state_dict(), os.path.join(out_dir, f"model_{model_name}.pt"))
        save_json({"meta": meta, "feature_cols": feature_cols}, os.path.join(out_dir, "meta.json"))

        test_pred_df = test2.loc[idx_te, ID_COLS + ["y"]].copy()
        test_pred_df["pred_prob"] = te_prob
        pred_path = os.path.join(out_dir, "predictions_test.csv")
        test_pred_df.to_csv(pred_path, index=False)

        return metrics, pred_path, best_thr

    # -------------------------
    # Graph models: GCN + GAT
    # -------------------------
    if model_name in {"gcn", "gat"}:
        if setting != "cross_platform":
            raise ValueError(f"{model_name.upper()} requires setting='cross_platform'")

        print(f"Training {model_name.upper()}...")

        platform_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        platform_encoder.fit(train[["Breach_Source"]].astype(str))

        def event_X(d: pd.DataFrame) -> np.ndarray:
            Xn = d[list(BASE_FEATURES) + list(CROSS_FEATURES)].astype(float).values
            oh = platform_encoder.transform(d[["Breach_Source"]].astype(str))
            return np.concatenate([Xn, oh], axis=1)

        Xtr_e, Xva_e, Xte_e = event_X(train), event_X(val), event_X(test)
        ytr_e = train["y"].astype(int).values
        yva_e = val["y"].astype(int).values
        yte_e = test["y"].astype(int).values

        email2n, plat2n, A = build_bipartite_graph(train)
        A_norm = normalize_sparse_adj(A)
        edge_index = A.indices()  # COO edges for GAT

        E, P = len(email2n), len(plat2n)
        N = E + P

        node_in_dim = Xtr_e.shape[1]
        X_nodes = np.zeros((N, node_in_dim), dtype=np.float32)
        counts = np.zeros(N, dtype=np.float32)

        # Aggregate event features into node features (mean per node)
        for i, r in train.iterrows():
            e, p = str(r["Email"]), str(r["Breach_Source"])
            pos_i = train.index.get_loc(i)

            if e in email2n:
                ni = email2n[e]
                X_nodes[ni] += Xtr_e[pos_i]
                counts[ni] += 1.0
            if p in plat2n:
                nj = E + plat2n[p]
                X_nodes[nj] += Xtr_e[pos_i]
                counts[nj] += 1.0

        counts = np.clip(counts, 1.0, None)
        X_nodes = (X_nodes / counts[:, None]).astype(np.float32)

        def map_nodes(d: pd.DataFrame):
            en = np.array([email2n.get(str(e), 0) for e in d["Email"].astype(str).values], dtype=np.int64)
            pn = np.array([E + plat2n.get(str(p), 0) for p in d["Breach_Source"].astype(str).values], dtype=np.int64)
            return en, pn

        en_tr, pn_tr = map_nodes(train)
        en_va, pn_va = map_nodes(val)
        en_te, pn_te = map_nodes(test)

        tr_loader = make_loader(GraphDataset(Xtr_e, ytr_e, en_tr, pn_tr), batch, True,  device, setting)
        va_loader = make_loader(GraphDataset(Xva_e, yva_e, en_va, pn_va), batch, False, device, setting)
        te_loader = make_loader(GraphDataset(Xte_e, yte_e, en_te, pn_te), batch, False, device, setting)

        if model_name == "gcn":
            net = BipartiteGCNClassifier(
                node_in_dim=node_in_dim,
                gcn_hidden=128,
                gcn_out=64,
                event_in_dim=Xtr_e.shape[1],
                dropout=0.2
            )
            graph_obj = A_norm
        else:
            net = BipartiteGATClassifier(
                node_in_dim=node_in_dim,
                gat_hidden=32,
                gat_out=32,
                heads1=4,
                heads2=4,
                event_in_dim=Xtr_e.shape[1],
                dropout=0.2
            )
            graph_obj = edge_index

        net, train_info = train_graph_binary(
            net,
            graph_obj=graph_obj,
            X_nodes=torch.tensor(X_nodes, dtype=torch.float32),
            train_loader=tr_loader,
            val_loader=va_loader,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            class_weight_pos=class_weight_pos,
            use_amp=use_amp
        )

        va_prob = predict_graph(net, graph_obj, torch.tensor(X_nodes, dtype=torch.float32),
                                va_loader, device=device, use_amp=use_amp)
        best_thr = choose_threshold_by_accuracy(yva_e, va_prob)

        te_prob = predict_graph(net, graph_obj, torch.tensor(X_nodes, dtype=torch.float32),
                                te_loader, device=device, use_amp=use_amp)

        metrics = compute_metrics(yte_e, te_prob, best_thr)
        metrics["train_info"] = train_info

        torch.save(net.state_dict(), os.path.join(out_dir, f"model_{model_name}.pt"))
        save_json({"meta": meta, "node_in_dim": node_in_dim, "email_count": E, "platform_count": P},
                  os.path.join(out_dir, "meta.json"))

        test_pred_df = test[ID_COLS + ["y"]].copy()
        test_pred_df["pred_prob"] = te_prob
        pred_path = os.path.join(out_dir, "predictions_test.csv")
        test_pred_df.to_csv(pred_path, index=False)

        return metrics, pred_path, best_thr

    raise ValueError(f"Unknown model: {model_name}")


# -------------------------
# Main
# -------------------------

def main():
    # =========================
    # USER CONFIGURATION
    # =========================
    DATA_DIR = "/home/afam/Passwords/Analysis"
    OUT_DIR = "/home/afam/Passwords/Analysis/Detection_Output"

    SETTINGS = ["cross_platform", "per_platform"]
    MODELS = ["fta_lstm", "fta_gru", "lstm", "gru", "gnn", "ifcnn_tpp", "cep3", "gcn", "gat"]

    SEQ_LEN = 10
    BATCH_SIZE = 256
    EPOCHS = 30
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 8
    SEED = 42

    DEVICE = None          # "cuda" | "cpu" | "mps" | None(auto)
    USE_AMP = True         # set False to disable
    FAIL_FAST = False      # True: stop on first failure; False: continue

    # =========================
    # DEVICE SETUP
    # =========================
    if DEVICE:
        device = torch.device(DEVICE)
    else:
        device = get_device()

    print(f"Using device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # =========================
    # LOAD DATA ONCE
    # =========================
    print(f"Loading cached data from: {DATA_DIR}")
    train, val, test = load_cached(DATA_DIR)

    # =========================
    # RUN GRID: settings x models
    # =========================
    all_runs = []
    total = len(SETTINGS) * len(MODELS)
    run_idx = 0

    for setting in SETTINGS:
        for model_name in MODELS:
            if model_name in {"gcn", "gat"} and setting != "cross_platform":
                print(f"Skipping invalid combo: {setting} + {model_name}")
                continue

            run_idx += 1
            run_tag = f"{setting}__{model_name}"
            run_out_dir = os.path.join(OUT_DIR, run_tag)
            os.makedirs(run_out_dir, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"[{run_idx}/{total}] Training model={model_name} setting={setting}")
            print(f"Run output dir: {run_out_dir}")
            print("=" * 80)

            started = time.time()
            run_record = {
                "setting": setting,
                "model": model_name,
                "out_dir": run_out_dir,
                "status": "started",
                "started_at": started,
            }

            try:
                results = train_and_evaluate(
                    train=train,
                    val=val,
                    test=test,
                    out_dir=run_out_dir,
                    setting=setting,
                    model_name=model_name,
                    seq_len=SEQ_LEN,
                    batch=BATCH_SIZE,
                    epochs=EPOCHS,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    patience=PATIENCE,
                    seed=SEED,
                    device=device,
                    use_amp=USE_AMP,
                )

                elapsed = time.time() - started
                run_record["status"] = "ok"
                run_record["elapsed_sec"] = elapsed

                if isinstance(results, dict) and "metrics" in results and isinstance(results["metrics"], dict):
                    run_record["metrics"] = results["metrics"]
                    acc = results["metrics"].get("accuracy", None)
                    f1 = results["metrics"].get("f1", None)
                    auroc = results["metrics"].get("auroc", None)
                    auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
                    print(f"Done: {run_tag} | elapsed={elapsed:.1f}s | ACC={acc:.4f} | F1={f1:.4f} | AUROC={auroc_str}")
                else:
                    run_record["metrics"] = None
                    print(f"Done: {run_tag} | elapsed={elapsed:.1f}s | (no metrics returned)")

                with open(os.path.join(run_out_dir, "run_summary.json"), "w") as f:
                    json.dump(run_record, f, indent=2)

            except Exception as e:
                elapsed = time.time() - started
                run_record["status"] = "error"
                run_record["elapsed_sec"] = elapsed
                run_record["error"] = str(e)
                run_record["traceback"] = traceback.format_exc()

                print(f"ERROR in {run_tag} after {elapsed:.1f}s: {e}")

                with open(os.path.join(run_out_dir, "run_summary.json"), "w") as f:
                    json.dump(run_record, f, indent=2)

                if FAIL_FAST:
                    raise

            # Per-run cleanup (helps long grids)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_runs.append(run_record)

    # =========================
    # SAVE MASTER SUMMARY
    # =========================
    master_path = os.path.join(OUT_DIR, "all_runs_summary.json")
    with open(master_path, "w") as f:
        json.dump(all_runs, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE")
    print(f"Master summary: {master_path}")
    print("=" * 80)

    scored = []
    for r in all_runs:
        if r.get("status") == "ok" and isinstance(r.get("metrics"), dict):
            acc = r["metrics"].get("accuracy", None)
            f1 = r["metrics"].get("f1", None)
            auroc = r["metrics"].get("auroc", None)
            if acc is not None:
                scored.append((float(acc), f1, auroc, r["setting"], r["model"], r["out_dir"]))

    if scored:
        scored.sort(reverse=True)
        print("\nTop runs by Accuracy:")
        for i, (acc, f1, auroc, setting, model, out_dir) in enumerate(scored[:10], 1):
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            print(f"{i:>2}. ACC={acc:.4f} F1={f1_str} AUROC={auroc_str} | {setting:>13} | {model:>8}")
    else:
        print("\nNo accuracy scores found in returned metrics.")


if __name__ == "__main__":
    main()