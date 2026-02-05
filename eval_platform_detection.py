#!/usr/bin/env python3
"""
Time-to-Detection Analysis for Cross-Platform Breach Platform Classification

Measures how quickly we can correctly identify the breach platform over time.

Models:
- LSTM classifier (PyTorch with CUDA)
- GRU classifier (PyTorch with CUDA)
- FTA-GRU simplified classifier (PyTorch with CUDA)
- FTA-LSTM simplified classifier (PyTorch with CUDA)
- Temporal GNN classifier (timesteps-as-nodes; PyTorch with CUDA)
- IFCNN-TPP classifier (Multi-kernel CNN; PyTorch with CUDA)   <-- UPDATED
- CEP3 classifier (dilated temporal convolution encoder; PyTorch with CUDA)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL ARCHITECTURES (must match training code)
# ============================================================================

class LSTMClassifier(nn.Module):
    """LSTM-based time-series classifier"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)   # hidden: (num_layers, B, H)
        out = self.dropout(hidden[-1])  # (B, H)
        return self.fc(out)             # (B, C)


class GRUClassifier(nn.Module):
    """GRU-based time-series classifier"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, hidden = self.gru(x)         # hidden: (num_layers, B, H)
        out = self.dropout(hidden[-1])  # (B, H)
        return self.fc(out)             # (B, C)


class FTAGru(nn.Module):
    """FTA-GRU: Feature-Temporal Attention GRU"""
    def __init__(self, input_size, hidden_size, num_classes,
                 num_gru_layers=2, attn_dim=64, num_attn_heads=4, dropout=0.3):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=False
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_pooling = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        feat_w = self.feature_attention(x)
        x = x * feat_w
        gru_out, _ = self.gru(x)
        attn_out, _ = self.temporal_attention(gru_out, gru_out, gru_out)
        scores = self.attn_pooling(attn_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (attn_out * weights).sum(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class FTALstm(nn.Module):
    """FTA-LSTM: Feature-Temporal Attention LSTM"""
    def __init__(self, input_size, hidden_size, num_classes,
                 num_lstm_layers=2, attn_dim=64, num_attn_heads=4, dropout=0.3):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_pooling = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        feat_w = self.feature_attention(x)
        x = x * feat_w
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        scores = self.attn_pooling(attn_out).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (attn_out * weights).sum(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for dense timesteps-as-nodes graph"""
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj=None):
        B, T, _ = x.size()
        h = torch.matmul(x, self.W)  # (B,T,Fout)

        h1 = h.repeat_interleave(T, dim=1)
        h2 = h.repeat(1, T, 1)
        hcat = torch.cat([h1, h2], dim=-1)

        e = self.leakyrelu(torch.matmul(hcat, self.a)).view(B, T, T)

        if adj is not None:
            e = e.masked_fill(adj == 0, -1e9)

        att = F.softmax(e, dim=-1)
        att = F.dropout(att, self.dropout, training=self.training)
        return torch.matmul(att, h)


class GNNClassifier(nn.Module):
    """Temporal GNN: GRU -> GAT across timesteps -> temporal attention -> pool -> classify"""
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, gat_heads=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_size, hidden_size // gat_heads, dropout=dropout)
            for _ in range(gat_heads)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)  # (B,T,H)
        gat_outs = [gat(gru_out) for gat in self.gat_layers]
        gat_combined = torch.cat(gat_outs, dim=-1)  # (B,T,H)
        h = self.layer_norm(gru_out + gat_combined)
        attn_out, _ = self.temporal_attention(h, h, h)
        pooled = attn_out.mean(dim=1)
        out = self.dropout(pooled)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


# ----------------------------------------------------------------------------
# UPDATED: IFCNN-TPP (code-1 multi-kernel CNN version)
# ----------------------------------------------------------------------------

class IFCNNTPPClassifier(nn.Module):
    """
    IFCNN-TPP (Multi-kernel CNN, code-1 style):
      - in_proj: Linear(F -> H)
      - convs: multi-kernel Conv1d over time on hidden channels
      - mean of conv branches
      - mean pool over time
      - head MLP

    NOTE: This matches checkpoints with keys like:
      in_proj.*, convs.*, norm.*, head.*
    """
    def __init__(self, input_size, hidden_size, num_classes, kernels=(3, 5, 7), dropout=0.3):
        super().__init__()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x: (B,T,F)
        h = self.norm(self.in_proj(x))  # (B,T,H)
        h = h.transpose(1, 2)           # (B,H,T)

        outs = [F.relu(conv(h)) for conv in self.convs]   # each (B,H,T)
        z = torch.stack(outs, dim=0).mean(dim=0)          # (B,H,T)

        z = z.transpose(1, 2)       # (B,T,H)
        pooled = z.mean(dim=1)      # (B,H)
        return self.head(pooled)    # (B,C)


# ----------------------------------------------------------------------------
# CEP3 (dilated temporal convolution encoder)
# ----------------------------------------------------------------------------

class CEP3Classifier(nn.Module):
    """
    CEP3-like dilated TCN with residual blocks.
    Input: (B, T, F)
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        channels: int = 128,
        kernel_size: int = 3,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size."

        self.in_proj = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            dilation = 2 ** i
            pad = (kernel_size // 2) * dilation
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        self.out_norm = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        h = self.in_proj(x)  # (B,C,T)

        for blk in self.blocks:
            r = h
            h = blk(h)
            h = h + r

        pooled = h.mean(dim=-1)  # (B,C)
        pooled = self.out_norm(pooled)
        return self.fc(pooled)


# ============================================================================
# DATA LOADING AND MODEL LOADING
# ============================================================================

def load_dataset(npz_path):
    print(f"\nLoading dataset from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    X = data["X"]
    y_platform = data["y_platform"]
    platform_classes = data["platform_classes"]

    print(f"X shape: {X.shape}")
    print(f"Number of platform classes: {len(platform_classes)}")
    return X, y_platform, platform_classes


def load_trained_model(model_type, input_size, num_classes, model_path):
    """
    Load a trained model (must match the training script’s architecture/hparams).

    Includes a safe auto-detect for IFCNN-TPP checkpoints to avoid key mismatches.
    """
    state = torch.load(model_path, map_location=device)

    if model_type == "LSTM":
        model = LSTMClassifier(input_size, 128, 2, num_classes, dropout=0.3)

    elif model_type == "GRU":
        model = GRUClassifier(input_size, 128, 2, num_classes, dropout=0.3)

    elif model_type == "FTA-GRU":
        model = FTAGru(
            input_size=input_size,
            hidden_size=128,
            num_classes=num_classes,
            num_gru_layers=2,
            attn_dim=64,
            num_attn_heads=4,
            dropout=0.3,
        )

    elif model_type == "FTA-LSTM":
        model = FTALstm(
            input_size=input_size,
            hidden_size=128,
            num_classes=num_classes,
            num_lstm_layers=2,
            attn_dim=64,
            num_attn_heads=4,
            dropout=0.3,
        )

    elif model_type == "GNN":
        model = GNNClassifier(
            input_size=input_size,
            hidden_size=128,
            num_classes=num_classes,
            num_layers=2,
            gat_heads=4,
            dropout=0.3,
        )

    elif model_type == "IFCNN-TPP":
        # Expecting code-1 multi-kernel checkpoint keys:
        # in_proj.*, convs.*, norm.*, head.*
        if any(k.startswith("in_proj.") for k in state.keys()):
            model = IFCNNTPPClassifier(
                input_size=input_size,
                hidden_size=128,
                num_classes=num_classes,
                kernels=(3, 5, 7),
                dropout=0.3,
            )
        else:
            raise RuntimeError(
                "IFCNN-TPP checkpoint does not look like the multi-kernel code-1 variant. "
                "Expected keys like 'in_proj.*', 'convs.*', 'norm.*', 'head.*'."
            )

    elif model_type == "CEP3":
        model = CEP3Classifier(
            input_size=input_size,
            num_classes=num_classes,
            channels=128,
            kernel_size=3,
            n_blocks=4,
            dropout=0.3,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# TIME-TO-DETECTION ANALYSIS
# ============================================================================

def compute_detection_time(model, X, y_true, confidence_threshold=0.7):
    """
    Detection happens at time step t when:
      1) predicted platform == true platform
      2) confidence(pred) >= threshold

    We evaluate partial prefixes by padding to full seq_len (so all models keep same input shape).
    """
    detection_times = []
    detection_confidences = []

    seq_len = X.shape[1]
    feat_dim = X.shape[2]

    for i in range(len(X)):
        sequence = X[i]  # (seq_len, feat_dim)
        true_label = int(y_true[i])

        detection_time = -1
        detection_conf = 0.0

        for t in range(1, seq_len + 1):
            partial = sequence[:t]  # (t, feat_dim)

            if t < seq_len:
                padding = np.zeros((seq_len - t, feat_dim), dtype=sequence.dtype)
                padded = np.vstack([partial, padding])
            else:
                padded = partial

            with torch.no_grad():
                X_input = torch.FloatTensor(padded).unsqueeze(0).to(device)  # (1, seq_len, feat_dim)
                out = model(X_input)
                probs = torch.softmax(out, dim=1)
                pred = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0, pred].item())

            if pred == true_label and conf >= confidence_threshold:
                detection_time = t
                detection_conf = conf
                break

        detection_times.append(int(detection_time))
        detection_confidences.append(float(detection_conf))

    return detection_times, detection_confidences


def analyze_detection_metrics(detection_times):
    valid = [t for t in detection_times if t > 0]
    never = sum(1 for t in detection_times if t <= 0)

    return {
        "total_sequences": int(len(detection_times)),
        "detected": int(len(valid)),
        "never_detected": int(never),
        "detection_rate": float(len(valid) / len(detection_times) * 100) if len(detection_times) else 0.0,
        "mean_detection_time": float(np.mean(valid)) if valid else 0.0,
        "median_detection_time": float(np.median(valid)) if valid else 0.0,
        "min_detection_time": float(np.min(valid)) if valid else 0.0,
        "max_detection_time": float(np.max(valid)) if valid else 0.0,
        "std_detection_time": float(np.std(valid)) if valid else 0.0,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_detection_time_distribution(detection_times, model_name, output_path):
    valid_times = [t for t in detection_times if t > 0]
    if not valid_times:
        print(f"No valid detections for {model_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_time = max(valid_times)
    bins = range(1, max_time + 2)

    axes[0].hist(valid_times, bins=bins, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Detection Time (timesteps)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{model_name} - Distribution of Detection Times")
    axes[0].grid(True, alpha=0.3)

    sorted_times = sorted(valid_times)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(detection_times) * 100

    axes[1].plot(sorted_times, cumulative, marker="o", markersize=4, linewidth=2)
    axes[1].set_xlabel("Detection Time (timesteps)")
    axes[1].set_ylabel("Cumulative Detection Rate (%)")
    axes[1].set_title(f"{model_name} - Cumulative Detection Curve")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=50, linestyle="--", alpha=0.7, label="50%")
    axes[1].axhline(y=90, linestyle="--", alpha=0.7, label="90%")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Detection distribution saved to {output_path}")


def plot_detection_by_platform(detection_times, y_true, platform_classes, model_name, output_path):
    platform_times = defaultdict(list)

    for t, label in zip(detection_times, y_true):
        if t > 0:
            platform_times[platform_classes[int(label)]].append(int(t))

    if not platform_times:
        print(f"No valid detections for {model_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    platforms = list(platform_times.keys())
    times_by_platform = [platform_times[p] for p in platforms]

    ax.boxplot(times_by_platform, labels=platforms, patch_artist=True)
    ax.set_xlabel("Platform", fontsize=12, fontweight="bold")
    ax.set_ylabel("Detection Time (timesteps)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} - Detection Time by Breach Platform", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Detection by platform saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CROSS-PLATFORM TIME-TO-DETECTION ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR = "/home/afam/Passwords/Analysis/Platform_Output"
    DATA_PATH = "/home/afam/Passwords/Analysis/platforms.npz"

    STAGE3_DIR = os.path.join(OUTPUT_DIR, "stage3")
    DELAY_DIR = os.path.join(STAGE3_DIR, "delay")
    os.makedirs(DELAY_DIR, exist_ok=True)

    X, y_platform, platform_classes = load_dataset(DATA_PATH)
    input_size = X.shape[2]
    num_classes = len(platform_classes)

    # NOTE: Your training script may have saved IFCNN as "ifcnn_tpp_model.pth"
    # We will try a few common filenames automatically for IFCNN-TPP.
    def resolve_model_path(primary_path: str, fallbacks: list[str]) -> str:
        if os.path.exists(primary_path):
            return primary_path
        for p in fallbacks:
            if os.path.exists(p):
                return p
        return primary_path  # return primary even if missing; caller will handle

    models_to_analyze = {
        "LSTM":      f"{OUTPUT_DIR}/LSTM_model.pth",
        "GRU":       f"{OUTPUT_DIR}/GRU_model.pth",
        "FTA-GRU":   f"{OUTPUT_DIR}/FTA-GRU_model.pth",
        "FTA-LSTM":  f"{OUTPUT_DIR}/FTA-LSTM_model.pth",
        "GNN":       f"{OUTPUT_DIR}/GNN_model.pth",
        "IFCNN-TPP": resolve_model_path(
            f"{OUTPUT_DIR}/IFCNN-TPP_model.pth",
            [
                f"{OUTPUT_DIR}/ifcnn_tpp_model.pth",
                f"{OUTPUT_DIR}/IFCNN_TPP_model.pth",
                f"{OUTPUT_DIR}/IFCNNTPP_model.pth",
            ],
        ),
        "CEP3":      f"{OUTPUT_DIR}/CEP3_model.pth",
    }

    all_stats = []

    for model_name, model_path in models_to_analyze.items():
        print(f"\n{'=' * 70}")
        print(f"Analyzing {model_name} Model")
        print(f"{'=' * 70}")

        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                print("Please train the models first using the training script")
                continue

            model = load_trained_model(model_name, input_size, num_classes, model_path)

            print(f"\nComputing detection times for {model_name}...")
            detection_times, confidences = compute_detection_time(
                model, X, y_platform, confidence_threshold=0.7
            )

            stats = analyze_detection_metrics(detection_times)

            print("\nDetection Statistics:")
            print(f"  Total sequences: {stats['total_sequences']}")
            print(f"  Detected: {stats['detected']} ({stats['detection_rate']:.2f}%)")
            print(f"  Never detected: {stats['never_detected']}")
            print("\nDetection Time (timesteps):")
            print(f"  Mean: {stats['mean_detection_time']:.2f}")
            print(f"  Median: {stats['median_detection_time']:.2f}")
            print(f"  Std: {stats['std_detection_time']:.2f}")
            print(f"  Range: [{stats['min_detection_time']:.0f}, {stats['max_detection_time']:.0f}]")

            # Stage3-compatible JSON
            delay_json = {
                "model": model_name,
                "setting": "cross_platform",
                "threshold": 0.7,
                "n_rows": int(len(X)),
                "n_entities_detected": int(stats["detected"]),
                "n_entities_with_positive": int(stats["total_sequences"]),
                "detection_rate_entities": float(stats["detection_rate"] / 100.0),
                "delay_mean": float(stats["mean_detection_time"]),
                "delay_median": float(stats["median_detection_time"]),
                "delay_min": float(stats["min_detection_time"]),
                "delay_max": float(stats["max_detection_time"]),
            }

            delay_json_path = os.path.join(
                DELAY_DIR,
                f"cross_platform__{model_name.lower().replace('-', '_')}.json"
            )
            with open(delay_json_path, "w") as f:
                json.dump(delay_json, f, indent=2)
            print(f"Saved delay metrics to: {delay_json_path}")

            plot_detection_time_distribution(
                detection_times, model_name,
                f"{OUTPUT_DIR}/{model_name}_detection_distribution.png"
            )

            plot_detection_by_platform(
                detection_times, y_platform, platform_classes, model_name,
                f"{OUTPUT_DIR}/{model_name}_detection_by_platform.png"
            )

            all_stats.append({
                "Model": model_name,
                "Detection Rate (%)": stats["detection_rate"],
                "Mean Time (timesteps)": stats["mean_detection_time"],
                "Median Time (timesteps)": stats["median_detection_time"],
            })

        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if all_stats:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON - TIME-TO-DETECTION")
        print("=" * 70)

        comparison_df = pd.DataFrame(all_stats).sort_values("Detection Rate (%)", ascending=False)
        print("\n" + comparison_df.to_string(index=False))
        comparison_df.to_csv(f"{OUTPUT_DIR}/detection_time_comparison.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(comparison_df)))
        axes[0].bar(comparison_df["Model"], comparison_df["Detection Rate (%)"], color=colors1)
        axes[0].set_ylabel("Detection Rate (%)", fontsize=11, fontweight="bold")
        axes[0].set_title("Detection Rate Comparison", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis="x", rotation=45)

        for i, (_, row) in enumerate(comparison_df.iterrows()):
            axes[0].text(
                i, row["Detection Rate (%)"],
                f'{row["Detection Rate (%)"]:.1f}%',
                ha="center", va="bottom", fontweight="bold"
            )

        colors2 = plt.cm.plasma(np.linspace(0.3, 0.9, len(comparison_df)))
        axes[1].bar(comparison_df["Model"], comparison_df["Mean Time (timesteps)"], color=colors2)
        axes[1].set_ylabel("Mean Detection Time (timesteps)", fontsize=11, fontweight="bold")
        axes[1].set_title("Mean Detection Time Comparison", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].tick_params(axis="x", rotation=45)

        for i, (_, row) in enumerate(comparison_df.iterrows()):
            axes[1].text(
                i, row["Mean Time (timesteps)"],
                f'{row["Mean Time (timesteps)"]:.1f}',
                ha="center", va="bottom", fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/model_detection_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nComparison plot saved to {OUTPUT_DIR}/model_detection_comparison.png")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - *_detection_distribution.png: Detection time distributions")
    print("  - *_detection_by_platform.png: Detection times by platform")
    print("  - detection_time_comparison.csv: Model comparison table")
    print("  - model_detection_comparison.png: Visual comparison")
    print("  - stage3/delay/*.json: Detection metrics (Stage3 compatible)")
    print("\nModels analyzed: LSTM, GRU, FTA-GRU, FTA-LSTM, GNN, IFCNN-TPP, CEP3 (cross-platform)")


if __name__ == "__main__":
    main()
