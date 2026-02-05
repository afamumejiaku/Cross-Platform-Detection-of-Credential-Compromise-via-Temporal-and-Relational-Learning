#!/usr/bin/env python3
"""
GPU-Optimized Time-Series Classification Models for Platform Detection

Handles:
- LSTM classifier (PyTorch with CUDA)
- GRU classifier (PyTorch with CUDA)
- FTA-GRU simplified classifier (PyTorch with CUDA)
- FTA-LSTM simplified classifier (PyTorch with CUDA)
- Temporal GNN classifier (timesteps-as-nodes; PyTorch with CUDA)
- IFCNN-TPP classifier (multi-kernel temporal CNN + mean pooling; PyTorch with CUDA)   <-- UPDATED
- CEP3 classifier (dilated temporal convolution encoder; PyTorch with CUDA)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DEVICE
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")


# =============================================================================
# BASIC MODELS (LSTM, GRU)
# =============================================================================

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
        _, (hidden, _) = self.lstm(x)         # (num_layers, B, H)
        out = self.dropout(hidden[-1])        # (B, H)
        return self.fc(out)                   # (B, C)


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
        _, hidden = self.gru(x)               # (num_layers, B, H)
        out = self.dropout(hidden[-1])        # (B, H)
        return self.fc(out)                   # (B, C)


# =============================================================================
# FTA ENCODERS (FTA-GRU, FTA-LSTM)
# =============================================================================

class FTAGru(nn.Module):
    """
    FTA-GRU:
    - Feature-level attention (gate per feature per timestep)
    - Multi-layer GRU
    - Multi-head temporal attention
    - Attention pooling
    """
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
    """
    FTA-LSTM:
    - Feature-level attention
    - Multi-layer LSTM
    - Multi-head temporal attention
    - Attention pooling
    """
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


# =============================================================================
# TEMPORAL GNN (timesteps-as-nodes)
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """Dense graph-attention across timesteps (treat T as nodes)."""
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj=None):
        B, T, _ = x.size()
        h = torch.matmul(x, self.W)  # (B, T, Fout)

        h_repeat_1 = h.repeat_interleave(T, dim=1)
        h_repeat_2 = h.repeat(1, T, 1)
        h_concat = torch.cat([h_repeat_1, h_repeat_2], dim=-1)

        e = self.leakyrelu(torch.matmul(h_concat, self.a)).view(B, T, T)

        if adj is not None:
            e = e.masked_fill(adj == 0, -1e9)

        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(attention, h)


class GNNClassifier(nn.Module):
    """
    Temporal GNN:
    - GRU encodes per-timestep hidden states
    - GAT across timesteps (dense)
    - Multihead temporal attention
    - Mean pooling -> classifier
    """
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
        gru_out, _ = self.gru(x)  # (B, T, H)
        gat_outputs = [gat(gru_out) for gat in self.gat_layers]
        gat_combined = torch.cat(gat_outputs, dim=-1)  # (B, T, H)

        gru_out = self.layer_norm(gru_out + gat_combined)
        attn_out, _ = self.temporal_attention(gru_out, gru_out, gru_out)
        pooled = attn_out.mean(dim=1)

        out = self.dropout(pooled)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


# =============================================================================
# IFCNN-TPP (UPDATED: multi-kernel temporal CNN + mean pooling)
# =============================================================================

class IFCNNTPPClassifier(nn.Module):
    """
    ifcnn_tpp (multi-kernel temporal CNN proxy):
      - Linear projection: F -> H
      - Parallel Conv1d branches over time with different kernel sizes (e.g., 3/5/7)
      - Average branch outputs
      - Mean pool over time
      - MLP head -> logits

    Input:  x (B, T, F)
    Conv1d: expects (B, H, T)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        kernels=(3, 5, 7),
        dropout: float = 0.3,
    ):
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
        # x: (B, T, F) -> h: (B, T, H)
        h = self.norm(self.in_proj(x))

        # Conv over time: (B, H, T)
        h = h.transpose(1, 2)

        outs = [F.relu(conv(h)) for conv in self.convs]           # list of (B, H, T)
        z = torch.stack(outs, dim=0).mean(dim=0)                  # (B, H, T)

        # back to (B, T, H) then mean pool over time -> (B, H)
        z = z.transpose(1, 2)
        pooled = z.mean(dim=1)

        return self.head(pooled)


# =============================================================================
# CEP3 (Dilated Temporal Convolution Encoder)
# =============================================================================

class CEP3Classifier(nn.Module):
    """
    CEP3-like dilated TCN:
      - Stacked dilated Conv1d blocks with residual connections
      - Global average pooling over time
      - Classification head

    Input: (B, T, F)
      Conv1d works on (B, F, T)
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
        assert kernel_size % 2 == 1, "Use odd kernel_size for simple same-length padding."

        self.in_proj = nn.Conv1d(input_size, channels, kernel_size=1)

        blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i
            pad = (kernel_size // 2) * dilation

            blocks.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.in_proj(x)  # (B, C, T)

        # residual dilated blocks (length preserved by padding choice)
        for blk in self.blocks:
            r = h
            h = blk(h)
            h = h + r

        # global average pool over time: (B, C)
        pooled = h.mean(dim=-1)          # (B, C)
        pooled = self.out_norm(pooled)   # LayerNorm over channels
        return self.fc(pooled)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class TimeSeriesAugmentation:
    @staticmethod
    def jitter(x, sigma=0.03):
        noise = np.random.normal(0, sigma, x.shape).astype(np.float32)
        return x + noise

    @staticmethod
    def scaling(x, sigma=0.1):
        factor = np.random.normal(1.0, sigma, (x.shape[0], 1, x.shape[2])).astype(np.float32)
        return x * factor


def augment_batch(X_batch, y_batch, aug_prob=0.3):
    if np.random.random() < aug_prob:
        choice = np.random.choice(['jitter', 'scaling'])
        X_aug = X_batch.detach().cpu().numpy()
        X_aug = TimeSeriesAugmentation.jitter(X_aug) if choice == 'jitter' else TimeSeriesAugmentation.scaling(X_aug)
        return torch.FloatTensor(X_aug).to(X_batch.device), y_batch
    return X_batch, y_batch


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_dataset(npz_path):
    print(f"\nLoading dataset from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    X = data['X']
    y_platform = data['y_platform']
    y_time_bucket = data['y_time_bucket']
    platform_classes = data['platform_classes']

    print(f"X shape: {X.shape}")
    print(f"Number of platform classes: {len(platform_classes)}")
    print(f"Example classes: {platform_classes[:5]}")
    return X, y_platform, y_time_bucket, platform_classes


def prepare_dataloaders(X, y, batch_size=64, train_ratio=0.7, val_ratio=0.15):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=(1 - train_ratio - val_ratio), random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=42,
        stratify=y_temp
    )

    print(f"\nDataset splits:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),     batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, use_augmentation=False, aug_prob=0.3):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        if use_augmentation:
            X_batch, y_batch = augment_batch(X_batch, y_batch, aug_prob)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return total_loss / max(1, len(train_loader)), 100.0 * correct / max(1, total)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += float(loss.item())

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(y_batch.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    return (
        total_loss / max(1, len(data_loader)),
        100.0 * correct / max(1, total),
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-3,
                patience=10, use_label_smoothing=True, use_augmentation=True):
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        print("Using Label Smoothing (0.1)")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_path = '/tmp/best_model.pth'

    print("\nTraining started...")
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Train Acc':>10} | {'Val Loss':>12} | {'Val Acc':>10}")
    print("-" * 70)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            use_augmentation=use_augmentation, aug_prob=0.3
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()

        print(f"{epoch+1:5d} | {train_loss:12.4f} | {train_acc:9.2f}% | {val_loss:12.4f} | {val_acc:9.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history


def compute_metrics(y_true, y_pred, y_probs, class_names):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    if len(class_names) > 2:
        roc_auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': dict(zip(class_names, f1_per_class)),
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }


# =============================================================================
# VISUALIZATION (unchanged)
# =============================================================================

def plot_training_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {output_path}")


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curves(y_true, y_probs, class_names, output_path):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {output_path}")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    OUTPUT_DIR = '/home/Passwords/Analysis/Platform_Output'
    DATA_PATH = '/home/Passwords/Analysis/platforms.npz'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y_platform, y_time_bucket, platform_classes = load_dataset(DATA_PATH)
    train_loader, val_loader, test_loader = prepare_dataloaders(X, y_platform, batch_size=64)

    input_size = X.shape[2]
    num_classes = len(platform_classes)

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")

    models_to_train = {
        'LSTM': LSTMClassifier(input_size, 128, 2, num_classes, dropout=0.3).to(device),
        'GRU': GRUClassifier(input_size, 128, 2, num_classes, dropout=0.3).to(device),
        'FTA-GRU': FTAGru(input_size, 128, num_classes, num_gru_layers=2, attn_dim=64, num_attn_heads=4, dropout=0.3).to(device),
        'FTA-LSTM': FTALstm(input_size, 128, num_classes, num_lstm_layers=2, attn_dim=64, num_attn_heads=4, dropout=0.3).to(device),
        'GNN': GNNClassifier(input_size, 128, num_classes, num_layers=2, gat_heads=4, dropout=0.3).to(device),

        # UPDATED: multi-kernel CNN proxy (no GRU/LSTM encoder)
        'IFCNN-TPP': IFCNNTPPClassifier(
            input_size=input_size,
            hidden_size=128,
            num_classes=num_classes,
            kernels=(3, 5, 7),
            dropout=0.3,
        ).to(device),

        'CEP3': CEP3Classifier(
            input_size=input_size,
            num_classes=num_classes,
            channels=128,
            kernel_size=3,
            n_blocks=4,
            dropout=0.3,
        ).to(device),
    }

    results = {}

    for model_name, model in models_to_train.items():
        print(f"\n{'=' * 70}")
        print(f"Training {model_name} Model")
        print(f"{'=' * 70}")

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params:,}")

        # Treat attention/CNN/TCN as "advanced"
        use_advanced = model_name in ['FTA-GRU', 'FTA-LSTM', 'IFCNN-TPP', 'CEP3']

        trained_model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=200,
            learning_rate=0.001,
            patience=10,
            use_label_smoothing=use_advanced,
            use_augmentation=use_advanced
        )

        print(f"\nEvaluating {model_name} on test set...")
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, y_pred, y_true, y_probs = evaluate(trained_model, test_loader, criterion)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        metrics = compute_metrics(y_true, y_pred, y_probs, platform_classes)

        print(f"\nF1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        results[model_name] = {
            'model': trained_model,
            'history': history,
            'test_metrics': metrics,
            'test_accuracy': test_acc
        }

        plot_training_history(history, f'{OUTPUT_DIR}/{model_name}_training_history.png')
        plot_confusion_matrix(metrics['confusion_matrix'], platform_classes, f'{OUTPUT_DIR}/{model_name}_confusion_matrix.png')
        plot_roc_curves(y_true, y_probs, platform_classes, f'{OUTPUT_DIR}/{model_name}_roc_curves.png')

        torch.save(trained_model.state_dict(), f'{OUTPUT_DIR}/{model_name}_model.pth')
        print(f"\n{model_name} training complete!")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy (%)': [r['test_accuracy'] for r in results.values()],
        'F1 Macro': [r['test_metrics']['f1_macro'] for r in results.values()],
        'F1 Weighted': [r['test_metrics']['f1_weighted'] for r in results.values()],
        'ROC-AUC': [r['test_metrics']['roc_auc'] for r in results.values()]
    }).sort_values('Test Accuracy (%)', ascending=False)

    print("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)

    # Comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = [
        ('Test Accuracy (%)', axes[0, 0]),
        ('F1 Macro', axes[0, 1]),
        ('F1 Weighted', axes[1, 0]),
        ('ROC-AUC', axes[1, 1])
    ]

    for metric_name, ax in metrics_to_plot:
        data = comparison_df.sort_values(metric_name, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
        ax.barh(data['Model'], data[metric_name], color=colors)
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(data[metric_name]):
            ax.text(v, i, f' {v:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/model_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {OUTPUT_DIR}/model_comparison_chart.png")

    best_model = comparison_df.iloc[0]['Model']
    best_acc = comparison_df.iloc[0]['Test Accuracy (%)']
    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_model} with {best_acc:.2f}% Test Accuracy")
    print(f"{'=' * 70}")

    print("\nGenerated files:")
    print("  - *_model.pth: Trained model weights")
    print("  - *_training_history.png: Training curves")
    print("  - *_confusion_matrix.png: Confusion matrices")
    print("  - *_roc_curves.png: ROC curves")
    print("  - model_comparison.csv: Performance comparison table")
    print("  - model_comparison_chart.png: Visual comparison")


if __name__ == "__main__":
    main()
