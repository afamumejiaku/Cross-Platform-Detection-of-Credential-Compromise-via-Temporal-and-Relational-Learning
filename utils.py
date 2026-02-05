#!/usr/bin/env python3
"""
Utility Functions
================
Common utility functions used across the pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import os


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_parquet(filepath: str) -> pd.DataFrame:
    """Load parquet file."""
    return pd.read_parquet(filepath)


def save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to parquet file."""
    df.to_parquet(filepath, index=False)


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load CSV file with common defaults."""
    defaults = {
        'encoding': 'utf-8',
        'low_memory': False
    }
    defaults.update(kwargs)
    return pd.read_csv(filepath, **defaults)


def save_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save DataFrame to CSV file."""
    defaults = {
        'index': False,
        'encoding': 'utf-8'
    }
    defaults.update(kwargs)
    df.to_csv(filepath, **defaults)


def create_directory(dirpath: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(dirpath, exist_ok=True)


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def count_lines(filepath: str) -> int:
    """Count lines in a file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


def split_list(lst: List, n_chunks: int) -> List[List]:
    """Split list into n roughly equal chunks."""
    chunk_size = len(lst) // n_chunks
    remainder = len(lst) % n_chunks
    
    chunks = []
    start = 0
    
    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    
    return chunks


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print useful information about a DataFrame."""
    print(f"\n{'='*60}")
    print(f"{name} Info")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nFirst few rows:")
    print(df.head())


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Validate that DataFrame has required columns."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is 0."""
    return numerator / denominator if denominator != 0 else default


def normalize_email(email: str) -> str:
    """Normalize email address (lowercase, strip whitespace)."""
    return email.lower().strip() if email else ""


def batch_process(items: List, batch_size: int, process_fn):
    """Process items in batches."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    return results


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "") -> None:
    """Print progress bar."""
    bar_length = 50
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = f"{100 * (current / total):.1f}"
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }
    
    values_array = np.array(values)
    return {
        'count': len(values),
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q75': float(np.percentile(values_array, 75))
    }


# ==========================================
# Modeling and Evaluation Utilities
# ==========================================

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS GPU")
            return device
        else:
            device = torch.device("cpu")
            print("Using CPU")
            return device
    except ImportError:
        print("PyTorch not available, using CPU")
        return "cpu"


def count_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced datasets."""
    from collections import Counter
    counts = Counter(y)
    total = len(y)
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return weights


def load_model_checkpoint(model, checkpoint_path: str):
    """Load model weights from checkpoint."""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def save_model_checkpoint(model, optimizer, epoch: int, metrics: Dict, path: str):
    """Save model checkpoint with training state."""
    try:
        import torch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        raise


def plot_training_history(history: Dict[str, List[float]], output_path: str):
    """Plot training and validation metrics over epochs."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy')
            axes[1].plot(history['val_acc'], label='Val Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training history plot to {output_path}")
    except Exception as e:
        print(f"Error plotting training history: {e}")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: str):
    """Plot confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }
    
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'] = float(p)
    metrics['recall'] = float(r)
    metrics['f1'] = float(f1)
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except:
            metrics['roc_auc'] = None
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
        except:
            metrics['pr_auc'] = None
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics_summary(metrics: Dict[str, Any], title: str = "Metrics Summary"):
    """Print formatted metrics summary."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Utility functions module")
    print("Import this module to use utility functions:")
    print("  from utils import *")
