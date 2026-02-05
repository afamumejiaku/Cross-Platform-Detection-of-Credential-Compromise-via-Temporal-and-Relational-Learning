#!/usr/bin/env python3
"""
Complete Model Training Script
==============================
Train and evaluate breach detection and platform detection models.

Usage:
    python train_models.py --task breach --model lstm --setting cross_platform
    python train_models.py --task platform --model gru --epochs 50
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any

# Import training modules
try:
    from models_breach_detection import (
        load_cached, build_design_matrices,
        train_and_evaluate_lstm, train_and_evaluate_gru,
        train_and_evaluate_fta_gru, train_and_evaluate_fta_lstm,
        train_and_evaluate_temporal_gnn, train_and_evaluate_ifcnn_tpp,
        train_and_evaluate_cep3, train_and_evaluate_gcn, train_and_evaluate_gat
    )
    BREACH_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Breach detection models not available: {e}")
    BREACH_MODELS_AVAILABLE = False

try:
    from models_platform_detection import (
        load_dataset, train_lstm_classifier, train_gru_classifier,
        train_temporal_cnn, train_transformer
    )
    PLATFORM_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Platform detection models not available: {e}")
    PLATFORM_MODELS_AVAILABLE = False

from utils import (
    seed_everything, ensure_dir, save_json,
    print_metrics_summary, get_device
)


def train_breach_detection(
    data_dir: str,
    model_name: str,
    setting: str,
    output_dir: str,
    config: Dict[str, Any]
):
    """Train breach detection model."""
    if not BREACH_MODELS_AVAILABLE:
        raise ImportError("Breach detection models not available")
    
    print(f"\n{'='*80}")
    print(f"Training Breach Detection: {model_name.upper()} ({setting})")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    train, val, test = load_cached(data_dir)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Training function map
    model_functions = {
        'lstm': train_and_evaluate_lstm,
        'gru': train_and_evaluate_gru,
        'fta_gru': train_and_evaluate_fta_gru,
        'fta_lstm': train_and_evaluate_fta_lstm,
        'temporal_gnn': train_and_evaluate_temporal_gnn,
        'ifcnn_tpp': train_and_evaluate_ifcnn_tpp,
        'cep3': train_and_evaluate_cep3,
        'gcn': train_and_evaluate_gcn,
        'gat': train_and_evaluate_gat
    }
    
    if model_name not in model_functions:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_functions.keys())}")
    
    # Train model
    train_fn = model_functions[model_name]
    results = train_fn(
        train, val, test,
        setting=setting,
        hidden=config.get('hidden', 128),
        layers=config.get('layers', 2),
        seq_len=config.get('seq_len', 10),
        epochs=config.get('epochs', 30),
        batch_size=config.get('batch_size', 256),
        lr=config.get('lr', 1e-3),
        dropout=config.get('dropout', 0.2),
        patience=config.get('patience', 8)
    )
    
    # Save results
    ensure_dir(output_dir)
    result_path = os.path.join(output_dir, f"{model_name}_{setting}_results.json")
    save_json(results, result_path)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {result_path}")
    
    # Print summary
    if 'test_metrics' in results:
        print_metrics_summary(results['test_metrics'], "Test Set Metrics")
    
    return results


def train_platform_detection(
    data_path: str,
    model_name: str,
    output_dir: str,
    config: Dict[str, Any]
):
    """Train platform detection model."""
    if not PLATFORM_MODELS_AVAILABLE:
        raise ImportError("Platform detection models not available")
    
    print(f"\n{'='*80}")
    print(f"Training Platform Detection: {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"Loading data from {data_path}...")
    X, y_platform, y_time, metadata, feature_names = load_dataset(data_path)
    print(f"Data shape: {X.shape}")
    print(f"Platforms: {len(np.unique(y_platform))}")
    
    # Training function map
    model_functions = {
        'lstm': train_lstm_classifier,
        'gru': train_gru_classifier,
        'cnn': train_temporal_cnn,
        'transformer': train_transformer
    }
    
    if model_name not in model_functions:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_functions.keys())}")
    
    # Train model
    train_fn = model_functions[model_name]
    results = train_fn(
        X, y_platform,
        hidden_dim=config.get('hidden', 128),
        num_layers=config.get('layers', 2),
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 64),
        lr=config.get('lr', 1e-3),
        dropout=config.get('dropout', 0.2),
        patience=config.get('patience', 10)
    )
    
    # Save results
    ensure_dir(output_dir)
    result_path = os.path.join(output_dir, f"{model_name}_platform_results.json")
    save_json(results, result_path)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {result_path}")
    
    # Print summary
    if 'test_metrics' in results:
        print_metrics_summary(results['test_metrics'], "Test Set Metrics")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train breach detection and platform detection models'
    )
    
    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['breach', 'platform'],
        help='Task to train: breach detection or platform detection'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model architecture (lstm, gru, fta_gru, temporal_gnn, etc.)'
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default='outputFolder',
        help='Directory with cached train/val/test data (for breach detection)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='breach_dataset_monthly.npz',
        help='Path to time-series dataset (for platform detection)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_results',
        help='Directory to save results'
    )
    
    # Breach detection specific
    parser.add_argument(
        '--setting',
        type=str,
        default='cross_platform',
        choices=['per_platform', 'cross_platform'],
        help='Training setting for breach detection'
    )
    
    # Model hyperparameters
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--seq-len', type=int, default=10, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu/mps)')
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Get device
    if args.device:
        import torch
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # Build config
    config = {
        'hidden': args.hidden,
        'layers': args.layers,
        'seq_len': args.seq_len,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'dropout': args.dropout,
        'patience': args.patience,
        'seed': args.seed,
        'device': str(device)
    }
    
    # Train model
    if args.task == 'breach':
        results = train_breach_detection(
            args.data_dir,
            args.model,
            args.setting,
            args.output_dir,
            config
        )
    else:  # platform
        results = train_platform_detection(
            args.data_path,
            args.model,
            args.output_dir,
            config
        )
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
