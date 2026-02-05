#!/usr/bin/env python3
"""
Model Evaluation Script
======================
Evaluate trained models and generate analysis reports.

Usage:
    python evaluate_models.py --task breach --results-dir model_results
    python evaluate_models.py --task platform --results-dir model_results
"""

import argparse
import os
import json
from typing import Dict, Any

try:
    from eval_breach_detection import (
        analyze_detection_and_delay,
        plot_detection_curves,
        compare_models
    )
    BREACH_EVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Breach evaluation not available: {e}")
    BREACH_EVAL_AVAILABLE = False

try:
    from eval_platform_detection import (
        evaluate_platform_detection,
        compute_time_to_detection,
        plot_platform_confusion_matrix
    )
    PLATFORM_EVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Platform evaluation not available: {e}")
    PLATFORM_EVAL_AVAILABLE = False

from utils import ensure_dir, print_metrics_summary


def evaluate_breach_detection(results_dir: str, output_dir: str):
    """Evaluate breach detection models."""
    if not BREACH_EVAL_AVAILABLE:
        raise ImportError("Breach detection evaluation not available")
    
    print(f"\n{'='*80}")
    print("Evaluating Breach Detection Models")
    print(f"{'='*80}\n")
    
    ensure_dir(output_dir)
    
    # Analyze detection and delay
    print("Analyzing detection rate and delay...")
    analysis_results = analyze_detection_and_delay(
        results_dir=results_dir,
        output_dir=output_dir
    )
    
    # Plot detection curves
    print("Generating detection curves...")
    plot_detection_curves(
        analysis_results,
        output_path=os.path.join(output_dir, "detection_curves.png")
    )
    
    # Compare models
    print("Comparing models...")
    comparison = compare_models(
        results_dir=results_dir,
        output_path=os.path.join(output_dir, "model_comparison.json")
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    
    if comparison:
        print("\nModel Comparison Summary:")
        for model_name, metrics in comparison.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric:20s}: {value:.4f}")
    
    return analysis_results


def evaluate_platform_detection(results_dir: str, output_dir: str):
    """Evaluate platform detection models."""
    if not PLATFORM_EVAL_AVAILABLE:
        raise ImportError("Platform detection evaluation not available")
    
    print(f"\n{'='*80}")
    print("Evaluating Platform Detection Models")
    print(f"{'='*80}\n")
    
    ensure_dir(output_dir)
    
    # Load all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    
    for result_file in result_files:
        print(f"\nProcessing {result_file}...")
        
        result_path = os.path.join(results_dir, result_file)
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        model_name = result_file.replace('_platform_results.json', '')
        model_output_dir = os.path.join(output_dir, model_name)
        ensure_dir(model_output_dir)
        
        # Evaluate platform detection
        eval_results = evaluate_platform_detection(
            results,
            output_dir=model_output_dir
        )
        
        # Compute time-to-detection
        if 'predictions' in results and 'ground_truth' in results:
            ttd_results = compute_time_to_detection(
                results['predictions'],
                results['ground_truth'],
                time_buckets=['<1h', '1-6h', '6-24h', '1-7d', '>7d']
            )
            
            # Save TTD results
            ttd_path = os.path.join(model_output_dir, 'time_to_detection.json')
            with open(ttd_path, 'w') as f:
                json.dump(ttd_results, f, indent=2)
        
        # Plot confusion matrix
        if 'confusion_matrix' in results:
            plot_platform_confusion_matrix(
                results['confusion_matrix'],
                output_path=os.path.join(model_output_dir, 'confusion_matrix.png')
            )
        
        # Print metrics
        if 'test_metrics' in results:
            print_metrics_summary(results['test_metrics'], f"{model_name} Test Metrics")
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['breach', 'platform'],
        help='Task to evaluate: breach detection or platform detection'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='model_results',
        help='Directory containing model results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Evaluate models
    if args.task == 'breach':
        results = evaluate_breach_detection(
            args.results_dir,
            args.output_dir
        )
    else:  # platform
        results = evaluate_platform_detection(
            args.results_dir,
            args.output_dir
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
