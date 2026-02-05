#!/usr/bin/env python3
"""
Complete Training Pipeline
=========================
End-to-end pipeline for breach detection training.

This script orchestrates all stages of the pipeline:
1. Data collection
2. Data enrichment
3. Data preparation
4. Feature engineering and training data generation

Usage:
    python pipeline.py --config config.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any

# Import pipeline modules
from data_collection import (
    read_and_consolidate_email_passwords,
    filter_multiple_breach_accounts,
    enrich_with_metadata
)
from data_enrichment import (
    merge_breaches_roundrobin,
    enrich_with_honeywords
)
from data_preparation import (
    split_by_breach_name,
    add_breach_time_to_csvs,
    merge_all_csv_files
)
from train_breach_detection import (
    BuildConfig,
    SplitConfig,
    build_and_cache
)


class PipelineConfig:
    """Configuration for the complete pipeline."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Input files
        self.breach_files = config_dict.get('breach_files', [])
        self.hibp_api_key = config_dict.get('hibp_api_key', None)
        
        # Output directories
        self.work_dir = config_dict.get('work_dir', 'pipeline_output')
        self.output_dir = config_dict.get('output_dir', 'training_data')
        
        # Processing parameters
        self.honeywords_per_password = config_dict.get('honeywords_per_password', 10)
        self.rate_limit_delay = config_dict.get('rate_limit_delay', 1.5)
        
        # Training parameters
        self.n_honeywords = config_dict.get('n_honeywords', 50)
        self.tfidf_ngram = config_dict.get('tfidf_ngram', 3)
        self.max_features = config_dict.get('max_features', 5000)
        self.sim_threshold_noisy = config_dict.get('sim_threshold_noisy', 0.85)
        self.early_frac = config_dict.get('early_frac', 0.40)
        self.cross_platform_leak_days = config_dict.get('cross_platform_leak_days', 30)
        
        # Split parameters
        self.train_frac = config_dict.get('train_frac', 0.60)
        self.val_frac = config_dict.get('val_frac', 0.20)
        self.test_frac = config_dict.get('test_frac', 0.20)
        
        # Pipeline control
        self.skip_collection = config_dict.get('skip_collection', False)
        self.skip_enrichment = config_dict.get('skip_enrichment', False)
        self.skip_preparation = config_dict.get('skip_preparation', False)
        
    def validate(self):
        """Validate configuration."""
        if not self.skip_collection and not self.breach_files:
            raise ValueError("breach_files must be provided if not skipping collection")
        
        if not self.skip_collection and not self.hibp_api_key:
            print("Warning: HIBP API key not provided. Metadata enrichment will be skipped.")
        
        if abs(self.train_frac + self.val_frac + self.test_frac - 1.0) > 0.01:
            raise ValueError("train_frac + val_frac + test_frac must equal 1.0")


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration."""
    return {
        "breach_files": [
            "breach1.txt",
            "breach2.txt",
            "breach3.txt"
        ],
        "hibp_api_key": "your-api-key-here",
        "work_dir": "pipeline_output",
        "output_dir": "training_data",
        "honeywords_per_password": 10,
        "rate_limit_delay": 1.5,
        "n_honeywords": 50,
        "tfidf_ngram": 3,
        "max_features": 5000,
        "sim_threshold_noisy": 0.85,
        "early_frac": 0.40,
        "cross_platform_leak_days": 30,
        "train_frac": 0.60,
        "val_frac": 0.20,
        "test_frac": 0.20,
        "skip_collection": False,
        "skip_enrichment": False,
        "skip_preparation": False
    }


def run_pipeline(config: PipelineConfig):
    """Run the complete training pipeline."""
    
    print("="*80)
    print("TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Create work directories
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Define file paths
    paths = {
        'email_passwords': os.path.join(config.work_dir, 'email_passwords.json'),
        'filtered': os.path.join(config.work_dir, 'email_passwords_filtered.json'),
        'train_data': os.path.join(config.work_dir, 'trainData.json'),
        'enriched': os.path.join(config.work_dir, 'trainData_enriched_with_honeywords.json'),
        'breach_csv_dir': os.path.join(config.work_dir, 'breach_csv_files'),
        'breach_csv_time_dir': os.path.join(config.work_dir, 'breach_csv_files_with_time'),
        'merged_csv': os.path.join(config.work_dir, 'merged_training_data.csv')
    }
    
    # Stage 1: Data Collection
    if not config.skip_collection:
        print("\n" + "="*80)
        print("STAGE 1: DATA COLLECTION")
        print("="*80)
        
        print("\n[1.1] Reading and consolidating breach files...")
        read_and_consolidate_email_passwords(
            config.breach_files,
            paths['email_passwords']
        )
        
        print("\n[1.2] Filtering accounts with multiple breaches...")
        filter_multiple_breach_accounts(
            paths['email_passwords'],
            paths['filtered']
        )
        
        if config.hibp_api_key:
            print("\n[1.3] Enriching with HIBP metadata...")
            enrich_with_metadata(
                paths['filtered'],
                paths['train_data'],
                config.hibp_api_key,
                config.rate_limit_delay
            )
        else:
            print("\n[1.3] Skipping HIBP metadata (no API key)")
            # Copy filtered to train_data
            import shutil
            shutil.copy(paths['filtered'], paths['train_data'])
    else:
        print("\n[SKIPPED] Stage 1: Data Collection")
    
    # Stage 2: Data Enrichment
    if not config.skip_enrichment:
        print("\n" + "="*80)
        print("STAGE 2: DATA ENRICHMENT")
        print("="*80)
        
        print("\n[2.1] Generating honeywords...")
        enrich_with_honeywords(
            paths['train_data'],
            paths['enriched'],
            config.honeywords_per_password
        )
    else:
        print("\n[SKIPPED] Stage 2: Data Enrichment")
    
    # Stage 3: Data Preparation
    if not config.skip_preparation:
        print("\n" + "="*80)
        print("STAGE 3: DATA PREPARATION")
        print("="*80)
        
        print("\n[3.1] Splitting by breach name...")
        split_by_breach_name(
            paths['enriched'],
            paths['breach_csv_dir']
        )
        
        print("\n[3.2] Adding breach times...")
        add_breach_time_to_csvs(
            paths['breach_csv_dir'],
            paths['breach_csv_time_dir']
        )
        
        print("\n[3.3] Merging all CSV files...")
        merge_all_csv_files(
            paths['breach_csv_time_dir'],
            paths['merged_csv']
        )
    else:
        print("\n[SKIPPED] Stage 3: Data Preparation")
    
    # Stage 4: Feature Engineering and Training Data
    print("\n" + "="*80)
    print("STAGE 4: FEATURE ENGINEERING")
    print("="*80)
    
    build_cfg = BuildConfig(
        n_honeywords=config.n_honeywords,
        tfidf_ngram=config.tfidf_ngram,
        max_features=config.max_features,
        sim_threshold_noisy=config.sim_threshold_noisy,
        early_frac=config.early_frac,
        cross_platform_leak_days=config.cross_platform_leak_days
    )
    
    split_cfg = SplitConfig(
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        test_frac=config.test_frac
    )
    
    print("\n[4.1] Building features and creating splits...")
    build_and_cache(
        csv_path=paths['merged_csv'],
        out_dir=config.output_dir,
        build_cfg=build_cfg,
        split_cfg=split_cfg
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Completed at: {datetime.now()}")
    print(f"\nTraining data saved to: {config.output_dir}")
    print("\nFiles created:")
    print(f"  - {os.path.join(config.output_dir, 'train.parquet')}")
    print(f"  - {os.path.join(config.output_dir, 'val.parquet')}")
    print(f"  - {os.path.join(config.output_dir, 'test.parquet')}")
    print(f"  - {os.path.join(config.output_dir, 'build_meta.json')}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete training pipeline for breach detection'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config.json',
        help='Output path for default configuration (used with --create-config)'
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        # Create default config
        default_config = create_default_config()
        with open(args.output, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default configuration at: {args.output}")
        print("Edit this file and run: python pipeline.py --config config.json")
        return
    
    if args.config:
        # Load configuration
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PipelineConfig(config_dict)
    else:
        print("No configuration provided. Creating default configuration...")
        config = PipelineConfig(create_default_config())
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    # Run pipeline
    try:
        run_pipeline(config)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
