# Training Pipeline for Breach Detection and Platform Detection

This repository contains the complete data processing and training pipeline for breach detection and platform identification using password leak data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Generate config file
python pipeline.py --create-config

# 2. Edit config.json with your settings and API keys

# 3. Run complete pipeline
python pipeline.py --config config.json

# 4. Train breach detection model
python train_models.py --task breach --model lstm --setting cross_platform

# 5. Train platform detection model
python train_models.py --task platform --model gru --epochs 50

# 6. Evaluate models
python evaluate_models.py --task breach --results-dir model_results
python evaluate_models.py --task platform --results-dir model_results
```

## Overview

The pipeline consists of several stages:

1. **Data Collection** - Reading and consolidating email-password data from breach files
2. **Data Enrichment** - Merging with metadata and generating honeywords
3. **Data Preparation** - Splitting by breach, adding timestamps, and merging
4. **Breach Detection Training** - Feature engineering and temporal splits for breach detection
5. **Platform Detection Training** - Time-series classification for platform identification

## Repository Structure

```
.
├── data_collection.py              # Stage 1: Data collection and HIBP metadata
├── data_enrichment.py              # Stage 2: Merging and honeyword generation
├── data_preparation.py             # Stage 3: CSV preparation and merging
├── train_breach_detection.py      # Stage 4: Breach detection training pipeline
├── train_platform_detection.py    # Stage 5: Platform detection training pipeline
├── models_breach_detection.py     # Breach detection models (LSTM, GRU, FTA-GRU, etc.)
├── eval_breach_detection.py       # Breach detection evaluation and metrics
├── models_platform_detection.py   # Platform detection models (LSTM, GRU, etc.)
├── eval_platform_detection.py     # Platform detection evaluation and metrics
├── pipeline.py                     # Complete pipeline orchestrator
├── utils.py                        # Common utility functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Collection (`data_collection.py`)

Collects email-password data from breach files and enriches with HIBP metadata.

```python
from data_collection import *

# Step 1: Read and consolidate
input_files = ["breach1.txt", "breach2.txt", "breach3.txt"]
read_and_consolidate_email_passwords(input_files)

# Step 2: Filter multiple breaches
filter_multiple_breach_accounts()

# Step 3: Enrich with metadata
HIBP_API_KEY = "your-api-key-here"
enrich_with_metadata(api_key=HIBP_API_KEY)
```

### 2. Data Enrichment (`data_enrichment.py`)

Merges data and generates honeywords.

```python
from data_enrichment import *

# Merge breaches using round-robin
merge_breaches_roundrobin(email_passwords, metadata)

# Enrich with honeywords
enrich_with_honeywords(
    input_file="trainData.json",
    output_file="trainData_enriched_with_honeywords.json",
    honeywords_per_password=10
)
```

### 3. Data Preparation (`data_preparation.py`)

Prepares CSV files for training.

```python
from data_preparation import *

# Split by breach name
split_by_breach_name()

# Add breach times
add_breach_time_to_csvs()

# Merge all CSVs
merge_all_csv_files()
```

### 4. Breach Detection Training (`train_breach_detection.py`)

Trains breach detection models with feature engineering and temporal splits.

```python
from train_breach_detection import *

# Configure
build_cfg = BuildConfig(
    n_honeywords=50,
    tfidf_ngram=3,
    max_features=5000,
    sim_threshold_noisy=0.85,
    early_frac=0.40,
    cross_platform_leak_days=30
)

split_cfg = SplitConfig(
    train_frac=0.60,
    val_frac=0.20,
    test_frac=0.20
)

# Run pipeline
build_and_cache(
    csv_path="clean_merged_all_breaches_50hw.csv",
    out_dir="outputFolder",
    build_cfg=build_cfg,
    split_cfg=split_cfg
)
```

**Features computed:**
- Password similarity features (current vs historical)
- Temporal features (time since last leak/attack)
- Cross-platform features (platform overlap, reuse detection)
- Honeyword trigger detection

**Output:**
- `features.parquet` - Full feature set
- `train.parquet` - Training split (60%)
- `val.parquet` - Validation split (20%)
- `test.parquet` - Test split (20%)
- `build_meta.json` - Metadata

### 5. Platform Detection Training (`train_platform_detection.py`)

Trains time-series models for platform identification.

```python
from train_platform_detection import *

# Initialize processor
processor = BreachTimeSeriesProcessor(lookback_months=12)

# Load and process
df = processor.load_and_prepare_data('merged_all_breaches_100hw.csv')
df = processor.create_time_buckets(df)

# Build sequences
sequences, labels_platform, labels_time_bucket, metadata = \
    processor.build_monthly_sequences(df)

# Prepare GPU-ready format
X, y_platform, y_time_bucket, metadata = \
    processor.prepare_gpu_ready_dataset(
        sequences, labels_platform, labels_time_bucket, metadata
    )

# Save
processor.save_dataset(
    X, y_platform, y_time_bucket, metadata,
    'breach_dataset_monthly.npz'
)
```

**Features computed:**
- Monthly aggregated attack statistics
- Honeyword similarity features (12 metrics)
- Historical password features (8 metrics)
- Temporal features (4 metrics)

**Output:**
- `breach_dataset_monthly.npz` - Time-series dataset (X, y)
- `breach_metadata.csv` - Sequence metadata
- `dataset_stats.json` - Summary statistics

### 6. Breach Detection Modeling (`models_breach_detection.py`)

Train various deep learning models for breach detection.

**Available Models:**
- **LSTM Classifier** - Long Short-Term Memory networks
- **GRU Classifier** - Gated Recurrent Units
- **FTA-GRU** - Feature-Temporal Attention GRU
- **FTA-LSTM** - Feature-Temporal Attention LSTM
- **Temporal GNN** - Graph Neural Network treating timesteps as nodes
- **IFCNN-TPP** - CNN + GRU with attention pooling
- **CEP3** - Dilated causal convolutions with attention
- **GCN Bipartite** - Graph Convolutional Network on email-platform graph
- **GAT Bipartite** - Graph Attention Network on email-platform graph

```python
from models_breach_detection import *

# Load training data
train, val, test = load_cached("outputFolder")

# Build design matrices
Xtr, ytr, Xva, yva, Xte, yte, encoder, meta = build_design_matrices(
    train, val, test, 
    setting="cross_platform"  # or "per_platform"
)

# Train LSTM model
results = train_and_evaluate_lstm(
    train, val, test,
    setting="cross_platform",
    hidden=128,
    layers=2,
    seq_len=10,
    epochs=30,
    batch_size=256
)
```

**Model Settings:**
- `per_platform`: Train separate models for each platform
- `cross_platform`: Single model across all platforms with platform features

### 7. Breach Detection Evaluation (`eval_breach_detection.py`)

Evaluate breach detection models with time-to-detection analysis.

```python
from eval_breach_detection import *

# Analyze detection rate and delay
results = analyze_detection_and_delay(
    results_dir="results_breach",
    output_dir="analysis_breach"
)

# Generate comparative plots
plot_detection_curves(results, output_path="detection_curves.png")
```

**Metrics Computed:**
- Detection rate at various time windows (1h, 6h, 24h, 1d, 7d)
- Mean time to detection
- Detection delay distribution
- ROC-AUC, PR-AUC
- Accuracy, Precision, Recall, F1

### 8. Platform Detection Modeling (`models_platform_detection.py`)

Train models for platform identification from breach patterns.

**Available Models:**
- **LSTM Classifier** - Time-series LSTM
- **GRU Classifier** - Time-series GRU
- **Temporal CNN** - 1D Convolutional networks
- **Transformer** - Self-attention based models
- **Hybrid Models** - CNN + RNN combinations

```python
from models_platform_detection import *

# Load time-series data
data = np.load('breach_dataset_monthly.npz')
X = data['X']  # (n_samples, lookback_months, n_features)
y_platform = data['y_platform']
y_time = data['y_time_bucket']

# Train LSTM for platform detection
model, history = train_platform_lstm(
    X, y_platform,
    hidden_dim=128,
    num_layers=2,
    epochs=50,
    batch_size=64
)
```

### 9. Platform Detection Evaluation (`eval_platform_detection.py`)

Evaluate platform detection models with time-to-detection metrics.

```python
from eval_platform_detection import *

# Analyze platform detection performance
results = evaluate_platform_detection(
    model, test_data,
    output_dir="platform_analysis"
)

# Time-to-detection analysis
ttd_results = compute_time_to_detection(
    predictions, ground_truth,
    time_buckets=['<1h', '1-6h', '6-24h', '1-7d', '>7d']
)
```

**Metrics:**
- Platform classification accuracy
- Per-platform precision/recall/F1
- Confusion matrix
- Time-to-correct-detection
- Mean detection delay by platform

## Data Format

### Input CSV Format (for breach detection)
```csv
Breach_Source,Email,Password,Honeyword_1,...,Honeyword_50,Timestamp,AttackTime
```

### Input CSV Format (for platform detection)
```csv
Breach_Source,Email,Password,Honeyword_1,...,Honeyword_100,Timestamp,AttackTime
```

### Output Format (breach detection)
Parquet files with features:
- `sim_to_prev_passwords_max`
- `sim_to_prev_passwords_mean`
- `exact_password_reuse`
- `sim_to_prev_honeywords_max`
- `sim_to_prev_honeywords_mean`
- `honeyword_trigger`
- `current_hw_matches_prev_pw`
- `time_since_last_leak`
- `time_since_last_attack`
- `attack_sequence_index`
- `delta_similarity`
- `num_platforms_seen`
- `platform_overlap_score`
- `cross_platform_reuse`
- `y` (label: 0 or 1)

### Output Format (platform detection)
NPZ file with:
- `X`: shape `(n_samples, lookback_months, n_features)` - float32
- `y_platform`: shape `(n_samples,)` - int (encoded platform labels)
- `y_time_bucket`: shape `(n_samples,)` - int (time bucket: 0-4)
- `metadata`: list of dicts with email, target_month, etc.
- `platform_classes`: array of platform names
- `feature_names`: list of feature names

## GPU Acceleration

Both training pipelines support GPU acceleration:

- **Breach Detection**: Uses PyTorch for TF-IDF similarity computation
- **Platform Detection**: Outputs GPU-ready float32 arrays for PyTorch/TensorFlow

Auto-detection priority: CUDA > MPS (Apple Silicon) > CPU

## Configuration

### Breach Detection Config
```python
BuildConfig(
    n_honeywords=50,              # Number of honeywords per password
    tfidf_ngram=3,                # Character n-gram size for TF-IDF
    max_features=5000,            # Max vocabulary size
    sim_threshold_noisy=0.85,     # Similarity threshold for noisy labels
    early_frac=0.40,              # Fraction considered "early" in timeline
    cross_platform_leak_days=30   # Days for cross-platform leak window
)
```

### Platform Detection Config
```python
BreachTimeSeriesProcessor(
    lookback_months=12  # Number of historical months in sequence
)
```

## Dependencies

- pandas
- numpy
- scikit-learn
- torch (PyTorch)
- torch-geometric (for GNN models)
- scipy
- requests (for HIBP API)
- psutil (for memory monitoring)
- tqdm (for progress bars)

See `requirements.txt` for version details.

## Notes

- **HIBP API Key**: Required for metadata enrichment (get from https://haveibeenpwned.com/API/Key)
- **Memory Usage**: Large datasets may require 16GB+ RAM for feature computation
- **GPU Memory**: Platform detection with 100 honeywords requires ~4GB GPU memory
- **Time Complexity**: Feature engineering is O(n²) per email due to historical comparisons

## Model Architectures

### Breach Detection Models

1. **LSTM/GRU**: Standard recurrent architectures for sequential breach detection
2. **FTA-GRU/FTA-LSTM**: Feature and temporal attention mechanisms
3. **Temporal GNN**: Treats timesteps as graph nodes with attention
4. **IFCNN-TPP**: CNN feature extraction + GRU temporal encoding
5. **CEP3**: Dilated causal convolutions for temporal patterns
6. **GCN/GAT Bipartite**: Graph models on email-platform relationships

### Platform Detection Models

1. **LSTM/GRU**: Sequential models for platform classification
2. **Temporal CNN**: Convolutional networks over time windows
3. **Transformer**: Self-attention for long-range dependencies
4. **Hybrid**: Combined CNN+RNN architectures

All models support:
- GPU acceleration (CUDA/MPS/CPU)
- Automatic mixed precision training
- Per-platform or cross-platform settings
- Configurable architectures (hidden dimensions, layers, etc.)

## Command-Line Usage

### Training Models

```bash
# Breach Detection Models
python train_models.py --task breach --model lstm --setting cross_platform \
    --hidden 128 --layers 2 --epochs 30 --batch-size 256

python train_models.py --task breach --model gru --setting per_platform \
    --hidden 256 --layers 3 --epochs 40

python train_models.py --task breach --model temporal_gnn --setting cross_platform \
    --hidden 128 --epochs 50

# Platform Detection Models
python train_models.py --task platform --model lstm \
    --hidden 128 --layers 2 --epochs 50 --batch-size 64

python train_models.py --task platform --model transformer \
    --hidden 256 --layers 4 --epochs 100
```

### Evaluating Models

```bash
# Breach Detection Evaluation
python evaluate_models.py --task breach \
    --results-dir model_results \
    --output-dir breach_evaluation

# Platform Detection Evaluation
python evaluate_models.py --task platform \
    --results-dir model_results \
    --output-dir platform_evaluation
```

### Available Models

**Breach Detection:**
- `lstm` - LSTM Classifier
- `gru` - GRU Classifier
- `fta_gru` - Feature-Temporal Attention GRU
- `fta_lstm` - Feature-Temporal Attention LSTM
- `temporal_gnn` - Temporal Graph Neural Network
- `ifcnn_tpp` - CNN + GRU with attention
- `cep3` - Dilated causal convolutions
- `gcn` - Graph Convolutional Network
- `gat` - Graph Attention Network

**Platform Detection:**
- `lstm` - LSTM Classifier
- `gru` - GRU Classifier
- `cnn` - Temporal CNN
- `transformer` - Transformer model

## Citation

If you use this pipeline in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]

## Contact

[Add contact information here]
