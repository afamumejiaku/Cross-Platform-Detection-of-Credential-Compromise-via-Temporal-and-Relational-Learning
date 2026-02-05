#!/usr/bin/env python3
"""
GPU-Optimized Monthly Time-Series Classification Pipeline for Platform Detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class BreachTimeSeriesProcessor:
    """
    Processes breach data into monthly time-series classification dataset
    Optimized for GPU training with efficient memory layout
    """
    
    def __init__(self, lookback_months=12):
        """
        Args:
            lookback_months: Number of past months to include in each sample (default: 12 months = 1 year)
        """
        self.lookback_months = lookback_months
        self.platform_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=500,  # Reduced for memory efficiency
            lowercase=True
        )
        self.num_honeywords = None  # Will be detected from data
        
    def load_and_prepare_data(self, csv_path):
        """Load CSV and prepare for time-series processing"""
        print("Loading data...")
        df = pd.read_csv(csv_path)
        
        # Detect honeyword columns
        honeyword_cols = [col for col in df.columns if col.startswith('Honeyword_')]
        self.num_honeywords = len(honeyword_cols)
        print(f"Detected {self.num_honeywords} honeyword columns")
        
        # Convert timestamps to datetime (handle mixed formats)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', errors='coerce')
        df['AttackTime'] = pd.to_datetime(df['AttackTime'], format='mixed', errors='coerce')
        
        # Drop any rows with invalid dates
        df = df.dropna(subset=['Timestamp', 'AttackTime'])
        
        # Drop rows with missing passwords
        df = df.dropna(subset=['Password'])
        df = df[df['Password'].astype(str).str.len() > 0]
        
        # Check honeywords - drop rows with too many missing honeywords
        min_honeywords = int(self.num_honeywords * 0.9)  # At least 90% present
        df = df.dropna(subset=honeyword_cols, thresh=min_honeywords)
        
        # Extract month-year for aggregation
        df['LeakMonth'] = df['Timestamp'].dt.to_period('M')
        df['AttackMonth'] = df['AttackTime'].dt.to_period('M')
        
        print(f"Loaded {len(df)} records")
        print(f"Date range: {df['Timestamp'].min()} to {df['AttackTime'].max()}")
        print(f"Platforms: {df['Breach_Source'].nunique()}")
        print(f"Unique emails: {df['Email'].nunique()}")
        
        return df
    
    def compute_password_embeddings(self, passwords):
        """Convert passwords to TF-IDF embeddings for similarity computation"""
        return self.vectorizer.fit_transform(passwords).toarray()
    
    def compute_honeyword_features(self, row, pw_embedding):
        """
        Compute similarity features between password and honeywords
        Returns: dictionary of honeyword-based features
        """
        honeywords = [row[f'Honeyword_{i}'] for i in range(1, self.num_honeywords + 1)]
        
        # Filter out NaN honeywords
        honeywords = [hw for hw in honeywords if pd.notna(hw) and str(hw).strip() != '']
        
        if len(honeywords) == 0:
            # No valid honeywords - return zeros
            return {
                'hw_mean_similarity': 0.0,
                'hw_max_similarity': 0.0,
                'hw_min_similarity': 0.0,
                'hw_std_similarity': 0.0,
                'hw_median_similarity': 0.0,
                'hw_top5_mean_similarity': 0.0,
                'hw_top10_mean_similarity': 0.0,
                'hw_bottom10_mean_similarity': 0.0,
                'hw_q25_similarity': 0.0,
                'hw_q75_similarity': 0.0,
                'hw_iqr_similarity': 0.0,
                'hw_range_similarity': 0.0,
            }
        
        # Compute embeddings for honeywords
        hw_embeddings = self.vectorizer.transform(honeywords).toarray()
        
        # Compute cosine similarities
        similarities = cosine_similarity(pw_embedding.reshape(1, -1), hw_embeddings)[0]
        
        # Handle edge cases for top-k means
        top5 = min(5, len(similarities))
        top10 = min(10, len(similarities))
        bottom10 = min(10, len(similarities))
        
        return {
            'hw_mean_similarity': np.mean(similarities),
            'hw_max_similarity': np.max(similarities),
            'hw_min_similarity': np.min(similarities),
            'hw_std_similarity': np.std(similarities),
            'hw_median_similarity': np.median(similarities),
            'hw_top5_mean_similarity': np.mean(np.sort(similarities)[-top5:]),
            'hw_top10_mean_similarity': np.mean(np.sort(similarities)[-top10:]),
            'hw_bottom10_mean_similarity': np.mean(np.sort(similarities)[:bottom10]),
            'hw_q25_similarity': np.percentile(similarities, 25),
            'hw_q75_similarity': np.percentile(similarities, 75),
            'hw_iqr_similarity': np.percentile(similarities, 75) - np.percentile(similarities, 25),
            'hw_range_similarity': np.max(similarities) - np.min(similarities),
        }
    
    def compute_historical_features(self, email, current_leak_month, current_password, 
                                   historical_data, pw_embedding):
        """
        Compute features based on historical leaked passwords for this email
        Uses only data from leaks BEFORE the current leak time
        """
        # Get historical leaks for this email (only those BEFORE current leak)
        # Convert Period to Timestamp for comparison
        current_leak_ts = current_leak_month.to_timestamp()
        
        hist = historical_data[
            (historical_data['Email'] == email) & 
            (historical_data['Timestamp'] < current_leak_ts)
        ].sort_values('Timestamp')
        
        if len(hist) == 0:
            # No historical data - return default features
            return {
                'hist_leak_count': 0,
                'hist_months_since_last_leak': 0,
                'hist_max_pw_similarity': 0.0,
                'hist_mean_pw_similarity': 0.0,
                'hist_max_hw_similarity': 0.0,
                'hist_mean_hw_similarity': 0.0,
                'hist_platform_diversity': 0,
                'hist_reuse_indicator': 0.0,
            }
        
        # Get most recent historical leak
        most_recent_leak = hist.iloc[-1]
        
        # Compute time since last leak (in months)
        months_since = (current_leak_ts - 
                       most_recent_leak['Timestamp']).days / 30.44
        
        # Compute password similarity with historical passwords
        hist_passwords = hist['Password'].tolist()
        hist_pw_embeddings = self.vectorizer.transform(hist_passwords).toarray()
        pw_similarities = cosine_similarity(pw_embedding.reshape(1, -1), hist_pw_embeddings)[0]
        
        # Compute similarity with historical honeywords
        all_hist_honeywords = []
        for _, hist_row in hist.iterrows():
            for i in range(1, self.num_honeywords + 1):
                hw = hist_row[f'Honeyword_{i}']
                if pd.notna(hw) and str(hw).strip() != '':
                    all_hist_honeywords.append(hw)
        
        if len(all_hist_honeywords) > 0:
            hist_hw_embeddings = self.vectorizer.transform(all_hist_honeywords).toarray()
            hw_similarities = cosine_similarity(pw_embedding.reshape(1, -1), hist_hw_embeddings)[0]
            max_hw_sim = np.max(hw_similarities)
            mean_hw_sim = np.mean(hw_similarities)
        else:
            max_hw_sim = 0.0
            mean_hw_sim = 0.0
        
        # Platform diversity
        platform_diversity = hist['Breach_Source'].nunique()
        
        # Password reuse indicator (high similarity suggests reuse)
        reuse_indicator = 1.0 if np.max(pw_similarities) > 0.8 else 0.0
        
        return {
            'hist_leak_count': len(hist),
            'hist_months_since_last_leak': months_since,
            'hist_max_pw_similarity': np.max(pw_similarities),
            'hist_mean_pw_similarity': np.mean(pw_similarities),
            'hist_max_hw_similarity': max_hw_sim,
            'hist_mean_hw_similarity': mean_hw_sim,
            'hist_platform_diversity': platform_diversity,
            'hist_reuse_indicator': reuse_indicator,
        }
    
    def create_time_buckets(self, df):
        """Create time bucket labels for time-to-detection"""
        time_deltas = (df['AttackTime'] - df['Timestamp']).dt.total_seconds() / 3600  # hours
        
        def bucket_time(hours):
            if hours < 1:
                return 0  # < 1 hour
            elif hours < 6:
                return 1  # 1-6 hours
            elif hours < 24:
                return 2  # 6-24 hours
            elif hours < 168:  # 7 days
                return 3  # 1-7 days
            else:
                return 4  # > 7 days
        
        df['time_bucket'] = time_deltas.apply(bucket_time)
        return df
    
    def build_monthly_sequences(self, df):
        """
        Build monthly time-series sequences for each email
        Each sequence spans lookback_months and includes aggregated features
        """
        print(f"\nBuilding monthly sequences (lookback={self.lookback_months} months)...")
        
        # Fit vectorizer on all passwords first
        print("Fitting password vectorizer...")
        self.vectorizer.fit(df['Password'].tolist())
        
        # Sort by email and attack time
        df = df.sort_values(['Email', 'AttackMonth'])
        
        sequences = []
        labels_platform = []
        labels_time_bucket = []
        metadata = []
        
        # Process each email
        for email in df['Email'].unique():
            email_data = df[df['Email'] == email].copy()
            
            # Get unique months where this email was attacked
            attack_months = email_data['AttackMonth'].unique()
            
            for target_month in attack_months:
                # Get all attacks in this target month
                target_attacks = email_data[email_data['AttackMonth'] == target_month]
                
                # For this month, we'll create features based on ALL attacks in lookback window
                # Define lookback period
                start_month = target_month - self.lookback_months
                
                # Get all data in lookback window
                lookback_data = email_data[
                    (email_data['AttackMonth'] > start_month) & 
                    (email_data['AttackMonth'] <= target_month)
                ]
                
                if len(lookback_data) == 0:
                    continue
                
                # Build monthly feature sequence
                monthly_features = []
                for month_offset in range(self.lookback_months):
                    current_month = target_month - (self.lookback_months - month_offset - 1)
                    
                    month_data = lookback_data[lookback_data['AttackMonth'] == current_month]
                    
                    if len(month_data) == 0:
                        # No attacks this month - use zeros
                        month_features = self._get_zero_features()
                    else:
                        # Aggregate features for this month
                        month_features = self._aggregate_month_features(
                            month_data, email, df
                        )
                    
                    monthly_features.append(month_features)
                
                # Stack into sequence: (lookback_months, num_features)
                sequence = np.vstack(monthly_features)
                sequences.append(sequence)
                
                # Label is the most common platform in target month
                target_platform = target_attacks['Breach_Source'].mode()[0]
                labels_platform.append(target_platform)
                
                # Time bucket is average of target month
                avg_time_bucket = int(target_attacks['time_bucket'].mean())
                labels_time_bucket.append(avg_time_bucket)
                
                # Store metadata
                metadata.append({
                    'email': email,
                    'target_month': str(target_month),
                    'num_attacks_in_month': len(target_attacks),
                    'platforms_in_month': target_attacks['Breach_Source'].unique().tolist()
                })
        
        print(f"Generated {len(sequences)} sequences")
        
        return sequences, labels_platform, labels_time_bucket, metadata
    
    def _get_zero_features(self):
        """Return zero feature vector for months with no attacks"""
        return np.zeros((1, 26))  # 26 features total: 2 basic + 12 honeyword + 8 historical + 4 time
    
    def _aggregate_month_features(self, month_data, email, full_df):
        """
        Aggregate features for a single month
        Returns: numpy array of shape (1, num_features)
        """
        features = []
        
        # Basic statistics
        features.append(len(month_data))  # num_attacks_in_month
        features.append(month_data['Breach_Source'].nunique())  # num_platforms
        
        # Compute average honeyword features
        hw_features_list = []
        for _, row in month_data.iterrows():
            pw_embedding = self.vectorizer.transform([row['Password']]).toarray()[0]
            hw_feats = self.compute_honeyword_features(row, pw_embedding)
            hw_features_list.append(hw_feats)
        
        # Average honeyword features across all attacks in month
        hw_features_df = pd.DataFrame(hw_features_list)
        for col in hw_features_df.columns:
            features.append(hw_features_df[col].mean())
        
        # Compute average historical features
        hist_features_list = []
        for _, row in month_data.iterrows():
            pw_embedding = self.vectorizer.transform([row['Password']]).toarray()[0]
            hist_feats = self.compute_historical_features(
                email, row['LeakMonth'], row['Password'], full_df, pw_embedding
            )
            hist_features_list.append(hist_feats)
        
        # Average historical features across all attacks in month
        hist_features_df = pd.DataFrame(hist_features_list)
        for col in hist_features_df.columns:
            features.append(hist_features_df[col].mean())
        
        # Time-based features
        time_delta_hours = (month_data['AttackTime'] - month_data['Timestamp']).dt.total_seconds() / 3600
        features.append(time_delta_hours.mean())  # avg_time_to_attack_hours
        features.append(time_delta_hours.std())   # std_time_to_attack_hours
        features.append(time_delta_hours.min())   # min_time_to_attack_hours
        features.append(time_delta_hours.max())   # max_time_to_attack_hours
        
        return np.array(features).reshape(1, -1)
    
    def prepare_gpu_ready_dataset(self, sequences, labels_platform, labels_time_bucket, metadata):
        """
        Convert sequences to GPU-optimized format
        Returns: X (float32 array), y_platform (int), y_time_bucket (int), metadata
        """
        print("\nPreparing GPU-ready dataset...")
        
        # Stack sequences into 3D array: (num_samples, lookback_months, num_features)
        X = np.stack(sequences).astype(np.float32)
        
        # Encode platform labels
        y_platform = self.platform_encoder.fit_transform(labels_platform)
        
        # Time bucket labels (already integers)
        y_time_bucket = np.array(labels_time_bucket, dtype=np.int32)
        
        print(f"X shape: {X.shape} (samples, time_steps, features)")
        print(f"y_platform shape: {y_platform.shape} - {len(self.platform_encoder.classes_)} classes")
        print(f"y_time_bucket shape: {y_time_bucket.shape} - {len(np.unique(y_time_bucket))} classes")
        print(f"Memory usage: {X.nbytes / 1024**2:.2f} MB")
        
        return X, y_platform, y_time_bucket, metadata
    
    def get_feature_names(self):
        """Return names of all features in order"""
        feature_names = [
            'num_attacks_in_month',
            'num_platforms',
            # Honeyword features (12)
            'hw_mean_similarity',
            'hw_max_similarity',
            'hw_min_similarity',
            'hw_std_similarity',
            'hw_median_similarity',
            'hw_top5_mean_similarity',
            'hw_top10_mean_similarity',
            'hw_bottom10_mean_similarity',
            'hw_q25_similarity',
            'hw_q75_similarity',
            'hw_iqr_similarity',
            'hw_range_similarity',
            # Historical features (8)
            'hist_leak_count',
            'hist_months_since_last_leak',
            'hist_max_pw_similarity',
            'hist_mean_pw_similarity',
            'hist_max_hw_similarity',
            'hist_mean_hw_similarity',
            'hist_platform_diversity',
            'hist_reuse_indicator',
            # Time features (4)
            'avg_time_to_attack_hours',
            'std_time_to_attack_hours',
            'min_time_to_attack_hours',
            'max_time_to_attack_hours',
        ]
        return feature_names
    
    def save_dataset(self, X, y_platform, y_time_bucket, metadata, output_path):
        """Save processed dataset to disk"""
        print(f"\nSaving dataset to {output_path}...")
        
        np.savez_compressed(
            output_path,
            X=X,
            y_platform=y_platform,
            y_time_bucket=y_time_bucket,
            metadata=metadata,
            platform_classes=self.platform_encoder.classes_,
            feature_names=self.get_feature_names(),
            lookback_months=self.lookback_months
        )
        
        print("Dataset saved successfully!")
    
    def get_summary_stats(self, df, X, y_platform, y_time_bucket):
        """Generate summary statistics"""
        stats = {
            'total_records': len(df),
            'total_emails': df['Email'].nunique(),
            'total_platforms': df['Breach_Source'].nunique(),
            'date_range': f"{df['Timestamp'].min()} to {df['AttackTime'].max()}",
            'num_sequences': X.shape[0],
            'sequence_length': X.shape[1],
            'num_features': X.shape[2],
            'platform_distribution': dict(zip(
                self.platform_encoder.classes_,
                np.bincount(y_platform)
            )),
            'time_bucket_distribution': dict(zip(
                ['<1h', '1-6h', '6-24h', '1-7d', '>7d'],
                np.bincount(y_time_bucket)
            ))
        }
        return stats


def main():
    """Main processing pipeline"""
    
    # Initialize processor with 12-month lookback
    processor = BreachTimeSeriesProcessor(lookback_months=12)
    
    # Load data
    df = processor.load_and_prepare_data('merged_breaches.csv')
    
    # Create time buckets
    df = processor.create_time_buckets(df)
    
    # Build monthly sequences
    sequences, labels_platform, labels_time_bucket, metadata = processor.build_monthly_sequences(df)
    
    # Prepare GPU-ready format
    X, y_platform, y_time_bucket, metadata = processor.prepare_gpu_ready_dataset(
        sequences, labels_platform, labels_time_bucket, metadata
    )
    
    # Get summary statistics
    stats = processor.get_summary_stats(df, X, y_platform, y_time_bucket)
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Save dataset
    processor.save_dataset(
        X, y_platform, y_time_bucket, metadata,
        'breach_dataset_monthly.npz'
    )
    
    # Save metadata separately for easy inspection
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('breach_metadata.csv', index=False)
    
    # Save summary stats
    stats_df = pd.DataFrame([stats])
    stats_df.to_json('dataset_stats.json', orient='records', indent=2)
    
    print("\n" + "="*70)
    print("DATASET FILES CREATED:")
    print("="*70)
    print("1. breach_dataset_monthly.npz - Main dataset (X, y_platform, y_time_bucket)")
    print("2. breach_metadata.csv - Sequence metadata")
    print("3. dataset_stats.json - Summary statistics")
    print("\nReady for GPU training with PyTorch/TensorFlow!")


if __name__ == "__main__":
    main()
