"""
Step 3: Save XGBoost Predictions for Knowledge Distillation
Saves predictions (soft targets) for each sample in the dataset
Output is optimized for fast loading during neural network training
"""

import duckdb
import numpy as np
import pickle
from pathlib import Path
import time
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictionSaver:
    def __init__(self, model_path: str, ducklake_path: str, output_dir: str = "./predictions", 
                 batch_size: int = 50000):
        self.model_path = Path(model_path)
        self.ducklake_path = Path(ducklake_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        
        logger.info("="*70)
        logger.info("XGBoost Predictions Saver")
        logger.info("="*70)
        logger.info(f"Model: {model_path}")
        logger.info(f"Database: {ducklake_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Batch size: {batch_size:,}\n")
        
        logger.info("Loading model...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_params = model_data.get('params', {})
        self.model_accuracy = model_data.get('accuracy', 0.0)
        self.stack_frames = model_data.get('stack_frames', 4)
        
        logger.info(f"✓ Model loaded")
        logger.info(f"  Accuracy: {self.model_accuracy:.6f}")
        logger.info(f"  Stack frames: {self.stack_frames}")
    
    def compute_and_save_predictions(self):
        """Compute predictions for all samples and save efficiently"""
        
        logger.info("\nConnecting to DuckDB lake...")
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        conn.execute("SET memory_limit = '32GB'")
        
        try:
            # Determine table to use
            obs_table = "observations_stacked" if self.stack_frames > 1 else "observations"
            
            # Get total count
            total_count = conn.execute(f"SELECT COUNT(*) FROM {obs_table}").fetchone()[0]
            logger.info(f"Total samples: {total_count:,}")
            
            # Prepare output files
            predictions_path = self.output_dir / "predictions.parquet"
            metadata_path = self.output_dir / "metadata.json"
            
            # Process in batches
            logger.info(f"\nProcessing predictions in batches of {self.batch_size:,}...")
            
            all_predictions = []
            all_probabilities = []
            all_true_labels = []
            all_indices = []
            
            offset = 0
            pbar = tqdm(total=total_count, desc="Computing predictions")
            
            while offset < total_count:
                # Load batch
                query = f"""
                    SELECT 
                        o.global_idx,
                        o.* EXCLUDE (global_idx, replay_id, step, filename),
                        a.action as true_label
                    FROM {obs_table} o
                    JOIN actions a ON o.global_idx = a.global_idx
                    ORDER BY o.global_idx
                    LIMIT {self.batch_size} OFFSET {offset}
                """
                
                batch_df = conn.execute(query).df()
                
                if len(batch_df) == 0:
                    break
                
                # Extract features and labels
                indices = batch_df['global_idx'].values
                X_batch = batch_df.drop(['global_idx', 'true_label'], axis=1).values.astype(np.float32)
                y_true = batch_df['true_label'].values
                
                # Clean data
                X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get predictions
                y_pred = self.model.predict(X_batch)
                y_proba = self.model.predict_proba(X_batch)
                
                # Store results
                all_indices.extend(indices)
                all_predictions.extend(y_pred)
                all_probabilities.extend(y_proba)
                all_true_labels.extend(y_true)
                
                # Update progress
                pbar.update(len(batch_df))
                offset += self.batch_size
                
                # Memory cleanup every 10 batches
                if offset % (self.batch_size * 10) == 0:
                    gc.collect()
            
            pbar.close()
            
            logger.info("\nCreating output DataFrame...")
            
            # Convert to numpy arrays
            all_indices = np.array(all_indices)
            all_predictions = np.array(all_predictions)
            all_probabilities = np.array(all_probabilities)
            all_true_labels = np.array(all_true_labels)
            
            # Create DataFrame with predictions
            predictions_dict = {
                'sample_idx': all_indices,
                'true_label': all_true_labels,
                'predicted_label': all_predictions
            }
            
            # Add probability columns
            for class_idx in range(19):
                predictions_dict[f'prob_class_{class_idx}'] = all_probabilities[:, class_idx].astype(np.float32)
            
            predictions_df = pa.Table.from_pydict(predictions_dict)
            
            # Save as Parquet
            logger.info(f"Saving predictions to {predictions_path}...")
            pq.write_table(
                predictions_df, 
                predictions_path,
                compression='snappy',
                use_dictionary=True,
                compression_level=None
            )
            
            file_size_gb = predictions_path.stat().st_size / 1e9
            logger.info(f"✓ Predictions saved: {file_size_gb:.2f} GB")
            
            # Calculate accuracy
            accuracy = (all_predictions == all_true_labels).mean()
            logger.info(f"✓ Prediction accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
            
            # Save metadata
            import json
            metadata = {
                'model_path': str(self.model_path),
                'ducklake_path': str(self.ducklake_path),
                'total_samples': int(total_count),
                'accuracy': float(accuracy),
                'model_accuracy': float(self.model_accuracy),
                'stack_frames': int(self.stack_frames),
                'n_classes': 19,
                'batch_size': int(self.batch_size),
                'file_size_gb': float(file_size_gb),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Metadata saved: {metadata_path}")
            
            # Create efficient loader example
            loader_example_path = self.output_dir / "example_loader.py"
            with open(loader_example_path, 'w') as f:
                f.write(f'''"""
Example: How to load predictions for Knowledge Distillation
"""

import pyarrow.parquet as pq
import torch
import numpy as np

def load_predictions_batch(start_idx, batch_size):
    """Load a batch of predictions for training"""
    
    # Load specific rows
    table = pq.read_table(
        '{predictions_path}',
        columns=['sample_idx', 'true_label'] + [f'prob_class_{{i}}' for i in range(19)],
        filters=[
            ('sample_idx', '>=', start_idx),
            ('sample_idx', '<', start_idx + batch_size)
        ]
    )
    
    df = table.to_pandas()
    
    # Extract soft targets (probabilities)
    soft_targets = df[[f'prob_class_{{i}}' for i in range(19)]].values
    true_labels = df['true_label'].values
    
    return torch.FloatTensor(soft_targets), torch.LongTensor(true_labels)

# Or load everything for small datasets
def load_all_predictions():
    """Load all predictions at once"""
    table = pq.read_table('{predictions_path}')
    df = table.to_pandas()
    
    soft_targets = df[[f'prob_class_{{i}}' for i in range(19)]].values
    true_labels = df['true_label'].values
    
    return torch.FloatTensor(soft_targets), torch.LongTensor(true_labels)

# Usage in training loop:
# soft_targets, true_labels = load_predictions_batch(batch_start, batch_size)
''')
            
            logger.info(f"✓ Example loader saved: {loader_example_path}")
            
        finally:
            conn.close()
        
        return predictions_path, accuracy
    
    def verify_predictions(self, predictions_path):
        """Verify the saved predictions"""
        logger.info("\n" + "="*70)
        logger.info("Verifying Predictions")
        logger.info("="*70)
        
        # Read sample
        table = pq.read_table(predictions_path, columns=['sample_idx', 'true_label', 'predicted_label'])
        df = table.to_pandas()
        
        logger.info(f"Total predictions: {len(df):,}")
        logger.info(f"Unique samples: {df['sample_idx'].nunique():,}")
        logger.info(f"Accuracy: {(df['true_label'] == df['predicted_label']).mean():.6f}")
        
        # Check probability sums
        prob_cols = [f'prob_class_{i}' for i in range(19)]
        sample_table = pq.read_table(predictions_path, columns=prob_cols, num_rows=1000)
        sample_df = sample_table.to_pandas()
        prob_sums = sample_df.sum(axis=1)
        
        logger.info(f"Probability sum check (should be ~1.0):")
        logger.info(f"  Mean: {prob_sums.mean():.6f}")
        logger.info(f"  Std: {prob_sums.std():.6f}")
        logger.info(f"  Min: {prob_sums.min():.6f}")
        logger.info(f"  Max: {prob_sums.max():.6f}")
        
        return True


def main():
    MODEL_PATH = r"C:\clones\rlib_gfootball\cold_start\xgboost_tuned\xgboost_memorized_4x_acc0.9999.pkl"
    DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
    OUTPUT_DIR = r"C:\clones\rlib_gfootball\cold_start\xgb_predictions"
    BATCH_SIZE = 50000

    saver = PredictionSaver(MODEL_PATH, DUCKLAKE_PATH, OUTPUT_DIR, BATCH_SIZE)
    predictions_path, accuracy = saver.compute_and_save_predictions()
    
    saver.verify_predictions(predictions_path)
    
    logger.info("\n" + "="*70)
    logger.info("✅ PREDICTIONS SAVED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"Location: {predictions_path}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"\n💡 Ready for Knowledge Distillation training!")
    logger.info(f"   Use the soft targets from: {predictions_path}")
    logger.info(f"   Example loader provided in: {OUTPUT_DIR}/example_loader.py")


if __name__ == "__main__":
    main()