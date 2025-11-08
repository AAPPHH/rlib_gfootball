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
import xgboost as xgb
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictionSaver:
    def __init__(self, model_path: str, ducklake_path: str, output_dir: str = "./predictions",
                 batch_size: int = 300_000):
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
        logger.info("\nConnecting to DuckDB lake...")
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        conn.execute("SET memory_limit = '32GB'")

        try:
            obs_table = "observations_stacked" if self.stack_frames > 1 else "observations"

            total_count = conn.execute(f"SELECT COUNT(*) FROM {obs_table}").fetchone()[0]
            logger.info(f"Total samples: {total_count:,}")
            predictions_path = self.output_dir / "predictions.parquet"
            metadata_path = self.output_dir / "metadata.json"

            logger.info(f"\nProcessing predictions in batches of {self.batch_size:,}...")

            all_predictions = []
            all_probabilities = []
            all_true_labels = []
            all_indices = []

            offset = 0
            pbar = tqdm(total=total_count, desc="Computing predictions")

            while offset < total_count:
                query = f"""
                    SELECT 
                        o.global_idx,
                        o.* EXCLUDE (global_idx, replay_id, step),
                        a.action as true_label
                    FROM {obs_table} o
                    JOIN actions a ON o.global_idx = a.global_idx
                    ORDER BY o.global_idx
                    LIMIT {self.batch_size} OFFSET {offset}
                """

                batch_df = conn.execute(query).df()

                if len(batch_df) == 0:
                    break

                indices = batch_df['global_idx'].values
                X_batch = batch_df.drop(['global_idx', 'true_label'], axis=1).values.astype(np.float32)
                y_true = batch_df['true_label'].values

                X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)

                d_batch = xgb.DMatrix(X_batch)

                y_proba = self.model.predict(d_batch)
                y_pred = np.argmax(y_proba, axis=1)
                
                all_indices.extend(indices)
                all_predictions.extend(y_pred)
                all_probabilities.extend(y_proba)
                all_true_labels.extend(y_true)

                pbar.update(len(batch_df))
                offset += self.batch_size

                if offset % (self.batch_size * 10) == 0:
                    gc.collect()

            pbar.close()

            logger.info("\nCreating output DataFrame...")

            all_indices = np.array(all_indices)
            all_predictions = np.array(all_predictions)
            all_probabilities = np.array(all_probabilities)
            all_true_labels = np.array(all_true_labels)

            predictions_dict = {
                'sample_idx': all_indices,
                'true_label': all_true_labels,
                'predicted_label': all_predictions
            }

            n_classes = all_probabilities.shape[1]
            if n_classes != 19:
                logger.warning(f"Modell hat {n_classes} Klassen zurückgegeben, nicht 19. Passe Spalten an.")

            for class_idx in range(n_classes):
                predictions_dict[f'prob_class_{class_idx}'] = all_probabilities[:, class_idx].astype(np.float32)
            
            for class_idx in range(n_classes, 19):
                logger.warning(f"Fülle fehlende prob_class_{class_idx} mit Nullen auf.")
                predictions_dict[f'prob_class_{class_idx}'] = np.zeros(len(all_indices), dtype=np.float32)


            predictions_df = pa.Table.from_pydict(predictions_dict)

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

            accuracy = (all_predictions == all_true_labels).mean()
            logger.info(f"✓ Prediction accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")

            metadata = {
                'model_path': str(self.model_path),
                'ducklake_path': str(self.ducklake_path),
                'total_samples': int(total_count),
                'accuracy': float(accuracy),
                'model_accuracy': float(self.model_accuracy),
                'stack_frames': int(self.stack_frames),
                'n_classes': int(n_classes),
                'batch_size': int(self.batch_size),
                'file_size_gb': float(file_size_gb),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✓ Metadata saved: {metadata_path}")

        finally:
            conn.close()

        return predictions_path, accuracy

    def verify_predictions(self, predictions_path):
        """Verify the saved predictions"""
        logger.info("\n" + "="*70)
        logger.info("Verifying Predictions")
        logger.info("="*70)

        table = pq.read_table(predictions_path, columns=['sample_idx', 'true_label', 'predicted_label'])
        df = table.to_pandas()

        logger.info(f"Total predictions: {len(df):,}")
        logger.info(f"Unique samples: {df['sample_idx'].nunique():,}")
        logger.info(f"Accuracy: {(df['true_label'] == df['predicted_label']).mean():.6f}")

        try:
            with open(self.output_dir / "metadata.json", 'r') as f:
                n_classes = json.load(f).get('n_classes', 19)
        except FileNotFoundError:
            n_classes = 19
            logger.warning("Konnte Metadaten nicht lesen, nehme 19 Klassen für Verifizierung an.")
        prob_cols = [f'prob_class_{i}' for i in range(n_classes)]

        sample_table = pq.read_table(predictions_path, columns=prob_cols)
        sample_table_slice = sample_table.slice(length=1000)
        sample_df = sample_table_slice.to_pandas()
        
        prob_sums = sample_df.sum(axis=1)

        logger.info(f"Probability sum check (should be ~1.0):")
        logger.info(f"  Mean: {prob_sums.mean():.6f}")
        logger.info(f"  Std: {prob_sums.std():.6f}")
        logger.info(f"  Min: {prob_sums.min():.6f}")
        logger.info(f"  Max: {prob_sums.max():.6f}")

        return True


def main():
    MODEL_PATH = r"C:\clones\rlib_gfootball\cold_start\manual_model_output\xgboost_manual_trial2_acc0.6201_20251107_181416.pkl"
    DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
    OUTPUT_DIR = r"C:\clones\rlib_gfootball\cold_start\xgb_predictions"
    BATCH_SIZE = 250_000

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