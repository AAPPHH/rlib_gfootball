"""
Step 4: Neural Network Training with Saved XGBoost Predictions
Uses the predictions saved in Step 3 for efficient knowledge distillation
Directly loads from DuckDB lake and prediction parquet files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from gymnasium import spaces
from torch.utils.data import Dataset, DataLoader
import duckdb
import pyarrow.parquet as pq
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import gc
import logging

# Import your existing model classes
from modular_models import (
    LinearEncoder, CNNEncoder, GNNEncoder,
    GRUSequence, LSTMSequence, TCNSequence, MambaSequence,
    MLPHead, KANHead,
    ModularStudentModel
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DuckLakePredictionDataset(Dataset):
    """
    Dataset that efficiently loads from DuckDB lake and pre-computed predictions
    """
    def __init__(self, ducklake_path: str, predictions_path: str, 
                 stack_frames: int = 4, indices: Optional[np.ndarray] = None):
        """
        Args:
            ducklake_path: Path to DuckDB lake
            predictions_path: Path to saved XGBoost predictions parquet
            stack_frames: Number of stacked frames
            indices: Optional indices for train/val split
        """
        self.ducklake_path = Path(ducklake_path)
        self.predictions_path = Path(predictions_path)
        self.stack_frames = stack_frames
        
        logger.info(f"Loading dataset from DuckDB lake and predictions...")
        
        # Load predictions (soft targets)
        pred_table = pq.read_table(self.predictions_path)
        self.pred_df = pred_table.to_pandas()
        
        # If indices provided, filter
        if indices is not None:
            self.pred_df = self.pred_df[self.pred_df['sample_idx'].isin(indices)]
        
        self.pred_df = self.pred_df.sort_values('sample_idx').reset_index(drop=True)
        
        # Extract soft targets and labels
        prob_cols = [f'prob_class_{i}' for i in range(19)]
        self.soft_targets = self.pred_df[prob_cols].values.astype(np.float32)
        self.true_labels = self.pred_df['true_label'].values.astype(np.int64)
        self.sample_indices = self.pred_df['sample_idx'].values
        
        # Load features from DuckDB
        self._load_features()
        
        logger.info(f"✓ Dataset loaded: {len(self)} samples")
    
    def _load_features(self):
        """Load feature data from DuckDB lake"""
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        
        try:
            # Determine table to use
            obs_table = "observations_stacked" if self.stack_frames > 1 else "observations"
            
            # Create temp table with our indices
            indices_str = ','.join(map(str, self.sample_indices))
            conn.execute(f"""
                CREATE TEMP TABLE selected_indices AS 
                SELECT * FROM (VALUES {','.join([f'({i})' for i in self.sample_indices])}) AS t(idx)
            """)
            
            # Load features for our indices
            query = f"""
                SELECT o.* EXCLUDE (global_idx, replay_id, step, filename)
                FROM {obs_table} o
                WHERE o.global_idx IN (SELECT idx FROM selected_indices)
                ORDER BY o.global_idx
            """
            
            features_df = conn.execute(query).df()
            self.features = features_df.values.astype(np.float32)
            
            # Clean data
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
            
        finally:
            conn.close()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.soft_targets[idx]),
            self.true_labels[idx]
        )


class ImprovedDistillationTrainer:
    """
    Improved trainer that uses pre-computed XGBoost predictions
    """
    def __init__(self, student_model: nn.Module, device: str = 'cuda', 
                 temperature: float = 2.0, alpha: float = 0.7):
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.student_model = student_model.to(device)
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    def distillation_loss(self, student_logits, teacher_probs, true_labels):
        """Combined distillation and hard label loss"""
        # Hard loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, true_labels)
        
        # Soft loss (KL divergence with teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_probs / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        # Calculate accuracy
        accuracy = (student_logits.argmax(dim=-1) == true_labels).float().mean()
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'accuracy': accuracy.item()
        }
    
    def train_epoch(self, loader, optimizer, scheduler=None):
        """Train for one epoch"""
        self.student_model.train()
        metrics = {'total_loss': 0.0, 'hard_loss': 0.0, 'soft_loss': 0.0, 'accuracy': 0.0}
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for features, soft_targets, true_labels in pbar:
            # Move to device
            features = features.to(self.device)
            soft_targets = soft_targets.to(self.device)
            true_labels = true_labels.to(self.device).long()
            
            # Prepare input
            input_dict = {
                "obs_flat": features,
                "prev_actions": torch.zeros(features.size(0), dtype=torch.long, device=self.device)
            }
            
            # Get initial state
            state = [s.to(self.device) for s in self.student_model.get_initial_state()]
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, _ = self.student_model(
                        input_dict, state, 
                        torch.ones(features.size(0), dtype=torch.long, device=self.device)
                    )
                    loss, batch_metrics = self.distillation_loss(logits, soft_targets, true_labels)
                
                # Backward pass with scaling
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                logits, _ = self.student_model(
                    input_dict, state,
                    torch.ones(features.size(0), dtype=torch.long, device=self.device)
                )
                loss, batch_metrics = self.distillation_loss(logits, soft_targets, true_labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                optimizer.step()
            
            # Update metrics
            for k in metrics:
                metrics[k] += batch_metrics[k]
            
            # Update progress bar
            pbar.set_postfix({'loss': batch_metrics['total_loss'], 
                             'acc': batch_metrics['accuracy']})
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(loader)
        
        if scheduler is not None:
            scheduler.step()
        
        return metrics
    
    def validate(self, loader):
        """Validation pass"""
        self.student_model.eval()
        metrics = {'total_loss': 0.0, 'hard_loss': 0.0, 'soft_loss': 0.0, 'accuracy': 0.0}
        
        with torch.no_grad():
            for features, soft_targets, true_labels in tqdm(loader, desc="Validation", leave=False):
                features = features.to(self.device)
                soft_targets = soft_targets.to(self.device)
                true_labels = true_labels.to(self.device).long()
                
                input_dict = {
                    "obs_flat": features,
                    "prev_actions": torch.zeros(features.size(0), dtype=torch.long, device=self.device)
                }
                
                state = [s.to(self.device) for s in self.student_model.get_initial_state()]
                
                logits, _ = self.student_model(
                    input_dict, state,
                    torch.ones(features.size(0), dtype=torch.long, device=self.device)
                )
                
                _, batch_metrics = self.distillation_loss(logits, soft_targets, true_labels)
                
                for k in metrics:
                    metrics[k] += batch_metrics[k]
        
        for k in metrics:
            metrics[k] /= len(loader)
        
        return metrics


def train_with_ducklake_predictions(
    ducklake_path: str,
    predictions_path: str,
    output_dir: str,
    config: Dict[str, Any],
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    val_split: float = 0.1
):
    """
    Main training function using DuckDB lake and saved predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("Training with DuckLake and XGBoost Predictions")
    logger.info("="*70)
    logger.info(f"DuckLake: {ducklake_path}")
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load metadata to get dataset info
    pred_table = pq.read_table(predictions_path, columns=['sample_idx'])
    all_indices = pred_table['sample_idx'].to_numpy()
    n_samples = len(all_indices)
    
    # Split indices
    n_train = int(n_samples * (1 - val_split))
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train_indices):,}")
    logger.info(f"  Val: {len(val_indices):,}")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = DuckLakePredictionDataset(
        ducklake_path, predictions_path, 
        stack_frames=config.get('stack_frames', 4),
        indices=train_indices
    )
    
    val_dataset = DuckLakePredictionDataset(
        ducklake_path, predictions_path,
        stack_frames=config.get('stack_frames', 4),
        indices=val_indices
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    logger.info("\nCreating model...")
    obs_space = spaces.Box(
        -np.inf, np.inf, 
        (config['stack_frames'], config['n_features']), 
        dtype=np.float32
    )
    
    model = ModularStudentModel(
        obs_space, spaces.Discrete(19), 19, config
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ImprovedDistillationTrainer(
        model, device,
        temperature=config.get('temperature', 2.0),
        alpha=config.get('alpha', 0.7)
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )
    
    # Training loop
    logger.info(f"\nStarting training for {epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler)
        logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        logger.info(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✓ Saved best model (acc: {best_val_acc:.4f})")
        
        # Memory cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ Training Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    
    return best_val_acc


def main():
    # Paths
    DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
    PREDICTIONS_PATH = r"C:\clones\rlib_gfootball\cold_start\xgb_predictions\predictions.parquet"
    OUTPUT_DIR = r"C:\clones\rlib_gfootball\cold_start\neural_models"
    
    # Best config from your search (example - use your actual best config)
    config = {
        'stack_frames': 4,
        'n_features': 115,  # base features
        
        # Architecture
        'encoder_type': 'gnn',
        'sequence_type': 'mamba',
        'head_type': 'kan',
        
        # Encoder params
        'encoder_output_dim': 48,
        'encoder_hidden_dim': 256,
        'gnn_type': 'sage',
        'gnn_hidden': 24,
        'gnn_layers': 2,
        'gnn_k_neighbors': 6,
        'gnn_dropout': 0.1,
        
        # Sequence params
        'sequence_hidden_dim': 48,
        'sequence_num_layers': 2,
        'mamba_state_dim': 6,
        
        # Head params
        'policy_hidden_dims': [32, 16],
        'value_hidden_dims': [24, 12],
        'head_dropout': 0.1,
        'kan_grid': 3,
        
        # Other
        'prev_action_emb_dim': 8,
        'num_frames': 4,
        
        # Training params
        'temperature': 2.0,
        'alpha': 0.7,
        'weight_decay': 1e-4
    }
    
    # Train
    best_acc = train_with_ducklake_predictions(
        DUCKLAKE_PATH,
        PREDICTIONS_PATH,
        OUTPUT_DIR,
        config,
        epochs=30,
        batch_size=512,
        learning_rate=1e-3,
        val_split=0.1
    )
    
    print(f"\n🎯 Final best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()