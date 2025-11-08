import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
import time
import gc
from torch.utils.data import Dataset, DataLoader
import duckdb
import pyarrow.parquet as pq
from gymnasium import spaces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
PREDICTIONS_PATH = r"C:\clones\rlib_gfootball\cold_start\xgb_predictions\predictions.parquet"
OUTPUT_DIR = Path("C:/clones/rlib_gfootball/cold_start/final_model_training")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class DuckLakePredictionDataset(Dataset):
    """Dataset loader aus deinem ersten Script"""
    def __init__(self, ducklake_path: str, predictions_path: str, 
                 stack_frames: int = 4, indices: Optional[np.ndarray] = None):
        self.ducklake_path = Path(ducklake_path)
        self.predictions_path = Path(predictions_path)
        self.stack_frames = stack_frames
        
        pred_table = pq.read_table(self.predictions_path)
        self.pred_df = pred_table.to_pandas()
        
        if indices is not None:
            self.pred_df = self.pred_df[self.pred_df['sample_idx'].isin(indices)]
        
        self.pred_df = self.pred_df.sort_values('sample_idx').reset_index(drop=True)
        
        prob_cols = [f'prob_class_{i}' for i in range(19)]
        self.soft_targets = self.pred_df[prob_cols].values.astype(np.float32)
        self.true_labels = self.pred_df['true_label'].values.astype(np.int64)
        self.sample_indices = self.pred_df['sample_idx'].values
        
        self._load_features()
    
    def _load_features(self):
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        
        try:
            obs_table = "observations_stacked" if self.stack_frames > 1 else "observations"
            
            conn.execute(f"""
                CREATE TEMP TABLE selected_indices AS 
                SELECT * FROM (VALUES {','.join([f'({i})' for i in self.sample_indices])}) AS t(idx)
            """)
            
            query = f"""
                SELECT o.* EXCLUDE (global_idx, replay_id, step)
                FROM {obs_table} o
                JOIN selected_indices s ON o.global_idx = s.idx
                ORDER BY o.global_idx
            """
            
            features_df = conn.execute(query).df()
            self.features = features_df.values.astype(np.float32)
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

def load_best_config():
    """Lädt die beste Konfiguration aus der Architecture Search"""
    results_path = Path("C:/clones/rlib_gfootball/cold_start/architecture_search_scaled/results.json")
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        best = results[0]  # Assuming sorted by val_acc
        
        config = {
            'encoder_type': best['encoder'],
            'sequence_type': best['sequence'],
            'head_type': best['head'],
            'model_scale': best['size'],
            'n_features': 115,
            'stack_frames': 4,
            'num_frames': 4,
            
            # Standard parameters
            'encoder_output_dim': 48,
            'sequence_hidden_dim': 48,
            'encoder_hidden_dim': 256,
            'prev_action_emb_dim': 8,
            'policy_hidden_dims': [32, 16],
            'value_hidden_dims': [24, 12],
            'head_dropout': 0.1,
            
            # Component-specific
            'cnn_channels': [32, 64],
            'cnn_kernels': [8, 4],
            'gnn_hidden': 24,
            'gnn_layers': 2,
            'tcn_channels': [48, 48],
            'tcn_kernel_size': 3,
            'mamba_state_dim': 6,
            'kan_grid': 3,
        }
        
        logger.info(f"Beste Architektur: {best['name']} (Acc: {best['val_acc']:.4f})")
        return config
    else:
        # Fallback config wenn keine Results gefunden
        logger.warning("Keine results.json gefunden, nutze Standard-Config")
        return {
            'encoder_type': 'linear',
            'sequence_type': 'gru',
            'head_type': 'mlp',
            'model_scale': 'm',
            'n_features': 115,
            'stack_frames': 4,
            'num_frames': 4,
            'encoder_output_dim': 48,
            'sequence_hidden_dim': 48,
            'encoder_hidden_dim': 256,
            'prev_action_emb_dim': 8,
            'policy_hidden_dims': [32, 16],
            'value_hidden_dims': [24, 12],
            'head_dropout': 0.1,
        }

def train_simple():
    """Einfaches Training ohne fancy features"""
    
    # Config laden
    config = load_best_config()
    
    # Hyperparameters
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_SAMPLES = 100_000  # Erstmal mit weniger Samples testen
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Daten laden
    logger.info(f"Lade {NUM_SAMPLES} Samples...")
    pred_table = pq.read_table(PREDICTIONS_PATH, columns=['sample_idx'])
    all_indices = pred_table['sample_idx'].to_numpy()
    
    if NUM_SAMPLES < len(all_indices):
        all_indices = np.random.choice(all_indices, NUM_SAMPLES, replace=False)
    
    # Train/Val Split
    n_train = int(len(all_indices) * 0.8)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    
    logger.info("Erstelle Datasets...")
    train_dataset = DuckLakePredictionDataset(
        DUCKLAKE_PATH, PREDICTIONS_PATH, 
        stack_frames=config['stack_frames'], 
        indices=train_indices
    )
    val_dataset = DuckLakePredictionDataset(
        DUCKLAKE_PATH, PREDICTIONS_PATH,
        stack_frames=config['stack_frames'],
        indices=val_indices
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model erstellen
    from modular_models import ModularStudentModel
    
    obs_space = spaces.Box(-np.inf, np.inf, 
                           (config['stack_frames'], config['n_features']), 
                           dtype=np.float32)
    action_space = spaces.Discrete(19)
    
    model = ModularStudentModel(obs_space, action_space, 19, config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {num_params/1e6:.2f}M")
    
    # Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    temperature = 2.0
    alpha = 0.7
    
    best_val_acc = 0
    
    # Training Loop
    logger.info("Starte Training...")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        train_batches = 0
        
        for features, soft_targets, true_labels in train_loader:
            features = features.to(device)
            soft_targets = soft_targets.to(device)
            true_labels = true_labels.to(device).long()
            
            # Forward pass
            input_dict = {
                "obs_flat": features,
                "prev_actions": torch.zeros(features.size(0), dtype=torch.long, device=device)
            }
            
            batch_size = features.size(0)
            state = model.get_initial_state(batch_size, device)
            
            logits, _ = model(
                input_dict, state,
                torch.ones(features.size(0), dtype=torch.long, device=device)
            )
            
            # Loss calculation
            hard_loss = F.cross_entropy(logits, true_labels)
            
            soft_loss = F.kl_div(
                F.log_softmax(logits / temperature, dim=-1),
                F.softmax(soft_targets / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            loss = (1 - alpha) * hard_loss + alpha * soft_loss
            accuracy = (logits.argmax(dim=-1) == true_labels).float().mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_batches = 0
        
        with torch.no_grad():
            for features, soft_targets, true_labels in val_loader:
                features = features.to(device)
                soft_targets = soft_targets.to(device)
                true_labels = true_labels.to(device).long()
                
                input_dict = {
                    "obs_flat": features,
                    "prev_actions": torch.zeros(features.size(0), dtype=torch.long, device=device)
                }
                
                batch_size = features.size(0)
                state = model.get_initial_state(batch_size, device)
                
                logits, _ = model(
                    input_dict, state,
                    torch.ones(features.size(0), dtype=torch.long, device=device)
                )
                
                hard_loss = F.cross_entropy(logits, true_labels)
                soft_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    F.softmax(soft_targets / temperature, dim=-1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                loss = (1 - alpha) * hard_loss + alpha * soft_loss
                accuracy = (logits.argmax(dim=-1) == true_labels).float().mean()
                
                val_loss += loss.item()
                val_acc += accuracy.item()
                val_batches += 1
        
        # Logging
        avg_train_loss = train_loss / train_batches
        avg_train_acc = train_acc / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config
            }, OUTPUT_DIR / 'best_model.pth')
            logger.info(f"  -> Neues bestes Modell gespeichert!")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    logger.info(f"\nTraining abgeschlossen!")
    logger.info(f"Beste Val Accuracy: {best_val_acc:.4f}")
    
    return model, best_val_acc

if __name__ == "__main__":
    model, best_acc = train_simple()