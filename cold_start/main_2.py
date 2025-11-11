import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import time
import gc
import pickle
from torch.utils.data import Dataset, DataLoader
import duckdb
import pyarrow.parquet as pq
from gymnasium import spaces

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
XGBOOST_MODEL_PATH = r"C:\clones\rlib_gfootball\cold_start\xgboost_coldstart_acc0.6813_20251109_043825.pkl"
OUTPUT_DIR = Path("C:/clones/rlib_gfootball/cold_start/mamba_distillation_training")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def calculate_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Calculate top-k accuracy (simplified version)"""
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        return correct.item()

class DuckLakeDataset(Dataset):
    """Dataset loader for raw features from DuckLake"""
    def __init__(self, ducklake_path: str, stack_frames: int = 4, 
                 num_samples: int = 100000, offset: int = 0):
        self.ducklake_path = Path(ducklake_path)
        self.stack_frames = stack_frames
        self.num_samples = num_samples
        self.offset = offset
        
        self._load_data()
    
    def _load_data(self):
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        
        try:
            obs_table = "observations_stacked" if self.stack_frames > 1 else "observations"
            
            query_count = f"SELECT COUNT(*) FROM {obs_table}"
            total_count = conn.execute(query_count).fetchone()[0]
            
            end_idx = min(self.offset + self.num_samples, total_count)
            actual_samples = end_idx - self.offset
            
            logger.info(f"Loading {actual_samples} samples from {total_count} total (offset: {self.offset})")
            
            if self.stack_frames > 1:
                sample = conn.execute(f"SELECT * FROM {obs_table} LIMIT 1").df()
                feature_cols = [c for c in sample.columns if c.startswith('feat_')]
                features_str = ', '.join(feature_cols)
                
                query = f"""
                    SELECT o.global_idx, {features_str}
                    FROM {obs_table} o
                    WHERE o.global_idx >= {self.offset} AND o.global_idx < {end_idx}
                    ORDER BY o.global_idx
                """
            else:
                query = f"""
                    SELECT o.global_idx, o.* EXCLUDE (global_idx, replay_id, step)
                    FROM {obs_table} o
                    WHERE o.global_idx >= {self.offset} AND o.global_idx < {end_idx}
                    ORDER BY o.global_idx
                """
            
            obs_df = conn.execute(query).df()
            
            indices = obs_df['global_idx'].values
            indices_str = ','.join(map(str, indices))
            
            action_query = f"""
                SELECT a.global_idx, a.action
                FROM actions a
                WHERE a.global_idx IN ({indices_str})
                ORDER BY a.global_idx
            """
            
            actions_df = conn.execute(action_query).df()
            merged_df = obs_df.merge(actions_df, on='global_idx', how='left')
            
            feature_cols = [col for col in merged_df.columns if col.startswith('feat_')]
            self.features = merged_df[feature_cols].values.astype(np.float32)
            self.labels = merged_df['action'].fillna(0).values.astype(np.int64)
            
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Loaded {len(self.features)} samples with {self.features.shape[1]} features")
            
        finally:
            conn.close()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            self.labels[idx]
        )

class XGBoostTeacher:
    """Wrapper for XGBoost model to generate soft targets"""
    def __init__(self, model_path: str):
        logger.info(f"Loading XGBoost model from {model_path}")
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        
        if isinstance(loaded_obj, dict):
            if 'model' in loaded_obj:
                self.model = loaded_obj['model']
            elif 'xgb_model' in loaded_obj:
                self.model = loaded_obj['xgb_model']
            elif 'classifier' in loaded_obj:
                self.model = loaded_obj['classifier']
            else:
                for key, value in loaded_obj.items():
                    if hasattr(value, 'predict') or hasattr(value, 'predict_proba'):
                        self.model = value
                        logger.info(f"Found model under key: {key}")
                        break
                else:
                    raise ValueError(f"Could not find model in dictionary. Keys: {list(loaded_obj.keys())}")
        else:
            self.model = loaded_obj
        
        logger.info(f"XGBoost model type: {type(self.model)}")
        
        if hasattr(self.model, 'predict_proba'):
            self.predict_fn = self.model.predict_proba
            logger.info("Using predict_proba method")
        elif hasattr(self.model, 'predict'):
            try:
                import xgboost as xgb
                if isinstance(self.model, xgb.XGBClassifier):
                    self.predict_fn = self.model.predict_proba
                else:
                    # Native XGBoost booster
                    self.predict_fn = lambda x: self.model.predict(xgb.DMatrix(x))
                logger.info("Using XGBoost predict method")
            except ImportError:
                self.predict_fn = self.model.predict
                logger.info("Using generic predict method")
        else:
            raise ValueError(f"Model has no predict or predict_proba method: {type(self.model)}")
    
    @torch.no_grad()
    def get_soft_targets(self, features: np.ndarray) -> np.ndarray:
        """Generate soft targets from XGBoost model - FIX: NO temperature here"""
        if len(features.shape) > 2:
            batch_size = features.shape[0]
            features = features.reshape(batch_size, -1)
        
        try:
            probs = self.predict_fn(features)
            
            if len(probs.shape) == 1:
                # Convert to one-hot if it's just class predictions
                batch_size = features.shape[0]
                probs_2d = np.zeros((batch_size, 19), dtype=np.float32)
                probs_2d[np.arange(batch_size), probs.astype(int)] = 1.0
                probs = probs_2d
            
        except Exception as e:
            logger.warning(f"Prediction failed: {e}, using uniform distribution")
            batch_size = features.shape[0]
            probs = np.ones((batch_size, 19), dtype=np.float32) / 19.0
        
        # Sicherheitshalber normalisieren (ohne Temperature!)
        probs = probs.astype(np.float32)
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        
        return probs

def create_mamba_model(config: dict, device: torch.device) -> nn.Module:
    """Create the GFootballMamba model"""
    # FIX: Correct import path
    from model_3 import GFootballMamba
    
    n_features = config['n_features'] * config['stack_frames']
    obs_space = spaces.Box(-np.inf, np.inf, (n_features,), dtype=np.float32)
    action_space = spaces.Discrete(19)
    
    model_config = {
        "custom_model_config": {
            "d_model": config['d_model'],
            "mamba_state": config['mamba_state'],
            "num_mamba_layers": config['num_mamba_layers'],
            "prev_action_emb": config['prev_action_emb'],
            "gradient_checkpointing": config['gradient_checkpointing'],
            "mlp_hidden_dims": config['mlp_hidden_dims'],
            "mlp_activation": config['mlp_activation'],
            "head_hidden_dims": config['head_hidden_dims'],
            "head_activation": config['head_activation'],
            "use_noisy": config['use_noisy'],
            "use_distributional": config['use_distributional'],
            "v_min": config['v_min'],
            "v_max": config['v_max'],
            "num_atoms": config['num_atoms'],
        }
    }
    
    model = GFootballMamba(obs_space, action_space, 19, model_config, "mamba_student")
    return model.to(device)

def train_with_distillation():
    """Main training function with XGBoost distillation"""
    
    config = {
        'n_features': 115,
        'stack_frames': 4,
        'd_model': 48,
        'mamba_state': 6,
        'num_mamba_layers': 6,
        'prev_action_emb': 8,
        'gradient_checkpointing': True,
        'mlp_hidden_dims': [256, 128],
        'mlp_activation': 'silu',
        'head_hidden_dims': [128],
        'head_activation': 'silu',
        'use_noisy': True,
        'use_distributional': True,
        'v_min': -10.0,
        'v_max': 10.0,
        'num_atoms': 51,
    }
    
    # Training hyperparameters
    BATCH_SIZE = 8192
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    NUM_SAMPLES = 200000
    TEACHER_TEMP = 1.5  # Erhöht für mehr Softness
    STUDENT_TEMP = 1.5  # Gleich wie Teacher für Konsistenz
    ALPHA_INITIAL = 0.9
    ALPHA_FINAL = 0.1
    ALPHA_DECAY_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 20
    TOP_K_VALUES = [1, 3, 5]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    teacher = XGBoostTeacher(XGBOOST_MODEL_PATH)
    
    conn = duckdb.connect(str(DUCKLAKE_PATH), read_only=True)
    obs_table = "observations_stacked" if config['stack_frames'] > 1 else "observations"
    total_available = conn.execute(f"SELECT COUNT(*) FROM {obs_table}").fetchone()[0]
    conn.close()
    
    NUM_SAMPLES = min(NUM_SAMPLES, total_available)
    logger.info(f"Total available samples: {total_available}, using: {NUM_SAMPLES}")
    
    logger.info(f"Loading {NUM_SAMPLES} samples...")
    n_train = int(NUM_SAMPLES * 0.8)
    n_val = NUM_SAMPLES - n_train
    
    train_dataset = DuckLakeDataset(DUCKLAKE_PATH, config['stack_frames'], 
                                   num_samples=n_train, offset=0)
    val_dataset = DuckLakeDataset(DUCKLAKE_PATH, config['stack_frames'],
                                 num_samples=n_val, offset=n_train)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    model = create_mamba_model(config, device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {num_params/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                                 weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    best_val_top5 = 0
    epochs_no_improve = 0
    
    logger.info("Starting distillation training with FIXED temperature scaling...")
    logger.info(f"Teacher Temp: {TEACHER_TEMP}, Student Temp: {STUDENT_TEMP}")
    
    for epoch in range(EPOCHS):
        if epoch < ALPHA_DECAY_EPOCHS:
            progress = epoch / ALPHA_DECAY_EPOCHS
            alpha = ALPHA_INITIAL * (1 - progress) + ALPHA_FINAL * progress
        else:
            alpha = ALPHA_FINAL
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Alpha (teacher weight): {alpha:.3f}")
        logger.info(f"{'='*60}")
        
        # Training
        model.train()
        if hasattr(model, 'reset_noise'):
            model.reset_noise()
        
        train_loss = 0
        train_topk_accs = {k: 0 for k in TOP_K_VALUES}
        train_batches = 0
        
        for batch_idx, (features, true_labels) in enumerate(train_loader):
            features = features.to(device)
            true_labels = true_labels.to(device).long()
            batch_size = features.size(0)
            
            # Get teacher predictions (NO temperature in get_soft_targets!)
            features_np = features.cpu().numpy()
            soft_targets = teacher.get_soft_targets(features_np)
            soft_targets = torch.from_numpy(soft_targets).to(device)
            
            input_dict = {
                "obs": features.view(batch_size, -1),
                "obs_flat": features.view(batch_size, -1),
                "prev_actions": torch.zeros(batch_size, dtype=torch.long, device=device)
            }
            
            state = [torch.zeros(batch_size, model.state_size, device=device) 
                    for _ in range(model.num_mamba_layers)]
            seq_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
            
            logits, new_state = model(input_dict, state, seq_lens)
            
            # Hard loss
            hard_loss = F.cross_entropy(logits, true_labels)
            
            # FIX: Correct distillation loss
            # Convert teacher probs to logits for temperature scaling
            log_probs_student = F.log_softmax(logits / STUDENT_TEMP, dim=-1)
            # Apply temperature to teacher probs via log-space
            teacher_logits = torch.log(soft_targets + 1e-8)
            probs_teacher = F.softmax(teacher_logits / TEACHER_TEMP, dim=-1)
            
            soft_loss = F.kl_div(
                log_probs_student,
                probs_teacher,
                reduction='batchmean'
            ) * (TEACHER_TEMP ** 2)
            
            # Combined loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            # Calculate Top-k accuracies
            for k in TOP_K_VALUES:
                acc_k = calculate_topk_accuracy(logits, true_labels, k)
                train_topk_accs[k] += acc_k
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, "
                          f"Top-1={train_topk_accs[1]/(batch_idx+1):.3f}, "
                          f"Top-3={train_topk_accs[3]/(batch_idx+1):.3f}, "
                          f"Top-5={train_topk_accs[5]/(batch_idx+1):.3f}, "
                          f"LR={current_lr:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_topk_accs = {k: 0 for k in TOP_K_VALUES}
        val_teacher_agreement = 0
        val_batches = 0
        
        with torch.no_grad():
            for features, true_labels in val_loader:
                features = features.to(device)
                true_labels = true_labels.to(device).long()
                batch_size = features.size(0)
                
                # Get teacher predictions (NO temperature here!)
                features_np = features.cpu().numpy()
                soft_targets = teacher.get_soft_targets(features_np)
                soft_targets_tensor = torch.from_numpy(soft_targets).to(device)
                teacher_preds = soft_targets_tensor.argmax(dim=-1)
                
                input_dict = {
                    "obs": features.view(batch_size, -1),
                    "obs_flat": features.view(batch_size, -1),
                    "prev_actions": torch.zeros(batch_size, dtype=torch.long, device=device)
                }
                
                state = [torch.zeros(batch_size, model.state_size, device=device) 
                        for _ in range(model.num_mamba_layers)]
                seq_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
                
                logits, _ = model(input_dict, state, seq_lens)
                
                # Losses
                hard_loss = F.cross_entropy(logits, true_labels)
                
                # FIX: Same as training
                log_probs_student = F.log_softmax(logits / STUDENT_TEMP, dim=-1)
                teacher_logits = torch.log(soft_targets_tensor + 1e-8)
                probs_teacher = F.softmax(teacher_logits / TEACHER_TEMP, dim=-1)
                
                soft_loss = F.kl_div(
                    log_probs_student,
                    probs_teacher,
                    reduction='batchmean'
                ) * (TEACHER_TEMP ** 2)
                
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                # Calculate Top-k accuracies
                for k in TOP_K_VALUES:
                    acc_k = calculate_topk_accuracy(logits, true_labels, k)
                    val_topk_accs[k] += acc_k
                
                student_preds = logits.argmax(dim=-1)
                teacher_agree = (student_preds == teacher_preds).float().mean()
                
                val_loss += loss.item()
                val_teacher_agreement += teacher_agree.item()
                val_batches += 1
        
        # Epoch summary
        avg_train_loss = train_loss / train_batches
        avg_train_topk = {k: train_topk_accs[k] / train_batches for k in TOP_K_VALUES}
        avg_val_loss = val_loss / val_batches
        avg_val_topk = {k: val_topk_accs[k] / val_batches for k in TOP_K_VALUES}
        avg_teacher_agree = val_teacher_agreement / val_batches
        
        logger.info(f"\n{'-'*40}")
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Train - Loss: {avg_train_loss:.4f}")
        logger.info(f"         Top-1: {avg_train_topk[1]:.4f}, Top-3: {avg_train_topk[3]:.4f}, Top-5: {avg_train_topk[5]:.4f}")
        logger.info(f"  Val   - Loss: {avg_val_loss:.4f}")
        logger.info(f"         Top-1: {avg_val_topk[1]:.4f}, Top-3: {avg_val_topk[3]:.4f}, Top-5: {avg_val_topk[5]:.4f}")
        logger.info(f"  Teacher Agreement: {avg_teacher_agree:.4f}")
        
        # Save based on Top-5 accuracy
        if avg_val_topk[5] > best_val_top5:
            best_val_top5 = avg_val_topk[5]
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_top1': avg_val_topk[1],
                'val_top3': avg_val_topk[3],
                'val_top5': avg_val_topk[5],
                'teacher_agreement': avg_teacher_agree,
                'config': config,
                'alpha': alpha
            }
            torch.save(checkpoint, OUTPUT_DIR / 'best_model.pth')
            logger.info(f"  ✓ New best model saved! (Top-5: {best_val_top5:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"  → No improvement. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"\n🛑 Early stopping triggered after {epoch + 1} epochs.")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_topk': avg_val_topk,
                'config': config
            }
            torch.save(checkpoint, OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pth')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"Best Validation Top-5 Accuracy: {best_val_top5:.4f}")
    logger.info(f"{'='*60}")
    
    return model, best_val_top5

if __name__ == "__main__":
    model, best_top5 = train_with_distillation()
    print(f"\n✅ Training complete! Best Top-5 accuracy: {best_top5:.4f}")