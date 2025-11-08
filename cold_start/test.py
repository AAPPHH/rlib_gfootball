#!/usr/bin/env python3

import os
import sys
import time
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['NUMEXPR_MAX_THREADS'] = '20'

# Globals to hold the data
X_global = None
y_global = None
groups_global = None

# --- Data Loading Function (copied from your original script) ---
def load_data(ducklake_path, stack_frames):
    global X_global, y_global, groups_global
    
    conn = duckdb.connect(str(ducklake_path), read_only=True)
    conn.execute("SET threads TO 2")
    conn.execute("SET memory_limit = '4GB'")
    
    try:
        obs_table = "observations_stacked" if stack_frames > 1 else "observations"
        
        query = f"""
            SELECT 
                o.*,
                a.action
            FROM {obs_table} o
            JOIN actions a ON o.global_idx = a.global_idx
            ORDER BY o.global_idx
        """
        
        data = conn.execute(query).df()
        
        groups_global = data["replay_id"].values
        X_global = data.drop(columns=["action", "replay_id", "global_idx", "step"], errors="ignore").values.astype(np.float32)
        y_global = data['action'].values.astype(np.int32)
        X_global = np.nan_to_num(X_global, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Loaded {len(X_global):,} samples from {len(np.unique(groups_global))} replays")
        logger.info(f"Features: {X_global.shape[1]}, Classes: {len(np.unique(y_global))}")
        
        gc.collect()
        
    finally:
        conn.close()

def train_and_save_single_model():
    # --- Konfiguration ---
    DUCKLAKE_PATH =r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
    # (Ich habe ein neues Verzeichnis für die manuelle Speicherung erstellt)
    OUTPUT_DIR = r"C:\clones\rlib_gfootball\cold_start\manual_model_output"
    STACK_FRAMES = 4
    
    # --- 1. Daten laden ---
    logger.info("Loading data...")
    load_data(DUCKLAKE_PATH, STACK_FRAMES)
    
    # --- 2. Parameter von Trial 2 ---
    # (Parameter aus deinem Log-Eintrag)
    trial_params = {
        'n_estimators': 900, 
        'max_depth': 6, 
        'learning_rate': 0.0438161514621447, 
        'subsample': 0.8542703315240835, 
        'colsample_bytree': 0.836965827544817, 
        'colsample_bylevel': 0.6185801650879991, 
        'colsample_bynode': 0.8430179407605753, 
        'reg_lambda': 1.4808945119975185, 
        'reg_alpha': 0.3252579649263976, 
        'min_child_weight': 29, 
        'gamma': 1.9312640661491187, 
        'max_delta_step': 4.041986740582305, 
        'max_bin': 256
    }
    
    # --- 3. Basis-Parameter (aus deinem originalen train_final_model) ---
    # (Sicherstellen, dass GPU, Objective etc. gesetzt sind)
    params = {
        'objective': 'multi:softprob',
        'num_class': 19,
        'tree_method': 'hist',
        'device': 'cuda:0', # (Sicherstellen, dass GPU 0 verwendet wird)
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 42,
        'verbosity': 2,
    }
    
    # `n_estimators` aus den Trial-Parametern wird zu `num_boost_round`
    num_boost_round = trial_params.pop('n_estimators')
    
    # Kombiniere die Basis-Parameter mit den Trial-Parametern
    params.update(trial_params)
    
    logger.info(f"Training final model with {num_boost_round} rounds...")
    logger.info(f"Parameters: {params}")
    
    # --- 4. Training ---
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # (Setzt die sichtbare GPU)
    
    dtrain = xgb.DMatrix(X_global, label=y_global)
    
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=5 # Zeigt alle 50 Runden den Fortschritt
    )
    
    train_time = time.time() - start_time
    
    # --- 5. Genauigkeit berechnen (auf Trainingsdaten) ---
    train_pred = model.predict(dtrain)
    train_accuracy = np.mean(np.argmax(train_pred, axis=1) == y_global)
    
    logger.info(f"Training finished in {train_time:.2f}s")
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    
    # --- 6. Speichern (wie in deinem originalen Skript) ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_manual_trial2_acc{train_accuracy:.4f}_{timestamp}"
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Als Pickle speichern
    pkl_path = output_path / f"{model_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'params': params,
            'train_accuracy': train_accuracy,
            'train_time': train_time,
            'num_boost_round': num_boost_round,
        }, f)
    
    # Als JSON (XGBoost-Format) speichern
    json_path = output_path / f"{model_name}.json"
    model.save_model(str(json_path))
    
    logger.info(f"Model saved to: {pkl_path}")
    logger.info(f"Model also saved to: {json_path}")
    logger.info("Training complete!")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(train_and_save_single_model())
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)