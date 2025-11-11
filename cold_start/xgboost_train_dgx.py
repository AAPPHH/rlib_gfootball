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
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc
import joblib

# Imports für die GPU-Queue
import multiprocessing
import functools

os.environ['NUMEXPR_MAX_THREADS'] = '256'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

X_global = None
y_global = None
groups_global = None

def load_data(ducklake_path, stack_frames):
    global X_global, y_global, groups_global
    
    conn = duckdb.connect(str(ducklake_path), read_only=True)
    conn.execute("SET threads TO 64")
    conn.execute("SET memory_limit = '400GB'")
    
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

# --- KORRIGIERTE Objective-Funktion ---
def objective(trial, gpu_queue):
    gpu_id = gpu_queue.get()
    
    # Korrekte Device-Zuweisung für XGBoost >= 2.0
    device_string = f'cuda:{gpu_id}'
    trial.set_user_attr('gpu_id', gpu_id)
    
    try:
        params = {
            'objective': 'multi:softprob',
            'num_class': 19,
            'tree_method': 'hist',
            'device': device_string,  # <-- KORREKTUR: 'cuda:0', 'cuda:1' etc.
            # 'gpu_id': gpu_id,      # <-- ENTFERNT: Verursacht Absturz in >= 2.0
            'eval_metric': ['mlogloss', 'merror'],
            'seed': 42,
            'verbosity': 0,
            
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'gamma': trial.suggest_float('gamma', 0.0, 3.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 5.0),
            'max_bin': trial.suggest_int('max_bin', 128, 512),
        }

        gkf = GroupKFold(n_splits=5)
        accuracies = []
        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_global, y_global, groups_global)):
            X_train, X_val = X_global[train_idx], X_global[val_idx]
            y_train, y_val = y_global[train_idx], y_global[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'eval')],
                early_stopping_rounds=50,
                evals_result=evals_result,
                verbose_eval=False
            )

            val_pred = model.predict(dval)
            acc = (np.argmax(val_pred, axis=1) == y_val).mean()
            accuracies.append(acc)

            if hasattr(model, 'best_iteration'):
                best_iterations.append(model.best_iteration)

            del model, dtrain, dval
            gc.collect()

            trial.report(np.mean(accuracies), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best_iterations:
            trial.set_user_attr('avg_best_iteration', float(np.mean(best_iterations)))

        return float(np.mean(accuracies))
    
    except Exception as e:
        logger.error(f"Trial {trial.number} auf GPU {gpu_id} (device {device_string}) fehlgeschlagen: {e}", exc_info=True)
        return 0.0 # Harter Fehler, aber Optuna soll weitermachen

    finally:
        # WICHTIG: GPU-ID immer zurückgeben
        gpu_queue.put(gpu_id)


# --- KORRIGIERTE Final Model-Funktion ---
def train_final_model(best_params, best_trial, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    dtrain = xgb.DMatrix(X_global, label=y_global)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 19,
        'tree_method': 'hist',
        'device': 'cuda:0',    # <-- KORREKTUR: 'cuda:0'
        # 'gpu_id': 0,         # <-- ENTFERNT
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 42,
        'verbosity': 1,
    }
    params.update(best_params)
    
    # Logik von 100% der CV-Runden (unverändert, ist gut so)
    if 'avg_best_iteration' in best_trial.user_attrs:
        num_boost_round = int(best_trial.user_attrs['avg_best_iteration'])
    else:
        num_boost_round = 800  # Fallback
    
    logger.info(f"Training final model with {num_boost_round} rounds on device 'cuda:0'")
    
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=50
    )
    
    train_time = time.time() - start_time
    
    train_pred = model.predict(dtrain)
    train_accuracy = np.mean(np.argmax(train_pred, axis=1) == y_global)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_coldstart_acc{train_accuracy:.4f}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pkl_path = output_path / f"{model_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'params': params,
            'train_accuracy': train_accuracy,
            'train_time': train_time,
            'num_boost_round': num_boost_round,
        }, f)
    
    json_path = output_path / f"{model_name}.json"
    model.save_model(str(json_path))
    
    logger.info(f"Model saved to: {pkl_path}")
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    
    return model, train_accuracy


def main():
    DUCKLAKE_PATH = "/home/john/rlib_gfootball/cold_start/ducklake/replay_lake.duckdb"
    OUTPUT_DIR = Path("/home/john/rlib_gfootball/cold_start/xgboost_optuna_output")
    STACK_FRAMES = 4
    N_TRIALS = 500
    N_JOBS = 8
    N_GPUS = 8 # Sicherstellen, dass dies zur Anzahl der GPUs passt
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    load_data(DUCKLAKE_PATH, STACK_FRAMES)
    logger.info(f"Python script sees CUDA_VISIBLE_DEVICES as: [{os.environ.get('CUDA_VISIBLE_DEVICES')}]")

    db_path = OUTPUT_DIR / "optuna_study.db"
    storage_name = f"sqlite:///{db_path.resolve()}"
    logger.info(f"Optuna-Studie wird in Datenbank gespeichert: {storage_name}")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        storage=storage_name,
        load_if_exists=True,
        study_name='xgboost_coldstart_optimization'
    )
    
    # --- GPU-QUEUE LOGIK (Unverändert) ---
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    
    for i in range(N_GPUS):
        gpu_queue.put(i)
        
    objective_with_queue = functools.partial(objective, gpu_queue=gpu_queue)
    # --- ENDE GPU-QUEUE LOGIK ---

    logger.info(f"Starte Optimierung mit {N_JOBS} Jobs auf {N_GPUS} GPUs...")

    study.optimize(
        objective_with_queue,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True
    )
    
    best_trial = study.best_trial
    
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best CV accuracy: {best_trial.value:.4f}")
    logger.info("Best parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    study_path = OUTPUT_DIR / "optuna_study.pkl"
    joblib.dump(study, study_path)
    logger.info(f"Optuna study (PKL) saved to: {study_path}")
    
    final_model, train_acc = train_final_model(
        best_trial.params,
        best_trial,
        OUTPUT_DIR
    )
    
    logger.info(f"\nFinal model training accuracy: {train_acc:.4f}")
    logger.info("Training complete!")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)