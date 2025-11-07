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
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc
import joblib

os.environ['NUMEXPR_MAX_THREADS'] = '256'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

X_train_global = None
X_val_global = None
y_train_global = None
y_val_global = None

def load_data(ducklake_path, stack_frames):
    global X_train_global, X_val_global, y_train_global, y_val_global
    
    conn = duckdb.connect(str(ducklake_path), read_only=True)
    conn.execute("SET threads TO 64")
    conn.execute("SET memory_limit = '400GB'")
    
    try:
        obs_table = "observations_stacked" if stack_frames > 1 else "observations"
        
        query = f"""
            SELECT 
                o.* EXCLUDE (global_idx, replay_id, step),
                a.action
            FROM {obs_table} o
            JOIN actions a ON o.global_idx = a.global_idx
            ORDER BY o.global_idx
        """
        
        data = conn.execute(query).df()
        
        X = data.drop('action', axis=1).values.astype(np.float32)
        y = data['action'].values.astype(np.int32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        del X, y
        gc.collect()
        
    finally:
        conn.close()

def objective(trial):
    gpu_id = trial.number % 8
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 19,
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 42,
        'verbosity': 0,
        
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'grow_policy': trial.suggest_categorical('grow_policy', ['lossguide']),
        'max_leaves': trial.suggest_int('max_leaves', 1024, 8192, log=True),

        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.7, 1.0),

        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 50.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),

        'max_bin': trial.suggest_categorical('max_bin', [256]),
        'single_precision_histogram': trial.suggest_categorical('single_precision_histogram', [1]),
        'sampling_method': trial.suggest_categorical('sampling_method', ['gradient_based']),
    }
    
    dtrain = xgb.DMatrix(X_train_global, label=y_train_global)
    dval = xgb.DMatrix(X_val_global, label=y_val_global)
    
    num_boost_round = 1000
    early_stopping_rounds = 5
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'eval-merror')
    
    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )
        
        val_pred = model.predict(dval)
        val_accuracy = np.mean(np.argmax(val_pred, axis=1) == y_val_global)
        
        train_pred = model.predict(dtrain)
        train_accuracy = np.mean(np.argmax(train_pred, axis=1) == y_train_global)
        
        trial.set_user_attr('train_accuracy', train_accuracy)
        trial.set_user_attr('gpu_id', gpu_id)
        trial.set_user_attr('best_iteration', model.best_iteration if hasattr(model, 'best_iteration') else None)
        
        return val_accuracy
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0

def train_final_model(best_params, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    dtrain = xgb.DMatrix(X_train_global, label=y_train_global)
    dval = xgb.DMatrix(X_val_global, label=y_val_global)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 19,
        'tree_method': 'hist',
        'device': 'cuda:0',
        'eval_metric': ['mlogloss', 'merror'],
        'seed': 42,
        'verbosity': 2,
    }
    params.update(best_params)
    params.update(best_params)
    
    num_boost_round = 4000
    early_stopping_rounds = 200
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50
    )
    
    train_time = time.time() - start_time
    
    val_pred = model.predict(dval)
    val_accuracy = np.mean(np.argmax(val_pred, axis=1) == y_val_global)
    
    train_pred = model.predict(dtrain)
    train_accuracy = np.mean(np.argmax(train_pred, axis=1) == y_train_global)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_optuna_final_val{val_accuracy:.4f}_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pkl_path = output_path / f"{model_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'params': params,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_time': train_time,
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None,
        }, f)
    
    json_path = output_path / f"{model_name}.json"
    model.save_model(str(json_path))
    
    logger.info(f"Model saved to: {pkl_path}")
    
    return model, train_accuracy, val_accuracy

def main():
    DUCKLAKE_PATH = "/home/john/rlib_gfootball/cold_start/ducklake/replay_lake.duckdb"
    OUTPUT_DIR = "/home/john/rlib_gfootball/cold_start/xgboost_optuna_output"
    STACK_FRAMES = 4
    N_TRIALS = 100
    N_JOBS = 8
    
    load_data(DUCKLAKE_PATH, STACK_FRAMES)
    logger.info(f"Python script sees CUDA_VISIBLE_DEVICES as: [{os.environ.get('CUDA_VISIBLE_DEVICES')}]")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20),
        study_name='xgboost_dgx_optimization'
    )
    
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True
    )
    
    best_trial = study.best_trial
    
    study_path = Path(OUTPUT_DIR) / "optuna_study.pkl"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, study_path)
    logger.info(f"Optuna study saved to: {study_path}")
    
    final_model, train_acc, val_acc = train_final_model(
        best_trial.params,
        OUTPUT_DIR
    )
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)