import duckdb
import numpy as np
import xgboost as xgb
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import time
import pickle
from sklearn.model_selection import GroupKFold
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostMemorizationTuner:
    def __init__(self, ducklake_path: str, output_dir: str = "./xgboost_tuned", stack_frames: int = 4):
        self.ducklake_path = Path(ducklake_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stack_frames = stack_frames
        
        if not self.ducklake_path.exists():
            raise FileNotFoundError(f"DuckDB lake not found: {self.ducklake_path}")
        
        logger.info("="*70)
        logger.info("XGBoost Cold-Start Policy Tuner")
        logger.info("="*70)
        logger.info(f"Database: {ducklake_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Stack frames: {stack_frames}\n")
    
    def load_full_dataset(self):
        logger.info("Loading full dataset from DuckDB lake...")
        start = time.time()
        
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        conn.execute("SET memory_limit = '32GB'")
        
        try:
            if self.stack_frames > 1:
                obs_table = "observations_stacked"
            else:
                obs_table = "observations"
            
            query = f"""
                SELECT 
                    o.*,
                    a.action
                FROM {obs_table} o
                JOIN actions a ON o.global_idx = a.global_idx
                ORDER BY o.global_idx
            """
            
            data = conn.execute(query).df()
            
            groups = data["replay_id"].values
            X = data.drop(columns=["action", "replay_id", "global_idx", "step"], errors="ignore").values.astype(np.float32)
            y = data["action"].values.astype(np.int32)
            
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            elapsed = time.time() - start
            
            logger.info(f"✓ Loaded {len(X):,} samples in {elapsed:.1f}s")
            logger.info(f"  Shape: {X.shape}")
            logger.info(f"  Features: {X.shape[1]}")
            logger.info(f"  Classes: {len(np.unique(y))}")
            logger.info(f"  Replays: {len(np.unique(groups))}")
            logger.info(f"  Memory: {X.nbytes / 1e9:.2f} GB\n")
            
        finally:
            conn.close()
        
        return X, y, groups
    
    def objective(self, trial, X, y, groups):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1600, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 5),
            'max_bin': trial.suggest_int('max_bin', 256, 256),
        }
        
        params.update({
            'tree_method': 'hist',
            'device': 'cuda',
            'num_class': 19,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': 0
        })
        
        gkf = GroupKFold(n_splits=5)
        accuracies = []
        best_iterations = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                best_iter = model.best_iteration
                best_iterations.append(best_iter)
                logger.debug(f"Fold {fold+1}: Early stopped at iteration {best_iter}/{params['n_estimators']}")
            else:
                logger.debug(f"Fold {fold+1}: Used all {params['n_estimators']} iterations")
            
            acc = model.score(X_val, y_val)
            accuracies.append(acc)
            
            del model
            gc.collect()
            
            trial.report(np.mean(accuracies), fold)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if best_iterations:
            avg_best = np.mean(best_iterations)
            trial.set_user_attr('avg_best_iteration', avg_best)
            trial.set_user_attr('early_stopped', True)
        else:
            trial.set_user_attr('early_stopped', False)
        
        return np.mean(accuracies)
    
    def tune_hyperparameters(self, X, y, groups, n_trials=100):
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        logger.info("Using 5-fold GroupKFold (replay-separated) with early stopping\n")

        OUTPUT_DIR = r"C:\clones\rlib_gfootball\cold_start\xgboost_tuned"
        
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        db_path = output_path / "optuna_study.db"
        storage_name = f"sqlite:///{db_path.resolve()}"
        logger.info(f"Optuna-Studie wird in Datenbank gespeichert: {storage_name}")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
            storage=storage_name,
            load_if_exists=True
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X, y, groups),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        logger.info("\n" + "="*70)
        logger.info("Optimization Complete!")
        logger.info("="*70)
        logger.info(f"Best trial: #{study.best_trial.number}")
        logger.info(f"Best CV accuracy: {study.best_value:.6f}")
        
        if study.best_trial.user_attrs.get('early_stopped', False):
            avg_iter = study.best_trial.user_attrs.get('avg_best_iteration', 0)
            logger.info(f"Early stopping used: avg iterations = {avg_iter:.0f}")
        
        logger.info("\nBest parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        study_path = self.output_dir / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"\n✓ Study saved to: {study_path}")
        
        return study.best_params, study.best_trial
    
    def train_final_model(self, X, y, params, best_trial):
        logger.info("\n" + "="*70)
        logger.info("Training Final Model on FULL Dataset")
        logger.info("="*70)
        
        final_params = params.copy()
        final_params.update({
            'tree_method': 'hist',
            'device': 'cuda',
            'num_class': 19,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 0,
        })
        
        if best_trial.user_attrs.get('early_stopped', False):
            avg_iter = best_trial.user_attrs.get('avg_best_iteration', 1000)
            final_params['n_estimators'] = int(avg_iter * 1.1)
            logger.info(f"Using n_estimators={final_params['n_estimators']} (110% of CV best)")
        
        logger.info("\nFinal parameters:")
        for key, value in final_params.items():
            if key not in ['tree_method', 'device', 'num_class', 'objective', 
                          'eval_metric', 'random_state', 'verbosity', 'n_jobs']:
                logger.info(f"  {key}: {value}")
        
        model = xgb.XGBClassifier(**final_params)
        
        logger.info(f"\nTraining with {len(X):,} samples...")
        
        start_time = time.time()
        model.fit(X, y, verbose=True)
        train_time = time.time() - start_time
        
        train_accuracy = model.score(X, y)
        
        logger.info(f"\n✓ Training complete in {train_time/60:.1f} minutes")
        logger.info(f"✓ Training accuracy: {train_accuracy:.6f} ({train_accuracy*100:.2f}%)")
        
        return model, train_accuracy, train_time
    
    def save_model(self, model, params, accuracy, train_time):
        base_name = f"xgboost_coldstart_{self.stack_frames}x_acc{accuracy:.4f}"
        
        model_path = self.output_dir / f"{base_name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'params': params,
                'accuracy': accuracy,
                'train_time': train_time,
                'stack_frames': self.stack_frames,
                'ducklake_path': str(self.ducklake_path),
                'n_estimators_trained': model.n_estimators
            }, f)
        
        logger.info(f"\n✓ Model saved to: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / 1e6:.2f} MB")
        
        xgb_path = self.output_dir / f"{base_name}.json"
        model.save_model(str(xgb_path))
        logger.info(f"✓ XGBoost format saved to: {xgb_path}")
        
        return model_path
    
    def run_full_pipeline(self, n_trials=100):
        X, y, groups = self.load_full_dataset()
        
        best_params, best_trial = self.tune_hyperparameters(X, y, groups, n_trials)
        
        model, accuracy, train_time = self.train_final_model(X, y, best_params, best_trial)
        
        model_path = self.save_model(model, best_params, accuracy, train_time)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Final accuracy: {accuracy*100:.2f}%")
        logger.info(f"Training time: {train_time/60:.1f} minutes")
        logger.info(f"Model saved: {model_path}")
        
        return model_path, accuracy

def main():
    DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
    OUTPUT_DIR = r"C:\clones\rlib_gfootball\cold_start\xgboost_tuned"
    STACK_FRAMES = 4
    N_TRIALS = 50
    
    tuner = XGBoostMemorizationTuner(DUCKLAKE_PATH, OUTPUT_DIR, STACK_FRAMES)
    model_path, accuracy = tuner.run_full_pipeline(n_trials=N_TRIALS)
    
    print(f"\n🎯 Final accuracy: {accuracy*100:.2f}%")
    print(f"📂 Model location: {model_path}")
    print("\n💡 Model trained with replay-separated cross-validation")
    print("   Ready for cold-start deployment to new games!")


if __name__ == "__main__":
    main()