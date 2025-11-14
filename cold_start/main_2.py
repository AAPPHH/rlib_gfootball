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
import pyarrow as pa
import pyarrow.parquet as pq
from gymnasium import spaces
from tqdm import tqdm

DUCKLAKE_PATH = r"/home/john/rlib_gfootball/cold_start/ducklake/replay_lake.duckdb"
XGBOOST_MODEL_PATH = r"/home/john/rlib_gfootball/cold_start/xgboost_optuna_output/xgboost_coldstart_acc0.6813_20251109_043825.pkl"
OUTPUT_DIR = Path("/home/john/rlib_gfootball/cold_start/mamba_distillation_training")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Cache directory for preprocessed sequences (Parquet)
CACHE_DIR = OUTPUT_DIR / "cached_sequences"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def calculate_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
        return correct.item()


class XGBoostTeacher:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            loaded_obj = pickle.load(f)

        if isinstance(loaded_obj, dict):
            if "model" in loaded_obj:
                self.model = loaded_obj["model"]
            elif "xgb_model" in loaded_obj:
                self.model = loaded_obj["xgb_model"]
            elif "classifier" in loaded_obj:
                self.model = loaded_obj["classifier"]
            else:
                for key, value in loaded_obj.items():
                    if hasattr(value, "predict") or hasattr(value, "predict_proba"):
                        self.model = value
                        break
                else:
                    raise ValueError(
                        f"Could not find model in dictionary. Keys: {list(loaded_obj.keys())}"
                    )
        else:
            self.model = loaded_obj

        if hasattr(self.model, "predict_proba"):
            self.predict_fn = self.model.predict_proba
        elif hasattr(self.model, "predict"):
            try:
                import xgboost as xgb

                if isinstance(self.model, xgb.XGBClassifier):
                    self.predict_fn = self.model.predict_proba
                else:
                    self.predict_fn = lambda x: self.model.predict(xgb.DMatrix(x))
            except ImportError:
                self.predict_fn = self.model.predict
        else:
            raise ValueError(
                f"Model has no predict or predict_proba method: {type(self.model)}"
            )

    @torch.no_grad()
    def get_soft_targets_batch(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) > 2:
            orig_shape = features.shape
            features = features.reshape(orig_shape[0], -1)

        try:
            probs = self.predict_fn(features)

            if len(probs.shape) == 1:
                batch_size = features.shape[0]
                probs_2d = np.zeros((batch_size, 19), dtype=np.float32)
                probs_2d[np.arange(batch_size), probs.astype(int)] = 1.0
                probs = probs_2d

        except Exception as e:
            batch_size = features.shape[0]
            probs = np.ones((batch_size, 19), dtype=np.float32) / 19.0

        probs = probs.astype(np.float32)
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

        return probs


def precompute_teacher_predictions_duckdb(
    teacher, ducklake_path: str, stack_frames: int = 4
):
    """Precompute teacher predictions and store in DuckDB"""
    conn = duckdb.connect(str(ducklake_path))
    conn.execute("SET threads TO 16")
    conn.execute("SET memory_limit = '16GB'")

    try:
        obs_table = "observations_stacked" if stack_frames > 1 else "observations"

        tables = conn.execute("SHOW TABLES").df()
        if "teacher_predictions" in tables["name"].values:
            user_input = input("Recompute teacher predictions? (y/n): ")
            if user_input.lower() != "y":
                print("Using existing teacher predictions")
                return
            else:
                print("Dropping existing table...")
                conn.execute("DROP TABLE teacher_predictions")

        total_count = conn.execute(f"SELECT COUNT(*) FROM {obs_table}").fetchone()[0]
        print(f"Total frames to process: {total_count:,}")

        sample = conn.execute(f"SELECT * FROM {obs_table} LIMIT 1").df()
        feature_cols = [c for c in sample.columns if c.startswith("feat_")]

        features_str = ", ".join(feature_cols)
        query = f"""
            SELECT global_idx, {features_str}
            FROM {obs_table}
            ORDER BY global_idx
        """

        arrow_result = conn.execute(query).fetch_arrow_table()

        global_indices = arrow_result["global_idx"].to_numpy()
        features = np.column_stack(
            [arrow_result[col].to_numpy() for col in feature_cols]
        ).astype(np.float32)

        print("Running XGBoost predictions...")
        PRED_BATCH_SIZE = 100000
        all_probs = []

        for i in tqdm(range(0, len(features), PRED_BATCH_SIZE), desc="XGBoost batches"):
            batch_features = features[i : i + PRED_BATCH_SIZE]
            batch_probs = teacher.get_soft_targets_batch(batch_features)
            all_probs.append(batch_probs)

        teacher_probs = np.vstack(all_probs)

        print("Creating DuckDB table...")
        conn.execute(
            """
            CREATE TABLE teacher_predictions (
                global_idx INTEGER PRIMARY KEY,
                prob_0 FLOAT, prob_1 FLOAT, prob_2 FLOAT, prob_3 FLOAT, prob_4 FLOAT,
                prob_5 FLOAT, prob_6 FLOAT, prob_7 FLOAT, prob_8 FLOAT, prob_9 FLOAT,
                prob_10 FLOAT, prob_11 FLOAT, prob_12 FLOAT, prob_13 FLOAT, prob_14 FLOAT,
                prob_15 FLOAT, prob_16 FLOAT, prob_17 FLOAT, prob_18 FLOAT
            )
        """
        )

        data_dict = {
            "global_idx": global_indices.astype(np.int32),
            **{f"prob_{i}": teacher_probs[:, i] for i in range(19)}
        }

        conn.register("temp_predictions", data_dict)
        conn.execute("INSERT INTO teacher_predictions SELECT * FROM temp_predictions")
        conn.unregister("temp_predictions")

        print("Creating index...")
        conn.execute("CREATE INDEX idx_teacher_global ON teacher_predictions(global_idx)")
        print("✓ Teacher predictions stored in DuckDB")

    finally:
        conn.close()


def preprocess_and_cache_sequences(
    ducklake_path: str,
    cache_dir: Path,
    seq_len: int = 32,
    stack_frames: int = 4,
    num_sequences: int = 20000,
    train_split: float = 0.8
):
    """
    Extract all sequences from DuckDB once and cache them in Parquet format
    OPTIMIZED: Bulk load all data at once instead of per-sequence queries
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING SEQUENCES INTO PARQUET CACHE (OPTIMIZED)")
    print(f"{'='*60}")
    
    conn = duckdb.connect(str(ducklake_path), read_only=True)
    conn.execute("SET threads TO 16")
    conn.execute("SET memory_limit = '24GB'")

    try:
        obs_table = "observations_stacked" if stack_frames > 1 else "observations"

        # Get feature columns
        sample = conn.execute(f"SELECT * FROM {obs_table} LIMIT 1").df()
        feature_cols = [c for c in sample.columns if c.startswith("feat_")]
        n_features = len(feature_cols)
        features_str = ", ".join(feature_cols)

        # Split replays into train/val
        replay_query = f"SELECT DISTINCT replay_id FROM {obs_table} ORDER BY replay_id"
        replay_ids = conn.execute(replay_query).df()["replay_id"].values
        
        n_train_replays = int(len(replay_ids) * train_split)
        train_replays = replay_ids[:n_train_replays]
        val_replays = replay_ids[n_train_replays:]

        print(f"Total replays: {len(replay_ids)}, Train: {len(train_replays)}, Val: {len(val_replays)}")

        # Process train and val separately
        for split_name, split_replays in [('train', train_replays), ('val', val_replays)]:
            n_seqs = int(num_sequences * train_split) if split_name == 'train' else int(num_sequences * (1 - train_split))
            
            print(f"\nProcessing {split_name} split ({len(split_replays)} replays, {n_seqs} sequences)...")
            
            replays_str = ",".join(map(str, split_replays))

            # OPTIMIZATION 1: Load ALL data for these replays at once into Arrow format
            print(f"Bulk loading all data for {split_name} replays...")
            bulk_query = f"""
                SELECT 
                    o.replay_id,
                    o.global_idx,
                    o.step,
                    {features_str},
                    a.action,
                    t.prob_0, t.prob_1, t.prob_2, t.prob_3, t.prob_4,
                    t.prob_5, t.prob_6, t.prob_7, t.prob_8, t.prob_9,
                    t.prob_10, t.prob_11, t.prob_12, t.prob_13, t.prob_14,
                    t.prob_15, t.prob_16, t.prob_17, t.prob_18
                FROM {obs_table} o
                JOIN actions a ON o.global_idx = a.global_idx
                LEFT JOIN teacher_predictions t ON o.global_idx = t.global_idx
                WHERE o.replay_id IN ({replays_str})
                ORDER BY o.replay_id, o.step
            """
            
            # Use Arrow for efficient memory handling
            arrow_table = conn.execute(bulk_query).fetch_arrow_table()
            
            # Convert to pandas for easier manipulation (but keep memory efficient)
            df_all = arrow_table.to_pandas()
            
            print(f"Loaded {len(df_all):,} frames for {split_name}")
            
            # OPTIMIZATION 2: Vectorized sequence extraction
            print(f"Extracting sequences (vectorized)...")
            
            # Group by replay_id for efficient sequence extraction
            replay_groups = df_all.groupby('replay_id')
            
            all_sequences = []
            sequences_collected = 0
            
            # Probability columns
            prob_cols = [f"prob_{i}" for i in range(19)]
            
            for replay_id, replay_df in replay_groups:
                if sequences_collected >= n_seqs:
                    break
                    
                replay_len = len(replay_df)
                if replay_len < seq_len:
                    continue
                
                # Calculate how many sequences we can extract from this replay
                max_seqs_from_replay = replay_len - seq_len + 1
                # Sample some start positions
                n_seqs_from_replay = min(
                    max(1, n_seqs // len(split_replays)),  # Target per replay
                    max_seqs_from_replay,
                    n_seqs - sequences_collected  # Don't exceed total needed
                )
                
                if n_seqs_from_replay > 0:
                    # Random sample start positions
                    if max_seqs_from_replay > n_seqs_from_replay:
                        start_positions = np.random.choice(
                            max_seqs_from_replay, 
                            n_seqs_from_replay, 
                            replace=False
                        )
                    else:
                        start_positions = np.arange(max_seqs_from_replay)
                    
                    # Extract sequences vectorized
                    for start_pos in start_positions:
                        end_pos = start_pos + seq_len
                        seq_df = replay_df.iloc[start_pos:end_pos]
                        
                        # Extract data
                        features = seq_df[feature_cols].values.astype(np.float32)
                        actions = seq_df["action"].values.astype(np.int64)
                        teacher_probs = seq_df[prob_cols].values.astype(np.float32)
                        
                        # Clean data
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        teacher_probs = np.nan_to_num(teacher_probs, nan=1.0/19.0, posinf=0.0, neginf=0.0)
                        
                        all_sequences.append({
                            'features': features.flatten(),
                            'actions': actions,
                            'teacher_probs': teacher_probs.flatten()
                        })
                        
                        sequences_collected += 1
                        if sequences_collected >= n_seqs:
                            break
            
            print(f"Collected {len(all_sequences)} sequences")
            
            # OPTIMIZATION 3: Efficient Parquet writing with Arrow
            print(f"Writing to Parquet (optimized)...")
            
            # Stack all sequences (This is good and fast)
            features_array = np.stack([s['features'] for s in all_sequences])
            actions_array = np.stack([s['actions'] for s in all_sequences])
            teacher_probs_array = np.stack([s['teacher_probs'] for s in all_sequences])
            
            # --- START FIX ---
            # Create Arrow arrays by first flattening the 2D arrays 
            # and then using FixedSizeListArray.from_arrays

            # 1. Create 1D "values" arrays from the flattened 2D numpy arrays
            features_values_flat = pa.array(features_array.flatten(), type=pa.float32())
            actions_values_flat = pa.array(actions_array.flatten(), type=pa.int64())
            teacher_probs_values_flat = pa.array(teacher_probs_array.flatten(), type=pa.float32())

            # 2. Create the FixedSizeListArray from the 1D values and the known list size
            features_arrow = pa.FixedSizeListArray.from_arrays(
                features_values_flat, list_size=seq_len * n_features
            )
            actions_arrow = pa.FixedSizeListArray.from_arrays(
                actions_values_flat, list_size=seq_len
            )
            teacher_probs_arrow = pa.FixedSizeListArray.from_arrays(
                teacher_probs_values_flat, list_size=seq_len * 19
            )
            
            # Create schema with fixed-size lists for better performance
            schema = pa.schema([
                ('features', pa.list_(pa.float32(), seq_len * n_features)),
                ('actions', pa.list_(pa.int64(), seq_len)),
                ('teacher_probs', pa.list_(pa.float32(), seq_len * 19))
            ])
            
            # Create table with explicit schema
            table = pa.table(
                [features_arrow, actions_arrow, teacher_probs_arrow],
                schema=schema # Use schema= instead of names=
            )
            # --- END FIX ---

            # Write with optimized settings
            output_file = cache_dir / f"{split_name}.parquet"
            pq.write_table(
                table, 
                output_file,
                compression='snappy',
                use_dictionary=False,
                write_statistics=True,
                row_group_size=1000  # Optimize row group size for batched reading
            )
            
            print(f"✓ Wrote {split_name} to {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Clean up memory
            del df_all, arrow_table, all_sequences, features_array, actions_array, teacher_probs_array
            del features_arrow, actions_arrow, teacher_probs_arrow, table
            gc.collect()

    finally:
        conn.close()

    print(f"\n✓ Sequences cached to {cache_dir}")
    print(f"{'='*60}\n")

class CachedSequenceDataset(Dataset):
    """
    Fast dataset that loads preprocessed sequences from Parquet cache
    No more DuckDB queries during training!
    """
    def __init__(self, cache_dir: Path, split: str = 'train', seq_len: int = 32, n_features: int = 115):
        self.cache_dir = cache_dir
        self.split = split
        self.seq_len = seq_len
        self.n_features = n_features
        
        parquet_file = cache_dir / f"{split}.parquet"
        
        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
        
        # Load entire Parquet file into memory (fast!)
        print(f"Loading {split} dataset from {parquet_file}...")
        table = pq.read_table(parquet_file)
        
        # Convert to numpy arrays (kept in memory for fast access)
        features_flat = np.array(table['features'].to_pylist(), dtype=np.float32)
        actions_flat = np.array(table['actions'].to_pylist(), dtype=np.int64)
        teacher_probs_flat = np.array(table['teacher_probs'].to_pylist(), dtype=np.float32)
        
        # Reshape to proper dimensions
        self.features = features_flat.reshape(-1, seq_len, n_features)
        self.actions = actions_flat.reshape(-1, seq_len)
        self.teacher_probs = teacher_probs_flat.reshape(-1, seq_len, 19)
        
        self.length = self.features.shape[0]
        
        print(f"✓ Loaded {split} dataset: {self.length:,} sequences ({self.features.nbytes / 1024 / 1024:.1f} MB features)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Direct numpy array access - extremely fast!
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.teacher_probs[idx])
        )


def create_mamba_model(config: dict, device: torch.device) -> nn.Module:
    from model_3 import GFootballMamba

    n_features = config["n_features"] * config["stack_frames"]
    obs_space = spaces.Box(-np.inf, np.inf, (n_features,), dtype=np.float32)
    action_space = spaces.Discrete(19)

    model_config = {
        "custom_model_config": {
            "d_model": config["d_model"],
            "mamba_state": config["mamba_state"],
            "num_mamba_layers": config["num_mamba_layers"],
            "prev_action_emb": config["prev_action_emb"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "mlp_hidden_dims": config["mlp_hidden_dims"],
            "mlp_activation": config["mlp_activation"],
            "head_hidden_dims": config["head_hidden_dims"],
            "head_activation": config["head_activation"],
            "use_noisy": config["use_noisy"],
            "use_distributional": config["use_distributional"],
            "v_min": config["v_min"],
            "v_max": config["v_max"],
            "num_atoms": config["num_atoms"],
        }
    }

    model = GFootballMamba(obs_space, action_space, 19, model_config, "mamba_student")
    return model.to(device)


def train_with_distillation():
    config = {
        "stack_frames": 4,
        "n_features": 115,
        "seq_len": 32,
        "custom_model": "GFootballMamba",
        "max_seq_len": 256,
        "custom_model_config": {
            "d_model": 48,
            "mamba_state": 6,
            "num_mamba_layers": 6,
            "prev_action_emb": 8,
            "gradient_checkpointing": True,
            "mlp_hidden_dims": [256, 128],
            "mlp_activation": "silu",
            "head_hidden_dims": [128],
            "head_activation": "silu",
            "use_noisy": True,
            "use_distributional": True,
            "v_min": -10.0,
            "v_max": 10.0,
            "num_atoms": 51,
        },
    }

    SEQ_LEN = config["seq_len"]
    BATCH_SIZE = 512
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    NUM_SEQUENCES = 20000
    TEACHER_TEMP = 1.5
    STUDENT_TEMP = 1.5
    ALPHA_INITIAL = 0.9
    ALPHA_FINAL = 0.3
    ALPHA_DECAY_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    TOP_K_VALUES = [1, 3, 5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Precompute teacher predictions if needed
    teacher = XGBoostTeacher(XGBOOST_MODEL_PATH)
    precompute_teacher_predictions_duckdb(teacher, DUCKLAKE_PATH, config["stack_frames"])
    del teacher  # Free memory
    gc.collect()

    # Step 2: Preprocess sequences into cache if not exists
    train_cache = CACHE_DIR / "train.parquet"
    val_cache = CACHE_DIR / "val.parquet"
    
    if not train_cache.exists() or not val_cache.exists():
        print(f"\n📦 Cache files not found. Preprocessing sequences...")
        preprocess_and_cache_sequences(
            DUCKLAKE_PATH,
            CACHE_DIR,
            seq_len=SEQ_LEN,
            stack_frames=config["stack_frames"],
            num_sequences=NUM_SEQUENCES,
            train_split=0.8
        )
    else:
        print(f"\n✓ Using cached sequences from {CACHE_DIR}")

    # Step 3: Create datasets from cache
    train_dataset = CachedSequenceDataset(
        CACHE_DIR, 
        split='train',
        seq_len=SEQ_LEN,
        n_features=config["n_features"] * config["stack_frames"]
    )
    val_dataset = CachedSequenceDataset(
        CACHE_DIR,
        split='val',
        seq_len=SEQ_LEN,
        n_features=config["n_features"] * config["stack_frames"]
    )

    # Step 4: Create DataLoaders (much faster now!)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Step 5: Create model
    model_init_config = {
        "n_features": config["n_features"],
        "stack_frames": config["stack_frames"],
        **config["custom_model_config"],
    }

    model = create_mamba_model(model_init_config, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_top5 = 0
    epochs_no_improve = 0

    print("\n" + "=" * 60)
    print("STARTING SEQUENCE-BASED DISTILLATION TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):
        if epoch < ALPHA_DECAY_EPOCHS:
            progress = epoch / ALPHA_DECAY_EPOCHS
            alpha = ALPHA_INITIAL * (1 - progress) + ALPHA_FINAL * progress
        else:
            alpha = ALPHA_FINAL

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} - Alpha (teacher weight): {alpha:.3f}")
        print(f"{'='*60}")

        model.train()
        if hasattr(model, "reset_noise"):
            model.reset_noise()

        train_loss = 0
        train_topk_accs = {k: 0 for k in TOP_K_VALUES}
        train_batches = 0

        epoch_start = time.time()

        for batch_idx, (seq_features, seq_actions, teacher_probs) in enumerate(
            tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        ):
            batch_size = seq_features.size(0)
            seq_features = seq_features.to(device, non_blocking=True)
            seq_actions = seq_actions.to(device, non_blocking=True)
            teacher_probs = teacher_probs.to(device, non_blocking=True)

            # Initialize state at beginning of sequence
            state = [
                torch.zeros(batch_size, model.state_size, device=device)
                for _ in range(model.num_mamba_layers)
            ]

            batch_loss = 0
            batch_topk = {k: 0 for k in TOP_K_VALUES}

            for t in range(SEQ_LEN):
                features_t = seq_features[:, t, :]
                action_t = seq_actions[:, t]
                teacher_probs_t = teacher_probs[:, t, :]

                input_dict = {
                    "obs": features_t,
                    "obs_flat": features_t,
                    "prev_actions": seq_actions[:, t - 1]
                    if t > 0
                    else torch.zeros(batch_size, dtype=torch.long, device=device),
                }

                seq_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
                
                # FIX: Properly update state for next timestep
                logits, new_state = model(input_dict, state, seq_lens)
                state = new_state  # <-- THIS IS THE KEY FIX!

                hard_loss = F.cross_entropy(logits, action_t)

                log_probs_student = F.log_softmax(logits / STUDENT_TEMP, dim=-1)
                teacher_logits = torch.log(teacher_probs_t + 1e-8)
                probs_teacher = F.softmax(teacher_logits / TEACHER_TEMP, dim=-1)

                soft_loss = (
                    F.kl_div(
                        log_probs_student, probs_teacher, reduction="batchmean"
                    )
                    * (TEACHER_TEMP**2)
                )

                step_loss = alpha * soft_loss + (1 - alpha) * hard_loss
                batch_loss += step_loss

                for k in TOP_K_VALUES:
                    batch_topk[k] += calculate_topk_accuracy(logits, action_t, k)

            avg_batch_loss = batch_loss / SEQ_LEN

            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += avg_batch_loss.item()
            for k in TOP_K_VALUES:
                train_topk_accs[k] += batch_topk[k] / SEQ_LEN
            train_batches += 1

        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        val_loss = 0
        val_topk_accs = {k: 0 for k in TOP_K_VALUES}
        val_teacher_agreement = 0
        val_batches = 0

        with torch.no_grad():
            for seq_features, seq_actions, teacher_probs in tqdm(val_loader, desc="Validation"):
                batch_size = seq_features.size(0)
                seq_features = seq_features.to(device, non_blocking=True)
                seq_actions = seq_actions.to(device, non_blocking=True)
                teacher_probs = teacher_probs.to(device, non_blocking=True)

                # Initialize state at beginning of sequence
                state = [
                    torch.zeros(batch_size, model.state_size, device=device)
                    for _ in range(model.num_mamba_layers)
                ]

                batch_loss = 0
                batch_topk = {k: 0 for k in TOP_K_VALUES}
                batch_agreement = 0

                for t in range(SEQ_LEN):
                    features_t = seq_features[:, t, :]
                    action_t = seq_actions[:, t]
                    teacher_probs_t = teacher_probs[:, t, :]

                    teacher_preds = teacher_probs_t.argmax(dim=-1)

                    input_dict = {
                        "obs": features_t,
                        "obs_flat": features_t,
                        "prev_actions": seq_actions[:, t - 1]
                        if t > 0
                        else torch.zeros(batch_size, dtype=torch.long, device=device),
                    }

                    seq_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
                    
                    # FIX: Properly update state for next timestep
                    logits, new_state = model(input_dict, state, seq_lens)
                    state = new_state  # <-- THIS IS THE KEY FIX!

                    hard_loss = F.cross_entropy(logits, action_t)

                    log_probs_student = F.log_softmax(logits / STUDENT_TEMP, dim=-1)
                    teacher_logits = torch.log(teacher_probs_t + 1e-8)
                    probs_teacher = F.softmax(teacher_logits / TEACHER_TEMP, dim=-1)

                    soft_loss = (
                        F.kl_div(
                            log_probs_student, probs_teacher, reduction="batchmean"
                        )
                        * (TEACHER_TEMP**2)
                    )

                    step_loss = alpha * soft_loss + (1 - alpha) * hard_loss
                    batch_loss += step_loss

                    for k in TOP_K_VALUES:
                        batch_topk[k] += calculate_topk_accuracy(logits, action_t, k)

                    student_preds = logits.argmax(dim=-1)
                    batch_agreement += (
                        (student_preds == teacher_preds).float().mean().item()
                    )

                val_loss += batch_loss.item() / SEQ_LEN
                for k in TOP_K_VALUES:
                    val_topk_accs[k] += batch_topk[k] / SEQ_LEN
                val_teacher_agreement += batch_agreement / SEQ_LEN
                val_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_train_topk = {k: train_topk_accs[k] / train_batches for k in TOP_K_VALUES}
        avg_val_loss = val_loss / val_batches
        avg_val_topk = {k: val_topk_accs[k] / val_batches for k in TOP_K_VALUES}
        avg_teacher_agree = val_teacher_agreement / val_batches

        print(f"\n{'-'*40}")
        print(f"Epoch {epoch+1} Summary (Time: {epoch_time:.1f}s):")
        print(f"  Train - Loss: {avg_train_loss:.4f}")
        print(
            f"        Top-1: {avg_train_topk[1]:.4f}, Top-3: {avg_train_topk[3]:.4f}, Top-5: {avg_train_topk[5]:.4f}"
        )
        print(f"  Val   - Loss: {avg_val_loss:.4f}")
        print(
            f"        Top-1: {avg_val_topk[1]:.4f}, Top-3: {avg_val_topk[3]:.4f}, Top-5: {avg_val_topk[5]:.4f}"
        )
        print(f"  Teacher Agreement: {avg_teacher_agree:.4f}")

        if avg_val_topk[5] > best_val_top5:
            best_val_top5 = avg_val_topk[5]
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_top1": avg_val_topk[1],
                "val_top3": avg_val_topk[3],
                "val_top5": avg_val_topk[5],
                "teacher_agreement": avg_teacher_agree,
                "config": config,
                "alpha": alpha,
                "seq_len": SEQ_LEN,
            }
            torch.save(checkpoint, OUTPUT_DIR / "best_model.pth")
            print(f"  ✓ New best model saved! (Top-5: {best_val_top5:.4f})")
        else:
            epochs_no_improve += 1
            print(
                f"  → No improvement. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"
            )

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs.")
            break

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_topk": avg_val_topk,
                "config": config,
                "seq_len": SEQ_LEN,
            }
            torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pth")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Validation Top-5 Accuracy: {best_val_top5:.4f}")
    print(f"{'='*60}")

    return model, best_val_top5

if __name__ == "__main__":
    model, best_top5 = train_with_distillation()
    print(f"\n✅ Training complete! Best Top-5 accuracy: {best_top5:.4f}")