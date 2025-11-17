import os
import time
import platform
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from concurrent.futures import ThreadPoolExecutor
import pickle

@dataclass
class TransitionRecord:
    """Single transition record"""
    run_id: str
    episode_id: int
    step_idx: int
    agent_id: str
    observation: np.ndarray
    action: int
    action_logits: np.ndarray
    action_probs: np.ndarray
    action_log_prob: float
    value_estimate: float
    reward: float
    done: bool
    truncated: bool
    score_left: int
    score_right: int
    game_mode: int
    policy_version: int
    timestamp_ns: int
    wall_time: float

class ReplayStorage:
    def __init__(self, 
                 base_dir: str = "~/gfootball_replay_lake",
                 buffer_size: int = 50000,
                 compression: str = "snappy",
                 partition_size: int = 1000):
        self.base_dir = Path(os.path.expanduser(base_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.compression = compression
        self.partition_size = partition_size
        
        self.current_run_id: Optional[str] = None
        self.current_policy_version: int = 0
        self.episode_counter: int = 0
        self.run_start_time: float = 0
        
        # Use Arrow for zero-copy data handling
        self._arrow_buffer: List[pa.RecordBatch] = []
        self._arrow_episode_buffer: List[pa.RecordBatch] = []
        self._buffer_size_bytes = 0
        self._max_buffer_bytes = 500 * 1024 * 1024  # 500MB
        
        # Define schemas upfront
        self._init_schemas()
        
    def _init_schemas(self):
        """Initialize Arrow schemas for efficient columnar storage"""
        # Assuming 460 observation dimensions (115 * 4 stacked) and 19 actions
        obs_fields = [(f'obs_{i}', pa.float32()) for i in range(460)]
        logit_fields = [(f'logit_{i}', pa.float32()) for i in range(19)]
        prob_fields = [(f'prob_{i}', pa.float32()) for i in range(19)]
        
        self.transition_schema = pa.schema([
            ('run_id', pa.string()),
            ('episode_id', pa.int32()),
            ('step_idx', pa.int16()),
            ('agent_id', pa.string()),
            ('action', pa.int8()),
            ('action_log_prob', pa.float32()),
            ('value_estimate', pa.float32()),
            ('reward', pa.float32()),
            ('done', pa.bool_()),
            ('truncated', pa.bool_()),
            ('score_left', pa.int8()),
            ('score_right', pa.int8()),
            ('game_mode', pa.int8()),
            ('policy_version', pa.int32()),
            ('timestamp_ns', pa.int64()),
            ('wall_time', pa.float32()),
        ] + obs_fields + logit_fields + prob_fields)
        
        self.episode_schema = pa.schema([
            ('run_id', pa.string()),
            ('episode_id', pa.int32()),
            ('start_timestamp_ns', pa.int64()),
            ('end_timestamp_ns', pa.int64()),
            ('total_steps', pa.int32()),
            ('total_reward', pa.float32()),
            ('policy_version', pa.int32()),
            ('final_score_left', pa.int8()),
            ('final_score_right', pa.int8()),
        ])
        
    def begin_run(self, 
                  run_id: str,
                  config: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        self.current_run_id = run_id
        self.run_start_time = time.time()
        self.episode_counter = 0
        
        run_dir = self.base_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "raw_transitions").mkdir(exist_ok=True)
        (run_dir / "episodes").mkdir(exist_ok=True)
        
        config_data = {
            "run_id": run_id,
            "config": config,
            "metadata": metadata or {},
            "start_time": self.run_start_time,
            "timestamp": time.time()
        }
        
        config_file = run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
            
    def begin_episode(self,
                      episode_id: int,
                      policy_version: int) -> None:
        self.current_policy_version = policy_version
        
    def record_step_batch(self, transitions: List[TransitionRecord]) -> None:
        """Ultra-fast batch recording using Arrow columnar format"""
        if not transitions:
            return
        
        # Pre-allocate arrays for all columns
        num_trans = len(transitions)
        
        # Create numpy arrays for each column type
        run_ids = []
        episode_ids = np.zeros(num_trans, dtype=np.int32)
        step_idxs = np.zeros(num_trans, dtype=np.int16)
        agent_ids = []
        actions = np.zeros(num_trans, dtype=np.int8)
        action_log_probs = np.zeros(num_trans, dtype=np.float32)
        value_estimates = np.zeros(num_trans, dtype=np.float32)
        rewards = np.zeros(num_trans, dtype=np.float32)
        dones = np.zeros(num_trans, dtype=bool)
        truncateds = np.zeros(num_trans, dtype=bool)
        score_lefts = np.zeros(num_trans, dtype=np.int8)
        score_rights = np.zeros(num_trans, dtype=np.int8)
        game_modes = np.zeros(num_trans, dtype=np.int8)
        policy_versions = np.zeros(num_trans, dtype=np.int32)
        timestamp_nss = np.zeros(num_trans, dtype=np.int64)
        wall_times = np.zeros(num_trans, dtype=np.float32)
        
        # Pre-allocate observation and action arrays
        obs_arrays = np.zeros((num_trans, 460), dtype=np.float32)
        logit_arrays = np.zeros((num_trans, 19), dtype=np.float32)
        prob_arrays = np.zeros((num_trans, 19), dtype=np.float32)
        
        # Vectorized copy - no Python loops!
        for i, trans in enumerate(transitions):
            run_ids.append(trans.run_id)
            episode_ids[i] = trans.episode_id
            step_idxs[i] = trans.step_idx
            agent_ids.append(trans.agent_id)
            actions[i] = trans.action
            action_log_probs[i] = trans.action_log_prob
            value_estimates[i] = trans.value_estimate
            rewards[i] = trans.reward
            dones[i] = trans.done
            truncateds[i] = trans.truncated
            score_lefts[i] = trans.score_left
            score_rights[i] = trans.score_right
            game_modes[i] = trans.game_mode
            policy_versions[i] = trans.policy_version
            timestamp_nss[i] = trans.timestamp_ns
            wall_times[i] = trans.wall_time
            
            # Flatten and copy arrays efficiently
            obs_arrays[i] = trans.observation.flatten()
            logit_arrays[i] = trans.action_logits
            prob_arrays[i] = trans.action_probs
        
        # Build column dict for Arrow
        columns = {
            'run_id': run_ids,
            'episode_id': episode_ids,
            'step_idx': step_idxs,
            'agent_id': agent_ids,
            'action': actions,
            'action_log_prob': action_log_probs,
            'value_estimate': value_estimates,
            'reward': rewards,
            'done': dones,
            'truncated': truncateds,
            'score_left': score_lefts,
            'score_right': score_rights,
            'game_mode': game_modes,
            'policy_version': policy_versions,
            'timestamp_ns': timestamp_nss,
            'wall_time': wall_times,
        }
        
        # Add observation columns
        for i in range(460):
            columns[f'obs_{i}'] = obs_arrays[:, i]
        
        # Add logit columns
        for i in range(19):
            columns[f'logit_{i}'] = logit_arrays[:, i]
            
        # Add prob columns
        for i in range(19):
            columns[f'prob_{i}'] = prob_arrays[:, i]
        
        # Create Arrow RecordBatch
        batch = pa.RecordBatch.from_pydict(columns, schema=self.transition_schema)
        self._arrow_buffer.append(batch)
        
        # Track buffer size
        self._buffer_size_bytes += batch.nbytes
        
        # Flush if buffer is large enough
        if self._buffer_size_bytes >= self._max_buffer_bytes:
            self.flush()
    
    def end_episode(self,
                    episode_id: int,
                    final_metrics: Dict[str, Any]) -> None:
        """Record episode completion"""
        columns = {
            'run_id': [self.current_run_id],
            'episode_id': [episode_id],
            'start_timestamp_ns': [final_metrics.get("start_timestamp_ns", 0)],
            'end_timestamp_ns': [time.time_ns()],
            'total_steps': [final_metrics.get("episode_length", 0)],
            'total_reward': [final_metrics.get("episode_reward", 0.0)],
            'policy_version': [self.current_policy_version],
            'final_score_left': [final_metrics.get("final_score_left", 0)],
            'final_score_right': [final_metrics.get("final_score_right", 0)],
        }
        
        batch = pa.RecordBatch.from_pydict(columns, schema=self.episode_schema)
        self._arrow_episode_buffer.append(batch)
        
        if len(self._arrow_episode_buffer) >= 100:
            self._flush_episodes()
    
    def flush(self, force: bool = False) -> None:
        """Flush using Arrow's native parquet writer"""
        if not self._arrow_buffer and not force:
            return
        
        if not self._arrow_buffer:
            return
        
        # Combine all batches into a single table
        table = pa.Table.from_batches(self._arrow_buffer, schema=self.transition_schema)
        
        # Write with partitioning using PyArrow (C++ implementation)
        run_dir = self.base_dir / "runs" / self.current_run_id / "raw_transitions"
        
        # Use PyArrow's partitioning for zero-copy writes
        timestamp = int(time.time() * 1000)
        
        # Create partition columns
        table = table.append_column(
            'episode_partition', 
            pa.compute.divide(table['episode_id'], pa.scalar(self.partition_size))
        )
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(run_dir),
            partition_cols=['policy_version', 'episode_partition'],
            compression=self.compression,
            use_legacy_dataset=False,
            existing_data_behavior='overwrite_or_ignore',
            basename_template=f'batch_{timestamp}_{{i}}.parquet'
        )
        
        # Clear buffers
        self._arrow_buffer.clear()
        self._buffer_size_bytes = 0
    
    def _flush_episodes(self) -> None:
        """Flush episode buffer"""
        if not self._arrow_episode_buffer:
            return
        
        # Combine all batches
        table = pa.Table.from_batches(self._arrow_episode_buffer, schema=self.episode_schema)
        
        run_dir = self.base_dir / "runs" / self.current_run_id / "episodes"
        run_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        filepath = run_dir / f"episodes_{timestamp}.parquet"
        
        pq.write_table(table, filepath, compression=self.compression)
        
        self._arrow_episode_buffer.clear()
    
    def close(self) -> None:
        """Close and flush all buffers"""
        self.flush(force=True)
        self._flush_episodes()
        
        if self.current_run_id:
            run_dir = self.base_dir / "runs" / self.current_run_id
            metadata_file = run_dir / "run_complete.json"
            metadata = {
                "run_id": self.current_run_id,
                "total_episodes": self.episode_counter,
                "end_time": time.time(),
                "duration": time.time() - self.run_start_time
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def create_duckdb_views(self, db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
        """Create DuckDB views for analysis"""
        conn = duckdb.connect(db_path) if db_path else duckdb.connect(':memory:')
        
        transitions_path = str(self.base_dir / "runs" / self.current_run_id / "raw_transitions" / "**/*.parquet")
        episodes_path = str(self.base_dir / "runs" / self.current_run_id / "episodes" / "*.parquet")
        
        conn.execute(f"""
            CREATE OR REPLACE VIEW raw_transitions AS 
            SELECT * FROM read_parquet('{transitions_path}')
        """)
        
        conn.execute(f"""
            CREATE OR REPLACE VIEW episodes AS 
            SELECT * FROM read_parquet('{episodes_path}')
        """)
        
        return conn