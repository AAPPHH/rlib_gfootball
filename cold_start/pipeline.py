"""
Step 1: Create DuckDB Data Lake from Google Football JSON Replay Files
Processes raw JSON files and builds an efficient DuckDB database
Compatible with simple115v2 feature extraction (115 features)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DuckDBLakeBuilder:
    """Build DuckDB lake directly from Google Football JSON replay files"""
    
    def __init__(self, json_replay_dir: str, output_dir: str = "./ducklake", stack_frames: int = 4):
        self.json_replay_dir = Path(json_replay_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stack_frames = stack_frames
        
        # Database path
        self.db_path = self.output_dir / "replay_lake.duckdb"
        
        # Find all JSON files
        self.json_files = sorted(list(self.json_replay_dir.glob("*.json")))
        
        if not self.json_files:
            raise FileNotFoundError(f"No JSON files found in {self.json_replay_dir}")
        
        logger.info("="*70)
        logger.info("DuckDB Lake Builder - Google Football JSON Processor")
        logger.info("="*70)
        logger.info(f"Input: {json_replay_dir}")
        logger.info(f"Output: {self.db_path}")
        logger.info(f"Stack frames: {stack_frames}")
        logger.info(f"Found {len(self.json_files)} JSON replay files\n")
    
    def _parse_replay(self, filepath: Path) -> Dict:
        """Parse JSON replay file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Parse error {filepath.name}: {e}")
            return None
    
    def _one_hot_game_mode(self, mode: int) -> List[float]:
        """One-hot encode game mode (7 values)"""
        one_hot = [0.0] * 7
        if 0 <= mode < 7:
            one_hot[mode] = 1.0
        return one_hot
    
    def _one_hot_ball_owned(self, owned_team: int) -> List[float]:
        """One-hot encode ball ownership (3 values: -1, 0, 1)"""
        if owned_team == -1:
            return [1.0, 0.0, 0.0]
        elif owned_team == 0:
            return [0.0, 1.0, 0.0]
        else:  # 1
            return [0.0, 0.0, 1.0]
    
    def _one_hot_active(self, active_player: int) -> List[float]:
        """One-hot encode active player (11 values)"""
        one_hot = [0.0] * 11
        if 0 <= active_player < 11:
            one_hot[active_player] = 1.0
        return one_hot
    
    def _extract_simple115_features(self, replay_data: Dict, replay_id: str) -> Tuple[List, List, List]:
        """Extract simple115v2 compatible features from JSON replay"""
        
        observations = []
        actions_list = []
        rewards_list = []
        
        try:
            steps = replay_data.get("steps", [])
            
            for step_idx, step in enumerate(steps):
                if not isinstance(step, list) or len(step) == 0:
                    continue
                
                agent_step = step[0]
                obs_data = agent_step.get("observation", {})
                players_raw = obs_data.get("players_raw", [])
                
                if not players_raw:
                    continue
                
                player_obs = players_raw[0]
                
                # === SIMPLE115V2 FEATURES ===
                
                # Ball Position (3)
                ball = player_obs.get("ball", [0, 0, 0])
                ball_features = [
                    float(ball[0]) if len(ball) > 0 else 0.0,
                    float(ball[1]) if len(ball) > 1 else 0.0,
                    float(ball[2]) if len(ball) > 2 else 0.0,
                ]
                
                # Ball Direction (3)
                ball_dir = player_obs.get("ball_direction", [0, 0, 0])
                ball_dir_features = [
                    float(ball_dir[0]) if len(ball_dir) > 0 else 0.0,
                    float(ball_dir[1]) if len(ball_dir) > 1 else 0.0,
                    float(ball_dir[2]) if len(ball_dir) > 2 else 0.0,
                ]
                
                # Left Team Position (22)
                left_team = player_obs.get("left_team", [])
                left_pos = []
                for i in range(11):
                    if i < len(left_team) and isinstance(left_team[i], (list, tuple)):
                        left_pos.append(float(left_team[i][0]) if len(left_team[i]) > 0 else 0.0)
                        left_pos.append(float(left_team[i][1]) if len(left_team[i]) > 1 else 0.0)
                    else:
                        left_pos.extend([0.0, 0.0])
                
                # Left Team Direction (22)
                left_dir = player_obs.get("left_team_direction", [])
                left_vel = []
                for i in range(11):
                    if i < len(left_dir) and isinstance(left_dir[i], (list, tuple)):
                        left_vel.append(float(left_dir[i][0]) if len(left_dir[i]) > 0 else 0.0)
                        left_vel.append(float(left_dir[i][1]) if len(left_dir[i]) > 1 else 0.0)
                    else:
                        left_vel.extend([0.0, 0.0])
                
                # Right Team Position (22)
                right_team = player_obs.get("right_team", [])
                right_pos = []
                for i in range(11):
                    if i < len(right_team) and isinstance(right_team[i], (list, tuple)):
                        right_pos.append(float(right_team[i][0]) if len(right_team[i]) > 0 else 0.0)
                        right_pos.append(float(right_team[i][1]) if len(right_team[i]) > 1 else 0.0)
                    else:
                        right_pos.extend([0.0, 0.0])
                
                # Right Team Direction (22)
                right_dir = player_obs.get("right_team_direction", [])
                right_vel = []
                for i in range(11):
                    if i < len(right_dir) and isinstance(right_dir[i], (list, tuple)):
                        right_vel.append(float(right_dir[i][0]) if len(right_dir[i]) > 0 else 0.0)
                        right_vel.append(float(right_dir[i][1]) if len(right_dir[i]) > 1 else 0.0)
                    else:
                        right_vel.extend([0.0, 0.0])
                
                # Game Mode (7) - One-hot
                game_mode = int(player_obs.get("game_mode", 0))
                game_mode_features = self._one_hot_game_mode(game_mode)
                
                # Ball Owned Team (3) - One-hot
                ball_owned_team = int(player_obs.get("ball_owned_team", -1))
                ball_owned_features = self._one_hot_ball_owned(ball_owned_team)
                
                # Active Player (11) - One-hot
                active_player = int(player_obs.get("active", -1))
                active_features = self._one_hot_active(active_player)
                
                # Combine all features (115 total)
                all_features = (
                    ball_features +          # 3
                    ball_dir_features +      # 3
                    left_pos +               # 22
                    left_vel +               # 22
                    right_pos +              # 22
                    right_vel +              # 22
                    game_mode_features +     # 7
                    ball_owned_features +    # 3
                    active_features          # 11
                )  # Total: 115
                
                # Create observation record
                obs_record = {
                    "replay_id": replay_id,
                    "step": step_idx
                }
                for i, val in enumerate(all_features):
                    obs_record[f"feat_{i}"] = val
                
                observations.append(obs_record)
                
                # Action
                action = agent_step.get("action", [0])
                if isinstance(action, list) and len(action) > 0:
                    action_val = int(action[0])
                else:
                    action_val = 0
                    
                actions_list.append({
                    "replay_id": replay_id,
                    "step": step_idx,
                    "action": action_val
                })
                
                # Reward
                reward = agent_step.get("reward", 0)
                rewards_list.append({
                    "replay_id": replay_id,
                    "step": step_idx,
                    "reward": float(reward) if reward is not None else 0.0
                })
        
        except Exception as e:
            logger.error(f"Extract error for replay {replay_id}: {e}")
        
        return (observations, actions_list, rewards_list)

    def build_lake(self):
            if self.db_path.exists():
                logger.info(f"Removing existing database: {self.db_path}")
                self.db_path.unlink()
            
            all_obs = []
            all_act = []
            all_rew = []
            
            logger.info(f"Parsing {len(self.json_files)} JSON files...")
            for json_file in tqdm(self.json_files, desc="Processing JSON"):
                replay_id = json_file.stem
                replay_data = self._parse_replay(json_file)
                if replay_data:
                    obs, act, rew = self._extract_simple115_features(replay_data, replay_id)
                    all_obs.extend(obs)
                    all_act.extend(act)
                    all_rew.extend(rew)
            
            if not all_obs:
                logger.error("No data extracted. Aborting.")
                return

            logger.info("Converting to DataFrames...")
            obs_df = pd.DataFrame(all_obs)
            act_df = pd.DataFrame(all_act)
            rew_df = pd.DataFrame(all_rew)

            conn = duckdb.connect(str(self.db_path))
            
            conn.execute("SET threads TO 24")
            conn.execute("SET memory_limit = '32GB'")
            
            try:
                logger.info("Creating tables and importing data...")
                start_time = time.time()

                logger.info("\n📥 Importing observations...")
                conn.execute("""
                    CREATE TABLE observations AS 
                    SELECT *, row_number() OVER (ORDER BY replay_id, step) - 1 as global_idx 
                    FROM obs_df
                """)
                obs_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
                logger.info(f"✓ Imported {obs_count:,} observations")
                
                logger.info("\n📥 Importing actions...")
                conn.execute("""
                    CREATE TABLE actions AS 
                    SELECT *, row_number() OVER (ORDER BY replay_id, step) - 1 as global_idx 
                    FROM act_df
                """)
                act_count = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
                logger.info(f"✓ Imported {act_count:,} actions")
                
                logger.info("\n📥 Importing rewards...")
                conn.execute("""
                    CREATE TABLE rewards AS 
                    SELECT *, row_number() OVER (ORDER BY replay_id, step) - 1 as global_idx 
                    FROM rew_df
                """)
                rew_count = conn.execute("SELECT COUNT(*) FROM rewards").fetchone()[0]
                logger.info(f"✓ Imported {rew_count:,} rewards")
                
                if self.stack_frames > 1:
                    logger.info(f"\n🔄 Creating {self.stack_frames}x stacked features view...")
                    
                    sample = conn.execute("SELECT * FROM observations LIMIT 1").df()
                    feature_cols = [c for c in sample.columns if c.startswith('feat_')]
                    n_base_features = len(feature_cols)
                    
                    if n_base_features == 0:
                        logger.warning("No 'feat_' columns found in observations, skipping stacking.")
                    else:
                        lag_clauses = []
                        for offset in range(self.stack_frames):
                            for col in feature_cols:
                                if offset == 0:
                                    lag_clauses.append(f"{col}")
                                else:
                                    lag_clause = (
                                        f"COALESCE("
                                        f"LAG({col}, {offset}) OVER (PARTITION BY replay_id ORDER BY step), "
                                        f"{col}"
                                        f") AS {col}_lag{offset}"
                                    )
                                    lag_clauses.append(lag_clause)
                        
                        conn.execute(f"""
                            CREATE VIEW observations_stacked AS
                            SELECT 
                                global_idx,
                                replay_id,
                                step,
                                {', '.join(lag_clauses)}
                            FROM observations
                        """)
                        
                        logger.info(f"✓ Created stacked view with {n_base_features * self.stack_frames} features")
                
                logger.info("\n🔧 Creating indices...")
                conn.execute("CREATE INDEX idx_obs_global ON observations(global_idx)")
                conn.execute("CREATE INDEX idx_obs_replay ON observations(replay_id, step)")
                conn.execute("CREATE INDEX idx_act_global ON actions(global_idx)")
                conn.execute("CREATE INDEX idx_act_replay ON actions(replay_id, step)")
                conn.execute("CREATE INDEX idx_rew_global ON rewards(global_idx)")
                conn.execute("CREATE INDEX idx_rew_replay ON rewards(replay_id, step)")
                
                logger.info("\n📊 Computing statistics...")
                stats = conn.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM observations) as n_observations,
                        (SELECT COUNT(*) FROM actions) as n_actions,
                        (SELECT COUNT(*) FROM rewards) as n_rewards,
                        (SELECT COUNT(DISTINCT replay_id) FROM observations) as n_replays,
                        (SELECT MIN(step) FROM observations) as min_step,
                        (SELECT MAX(step) FROM observations) as max_step,
                        (SELECT AVG(step) FROM observations) as avg_step
                """).df()
                
                conn.execute(f"""
                    CREATE TABLE metadata AS 
                    SELECT 
                        '{self.db_path.name}' as lake_name,
                        CURRENT_TIMESTAMP as created_at,
                        {self.stack_frames} as stack_frames,
                        {stats['n_observations'].iloc[0]} as total_observations,
                        {stats['n_actions'].iloc[0]} as total_actions,
                        {stats['n_rewards'].iloc[0]} as total_rewards,
                        {stats['n_replays'].iloc[0]} as total_replays
                """)
                
                elapsed = time.time() - start_time
                
                logger.info("\n" + "="*70)
                logger.info("✅ DuckDB Lake Created Successfully!")
                logger.info("="*70)
                logger.info(f"Database: {self.db_path}")
                logger.info(f"Size: {self.db_path.stat().st_size / 1e9:.2f} GB")
                logger.info(f"Time: {elapsed:.1f} seconds")
                logger.info("\nStatistics:")
                logger.info(f"  Observations: {stats['n_observations'].iloc[0]:,}")
                logger.info(f"  Actions: {stats['n_actions'].iloc[0]:,}")
                logger.info(f"  Rewards: {stats['n_rewards'].iloc[0]:,}")
                logger.info(f"  Replays: {stats['n_replays'].iloc[0]:,}")
                
                logger.info("\nAvailable tables:")
                tables = conn.execute("SHOW TABLES").df()
                for table in tables['name']:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    logger.info(f"  - {table}: {count:,} rows")
                
                if self.stack_frames > 1:
                    logger.info(f"\nAvailable views:")
                    logger.info(f"  - observations_stacked: {self.stack_frames}x stacked features")
                
                queries_file = self.output_dir / "example_queries.sql"
                with open(queries_file, 'w') as f:
                    f.write(f"""-- Example DuckDB Queries for the Replay Lake

    -- Connect to database:
    -- conn = duckdb.connect('{self.db_path}')

    -- Get all data for training (stacked):
    SELECT o.*, a.action 
    FROM observations{'_stacked' if self.stack_frames > 1 else ''} o
    JOIN actions a ON o.global_idx = a.global_idx
    ORDER BY o.global_idx;

    -- Get data from specific replay:
    SELECT * FROM observations 
    WHERE replay_id = (SELECT replay_id FROM observations LIMIT 1)
    ORDER BY step;

    -- Get random sample:
    SELECT * FROM observations 
    USING SAMPLE 10000;

    -- Get metadata:
    SELECT * FROM metadata;

    -- Count samples per replay:
    SELECT replay_id, COUNT(*) as n_samples 
    FROM observations 
    GROUP BY replay_id 
    ORDER BY n_samples DESC;
    """)
                
                logger.info(f"\n💡 Example queries saved to: {queries_file}")
                
            finally:
                conn.close()
            
            return self.db_path


def main():
    REPLAY_DIR = r"/home/john/rlib_gfootball/cold_start/archive" 
    OUTPUT_DIR = r"/home/john/rlib_gfootball/cold_start/ducklake"
    STACK_FRAMES = 4
    
    try:
        builder = DuckDBLakeBuilder(REPLAY_DIR, OUTPUT_DIR, STACK_FRAMES)
        db_path = builder.build_lake()
        
        if db_path:
            print("\n🔍 Testing database connection...")
            conn = duckdb.connect(str(db_path), read_only=True)
            
            result = conn.execute("SELECT COUNT(*) as total FROM observations").fetchone()
            print(f"✓ Database accessible: {result[0]:,} observations")
            
            conn.close()
            print("\n✅ DuckDB Lake ready for use!")
            print(f"📂 Location: {db_path}")
        else:
            print("❌ Database building failed. Check logs.")
            
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Please check if the REPLAY_DIR is correct and contains .json files.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()