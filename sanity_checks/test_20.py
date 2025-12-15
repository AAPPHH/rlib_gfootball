import numpy as np
import gfootball.env as football_env
from dataclasses import dataclass

@dataclass
class TrainingStage:
    name: str
    env_name: str
    representation: str 
    left_players: int
    right_players: int
    target_reward: float
    max_timesteps: int
    description: str = ""

TRAINING_STAGES = [
    TrainingStage("stage_1_basic", "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 0.75, 10_000_000, "1 Spieler gegen Keeper"),
    TrainingStage("stage_2_1v1", "academy_3_vs_1_with_keeper", "simple115v2", 1, 1, 0.75, 20_000_000, "1v1 Spiel"),
    TrainingStage("stage_3_3v3", "11_vs_11_easy_stochastic", "simple115v2", 3, 3, 1.0, 50_000_000, "3v3 Kleinfeld"),
    TrainingStage("stage_4_5v5", "11_vs_11_stochastic", "simple115v2", 5, 5, 1.0, 100_000_000, "5v5 Mittelfeld"),
    TrainingStage("stage_5_11v11", "11_vs_11_stochastic", "simple115v2", 11, 11, 1.0, 500_000_000, "11v11 Vollspiel"),
]

def test_stage(stage: TrainingStage, stacked: bool = True):
    """Test eine Training Stage"""
    print("\n" + "="*80)
    print(f"Stage: {stage.name}")
    print(f"  Env: {stage.env_name}")
    print(f"  Players: {stage.left_players} vs {stage.right_players}")
    print(f"  Representation: {stage.representation}")
    print(f"  Stacked: {stacked}")
    print("="*80)
    
    try:
        env = football_env.create_environment(
            env_name=stage.env_name,
            representation=stage.representation,
            stacked=stacked,
            logdir='/tmp/gfootball_test',
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            number_of_left_players_agent_controls=stage.left_players,
            number_of_right_players_agent_controls=stage.right_players,
        )
        
        obs = env.reset()
        
        print(f"\n📊 Observation Info:")
        print(f"   Type: {type(obs)}")
        
        if isinstance(obs, (list, tuple)):
            print(f"   Multi-Agent: {len(obs)} agents")
            first_obs = obs[0]
            print(f"   Single Agent Obs Shape: {first_obs.shape}")
            print(f"   Single Agent Obs Dtype: {first_obs.dtype}")
        elif isinstance(obs, np.ndarray):
            print(f"   Shape: {obs.shape}")
            print(f"   Dtype: {obs.dtype}")
            first_obs = obs
        else:
            print(f"   Unexpected type: {type(obs)}")
            first_obs = obs
        
        if hasattr(first_obs, 'shape'):
            shape = first_obs.shape
            total_elements = np.prod(shape)
            
            print(f"   Total elements: {total_elements}")
            print(f"   Min value: {np.min(first_obs):.4f}")
            print(f"   Max value: {np.max(first_obs):.4f}")
            
            if len(shape) >= 3:
                print(f"\n🖼️  PIXEL DATA detected!")
                if len(shape) == 3:
                    H, W, C = shape
                    print(f"   Dimensions: Height={H}, Width={W}, Channels={C}")
                    if stacked:
                        frames = C // 4 if C % 4 == 0 else "unknown"
                        print(f"   Likely {frames} stacked frames (4 channels each)")
                else:
                    print(f"   Complex shape: {shape}")
                print(f"   ⚠️  CNN REQUIRED for this representation!")
            elif len(shape) == 2:
                print(f"\n📈 2D ARRAY detected!")
                print(f"   Dimensions: {shape[0]} x {shape[1]}")
                print(f"   Could be [Agents, Features] or [Frames, Features]")
            elif len(shape) == 1:
                print(f"\n📈 FLAT FEATURE VECTOR detected!")
                print(f"   Dimensions: {shape[0]} features")
                if stacked and stage.representation == "simple115v2":
                    frames = shape[0] // 115 if shape[0] % 115 == 0 else "unknown"
                    print(f"   Likely {frames} stacked frames (115 features each)")
                print(f"   ✅ Linear layers work fine!")
            
            # Show sample values
            if len(shape) == 1:
                print(f"\n🔍 First 20 values:")
                print(f"   {first_obs[:20]}")
            elif len(shape) == 2 and shape[0] <= 10:
                print(f"\n🔍 First agent, first 20 values:")
                print(f"   {first_obs[0, :20]}")
        
        # Test a step
        if isinstance(obs, (list, tuple)):
            # Multi-agent: provide action for each agent
            total_agents = stage.left_players + stage.right_players
            actions = [0] * total_agents
            obs, reward, done, info = env.step(actions)
        else:
            obs, reward, done, info = env.step(0)
        
        print(f"\n✅ After step:")
        if isinstance(obs, (list, tuple)):
            print(f"   Obs list length: {len(obs)}")
            print(f"   Single obs shape: {obs[0].shape}")
        else:
            print(f"   Obs shape: {obs.shape}")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_representation_comparison(stage: TrainingStage):
    """Vergleiche extracted vs simple115v2 für eine Stage"""
    print("\n" + "🔄"*40)
    print(f"REPRESENTATION COMPARISON for {stage.name}")
    print("🔄"*40)
    
    representations = ["extracted", "simple115v2"]
    results = {}
    
    for rep in representations:
        print(f"\n--- Testing {rep} ---")
        try:
            env = football_env.create_environment(
                env_name=stage.env_name,
                representation=rep,
                stacked=True,
                logdir='/tmp/gfootball_test',
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=False,
                number_of_left_players_agent_controls=stage.left_players,
                number_of_right_players_agent_controls=stage.right_players,
            )
            
            obs = env.reset()
            if isinstance(obs, (list, tuple)):
                obs = obs[0]
            
            results[rep] = {
                'shape': obs.shape,
                'dtype': obs.dtype,
                'total_elements': np.prod(obs.shape),
                'is_pixel': len(obs.shape) >= 3
            }
            
            print(f"✅ {rep}: shape={obs.shape}, elements={results[rep]['total_elements']}")
            env.close()
            
        except Exception as e:
            print(f"❌ {rep}: Error - {e}")
            results[rep] = None
    
    # Compare
    print(f"\n📊 Comparison:")
    for rep, data in results.items():
        if data:
            data_type = "PIXEL" if data['is_pixel'] else "FEATURES"
            print(f"  {rep:15s}: {str(data['shape']):20s} | {data['total_elements']:8d} elements | {data_type}")
    
    return results

def main():
    print("\n" + "⚽"*40)
    print("GOOGLE FOOTBALL - ALL TRAINING STAGES TEST")
    print("⚽"*40)
    
    # Test alle Stages mit extracted + stacked
    print("\n" + "="*80)
    print("PART 1: Testing all stages with 'extracted' + stacked=True")
    print("="*80)
    
    success_count = 0
    for i, stage in enumerate(TRAINING_STAGES):
        success = test_stage(stage, stacked=True)
        if success:
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Tested {len(TRAINING_STAGES)} stages, {success_count} successful")
    print(f"{'='*80}")
    
    # Test erste Stage mit verschiedenen Representations
    print("\n" + "="*80)
    print("PART 2: Representation comparison for first stage")
    print("="*80)
    
    test_representation_comparison(TRAINING_STAGES[0])
    
    # Final recommendation
    print("\n" + "🎯"*40)
    print("EMPFEHLUNG")
    print("🎯"*40)
    print("""
Basierend auf den Tests:

1. 'extracted' (euer aktuelles Setup):
   ❌ Liefert PIXEL-Daten (72, 96, 16)
   ⚠️  Benötigt CNN in der Architektur
   ⚠️  Langsamer zu trainieren
   ⚠️  Mehr GPU-Memory
   ✅ Mehr visuelle Information

2. 'simple115v2' (Alternative):
   ✅ Liefert FEATURE VECTOR (460 wenn stacked)
   ✅ Eure aktuelle Architektur funktioniert SOFORT
   ✅ Viel schneller zu trainieren
   ✅ Weniger GPU-Memory
   ✅ Strukturierte Features (Positionen, Geschwindigkeiten, etc.)

ENTSCHEIDUNG:
-----------
Für schnelles Prototyping und erste Experimente: 
→ Wechsle zu 'simple115v2' in TRAINING_STAGES

Für maximale Performance mit mehr Aufwand:
→ Behalte 'extracted' und füge CNN hinzu (siehe model_with_cnn.py)
    """)
    print("🎯"*40 + "\n")

if __name__ == "__main__":
    main()