import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_LOGDIR = r"C:\clones\rlib_gfootball\logs_vtrace"
OUT_CSV = r"C:\clones\rlib_gfootball\feature_importance_analysis.csv"

# ============================================================================
# FEATURE IMPORTANCE TAGS
# ============================================================================

# Alle 32 einzelnen Features
FEATURE_NAMES = [
    'dist_to_goal', 'dist_to_own_goal', 'goal_angle', 'in_shooting_range',
    'in_penalty_area', 'ball_speed', 'moving_to_goal', 'keeper_dist',
    'keeper_angle_to_ball', 'shooting_angle', 'ball_progress', 'ball_z',
    'teammates_ahead', 'defenders_ahead', 'numerical_advantage', 'keeper_close',
    'good_shooting_angle', 'ball_x', 'ball_y', 'ball_dir_x',
    'ball_dir_y', 'sticky_0', 'sticky_1', 'sticky_2',
    'sticky_3', 'in_attack_third', 'in_defense_third', 'on_wing',
    'proximity_to_goal', 'shooting_opportunity', 'attack_momentum', 'danger_zone'
]

# Feature Gruppen
FEATURE_GROUPS = [
    'ball_position',   # ball_progress, ball_z, ball_x, ball_y
    'ball_movement',   # ball_speed, moving_to_goal, ball_dir_x/y
    'goal_threat',     # dist_to_goal, goal_angle, in_shooting_range, in_penalty, shooting_angle
    'keeper',          # keeper_dist, keeper_angle, keeper_close
    'team_structure',  # teammates_ahead, defenders_ahead, numerical_adv
    'zones',           # attack_third, defense_third, on_wing
    'composite',       # proximity, shooting_opp, attack_momentum, danger_zone
    'sticky',          # sticky actions
    'defense',         # dist_to_own_goal
    'flags',           # good_shooting_angle
]

# Zusätzliche Training-Metriken für Kontext
TRAINING_TAGS = [
    "loss/total",
    "loss/policy", 
    "loss/value",
    "ppo/explained_variance",
    "episode/win_rate_stage_0",
    "episode/win_rate_stage_1",
    "episode/win_rate_stage_2",
    "episode/win_rate_stage_3",
    "episode/win_rate_stage_4",
    "episode/win_rate_stage_5",
    "episode/win_rate_stage_6",
    "episode/win_rate_stage_7",
    "episode/win_rate_stage_8",
    "episode/win_rate_stage_9",
    "episode/win_rate_stage_10",
]

# Baue vollständige Tag-Liste
FILTER_TAGS = []

# Feature Importance pro Feature (global)
for name in FEATURE_NAMES:
    FILTER_TAGS.append(f"feature_importance/{name}")

# Feature Importance pro Stage (0-12)
for stage in range(13):
    for name in FEATURE_NAMES:
        FILTER_TAGS.append(f"feature_importance_stage{stage}/{name}")

# Feature Group Importance
for group in FEATURE_GROUPS:
    FILTER_TAGS.append(f"feature_group_importance/{group}")

# Training Metriken
FILTER_TAGS.extend(TRAINING_TAGS)

# Jeden N-ten Datenpunkt
DOWNSAMPLE = 5


def iter_event_dirs(root: Path):
    for p in root.rglob("*"):
        if p.is_dir():
            try:
                files = os.listdir(p)
            except PermissionError:
                continue
            if any(f.startswith("events.out.tfevents") for f in files):
                yield p


def clean_run_name(root: Path, log_dir: Path):
    rel_path = str(log_dir.relative_to(root)).replace("\\", "/")
    parts = rel_path.split("/")
    return parts[-1] if parts else rel_path


def tag_matches(tag: str, filters: list) -> bool:
    # Match feature_importance tags direkt
    if 'feature_importance' in tag:
        return True
    return tag in filters or any(ft in tag for ft in filters)


def main():
    root = Path(ROOT_LOGDIR)
    if not root.exists():
        print(f"Pfad existiert nicht: {root}")
        return

    rows = 0
    feature_importance_found = 0
    
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "tag", "step", "value"])

        for log_dir in iter_event_dirs(root):
            run_name = clean_run_name(root, log_dir)
            print(f"Lade: {run_name}")
            
            ea = EventAccumulator(str(log_dir), size_guidance={'scalars': 0})
            
            try:
                ea.Reload()
            except Exception as e:
                print(f"  Fehler: {e}")
                continue

            available_tags = ea.Tags().get('scalars', [])
            
            # Zähle Feature Importance Tags
            fi_tags = [t for t in available_tags if 'feature_importance' in t]
            if fi_tags:
                print(f"  ✓ {len(fi_tags)} Feature Importance Tags gefunden")
                feature_importance_found += len(fi_tags)
            
            matched_tags = [t for t in available_tags if tag_matches(t, FILTER_TAGS)]
            print(f"  {len(matched_tags)} relevante Tags gesamt")
            
            for tag in matched_tags:
                try:
                    scalars = ea.Scalars(tag)
                    for i, e in enumerate(scalars):
                        if i % DOWNSAMPLE == 0:
                            w.writerow([run_name, tag, e.step, round(e.value, 6)])
                            rows += 1
                except KeyError:
                    continue

    file_size_mb = os.path.getsize(OUT_CSV) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Fertig: {rows} Zeilen, {file_size_mb:.2f} MB -> {OUT_CSV}")
    print(f"Feature Importance Tags gefunden: {feature_importance_found}")
    
    if feature_importance_found == 0:
        print("\n⚠️ KEINE Feature Importance Daten gefunden!")
        print("   Starte Training mit main_32_features.py und warte ~500 Updates")
    
    if file_size_mb > 25:
        print("⚠️ WARNUNG: Datei > 25MB, erhöhe DOWNSAMPLE!")


def analyze_csv():
    import pandas as pd
    
    df = pd.read_csv(OUT_CSV)
    
    print(f"\nGeladene Zeilen: {len(df)}")
    print(f"Unique Tags: {df['tag'].nunique()}")
    
    # Zeige alle feature_importance Tags
    fi_tags = df[df['tag'].str.contains('feature_importance')]['tag'].unique()
    print(f"\nFeature Importance Tags: {len(fi_tags)}")
    if len(fi_tags) > 0:
        print("Beispiele:", list(fi_tags)[:5])
    
    # === GLOBAL IMPORTANCE ===
    # Neues Format: feature_importance/global/feature_name
    fi_global = df[df['tag'].str.contains('feature_importance/global/')]
    
    if not fi_global.empty:
        latest = fi_global.groupby('tag').apply(lambda x: x.nlargest(20, 'step')['value'].mean())
        latest = latest.sort_values(ascending=False)
        
        print("\n" + "="*60)
        print("GLOBAL FEATURE IMPORTANCE")
        print("="*60)
        
        print("\nTOP 10:")
        for i, (tag, val) in enumerate(latest.head(10).items()):
            name = tag.split('/')[-1]
            print(f"  {i+1:2}. {name:25} {val:.6f}")
        
        print("\nBOTTOM 10:")
        for i, (tag, val) in enumerate(latest.tail(10).items()):
            name = tag.split('/')[-1]
            print(f"  {i+1:2}. {name:25} {val:.6f}")
    else:
        print("\nKeine global feature importance Daten")
    
    fi_groups = df[df['tag'].str.contains('feature_importance/groups/')]
    
    if not fi_groups.empty:
        group_latest = fi_groups.groupby('tag').apply(lambda x: x.nlargest(20, 'step')['value'].mean())
        group_latest = group_latest.sort_values(ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE GRUPPEN RANKING")
        print("="*60)
        
        for tag, val in group_latest.items():
            name = tag.split('/')[-1]
            print(f"  {name:20} {val:.6f}")
    
    print("\n" + "="*60)
    print("STAGE-SPEZIFISCHE FEATURE IMPORTANCE")
    print("="*60)
    
    for stage in range(13):
        stage_df = df[df['tag'].str.contains(f'feature_importance/stage_{stage:02d}/')]
        if stage_df.empty:
            continue
        
        stage_latest = stage_df.groupby('tag').apply(lambda x: x.nlargest(10, 'step')['value'].mean())
        stage_latest = stage_latest.sort_values(ascending=False)
        
        if len(stage_latest) > 0:
            top3 = [(t.split('/')[-1], v) for t, v in stage_latest.head(3).items()]
            bottom3 = [(t.split('/')[-1], v) for t, v in stage_latest.tail(3).items()]
            
            print(f"\nStage {stage:2}:")
            print(f"  TOP:    {', '.join([f'{n}:{v:.4f}' for n,v in top3])}")
            print(f"  BOTTOM: {', '.join([f'{n}:{v:.4f}' for n,v in bottom3])}")
    
    print("\n" + "="*60)
    print("GRUPPEN IMPORTANCE PRO STAGE")
    print("="*60)
    
    for stage in range(13):
        stage_groups = df[df['tag'].str.contains(f'feature_importance/stage_{stage:02d}_groups/')]
        if stage_groups.empty:
            continue
        
        group_latest = stage_groups.groupby('tag').apply(lambda x: x.nlargest(10, 'step')['value'].mean())
        group_latest = group_latest.sort_values(ascending=False)
        
        top3 = [(t.split('/')[-1], v) for t, v in group_latest.head(3).items()]
        print(f"Stage {stage:2}: {', '.join([f'{n}:{v:.4f}' for n,v in top3])}")
    
    if not fi_global.empty:
        print("\n" + "="*60)
        print("REDUNDANZ-CHECK")
        print("="*60)
        
        redundant_pairs = [
            ('ball_progress', 'ball_x'),
            ('keeper_close', 'keeper_dist'),
            ('good_shooting_angle', 'shooting_angle'),
            ('proximity_to_goal', 'dist_to_goal'),
            ('danger_zone', 'in_penalty_area'),
            ('in_attack_third', 'ball_x'),
            ('in_defense_third', 'ball_x'),
        ]
        
        for f1, f2 in redundant_pairs:
            v1 = latest.get(f'feature_importance/global/{f1}', 0)
            v2 = latest.get(f'feature_importance/global/{f2}', 0)
            if v1 > 0 or v2 > 0:
                print(f"  {f1:20} {v1:.6f} vs {f2:20} {v2:.6f}")
                if min(v1, v2) > 0.01 and min(v1, v2) / (max(v1, v2) + 1e-8) > 0.7:
                    keep = f2 if v2 > v1 else f1
                    drop = f1 if v2 > v1 else f2
                    print(f"    -> Redundant: consider dropping {drop}")


if __name__ == "__main__":
    main()
    
    try:
        analyze_csv()
    except ImportError:
        print("\nFür Analyse: pip install pandas")
    except Exception as e:
        print(f"\nAnalyse-Fehler: {e}")