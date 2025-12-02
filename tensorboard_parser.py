import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_LOGDIR = r"C:\clones\rlib_gfootball\logs_mamba"
OUT_CSV = r"C:\clones\rlib_gfootball\training_analysis.csv"

# Fokus auf Stage 8 und 9 Transition + Curriculum Dynamik
FILTER_TAGS = [
    # Loss Tracking
    "loss/total",
    "loss/policy",
    "loss/value",
    
    # Win Rates pro Stage (besonders 8 und 9)
    "episode/win_rate_stage_8",
    "episode/win_rate_stage_9",
    "episode/win_rate_stage_10",
    
    # Episode Returns
    "episode/return_stage_8",
    "episode/return_stage_9",
    "episode/return_mean",
    "episode/return_max",
    
    # Max Rewards (zeigt Fortschritt)
    "episode/max_reward_stage_8",
    "episode/max_reward_stage_9",
    
    # Curriculum EMA (wichtig für Forgetting-Analyse)
    "curriculum/ema_win_stage_8",
    "curriculum/ema_win_stage_9",
    "curriculum/sustained_peak_stage_8",
    "curriculum/sustained_peak_stage_9",
    
    # Sample Probabilities
    "curriculum/sample_prob_stage_8",
    "curriculum/sample_prob_stage_9",
    
    # PPO Metriken
    "ppo/clip_fraction",
    "ppo/approx_kl",
    "ppo/explained_variance",
    
    # Training
    "train/lr",
    "train/grad_norm",
]

# Jeden 5. Datenpunkt für gute Auflösung aber kompakte Größe
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
    return any(ft in tag for ft in filters)

def main():
    root = Path(ROOT_LOGDIR)
    if not root.exists():
        print(f"Pfad existiert nicht: {root}")
        return

    rows = 0
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
            matched_tags = [t for t in available_tags if tag_matches(t, FILTER_TAGS)]
            print(f"  {len(matched_tags)} relevante Tags gefunden")
            
            for tag in matched_tags:
                try:
                    scalars = ea.Scalars(tag)
                    for i, e in enumerate(scalars):
                        if i % DOWNSAMPLE == 0:
                            w.writerow([run_name, tag, e.step, round(e.value, 4)])
                            rows += 1
                except KeyError:
                    continue

    file_size_mb = os.path.getsize(OUT_CSV) / (1024 * 1024)
    print(f"\nFertig: {rows} Zeilen, {file_size_mb:.2f} MB -> {OUT_CSV}")
    
    if file_size_mb > 25:
        print("⚠️ WARNUNG: Datei > 25MB, erhöhe DOWNSAMPLE!")

if __name__ == "__main__":
    main()