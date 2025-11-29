import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_LOGDIR = r"C:\clones\rlib_gfootball\logs"  # Angepasst auf neuen log_dir
OUT_CSV = r"C:\clones\rlib_gfootball\training_export.csv"

# Angepasst auf das neue Logging-Format
FILTER_TAGS = [
    # Loss metrics
    "loss/total",
    "loss/policy",
    "loss/value",
    "loss/entropy",
    
    # PPO metrics
    "ppo/clip_fraction",
    "ppo/approx_kl",
    "ppo/explained_variance",
    
    # Training metrics
    "train/lr",
    "train/grad_norm",
    "train/nan_count",
    
    # Episode metrics (global)
    "episode/return_mean",
    "episode/return_std",
    "episode/return_min",
    "episode/return_max",
    "episode/win_rate",
    "episode/length_mean",
    
    # Episode metrics (per stage)
    "episode/return_stage_",
    "episode/win_rate_stage_",
    "episode/length_stage_",
    
    # Curriculum metrics
    "curriculum/ema_return_stage_",
    "curriculum/ema_win_stage_",
    "curriculum/normalized_return_stage_",
    "curriculum/episodes_stage_",
    
    # Throughput
    "throughput/steps_per_second",
    "throughput/episodes",
    "throughput/updates",
]

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
            ea = EventAccumulator(str(log_dir), size_guidance={'scalars': 0})
            
            try:
                ea.Reload()
            except Exception as e:
                print(f"Fehler beim Laden von {log_dir}: {e}")
                continue

            for tag in ea.Tags().get('scalars', []):
                if not any(ft in tag for ft in FILTER_TAGS):
                    continue
                try:
                    for e in ea.Scalars(tag):
                        w.writerow([run_name, tag, e.step, e.value])
                        rows += 1
                except KeyError:
                    continue

    print(f"Fertig: {rows} Zeilen -> {OUT_CSV}")

if __name__ == "__main__":
    main()