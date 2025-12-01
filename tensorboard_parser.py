import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_LOGDIR = r"C:\clones\rlib_gfootball\logs"
OUT_CSV = r"C:\clones\rlib_gfootball\training_compact.csv"

# Nur die wichtigsten Metriken - stark reduziert
FILTER_TAGS = [
    "loss/total",
    "episode/win_rate",
    "curriculum/ema_win_stage_",
    "curriculum/sample_prob_stage_",
    "curriculum/num_learned",
]

# Nur jeden N-ten Datenpunkt (z.B. alle 10)
DOWNSAMPLE = 10

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
                    scalars = ea.Scalars(tag)
                    for i, e in enumerate(scalars):
                        if i % DOWNSAMPLE == 0:  # Nur jeden N-ten Punkt
                            w.writerow([run_name, tag, e.step, round(e.value, 4)])
                            rows += 1
                except KeyError:
                    continue

    print(f"Fertig: {rows} Zeilen -> {OUT_CSV}")

if __name__ == "__main__":
    main()