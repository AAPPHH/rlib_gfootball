import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_LOGDIR = r"C:\clones\rlib_gfootball\ray_results"
OUT_CSV = r"C:\clones\rlib_gfootball\curriculum_export.csv"

FILTER_TAGS = [
    "stage_0", "stage_1", "stage_2", "stage_3",
    "stage_4", "stage_5", "stage_6", "stage_7",
    "curriculum/",
    "pbt_metric",
    "episode_return_mean",
    "episode_len_mean",
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
            except Exception:
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