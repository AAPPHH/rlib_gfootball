import os, csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT_LOGDIR = r"C:\clones\rlib_gfootball\training_results_transfer_kan"
OUT_CSV     = r"C:\clones\rlib_gfootball\tb_export_csv"

def iter_event_dirs(root: Path):
    for p in root.rglob("*"):
        if p.is_dir():
            try:
                files = os.listdir(p)
            except PermissionError:
                continue
            if any(f.startswith("events.out.tfevents") for f in files):
                yield p

def main():
    root = Path(ROOT_LOGDIR)
    rows = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "tag", "step", "value", "wall_time"])
        for log_dir in iter_event_dirs(root):
            run_name = str(log_dir).replace(str(root), "").strip("\\/").replace("\\", "/")
            ea = EventAccumulator(str(log_dir), size_guidance={'scalars': 0})
            try:
                ea.Reload()
            except Exception as e:
                print(f"Skip {log_dir}: {e}")
                continue
            scalar_tags = ea.Tags().get('scalars', [])
            for tag in scalar_tags:
                for e in ea.Scalars(tag):
                    w.writerow([run_name, tag, e.step, e.value, e.wall_time])
                    rows += 1
    print(f"✅ wrote {rows} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
