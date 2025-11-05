# load_simple115.py
import pickle, numpy as np, pandas as pd
from pathlib import Path

def extract_simple115(obs):
    if isinstance(obs, np.ndarray) and obs.ndim==1 and obs.shape[0]==115:
        return obs.astype(np.float32, copy=False)
    if isinstance(obs, (list, tuple)) and obs:
        a0 = np.asarray(obs[0])
        if a0.ndim==1 and a0.shape[0]==115:
            return a0.astype(np.float32, copy=False)
    return None

def load_all(d):
    rows=[]
    for f in sorted(Path(d).glob("*.dump")):
        with open(f, "rb") as fh:
            while True:
                try:
                    step = pickle.load(fh)
                except EOFError:
                    break
                v = extract_simple115(step.get("observation"))
                if v is None:
                    continue
                rew = float(step.get("reward", 0.0) or 0.0)
                dbg = step.get("debug") or {}
                done = bool(dbg.get("done", step.get("done", False)))
                rows.append({**{f"f_{i}": v[i] for i in range(115)},
                             "reward": rew, "done": done, "source_file": f.name})
    if not rows:
        raise RuntimeError("Keine simple115-Observationen gefunden.")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_all("gfootball_dumps")
    df.to_parquet("gfootball_simple115.parquet", index=False)
    print(df.head(), "\n", df.shape)
