"""Behavioral-Cloning-Warmstart fuer die IMPALA-League.

Trainiert das ORIGINAL-Netz aus full_league.py (inkl. Value-Heads, die vom
BC-Loss unberuehrt bleiben) supervised auf Expertendaten. Dadurch ist das
gespeicherte Checkpoint strict-kompatibel mit dem League-Learner und den
Workern — dieselbe Konstruktion wie bc_train.py/net.py auf der main-Branch,
damit Netz-Umbauten in full_league.py nie wieder still am Warmstart
vorbeilaufen koennen.
"""
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from full_league import Net, FeatureEngineer, OBS_DIM, FEATURE_DIM, NUM_ACTIONS


class ExpertDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path}...")
        df = pq.read_table(parquet_path).to_pandas()
        self.obs = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
        self.actions = df['action'].values.astype(np.int64)
        self.active = df['active'].values.astype(np.int64)
        print(f"Computing features for {len(self.obs)} samples...")
        self.feat = np.array([FeatureEngineer.extract(o, a) for o, a in zip(self.obs, self.active)])
        print(f"Dataset ready: {len(self.obs)} samples")
        print("Action distribution:")
        counts = np.bincount(self.actions, minlength=NUM_ACTIONS)
        for i, c in enumerate(counts):
            if c > 0:
                print(f"  {i:2d}: {c:6d} ({c/len(self.actions)*100:5.1f}%)")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            'obs': torch.from_numpy(self.obs[idx]).float(),
            'feat': torch.from_numpy(self.feat[idx]).float(),
            'action': torch.tensor(self.actions[idx]).long(),
        }


def train_bc(parquet_path, epochs=20, batch_size=512, lr=1e-3, save_path="bc_warmstart_v2.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    dataset = ExpertDataset(parquet_path)
    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train: {n_train} | Val: {n_val}\n")
    model = Net(d_model=512, lstm_hidden=512).to(device)
    print(f"Net: {sum(p.numel() for p in model.parameters()):,} params")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9}")
    print("-" * 60)
    best_val_acc = -1.0
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in train_loader:
            obs = batch['obs'].to(device)
            feat = batch['feat'].to(device)
            actions = batch['action'].to(device)
            logits, _, _ = model(obs, feat)
            loss = F.cross_entropy(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(obs)
            train_correct += (logits.argmax(dim=-1) == actions).sum().item()
            train_total += len(obs)
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                feat = batch['feat'].to(device)
                actions = batch['action'].to(device)
                logits, _, _ = model(obs, feat)
                loss = F.cross_entropy(logits, actions)
                val_loss += loss.item() * len(obs)
                val_correct += (logits.argmax(dim=-1) == actions).sum().item()
                val_total += len(obs)
        train_loss /= train_total
        train_acc = train_correct / train_total * 100
        val_loss /= val_total
        val_acc = val_correct / val_total * 100
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.1f}% | {val_loss:10.4f} | {val_acc:8.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch + 1}, save_path)
    print(f"\nFinal: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
    print(f"Best model saved to {save_path} (Val Acc: {best_val_acc:.1f}%)")
    if val_acc > 30:
        print("\nSANITY CHECK PASSED - Network can learn from expert data")
    else:
        print("\nSANITY CHECK FAILED - Network struggles to learn")
    return model, val_acc


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\clones\rlib_gfootball\main\expert.parquet"
    save_path = sys.argv[2] if len(sys.argv) > 2 else r"C:\clones\rlib_gfootball\bc_warmstart_v2.pt"
    train_bc(parquet_path, epochs=20, batch_size=512, lr=1e-3, save_path=save_path)
