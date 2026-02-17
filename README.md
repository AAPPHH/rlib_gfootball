# GFootball IMPALA League — Self-Play with TrueSkill Matchmaking

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Google Research Football](https://img.shields.io/badge/Env-GFootball-0a9d57)](https://github.com/google-research/football)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> Trains an 11v11 Google Research Football agent using **IMPALA + V-trace**,
> **TrueSkill-based League Play**, **Self-Imitation Learning**, and
> **Behavioral Cloning Warmstart** — completely from scratch to beating the Hard Bot.

---

## Demo

[![GFootball Agent Demo](https://img.youtube.com/vi/_oVHxy5FDkQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=_oVHxy5FDkQ)

> Click on the image to watch the video on YouTube.

---

## Features

- **IMPALA with V-trace** off-policy correction and mixed-precision training
- **TrueSkill League** with automatic matchmaking (champion, skill-matched, and exploration selection)
- **Self-Imitation Learning (SIL)** from golden memory of the best episodes
- **Behavioral Cloning Warmstart** from expert demonstrations (Parquet)
- **PopArt Value Normalization** for stable value estimates under non-stationary targets
- **Automatic Snapshotting** on rating gains or champion victories
- **Distributed Rollout Collection** via Ray with 32+ parallel workers

---

## Project Structure

```
main/
├── feature_engineer.py   # 93-dim feature extraction from 115-dim observations
├── net.py                # Encoder-LSTM-Policy/Value network with PopArt
├── bc_train.py           # Behavioral Cloning trainer (warmstart)
├── league_learner.py     # IMPALA League training orchestrator
└── expert.parquet        # Expert demonstrations
```

---

## Architecture

### Network (`net.py`)

```
[obs(115) | feat(93) | action_emb(16)] → 2-Layer MLP → LSTM → Policy + PopArt Value
```

- **Encoder:** 2-layer MLP with LayerNorm + ReLU
- **Sequence Model:** LSTM (512 hidden) for temporal dependencies
- **Policy Head:** Linear → ReLU → Linear (19 actions)
- **Value Head:** PopArt-normalized with adaptive weight correction
- **Init:** Orthogonal initialization (gain=sqrt(2), policy gain=0.01)

### Feature Engineering (`feature_engineer.py`)

93 handcrafted features from the `simple115v2` representation:

| Feature Group          | Dim  | Description                               |
|------------------------|:----:|-------------------------------------------|
| Ball State             | 7    | Position, height, speed, direction, possession |
| Relative Ball Position | 4    | Distance and angle to active player        |
| Goal Geometry          | 5    | Distance, angle, shooting zone, own goal   |
| Keeper Tracking        | 2    | Distance and angle to opposing keeper      |
| 5 Nearest Teammates    | 20   | Relative position + direction              |
| 5 Nearest Opponents    | 20   | Relative position + direction              |
| Formation              | 3    | Players ahead/behind ball, balance         |
| Offside                | 2    | Offside line, offside flag                 |
| Zones                  | 5    | Attack/midfield/defense, lateral           |
| Sticky Actions         | 11   | Sprint, dribble, direction vector          |
| Game Mode              | 7    | One-hot game mode                          |
| Reserved               | 8    | Zeros (extensibility)                      |

### League System (`league_learner.py`)

```
                    ┌─────────────┐
                    │  Champion   │ ← 50% selection
                    └──────┬──────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
┌───▼───┐           ┌─────▼─────┐          ┌─────▼─────┐
│ Bots  │           │ Snapshots │          │ Agent     │
│ E/M/H │           │ (Top-15)  │          │ (Learner) │
└───────┘           └───────────┘          └───────────┘
  random              skill_matched           V-trace
  lazy                35% selection           + SIL
  easy/med/hard       15% exploration         + Expert BC
```

**Opponent Pool:**
- **Built-in Bots:** random, lazy, easy, medium, hard (fixed TrueSkill ratings)
- **Snapshots:** Past agent versions (max 15, automatically pruned)
- **Selection:** 50% champion, 35% skill-matched, 15% exploration

---

## Installation

```bash
# Environment
conda create -n gfootball python=3.10 -y
conda activate gfootball

# PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install gfootball ray numpy trueskill pyarrow
```

---

## Quick Start

### 1) Behavioral Cloning Warmstart

```bash
cd main
python bc_train.py expert.parquet bc_warmstart.pt
```

Trains a policy network supervised on expert data (20 epochs, ~45% val accuracy).

### 2) League Training

```bash
python league_learner.py
```

Starts the full league training with:
- 32 Ray workers for parallel rollout collection
- V-trace IMPALA updates every 128 rollouts
- SIL from golden memory + expert buffer
- Automatic snapshotting on rating gains

### Configuration

Hyperparameters are directly configurable in the `__main__` block of `league_learner.py`:

| Parameter          | Default | Description                            |
|--------------------|:-------:|----------------------------------------|
| `num_workers`      | 32      | Number of Ray rollout workers          |
| `rollout_len`      | 512     | Steps per rollout                      |
| `batch_size`       | 128     | Rollouts per update                    |
| `lr`               | 5e-4    | Learning rate (Adam)                   |
| `d_model`          | 512     | Encoder hidden dimension               |
| `lstm_hidden`      | 512     | LSTM hidden dimension                  |
| `gamma`            | 0.997   | Discount factor                        |
| `entropy_coeff`    | 0.01    | Entropy bonus                          |
| `max_snapshots`    | 15      | Max league snapshots                   |
| `expert_threshold` | 0.7     | Win rate vs Hard Bot to disable expert |

---

## Training Progress

Typical log output:

```
[  10] 0.7M 1k/s 8m | W: 78% R:+3.1(+11) | μ=41.8 σ=1.55 | E:50%(0) M:50%(0) H:62%(17)
```

| Field      | Meaning                                          |
|------------|--------------------------------------------------|
| `[  10]`   | Update number                                    |
| `0.7M`     | Total environment steps                          |
| `W: 78%`   | Win rate (last 100 games)                        |
| `R:+3.1`   | Average return                                   |
| `μ=41.8`   | TrueSkill rating                                 |
| `E/M/H`    | Win rate vs Easy/Medium/Hard Bot                 |
| `VT`       | V-trace loss metrics                             |
| `EXP/SIL`  | Expert and self-imitation metrics                |
| `GM`       | Golden memory size (fresh entries)               |

---

## Results

The agent achieves after approximately 100 updates:
- **100%** win rate vs Easy Bot
- **60%+** win rate vs Medium Bot
- **62%+** win rate vs Hard Bot
- **TrueSkill μ > 40** (conservative skill > 35)

---

## License

MIT License. See [LICENSE](LICENSE).