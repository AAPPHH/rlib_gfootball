# 🏆 IMPALA League: Total Domination in Google Research Football

A high-performance reinforcement learning system for Google Research Football that achieves **93% win rate** against the hard bot with an average goal difference of **+3.2** — demonstrating total domination rather than marginal wins.

## 🎯 Key Results

| Metric | Value | Comparison |
|--------|-------|------------|
| Win Rate vs Hard Bot | **93%** | vs. "barely positive" (Google 2019) |
| Goal Difference | **+3.2** | Complete dominance |
| Training Steps | **1.3M** | vs. 500M (Google) / 2M (Light-MALib) |
| Reward Shaping | **None** | Pure scoring reward only |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IMPALA League System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Worker 1  │    │   Worker 2  │    │  Worker N   │     │
│  │  (Ray Actor)│    │  (Ray Actor)│    │  (Ray Actor)│     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Pure League                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ bot_easy│ │bot_medium│ │bot_hard │ │snapshots│   │   │
│  │  │  μ=18   │ │  μ=28   │ │  μ=38   │ │  μ=var  │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  │           TrueSkill Matchmaking                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Learner (GPU)                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │  V-Trace │ │   SIL    │ │  Expert  │            │   │
│  │  │  Update  │ │  Update  │ │  Buffer  │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  │              PopArt Value Normalization              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Core Components

### 1. Pure League System
- **TrueSkill Rating**: Bayesian skill estimation for all agents
- **Dynamic Opponent Selection**:
  - 50% Champion (strongest member)
  - 35% Skill-matched (similar rating)
  - 15% Exploration (least-played)
- **Automatic Snapshots**: Saved when rating improves by 2.0 or 5 wins vs champion
- **Strong Opponent Filter**: Prevents games against obsolete weak snapshots

### 2. IMPALA with V-Trace
- Off-policy actor-critic with importance sampling correction
- Clipped importance weights (ρ̄=1.0, c̄=1.0)
- Handles asynchronous weight updates gracefully

### 3. Self-Imitation Learning (SIL)
- Learns from best past experiences stored in Golden Memory
- Weighted sampling by return and opponent strength
- Maximum 8 uses per experience to prevent overfitting

### 4. Expert Buffer
- Imitation learning from expert demonstrations
- Automatically disabled when agent surpasses 70% win rate vs hard bot
- Prioritized sampling by episode return

### 5. PopArt Value Normalization
- Adaptive normalization of value targets
- Prevents value function collapse during rapid improvement
- Preserves output scale through weight rescaling

## 📦 Installation

```bash
# Create conda environment
conda create -n grf python=3.10
conda activate grf

# Install Google Research Football
pip install gfootball

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ray[default] trueskill pyarrow numpy

# Clone repository
git clone https://github.com/yourusername/grf-league.git
cd grf-league
```

## 🚀 Usage

### Training from Scratch
```python
from impala_league import LeagueLearner

learner = LeagueLearner(
    num_workers=32,
    rollout_len=512,
    batch_size=128,
    lr=0.0005,
    gamma=0.997,
    entropy_coeff=0.01,
    value_coeff=0.5,
    sil_coeff=0.5,
    d_model=512,
    lstm_hidden=512,
    checkpoint_dir="./checkpoints_league",
    max_snapshots=15,
    skill_matched_prob=0.35,
    champion_prob=0.50,
    exploration_prob=0.15,
)

learner.train(max_time=360000)  # 100 hours
```

### Training with Warmstart
```python
learner = LeagueLearner(
    # ... same as above ...
    warmstart_path="path/to/checkpoint.pt",
    expert_parquet="path/to/expert.parquet",
    expert_threshold=0.7,  # Disable expert when >70% vs hard
)

learner.train(max_time=360000)
```

### Monitoring Output
```
[  20] 1.3M 1k/s 19m | W: 89% R:+4.7(+12) | μ=48.3 σ=0.86 | E:95%(82) M:91%(85) H:93%(130) RvH:+3.2 | ...
        │     │    │     │      │           │               │
        │     │    │     │      │           │               └── Win rates vs Easy/Medium/Hard (games)
        │     │    │     │      │           └── TrueSkill rating
        │     │    │     │      └── Average return (max return)
        │     │    │     └── Overall win rate
        │     │    └── Training time
        │     └── Steps per second
        └── Total steps
```

## 📁 Project Structure

```
main/
├── impala_league.py      # Main training script (monolithic)
├── expert.parquet        # Expert demonstrations (optional)
├── checkpoints_league/   # Training checkpoints
│   ├── league.pkl        # League state (ratings, members)
│   ├── snapshot_*.pt     # Agent snapshots
│   └── ckpt_*.pt         # Training checkpoints
└── README.md
```

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_workers` | 32 | Parallel rollout workers |
| `rollout_len` | 512 | Steps per rollout |
| `batch_size` | 128 | Rollouts per update |
| `lr` | 5e-4 | Learning rate |
| `gamma` | 0.997 | Discount factor |
| `entropy_coeff` | 0.01 | Entropy bonus |
| `value_coeff` | 0.5 | Value loss weight |
| `sil_coeff` | 0.5 | SIL loss weight |
| `d_model` | 512 | Transformer/MLP hidden dim |
| `lstm_hidden` | 512 | LSTM hidden dim |
| `max_snapshots` | 15 | Maximum league snapshots |
| `champion_prob` | 0.50 | Probability of playing champion |
| `skill_matched_prob` | 0.35 | Probability of skill-matched opponent |

## 🔬 Technical Details

### Network Architecture
- **Input**: 115-dim observation + 93-dim engineered features + 16-dim action embedding
- **Encoder**: 2-layer MLP with LayerNorm and ReLU
- **Sequence Model**: Single-layer LSTM (512 hidden)
- **Policy Head**: 2-layer MLP → 19 actions
- **Value Head**: PopArt normalized single output

### Feature Engineering
- Ball position, velocity, ownership
- Relative positions to goal, opponents, teammates
- Offside line detection
- Game mode encoding
- Sticky action states

### TrueSkill Configuration
```python
TrueSkill(
    mu=25.0,        # Initial skill mean
    sigma=8.333,    # Initial uncertainty
    beta=4.166,     # Performance variance
    tau=0.083,      # Dynamics factor
    draw_probability=0.05
)
```

## 📊 Comparison with Prior Work

| Method | Steps to Beat Hard | Win Rate | Goal Diff | Reward |
|--------|-------------------|----------|-----------|--------|
| Google IMPALA (2019) | 500M | ~50% | ~0 | Checkpoint |
| Light-MALib IPPO (2023) | 2M | >50% | N/A | Dense |
| **Ours** | **1.3M** | **93%** | **+3.2** | **Scoring only** |


## 🙏 Acknowledgments

- Google Research Football Team

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Total Domination Achieved.** 🏆