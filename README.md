# GFootball IMPALA League — Self-Play mit TrueSkill-Matchmaking

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Google Research Football](https://img.shields.io/badge/Env-GFootball-0a9d57)](https://github.com/google-research/football)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> Trainiert einen 11v11 Google Research Football Agent mit **IMPALA + V-trace**,
> **TrueSkill-basiertem League-Play**, **Self-Imitation Learning** und
> **Behavioral Cloning Warmstart** — komplett von Grund auf bis zum Sieg gegen den Hard-Bot.

---

## Demo

[![GFootball Agent Demo](https://img.youtube.com/vi/_oVHxy5FDkQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=_oVHxy5FDkQ)

> Klick auf das Bild um das Video auf YouTube anzusehen.

---

## Features

- **IMPALA mit V-trace** Off-Policy-Korrektur und Mixed-Precision Training
- **TrueSkill League** mit automatischem Matchmaking (Champion-, Skill-Matched- und Exploration-Selektion)
- **Self-Imitation Learning (SIL)** aus Golden Memory der besten Episoden
- **Behavioral Cloning Warmstart** aus Expert-Demonstrationen (Parquet)
- **PopArt Value Normalization** fuer stabile Wertschaetzungen bei nicht-stationaeren Targets
- **Automatisches Snapshotting** bei Rating-Gewinn oder Champion-Siege
- **Verteiltes Rollout-Sammeln** ueber Ray mit 32+ parallelen Workern

---

## Projektstruktur

```
main/
├── feature_engineer.py   # 93-dim Feature-Extraktion aus 115-dim Observationen
├── net.py                # Encoder-LSTM-Policy/Value Netz mit PopArt
├── bc_train.py           # Behavioral Cloning Trainer (Warmstart)
├── league_learner.py     # IMPALA League Training Orchestrator
└── expert.parquet        # Expert-Demonstrationen
```

---

## Architektur

### Netzwerk (`net.py`)

```
[obs(115) | feat(93) | action_emb(16)] → 2-Layer MLP → LSTM → Policy + PopArt Value
```

- **Encoder:** 2-schichtiges MLP mit LayerNorm + ReLU
- **Sequenzmodell:** LSTM (512 hidden) fuer temporale Abhaengigkeiten
- **Policy Head:** Linear → ReLU → Linear (19 Actions)
- **Value Head:** PopArt-normalisiert mit adaptiver Gewichtskorrektur
- **Init:** Orthogonale Initialisierung (gain=sqrt(2), Policy gain=0.01)

### Feature Engineering (`feature_engineer.py`)

93 handgefertigte Features aus der `simple115v2` Repraesentation:

| Feature-Gruppe         | Dim  | Beschreibung                              |
|------------------------|:----:|-------------------------------------------|
| Ball-Zustand           | 7    | Position, Hoehe, Speed, Richtung, Besitz  |
| Relative Ball-Position | 4    | Distanz und Winkel zum aktiven Spieler     |
| Tor-Geometrie          | 5    | Distanz, Winkel, Schusszone, Eigentor      |
| Keeper-Tracking        | 2    | Distanz und Winkel zum gegnerischen Keeper |
| 5 naechste Mitspieler  | 20   | Relative Position + Richtung               |
| 5 naechste Gegner      | 20   | Relative Position + Richtung               |
| Formation              | 3    | Spieler vor/hinter Ball, Balance           |
| Abseits                | 2    | Abseitslinie, Abseits-Flag                 |
| Zonen                  | 5    | Angriff/Mittelfeld/Verteidigung, Lateral   |
| Sticky Actions         | 11   | Sprint, Dribble, Richtungsvektor           |
| Game Mode              | 7    | One-hot Spielmodus                         |
| Reserviert             | 8    | Nullen (Erweiterbarkeit)                   |

### League System (`league_learner.py`)

```
                    ┌─────────────┐
                    │  Champion   │ ← 50% Selektion
                    └──────┬──────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
┌───▼───┐           ┌─────▼─────┐          ┌─────▼─────┐
│ Bots  │           │ Snapshots │          │ Agent     │
│ E/M/H │           │ (Top-15)  │          │ (Learner) │
└───────┘           └───────────┘          └───────────┘
  random              skill_matched           V-trace
  lazy                35% Selektion           + SIL
  easy/med/hard       15% exploration         + Expert BC
```

**Gegner-Pool:**
- **Built-in Bots:** random, lazy, easy, medium, hard (feste TrueSkill-Ratings)
- **Snapshots:** Vergangene Agent-Versionen (max 15, automatisch gepruned)
- **Selektion:** 50% Champion, 35% Skill-Matched, 15% Exploration

---

## Installation

```bash
# Umgebung
conda create -n gfootball python=3.10 -y
conda activate gfootball

# PyTorch (CUDA-Version anpassen)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install gfootball ray numpy trueskill pyarrow
```

---

## Schnellstart

### 1) Behavioral Cloning Warmstart

```bash
cd main
python bc_train.py expert.parquet bc_warmstart.pt
```

Trainiert ein Policy-Netz supervised auf Expert-Daten (20 Epochen, ~45% Val-Accuracy).

### 2) League Training

```bash
python league_learner.py
```

Startet das volle League-Training mit:
- 32 Ray-Workern fuer paralleles Rollout-Sammeln
- V-trace IMPALA Updates alle 128 Rollouts
- SIL aus Golden Memory + Expert Buffer
- Automatisches Snapshotting bei Rating-Gewinn

### Konfiguration

Die Hyperparameter sind direkt im `__main__`-Block von `league_learner.py` einstellbar:

| Parameter          | Default | Beschreibung                           |
|--------------------|:-------:|----------------------------------------|
| `num_workers`      | 32      | Anzahl Ray Rollout-Worker              |
| `rollout_len`      | 512     | Steps pro Rollout                      |
| `batch_size`       | 128     | Rollouts pro Update                    |
| `lr`               | 5e-4    | Lernrate (Adam)                        |
| `d_model`          | 512     | Encoder Hidden-Dimension               |
| `lstm_hidden`      | 512     | LSTM Hidden-Dimension                  |
| `gamma`            | 0.997   | Discount-Faktor                        |
| `entropy_coeff`    | 0.01    | Entropie-Bonus                         |
| `max_snapshots`    | 15      | Max League-Snapshots                   |
| `expert_threshold` | 0.7     | Win-Rate vs Hard-Bot zum Expert-Abschalten |

---

## Trainings-Verlauf

Typischer Log-Output:

```
[  10] 0.7M 1k/s 8m | W: 78% R:+3.1(+11) | μ=41.8 σ=1.55 | E:50%(0) M:50%(0) H:62%(17)
```

| Feld       | Bedeutung                                        |
|------------|--------------------------------------------------|
| `[  10]`   | Update-Nummer                                    |
| `0.7M`     | Gesamte Environment-Steps                        |
| `W: 78%`   | Win-Rate (letzte 100 Spiele)                     |
| `R:+3.1`   | Durchschnitts-Return                             |
| `μ=41.8`   | TrueSkill Rating                                 |
| `E/M/H`    | Win-Rate vs Easy/Medium/Hard Bot                 |
| `VT`       | V-trace Loss-Metriken                            |
| `EXP/SIL`  | Expert- und Self-Imitation-Metriken              |
| `GM`       | Golden Memory Groesse (frische Eintraege)        |

---

## Ergebnisse

Der Agent erreicht nach ca. 100 Updates:
- **100%** Win-Rate vs Easy Bot
- **60%+** Win-Rate vs Medium Bot
- **62%+** Win-Rate vs Hard Bot
- **TrueSkill μ > 40** (konservativer Skill > 35)

---

## License

MIT License. Siehe [LICENSE](LICENSE).
