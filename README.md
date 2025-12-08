# RLlib GFootball – Progressive Self-Play mit Mamba-Hybrid-Netz

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Ray RLlib](https://img.shields.io/badge/Ray-RLlib-5b5ce2)](https://docs.ray.io/en/latest/rllib/index.html)
[![Google Research Football](https://img.shields.io/badge/Env-GFootball-0a9d57)](https://github.com/google-research/football)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> **Kurzfassung:** Dieses Repo trainiert Google Research Football Agents mit RLlib/IMPALA, einem **Mamba+Attention Hybrid-Netz**, **Trueskill-basiertem Self-Play**, **Champion-Tracking** und **Auto-Pruning** der Policy-Historie. Zusätzlich gibt’s einen **Video-Recorder**, der Episoden in „pixels“ rendert – perfekt für Demos und Präsentationen. ⚽️🧠

---

## ✨ Features

- **Progressives Curriculum** über mehrere Stages (von *Empty Goal* bis *11v11 stochastic*).
- **Self-Play Engine** mit:
  - Trueskill-Ratings (1v1), konservativer Skill (μ−3σ)
  - **Champion-Tracking** & **geschützte Versionen**
  - **Aktive Zone** + **Top-N** + **Auto-Pruning** alter Gewichte
  - **Gegner-Sampling** nach Match-Quality (+Champion-Bias, Exploration)
- **Population Based Training (PBT)** für Hyperparameter (LR, Entropy, VF-Loss).
- **Mamba-Hybrid-Modell** (Mamba-Blöcke + Multi-Head Attention) und **Lite-Variante**.
- **Multi-Agent-Wrapper** für GFootball (Links/Rechts-Teams, flexible Spieleranzahl).
- **Video-Demos**: Trained vs. Random-Policy, direkt als MP4/Full Dumps.

---

## 📦 Projektstruktur

```
.
├─ train.py                 # Hauptskript: Curriculum, Self-Play, PBT, PolicyPool
├─ model.py                 # GFootballMambaHybrid2025 & GFootballMambaLite2025
├─ demo_record.py           # DualEnvironmentRecorder + Video-Demos
├─ requirements.txt         # (Empfohlen) – siehe Installation
└─ README.md                # Diese Datei
```

> Falls deine Dateinamen abweichen: einfach oben anpassen. Die Klasse-/Funktionsnamen sind identisch.

---

## 🚀 Installation

> **Voraussetzungen:** Python 3.10+, CUDA optional, FFmpeg (für Video-Export empfohlen)

**1) Umgebung anlegen**
```bash
conda create -n gfootball-rllib python=3.10 -y
conda activate gfootball-rllib
```

**2) Abhängigkeiten**
```bash
# PyTorch (wähle passende CUDA-Variante unter https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Ray & RLlib + Extras
pip install "ray[default]" "ray[rllib]" "ray[tune]" "ray[air]"

# Google Research Football
pip install gfootball  # nutzt vorgebaute wheels; bei Problemen siehe GFootball-Doku

# Sonstiges
pip install gymnasium numpy trueskill dataclasses-json
```

> **Windows-Hinweis:** Wenn GFootball-Wheels zicken, nutze WSL2/Ubuntu oder Docker; alternativ vorgebaute Wheels aus Community-Repos.

---

## 🏁 Schnellstart

### 1) Training progressiv starten

In `main()` werden Standard-Umgebungsvariablen gelesen:

- `GFOOTBALL_DEBUG=true|false` – Render/Video, kurze Rollouts, lokaler Modus
- `GFOOTBALL_TRANSFER=true|false` – Transfer über Stages
- `GFOOTBALL_END_STAGE` – Letzte Stage (Index)

**Beispiel (Linux/macOS):**
```bash
export GFOOTBALL_DEBUG=false
export GFOOTBALL_TRANSFER=true
# optional: export GFOOTBALL_END_STAGE=4
python train.py
```

**Beispiel (Windows PowerShell):**
```powershell
$env:GFOOTBALL_DEBUG="false"
$env:GFOOTBALL_TRANSFER="true"
# optional: $env:GFOOTBALL_END_STAGE="4"
python .	rain.py
```

> In `main()` ist ein **Beispiel-Checkpoint** hinterlegt (`initial_checkpoint`). Passe den Pfad an oder setze ihn auf `None`, um von Scratch zu starten.

### 2) TensorBoard

Nach jedem Stage-Run wird ein TensorBoard-Kommando in `training_results_transfer_pbt/tensorboard_commands.txt` ergänzt:
```bash
tensorboard --logdir "training_results_transfer_pbt/<experiment>"
```

---

## 📚 Trainings-Curriculum

| Stage | Name                   | Env                                | Left | Right | Target R | Max Steps    | Beschreibung                  |
|:-----:|------------------------|------------------------------------|:----:|:-----:|:--------:|-------------:|------------------------------|
| 1     | stage_1_basic_0        | `academy_empty_goal_close`         | 1    | 0     | 0.75     | 10,000,000   | 1 Spieler vor Tor            |
| 2     | stage_1_basic_1        | `academy_run_to_score_with_keeper` | 1    | 0     | 0.75     | 10,000,000   | 1 Spieler rennt zum Tor      |
| 3     | stage_1_basic          | `academy_pass_and_shoot_with_keeper` | 1  | 0     | 0.75     | 10,000,000   | 1 Spieler gegen Keeper       |
| 4     | stage_2_1v1            | `academy_3_vs_1_with_keeper`       | 3    | 0     | 0.75     | 20,000,000   | 1v1 (3 vs 1 Setup)           |
| 5     | stage_3_3v3            | `11_vs_11_easy_stochastic`         | 3    | 0     | 1.0      | 50,000,000   | 3v3 (easy)                   |
| 6     | stage_4_3v3            | `11_vs_11_easy_stochastic`         | 3    | 3     | 1.0      | 100,000,000  | 3v3 (easy, beide Seiten)     |
| 7     | stage_5_3v3            | `11_vs_11_stochastic`              | 3    | 3     | 1.0      | 500,000,000  | 3v3 (stochastic, schwerer)   |

> **Hinweis:** Die Stop-Kriterien nutzen `timesteps_total`. Das `target_reward` ist als Orientierung gedacht und wird (je nach Bedarf) nicht hart erzwungen.

---

## 🧠 Architektur

### Mamba-Hybrid-Netz
- **MambaBlock:** depthwise Conv-Vorverarbeitung → vereinfachte SSM-Dynamik (A/B mean pooling) → Gate → Output-Proj.
- **MultiHeadSpatialAttention** (optional in jedem zweiten Block)
- **Feed-Forward (Mish)** + Residuals/LayerNorm
- **Head-Design:** getrennte Policy/Value-Heads, orthogonale Init (NormC)

### IMPALA + RLlib
- **Framework:** PyTorch AMP (optional), Grad-Clip, große `train_batch_size`
- **PBT:** Perturbation alle 64k Steps, Mutationen auf LR/Entropy/VF-Coeff
- **Multi-Agent:** Left/Right Policies, dynamische `policies_to_train`

---

## 🥊 Self-Play & PolicyPool

- **Versioning:** Neue Versionen werden periodisch gespeichert (`version_save_interval`).
- **Trueskill-Update:** Aggregiert Episoden-Scores; Sieg/Unentschieden/Niederlage → `rate_1vs1`.
- **Champion-Schutz:** Beste konservative Skill-Version wird nie gepruned.
- **Active Set:** `keep_top_n` (nach Skill), jüngste `active_zone_size`, aktuelle Version.
- **Auto-Pruning:** Löscht Gewichte & Ratings alter Versionen (außer Schutz/Active).

Konfigurierbar in `EnhancedSelfPlayCallback(...)` sowie `PolicyPool(...)`.

---

## 🎬 Video-Demos

`demo_record.py` rendert Episoden in „pixels“ parallel zur Trainingsumgebung („simple115v2“).  
Du kannst **trainierte Checkpoints** oder **Random-Policies** aufnehmen.

**Beispiel:**
```python
USE_RANDOM_WEIGHTS = False
CHECKPOINT_PFAD = r"C:\clones\rlib_gfootball\training_results_transfer_pbt\stage_1_basic_1_20251019_191114\9e3b0_00000\checkpoint_000095"
STAGE_KEY = "s3_pass_shoot"   # siehe DEMO_STAGES
NUM_EPISODEN = 5

create_video_demo(CHECKPOINT_PFAD, STAGE_KEY, NUM_EPISODEN)
```

Ausgabe landet unter `presentation_demo_<stage>_trained_<timestamp>/videos/`.

---

## ⚙️ Nützliche Umgebungsvariablen

```bash
# Debug/Render kurz halten, lokaler Modus, 0 Runner
export GFOOTBALL_DEBUG=true

# Transfer Learning über die Curriculum-Stages
export GFOOTBALL_TRANSFER=true

# Optional: letzte Stage als Index (0-basiert)
export GFOOTBALL_END_STAGE=4
```

---

## 🧪 Troubleshooting

- **GFootball hängt/kein Render:** Stelle sicher, dass FFmpeg installiert ist; ggf. `write_video=False` setzen.
- **Windows Pfade:** Backslashes escapen oder `r"raw\strings"` nutzen (siehe Beispiele).
- **CUDA/AMP Fehler:** Setze `use_amp=False` im `custom_model_config`.
- **„Cant call step() once episode finished“ beim Video:** Wird intern abgefangen; tritt bei sehr kurzen Episoden auf.
- **Speicherverbrauch hoch:** `train_batch_size` reduzieren, `num_env_runners`/`num_envs_per_env_runner` anpassen.

---

## 📈 Konfiguration anpassen (Beispiel)

```python
config = ImpalaConfig()   .environment("gfootball_multi", env_config={
      "env_name": "11_vs_11_easy_stochastic",
      "representation": "simple115v2",
      "number_of_left_players_agent_controls": 3,
      "number_of_right_players_agent_controls": 3,
      "rewards": "scoring,checkpoints",
      "stacked": True,
  })   .framework("torch")   .env_runners(num_env_runners=5, num_envs_per_env_runner=1)   .training(
      lr=1e-4, entropy_coeff=0.008, vf_loss_coeff=0.5,
      grad_clip=0.5, train_batch_size=4096,
      model={
        "custom_model": "mamba_hybrid_2025",
        "custom_model_config": {
          "d_model": 256, "num_layers": 4, "d_state": 16,
          "num_heads": 4, "use_attention": True,
          "dropout": 0.1, "use_amp": True
        }
      },
  )
```

---

## 🤝 Danksagung

- [Google Research Football](https://github.com/google-research/football)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- Arbeiten rund um **Mamba/SSM** als Inspiration für die Hybrid-Blöcke.

---

## 📝 License

Dieses Projekt steht unter der **MIT-Lizenz**. Siehe [LICENSE](LICENSE) für Details.

---

## 📫 Kontakt

Issues & PRs sind willkommen!  
Wenn du Benchmarks, Logs oder Videos teilen magst: gerne verlinken 👇

- TensorBoard: siehe `training_results_transfer_pbt/tensorboard_commands.txt`
- Videos: Ordner `presentation_demo_*/videos/`

---

Viel Spaß beim Trainieren & Kicken! ⚽️🔥