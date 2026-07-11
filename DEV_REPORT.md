# Bericht: Konvergenz-Fixes auf der dev-Branch

**Datum:** 2026-07-11 · **Basis:** dev (`2372476`)

---

## Session-Zusammenfassung (was in dieser Session gemacht wurde)

1. **Diagnose** des Konvergenz-Einbruchs auf dev via Branch-Diff main↔dev
   (Abschnitt 2) plus SOTA-Recherche (TiZero, Kaggle-Gewinner) als
   Referenzrahmen (Quellen am Ende).
2. **`main/full_league.py` repariert** (Abschnitte 3.1–3.5): Gegnerauswahl
   mit Lernfront-Gewichtung und 50 % Challenge-Self-Play, Hyperparameter
   zurück auf konvergenzfähige Werte, Rollouts von vollen Episoden auf
   512er-Chunks, Expert-Buffer auf 512er-Fenster, diverse Bugfixes
   (lexikografische Snapshot-Sortierung, Draw-Zählung).
3. **Reward-Entscheidung:** bleibt **scoring only** (Abschnitt 3.4) — ein
   testweise eingebautes Checkpoint-Curriculum wurde wieder entfernt.
4. **BC-Warmstart repariert** (Abschnitt 3.6): `weckick_im.py` nutzt jetzt
   das Netz aus `full_league.py` (main-Muster statt Duplikat-Definition),
   `bc_warmstart_v2.pt` neu trainiert — **Val-Accuracy 46.8 %**, lädt
   `strict=True` ins League-Netz. dev ist damit startklar.
5. **Aufgeräumt** (Abschnitt 3.7): 29 nicht mehr relevante Experiment-Files
   gelöscht (Alt-Architekturen, Scratch-Tests, SLURM-Skripte).

---

## 1. Ausgangslage

- **main** konvergierte: ~62 % Winrate vs. Hard-Bot, TrueSkill μ > 40.
- **dev** (der eigentlich ausgefeiltere Umbau: PFSP, Policy-Fingerprints, GPU-Buffer,
  Goal-Trajectories) konvergierte plötzlich nicht mehr.

Wichtig für die Einordnung: main startete vom RL-Checkpoint `ckpt_u900.pt`
(900 Updates Self-Play-Vortraining), dev startet vom BC-Warmstart
(~47 % Val-Accuracy). Ein Teil von mains „guter Konvergenz"
war schlicht geerbter Fortschritt — dev muss sich diese Rampe selbst bauen und
war gleichzeitig durch die unten beschriebenen Regressionen blockiert.

## 2. Diagnose — warum dev nicht konvergierte

| # | Ursache | Effekt |
|---|---------|--------|
| 1 | `bot_floor=1.0` im `__main__` (Debug-Überbleibsel) | PFSP-Zweig war toter Code → **nie Self-Play**, nur Bot-Spiele |
| 2 | Bot-Gewichtung `(bot_score)^0.5` | Bevorzugte Gegner, die den Agenten **schlagen** → Agent farmte Niederlagen vs. medium/hard → bei `rewards="scoring"` (±1 nur bei Toren) praktisch kein Lernsignal |
| 3 | `entropy_coeff=0.05` (main: 0.01) | Bei sparsem Reward dominiert der Entropie-Bonus den Policy-Gradient → BC-Policy erodiert Richtung Uniform-Random |
| 4 | `gamma=0.9997` (main: 0.997) | Quasi undiskontierte Value-Targets über 3000-Step-Episoden → extrem verrauschtes V-trace |
| 5 | Volle Episoden (~3000 Steps) statt 512er-Rollouts | LSTM-BPTT über 3000 Zeitschritte unter fp16 → schlechte Gradientenqualität; effektive Batchgröße sank von 128×512 auf 24 Episoden |

Kombinierter Mechanismus: Der Agent spielte fast nur Spiele, die er verliert
(1+2), bekam daraus kein positives Signal (scoring-only), und der verbleibende
Gradient war Entropie (3), die die Policy zufällig machte — sichtbar daran,
dass `H:` im Log nicht fällt.

## 3. Änderungen und Begründung

### 3.1 Gegnerauswahl (`PureLeague.select_opponent`)

**Vorher:** 100 % Bots (wg. `bot_floor=1.0`), gewichtet auf die härtesten Gegner.

**Nachher:** drei Zweige —

| Anteil | Zweig | Logik |
|-------:|-------|-------|
| 25 % | Bots | Gewicht `p·(1−p) + 0.1`, p = Agent-Score vs. Bot |
| 50 % | Neuester Snapshot | „Challenge Self-Play" — Gegner ist die jüngste eigene Version |
| 25 % | PFSP über alle Mitglieder | Gewicht `p·(1−p) + 0.05`, p = Mix aus TrueSkill-Winprob und empirischer Winrate; Neue-Mitglieder-Boost ×1.5 bleibt |

**Warum `p·(1−p)`:** Das Gewicht ist maximal bei ~50 % Winrate — der Agent
spielt gegen seine *Lernfront*. Beherrschte Gegner (p→1) und aussichtslose
Gegner (p→0) fallen auf den Floor. Das erzeugt ein emergentes Curriculum:
erst easy/medium, mit steigender Stärke automatisch medium/hard.
(Referenz: TiZero spielt 80 % gegen die neuesten Agenten; SaltyFish 70 %
gegen den neuesten Snapshot; Schwellen-basierter Aufstieg statt „immer der
Härteste".)

**Bugfix nebenbei:** „Neuester Snapshot" wurde bisher lexikografisch über den
Namen bestimmt (`snap_u1000_… < snap_u200_…`). Snapshots tragen jetzt einen
`created_idx` (persistiert in `league.pkl`); auch das Snapshot-Pruning
(`_maybe_prune_snapshots`, `_ensure_skill_spacing`) nutzt ihn.

### 3.2 Hyperparameter (`__main__`)

| Parameter | Vorher | Nachher | Begründung |
|-----------|-------:|--------:|------------|
| `bot_floor` | 1.0 | 0.25 | Self-Play wieder aktiv; Bots bleiben als Anker/Eval |
| `entropy_coeff` | 0.05 | 0.01 | mains Wert; auch TiZero nutzt 0.01. 0.05 zerstörte die BC-Policy |
| `gamma` | 0.9997 | 0.999 | TiZero-Wert; mit 512er-Bootstrap-Rollouts stabil (Horizont ~1000 Steps) |
| `value_coeff` | 0.5 | 0.2 | TiZero-Wert; weniger Value-Dominanz im Loss |
| `lr` | 3e-4 | 1e-4 | TiZero-Wert; konservativer bei größerem Batch |
| `batch_size` | 24 Episoden | 64 × 512 = 32k Steps | wieder in der Größenordnung von main (65k) |
| `checkpoint_dir` | v76 | **v77** | sauberer Neustart — v76-Liga entstand unter den kaputten Settings |

### 3.3 Rollout-Struktur (Worker + Learner)

**Vorher:** 1 `collect()` = 1 volle Episode (~3000 Steps), Training per
Full-Episode-BPTT, `bootstrap=0`.

**Nachher:** 1 `collect()` = 512 Steps. Der Worker hält den Episoden-Zustand
über Chunk-Grenzen (Obs, LSTM-Hidden, Score, Rolling-Buffer für
Goal-Trajectories) und liefert am Chunk-Ende einen Bootstrap-Value
`V(s_letzt)` für V-trace. Ein Rollout kann 0–n abgeschlossene Episoden
enthalten (`episodes`-Liste); der Train-Loop meldet jede einzeln an die Liga.

**Warum:** BPTT über 3000 Steps ist weder nötig noch üblich (TiZero:
Rollout ~500; main: 512) und war unter fp16-Autocast ein Stabilitätsrisiko.
Zudem sinkt die Policy-Lag (Worker bekommen nach jedem Chunk neue Gewichte,
nicht erst nach jeder Episode) — besser für V-trace.

Konsequent dazu: `ExpertBufferGPU` sampelt jetzt 512er-**Fenster** statt
kompletter Episoden (gleiche Semantik wie mains chunked ExpertBuffer:
abgeschnittene MC-Returns wirken durch den SIL-Clamp `max(R−V, 0)` nur
konservativ). Golden-Memory-Trajektorien waren schon ≤128 Steps — unverändert.

### 3.4 Reward: bleibt **scoring only** (bewusste Entscheidung)

Checkpoints waren in diesem Projekt in *allen* Setups aus (main, dev,
`impala_mlp_only`, `impala_mlp_ciriculum`). Das dichte Lernsignal kommt
stattdessen aus BC-Warmstart + Expert-SIL + Golden Memory. Ein
Checkpoint-Reward-Curriculum für die Bot-Phase (SaltyFish-Rezept) wurde
zwischenzeitlich implementiert und **auf Entscheidung wieder entfernt**, um
das bestehende Reward-Design nicht zu verändern. Die generische
`rewards`-Durchreichung im Worker (`_create_env`/`collect`, Default
`"scoring"`) ist geblieben — ohne Verhaltensänderung, aber künftig per
Parameter nutzbar.

### 3.5 Kleinigkeiten

- `print_ranking`: `agent_wr` zählte Draws als Siege → korrigiert
  (`1 − wins − draws`).
- Zwei Dispatch-Codepfade im Train-Loop zu `_dispatch()` zusammengefasst.

### 3.6 BC-Warmstart repariert (`main/weckick_im.py`)

Der Kaltstart war gebrochen: Das alte `bc_warmstart.pt` (17.02.) stammte von
mains Architektur (2-Layer-Encoder, Policy 512→128→19, PopArt auf 512) und
passte nicht mehr zu devs Netz — `strict=False` fängt nur fehlende Keys ab,
nicht Shape-Mismatches, also warf `full_league.py` beim Start einen
RuntimeError. Ursache: devs Architektur-Umbau pflegte in `weckick_im.py` eine
**Duplikat-Netzdefinition**, die nie zusammen mit dem BC-Checkpoint erneuert
wurde. Auf main konnte das nicht passieren, weil `bc_train.py` dasselbe Netz
aus `net.py` importiert.

**Fix (main-Muster übernommen):**

- `weckick_im.py` importiert `Net`/`FeatureEngineer`/Konstanten aus
  `full_league.py` — nur noch **eine** Netzdefinition im Projekt.
- BC trainiert das volle League-Netz (Loss nur auf den Policy-Logits,
  Value-Heads bleiben initialisiert) → Checkpoint ist `strict=True`-kompatibel.
- Neues Artefakt `bc_warmstart_v2.pt` (altes File bleibt unangetastet);
  `full_league.py` zeigt auf v2.
- Bugfix: Save-Guard `val_acc > best_val_acc` (Init 0) speicherte bei
  val_acc = 0.0 nie ein Checkpoint → Init jetzt −1.

**Ergebnis:** Neu trainiert auf `expert.parquet` (222k Samples, 20 Epochen):
**Val-Accuracy 46.8 %** (altes Checkpoint: 46.6 %) — gleichwertige Qualität
auf der korrekten Architektur, `strict=True`-Load verifiziert.

### 3.7 Aufräumen: gelöschte Alt-Experimente

Nicht mehr relevante Experiment-Files entfernt (auf main waren die
SLURM-Skripte bereits gelöscht, dev hinkte hinterher):

- `main/grif_im.py` (Griffin/RG-LRU-Architektur-Experiment)
- `main/impala_mlp_ciriculum.py`, `main/impala_mlp_only.py`
  (alte Einzel-Agent-/Curriculum-Experimente, alte 128er-Architektur)
- `sanity_checks/` (22 Scratch-Tests, u. a. `im_test.py` mit alter Architektur)
- `start.sh`, `start_mappo.sh`, `start_mobile.sh` (SLURM)
- `tensorboard_parser.py` (Auswertung alter `logs_vtrace`-Runs)

Aktive Pipeline danach: `main/full_league.py` (Training),
`main/weckick_im.py` (BC-Warmstart), `videos/` (Rendering).

## 4. Bewusst nicht angefasst

- **Goal-Trajectory-Golden-Memory** (SIL nur auf Tor-Sequenzen) — dev-Design,
  kein Bug; funktioniert jetzt sogar über Chunk-Grenzen.
- **Policy-Fingerprints** für Snapshot-Diversität — deckt sich mit der
  Literatur („Diversity is Strength", arXiv 2306.15903).
- **BC-Warmstart** statt RL-Checkpoint — gewollter Neuanfang; für einen fairen
  Vergleich mit main im Kopf behalten.
- Expert-SIL-Gate (aus ab 70 % WR vs. hard), `sil_coeff`, Netz-Architektur.

## 5. Woran man Konvergenz im Log erkennt

```
[  10] 0.3M ... | W/D/L:.. | μ=.. | E:../../..(n) M:.. H:.. | VT ... H:1.85 ...
```

1. **`H:` (Policy-Entropie) muss fallen** (BC-Start je nach Schärfe ~1.5–2.5,
   Richtung <1.0). Steigt sie gegen ln(19)≈2.9 → Policy zerfällt (altes Symptom).
2. **`E:`-Winrate (easy)** muss in den ersten ~50 Updates klar steigen —
   schnellster Indikator.
3. Startzeile prüfen: `Batch: 64 x 512 = 32,768 samples/update`,
   `bot_floor=25% latest_snap=50%`.
4. `μ` (TrueSkill) soll steigen, Snapshots sollen entstehen
   (`📸 Snapshot: snap_u…`), und ab dem ersten Snapshot tauchen
   Self-Play-Spiele im Mix auf.

## 6. Risiken / nächste Schritte

- **Stiller Start ohne Warmstart:** Existiert das `warmstart_path`-File
  nicht, startet der Learner kommentarlos from scratch (kein Fehler, keine
  Warnung). Nach Netz-Umbauten daran denken, `weckick_im.py` neu laufen zu
  lassen — durch den Import aus `full_league.py` (3.6) kann die Architektur
  selbst aber nicht mehr auseinanderlaufen.
- **Sparse Reward + schwacher BC-Start:** Ohne Checkpoint-Reward hängt der
  frühe Fortschritt an Expert-SIL/Golden-Memory. Wenn nach ~100 Updates die
  Easy-Winrate nicht steigt, ist das Reward-Curriculum (Abschnitt 3.4) der
  erste Kandidat zum Reaktivieren — die Plumbing dafür ist noch da.
- **γ=0.999 vs. 0.997:** Falls Value-Loss (`v:`) nicht sinkt, auf mains 0.997
  zurückgehen.
- **Fairer Vergleich:** dev vs. main nur mit gleichem Warmstart bewerten.
- Optional (SOTA, bewusst zurückgestellt): Action-Masking nach
  `ball_owned_team` — größter Einzel-Beschleuniger bei den
  Kaggle-Top-Teams.

## Quellen (SOTA-Referenzwerte)

- TiZero (arXiv 2302.07515): γ=0.999, ent=0.01, lr=1e-4, Rollout 500,
  value 0.2, 80 % Self-Play vs. neueste Agenten, adaptives Curriculum.
- SaltyFish (Kaggle GRF, Platz 2): IMPALA, PVE→Self-Play, 70 % neuester
  Snapshot, Checkpoint-Reward nur in PVE-Phase.
- TiKick (arXiv 2110.04507): BC/Offline-RL aus Expert-Demos als Warmstart.
- Empirische GRF-Studie (arXiv 2305.09458): Hard-Bot from scratch in ~2M Steps
  schaffbar → kein Rechenleistungsproblem.
