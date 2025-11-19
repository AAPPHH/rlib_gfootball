import os
from pathlib import Path
import ray
from marllib import marl

def main():
    # --- 1. Setup Pfade & Ray Init (analog zum Original) ---
    workspace_root = Path(__file__).resolve().parent
    ray_tmp_dir = workspace_root / "ray_tmp"
    results_dir = workspace_root / "ray_results"
    
    ray_tmp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    os.environ["RAY_TMPDIR"] = str(ray_tmp_dir)
    os.environ["RAY_TEMP_DIR"] = str(ray_tmp_dir)

    # Ray initialisieren (MARLlib benötigt Ray)
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        num_gpus=1, # GPU nutzen
        _temp_dir=str(ray_tmp_dir),
        log_to_driver=True
    )

    # --- 2. Environment Erstellen (MARLlib High-Level API) ---
    # MARLlib hat eingebaute Wrapper für Google Football
    env = marl.make_env(
        environment_name="football",
        map_name="11_vs_11_easy_stochastic",
        force_coop=True,  # Alle 11 linken Agenten kooperieren
        # Hier können wir die Reward-Settings übergeben
        env_args={
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints",
            "number_of_left_players_agent_controls": 11,
            "number_of_right_players_agent_controls": 0,
            "stacked": True,
        }
    )

    # --- 3. Algorithmus Wahl: MAPPO (State-of-the-Art für Football) ---
    # Wir nutzen MAPPO statt PPO, da es Parameter-Sharing und Centralized Critic bietet
    algo = marl.algos.MAPPO(hyperparam_source="common")

    # --- 4. Modell Konfiguration (LSTM & Layer) ---
    # Übersetzung deiner RLLib model config zu MARLlib
    model_config = {
        "use_rnn": True,             # use_lstm in RLLib
        "rnn_type": "lstm",
        "hidden_layer_sizes": [256, 128], # fcnet_hiddens
        "rnn_hidden_size": 512,      # lstm_cell_size
        "activation": "relu",        # silu ist in MARLlib ggf. nicht standard, fallback auf relu/tanh
        "gain": 0.01,                # Init gain
    }

    # --- 5. Training Konfiguration (Hyperparameter) ---
    # Hier setzen wir deine Werte aus dem PBT-Bereich (Mittelwerte oder Startwerte)
    algo_config = {
        "num_sgd_iter": 5,
        "lr": 5e-6,
        "gamma": 0.998,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        "clip_param": 0.3,           # Ähnlich zu grad_clip / clip param
        "grad_clip_norm": 0.5,       # grad_clip
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        "batch_size": 32768,         # train_batch_size
        "mini_batch_size": 8192,     # sgd_minibatch_size
        "use_gae": True,
        "gae_lambda": 0.95,
    }

    # --- 6. Training Starten ---
    print("Starte MARLlib Training mit MAPPO...")
    
    algo.fit(
        env,
        model_config=model_config,
        algo_config=algo_config,
        stop={
            "episode_reward_mean": 20.0,
            "training_iteration": 2000,
        },
        local_mode=False,
        num_gpus=1,            # Wie viele GPUs für das Training
        num_workers=10,        # num_rollout_workers
        share_policy="all",    # Parameter Sharing für alle 11 Agenten (Standard bei 11v11)
        checkpoint_freq=5,
        checkpoint_end=True,
        local_dir=str(results_dir),
        exp_name="MAPPO_GFootball_11v11"
    )

    print("Training abgeschlossen.")
    ray.shutdown()

if __name__ == "__main__":
    main()