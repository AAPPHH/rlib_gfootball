import gfootball.env as football_env
import numpy as np
import ray
import torch
from gymnasium import spaces
# Korrekte RLlib 2.x Imports
from ray.rllib.algorithms.impala import Impala, ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from typing import Any, Dict

# --- GFootballMultiAgentEnv (unverändert) ---
class GFootballMultiAgentEnv(MultiAgentEnv):
    """Eine minimalisierte Version der GFootballMultiAgentEnv."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        default_config = {
            "env_name": "11_vs_11_stochastic", 
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints", 
            "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "stacked": True, 
            "logdir": "/tmp/gfootball", 
            "write_goal_dumps": False,
            "write_full_episode_dumps": False, 
            "render": False,
            "write_video": False, 
            "dump_frequency": 1,
        }
        self.env_config = {**default_config, **config}
        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]

        creation_kwargs = self.env_config.copy()
        creation_kwargs.pop("debug_mode", None)
        creation_kwargs.pop("_reset_render_state", None)

        try:
            _temp_env = football_env.create_environment(**creation_kwargs)
            _initial_obs_sample = _temp_env.reset()
            if isinstance(_initial_obs_sample, tuple):
                 _initial_obs_sample = _initial_obs_sample[0]

            if self.left_players + self.right_players > 0:
                single_agent_obs_shape = _initial_obs_sample.shape[1:]
            else:
                single_agent_obs_shape = _initial_obs_sample.shape
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=single_agent_obs_shape, 
                dtype=_initial_obs_sample.dtype
            )
            self.action_space = spaces.Discrete(19)
            _temp_env.close()
        except Exception as e:
            print(f"FEHLER beim Ableiten der Spaces: {e}")
            obs_shape = (4, 115) if self.env_config.get("stacked") else (115,)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )
            self.action_space = spaces.Discrete(19)

        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + \
                         [f"right_{i}" for i in range(self.right_players)]
        self._agent_ids = set(self.agent_ids)

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("Reset nicht benötigt für reinen Modelltest")

    def step(self, action_dict):
        raise NotImplementedError("Step nicht benötigt für reinen Modelltest")


def get_policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """Mapping-Funktion für Multi-Agent Setup."""
    if agent_id.startswith("left_"):
        return "policy_left"
    else:
        return "policy_right" 


# --- Haupt-Testskript (KORREKTE RLlib 2.x API) ---

if __name__ == "__main__":
    
    print("Starte Ray im lokalen Modus...")
    ray.init(local_mode=True, ignore_reinit_error=True)

    # 1. Umgebung registrieren
    register_env("gfootball_multi_test", lambda cfg: GFootballMultiAgentEnv(cfg))

    # 2. Umgebungskonfiguration
    env_config = {
        "representation": "simple115v2",
        "env_name": "academy_single_goal_versus_lazy",
        "number_of_left_players_agent_controls": 3,
        "number_of_right_players_agent_controls": 0,
        "stacked": True, 
    }

    # 3. Dummy-Env erstellen für Spaces
    dummy_env = GFootballMultiAgentEnv(env_config)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    # 4. Modellkonfiguration
    standard_model_config = {
        "use_lstm": True,
        "lstm_cell_size": 256,
        "fcnet_hiddens": [256, 256],
    }

    # 5. IMPALA-Konfiguration mit KORREKTER RLlib 2.x Builder-API
    print("\nErstelle ImpalaConfig mit Builder-Pattern (RLlib 2.x)...")
    print(f"Typ von ImpalaConfig: {type(ImpalaConfig)}")
    
    config = (
        ImpalaConfig()
        # Environment Setup
        .environment(
            env="gfootball_multi_test",
            env_config=env_config,
        )
        # Framework
        .framework("torch")
        # Resources
        .resources(
            num_gpus=0,
        )
        .rollouts(
            num_rollout_workers=0,  # Lokaler Modus
        )
        # Model Config
        .training(
            model=standard_model_config,
        )
        # Multi-Agent Setup
        .multi_agent(
            policies={
                "policy_left": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                )
            },
            policy_mapping_fn=get_policy_mapping_fn,
            policies_to_train=["policy_left"],
        )
    )

    print("\nErstelle IMPALA-Algorithmus mit .build()...")
    algo = config.build()

    # 6. Policy und Modell holen
    policy = algo.get_policy("policy_left")
    model = policy.model
    model.eval()
    
    print("\n" + "="*60)
    print("MODELL- UND SPEICHER-ÜBERPRÜFUNG")
    print("="*60)
    print(f"Observation Space: {policy.observation_space}")
    print(f"Action Space:      {policy.action_space}")
    print(f"LSTM Cell Size:    {standard_model_config['lstm_cell_size']}")
    print("-"*60)

    # 7. Dummy-Daten für Forward-Pass
    B = 1  # Batch Size
    sample_obs = policy.observation_space.sample() 
    obs_batch = torch.from_numpy(np.expand_dims(sample_obs, 0)).float()
    initial_state = model.get_initial_state()
    seq_lens = torch.tensor([1] * B, dtype=torch.int32)
    input_dict = {"obs": obs_batch}

    print("\nMODELL-INPUT-FORMATE")
    print("-"*60)
    print(f"Input 'obs' Shape:         {input_dict['obs'].shape}")
    print(f"Anzahl LSTM-State-Tensoren: {len(initial_state)}")
    if len(initial_state) >= 2:
        print(f"LSTM 'hidden_state' Shape:  {initial_state[0].shape} (B, cell_size)")
        print(f"LSTM 'cell_state' Shape:    {initial_state[1].shape} (B, cell_size)")
    print(f"Sequenzlängen Shape:        {seq_lens.shape}")

    # 8. Forward-Pass durchführen
    print("\nFühre Forward-Pass durch...")
    with torch.no_grad():
        model_out, new_state = model(input_dict, initial_state, seq_lens)
        value_out = model.value_function()

    print("\nMODELL-OUTPUT-FORMATE")
    print("-"*60)
    print(f"Output 'logits' Shape:      {model_out.shape} (B, num_actions)")
    print(f"Output 'value' Shape:       {value_out.shape} (B,)")
    if len(new_state) >= 2:
        print(f"Neuer LSTM 'hidden' Shape:  {new_state[0].shape} (B, cell_size)")
        print(f"Neuer LSTM 'cell' Shape:    {new_state[1].shape} (B, cell_size)")
    print("-"*60)

    # 9. Assertions zur Validierung
    expected_obs_shape = (4, 115)
    expected_action_dim = 19
    lstm_size = standard_model_config["lstm_cell_size"]

    print("\nValidiere Formate...")
    try:
        assert policy.observation_space.shape == expected_obs_shape, \
            f"Obs Space: erwartet {expected_obs_shape}, erhalten {policy.observation_space.shape}"
        assert policy.action_space.n == expected_action_dim, \
            f"Action Space: erwartet {expected_action_dim}, erhalten {policy.action_space.n}"
        assert model_out.shape == (B, expected_action_dim), \
            f"Model Output: erwartet {(B, expected_action_dim)}, erhalten {model_out.shape}"
        assert value_out.shape == (B,), \
            f"Value Output: erwartet {(B,)}, erhalten {value_out.shape}"
        assert initial_state[0].shape == (B, lstm_size), \
            f"Initial Hidden: erwartet {(B, lstm_size)}, erhalten {initial_state[0].shape}"
        assert new_state[0].shape == (B, lstm_size), \
            f"New Hidden: erwartet {(B, lstm_size)}, erhalten {new_state[0].shape}"
        
        print("\n" + "="*60)
        print("✅ ALLE TESTS BESTANDEN!")
        print("="*60)
        print("Alle Formate entsprechen den Erwartungen.")
        
    except AssertionError as e:
        print(f"\n❌ FEHLER: {e}")
        
    finally:
        # 10. Cleanup
        print("\nBeende Algorithmus und Ray...")
        algo.stop()
        ray.shutdown()
        print("Test abgeschlossen.")