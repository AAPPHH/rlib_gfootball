import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from gymnasium import spaces

# Annahme: Deine ModularStudentModel-Klasse ist in 'modular_models.py'
# (Basierend auf deinem letzten Prompt)
try:
    from modular_models import ModularStudentModel
except ImportError:
    print("="*80)
    print("FEHLER: Konnte 'ModularStudentModel' nicht importieren.")
    print("Stelle sicher, dass die Datei 'modular_models.py' im selben Ordner")
    print("oder im Python-Pfad (C:/clones/rlib_gfootball/cold_start) liegt.")
    print("="*80)
    raise

class PretrainedGFootballModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # --- HIER IST DIE NEUE LOGIK ---
        
        # 1. Hole die Kontrollvariable aus train.py
        custom_config = model_config.get("custom_model_config", {})
        # `load_weights` ist der Name, den wir in create_impala_config definiert haben
        should_load_weights = custom_config.get("load_weights", True) 

        pretrained_path = Path("C:/clones/rlib_gfootball/cold_start/final_model_training/best_model.pth")
        config = None
        checkpoint = None

        # 2. Versuche immer, die ARCHITEKTUR-Config aus der Datei zu laden
        if pretrained_path.exists():
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            config = checkpoint.get('config')
            if config:
                print(f"✅ Konfiguration geladen von: {pretrained_path}")
                print(f"   Modell-Typ: {config.get('encoder_type', 'N/A')}-{config.get('sequence_type', 'N/A')}-{config.get('head_type', 'N/A')}")
            else:
                print(f"⚠️ Checkpoint gefunden, aber keinen 'config'-Schlüssel darin. Nutze Fallback.")
        
        # 3. Wenn keine Config geladen wurde (Datei fehlt oder Checkpoint alt), nutze Fallback
        if config is None:
            print("⚠️ Keine Config-Datei gefunden, nutze harte Fallback-Config.")
            config = {
                'encoder_type': 'linear', 'sequence_type': 'gru', 'head_type': 'mlp',
                'model_scale': 'm', 'n_features': 115, 'stack_frames': 4,
                'num_frames': 4, 'encoder_output_dim': 48, 'sequence_hidden_dim': 48,
                'encoder_hidden_dim': 256, 'prev_action_emb_dim': 8,
                'policy_hidden_dims': [32, 16], 'value_hidden_dims': [24, 12],
                'head_dropout': 0.1, 'cnn_channels': [32, 64], 'cnn_kernels': [8, 4],
                'gnn_hidden': 24, 'gnn_layers': 2, 'tcn_channels': [48, 48],
                'tcn_kernel_size': 3, 'mamba_state_dim': 6, 'kan_grid': 3,
            }
        
        # 4. Erstelle das Modell mit der geladenen oder Fallback-Config
        self.base_model = ModularStudentModel(obs_space, action_space, num_outputs, config)
        
        # 5. Lade die Gewichte NUR, wenn die Datei existiert UND should_load_weights=True ist
        if checkpoint and should_load_weights:
            print("Versuche, vortrainierte Gewichte zu laden...")
            try:
                state_dict = checkpoint['model_state_dict']
                self.base_model.load_state_dict(state_dict, strict=False)
                print("✅ Vortrainierte Gewichte erfolgreich geladen!")
            except Exception as e:
                print(f"⚠️ Gewichte konnten nicht geladen werden: {e}. Nutze zufällige Gewichte.")
        elif not checkpoint and should_load_weights:
            print("⚠️ `load_weights=True` gesetzt, aber keine Checkpoint-Datei gefunden. Nutze zufällige Gewichte.")
        else:
            # Dies trifft zu, wenn should_load_weights=False ist
            print("📝 Initialisiere mit zufälligen Gewichten (load_weights=False).")

        # --- ENDE DER NEUEN LOGIK ---

        self._last_batch_size = None
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        
        if "prev_actions" in input_dict:
            prev_actions = input_dict["prev_actions"]
        else:
            prev_actions = torch.zeros(obs.size(0), dtype=torch.long, device=obs.device)
        
        model_input = {
            "obs_flat": obs,
            "prev_actions": prev_actions
        }
        
        logits, new_state = self.base_model(model_input, state, seq_lens)
        self._value = self.base_model._value_out
        
        return logits, new_state
    
    @override(TorchModelV2)
    def value_function(self):
        return self._value
    
    @override(TorchModelV2) 
    def get_initial_state(self):
        batch_size = 1
        device = next(self.parameters()).device
        state = self.base_model.get_initial_state(batch_size, device)
        
        if state and state[0] is not None:
            if isinstance(state[0], tuple): # LSTM (h, c)
                h, c = state[0]
                # Squeeze die Batch-Dimension (RLlib erwartet [hidden_size])
                return [h.squeeze(0).contiguous(), c.squeeze(0).contiguous()]
            else: # GRU (h)
                return [state[0].squeeze(0).contiguous()]
        return []