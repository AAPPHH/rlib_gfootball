import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from modular_models import ModularStudentModel, count_parameters, MODEL_SCALES
from gymnasium.spaces import Box, Discrete
from gymnasium import spaces # Nötig für die 2D Obs-Space-Definition
from torch.utils.data import Dataset, DataLoader
import duckdb
import pyarrow.parquet as pq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)



# --- Pfade aus Skript 1 ---
DUCKLAKE_PATH = r"C:\clones\rlib_gfootball\cold_start\ducklake\replay_lake.duckdb"
PREDICTIONS_PATH = r"C:\clones\rlib_gfootball\cold_start\xgb_predictions\predictions.parquet"
OUTPUT_DIR = Path("C:/clones/rlib_gfootball/cold_start/architecture_search_scaled")


# --- Klassen aus Skript 1 ---

class DuckLakePredictionDataset(Dataset):
    def __init__(self, ducklake_path: str, predictions_path: str, 
                 stack_frames: int = 4, indices: Optional[np.ndarray] = None):
        self.ducklake_path = Path(ducklake_path)
        self.predictions_path = Path(predictions_path)
        self.stack_frames = stack_frames
        
        pred_table = pq.read_table(self.predictions_path)
        self.pred_df = pred_table.to_pandas()
        
        if indices is not None:
            self.pred_df = self.pred_df[self.pred_df['sample_idx'].isin(indices)]
        
        self.pred_df = self.pred_df.sort_values('sample_idx').reset_index(drop=True)
        
        prob_cols = [f'prob_class_{i}' for i in range(19)]
        self.soft_targets = self.pred_df[prob_cols].values.astype(np.float32)
        self.true_labels = self.pred_df['true_label'].values.astype(np.int64)
        self.sample_indices = self.pred_df['sample_idx'].values
        
        self._load_features()
    
    def _load_features(self):
        conn = duckdb.connect(str(self.ducklake_path), read_only=True)
        conn.execute("SET threads TO 16")
        
        try:
            obs_table = "observations_stacked" if self.stack_frames > 1 else "observations"
            
            # Verwenden einer temporären Tabelle für Indizes ist effizienter
            conn.execute(f"""
                CREATE TEMP TABLE selected_indices AS 
                SELECT * FROM (VALUES {','.join([f'({i})' for i in self.sample_indices])}) AS t(idx)
            """)
            
            query = f"""
                SELECT o.* EXCLUDE (global_idx, replay_id, step)
                FROM {obs_table} o
                JOIN selected_indices s ON o.global_idx = s.idx
                ORDER BY o.global_idx
            """
            
            features_df = conn.execute(query).df()
            self.features = features_df.values.astype(np.float32)
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
            
        finally:
            conn.close()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.soft_targets[idx]),
            self.true_labels[idx]
        )


class QuickDistillationTrainer:
    """
    Trainer-Klasse für die Knowledge Distillation.
    """
    def __init__(self, student_model: nn.Module, device: str = 'cuda', 
                 temperature: float = 2.0, alpha: float = 0.7):
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.student_model = student_model.to(device)
    
    def distillation_loss(self, student_logits, teacher_probs, true_labels):
        hard_loss = F.cross_entropy(student_logits, true_labels)
        
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_probs / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        accuracy = (student_logits.argmax(dim=-1) == true_labels).float().mean()
        
        return total_loss, accuracy.item()
    
    def train_epoch(self, loader, optimizer):
        self.student_model.train()
        total_acc = 0.0
        
        for features, soft_targets, true_labels in loader:
            features = features.to(self.device)
            soft_targets = soft_targets.to(self.device)
            true_labels = true_labels.to(self.device).long()
            
            # Eingabe-Dict, wie von ModularStudentModel (basierend auf Skript 1) erwartet
            input_dict = {
                "obs_flat": features,
                "prev_actions": torch.zeros(features.size(0), dtype=torch.long, device=self.device)
            }
            
            batch_size = features.size(0)
            state = self.student_model.get_initial_state(batch_size, self.device)
            
            logits, _ = self.student_model(
                input_dict, state, 
                torch.ones(features.size(0), dtype=torch.long, device=self.device)
            )
            
            loss, acc = self.distillation_loss(logits, soft_targets, true_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            optimizer.step()
            
            total_acc += acc
        
        return total_acc / len(loader)
    
    def validate(self, loader):
        self.student_model.eval()
        total_acc = 0.0
        
        with torch.no_grad():
            for features, soft_targets, true_labels in loader:
                features = features.to(self.device)
                soft_targets = soft_targets.to(self.device)
                true_labels = true_labels.to(self.device).long()
                
                input_dict = {
                    "obs_flat": features,
                    "prev_actions": torch.zeros(features.size(0), dtype=torch.long, device=self.device)
                }
                
                batch_size = features.size(0)
                state = self.student_model.get_initial_state(batch_size, self.device)
                
                logits, _ = self.student_model(
                    input_dict, state,
                    torch.ones(features.size(0), dtype=torch.long, device=self.device)
                )
                
                _, acc = self.distillation_loss(logits, soft_targets, true_labels)
                total_acc += acc
        
        return total_acc / len(loader)

def train_architecture(config: Dict, train_loader: DataLoader, val_loader: DataLoader, 
                       epochs: int = 5) -> Tuple[float, int, float]:

    obs_space = spaces.Box(-np.inf, np.inf, 
                           (config['stack_frames'], config['n_features']), 
                           dtype=np.float32)
    action_space = Discrete(19)
    model = ModularStudentModel(obs_space, action_space, 19, config).to(device)
    
    num_params = count_parameters(model)
    
    trainer = QuickDistillationTrainer(model, device, config['temperature'], config['alpha'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config['weight_decay'])
    
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_acc = trainer.train_epoch(train_loader, optimizer)
        val_acc = trainer.validate(val_loader)
        best_val_acc = max(best_val_acc, val_acc)
        
    training_time = time.time() - start_time
    
    return best_val_acc, num_params, training_time

def run_architecture_search(num_samples=20000, train_split=0.8, epochs_per_arch=5):
    logger.info("=" * 70)
    logger.info("Architecture Search mit DuckLake Loader & Distillation")
    logger.info("=" * 70)
    
    logger.info("Lade Indizes aus Parquet-Datei...")
    pred_table = pq.read_table(PREDICTIONS_PATH, columns=['sample_idx'])
    all_indices = pred_table['sample_idx'].to_numpy()
    
    if num_samples < len(all_indices):
        all_indices = np.random.choice(all_indices, num_samples, replace=False)
    
    n_train = int(len(all_indices) * train_split)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    
    logger.info(f"Erstelle Datasets... (Verwende {num_samples} Samples: {len(train_indices)} train, {len(val_indices)} val)")
    
    train_dataset = DuckLakePredictionDataset(DUCKLAKE_PATH, PREDICTIONS_PATH, stack_frames=2, indices=train_indices)
    val_dataset = DuckLakePredictionDataset(DUCKLAKE_PATH, PREDICTIONS_PATH, stack_frames=2, indices=val_indices)
    
    logger.info("Erstelle DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    
    logger.info("Daten-Setup abgeschlossen.")
 
    encoders = ['linear', 'cnn', 'gnn']
    sequences = ['gru', 'lstm', 'tcn', 'mamba']
    heads = ['mlp', 'kan']
    model_sizes = ['s', 'm', 'l', 'xl', 'xxl']
    
    architectures = []
    for encoder in encoders:
        for seq in sequences:
            for head in heads:
                for size in model_sizes:
                    architectures.append({
                        'encoder_type': encoder,
                        'sequence_type': seq,
                        'head_type': head,
                        'model_scale': size,
                        'name': f"{encoder}_{seq}_{head}_{size}"
                    })
    
    logger.info(f"\nTeste {len(architectures)} Architektur-Kombinationen...")
    logger.info(f"Jede wird für {epochs_per_arch} Epochen trainiert\n")
    
    results = []
    
    for idx, config in enumerate(architectures, 1):
        # Zusätzliche Konfigurationsparameter (aus Skript 1 und 2 kombiniert)
        full_config = {
            **config,
            # Distillation-Parameter (aus Skript 1)
            'temperature': 2.0,
            'alpha': 0.7,
            'weight_decay': 1e-4,
            
            # Feature-Parameter (aus Skript 1)
            'n_features': 115, # WICHTIG: Annahme basierend auf Skript 1
            'stack_frames': 4,
            'num_frames': 4, # Wird evtl. von 'stack_frames' überschrieben, aber zur Sicherheit
            
            # Architektur-Parameter (aus Skript 2)
            'encoder_output_dim': 48,
            'sequence_hidden_dim': 48,
            'encoder_hidden_dim': 256,
            'prev_action_emb_dim': 8,
            'policy_hidden_dims': [32, 16],
            'value_hidden_dims': [24, 12],
            'head_dropout': 0.1,
            # CNN
            'cnn_channels': [32, 64],
            'cnn_kernels': [8, 4],
            # GNN
            'gnn_hidden': 24,
            'gnn_layers': 2,
            # TCN
            'tcn_channels': [48, 48],
            'tcn_kernel_size': 3,
            # Mamba
            'mamba_state_dim': 6,
            # KAN
            'kan_grid': 3,
        }
        
        logger.info(f"[{idx}/{len(architectures)}] Teste: {config['name']}")
        
        try:
            # Rufe die *neue* train_architecture Funktion auf
            best_val_acc, params, train_time = train_architecture(
                full_config, train_loader, val_loader, epochs=epochs_per_arch
            )
            
            results.append({
                'name': config['name'],
                'encoder': config['encoder_type'],
                'sequence': config['sequence_type'],
                'head': config['head_type'],
                'size': config['model_scale'],
                'val_acc': best_val_acc, # Verwende die beste Validierungs-Acc
                'params': params,
                'params_m': params / 1e6,
                'train_time': train_time
            })
            
            logger.info(f"  Beste Val Acc: {best_val_acc:.4f} | Params: {params/1e6:.2f}M | Time: {train_time:.1f}s")
            
        except Exception as e:
            logger.error(f"  Fehlgeschlagen: {str(e)}", exc_info=True) # Detaillierterer Error-Log
            results.append({
                'name': config['name'],
                'encoder': config['encoder_type'],
                'sequence': config.get('sequence_type', 'N/A'),
                'head': config.get('head_type', 'N/A'),
                'size': config.get('model_scale', 'N/A'),
                'val_acc': 0.0,
                'params': 0, 'params_m': 0, 'train_time': 0
            })
        
        # Speicherbereinigung (aus Skript 1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Sortiere nach Genauigkeit
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    # Drucke Top-Ergebnisse
    logger.info("\n" + "=" * 70)
    logger.info("Top 10 Architekturen:")
    logger.info("=" * 70)
    
    for i, res in enumerate(results[:10], 1):
        logger.info(f"{i:2}. {res['name']:20s} | Acc: {res['val_acc']:.4f} | Params: {res['params_m']:.2f}M")
    
    # Analysiere nach Komponente und Größe
    logger.info("\nBeste nach Komponente und Größe:")
    logger.info("-" * 40)
    
    # (Restliche Analyse-Logik bleibt gleich)
    for component_type in ['Encoder', 'Sequence Model', 'Head']:
        logger.info(f"\n{component_type} Performance nach Größe:")
        component_key = component_type.split(' ')[0].lower()
        component_list = locals()[component_key + 's'] # Holt 'encoders', 'sequences', 'heads'
        
        for size in model_sizes:
            logger.info(f"\n{size.upper()} Modelle:")
            comp_results = {}
            for res in results:
                if res['size'] == size:
                    comp = res[component_key]
                    if comp not in comp_results:
                        comp_results[comp] = []
                    if res['val_acc'] > 0: # Ignoriere fehlgeschlagene Läufe
                        comp_results[comp].append(res['val_acc'])
            
            # Berechne Durchschnitt und sortiere
            avg_comp_results = []
            for comp in component_list:
                if comp in comp_results and comp_results[comp]:
                    avg_acc = np.mean(comp_results[comp])
                    avg_comp_results.append((comp, avg_acc))
            
            avg_comp_results.sort(key=lambda x: x[1], reverse=True)
            for comp, avg_acc in avg_comp_results:
                logger.info(f"  {comp:8s}: {avg_acc:.4f} (avg. over {len(comp_results[comp])} runs)")
    
    # Speichere Ergebnisse
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Erstelle Visualisierungen
    create_visualizations(results, OUTPUT_DIR)
    
    # Finde beste Gesamtarchitektur
    best = results[0]
    logger.info(f"\nBeste Architektur: {best['name']} mit {best['val_acc']:.4f} Genauigkeit")
    logger.info(f"Parameter: {best['params_m']:.2f}M")
    
    # Finde beste Parametereffizienz
    for res in results:
        if res['params_m'] > 0:
            res['efficiency'] = res['val_acc'] / res['params_m']
        else:
            res['efficiency'] = 0
    
    results.sort(key=lambda x: x['efficiency'], reverse=True)
    logger.info(f"\nParametereffizienteste: {results[0]['name']}")
    logger.info(f"Effizienz: {results[0]['efficiency']:.4f} acc/M params")
    
    return results

def create_visualizations(results: List[Dict], output_dir: Path):
    """Erstellt Visualisierungs-Plots für die Ergebnisse."""
    
    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("Keine Ergebnisse zum Visualisieren vorhanden.")
        return

    # Filtere fehlgeschlagene Läufe für Plots
    df = df[df['val_acc'] > 0].copy()
    if df.empty:
        logger.warning("Keine *erfolgreichen* Ergebnisse zum Visualisieren vorhanden.")
        return

    # Setze Stil
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    size_order = ['s', 'm', 'l']
    
    # 1. Genauigkeit nach Modellgröße
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        size_data = df.groupby('size')['val_acc'].agg(['mean', 'std']).reindex(size_order).reset_index()
        
        ax.bar(size_data['size'], size_data['mean'], yerr=size_data['std'], capsize=10, color=sns.color_palette("husl", 3))
        ax.set_xlabel('Modellgröße')
        ax.set_ylabel('Validierungsgenauigkeit')
        ax.set_title('Genauigkeit nach Modellgröße')
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_size.png', dpi=150)
        plt.close()
    except Exception as e:
        logger.error(f"Fehler bei Plot 1 (Acc by Size): {e}")

    # 2. Genauigkeit vs. Parameter Scatter-Plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        palette = dict(zip(size_order, sns.color_palette("husl", 3)))
        
        sns.scatterplot(data=df, x='params_m', y='val_acc', hue='size', 
                        style='encoder', size='sequence', 
                        hue_order=size_order, palette=palette,
                        ax=ax, s=150, alpha=0.8)
        
        ax.set_xlabel('Parameter (Millionen)')
        ax.set_ylabel('Validierungsgenauigkeit')
        ax.set_title('Genauigkeit vs. Modellgröße (Stil=Encoder, Größe=Sequenz)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_vs_params_detailed.png', dpi=150)
        plt.close()
    except Exception as e:
        logger.error(f"Fehler bei Plot 2 (Acc vs Params): {e}")

    # 3. Heatmap der durchschnittlichen Genauigkeit
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        if len(axes.shape) == 0: axes = [axes] # Fallback für 1 Subplot
        
        for idx, size in enumerate(size_order):
            if size not in df['size'].values: continue
            size_df = df[df['size'] == size]
            pivot_table = size_df.pivot_table(
                values='val_acc', 
                index=['encoder', 'head'], 
                columns='sequence', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                        ax=axes[idx], cbar_kws={'label': 'Val Accuracy'}, annot_kws={"size": 8})
            axes[idx].set_title(f'Größe {size.upper()}')
            axes[idx].set_ylabel('Encoder + Head')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'architecture_heatmap.png', dpi=150)
        plt.close()
    except Exception as e:
        logger.error(f"Fehler bei Plot 3 (Heatmap): {e}")

    # 4. Trainingszeit-Vergleich
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        time_data = df.groupby('size')['train_time'].mean().reindex(size_order).reset_index()
        
        ax.bar(time_data['size'], time_data['train_time'], color=sns.color_palette("husl", 3))
        ax.set_xlabel('Modellgröße')
        ax.set_ylabel('Trainingszeit (Sekunden)')
        ax.set_title('Durchschnittliche Trainingszeit nach Modellgröße')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_time.png', dpi=150)
        plt.close()
    except Exception as e:
        logger.error(f"Fehler bei Plot 4 (Trainingszeit): {e}")
    
    logger.info(f"\nVisualisierungen gespeichert in: {output_dir}")


if __name__ == "__main__":

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    results = run_architecture_search(
        num_samples=20_000, 
        epochs_per_arch=5
    )
    
    # Finale Zusammenfassung
    logger.info("\n" + "=" * 70)
    logger.info("ZUSAMMENFASSUNG")
    logger.info("=" * 70)
    
    df = pd.DataFrame(results)
    
    for size in ['s', 'm', 'l']:
        if not df.empty and size in df['size'].values:
            size_df = df[df['size'] == size].sort_values('val_acc', ascending=False)
            if not size_df.empty:
                logger.info(f"\nGröße {size.upper()}:")
                logger.info(f"  Avg Genauigkeit: {size_df['val_acc'].mean():.4f} ± {size_df['val_acc'].std():.4f}")
                logger.info(f"  Avg Parameter: {size_df['params_m'].mean():.2f}M")
                logger.info(f"  Beste: {size_df.iloc[0]['name']} ({size_df.iloc[0]['val_acc']:.4f})")
            else:
                 logger.info(f"\nGröße {size.upper()}: Keine erfolgreichen Läufe.")
        else:
            logger.info(f"\nGröße {size.upper()}: Keine Daten vorhanden.")