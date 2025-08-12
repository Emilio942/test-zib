"""
Neuronenvariations-Experiment
Testet verschiedene Neuronenzahlen und deren Auswirkung auf die Modellqualität
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import pandas as pd

from grouped_neural_network import GroupedNeuralNetwork, ModelConfig, create_model
from train import ExperimentDataset, ExperimentTrainer
from quick_test import create_test_config

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuronVariationExperiment:
    """
    Experimentiert mit verschiedenen Neuronenzahlen
    """
    def __init__(self, base_config: dict):
        self.base_config = base_config
        self.results = []
        self.models = {}
        
    def create_model_variants(self) -> Dict[str, ModelConfig]:
        """Erstellt verschiedene Modellvarianten mit unterschiedlichen Neuronenzahlen"""
        variants = {}
        
        # 1. Minimalistisches Modell (sehr wenige Neuronen)
        minimal_config = ModelConfig()
        minimal_config.embedding_dim = 32
        minimal_config.hidden_dim = 64
        minimal_config.neuron_groups = {
            'farben': 4, 'bewegungen': 4, 'objekte': 4, 'aktionen': 6,
            'zustände': 4, 'zahlen': 3, 'logik': 8, 'integration': 8
        }
        variants['minimal'] = minimal_config
        
        # 2. Kleines Modell (wenige Neuronen)
        small_config = ModelConfig()
        small_config.embedding_dim = 48
        small_config.hidden_dim = 96
        small_config.neuron_groups = {
            'farben': 8, 'bewegungen': 8, 'objekte': 8, 'aktionen': 12,
            'zustände': 8, 'zahlen': 6, 'logik': 16, 'integration': 16
        }
        variants['small'] = small_config
        
        # 3. Mittleres Modell (Standard)
        medium_config = ModelConfig()
        medium_config.embedding_dim = 64
        medium_config.hidden_dim = 128
        medium_config.neuron_groups = {
            'farben': 16, 'bewegungen': 16, 'objekte': 16, 'aktionen': 24,
            'zustände': 16, 'zahlen': 12, 'logik': 28, 'integration': 32
        }
        variants['medium'] = medium_config
        
        # 4. Großes Modell (viele Neuronen)
        large_config = ModelConfig()
        large_config.embedding_dim = 96
        large_config.hidden_dim = 192
        large_config.neuron_groups = {
            'farben': 32, 'bewegungen': 32, 'objekte': 32, 'aktionen': 48,
            'zustände': 32, 'zahlen': 24, 'logik': 56, 'integration': 64
        }
        variants['large'] = large_config
        
        # 5. Unbalanciertes Modell (ungleiche Verteilung)
        unbalanced_config = ModelConfig()
        unbalanced_config.embedding_dim = 64
        unbalanced_config.hidden_dim = 128
        unbalanced_config.neuron_groups = {
            'farben': 8, 'bewegungen': 8, 'objekte': 8, 'aktionen': 16,
            'zustände': 8, 'zahlen': 4, 'logik': 64, 'integration': 24  # Logik dominiert
        }
        variants['unbalanced'] = unbalanced_config
        
        return variants
    
    def train_model_variant(self, variant_name: str, config: ModelConfig, dataset: ExperimentDataset) -> Dict:
        """Trainiert eine Modellvariante und misst Performance"""
        logger.info(f"🚀 Trainiere Modell '{variant_name}'...")
        
        # Erstelle Modell
        model = create_model(config)
        
        # Bereite Daten vor
        small_dataset = torch.utils.data.Subset(dataset, range(min(200, len(dataset))))
        train_size = int(0.8 * len(small_dataset))
        val_size = len(small_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(small_dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Trainer
        trainer = ExperimentTrainer(model, self.base_config)
        
        # Training (verkürzt für Experiment)
        start_time = time.time()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(5):  # Nur 5 Epochen für schnelle Experimente
            train_loss = trainer.train_epoch(train_dataloader)
            val_loss = trainer.validate(val_dataloader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        training_time = time.time() - start_time
        
        # Modellstatistiken
        stats = model.get_model_stats()
        
        # Analyse der Neuronenaktivierungen
        activations = self.analyze_activations(model, val_dataloader)
        
        # Speichere Modell
        self.models[variant_name] = model
        
        result = {
            'variant_name': variant_name,
            'total_parameters': stats['total_parameters'],
            'total_neurons': stats['total_neurons'],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'neuron_groups': config.neuron_groups,
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'activations': activations
        }
        
        logger.info(f"✅ '{variant_name}' - Val Loss: {result['best_val_loss']:.4f}, "
                   f"Parameter: {result['total_parameters']:,}, Zeit: {training_time:.2f}s")
        
        return result
    
    def analyze_activations(self, model: GroupedNeuralNetwork, dataloader: DataLoader) -> Dict:
        """Analysiert Aktivierungsmuster eines Modells"""
        model.eval()
        group_activations = {group: [] for group in model.config.neuron_groups.keys()}
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                activations = model.get_neuron_activations(input_ids)
                
                for group_name, group_activation in activations.items():
                    mean_activation = group_activation.mean(dim=(0, 1)).cpu().numpy()
                    group_activations[group_name].append(mean_activation)
        
        # Berechne Statistiken
        activation_stats = {}
        for group_name, activations_list in group_activations.items():
            if activations_list:
                activations_array = np.stack(activations_list)
                activation_stats[group_name] = {
                    'mean': float(np.mean(activations_array)),
                    'std': float(np.std(activations_array)),
                    'activity_level': float(np.mean(np.abs(activations_array)))
                }
        
        return activation_stats
    
    def run_experiments(self, dataset: ExperimentDataset):
        """Führt alle Neuronenvariations-Experimente durch"""
        logger.info("🔬 NEURONENVARIATIONS-EXPERIMENT")
        logger.info("=" * 60)
        
        variants = self.create_model_variants()
        
        for variant_name, config in variants.items():
            try:
                result = self.train_model_variant(variant_name, config, dataset)
                self.results.append(result)
            except Exception as e:
                logger.error(f"❌ Fehler bei Variante '{variant_name}': {e}")
        
        self.analyze_results()
        self.create_visualizations()
    
    def analyze_results(self):
        """Analysiert die Ergebnisse aller Experimente"""
        logger.info("\n📊 EXPERIMENTANALYSE")
        logger.info("=" * 50)
        
        # Sortiere nach Performance (beste Validation Loss)
        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])
        
        logger.info("🏆 RANKING nach Validation Loss:")
        for i, result in enumerate(sorted_results):
            logger.info(f"{i+1}. {result['variant_name']:12} - "
                       f"Val Loss: {result['best_val_loss']:.4f}, "
                       f"Parameter: {result['total_parameters']:6,}, "
                       f"Neuronen: {result['total_neurons']:3}")
        
        # Effizienz-Analyse (Performance pro Parameter)
        logger.info("\n⚡ EFFIZIENZ-ANALYSE (niedrigere Loss pro Parameter ist besser):")
        for result in sorted_results:
            efficiency = result['best_val_loss'] / result['total_parameters'] * 1000000
            logger.info(f"{result['variant_name']:12} - "
                       f"Effizienz: {efficiency:.2f} (Loss/Million Parameter)")
        
        # Aktivierungsanalyse
        logger.info("\n🧠 AKTIVIERUNGSMUSTER:")
        for result in self.results:
            if result['activations']:
                total_activity = sum(stats['activity_level'] for stats in result['activations'].values())
                logger.info(f"{result['variant_name']:12} - Gesamt-Aktivität: {total_activity:.3f}")
    
    def create_visualizations(self):
        """Erstellt umfassende Visualisierungen der Ergebnisse"""
        logger.info("\n📊 Erstelle Visualisierungen...")
        
        # Vorbereitung der Daten
        variants = [r['variant_name'] for r in self.results]
        total_params = [r['total_parameters'] for r in self.results]
        total_neurons = [r['total_neurons'] for r in self.results]
        val_losses = [r['best_val_loss'] for r in self.results]
        train_times = [r['training_time'] for r in self.results]
        
        # Große Visualisierung mit 6 Subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Parameter vs Performance
        axes[0, 0].scatter(total_params, val_losses, s=100, alpha=0.7, c=['red', 'orange', 'green', 'blue', 'purple'])
        for i, variant in enumerate(variants):
            axes[0, 0].annotate(variant, (total_params[i], val_losses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 0].set_xlabel('Gesamt-Parameter')
        axes[0, 0].set_ylabel('Beste Validation Loss')
        axes[0, 0].set_title('Parameter vs Performance')
        axes[0, 0].grid(True)
        
        # 2. Neuronen vs Performance
        axes[0, 1].scatter(total_neurons, val_losses, s=100, alpha=0.7, c=['red', 'orange', 'green', 'blue', 'purple'])
        for i, variant in enumerate(variants):
            axes[0, 1].annotate(variant, (total_neurons[i], val_losses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 1].set_xlabel('Gesamt-Neuronen')
        axes[0, 1].set_ylabel('Beste Validation Loss')
        axes[0, 1].set_title('Neuronen vs Performance')
        axes[0, 1].grid(True)
        
        # 3. Trainingszeit vs Performance
        axes[0, 2].scatter(train_times, val_losses, s=100, alpha=0.7, c=['red', 'orange', 'green', 'blue', 'purple'])
        for i, variant in enumerate(variants):
            axes[0, 2].annotate(variant, (train_times[i], val_losses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 2].set_xlabel('Trainingszeit (s)')
        axes[0, 2].set_ylabel('Beste Validation Loss')
        axes[0, 2].set_title('Trainingszeit vs Performance')
        axes[0, 2].grid(True)
        
        # 4. Neuronenverteilung
        for i, result in enumerate(self.results):
            groups = list(result['neuron_groups'].keys())
            counts = list(result['neuron_groups'].values())
            x_pos = np.arange(len(groups)) + i * 0.15
            axes[1, 0].bar(x_pos, counts, width=0.15, label=result['variant_name'], alpha=0.7)
        
        axes[1, 0].set_xlabel('Neuronengruppen')
        axes[1, 0].set_ylabel('Anzahl Neuronen')
        axes[1, 0].set_title('Neuronenverteilung pro Modell')
        axes[1, 0].set_xticks(np.arange(len(groups)) + 0.3)
        axes[1, 0].set_xticklabels(groups, rotation=45)
        axes[1, 0].legend()
        
        # 5. Effizienz-Ranking
        efficiencies = [r['best_val_loss'] / r['total_parameters'] * 1000000 for r in self.results]
        sorted_indices = np.argsort(efficiencies)
        sorted_variants = [variants[i] for i in sorted_indices]
        sorted_efficiencies = [efficiencies[i] for i in sorted_indices]
        
        bars = axes[1, 1].bar(sorted_variants, sorted_efficiencies, color='lightblue', alpha=0.7)
        axes[1, 1].set_xlabel('Modell-Variante')
        axes[1, 1].set_ylabel('Effizienz (Loss/Million Parameter)')
        axes[1, 1].set_title('Modell-Effizienz Ranking')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Markiere das beste Modell
        best_idx = np.argmin(sorted_efficiencies)
        bars[best_idx].set_color('gold')
        
        # 6. Aktivierungsmuster-Heatmap
        if all(r['activations'] for r in self.results):
            activation_matrix = []
            group_names = list(self.results[0]['activations'].keys())
            
            for result in self.results:
                row = [result['activations'][group]['activity_level'] for group in group_names]
                activation_matrix.append(row)
            
            im = axes[1, 2].imshow(activation_matrix, cmap='viridis', aspect='auto')
            axes[1, 2].set_xticks(range(len(group_names)))
            axes[1, 2].set_xticklabels(group_names, rotation=45)
            axes[1, 2].set_yticks(range(len(variants)))
            axes[1, 2].set_yticklabels(variants)
            axes[1, 2].set_title('Aktivierungsmuster Heatmap')
            
            # Colorbar
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('evaluation/neuron_variation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("📊 Visualisierung gespeichert als 'evaluation/neuron_variation_analysis.png'")
    
    def save_results(self):
        """Speichert Ergebnisse als JSON und CSV"""
        # JSON für vollständige Daten
        with open('evaluation/neuron_variation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # CSV für einfache Analyse
        csv_data = []
        for result in self.results:
            csv_row = {
                'variant': result['variant_name'],
                'total_parameters': result['total_parameters'],
                'total_neurons': result['total_neurons'],
                'best_val_loss': result['best_val_loss'],
                'training_time': result['training_time'],
                'efficiency': result['best_val_loss'] / result['total_parameters'] * 1000000
            }
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv('evaluation/neuron_variation_summary.csv', index=False)
        
        logger.info("💾 Ergebnisse gespeichert in 'evaluation/'")

def main():
    """Hauptfunktion für Neuronenvariations-Experiment"""
    logger.info("🧬 NEURONENVARIATIONS-EXPERIMENT")
    logger.info("Testet verschiedene Neuronenzahlen und deren Auswirkung auf Performance")
    logger.info("=" * 70)
    
    # Lade Dataset
    config = create_test_config()
    dataset = ExperimentDataset(
        'data/training_examples.json',
        vocab_size=config['vocab_size'],
        max_length=config['max_length']
    )
    
    # Erstelle und führe Experiment durch
    experiment = NeuronVariationExperiment(config)
    experiment.run_experiments(dataset)
    
    # Speichere Ergebnisse
    experiment.save_results()
    
    logger.info("\n🎯 EXPERIMENT ABGESCHLOSSEN!")
    logger.info("✅ Verschiedene Neuronenkonfigurationen getestet")
    logger.info("📊 Ergebnisse visualisiert und gespeichert")
    logger.info("📈 Bereit für Modellkompression-Experimente")

if __name__ == "__main__":
    main()
