"""
Modellkompression-Experiment
Testet Pruning und Quantization für Modelloptimierung
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import copy

from grouped_neural_network import GroupedNeuralNetwork, ModelConfig, create_model
from train import ExperimentDataset, ExperimentTrainer
from quick_test import create_test_config

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCompressionExperiment:
    """
    Experimentiert mit Modellkompression-Techniken
    """
    def __init__(self, base_config: dict):
        self.base_config = base_config
        self.results = []
        
    def create_baseline_model(self, dataset: ExperimentDataset) -> Tuple[GroupedNeuralNetwork, Dict]:
        """Erstellt und trainiert Baseline-Modell"""
        logger.info("🚀 Erstelle Baseline-Modell...")
        
        # Standard-Konfiguration
        config = ModelConfig()
        config.embedding_dim = 64
        config.hidden_dim = 128
        config.neuron_groups = {
            'farben': 16, 'bewegungen': 16, 'objekte': 16, 'aktionen': 24,
            'zustände': 16, 'zahlen': 12, 'logik': 28, 'integration': 32
        }
        
        model = create_model(config)
        
        # Training
        train_result = self.train_model(model, dataset, "baseline")
        
        return model, train_result
    
    def train_model(self, model: GroupedNeuralNetwork, dataset: ExperimentDataset, name: str) -> Dict:
        """Trainiert ein Modell und gibt Metriken zurück"""
        
        # Bereite Daten vor
        small_dataset = torch.utils.data.Subset(dataset, range(min(300, len(dataset))))
        train_size = int(0.8 * len(small_dataset))
        val_size = len(small_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(small_dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Trainer
        trainer = ExperimentTrainer(model, self.base_config)
        
        # Training
        start_time = time.time()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(8):  # Mehr Epochen für stabilere Ergebnisse
            train_loss = trainer.train_epoch(train_dataloader)
            val_loss = trainer.validate(val_dataloader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        training_time = time.time() - start_time
        
        # Modellstatistiken
        stats = model.get_model_stats()
        
        result = {
            'name': name,
            'total_parameters': stats['total_parameters'],
            'total_neurons': stats['total_neurons'],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        logger.info(f"✅ '{name}' - Best Val Loss: {result['best_val_loss']:.4f}, "
                   f"Parameter: {result['total_parameters']:,}")
        
        return result
    
    def magnitude_pruning(self, model: GroupedNeuralNetwork, prune_ratio: float = 0.3) -> GroupedNeuralNetwork:
        """
        Implementiert Magnitude-basiertes Pruning
        Entfernt Neuronen mit den kleinsten Gewichtsmagnitudes
        """
        logger.info(f"✂️ Führe Magnitude Pruning durch (Ratio: {prune_ratio})")
        
        pruned_model = copy.deepcopy(model)
        
        # Analysiere Gewichtsmagnitudes pro Neuronengruppe
        for group_name, group_module in pruned_model.neuron_groups.items():
            linear_layer = group_module.linear
            weights = linear_layer.weight.data  # Shape: [group_size, input_dim]
            
            # Berechne Magnitude pro Neuron (L2-Norm über Input-Dimension)
            neuron_magnitudes = torch.norm(weights, p=2, dim=1)
            
            # Bestimme zu pruningde Neuronen
            num_neurons = weights.size(0)
            num_to_prune = int(num_neurons * prune_ratio)
            
            if num_to_prune > 0:
                # Finde Neuronen mit kleinsten Magnitudes
                _, prune_indices = torch.topk(neuron_magnitudes, num_to_prune, largest=False)
                
                # Setze Gewichte dieser Neuronen auf Null
                with torch.no_grad():
                    weights[prune_indices] = 0
                    if linear_layer.bias is not None:
                        linear_layer.bias.data[prune_indices] = 0
                
                logger.info(f"  🧠 {group_name}: {num_to_prune}/{num_neurons} Neuronen gepruned")
        
        return pruned_model
    
    def structured_pruning(self, model: GroupedNeuralNetwork, target_groups: List[str], reduce_ratio: float = 0.5) -> GroupedNeuralNetwork:
        """
        Strukturiertes Pruning: Reduziert Größe spezifischer Neuronengruppen
        """
        logger.info(f"🏗️ Führe strukturiertes Pruning durch für Gruppen: {target_groups}")
        
        # Erstelle neue Konfiguration mit reduzierten Gruppengrößen
        new_config = copy.deepcopy(model.config)
        
        for group_name in target_groups:
            if group_name in new_config.neuron_groups:
                original_size = new_config.neuron_groups[group_name]
                new_size = max(1, int(original_size * (1 - reduce_ratio)))
                new_config.neuron_groups[group_name] = new_size
                logger.info(f"  🧠 {group_name}: {original_size} → {new_size} Neuronen")
        
        # Erstelle neues Modell mit reduzierter Architektur
        pruned_model = create_model(new_config)
        
        # Übertrage Gewichte (wo möglich)
        self._transfer_weights(model, pruned_model, target_groups, reduce_ratio)
        
        return pruned_model
    
    def _transfer_weights(self, source_model: GroupedNeuralNetwork, target_model: GroupedNeuralNetwork, 
                         pruned_groups: List[str], reduce_ratio: float):
        """Überträgt Gewichte vom Source- zum Target-Modell"""
        with torch.no_grad():
            for group_name, target_group in target_model.neuron_groups.items():
                source_group = source_model.neuron_groups[group_name]
                
                if group_name in pruned_groups:
                    # Für geprunte Gruppen: nimm die wichtigsten Neuronen
                    source_weights = source_group.linear.weight.data
                    source_bias = source_group.linear.bias.data if source_group.linear.bias is not None else None
                    
                    # Berechne Wichtigkeit (L2-Norm)
                    neuron_importance = torch.norm(source_weights, p=2, dim=1)
                    
                    # Nimm die wichtigsten Neuronen
                    target_size = target_group.linear.weight.size(0)
                    _, important_indices = torch.topk(neuron_importance, target_size, largest=True)
                    
                    # Übertrage Gewichte
                    target_group.linear.weight.data = source_weights[important_indices]
                    if target_group.linear.bias is not None and source_bias is not None:
                        target_group.linear.bias.data = source_bias[important_indices]
                else:
                    # Für nicht-geprunte Gruppen: kopiere alle Gewichte
                    target_group.linear.weight.data = source_group.linear.weight.data.clone()
                    if target_group.linear.bias is not None and source_group.linear.bias is not None:
                        target_group.linear.bias.data = source_group.linear.bias.data.clone()
    
    def quantization_simulation(self, model: GroupedNeuralNetwork, bits: int = 8) -> GroupedNeuralNetwork:
        """
        Simuliert Quantization durch Reduzierung der Gewichtspräzision
        """
        logger.info(f"🔢 Führe {bits}-Bit Quantization durch")
        
        quantized_model = copy.deepcopy(model)
        
        for name, param in quantized_model.named_parameters():
            if param.requires_grad:
                # Simuliere Quantization
                param_min = param.data.min()
                param_max = param.data.max()
                
                # Quantization levels
                num_levels = 2 ** bits
                
                # Quantisiere
                param_range = param_max - param_min
                step_size = param_range / (num_levels - 1)
                
                quantized_param = torch.round((param.data - param_min) / step_size) * step_size + param_min
                param.data = quantized_param
        
        return quantized_model
    
    def run_compression_experiments(self, dataset: ExperimentDataset):
        """Führt alle Kompression-Experimente durch"""
        logger.info("🗜️ MODELLKOMPRESSION-EXPERIMENT")
        logger.info("=" * 60)
        
        # 1. Baseline-Modell
        baseline_model, baseline_result = self.create_baseline_model(dataset)
        self.results.append(baseline_result)
        
        # 2. Magnitude Pruning (verschiedene Ratios)
        for prune_ratio in [0.2, 0.4, 0.6]:
            logger.info(f"\n🔧 Magnitude Pruning {prune_ratio*100:.0f}%...")
            pruned_model = self.magnitude_pruning(baseline_model, prune_ratio)
            result = self.train_model(pruned_model, dataset, f"magnitude_prune_{prune_ratio:.1f}")
            self.results.append(result)
        
        # 3. Strukturiertes Pruning (verschiedene Strategien)
        strategies = [
            (['farben', 'zahlen'], 0.5, "simple_groups"),
            (['logik'], 0.3, "complex_group"),
            (['farben', 'bewegungen', 'objekte'], 0.4, "concrete_groups")
        ]
        
        for target_groups, reduce_ratio, strategy_name in strategies:
            logger.info(f"\n🏗️ Strukturiertes Pruning: {strategy_name}...")
            pruned_model = self.structured_pruning(baseline_model, target_groups, reduce_ratio)
            result = self.train_model(pruned_model, dataset, f"struct_prune_{strategy_name}")
            self.results.append(result)
        
        # 4. Quantization
        for bits in [8, 4]:
            logger.info(f"\n🔢 {bits}-Bit Quantization...")
            quantized_model = self.quantization_simulation(baseline_model, bits)
            result = self.train_model(quantized_model, dataset, f"quantized_{bits}bit")
            self.results.append(result)
        
        # 5. Kombinierte Kompression
        logger.info(f"\n🔀 Kombinierte Kompression...")
        combined_model = self.magnitude_pruning(baseline_model, 0.3)
        combined_model = self.quantization_simulation(combined_model, 8)
        result = self.train_model(combined_model, dataset, "combined_compression")
        self.results.append(result)
        
        self.analyze_compression_results()
        self.create_compression_visualizations()
    
    def analyze_compression_results(self):
        """Analysiert die Kompression-Ergebnisse"""
        logger.info("\n📊 KOMPRESSION-ANALYSE")
        logger.info("=" * 50)
        
        baseline = self.results[0]
        
        logger.info(f"📋 BASELINE: {baseline['name']}")
        logger.info(f"   Parameter: {baseline['total_parameters']:,}")
        logger.info(f"   Val Loss: {baseline['best_val_loss']:.4f}")
        
        logger.info(f"\n🏆 KOMPRESSION-ERGEBNISSE:")
        for result in self.results[1:]:
            param_reduction = (1 - result['total_parameters'] / baseline['total_parameters']) * 100
            loss_change = ((result['best_val_loss'] - baseline['best_val_loss']) / baseline['best_val_loss']) * 100
            
            logger.info(f"   {result['name']:20} - "
                       f"Parameter: -{param_reduction:5.1f}%, "
                       f"Loss Change: {loss_change:+6.1f}%")
        
        # Finde beste Kompromisse
        logger.info(f"\n⭐ BESTE KOMPROMISSE:")
        
        # Beste Parameter-Reduktion bei akzeptablem Loss-Anstieg (<10%)
        good_compressions = [r for r in self.results[1:] 
                           if ((r['best_val_loss'] - baseline['best_val_loss']) / baseline['best_val_loss']) < 0.1]
        
        if good_compressions:
            best_compression = min(good_compressions, key=lambda x: x['total_parameters'])
            param_reduction = (1 - best_compression['total_parameters'] / baseline['total_parameters']) * 100
            logger.info(f"   Beste Kompression: {best_compression['name']} (-{param_reduction:.1f}% Parameter)")
        
        # Beste Performance bei reduzierter Größe
        compressed_models = [r for r in self.results[1:] 
                           if r['total_parameters'] < baseline['total_parameters']]
        
        if compressed_models:
            best_performance = min(compressed_models, key=lambda x: x['best_val_loss'])
            logger.info(f"   Beste Performance: {best_performance['name']} (Loss: {best_performance['best_val_loss']:.4f})")
    
    def create_compression_visualizations(self):
        """Erstellt Visualisierungen der Kompression-Ergebnisse"""
        logger.info("\n📊 Erstelle Kompression-Visualisierungen...")
        
        baseline = self.results[0]
        
        # Daten vorbereiten
        names = [r['name'] for r in self.results]
        param_reductions = [(1 - r['total_parameters'] / baseline['total_parameters']) * 100 for r in self.results]
        loss_changes = [((r['best_val_loss'] - baseline['best_val_loss']) / baseline['best_val_loss']) * 100 for r in self.results]
        val_losses = [r['best_val_loss'] for r in self.results]
        
        # Visualisierung
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Parameter-Reduktion vs Loss-Änderung
        colors = ['red' if i == 0 else 'blue' for i in range(len(names))]
        ax1.scatter(param_reductions, loss_changes, c=colors, s=100, alpha=0.7)
        
        for i, name in enumerate(names):
            if i > 0:  # Nicht Baseline annotieren
                ax1.annotate(name.replace('_', '\n'), (param_reductions[i], loss_changes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Parameter-Reduktion (%)')
        ax1.set_ylabel('Loss-Änderung (%)')
        ax1.set_title('Kompression Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute Validation Losses
        bars = ax2.bar(range(len(names)), val_losses)
        bars[0].set_color('red')  # Baseline
        ax2.set_xlabel('Modell')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Performance Vergleich')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        # 3. Parameter-Anzahl
        total_params = [r['total_parameters'] for r in self.results]
        bars = ax3.bar(range(len(names)), total_params)
        bars[0].set_color('red')  # Baseline
        ax3.set_xlabel('Modell')
        ax3.set_ylabel('Anzahl Parameter')
        ax3.set_title('Modellgröße Vergleich')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right')
        
        # 4. Effizienz (Performance pro Parameter)
        efficiencies = [r['best_val_loss'] / r['total_parameters'] * 1000000 for r in self.results]
        bars = ax4.bar(range(len(names)), efficiencies)
        bars[0].set_color('red')  # Baseline
        
        # Markiere beste Effizienz
        best_idx = np.argmin(efficiencies)
        bars[best_idx].set_color('gold')
        
        ax4.set_xlabel('Modell')
        ax4.set_ylabel('Effizienz (Loss/Million Parameter)')
        ax4.set_title('Modell-Effizienz')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('evaluation/compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("📊 Visualisierung gespeichert als 'evaluation/compression_analysis.png'")
    
    def save_compression_results(self):
        """Speichert Kompression-Ergebnisse"""
        with open('evaluation/compression_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("💾 Kompression-Ergebnisse gespeichert")

def main():
    """Hauptfunktion für Modellkompression-Experiment"""
    logger.info("🗜️ MODELLKOMPRESSION-EXPERIMENT")
    logger.info("Testet Pruning, Quantization und kombinierte Kompression")
    logger.info("=" * 70)
    
    # Lade Dataset
    config = create_test_config()
    dataset = ExperimentDataset(
        'data/training_examples.json',
        vocab_size=config['vocab_size'],
        max_length=config['max_length']
    )
    
    # Erstelle und führe Experiment durch
    experiment = ModelCompressionExperiment(config)
    experiment.run_compression_experiments(dataset)
    
    # Speichere Ergebnisse
    experiment.save_compression_results()
    
    logger.info("\n🎯 KOMPRESSION-EXPERIMENT ABGESCHLOSSEN!")
    logger.info("✅ Verschiedene Kompression-Techniken getestet")
    logger.info("📊 Trade-offs zwischen Größe und Performance analysiert")
    logger.info("🏆 Optimale Kompression-Strategien identifiziert")

if __name__ == "__main__":
    main()
