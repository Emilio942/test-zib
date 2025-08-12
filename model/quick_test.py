"""
Schneller Test-Lauf für das Neurongruppen-Experiment
Reduzierte Konfiguration für erste Validierung
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from grouped_neural_network import GroupedNeuralNetwork, ModelConfig, create_model
from train import ExperimentDataset, ExperimentTrainer, create_experiment_config

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_config() -> dict:
    """Erstellt reduzierte Testkonfiguration"""
    return {
        'learning_rate': 0.01,
        'batch_size': 8,
        'num_epochs': 3,  # Nur 3 Epochen für schnellen Test
        'max_length': 20,  # Kürzere Sequenzen
        'vocab_size': 500,  # Kleineres Vokabular
        'train_split': 0.8,
        'val_split': 0.2
    }

def create_test_model_config() -> ModelConfig:
    """Erstellt reduzierte Modellkonfiguration für Tests"""
    config = ModelConfig()
    
    # Kleinere Dimensionen für schnelleres Training
    config.embedding_dim = 64
    config.hidden_dim = 128
    config.max_sequence_length = 20
    
    # Reduzierte Neuronengruppen
    config.neuron_groups = {
        'farben': 16,
        'bewegungen': 16, 
        'objekte': 16,
        'aktionen': 24,
        'zustände': 16,
        'zahlen': 12,
        'logik': 28,
        'integration': 32
    }
    
    return config

def test_model_functionality():
    """Testet grundlegende Modellfunktionalität"""
    logger.info("🧪 Teste Modell-Funktionalität...")
    
    # Erstelle Test-Modell
    config = create_test_model_config()
    model = create_model(config)
    
    # Test Forward Pass
    batch_size, seq_len = 4, 10
    test_input = torch.randint(0, 500, (batch_size, seq_len))
    
    outputs = model(test_input)
    
    logger.info(f"✅ Forward Pass: {outputs['logits'].shape}")
    logger.info(f"🧠 Neuronengruppen: {len(outputs['group_outputs'])}")
    
    # Test Neuron Activation Analysis
    activations = model.get_neuron_activations(test_input)
    logger.info(f"📊 Aktivierungen extrahiert für {len(activations)} Gruppen")
    
    return model

def test_dataset_loading():
    """Testet Dataset-Loading"""
    logger.info("📚 Teste Dataset-Loading...")
    
    test_config = create_test_config()
    
    try:
        dataset = ExperimentDataset(
            'data/training_examples.json',
            vocab_size=test_config['vocab_size'],
            max_length=test_config['max_length']
        )
        
        logger.info(f"✅ Dataset geladen: {len(dataset)} Beispiele")
        logger.info(f"📝 Vokabular-Größe: {len(dataset.vocab)}")
        
        # Test Datenpunkt
        sample = dataset[0]
        logger.info(f"📊 Beispiel-Shape: input_ids={sample['input_ids'].shape}, labels={sample['labels'].shape}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Dataset-Loading fehlgeschlagen: {e}")
        return None

def run_quick_training_test(model, dataset):
    """Führt einen sehr kurzen Trainingstest durch"""
    logger.info("🚀 Starte Schnell-Training...")
    
    test_config = create_test_config()
    
    # Sehr kleiner Subset für schnellen Test
    small_dataset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
    
    # Split
    train_size = int(0.8 * len(small_dataset))
    val_size = len(small_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(small_dataset, [train_size, val_size])
    
    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Trainer
    trainer = ExperimentTrainer(model, test_config)
    
    # Kurzes Training (nur 1 Epoche für Test)
    start_time = time.time()
    train_loss = trainer.train_epoch(train_dataloader)
    val_loss = trainer.validate(val_dataloader)
    end_time = time.time()
    
    logger.info(f"✅ Test-Training abgeschlossen in {end_time - start_time:.2f}s")
    logger.info(f"📊 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return trainer

def analyze_neuron_groups(model, dataset):
    """Analysiert Neuronengruppen-Verhalten"""
    logger.info("🔬 Analysiere Neuronengruppen...")
    
    # Erstelle kleinen DataLoader
    small_dataset = torch.utils.data.Subset(dataset, range(min(50, len(dataset))))
    dataloader = DataLoader(small_dataset, batch_size=8, shuffle=False)
    
    # Sammle Aktivierungen
    model.eval()
    group_activations = {group: [] for group in model.config.neuron_groups.keys()}
    device = next(model.parameters()).device  # Hole Device vom Modell
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)  # Move zu richtigem Device
            activations = model.get_neuron_activations(input_ids)
            
            for group_name, group_activation in activations.items():
                # Mittlere Aktivierung über Batch und Sequenz
                mean_activation = group_activation.mean(dim=(0, 1)).cpu().numpy()
                group_activations[group_name].append(mean_activation)
    
    # Statistiken berechnen
    for group_name, activations_list in group_activations.items():
        if activations_list:
            activations_array = np.stack(activations_list)
            mean_activity = np.mean(activations_array)
            std_activity = np.std(activations_array)
            
            logger.info(f"🧠 {group_name:12}: Ø={mean_activity:.4f}, σ={std_activity:.4f}")
    
    return group_activations

def create_quick_visualization(trainer, group_activations):
    """Erstellt schnelle Visualisierung der Ergebnisse"""
    logger.info("📊 Erstelle Visualisierungen...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Modellarchitektur (Neuronengruppen)
    groups = list(trainer.model.config.neuron_groups.keys())
    sizes = list(trainer.model.config.neuron_groups.values())
    
    ax1.bar(groups, sizes, color='skyblue')
    ax1.set_title('Neuronen pro Gruppe')
    ax1.set_ylabel('Anzahl Neuronen')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Gruppenaktivierungen (mittlere Aktivierung)
    if group_activations:
        group_means = []
        for group in groups:
            if group in group_activations and group_activations[group]:
                activations_array = np.stack(group_activations[group])
                group_means.append(np.mean(activations_array))
            else:
                group_means.append(0)
        
        ax2.bar(groups, group_means, color='lightgreen')
        ax2.set_title('Mittlere Gruppenaktivierung')
        ax2.set_ylabel('Aktivierung')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Training Loss (falls verfügbar)
    if hasattr(trainer, 'train_losses') and trainer.train_losses:
        ax3.plot(trainer.train_losses, 'b-', label='Train Loss')
        if hasattr(trainer, 'val_losses') and trainer.val_losses:
            ax3.plot(trainer.val_losses, 'r-', label='Val Loss')
        ax3.set_title('Training Verlauf')
        ax3.set_xlabel('Epoche')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
    
    # 4. Modellstatistiken
    stats = trainer.model.get_model_stats()
    
    ax4.text(0.1, 0.8, f"Gesamt Parameter: {stats['total_parameters']:,}", transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f"Gesamt Neuronen: {stats['total_neurons']:,}", transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f"Embedding Dim: {stats['embedding_dim']}", transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f"Hidden Dim: {stats['hidden_dim']}", transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f"Neuronengruppen: {len(stats['neuron_groups'])}", transform=ax4.transAxes)
    ax4.set_title('Modell-Statistiken')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation/quick_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Visualisierung gespeichert als 'evaluation/quick_test_results.png'")

def main():
    """Hauptfunktion für Schnell-Test"""
    logger.info("🔬 NEURONGRUPPEN-EXPERIMENT - SCHNELL-TEST")
    logger.info("=" * 60)
    
    try:
        # 1. Test Modell-Funktionalität
        model = test_model_functionality()
        
        # 2. Test Dataset-Loading
        dataset = test_dataset_loading()
        if dataset is None:
            logger.error("❌ Dataset-Loading fehlgeschlagen. Test abgebrochen.")
            return
        
        # 3. Kurzes Training
        trainer = run_quick_training_test(model, dataset)
        
        # 4. Neuronengruppen-Analyse
        group_activations = analyze_neuron_groups(model, dataset)
        
        # 5. Visualisierung
        create_quick_visualization(trainer, group_activations)
        
        logger.info("🎉 SCHNELL-TEST ERFOLGREICH ABGESCHLOSSEN!")
        logger.info("✅ Alle Komponenten funktionieren korrekt")
        logger.info("📈 Bereit für vollständiges Training mit: python model/train.py")
        
    except Exception as e:
        logger.error(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
