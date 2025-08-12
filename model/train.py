"""
Training-Pipeline für das gruppierte Neuronale Netzwerk
Experimentelles Training mit Neuronengruppen-Analyse
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from grouped_neural_network import GroupedNeuralNetwork, ModelConfig, create_model

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentDataset(Dataset):
    """
    Dataset für die Trainingsbeispiele
    """
    def __init__(self, data_path: str, vocab_size: int = 1000, max_length: int = 50):
        self.data = self._load_data(data_path)
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Erstelle einfaches Vokabular (für Experiment)
        self.vocab = self._create_vocabulary()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Lädt Trainingsdaten"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_vocabulary(self) -> List[str]:
        """Erstellt einfaches Vokabular aus den Daten"""
        words = set()
        
        for item in self.data:
            # Tokenisiere Input und Output
            input_words = item['input'].lower().split()
            output_words = item['output'].lower().split()
            words.update(input_words + output_words)
        
        # Füge spezielle Tokens hinzu
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        vocab_list = special_tokens + sorted(list(words))
        
        # Beschränke auf vocab_size
        return vocab_list[:self.vocab_size]
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenisiert Text zu Token-IDs"""
        words = text.lower().split()
        tokens = []
        
        tokens.append(self.word_to_idx.get('<start>', 0))
        
        for word in words:
            token_id = self.word_to_idx.get(word, self.word_to_idx.get('<unk>', 1))
            tokens.append(token_id)
        
        tokens.append(self.word_to_idx.get('<end>', 3))
        
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenisiere Input und Output
        input_tokens = self._tokenize(item['input'])
        output_tokens = self._tokenize(item['output'])
        
        # Padding/Truncation
        input_tokens = self._pad_sequence(input_tokens)
        output_tokens = self._pad_sequence(output_tokens)
        
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'labels': torch.tensor(output_tokens, dtype=torch.long),
            'category': item['category'],
            'complexity': item['complexity'],
            'type': item['type']
        }
    
    def _pad_sequence(self, tokens: List[int]) -> List[int]:
        """Padding/Truncation für einheitliche Länge"""
        if len(tokens) >= self.max_length:
            return tokens[:self.max_length]
        else:
            pad_token = self.word_to_idx.get('<pad>', 0)
            return tokens + [pad_token] * (self.max_length - len(tokens))

class ExperimentTrainer:
    """
    Trainer für das gruppierte Neuronale Netzwerk
    """
    def __init__(self, model: GroupedNeuralNetwork, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer und Loss
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignoriere Padding
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.neuron_activations_history = []
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Trainiert eine Epoche"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Berechne Loss (nur für nicht-padding tokens)
            # Reshape für CrossEntropyLoss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validiert das Modell"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def analyze_neuron_activations(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """Analysiert Neuronenaktivierungen während des Trainings"""
        self.model.eval()
        activations_by_category = {}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                categories = batch['category']
                
                # Hole Gruppenaktivierungen
                group_outputs = self.model.get_neuron_activations(input_ids)
                
                for i, category in enumerate(categories):
                    if category not in activations_by_category:
                        activations_by_category[category] = {
                            group: [] for group in group_outputs.keys()
                        }
                    
                    for group_name, activations in group_outputs.items():
                        # Mittlere Aktivierung für diese Eingabe
                        mean_activation = activations[i].mean(dim=0).cpu().numpy()
                        activations_by_category[category][group_name].append(mean_activation)
        
        # Berechne Statistiken
        activation_stats = {}
        for category, groups in activations_by_category.items():
            activation_stats[category] = {}
            for group_name, activations_list in groups.items():
                if activations_list:
                    activations_array = np.stack(activations_list)
                    activation_stats[category][group_name] = {
                        'mean': np.mean(activations_array, axis=0),
                        'std': np.std(activations_array, axis=0),
                        'max': np.max(activations_array, axis=0)
                    }
        
        return activation_stats
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, num_epochs: int):
        """Haupttraining-Loop"""
        logger.info(f"🚀 Starte Training für {num_epochs} Epochen")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"📊 Trainingsbeispiele: {len(train_dataloader.dataset)}")
        logger.info(f"📊 Validierungsbeispiele: {len(val_dataloader.dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation
            val_loss = self.validate(val_dataloader)
            
            # Tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Neuronenanalyse (jede 5. Epoche)
            if (epoch + 1) % 5 == 0:
                logger.info("🧠 Analysiere Neuronenaktivierungen...")
                activations = self.analyze_neuron_activations(val_dataloader)
                self.neuron_activations_history.append({
                    'epoch': epoch + 1,
                    'activations': activations
                })
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoche {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Zeit: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"model/best_model_epoch_{epoch+1}.pt")
                logger.info(f"✅ Neues bestes Modell gespeichert (Val Loss: {val_loss:.4f})")
        
        logger.info(f"🎉 Training abgeschlossen! Beste Val Loss: {best_val_loss:.4f}")
    
    def save_model(self, path: str):
        """Speichert das Modell"""
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'neuron_activations_history': self.neuron_activations_history
        }, path)
    
    def load_model(self, path: str):
        """Lädt das Modell"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.neuron_activations_history = checkpoint.get('neuron_activations_history', [])
    
    def plot_training_progress(self):
        """Visualisiert den Trainingsverlauf"""
        plt.figure(figsize=(12, 4))
        
        # Loss Curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Neuron Activations (if available)
        if self.neuron_activations_history:
            plt.subplot(1, 2, 2)
            
            # Zeige mittlere Aktivierung für eine Beispielgruppe
            epochs = [entry['epoch'] for entry in self.neuron_activations_history]
            
            # Verwende 'logik' Gruppe als Beispiel
            if self.neuron_activations_history[0]['activations'].get('logik'):
                logik_means = []
                for entry in self.neuron_activations_history:
                    logik_activations = entry['activations']['logik']
                    if 'logik' in logik_activations:
                        mean_activation = np.mean(logik_activations['logik']['mean'])
                        logik_means.append(mean_activation)
                
                if logik_means:
                    plt.plot(epochs, logik_means, 'g-', label='Logik-Gruppe Aktivierung')
                    plt.xlabel('Epoche')
                    plt.ylabel('Mittlere Aktivierung')
                    plt.title('Neuronenaktivierung')
                    plt.legend()
                    plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('evaluation/training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("📊 Trainingsverlauf gespeichert als 'evaluation/training_progress.png'")

def create_experiment_config() -> dict:
    """Erstellt Experimentkonfiguration"""
    return {
        'learning_rate': 0.001,
        'batch_size': 16,
        'num_epochs': 20,
        'max_length': 50,
        'vocab_size': 1000,
        'train_split': 0.8,
        'val_split': 0.2
    }

def main():
    """Hauptfunktion für das Training"""
    logger.info("🔬 Starte Neurongruppen-Experiment")
    
    # Konfiguration
    experiment_config = create_experiment_config()
    model_config = ModelConfig()
    
    # Erstelle Modell
    model = create_model(model_config)
    
    # Erstelle Dataset
    logger.info("📚 Lade Trainingsdaten...")
    dataset = ExperimentDataset(
        'data/training_examples.json',
        vocab_size=experiment_config['vocab_size'],
        max_length=experiment_config['max_length']
    )
    
    # Train/Val Split
    train_size = int(experiment_config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=experiment_config['batch_size'],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=experiment_config['batch_size'],
        shuffle=False
    )
    
    logger.info(f"📊 Vokabular-Größe: {len(dataset.vocab)}")
    logger.info(f"📊 Training/Validation Split: {train_size}/{val_size}")
    
    # Erstelle Trainer
    trainer = ExperimentTrainer(model, experiment_config)
    
    # Starte Training
    trainer.train(
        train_dataloader,
        val_dataloader,
        experiment_config['num_epochs']
    )
    
    # Visualisiere Ergebnisse
    trainer.plot_training_progress()
    
    # Speichere finales Modell
    trainer.save_model('model/final_model.pt')
    
    logger.info("🎯 Experiment abgeschlossen!")

if __name__ == "__main__":
    main()
