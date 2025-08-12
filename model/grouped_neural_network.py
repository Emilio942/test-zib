"""
Neuronales Netzwerk mit logisch gruppierten Neuronen
Experimentelles Sprachmodell für Neuronengruppen-Analyse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Konfiguration für das experimentelle Modell"""
    vocab_size: int = 1000
    embedding_dim: int = 128
    hidden_dim: int = 256
    neuron_groups: Dict[str, int] = None  # Anzahl Neuronen pro Gruppe
    output_dim: int = 1000
    dropout_rate: float = 0.1
    max_sequence_length: int = 50
    
    def __post_init__(self):
        if self.neuron_groups is None:
            # Standard-Neuronengruppen basierend auf Datensatz-Kategorien
            self.neuron_groups = {
                'farben': 32,
                'bewegungen': 32, 
                'objekte': 32,
                'aktionen': 48,  # Komplexere Kategorie -> mehr Neuronen
                'zustände': 32,
                'zahlen': 24,    # Einfachere Kategorie -> weniger Neuronen
                'logik': 56,     # Komplexeste Kategorie -> meiste Neuronen
                'integration': 64  # Gruppe für kategorieübergreifende Integration
            }

class NeuronGroup(nn.Module):
    """
    Einzelne Neuronengruppe für eine semantische Kategorie
    """
    def __init__(self, input_dim: int, group_size: int, group_name: str):
        super().__init__()
        self.group_name = group_name
        self.group_size = group_size
        
        # Haupttransformation der Gruppe
        self.linear = nn.Linear(input_dim, group_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Gruppe-spezifische Normalisierung
        self.layer_norm = nn.LayerNorm(group_size)
        
        # Interne Aufmerksamkeit innerhalb der Gruppe
        self.self_attention = nn.MultiheadAttention(
            embed_dim=group_size, 
            num_heads=4 if group_size >= 32 else 2,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass für die Neuronengruppe
        
        Args:
            x: Input Tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor [batch_size, seq_len, group_size]
        """
        # Lineare Transformation
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Layer Normalization
        out = self.layer_norm(out)
        
        # Selbst-Aufmerksamkeit innerhalb der Gruppe
        attended_out, attention_weights = self.self_attention(out, out, out)
        
        # Residual Connection
        out = out + attended_out
        
        return out

class GroupedNeuralNetwork(nn.Module):
    """
    Hauptmodell mit logisch gruppierten Neuronen
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        
        # Neuronengruppen erstellen
        self.neuron_groups = nn.ModuleDict()
        for group_name, group_size in config.neuron_groups.items():
            self.neuron_groups[group_name] = NeuronGroup(
                input_dim=config.embedding_dim,
                group_size=group_size,
                group_name=group_name
            )
        
        # Inter-Gruppen Aufmerksamkeit
        total_group_size = sum(config.neuron_groups.values())
        self.inter_group_attention = nn.MultiheadAttention(
            embed_dim=total_group_size,
            num_heads=8,
            batch_first=True
        )
        
        # Final Layers
        self.final_norm = nn.LayerNorm(total_group_size)
        self.output_projection = nn.Linear(total_group_size, config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward Pass des gruppierten Modells
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Aufmerksamkeitsmaske [batch_size, seq_len]
            
        Returns:
            Dictionary mit Outputs und Zwischenergebnissen
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Kombiniere Token und Position Embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Verarbeitung durch Neuronengruppen
        group_outputs = {}
        group_tensors = []
        
        for group_name, group_module in self.neuron_groups.items():
            group_output = group_module(embeddings)
            group_outputs[group_name] = group_output
            group_tensors.append(group_output)
        
        # Konkateniere alle Gruppenausgaben
        concatenated_groups = torch.cat(group_tensors, dim=-1)
        
        # Inter-Gruppen Aufmerksamkeit
        attended_output, inter_attention_weights = self.inter_group_attention(
            concatenated_groups, concatenated_groups, concatenated_groups
        )
        
        # Residual Connection
        attended_output = concatenated_groups + attended_output
        attended_output = self.final_norm(attended_output)
        
        # Final Projection
        hidden_output = self.output_projection(attended_output)
        hidden_output = F.relu(hidden_output)
        hidden_output = self.dropout(hidden_output)
        
        # Classification Head
        logits = self.classifier(hidden_output)
        
        return {
            'logits': logits,
            'hidden_states': hidden_output,
            'group_outputs': group_outputs,
            'inter_attention_weights': inter_attention_weights,
            'embeddings': embeddings
        }
    
    def get_neuron_activations(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extrahiert Neuronenaktivierungen für Analyse
        
        Returns:
            Dictionary mit Aktivierungen pro Gruppe
        """
        with torch.no_grad():
            outputs = self.forward(input_ids)
            return outputs['group_outputs']
    
    def analyze_neuron_importance(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Analysiert die Wichtigkeit einzelner Neuronen
        
        Returns:
            Dictionary mit Wichtigkeits-Scores pro Gruppe
        """
        self.eval()
        group_activations = {name: [] for name in self.config.neuron_groups.keys()}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                outputs = self.get_neuron_activations(input_ids)
                
                for group_name, activations in outputs.items():
                    # Mittlere Aktivierung über Sequenz und Batch
                    mean_activation = activations.mean(dim=(0, 1))
                    group_activations[group_name].append(mean_activation.cpu().numpy())
        
        # Berechne Wichtigkeits-Scores
        importance_scores = {}
        for group_name, activations_list in group_activations.items():
            activations_array = np.stack(activations_list)
            # Verwende Standardabweichung als Wichtigkeits-Maß
            importance_scores[group_name] = np.std(activations_array, axis=0)
        
        return importance_scores
    
    def prune_neurons(self, importance_scores: Dict[str, np.ndarray], prune_ratio: float = 0.2):
        """
        Entfernt unwichtige Neuronen basierend auf Wichtigkeits-Scores
        
        Args:
            importance_scores: Wichtigkeits-Scores pro Gruppe
            prune_ratio: Anteil der zu entfernenden Neuronen
        """
        logger.info(f"Pruning {prune_ratio*100:.1f}% der Neuronen...")
        
        for group_name, scores in importance_scores.items():
            group_module = self.neuron_groups[group_name]
            num_neurons = len(scores)
            num_to_prune = int(num_neurons * prune_ratio)
            
            # Finde die unwichtigsten Neuronen
            pruned_indices = np.argsort(scores)[:num_to_prune]
            
            logger.info(f"Gruppe {group_name}: {num_to_prune}/{num_neurons} Neuronen entfernt")
            
            # Hier würde die tatsächliche Pruning-Implementierung stehen
            # Für dieses Experiment loggen wir nur die Indices
            
    def get_model_stats(self) -> Dict:
        """Gibt Modell-Statistiken zurück"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        group_sizes = {name: size for name, size in self.config.neuron_groups.items()}
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'neuron_groups': group_sizes,
            'total_neurons': sum(group_sizes.values()),
            'embedding_dim': self.config.embedding_dim,
            'hidden_dim': self.config.hidden_dim
        }

def create_model(config: Optional[ModelConfig] = None) -> GroupedNeuralNetwork:
    """
    Factory-Funktion zur Modellerstellung
    
    Args:
        config: Modellkonfiguration (optional)
        
    Returns:
        Initialisiertes GroupedNeuralNetwork
    """
    if config is None:
        config = ModelConfig()
    
    model = GroupedNeuralNetwork(config)
    
    # Initialisierung der Gewichte
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
    
    model.apply(init_weights)
    
    logger.info("✅ Gruppiertes Neuronales Netzwerk erstellt")
    stats = model.get_model_stats()
    logger.info(f"📊 Modell-Statistiken: {stats}")
    
    return model

# Test der Modellarchitektur
if __name__ == "__main__":
    # Erstelle Test-Modell
    config = ModelConfig()
    model = create_model(config)
    
    # Test Forward Pass
    batch_size, seq_len = 2, 10
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("🧪 Teste Modellarchitektur...")
    outputs = model(test_input)
    
    print(f"✅ Forward Pass erfolgreich!")
    print(f"📊 Output Shape: {outputs['logits'].shape}")
    print(f"🧠 Neuronengruppen: {list(outputs['group_outputs'].keys())}")
    
    # Zeige Modell-Statistiken
    stats = model.get_model_stats()
    print(f"📈 Gesamt-Parameter: {stats['total_parameters']:,}")
    print(f"🎯 Gesamt-Neuronen: {stats['total_neurons']:,}")
