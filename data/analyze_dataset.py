"""
Datensatz-Analyse und Visualisierung
Analysiert die generierten Datensätze auf Qualität und Verteilung
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List

class DatasetAnalyzer:
    """Analysiert und visualisiert den generierten Datensatz"""
    
    def __init__(self, word_list_path: str, training_examples_path: str):
        self.word_list = self._load_json(word_list_path)
        self.training_examples = self._load_json(training_examples_path)
        
    def _load_json(self, filepath: str) -> List[Dict]:
        """Lädt JSON-Datei"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_word_distribution(self):
        """Analysiert die Verteilung der Wörter nach Kategorien"""
        print("📊 WORTVERTEILUNG NACH KATEGORIEN")
        print("=" * 50)
        
        # Zähle Wörter pro Kategorie
        category_counts = Counter([word['category'] for word in self.word_list])
        
        for category, count in category_counts.items():
            percentage = (count / len(self.word_list)) * 100
            print(f"{category:12}: {count:3} Wörter ({percentage:5.1f}%)")
        
        print(f"\nGesamt: {len(self.word_list)} Wörter")
        
        return category_counts
    
    def analyze_complexity_distribution(self):
        """Analysiert die Komplexitätsverteilung"""
        print("\n🎯 KOMPLEXITÄTSVERTEILUNG")
        print("=" * 50)
        
        complexity_counts = Counter([word['complexity'] for word in self.word_list])
        
        for complexity in sorted(complexity_counts.keys()):
            count = complexity_counts[complexity]
            percentage = (count / len(self.word_list)) * 100
            stars = "★" * complexity + "☆" * (5 - complexity)
            print(f"Level {complexity} {stars}: {count:3} Wörter ({percentage:5.1f}%)")
        
        return complexity_counts
    
    def analyze_training_examples(self):
        """Analysiert die Trainingsbeispiele"""
        print("\n📚 TRAININGSBEISPIELE ANALYSE")
        print("=" * 50)
        
        # Analyse nach Typ
        type_counts = Counter([ex['type'] for ex in self.training_examples])
        print("Nach Aufgabentyp:")
        for ex_type, count in type_counts.items():
            percentage = (count / len(self.training_examples)) * 100
            print(f"  {ex_type:20}: {count:4} ({percentage:5.1f}%)")
        
        # Analyse nach Kategorie
        category_counts = Counter([ex['category'] for ex in self.training_examples])
        print("\nNach Kategorie:")
        for category, count in category_counts.items():
            percentage = (count / len(self.training_examples)) * 100
            print(f"  {category:12}: {count:4} ({percentage:5.1f}%)")
        
        print(f"\nGesamt: {len(self.training_examples)} Trainingsbeispiele")
        
        return type_counts, category_counts
    
    def show_sample_data(self, n_samples: int = 5):
        """Zeigt Beispieldaten"""
        print(f"\n📋 BEISPIELDATEN ({n_samples} Samples)")
        print("=" * 50)
        
        print("Wortliste Beispiele:")
        for i, word in enumerate(self.word_list[:n_samples]):
            print(f"  {i+1}. '{word['word']}' ({word['category']}) -> {word['meaning']}")
        
        print(f"\nTrainingsbeispiele:")
        for i, example in enumerate(self.training_examples[:n_samples]):
            print(f"  {i+1}. Q: {example['input']}")
            print(f"     A: {example['output']}")
            print(f"     Type: {example['type']}, Category: {example['category']}")
            print()
    
    def visualize_distributions(self):
        """Erstellt Visualisierungen der Datenverteilungen"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Wörter pro Kategorie
        category_counts = Counter([word['category'] for word in self.word_list])
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        ax1.bar(categories, counts, color='skyblue')
        ax1.set_title('Wörter pro Kategorie')
        ax1.set_xlabel('Kategorie')
        ax1.set_ylabel('Anzahl Wörter')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Komplexitätsverteilung
        complexity_counts = Counter([word['complexity'] for word in self.word_list])
        complexities = list(complexity_counts.keys())
        comp_counts = list(complexity_counts.values())
        
        ax2.bar(complexities, comp_counts, color='lightgreen')
        ax2.set_title('Komplexitätsverteilung')
        ax2.set_xlabel('Komplexitätslevel')
        ax2.set_ylabel('Anzahl Wörter')
        
        # 3. Trainingsbeispiele nach Typ
        type_counts = Counter([ex['type'] for ex in self.training_examples])
        types = list(type_counts.keys())
        type_vals = list(type_counts.values())
        
        ax3.pie(type_vals, labels=types, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Trainingsbeispiele nach Typ')
        
        # 4. Beispiellängen
        input_lengths = [len(ex['input']) for ex in self.training_examples]
        output_lengths = [len(ex['output']) for ex in self.training_examples]
        
        ax4.hist([input_lengths, output_lengths], bins=20, alpha=0.7, 
                label=['Input-Länge', 'Output-Länge'], color=['orange', 'purple'])
        ax4.set_title('Verteilung der Text-Längen')
        ax4.set_xlabel('Zeichen-Anzahl')
        ax4.set_ylabel('Häufigkeit')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('evaluation/dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualisierung gespeichert als 'evaluation/dataset_analysis.png'")
    
    def validate_data_quality(self):
        """Validiert die Datenqualität"""
        print("\n✅ DATENQUALITÄT VALIDIERUNG")
        print("=" * 50)
        
        issues = []
        
        # 1. Prüfe auf leere Einträge
        empty_words = [w for w in self.word_list if not w['word'].strip()]
        if empty_words:
            issues.append(f"Gefunden: {len(empty_words)} leere Wörter")
        
        # 2. Prüfe auf doppelte Wörter
        words = [w['word'] for w in self.word_list]
        duplicates = [w for w in set(words) if words.count(w) > 1]
        if duplicates:
            issues.append(f"Gefunden: {len(duplicates)} doppelte Wörter")
        
        # 3. Prüfe Trainingsbeispiele auf leere Inputs/Outputs
        empty_inputs = [ex for ex in self.training_examples if not ex['input'].strip()]
        empty_outputs = [ex for ex in self.training_examples if not ex['output'].strip()]
        
        if empty_inputs:
            issues.append(f"Gefunden: {len(empty_inputs)} leere Inputs")
        if empty_outputs:
            issues.append(f"Gefunden: {len(empty_outputs)} leere Outputs")
        
        # 4. Prüfe auf extreme Längen
        very_long_inputs = [ex for ex in self.training_examples if len(ex['input']) > 200]
        very_short_outputs = [ex for ex in self.training_examples if len(ex['output']) < 2]
        
        if very_long_inputs:
            issues.append(f"Warnung: {len(very_long_inputs)} sehr lange Inputs (>200 Zeichen)")
        if very_short_outputs:
            issues.append(f"Warnung: {len(very_short_outputs)} sehr kurze Outputs (<2 Zeichen)")
        
        # Ergebnisse
        if not issues:
            print("🎉 Keine Probleme gefunden! Datenqualität ist gut.")
        else:
            print("⚠️  Gefundene Probleme:")
            for issue in issues:
                print(f"  - {issue}")
        
        return len(issues) == 0
    
    def generate_report(self):
        """Erstellt einen vollständigen Analysebericht"""
        print("🔍 VOLLSTÄNDIGER DATENSATZ-ANALYSEBERICHT")
        print("=" * 60)
        
        # Führe alle Analysen durch
        word_dist = self.analyze_word_distribution()
        comp_dist = self.analyze_complexity_distribution()
        type_dist, cat_dist = self.analyze_training_examples()
        self.show_sample_data()
        is_valid = self.validate_data_quality()
        
        # Erstelle Visualisierungen
        try:
            self.visualize_distributions()
        except Exception as e:
            print(f"⚠️  Konnte Visualisierungen nicht erstellen: {e}")
        
        # Zusammenfassung
        print(f"\n📝 ZUSAMMENFASSUNG")
        print("=" * 50)
        print(f"✓ {len(self.word_list)} Wörter in {len(word_dist)} Kategorien")
        print(f"✓ {len(self.training_examples)} Trainingsbeispiele")
        print(f"✓ Komplexitätslevel: {min(comp_dist.keys())} bis {max(comp_dist.keys())}")
        print(f"✓ Datenqualität: {'Gut' if is_valid else 'Probleme gefunden'}")
        
        return {
            'word_count': len(self.word_list),
            'training_examples_count': len(self.training_examples),
            'categories': len(word_dist),
            'is_valid': is_valid
        }

def main():
    """Hauptfunktion für die Datensatz-Analyse"""
    try:
        analyzer = DatasetAnalyzer(
            'data/word_list.json',
            'data/training_examples.json'
        )
        
        report = analyzer.generate_report()
        
        print(f"\n🎯 BEREIT FÜR PHASE 3!")
        print("Der Datensatz ist generiert und analysiert.")
        print("Sie können jetzt mit dem Modellaufbau beginnen.")
        
    except FileNotFoundError as e:
        print(f"❌ Datei nicht gefunden: {e}")
        print("Bitte führen Sie zuerst 'python3 data/create_dataset.py' aus.")
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")

if __name__ == "__main__":
    main()
