"""
Datensatz-Generator für das Sprachmodell-Experiment
Erstellt eine strukturierte Wortliste mit verschiedenen Bedeutungsgruppen
"""
import json
import csv
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class WordEntry:
    """Struktur für einen Wortliste-Eintrag"""
    word: str
    category: str
    meaning: str
    complexity: int  # 1-5, wobei 5 am komplexesten ist
    examples: List[str]

class DatasetGenerator:
    """Generator für strukturierte Wortlisten und Trainingsbeispiele"""
    
    def __init__(self):
        self.word_categories = {
            "farben": {
                "words": ["rot", "blau", "grün", "gelb", "orange", "violett", "schwarz", "weiß", "rosa", "braun"],
                "meanings": ["Farbe des Blutes", "Farbe des Himmels", "Farbe der Blätter", "Farbe der Sonne", 
                           "Mischfarbe rot-gelb", "Mischfarbe rot-blau", "Abwesenheit von Licht", "Alle Farben gemischt",
                           "Helle rote Farbe", "Farbe der Erde"],
                "complexity": 1
            },
            "bewegungen": {
                "words": ["laufen", "springen", "gehen", "rennen", "tanzen", "schwimmen", "fliegen", "kriechen", "rollen", "gleiten"],
                "meanings": ["Sich schnell fortbewegen", "Sich in die Luft bewegen", "Sich langsam fortbewegen", 
                           "Sehr schnell laufen", "Rhythmische Bewegung", "Sich im Wasser fortbewegen",
                           "Sich durch die Luft bewegen", "Sich am Boden fortbewegen", "Sich drehend bewegen", "Reibungslos bewegen"],
                "complexity": 2
            },
            "objekte": {
                "words": ["haus", "auto", "baum", "buch", "tisch", "stuhl", "computer", "telefon", "lampe", "fenster"],
                "meanings": ["Gebäude zum Wohnen", "Fahrzeug für Straßen", "Große Pflanze mit Stamm", 
                           "Sammlung von Seiten", "Möbel zum Arbeiten", "Möbel zum Sitzen", "Elektronisches Gerät",
                           "Gerät zur Kommunikation", "Gerät für Licht", "Öffnung in der Wand"],
                "complexity": 1
            },
            "aktionen": {
                "words": ["denken", "lernen", "verstehen", "erklären", "analysieren", "bewerten", "erschaffen", "lösen", "planen", "entscheiden"],
                "meanings": ["Geistige Tätigkeit", "Wissen aufnehmen", "Bedeutung erfassen", "Verständlich machen",
                           "Systematisch untersuchen", "Qualität beurteilen", "Etwas Neues schaffen", "Problem beheben",
                           "Vorgehensweise festlegen", "Wahl treffen"],
                "complexity": 4
            },
            "zustände": {
                "words": ["glücklich", "traurig", "müde", "wach", "hungrig", "satt", "kalt", "warm", "gesund", "krank"],
                "meanings": ["Positive Emotion", "Negative Emotion", "Bedürfnis nach Schlaf", "Nicht schläfrig",
                           "Bedürfnis nach Nahrung", "Gefühl nach dem Essen", "Niedrige Temperatur", "Hohe Temperatur",
                           "Körperlich in Ordnung", "Körperlich nicht in Ordnung"],
                "complexity": 2
            },
            "zahlen": {
                "words": ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn"],
                "meanings": ["Zahl 1", "Zahl 2", "Zahl 3", "Zahl 4", "Zahl 5", "Zahl 6", "Zahl 7", "Zahl 8", "Zahl 9", "Zahl 10"],
                "complexity": 1
            },
            "logik": {
                "words": ["wenn", "dann", "oder", "und", "nicht", "alle", "einige", "weil", "deshalb", "obwohl"],
                "meanings": ["Bedingung", "Folge", "Alternative", "Verbindung", "Negation", "Vollständige Menge",
                           "Teilmenge", "Begründung", "Schlussfolgerung", "Gegensatz"],
                "complexity": 5
            }
        }
    
    def generate_word_list(self, target_size: int = 1000) -> List[WordEntry]:
        """Generiert eine ausbalancierte Wortliste"""
        word_entries = []
        
        # Berechne Anzahl Wörter pro Kategorie
        categories = list(self.word_categories.keys())
        words_per_category = target_size // len(categories)
        
        for category_name, category_data in self.word_categories.items():
            base_words = category_data["words"]
            meanings = category_data["meanings"]
            complexity = category_data["complexity"]
            
            # Erweitere Wortliste durch Variationen
            extended_words = self._extend_word_list(base_words, words_per_category)
            extended_meanings = self._extend_meanings(meanings, len(extended_words))
            
            for i, word in enumerate(extended_words):
                meaning = extended_meanings[i % len(extended_meanings)]
                examples = self._generate_examples(word, category_name)
                
                entry = WordEntry(
                    word=word,
                    category=category_name,
                    meaning=meaning,
                    complexity=complexity,
                    examples=examples
                )
                word_entries.append(entry)
        
        # Mische die Liste für bessere Verteilung
        random.shuffle(word_entries)
        return word_entries[:target_size]
    
    def _extend_word_list(self, base_words: List[str], target_size: int) -> List[str]:
        """Erweitert eine Grundwortliste durch Variationen"""
        extended = base_words.copy()
        
        # Füge Pluralformen hinzu
        for word in base_words:
            if not word.endswith('en'):
                extended.append(word + "e")  # Einfache Pluralbildung
        
        # Füge zusammengesetzte Wörter hinzu
        for i, word1 in enumerate(base_words):
            for j, word2 in enumerate(base_words):
                if i != j and len(extended) < target_size:
                    extended.append(word1 + word2)
        
        # Fülle mit Variationen auf
        while len(extended) < target_size:
            base_word = random.choice(base_words)
            extended.append(base_word + "_var" + str(len(extended)))
        
        return extended[:target_size]
    
    def _extend_meanings(self, base_meanings: List[str], target_size: int) -> List[str]:
        """Erweitert Bedeutungsliste"""
        extended = base_meanings.copy()
        
        while len(extended) < target_size:
            base_meaning = random.choice(base_meanings)
            extended.append(base_meaning + " (Variante)")
        
        return extended
    
    def _generate_examples(self, word: str, category: str) -> List[str]:
        """Generiert Beispielsätze für ein Wort"""
        templates = {
            "farben": [f"Das ist {word}.", f"Ich sehe etwas {word}es.", f"{word.capitalize()} ist eine Farbe."],
            "bewegungen": [f"Ich kann {word}.", f"Wir {word} zusammen.", f"{word.capitalize()} ist eine Bewegung."],
            "objekte": [f"Das ist ein {word}.", f"Ich brauche einen {word}.", f"Der {word} ist hier."],
            "aktionen": [f"Ich {word} darüber.", f"Wir müssen {word}.", f"{word.capitalize()} ist wichtig."],
            "zustände": [f"Ich bin {word}.", f"Du siehst {word} aus.", f"{word.capitalize()} fühlen ist normal."],
            "zahlen": [f"Ich zähle bis {word}.", f"{word.capitalize()} Dinge.", f"Das sind {word} Objekte."],
            "logik": [f"{word.capitalize()} das stimmt.", f"Das ist {word} wahr.", f"{word.capitalize()} bedeutet..."]
        }
        
        return templates.get(category, [f"Das Wort ist {word}.", f"{word.capitalize()} ist ein Begriff."])
    
    def create_training_examples(self, word_entries: List[WordEntry]) -> List[Dict]:
        """Erstellt Trainingsbeispiele im JSON-Format"""
        training_examples = []
        
        for entry in word_entries:
            # Einfache Wort-zu-Bedeutung Beispiele
            training_examples.append({
                "input": f"Was bedeutet '{entry.word}'?",
                "output": entry.meaning,
                "category": entry.category,
                "complexity": entry.complexity,
                "type": "word_meaning"
            })
            
            # Kategorie-Klassifikation
            training_examples.append({
                "input": f"Zu welcher Kategorie gehört '{entry.word}'?",
                "output": entry.category,
                "category": entry.category,
                "complexity": entry.complexity,
                "type": "categorization"
            })
            
            # Beispielsatz-Generierung
            for example in entry.examples[:2]:  # Nur erste 2 Beispiele
                training_examples.append({
                    "input": f"Verwende '{entry.word}' in einem Satz.",
                    "output": example,
                    "category": entry.category,
                    "complexity": entry.complexity,
                    "type": "sentence_generation"
                })
        
        # Logische Aufgaben hinzufügen
        logical_examples = self._create_logical_tasks()
        training_examples.extend(logical_examples)
        
        return training_examples
    
    def _create_logical_tasks(self) -> List[Dict]:
        """Erstellt logische Denkaufgaben"""
        logical_tasks = [
            {
                "input": "Wenn es regnet, dann ist die Straße nass. Es regnet. Was folgt daraus?",
                "output": "Die Straße ist nass.",
                "category": "logik",
                "complexity": 3,
                "type": "logical_reasoning"
            },
            {
                "input": "Alle Hunde sind Tiere. Bello ist ein Hund. Was ist Bello?",
                "output": "Bello ist ein Tier.",
                "category": "logik", 
                "complexity": 3,
                "type": "logical_reasoning"
            },
            {
                "input": "Rot oder Blau. Nicht Rot. Was bleibt übrig?",
                "output": "Blau.",
                "category": "logik",
                "complexity": 2,
                "type": "logical_reasoning"
            }
        ]
        
        return logical_tasks
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Speichert Daten als JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Speichert Daten als CSV"""
        if not data:
            return
        
        fieldnames = data[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

# Hauptfunktion zum Generieren des Datensatzes
def generate_complete_dataset():
    """Generiert den kompletten Datensatz"""
    generator = DatasetGenerator()
    
    print("🚀 Generiere Wortliste...")
    word_entries = generator.generate_word_list(1000)
    
    print("📝 Erstelle Trainingsbeispiele...")
    training_examples = generator.create_training_examples(word_entries)
    
    print("💾 Speichere Datensätze...")
    
    # Wortliste speichern
    word_data = [
        {
            "word": entry.word,
            "category": entry.category,
            "meaning": entry.meaning,
            "complexity": entry.complexity,
            "examples": entry.examples
        }
        for entry in word_entries
    ]
    
    generator.save_to_json(word_data, "data/word_list.json")
    generator.save_to_csv(word_data, "data/word_list.csv")
    
    # Trainingsbeispiele speichern
    generator.save_to_json(training_examples, "data/training_examples.json")
    generator.save_to_csv(training_examples, "data/training_examples.csv")
    
    print(f"✅ Datensatz erstellt!")
    print(f"📊 {len(word_entries)} Wörter generiert")
    print(f"📚 {len(training_examples)} Trainingsbeispiele erstellt")
    print(f"📁 Dateien gespeichert in /data/")
    
    return word_entries, training_examples

if __name__ == "__main__":
    generate_complete_dataset()
