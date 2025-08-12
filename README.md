# Sprachmodell-Experiment mit Neuronengruppen

## Übersicht
Dieses Experiment untersucht die Auswirkungen variierender Neuronenzahlen in Sprachmodellen und entwickelt Schutzmechanismen gegen Prompt-Hacking.

## Projektstruktur

```
/data          - Rohdaten und Datensätze
/model         - Modellarchitektur und Trainingsskripte  
/evaluation    - Tests & Visualisierung
/security      - Prompt-Hacking-Schutzmechanismen
```

## Experimentziele

1. **Neuronengruppen-Analyse**: Untersuchung wie verschiedene Neuronenzahlen die Modellqualität beeinflussen
2. **Prompt-Hacking-Resistenz**: Entwicklung robuster Schutzmechanismen
3. **Logische Trennung**: Implementierung separater Verarbeitungsschritte

## Phasen

### Phase 1 - Vorbereitungen ✅
- [x] Projektstruktur angelegt
- [ ] Prompt-Hacking-Schutzplan definiert

### Phase 2 - Datensatz-Erstellung
- [ ] Wortliste erzeugen (ca. 1000 Wörter)
- [ ] Datensatz-Format festlegen

### Phase 3 - Modellaufbau  
- [ ] Basismodell entwerfen
- [ ] Neuronen in logische Gruppen clustern

## Getting Started

1. Installiere die Abhängigkeiten: `pip install -r requirements.txt`
2. Führe die Datensatz-Erstellung aus: `python data/create_dataset.py`
3. Starte das Training: `python model/train.py`

## Sicherheitshinweise

Dieses Experiment implementiert spezielle Schutzmechanismen gegen manipulative Eingaben. Siehe `/security` für Details.
