# VVVA F1 Predictor

Système de prédiction de résultats de Formule 1 basé sur 15 ans de données réelles (2009–2024).
Combine un simulateur de Grand Prix et un modèle de machine learning (Random Forest) avec analyse de l'impact météorologique.

---

## Prérequis

- **Python 3.9+**
- pip

```bash
python --version   # doit afficher 3.9 ou supérieur
```

---

## Installation

```bash
git clone <url-du-repo>
cd vvva
pip install -r requirements.txt
```

---

## Lancement

Depuis la **racine du projet** (ne pas lancer depuis `src/`) :

```bash
python -m streamlit run src/dashboard.py
```

Le dashboard s'ouvre sur `http://localhost:8501`.

---

## Fonctionnalités

| Onglet | Description |
|--------|-------------|
| **Simulateur** | Simule une course : circuit, météo → classement avec DNF possibles |
| **Prédicteur ML** | Prédit les positions 2025 ou affiche les vrais résultats historiques |
| **Statistiques** | Circuits, répartition par pays, analyse de l'impact météo |

### Modèle de simulation — Pondération

| Paramètre | Poids | Source |
|-----------|-------|--------|
| Force de la voiture (constructeur) | **40%** | `constructor_standings.csv` |
| Forme récente du pilote | **35%** | `f1_data.csv` |
| Historique sur le circuit | **15%** | `f1_data.csv` |
| Compétence pure du pilote | **10%** | Calculée (pilote vs voiture) |

### Impact météo — différencié par pilote

| Conditions | Vitesse | Amplificateur skill | DNF (par écurie) |
|------------|---------|---------------------|-----------------|
| ☀️ Ensoleillé | ×1.00 | ×1.0 | taux historique |
| ⛅ Nuageux | ×0.98 | ×1.1 | +léger |
| 🌧️ Pluie | ×0.85 | ×1.8 | +fort |

En pluie : la voiture pèse moins (25% au lieu de 40%), le skill pilote est amplifié ×1.8.
Les pilotes qui surpassent historiquement leur voiture bénéficient davantage du chaos sous la pluie.
Le risque de DNF est calculé **par écurie** d'après leur taux de fiabilité historique.

---

## Structure du projet

```
vvva/
├── csv/
│   ├── f1_data.csv               # Données F1 fusionnées (2009–2024)
│   ├── driver_standings.csv      # Points/positions championnat pilotes
│   ├── constructor_standings.csv # Points/positions championnat constructeurs
│   ├── qualifying.csv            # Positions de qualification
│   ├── status.csv                # Statuts de fin de course (DNF, etc.)
│   └── ...                       # Autres CSV Ergast
├── src/
│   ├── data_loader.py      # Chargement et préparation des données
│   ├── predictor.py        # Modèle ML Random Forest
│   ├── simulator.py        # Simulateur de Grand Prix (météo + DNF)
│   └── dashboard.py        # Interface Streamlit
├── docs/
│   └── DOCUMENTATION.md
├── requirements.txt
└── README.md
```

---

## Source des données

`csv/f1_data.csv` — dataset **Ergast F1** fusionné (pilotes, équipes, circuits, résultats 2009–2024), disponible sur Kaggle.

---

## Modèle ML — Features utilisées

| Feature | Source CSV | Description |
|---------|-----------|-------------|
| `quali_pos` | `qualifying.csv` | Position de qualification (plus précise que la grille) |
| `grid_position` | `f1_data.csv` | Position sur la grille de départ |
| `circuit_encoded` | `f1_data.csv` | Circuit encodé numériquement |
| `driver_age` | `f1_data.csv` | Âge du pilote |
| `year` | `f1_data.csv` | Année de la saison |
| `driver_champ_points` | `driver_standings.csv` | Points au championnat avant la course |
| `driver_champ_pos` | `driver_standings.csv` | Position au championnat avant la course |
| `driver_wins` | `driver_standings.csv` | Victoires cumulées avant la course |
| `constructor_points` | `constructor_standings.csv` | Points constructeur (= force de la voiture) |
| `constructor_pos` | `constructor_standings.csv` | Position constructeur au championnat |
| `dnf_rate` | `status.csv` | Taux d'abandon sur les 10 dernières courses |

Les standings utilisent les données de la course **précédente** pour éviter la fuite de données.

## Technologies

- **Streamlit** — dashboard interactif
- **Plotly** — visualisations dynamiques
- **Pandas / NumPy** — manipulation des données
- **Scikit-learn** — Random Forest (score R² typique : 0.6–0.7)

---

## Utilisation programmatique

```python
from src.simulator import F1Simulator

sim = F1Simulator()
results = sim.simulate_race(circuit_id=1, weather="rain", year=2025)
print(results)
```

```python
from src.predictor import F1Predictor

predictor = F1Predictor()
scores = predictor.train()
print(f"Score test : {scores['test_score']:.3f}")
```

---

## Auteurs

Projet VVVA — Epitech
