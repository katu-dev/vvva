# Documentation complète — Projet VVVA F1 Predictor

> Documentation en français, ligne par ligne, avec schémas.

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Structure des fichiers](#2-structure-des-fichiers)
3. [Les données CSV](#3-les-données-csv)
4. [data_loader.py — Chargement des données](#4-data_loaderpy--chargement-des-données)
5. [predictor.py — Modèle Machine Learning](#5-predictorpy--modèle-machine-learning)
6. [simulator.py — Simulateur de course](#6-simulatorpy--simulateur-de-course)
7. [dashboard.py — Interface visuelle](#7-dashboardpy--interface-visuelle)
8. [Schéma global du flux de données](#8-schéma-global-du-flux-de-données)
9. [Comment lancer le projet](#9-comment-lancer-le-projet)

---

## 1. Vue d'ensemble du projet

Le projet VVVA est un **système de prédiction de courses de Formule 1**.
Il utilise de vraies données historiques F1 (2009 → 2024) pour :

- **Simuler** une course GP avec des effets météo
- **Prédire** les positions finales grâce à un modèle IA (Random Forest)
- **Visualiser** tout ça dans un tableau de bord interactif

```
┌─────────────────────────────────────────────────────────┐
│                    PROJET VVVA F1                        │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  CSV F1  │───▶│ data_loader  │───▶│   simulator   │  │
│  │ (données)│    │  (lecture)   │    │  (simulation) │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│                         │                    │           │
│                         │            ┌───────────────┐  │
│                         └───────────▶│   predictor   │  │
│                                      │  (IA Random   │  │
│                                      │   Forest)     │  │
│                                      └───────────────┘  │
│                                              │           │
│                                      ┌───────────────┐  │
│                                      │   dashboard   │  │
│                                      │  (Streamlit)  │  │
│                                      └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Structure des fichiers

```
vvva/
│
├── csv/                      ← Données historiques F1 (fichiers bruts)
│   ├── races.csv             ← Toutes les courses (id, année, circuit...)
│   ├── results.csv           ← Résultats de chaque pilote par course
│   ├── drivers.csv           ← Infos sur chaque pilote (nom, date de naissance...)
│   ├── circuits.csv          ← Infos sur chaque circuit (nom, pays...)
│   ├── constructors.csv      ← Équipes (Ferrari, Mercedes, Red Bull...)
│   ├── qualifying.csv        ← Résultats de qualifications
│   ├── lap_times.csv         ← Temps au tour
│   ├── pit_stops.csv         ← Arrêts aux stands
│   └── ...                   ← Autres données (classements, sprints...)
│
├── src/
│   ├── data_loader.py        ← Lit et prépare les CSV
│   ├── predictor.py          ← Modèle IA (Random Forest)
│   ├── simulator.py          ← Simule une course
│   └── dashboard.py          ← Interface web (Streamlit)
│
├── requirements.txt          ← Liste des bibliothèques Python nécessaires
└── README.md                 ← Ce fichier
```

---

## 3. Les données CSV

Les fichiers CSV contiennent 15 ans de données F1 réelles. Voici comment ils sont liés :

```
┌─────────────┐         ┌─────────────────┐         ┌─────────────┐
│  drivers    │         │    results      │         │    races    │
│─────────────│         │─────────────────│         │─────────────│
│ driverId PK │◀────────│ driverId FK     │────────▶│ raceId PK   │
│ forename    │         │ raceId FK       │         │ year        │
│ surname     │         │ constructorId FK│         │ circuitId FK│
│ code        │         │ position        │         │ name        │
│ dob         │         │ grid            │         │ date        │
└─────────────┘         │ points          │         └──────┬──────┘
                        └─────────────────┘                │
┌──────────────────┐                              ┌────────▼────┐
│  constructors    │                              │  circuits   │
│──────────────────│                              │─────────────│
│ constructorId PK │◀─────────────────────────── │ circuitId PK│
│ name (équipe)    │   (via results)              │ name        │
└──────────────────┘                              │ location    │
                                                  │ country     │
                                                  └─────────────┘
```

**Colonnes importantes de `results.csv` :**
| Colonne | Signification |
|---------|--------------|
| `raceId` | Identifiant de la course |
| `driverId` | Identifiant du pilote |
| `constructorId` | Identifiant de l'équipe |
| `grid` | Position de départ (en qualif) |
| `position` | Position d'arrivée finale |
| `points` | Points marqués lors de la course |

---

## 4. `data_loader.py` — Chargement des données

Ce fichier est la **base de tout le projet**. Il lit les CSV et fournit des données prêtes à l'emploi aux autres modules.

```python
import pandas as pd      # Bibliothèque pour manipuler des tableaux de données (comme Excel en Python)
import numpy as np       # Bibliothèque pour les calculs mathématiques
from pathlib import Path # Permet de gérer les chemins de fichiers proprement (ex: csv/races.csv)
```

### La classe `F1DataLoader`

```
┌──────────────────────────────────────────────┐
│              F1DataLoader                    │
│──────────────────────────────────────────────│
│ Attributs :                                  │
│   data_path    → chemin vers le dossier csv/ │
│   races        → tableau des courses         │
│   results      → tableau des résultats       │
│   drivers      → tableau des pilotes         │
│   circuits     → tableau des circuits        │
│   constructors → tableau des équipes         │
│──────────────────────────────────────────────│
│ Méthodes :                                   │
│   __init__()           → initialise et charge│
│   load_data()          → lit les 5 CSV       │
│   get_driver_stats()   → stats d'un pilote   │
│   get_circuit_history()→ historique circuit  │
│   prepare_training_data()→ données pour l'IA │
│   get_recent_form()    → forme récente pilote│
└──────────────────────────────────────────────┘
```

### `__init__` — Constructeur

```python
def __init__(self, data_path='csv'):
    self.data_path = Path(data_path)  # Convertit 'csv' en chemin réel (ex: /projet/csv/)
    self.races = None                 # Ces variables sont vides pour l'instant (None = pas encore chargé)
    self.results = None
    self.drivers = None
    self.circuits = None
    self.constructors = None
    self.load_data()                  # Appelle immédiatement load_data() pour remplir ces variables
```

> Quand on crée un `F1DataLoader()`, Python appelle automatiquement `__init__`.
> Dès le départ, les 5 CSV sont lus et stockés en mémoire.

### `load_data` — Lecture des CSV

```python
def load_data(self):
    self.races = pd.read_csv(self.data_path / 'races.csv')
    # pd.read_csv() lit un fichier CSV et le transforme en tableau (DataFrame)
    # self.data_path / 'races.csv'  →  csv/races.csv

    self.results = pd.read_csv(self.data_path / 'results.csv')
    self.drivers = pd.read_csv(self.data_path / 'drivers.csv')
    self.circuits = pd.read_csv(self.data_path / 'circuits.csv')
    self.constructors = pd.read_csv(self.data_path / 'constructors.csv')
```

> Après cette méthode, chaque attribut (`self.races`, etc.) est un tableau avec toutes les lignes du CSV.

### `get_driver_stats` — Statistiques d'un pilote

```python
def get_driver_stats(self, driver_id):
    # Filtre results pour ne garder que les lignes du pilote demandé
    driver_results = self.results[self.results['driverId'] == driver_id]

    # Convertit la colonne 'position' en nombres
    # errors='coerce' → si une valeur n'est pas un nombre (ex: 'R' pour abandon), elle devient NaN
    positions = pd.to_numeric(driver_results['position'], errors='coerce')

    return {
        'total_races': len(driver_results),          # Nombre total de courses disputées
        'wins':        len(positions[positions == 1]),   # Nombre de victoires (position = 1)
        'podiums':     len(positions[positions <= 3]),   # Podiums (position 1, 2 ou 3)
        'avg_position': positions.mean(),            # Moyenne des positions (.mean() = somme / nb)
        'total_points': driver_results['points'].sum()   # Total de points marqués (.sum() = addition)
    }
```

### `get_circuit_history` — Historique d'un circuit

```python
def get_circuit_history(self, circuit_id, limit=10):
    # Récupère les 'limit' dernières courses sur ce circuit
    circuit_races = self.races[self.races['circuitId'] == circuit_id].tail(limit)
    # .tail(10) = les 10 dernières lignes du tableau

    race_ids = circuit_races['raceId'].values  # Extrait uniquement les IDs des courses sélectionnées

    # Récupère tous les résultats de ces courses
    history = self.results[self.results['raceId'].isin(race_ids)]
    # .isin([...]) = "est dans cette liste ?"

    # Ajoute les infos pilotes (nom, code...) au tableau
    history = history.merge(self.drivers, on='driverId', suffixes=('', '_driver'))
    # .merge() = jointure SQL : combine deux tableaux sur une colonne commune

    # Ajoute les infos de course (date, année...)
    history = history.merge(circuit_races, on='raceId', suffixes=('', '_race'))
    return history
```

### `prepare_training_data` — Données pour l'IA

C'est la méthode la plus importante pour le modèle ML. Elle prépare un tableau "propre" que l'IA peut apprendre.

```python
def prepare_training_data(self):
    # Étape 1 : Fusionner tous les tableaux en un seul grand tableau
    data = self.results.merge(self.races, on='raceId', suffixes=('', '_race'))
    # suffixes=('', '_race') : si deux colonnes ont le même nom, la 2ème reçoit le suffixe '_race'
    data = data.merge(self.drivers, on='driverId', suffixes=('', '_driver'))
    data = data.merge(self.circuits, on='circuitId', suffixes=('', '_circuit'))
    data = data.merge(self.constructors, on='constructorId', suffixes=('', '_constructor'))

    # Étape 2 : Calculer l'âge du pilote au moment de la course
    data['driver_age'] = pd.to_datetime(data['date']).dt.year - pd.to_datetime(data['dob']).dt.year
    # pd.to_datetime() convertit une chaîne "2024-03-15" en date Python
    # .dt.year extrait uniquement l'année

    # Étape 3 : Position de départ (grid), avec 20 si la valeur est manquante
    data['grid_position'] = data['grid'].fillna(20)
    # .fillna(20) remplace les cases vides (NaN) par 20

    # Étape 4 : Transformer les IDs de circuit en nombres (0, 1, 2, 3...)
    data['circuit_encoded'] = data['circuitId'].astype('category').cat.codes
    # astype('category') : dit à pandas que c'est une catégorie
    # .cat.codes : assigne un nombre unique à chaque catégorie

    # Étape 5 : Définir les colonnes d'entrée (X) et la colonne cible (y)
    features = ['grid_position', 'circuit_encoded', 'driver_age', 'year']
    target = 'position'   # Ce qu'on veut prédire : la position finale

    # Étape 6 : Nettoyer les données invalides
    data['position'] = pd.to_numeric(data['position'], errors='coerce')
    clean_data = data[features + [target]].dropna()
    # .dropna() supprime toutes les lignes qui ont au moins une valeur manquante
    clean_data = clean_data[clean_data['position'] > 0]    # Élimine positions négatives ou nulles
    clean_data = clean_data[clean_data['position'] <= 30]  # Élimine positions aberrantes (> 30)

    return clean_data[features], clean_data[target]
    # Retourne un tuple : (tableau des entrées X, colonne cible y)
```

```
Schéma du tableau final préparé :

┌──────────────┬──────────────────┬────────────┬──────┬──────────┐
│ grid_position│ circuit_encoded  │ driver_age │ year │ position │
│    (entrée)  │    (entrée)      │  (entrée)  │(ent) │  (cible) │
├──────────────┼──────────────────┼────────────┼──────┼──────────┤
│      1       │       12         │     28     │ 2024 │    1     │
│      5       │       12         │     35     │ 2024 │    3     │
│     12       │        7         │     22     │ 2023 │    8     │
│     ...      │       ...        │    ...     │ ...  │   ...    │
└──────────────┴──────────────────┴────────────┴──────┴──────────┘
```

### `get_recent_form` — Forme récente d'un pilote

```python
def get_recent_form(self, driver_id, n_races=5, year=None):
    driver_results = self.results[self.results['driverId'] == driver_id]

    if year is not None:
        # Filtre pour ne prendre que les courses jusqu'à l'année demandée
        race_ids = self.races[self.races['year'] <= year]['raceId']
        driver_results = driver_results[driver_results['raceId'].isin(race_ids)]

    driver_results = driver_results.tail(n_races)  # Garde uniquement les n dernières courses

    if len(driver_results) == 0:
        return 0.5  # Si aucune donnée, retourne une valeur neutre (50%)

    positions = pd.to_numeric(driver_results['position'], errors='coerce').fillna(20)
    # Les abandons (NaN) sont traités comme une 20ème place

    return 1 - (positions.mean() / 20)
    # Formule : si moyenne = 1  → 1 - (1/20)  = 0.95 (excellent)
    #           si moyenne = 10 → 1 - (10/20) = 0.50 (moyen)
    #           si moyenne = 20 → 1 - (20/20) = 0.00 (terrible)
```

```
Calcul de la forme récente :

Position moyenne des 5 dernières courses
        ↓
  forme = 1 - (moyenne / 20)

  Position 1  → forme = 0.95  ████████████████████
  Position 5  → forme = 0.75  ███████████████
  Position 10 → forme = 0.50  ██████████
  Position 15 → forme = 0.25  █████
  Position 20 → forme = 0.00  (abandon / dernier)
```

---

## 5. `predictor.py` — Modèle Machine Learning

Ce fichier contient le **cerveau IA** du projet. Il apprend des données historiques pour prédire des positions.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# RandomForestRegressor = algorithme IA "Forêt Aléatoire" pour prédire des nombres
from sklearn.model_selection import train_test_split
# train_test_split = divise les données en deux : entraînement + test
from data_loader import F1DataLoader  # Importe notre propre module
```

### C'est quoi un Random Forest ?

```
Un Random Forest = plusieurs arbres de décision qui votent ensemble

Exemple d'un arbre :
                    [grid_position ≤ 3 ?]
                    /                  \
               OUI                    NON
        [circuit == Monaco ?]    [driver_age ≤ 25 ?]
        /           \             /              \
   position≈1   position≈3  position≈5      position≈8

Le Random Forest crée 100 arbres comme ça (avec des variations),
puis fait la MOYENNE de toutes les prédictions.
→ Plus robuste qu'un seul arbre !
```

### La classe `F1Predictor`

```python
class F1Predictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # n_estimators=100 : crée 100 arbres de décision
        # random_state=42 : graine aléatoire fixe (pour reproduire les mêmes résultats)
        self.data_loader = F1DataLoader()  # Crée un chargeur de données
        self.is_trained = False            # Indique si le modèle a déjà été entraîné
        self.feature_names = None          # Noms des colonnes d'entrée (sauvegardés pour plus tard)
```

### `train` — Entraînement du modèle

```python
def train(self):
    X, y = self.data_loader.prepare_training_data()
    # X = tableau des entrées (grid, circuit, âge, année)
    # y = colonne cible (position finale)

    self.feature_names = X.columns.tolist()  # Sauvegarde les noms des colonnes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Coupe les données : 80% pour entraîner, 20% pour tester
    # test_size=0.2 → 20% de test

    self.model.fit(X_train, y_train)  # Entraîne le modèle sur les données d'entraînement
    self.is_trained = True

    train_score = self.model.score(X_train, y_train)
    test_score  = self.model.score(X_test, y_test)
    # .score() retourne le R² (entre 0 et 1)
    # R² = 1.0 → parfait | R² = 0.0 → le modèle ne prédit rien

    return {
        'train_score': train_score,
        'test_score':  test_score,
        'n_samples':   len(X)
    }
```

```
Schéma entraînement / test :

Toutes les données (ex: 50 000 lignes)
┌────────────────────────────────────────────────────────┐
│  80% — données d'entraînement          │ 20% test      │
│  Le modèle APPREND sur ces données     │ On VÉRIFIE    │
│  (il voit les réponses)                │ les prédictions│
└────────────────────────────────────────────────────────┘
```

### `predict_position` — Prédiction d'une position

```python
def predict_position(self, grid_position, circuit_id, driver_age, year):
    if not self.is_trained:
        raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        # raise ValueError = déclenche une erreur si on appelle predict sans train

    features = pd.DataFrame({
        'grid_position':   [grid_position],
        'circuit_encoded': [circuit_id],
        'driver_age':      [driver_age],
        'year':            [year]
    })
    # Crée un mini-tableau avec les données du pilote à prédire
    # Les [] sont importants : pd.DataFrame attend des listes

    prediction = self.model.predict(features)[0]
    # .predict() retourne une liste, [0] prend le premier (et seul) résultat

    return max(1, min(20, int(round(prediction))))
    # round() : arrondit à l'entier le plus proche (ex: 3.7 → 4)
    # int()   : convertit en entier
    # min(20, ...) : la position ne peut pas dépasser 20
    # max(1, ...)  : la position ne peut pas être inférieure à 1
```

### `get_feature_importance` — Importance des variables

```python
def get_feature_importance(self):
    if not self.is_trained:
        return None

    return pd.DataFrame({
        'feature':    self.feature_names,
        'importance': self.model.feature_importances_
        # feature_importances_ = attribut automatique du RandomForest
        # indique quel % de précision vient de chaque colonne
    }).sort_values('importance', ascending=False)
    # .sort_values() trie du plus important au moins important
```

```
Exemple de résultat :
┌──────────────────┬────────────┐
│ feature          │ importance │
├──────────────────┼────────────┤
│ grid_position    │   0.52     │  ← La position de départ est la plus importante !
│ circuit_encoded  │   0.25     │
│ year             │   0.15     │
│ driver_age       │   0.08     │
└──────────────────┴────────────┘
```

---

## 6. `simulator.py` — Simulateur de course

Ce fichier simule le déroulement complet d'un Grand Prix en se basant sur les données historiques et la météo.

```python
import numpy as np
import pandas as pd
from typing import Dict, List   # Pour indiquer les types dans les définitions de fonctions
from data_loader import F1DataLoader
```

### Les impacts météo

```python
self.weather_impact = {
    'sunny': {'speed': 1.0, 'reliability': 0.95},
    # Conditions normales : vitesse 100%, fiabilité 95%
    'cloudy': {'speed': 0.98, 'reliability': 0.93},
    # Nuageux : légèrement plus lent et moins fiable
    'rain': {'speed': 0.85, 'reliability': 0.80}
    # Pluie : beaucoup plus lent (-15%) et moins fiable (-20%)
}
```

```
Impact de la météo sur les performances :

                    Speed      Reliability
Sunny  ☀️   ████████████  100%  ███████████  95%
Cloudy ⛅   ███████████   98%  ██████████   93%
Rain   🌧️   █████████     85%  ████████     80%
```

### Structure de la classe

```
┌──────────────────────────────────────────────────────┐
│                   F1Simulator                        │
│──────────────────────────────────────────────────────│
│ Attributs :                                          │
│   data_loader    → instance de F1DataLoader          │
│   weather_impact → dict des impacts météo            │
│──────────────────────────────────────────────────────│
│ Méthodes publiques :                                 │
│   simulate_race()           → orchestre la simulation│
│   get_available_circuits()  → liste des circuits     │
│──────────────────────────────────────────────────────│
│ Méthodes privées (préfixe _) :                       │
│   _get_year_races()             → courses d'une année│
│   _get_active_drivers()         → pilotes actifs     │
│   _compute_driver_performance() → score d'un pilote  │
└──────────────────────────────────────────────────────┘
```

> Les méthodes `_` sont internes : elles ne sont pas destinées à être appelées depuis
> l'extérieur de la classe. Elles découpent `simulate_race` en étapes indépendantes.

### `_get_year_races` — Courses d'une année

```python
def _get_year_races(self, year: int) -> pd.DataFrame:
    year_races = self.data_loader.races[self.data_loader.races['year'] == year]
    if len(year_races) == 0:
        available_years = self.data_loader.races['year'].unique()
        # .unique() = liste des années sans doublons [2009, 2010, ..., 2024]
        closest_year = min(available_years, key=lambda x: abs(x - year))
        # min(..., key=lambda x: abs(x - year))
        # = trouve l'année dont la différence avec 'year' est la plus petite
        year_races = self.data_loader.races[self.data_loader.races['year'] == closest_year]
    return year_races
```

> Si l'année demandée n'existe pas dans les données, on prend automatiquement l'année la plus proche.

### `_get_active_drivers` — Pilotes actifs

```python
def _get_active_drivers(self, year_races: pd.DataFrame) -> pd.DataFrame:
    year_results = self.data_loader.results[
        self.data_loader.results['raceId'].isin(year_races['raceId'])
    ]
    return year_results.merge(
        self.data_loader.drivers, on='driverId'
    ).merge(
        self.data_loader.constructors, on='constructorId'
    )[['driverId', 'forename', 'surname', 'code', 'name']].drop_duplicates('driverId')
    # .drop_duplicates('driverId') = chaque pilote n'apparaît qu'une seule fois
```

> Fusionne les résultats, les pilotes et les équipes pour obtenir un tableau complet,
> avec un seul enregistrement par pilote.

### `_compute_driver_performance` — Score d'un pilote

```python
def _compute_driver_performance(self, driver, circuit_history, weather_data, year):
    recent_form = self.data_loader.get_recent_form(driver['driverId'], year=year)

    driver_circuit_history = circuit_history[circuit_history['driverId'] == driver['driverId']]
    if len(driver_circuit_history) > 0:
        circuit_positions = pd.to_numeric(driver_circuit_history['position'], errors='coerce')
        circuit_perf = circuit_positions.mean()
        if pd.isna(circuit_perf):  # pd.isna() vérifie si c'est NaN (valeur vide)
            circuit_perf = 10
    else:
        circuit_perf = 10          # Pas d'historique → position neutre

    circuit_skill = 1 - (circuit_perf / 20)
    # Transforme une position (1-20) en score (0-1)

    performance = (recent_form * 0.7 + circuit_skill * 0.3)
    # 70% forme récente, 30% historique sur ce circuit
    performance *= weather_data['speed']
    # Multiplie par le facteur météo (ex: 0.85 si pluie)
    performance += np.random.normal(0, 0.08)
    # Bruit aléatoire : simule l'imprévisibilité d'une vraie course
    return max(0, performance)
```

### `simulate_race` — Chef d'orchestre

```python
def simulate_race(self, circuit_id: int, weather: str, year: int = 2024) -> pd.DataFrame:
    circuit_history = self.data_loader.get_circuit_history(circuit_id, limit=20)
    year_races      = self._get_year_races(year)
    active_drivers  = self._get_active_drivers(year_races)
    weather_data    = self.weather_impact.get(weather, self.weather_impact['sunny'])

    num_drivers = min(len(active_drivers), 24)  # Maximum 24 pilotes sur une grille F1
    results = []
    for _, driver in active_drivers.head(num_drivers).iterrows():
        # .iterrows() = parcourt le tableau ligne par ligne
        results.append({
            'driver':        f"{driver['forename']} {driver['surname']}",
            'code':          driver['code'],
            'team':          driver['name'],
            'performance':   self._compute_driver_performance(driver, circuit_history, weather_data, year),
            'grid_position': len(results) + 1
        })

    df = pd.DataFrame(results)
    df = df.sort_values('performance', ascending=False).reset_index(drop=True)
    # ascending=False → décroissant (le plus performant en premier)
    # reset_index(drop=True) → remet les index à 0, 1, 2...
    df['final_position'] = df.index + 1  # index commence à 0, donc +1

    return df[['final_position', 'driver', 'code', 'team', 'grid_position', 'performance']]
```

```
Schéma du calcul de performance :

Pour chaque pilote :
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  forme_récente ──────(×0.7)──┐                            │
│  (5 dernières courses)       ├──▶ performance_base        │
│                              │         │                   │
│  historique_circuit ─(×0.3)──┘         │×météo (0.85-1.0)│
│  (sur ce circuit)                      │                   │
│                                        ▼                   │
│                              performance_météo             │
│                                        │                   │
│                              + bruit_aléatoire             │
│                                (±0.08 random)              │
│                                        │                   │
│                                        ▼                   │
│                              PERFORMANCE FINALE            │
└────────────────────────────────────────────────────────────┘

→ Trier par performance décroissante = classement final
```

---

## 7. `dashboard.py` — Interface visuelle

Ce fichier crée l'interface web interactive avec **Streamlit**.

```python
import streamlit as st          # Framework pour créer des apps web en Python
import pandas as pd
import plotly.express as px     # Graphiques interactifs simples
import plotly.graph_objects as go  # Graphiques interactifs avancés
from simulator import F1Simulator
from predictor import F1Predictor
```

### Configuration de la page

```python
st.set_page_config(page_title="VVVA F1 Predictor", layout="wide")
# Définit le titre de l'onglet navigateur et utilise toute la largeur de l'écran

st.title("VVVA - Simulateur et Prédicteur F1")
st.caption("Basé sur données historiques réelles de F1")
```

### Le cache Streamlit

```python
@st.cache_resource
def load_simulator():
    return F1Simulator()

@st.cache_resource
def load_predictor():
    predictor = F1Predictor()
    with st.spinner("Entraînement du modèle ML..."):
        # st.spinner() affiche un cercle de chargement pendant l'exécution du bloc
        scores = predictor.train()
    return predictor, scores
```

> `@st.cache_resource` est un **décorateur** : il dit à Streamlit de ne charger ces objets
> qu'une seule fois, même si la page est rechargée. Sans ça, le modèle se ré-entraînerait
> à chaque interaction de l'utilisateur (très lent !).

### Structure des onglets

```python
tab1, tab2, tab3 = st.tabs(["Simulateur", "Prédicteur ML", "Statistiques"])
# Crée 3 onglets et assigne chacun à une variable
```

```
Interface web :
┌─────────────────────────────────────────────────────────────┐
│  VVVA - Simulateur et Prédicteur F1                         │
│  Basé sur données historiques réelles de F1                 │
├──────────────┬──────────────────┬───────────────────────────┤
│  Simulateur  │  Prédicteur ML   │       Statistiques        │
├──────────────┴──────────────────┴───────────────────────────┤
│                                                             │
│  [Contenu de l'onglet actif]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Onglet 1 — Simulateur

```python
with tab1:
    st.header("Simulateur de Grand Prix")

    col1, col2, col3 = st.columns(3)  # Divise en 3 colonnes côte à côte

    with col1:
        selected_circuit = st.selectbox(
            "Circuit",
            circuits_df['circuitId'].values,
            # format_func : transforme l'ID en nom lisible pour l'affichage
            format_func=lambda x: circuits_df[circuits_df['circuitId']==x]['name'].values[0]
        )

    with col2:
        weather = st.selectbox("Météo", ["sunny", "cloudy", "rain"])

    with col3:
        year = st.number_input("Année", min_value=2009, max_value=2024, value=2024)
        # min_value/max_value : limites du champ numérique | value : valeur par défaut

    if st.button("Lancer la simulation", type="primary"):
        # Ce bloc ne s'exécute que si l'utilisateur clique le bouton
        with st.spinner("Simulation en cours..."):
            results = simulator.simulate_race(circuit_id=selected_circuit, weather=weather, year=year)

        st.success("Simulation terminée!")  # Affiche un message vert de succès

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Résultats de la course")
            st.dataframe(results, use_container_width=True, hide_index=True)
            # st.dataframe() affiche un tableau interactif
            # hide_index=True : cache la colonne d'index (0, 1, 2...)

        with col2:
            st.subheader("Performance par pilote")
            fig = px.bar(
                results.head(10),    # Seulement le Top 10
                x='code',            # Axe horizontal : codes pilotes (HAM, VER...)
                y='performance',     # Axe vertical : score de performance
                color='team',        # Couleur selon l'équipe
                title=f"Top 10 - {weather.capitalize()}"
                # .capitalize() : met la 1ère lettre en majuscule (ex: "rain" → "Rain")
            )
            st.plotly_chart(fig, use_container_width=True)

        # Graphique "évolution des positions"
        st.subheader("Progression Grid → Final")
        fig2 = go.Figure()  # Crée une figure Plotly vide

        for _, row in results.head(10).iterrows():
            fig2.add_trace(go.Scatter(
                x=['Grid', 'Final'],                           # 2 points sur l'axe X
                y=[row['grid_position'], row['final_position']],  # Positions correspondantes
                mode='lines+markers',                           # Ligne + points
                name=row['code'],                              # Légende = code pilote
                line=dict(width=2)
            ))

        fig2.update_layout(
            title="Évolution des positions (Top 10)",
            yaxis=dict(autorange='reversed', title='Position'),
            # autorange='reversed' : inverse l'axe Y (position 1 en haut, 20 en bas)
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
```

### Onglet 2 — Prédicteur ML

```python
with tab2:
    predictor, scores = load_predictor()  # Charge (ou récupère du cache) le modèle entraîné

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score Train", f"{scores['train_score']:.3f}")
        # st.metric() affiche un grand chiffre avec un label
        # :.3f = 3 décimales (ex: 0.724)
    with col2:
        st.metric("Score Test", f"{scores['test_score']:.3f}")
    with col3:
        st.metric("Échantillons", scores['n_samples'])

    st.subheader("Importance des features")
    importance = predictor.get_feature_importance()
    fig = px.bar(importance, x='importance', y='feature', orientation='h')
    # orientation='h' = barres horizontales (plus lisible pour des noms de features)
    st.plotly_chart(fig, use_container_width=True)
```

### Sidebar (barre latérale)

```python
st.sidebar.markdown("---")  # Ligne de séparation horizontale
st.sidebar.info("""
**Projet VVVA**
Prédiction de résultats F1 avec influence météo
Données: 2009-2024
""")
# st.sidebar.* = affiche dans la barre latérale gauche (toujours visible)
```

---

## 8. Schéma global du flux de données

```
                        ┌─────────────────────────────┐
                        │       Fichiers CSV           │
                        │  races / results / drivers   │
                        │  circuits / constructors     │
                        └──────────────┬──────────────┘
                                       │ pd.read_csv()
                                       ▼
                        ┌─────────────────────────────┐
                        │        F1DataLoader          │
                        │                             │
                        │  .races        DataFrame    │
                        │  .results      DataFrame    │
                        │  .drivers      DataFrame    │
                        │  .circuits     DataFrame    │
                        │  .constructors DataFrame    │
                        └──────┬──────────────┬───────┘
                               │              │
               ┌───────────────▼──┐    ┌──────▼─────────────────┐
               │   F1Simulator    │    │      F1Predictor        │
               │                  │    │                          │
               │ simulate_race()  │    │  prepare_training_data()│
               │                  │    │  model.fit()            │
               │  Pour chaque     │    │  model.predict()        │
               │  pilote:         │    │                          │
               │  • forme récente │    │  RandomForest           │
               │  • hist. circuit │    │  100 arbres             │
               │  • météo         │    │  R² ≈ 0.65              │
               │  • hasard        │    │                          │
               └───────┬──────────┘    └──────────┬──────────────┘
                       │                          │
                       └────────────┬─────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │      dashboard.py       │
                       │                         │
                       │  Streamlit + Plotly      │
                       │                         │
                       │  ┌─────────────────┐   │
                       │  │  Onglet 1        │   │
                       │  │  Simulateur      │   │
                       │  │  ↳ Sélect circuit│   │
                       │  │  ↳ Sélect météo  │   │
                       │  │  ↳ Sélect année  │   │
                       │  │  ↳ Bar chart     │   │
                       │  │  ↳ Line chart    │   │
                       │  ├─────────────────┤   │
                       │  │  Onglet 2        │   │
                       │  │  Prédicteur ML   │   │
                       │  │  ↳ Scores R²     │   │
                       │  │  ↳ Feature import│   │
                       │  ├─────────────────┤   │
                       │  │  Onglet 3        │   │
                       │  │  Statistiques    │   │
                       │  │  ↳ Table circuits│   │
                       │  └─────────────────┘   │
                       └────────────────────────┘
                                    │
                                    ▼
                          🌐 Navigateur web
                          localhost:8501
```

---

## 9. Comment lancer le projet

### Installation

```bash
# 1. Installer les dépendances
pip install -r requirements.txt
```

Les bibliothèques dans `requirements.txt` :
| Bibliothèque | Rôle |
|---|---|
| `streamlit` | Crée l'interface web |
| `pandas` | Manipulation de tableaux de données |
| `numpy` | Calculs mathématiques |
| `scikit-learn` | Modèle Random Forest |
| `plotly` | Graphiques interactifs |

### Lancement

```bash
# 2. Lancer le dashboard
python -m streamlit run src/dashboard.py
```

> Streamlit ouvre automatiquement `http://localhost:8501` dans ton navigateur.

### Flux d'exécution au démarrage

```
python -m streamlit run src/dashboard.py
        │
        ▼
dashboard.py charge
        │
        ├── load_simulator()
        │       └── F1Simulator()
        │               └── F1DataLoader()
        │                       └── lit les 5 CSV en mémoire
        │
        └── load_predictor()  (au 1er clic sur l'onglet ML)
                └── F1Predictor()
                        ├── F1DataLoader()
                        ├── prepare_training_data()  ← fusionne les CSV
                        └── model.fit()              ← entraîne le Random Forest
```

---

*Documentation générée pour le projet VVVA — F1 Race Prediction System (2009-2024)*
