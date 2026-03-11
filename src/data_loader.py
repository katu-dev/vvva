import pandas as pd
import numpy as np
from pathlib import Path

class F1DataLoader:
    """Charge et prépare les données F1 depuis les CSV"""
    
    def __init__(self, data_path='csv'):
        self.data_path = Path(data_path)
        self.races = None
        self.results = None
        self.drivers = None
        self.circuits = None
        self.constructors = None
        self.load_data()
    
    def load_data(self):
        """Charge tous les CSV nécessaires"""
        self.races = pd.read_csv(self.data_path / 'races.csv')
        self.results = pd.read_csv(self.data_path / 'results.csv')
        self.drivers = pd.read_csv(self.data_path / 'drivers.csv')
        self.circuits = pd.read_csv(self.data_path / 'circuits.csv')
        self.constructors = pd.read_csv(self.data_path / 'constructors.csv')
    
    def get_driver_stats(self, driver_id):
        """Récupère les statistiques d'un pilote"""
        driver_results = self.results[self.results['driverId'] == driver_id]
        
        # Convertir position en numérique
        positions = pd.to_numeric(driver_results['position'], errors='coerce')
        
        return {
            'total_races': len(driver_results),
            'wins': len(positions[positions == 1]),
            'podiums': len(positions[positions <= 3]),
            'avg_position': positions.mean(),
            'total_points': driver_results['points'].sum()
        }
    
    def get_circuit_history(self, circuit_id, limit=10):
        """Récupère l'historique des courses sur un circuit"""
        circuit_races = self.races[self.races['circuitId'] == circuit_id].tail(limit)
        race_ids = circuit_races['raceId'].values
        
        history = self.results[self.results['raceId'].isin(race_ids)]
        history = history.merge(self.drivers, on='driverId', suffixes=('', '_driver'))
        history = history.merge(circuit_races, on='raceId', suffixes=('', '_race'))
        return history
    
    def prepare_training_data(self):
        """Prépare les données pour l'entraînement du modèle"""
        # Merge des données avec suffixes pour éviter les conflits
        data = self.results.merge(self.races, on='raceId', suffixes=('', '_race'))
        data = data.merge(self.drivers, on='driverId', suffixes=('', '_driver'))
        data = data.merge(self.circuits, on='circuitId', suffixes=('', '_circuit'))
        data = data.merge(self.constructors, on='constructorId', suffixes=('', '_constructor'))
        
        # Calcul des features
        data['driver_age'] = pd.to_datetime(data['date']).dt.year - pd.to_datetime(data['dob']).dt.year
        data['grid_position'] = data['grid'].fillna(20)
        
        # Encodage des circuits
        data['circuit_encoded'] = data['circuitId'].astype('category').cat.codes
        
        # Sélection des features pertinentes
        features = ['grid_position', 'circuit_encoded', 'driver_age', 'year']
        target = 'position'
        
        # Nettoyage - convertir position en numérique et filtrer les valeurs invalides
        data['position'] = pd.to_numeric(data['position'], errors='coerce')
        clean_data = data[features + [target]].dropna()
        clean_data = clean_data[clean_data['position'] > 0]
        clean_data = clean_data[clean_data['position'] <= 30]  # Positions valides uniquement
        
        return clean_data[features], clean_data[target]
    
    def get_recent_form(self, driver_id, n_races=5, year=None):
        """Calcule la forme récente d'un pilote"""
        driver_results = self.results[self.results['driverId'] == driver_id]
        
        # Si une année est spécifiée, filtrer jusqu'à cette année
        if year is not None:
            race_ids = self.races[self.races['year'] <= year]['raceId']
            driver_results = driver_results[driver_results['raceId'].isin(race_ids)]
        
        driver_results = driver_results.tail(n_races)
        
        if len(driver_results) == 0:
            return 0.5
        
        # Convertir position en numérique et remplacer les valeurs invalides
        positions = pd.to_numeric(driver_results['position'], errors='coerce').fillna(20)
        return 1 - (positions.mean() / 20)
