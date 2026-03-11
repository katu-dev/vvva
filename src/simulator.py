import numpy as np
import pandas as pd
from typing import Dict, List
from data_loader import F1DataLoader

class F1Simulator:
    """Simulateur de Grand Prix de Formule 1 basé sur données réelles"""
    
    def __init__(self):
        self.data_loader = F1DataLoader()
        
        self.weather_impact = {
            'sunny': {'speed': 1.0, 'reliability': 0.95},
            'cloudy': {'speed': 0.98, 'reliability': 0.93},
            'rain': {'speed': 0.85, 'reliability': 0.80}
        }
    
    def simulate_race(self, circuit_id: int, weather: str, year: int = 2024) -> pd.DataFrame:
        """Simule une course complète basée sur les données historiques"""
        # Récupère l'historique du circuit
        circuit_history = self.data_loader.get_circuit_history(circuit_id, limit=20)
        
        # Récupère les pilotes actifs récents
        recent_races = self.data_loader.races[self.data_loader.races['year'] >= year - 1]
        recent_results = self.data_loader.results[
            self.data_loader.results['raceId'].isin(recent_races['raceId'])
        ]
        
        active_drivers = recent_results.merge(
            self.data_loader.drivers, on='driverId'
        ).merge(
            self.data_loader.constructors, on='constructorId'
        )[['driverId', 'forename', 'surname', 'code', 'name']].drop_duplicates('driverId')
        
        weather_data = self.weather_impact.get(weather, self.weather_impact['sunny'])
        
        results = []
        for _, driver in active_drivers.head(20).iterrows():
            driver_id = driver['driverId']
            
            # Calcule la forme récente
            recent_form = self.data_loader.get_recent_form(driver_id)
            
            # Performance sur ce circuit - convertir position en numérique
            driver_circuit_history = circuit_history[circuit_history['driverId'] == driver_id]
            if len(driver_circuit_history) > 0:
                circuit_positions = pd.to_numeric(driver_circuit_history['position'], errors='coerce')
                circuit_perf = circuit_positions.mean()
                if pd.isna(circuit_perf):
                    circuit_perf = 10
            else:
                circuit_perf = 10
            
            circuit_skill = 1 - (circuit_perf / 20)
            
            # Performance finale
            performance = (recent_form * 0.7 + circuit_skill * 0.3)
            performance *= weather_data['speed']
            performance += np.random.normal(0, 0.08)
            
            results.append({
                'driver': f"{driver['forename']} {driver['surname']}",
                'code': driver['code'],
                'team': driver['name'],
                'performance': max(0, performance),
                'grid_position': len(results) + 1
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('performance', ascending=False).reset_index(drop=True)
        df['final_position'] = df.index + 1
        
        return df[['final_position', 'driver', 'code', 'team', 'grid_position', 'performance']]
    
    def get_available_circuits(self):
        """Retourne la liste des circuits disponibles"""
        return self.data_loader.circuits[['circuitId', 'name', 'location', 'country']]
