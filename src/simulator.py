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

    def _get_year_races(self, year: int) -> pd.DataFrame:
        """Retourne les courses de l'année demandée, ou de l'année la plus proche si absente"""
        year_races = self.data_loader.races[self.data_loader.races['year'] == year]
        if len(year_races) == 0:
            available_years = self.data_loader.races['year'].unique()
            closest_year = min(available_years, key=lambda x: abs(x - year))
            year_races = self.data_loader.races[self.data_loader.races['year'] == closest_year]
        return year_races

    def _get_active_drivers(self, year_races: pd.DataFrame) -> pd.DataFrame:
        """Retourne les pilotes actifs pour les courses données, sans doublons"""
        year_results = self.data_loader.results[
            self.data_loader.results['raceId'].isin(year_races['raceId'])
        ]
        return year_results.merge(
            self.data_loader.drivers, on='driverId'
        ).merge(
            self.data_loader.constructors, on='constructorId'
        )[['driverId', 'forename', 'surname', 'code', 'name']].drop_duplicates('driverId')

    def _compute_driver_performance(self, driver: pd.Series, circuit_history: pd.DataFrame,
                                     weather_data: dict, year: int) -> float:
        """Calcule le score de performance final d'un pilote pour une course"""
        recent_form = self.data_loader.get_recent_form(driver['driverId'], year=year)

        driver_circuit_history = circuit_history[circuit_history['driverId'] == driver['driverId']]
        if len(driver_circuit_history) > 0:
            circuit_positions = pd.to_numeric(driver_circuit_history['position'], errors='coerce')
            circuit_perf = circuit_positions.mean()
            if pd.isna(circuit_perf):
                circuit_perf = 10
        else:
            circuit_perf = 10

        circuit_skill = 1 - (circuit_perf / 20)

        performance = (recent_form * 0.7 + circuit_skill * 0.3)
        performance *= weather_data['speed']
        performance += np.random.normal(0, 0.08)
        return max(0, performance)

    def simulate_race(self, circuit_id: int, weather: str, year: int = 2024) -> pd.DataFrame:
        """Simule une course complète basée sur les données historiques"""
        circuit_history = self.data_loader.get_circuit_history(circuit_id, limit=20)
        year_races = self._get_year_races(year)
        active_drivers = self._get_active_drivers(year_races)
        weather_data = self.weather_impact.get(weather, self.weather_impact['sunny'])

        num_drivers = min(len(active_drivers), 24)
        results = []
        for _, driver in active_drivers.head(num_drivers).iterrows():
            results.append({
                'driver': f"{driver['forename']} {driver['surname']}",
                'code': driver['code'],
                'team': driver['name'],
                'performance': self._compute_driver_performance(driver, circuit_history, weather_data, year),
                'grid_position': len(results) + 1
            })

        df = pd.DataFrame(results)
        df = df.sort_values('performance', ascending=False).reset_index(drop=True)
        df['final_position'] = df.index + 1

        return df[['final_position', 'driver', 'code', 'team', 'grid_position', 'performance']]

    def get_available_circuits(self):
        """Retourne la liste des circuits disponibles"""
        return self.data_loader.circuits[['circuitId', 'name', 'location', 'country']]
