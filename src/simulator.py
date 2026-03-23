import numpy as np
import pandas as pd
from data_loader import F1DataLoader


class F1Simulator:
    """Grand Prix simulator based on real historical F1 data"""

    def __init__(self):
        self.data_loader = F1DataLoader()

        self.weather_impact = {
            'sunny': {'speed': 1.0},
            'cloudy': {'speed': 0.98},
            'rain': {'speed': 0.85}
        }

    def simulate_race(self, circuit_id: int, weather: str, year: int = 2025) -> pd.DataFrame:
        """Simulate a 2025 race using 2024 driver lineup and historical circuit data"""
        weather_speed = self.weather_impact.get(weather, self.weather_impact['sunny'])['speed']
        # Use 2024 as the latest known lineup
        active_drivers = self.data_loader.get_drivers_for_year(2024)

        results = []
        for _, driver in active_drivers.iterrows():
            driver_id = driver['driverId']

            recent_form = self.data_loader.get_recent_form(driver_id, year=year)
            circuit_perf = self.data_loader.get_circuit_form(driver_id, circuit_id)
            circuit_skill = 1 - (circuit_perf / 20)

            performance = (recent_form * 0.7 + circuit_skill * 0.3) * weather_speed
            performance += np.random.normal(0, 0.08)

            results.append({
                'driver': f"{driver['forename']} {driver['surname']}",
                'code': driver['code'],
                'team': driver['team'],
                'performance': max(0, performance),
                'grid_position': len(results) + 1
            })

        df = pd.DataFrame(results)
        df = df.sort_values('performance', ascending=False).reset_index(drop=True)
        df['final_position'] = df.index + 1

        return df[['final_position', 'driver', 'code', 'team', 'grid_position', 'performance']]

    def get_available_circuits(self):
        """Returns the list of available circuits"""
        return self.data_loader.circuits[['circuitId', 'name', 'location', 'country']]
