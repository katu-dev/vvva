import pandas as pd
import numpy as np
from pathlib import Path


class F1DataLoader:
    """Loads and prepares F1 data from the merged CSV"""

    def __init__(self, data_path='csv'):
        self.data_path = Path(data_path)
        self.df = None
        self.circuits = None
        self.load_data()

    def load_data(self):
        """Load the merged CSV and derive circuit lookup"""
        self.df = pd.read_csv(self.data_path / 'f1_data.csv')
        self.df['position_num'] = pd.to_numeric(self.df['position'], errors='coerce')
        self.df['grid'] = pd.to_numeric(self.df['grid'], errors='coerce').fillna(20)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['driver_age'] = self.df['year'] - pd.to_datetime(self.df['dob'], errors='coerce').dt.year

        # Unique circuits table
        self.circuits = (
            self.df[['circuitId', 'circuit', 'location', 'country']]
            .drop_duplicates('circuitId')
            .rename(columns={'circuit': 'name'})
            .sort_values('name')
            .reset_index(drop=True)
        )

    def get_drivers_for_year(self, year):
        """Returns all drivers who raced in the given year"""
        year_df = self.df[self.df['year'] == year]
        if year_df.empty:
            available = self.df['year'].unique()
            year = min(available, key=lambda y: abs(y - year))
            year_df = self.df[self.df['year'] == year]

        return (
            year_df[['driverId', 'forename', 'surname', 'code', 'team']]
            .drop_duplicates('driverId')
            .reset_index(drop=True)
        )

    def get_recent_form(self, driver_id, n_races=5, year=None):
        """Recent average finishing position for a driver (up to given year)"""
        driver_df = self.df[self.df['driverId'] == driver_id]
        if year is not None:
            driver_df = driver_df[driver_df['year'] <= year]
        driver_df = driver_df.tail(n_races)

        if driver_df.empty:
            return 0.5

        avg = driver_df['position_num'].mean()
        if pd.isna(avg):
            return 0.5
        return 1 - (avg / 20)

    def get_circuit_form(self, driver_id, circuit_id):
        """Average finishing position for a driver on a specific circuit"""
        sub = self.df[(self.df['driverId'] == driver_id) & (self.df['circuitId'] == circuit_id)]
        if sub.empty:
            return 10
        avg = sub['position_num'].mean()
        return avg if not pd.isna(avg) else 10

    def prepare_training_data(self):
        """Prepare features and target for ML training, also return circuit encoding map"""
        df = self.df.dropna(subset=['position_num', 'grid', 'driver_age'])
        df = df[(df['position_num'] > 0) & (df['position_num'] <= 30)]

        circuit_series = df['circuitId'].astype('category')
        circuit_cat = dict(zip(circuit_series.cat.categories, circuit_series.cat.codes))
        circuit_encoded = circuit_series.cat.codes

        X = pd.DataFrame({
            'grid_position': df['grid'].values,
            'circuit_encoded': circuit_encoded.values,
            'driver_age': df['driver_age'].values,
            'year': df['year'].values,
        })
        y = df['position_num'].values

        return X, y, circuit_cat

    def get_available_years_for_circuit(self, circuit_id: int):
        """Returns sorted list of years with race data for a given circuit"""
        return sorted(self.df[self.df['circuitId'] == circuit_id]['year'].unique(), reverse=True)
