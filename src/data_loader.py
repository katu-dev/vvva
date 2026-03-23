import pandas as pd
import numpy as np
from pathlib import Path


class F1DataLoader:
    """Loads and prepares F1 data from all available CSVs"""

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'csv'
        self.data_path = Path(data_path)
        self.df = None
        self.circuits = None
        self._driver_standings = None
        self._constructor_standings = None
        self._qualifying = None
        self._status = None
        self.load_data()

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_data(self):
        """Load main CSV and all supplementary CSVs"""
        self.df = pd.read_csv(self.data_path / 'f1_data.csv')
        self.df['position_num'] = pd.to_numeric(self.df['position'], errors='coerce')
        self.df['grid'] = pd.to_numeric(self.df['grid'], errors='coerce').fillna(20)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['driver_age'] = self.df['year'] - pd.to_datetime(self.df['dob'], errors='coerce').dt.year

        self._driver_standings = pd.read_csv(self.data_path / 'driver_standings.csv')
        self._constructor_standings = pd.read_csv(self.data_path / 'constructor_standings.csv')
        self._qualifying = pd.read_csv(self.data_path / 'qualifying.csv')
        self._status = pd.read_csv(self.data_path / 'status.csv')

        self.circuits = (
            self.df[['circuitId', 'circuit', 'location', 'country']]
            .drop_duplicates('circuitId')
            .rename(columns={'circuit': 'name'})
            .sort_values('name')
            .reset_index(drop=True)
        )

    # ── Simulator helpers ────────────────────────────────────────────────────

    def get_drivers_for_year(self, year):
        """Returns all drivers who raced in the given year"""
        year_df = self.df[self.df['year'] == year]
        if year_df.empty:
            available = self.df['year'].unique()
            year = min(available, key=lambda y: abs(y - year))
            year_df = self.df[self.df['year'] == year]
        return (
            year_df[['driverId', 'forename', 'surname', 'code', 'team', 'constructorId']]
            .drop_duplicates('driverId')
            .reset_index(drop=True)
        )

    def get_recent_form(self, driver_id, n_races=5, year=None):
        """Recent average finishing position for a driver (0=best, 1=worst)"""
        driver_df = self.df[self.df['driverId'] == driver_id]
        if year is not None:
            driver_df = driver_df[driver_df['year'] <= year]
        driver_df = driver_df.tail(n_races)
        if driver_df.empty:
            return 0.5
        avg = driver_df['position_num'].mean()
        return 1 - (avg / 20) if not pd.isna(avg) else 0.5

    def get_circuit_form(self, driver_id, circuit_id):
        """Average finishing position for a driver on a specific circuit"""
        sub = self.df[(self.df['driverId'] == driver_id) & (self.df['circuitId'] == circuit_id)]
        if sub.empty:
            return 10
        avg = sub['position_num'].mean()
        return avg if not pd.isna(avg) else 10

    def get_driver_skill_vs_car(self):
        """
        Compute each driver's ability relative to their car.
        Positive = consistently outperforms their constructor's expected result.
        This is used as a wet-weather skill proxy: skilled drivers benefit more from rain chaos.
        """
        df = self.df[self.df['position_num'].notna()].copy()

        # Normalize position within each race (0=best, 1=worst)
        df['norm_pos'] = df.groupby('raceId')['position_num'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )

        # Constructor average normalized position per race
        c_avg = df.groupby(['raceId', 'constructorId'])['norm_pos'].mean().reset_index()
        c_avg.columns = ['raceId', 'constructorId', 'constructor_avg']
        df = df.merge(c_avg, on=['raceId', 'constructorId'])

        # Positive delta = driver outperforms their car
        df['skill_delta'] = df['constructor_avg'] - df['norm_pos']
        return df.groupby('driverId')['skill_delta'].mean()

    def get_constructor_reliability(self):
        """
        Historical DNF rate per constructor (0=very reliable, 1=always breaks down).
        Used to modulate DNF risk in the simulator.
        """
        dnf_ids = self._status[~self._status['status'].isin([
            'Finished', '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps', '+5 Laps',
            '+6 Laps', '+7 Laps', '+8 Laps', '+9 Laps'
        ])]['statusId'].values

        df = self.df.copy()
        df['is_dnf'] = df['statusId'].isin(dnf_ids).astype(int)
        return df.groupby('constructorId')['is_dnf'].mean()

    def get_constructor_strength(self, constructor_id, up_to_year):
        """
        Constructor strength (0=weakest, 1=strongest) based on last known championship standing.
        """
        races = self.df[self.df['year'] <= up_to_year]['raceId']
        cs = self._constructor_standings[self._constructor_standings['raceId'].isin(races)]
        latest = cs[cs['constructorId'] == constructor_id].sort_values('raceId').tail(1)
        if latest.empty:
            return 0.5
        pos = latest['position'].iloc[0]
        n = cs.groupby('raceId')['constructorId'].count().median()
        return max(0.05, 1 - (pos - 1) / max(n - 1, 1))

    def get_circuit_overtaking_factor(self, circuit_id):
        """
        How easy it is to overtake on this circuit (0=Monaco-like, 1=Monza-like).
        Based on average grid-to-finish position change historically.
        """
        df = self.df[self.df['circuitId'] == circuit_id].dropna(subset=['position_num', 'grid'])
        if df.empty:
            return 0.4
        avg_change = (df['grid'] - df['position_num']).abs().mean()
        return float(np.clip(avg_change / 6.0, 0.1, 1.0))

    # ── ML training helpers ──────────────────────────────────────────────────

    def _build_enriched(self):
        """Merge all supplementary data into the main dataframe (with lag on standings)."""
        df = self.df.copy()

        # Driver standings — lagged by 1 race (standings before this race)
        ds = self._driver_standings[['raceId', 'driverId', 'points', 'position', 'wins']].copy()
        ds.columns = ['raceId', 'driverId', 'driver_champ_points', 'driver_champ_pos', 'driver_wins']
        ds = ds.sort_values('raceId')
        for col in ['driver_champ_points', 'driver_champ_pos', 'driver_wins']:
            ds[col] = ds.groupby('driverId')[col].shift(1)
        df = df.merge(ds, on=['raceId', 'driverId'], how='left')
        df['driver_champ_points'] = df['driver_champ_points'].fillna(0)
        df['driver_champ_pos']    = df['driver_champ_pos'].fillna(20)
        df['driver_wins']         = df['driver_wins'].fillna(0)

        # Constructor standings — lagged by 1 race
        cs = self._constructor_standings[['raceId', 'constructorId', 'points', 'position']].copy()
        cs.columns = ['raceId', 'constructorId', 'constructor_points', 'constructor_pos']
        cs = cs.sort_values('raceId')
        for col in ['constructor_points', 'constructor_pos']:
            cs[col] = cs.groupby('constructorId')[col].shift(1)
        df = df.merge(cs, on=['raceId', 'constructorId'], how='left')
        df['constructor_points'] = df['constructor_points'].fillna(0)
        df['constructor_pos']    = df['constructor_pos'].fillna(10)

        # Qualifying position (fallback to grid if missing)
        q = self._qualifying[['raceId', 'driverId', 'position']].copy()
        q.columns = ['raceId', 'driverId', 'quali_pos']
        df = df.merge(q, on=['raceId', 'driverId'], how='left')
        df['quali_pos'] = df['quali_pos'].fillna(df['grid'])

        # DNF rate (rolling 10 races, lagged)
        dnf_ids = self._status[~self._status['status'].isin([
            'Finished', '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps', '+5 Laps',
            '+6 Laps', '+7 Laps', '+8 Laps', '+9 Laps'
        ])]['statusId'].values
        df['is_dnf'] = df['statusId'].isin(dnf_ids).astype(int)
        df['dnf_rate'] = (
            df.sort_values('raceId')
            .groupby('driverId')['is_dnf']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            .fillna(0)
        )

        return df

    def prepare_training_data(self):
        """Prepare features and target for ML training"""
        df = self._build_enriched()
        df = df.dropna(subset=['position_num', 'grid', 'driver_age'])
        df = df[(df['position_num'] > 0) & (df['position_num'] <= 30)]

        circuit_series = df['circuitId'].astype('category')
        circuit_cat = dict(zip(circuit_series.cat.categories, circuit_series.cat.codes))

        X = pd.DataFrame({
            'quali_pos':           df['quali_pos'].values,
            'grid_position':       df['grid'].values,
            'circuit_encoded':     circuit_series.cat.codes.values,
            'driver_age':          df['driver_age'].values,
            'year':                df['year'].values,
            'driver_champ_points': df['driver_champ_points'].values,
            'driver_champ_pos':    df['driver_champ_pos'].values,
            'driver_wins':         df['driver_wins'].values,
            'constructor_points':  df['constructor_points'].values,
            'constructor_pos':     df['constructor_pos'].values,
            'dnf_rate':            df['dnf_rate'].values,
        })
        y = df['position_num'].values
        return X, y, circuit_cat

    def get_available_years_for_circuit(self, circuit_id: int):
        """Returns sorted list of years with race data for a given circuit"""
        return sorted(self.df[self.df['circuitId'] == circuit_id]['year'].unique(), reverse=True)
