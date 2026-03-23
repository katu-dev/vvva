import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
try:
    from data_loader import F1DataLoader
except ImportError:
    from src.data_loader import F1DataLoader


class F1Predictor:
    """ML model trained on historical F1 data to predict 2025 race results"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.data_loader = F1DataLoader()
        self.is_trained = False
        self.feature_names = None
        # Circuit encoding map built at train time, reused at predict time
        self._circuit_cat = None

    def train(self):
        """Train on all historical data (2009-2024)"""
        X, y, circuit_cat = self.data_loader.prepare_training_data()
        self._circuit_cat = circuit_cat
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        return {
            'train_score': self.model.score(X_train, y_train),
            'test_score': self.model.score(X_test, y_test),
            'n_samples': len(X)
        }

    def predict_2026_race(self, circuit_id: int) -> pd.DataFrame:
        """
        Predict finishing positions for a 2025 race on the given circuit.
        Uses 2024 drivers and their end-of-2024 standings as features.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting")

        drivers = self.data_loader.get_drivers_for_year(2024)
        circuit_code = self._circuit_cat.get(circuit_id, 0)

        # End-of-2024 driver standings (last raceId of 2024 season)
        last_race_2024 = self.data_loader.df[self.data_loader.df['year'] == 2024]['raceId'].max()
        ds = self.data_loader._driver_standings
        ds_2024 = ds[ds['raceId'] == last_race_2024].set_index('driverId')

        # End-of-2024 constructor standings
        cs = self.data_loader._constructor_standings
        cs_2024 = cs[cs['raceId'] == last_race_2024].set_index('constructorId')

        rows = []
        for i, (_, driver) in enumerate(drivers.iterrows()):
            did = driver['driverId']
            cid = driver['constructorId']

            driver_rows = self.data_loader.df[self.data_loader.df['driverId'] == did]
            age = 2025 - pd.to_datetime(driver_rows['dob'].iloc[0], errors='coerce').year \
                if not driver_rows.empty else 30

            d_pts  = ds_2024.loc[did, 'points']   if did in ds_2024.index else 0
            d_pos  = ds_2024.loc[did, 'position']  if did in ds_2024.index else 20
            d_wins = ds_2024.loc[did, 'wins']      if did in ds_2024.index else 0
            c_pts  = cs_2024.loc[cid, 'points']   if cid in cs_2024.index else 0
            c_pos  = cs_2024.loc[cid, 'position']  if cid in cs_2024.index else 10

            rows.append({
                'driver':               f"{driver['forename']} {driver['surname']}",
                'code':                 driver['code'],
                'team':                 driver['team'],
                'quali_pos':            i + 1,
                'grid_position':        i + 1,
                'circuit_encoded':      circuit_code,
                'driver_age':           age,
                'year':                 2025,
                'driver_champ_points':  d_pts,
                'driver_champ_pos':     d_pos,
                'driver_wins':          d_wins,
                'constructor_points':   c_pts,
                'constructor_pos':      c_pos,
                'dnf_rate':             0.05,
            })

        df = pd.DataFrame(rows)
        df['predicted_position'] = self.model.predict(df[self.feature_names]).round().astype(int).clip(1, 20)
        df = df.sort_values('predicted_position').reset_index(drop=True)
        df['predicted_position'] = range(1, len(df) + 1)

        return df[['predicted_position', 'driver', 'code', 'team', 'grid_position']]

    def get_real_results(self, circuit_id: int, year: int) -> pd.DataFrame:
        """Return actual race results for a given circuit and year"""
        df = self.data_loader.df
        race = df[(df['circuitId'] == circuit_id) & (df['year'] == year)]

        if race.empty:
            return pd.DataFrame()

        # Take the most recent round if multiple races on same circuit that year
        latest_race_id = race.sort_values('round')['raceId'].iloc[-1]
        race = df[df['raceId'] == latest_race_id].copy()

        race['position_num'] = pd.to_numeric(race['position'], errors='coerce')
        race = race.dropna(subset=['position_num'])
        race = race.sort_values('position_num')

        race['driver'] = race['forename'] + ' ' + race['surname']
        race['final_position'] = race['position_num'].astype(int)

        return race[['final_position', 'driver', 'code', 'team', 'grid', 'points']].reset_index(drop=True)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return None
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
