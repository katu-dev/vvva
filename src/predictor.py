import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_loader import F1DataLoader


class F1Predictor:
    """ML model trained on historical F1 data to predict 2026 race results"""

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
        Predict finishing positions for a 2026 race on the given circuit.
        Uses 2024 drivers as the base grid (latest known lineup).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting")

        drivers = self.data_loader.get_drivers_for_year(2024)

        # Encode circuit the same way as training
        circuit_code = self._circuit_cat.get(circuit_id, 0)

        rows = []
        for i, (_, driver) in enumerate(drivers.iterrows()):
            age = 2026 - pd.to_datetime(
                self.data_loader.df[self.data_loader.df['driverId'] == driver['driverId']]['dob'].iloc[0],
                errors='coerce'
            ).year if not self.data_loader.df[self.data_loader.df['driverId'] == driver['driverId']].empty else 30

            rows.append({
                'driverId': driver['driverId'],
                'driver': f"{driver['forename']} {driver['surname']}",
                'code': driver['code'],
                'team': driver['team'],
                'grid_position': i + 1,
                'circuit_encoded': circuit_code,
                'driver_age': age,
                'year': 2026
            })

        df = pd.DataFrame(rows)
        features = df[self.feature_names]
        df['predicted_position'] = self.model.predict(features).round().astype(int).clip(1, 20)
        df = df.sort_values('predicted_position').reset_index(drop=True)
        df['predicted_position'] = range(1, len(df) + 1)  # deduplicate ranks

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
