import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_loader import F1DataLoader

class F1Predictor:
    """Modèle de prédiction des résultats de courses F1 basé sur données réelles"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.data_loader = F1DataLoader()
        self.is_trained = False
        self.feature_names = None
    
    def train(self):
        """Entraîne le modèle sur les données historiques"""
        X, y = self.data_loader.prepare_training_data()
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'n_samples': len(X)
        }
    
    def predict_position(self, grid_position, circuit_id, driver_age, year):
        """Prédit la position finale d'un pilote"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        features = pd.DataFrame({
            'grid_position': [grid_position],
            'circuit_encoded': [circuit_id],
            'driver_age': [driver_age],
            'year': [year]
        })
        
        prediction = self.model.predict(features)[0]
        return max(1, min(20, int(round(prediction))))
    
    def get_feature_importance(self):
        """Retourne l'importance des features"""
        if not self.is_trained:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
