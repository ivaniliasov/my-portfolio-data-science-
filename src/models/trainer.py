import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict

class ModelTrainer:
    def __init__(self, model_params: Dict = None):
        self.model = None
        self.feature_importance = None
        self.model_params = model_params or {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train_demand_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Обучение ML модели для прогнозирования спроса"""
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        print("Cross-Validation результаты:")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = RandomForestRegressor(**self.model_params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores.append(mae)
            print(f"Fold {fold + 1}: MAE = {mae:.3f}")
        
        # Финальная модель
        self.model = RandomForestRegressor(**self.model_params)
        self.model.fit(X, y)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mean_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)
        print(f"\nСредний MAE: {mean_mae:.3f} (+/- {std_mae:.3f})")
        
        return mean_mae
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Прогнозирование спроса"""
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train_demand_model()")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Получение важности признаков"""
        return self.feature_importance
