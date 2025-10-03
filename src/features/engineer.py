import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering для временных рядов"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Временные признаки
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Лаги и скользящие средние
        for part in df['part_name'].unique():
            part_mask = df['part_name'] == part
            df.loc[part_mask, 'demand_lag_7'] = df.loc[part_mask, 'demand'].shift(7)
            df.loc[part_mask, 'demand_rolling_mean_7'] = df.loc[part_mask, 'demand'].rolling(7).mean()
            df.loc[part_mask, 'demand_rolling_std_7'] = df.loc[part_mask, 'demand'].rolling(7).std()
        
        # Взаимодействие признаков
        df['stock_demand_ratio'] = df['stock'] / (df['demand'] + 1)
        df['price_category'] = pd.cut(df['price'], bins=3, labels=['low', 'medium', 'high'])
        
        return df.dropna()
    
    def prepare_features_for_training(self, df_processed: pd.DataFrame) -> tuple:
        """Подготовка признаков для обучения модели"""
        feature_columns = ['day_of_week', 'month', 'quarter', 'day_of_year', 'is_weekend',
                          'stock', 'price', 'demand_lag_7', 'demand_rolling_mean_7', 
                          'demand_rolling_std_7', 'stock_demand_ratio']
        
        X = pd.get_dummies(df_processed[feature_columns], columns=['day_of_week', 'month', 'quarter'])
        y = df_processed['demand']
        
        return X, y
