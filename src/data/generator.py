import pandas as pd
import numpy as np
from typing import Dict

class DataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.parts_config = {
            'Двигатель': {'base_demand': 8, 'price_range': (50000, 100000), 'failure_rate': 0.05},
            'Шасси': {'base_demand': 15, 'price_range': (30000, 60000), 'failure_rate': 0.15},
            'Авионика': {'base_demand': 10, 'price_range': (40000, 80000), 'failure_rate': 0.08},
            'Электрика': {'base_demand': 12, 'price_range': (20000, 50000), 'failure_rate': 0.12},
            'Гидравлика': {'base_demand': 11, 'price_range': (25000, 60000), 'failure_rate': 0.10}
        }
    
    def generate_business_data(self, start_date: str = '2022-01-01', end_date: str = '2024-01-01') -> pd.DataFrame:
        """Генерация реалистичных бизнес-данных"""
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        for date in dates:
            for part, config in self.parts_config.items():
                demand = self._generate_demand(date, config, dates[0])
                stock = self._generate_stock(demand, config)
                price = np.random.randint(config['price_range'][0], config['price_range'][1])
                
                data.append({
                    'date': date,
                    'part_name': part,
                    'demand': demand,
                    'stock': stock,
                    'price': price,
                    'is_anomaly': demand > config['base_demand'] * 2.5
                })
        
        return pd.DataFrame(data)
    
    def _generate_demand(self, date: pd.Timestamp, config: Dict, start_date: pd.Timestamp) -> int:
        """Генерация спроса с трендом и сезонностью"""
        trend = 0.001 * (date - start_date).days
        seasonal = 3 * np.sin(2 * np.pi * date.dayofyear / 365)
        weekly = 1.5 * np.sin(2 * np.pi * date.dayofweek / 7)
        
        base_demand = config['base_demand']
        demand = max(2, int(base_demand + seasonal + weekly + trend + np.random.poisson(2)))
        
        # Генерация аномалий на основе failure_rate
        if np.random.random() < config['failure_rate']:
            demand = demand * np.random.choice([3, 4, 5])
        
        return demand
    
    def _generate_stock(self, demand: int, config: Dict) -> int:
        """Умная генерация остатков"""
        optimal_stock = max(demand * 2, 20)
        stock_variation = np.random.randint(-15, 30)
        stock = max(5, optimal_stock + stock_variation)
        return stock
