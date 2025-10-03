import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MetricsCalculator:
    def __init__(self):
        pass
    
    def calculate_ml_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           baseline_mae: float = None) -> Dict:
        """Расчет метрик ML модели"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Процент улучшения относительно baseline
        improvement = None
        if baseline_mae:
            improvement = ((baseline_mae - mae) / baseline_mae) * 100
        
        return {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'MAPE': round(mape, 1),
            'R2_score': round(r2_score(y_true, y_pred), 3),
            'Improvement_vs_baseline': round(improvement, 1) if improvement else None
        }
    
    def calculate_business_metrics(self, df: pd.DataFrame, recommendations: pd.DataFrame) -> Dict:
        """Расчет бизнес-метрик"""
        # Метрики запасов
        inventory_metrics = self._calculate_inventory_metrics(df, recommendations)
        
        # Финансовые метрики
        financial_metrics = self._calculate_financial_metrics(df, recommendations)
        
        # Метрики сервиса
        service_metrics = self._calculate_service_metrics(df)
        
        return {
            **inventory_metrics,
            **financial_metrics,
            **service_metrics
        }
    
    def _calculate_inventory_metrics(self, df: pd.DataFrame, recommendations: pd.DataFrame) -> Dict:
        """Метрики управления запасами"""
        total_current_stock = df.groupby('part_name')['stock'].last().sum()
        total_recommended_stock = recommendations['Рекомендуемый запас'].sum()
        
        # Дефицитные и избыточные позиции
        deficit_items = len(recommendations[recommendations['Статус'] == 'Недостаточный'])
        excess_items = len(recommendations[recommendations['Рекомендуемый запас'] < 
                                         recommendations['Текущий запас'] * 0.7])
        
        # Оборачиваемость запасов (примерная)
        avg_daily_demand = df['demand'].mean()
        avg_stock = df['stock'].mean()
        turnover_ratio = avg_daily_demand / avg_stock if avg_stock > 0 else 0
        
        return {
            'total_current_stock': int(total_current_stock),
            'total_recommended_stock': int(total_recommended_stock),
            'stock_change_percent': round(((total_recommended_stock - total_current_stock) / total_current_stock) * 100, 1),
            'deficit_items_count': deficit_items,
            'excess_items_count': excess_items,
            'inventory_turnover_ratio': round(turnover_ratio, 3),
            'avg_days_of_supply': round(avg_stock / avg_daily_demand, 1) if avg_daily_demand > 0 else 0
        }
    
    def _calculate_financial_metrics(self, df: pd.DataFrame, recommendations: pd.DataFrame) -> Dict:
        """Финансовые метрики"""
        total_revenue = (df['demand'] * df['price']).sum()
        avg_price = df['price'].mean()
        
        # Оценка экономии (примерная)
        current_stock_value = df.groupby('part_name')['stock'].last().sum() * avg_price
        recommended_stock_value = recommendations['Рекомендуемый запас'].sum() * avg_price
        potential_savings = current_stock_value - recommended_stock_value
        
        # Стоимость дефицита
        shortage_events = df[df['demand'] > df['stock']]
        shortage_cost = (shortage_events['demand'] - shortage_events['stock']).sum() * avg_price * 0.1
        
        return {
            'total_revenue_millions': round(total_revenue / 1e6, 2),
            'avg_item_price': round(avg_price, 2),
            'current_inventory_value_millions': round(current_stock_value / 1e6, 2),
            'recommended_inventory_value_millions': round(recommended_stock_value / 1e6, 2),
            'potential_annual_savings_millions': round(abs(potential_savings) / 1e6, 2),
            'shortage_cost_millions': round(shortage_cost / 1e6, 2)
        }
    
    def _calculate_service_metrics(self, df: pd.DataFrame) -> Dict:
        """Метрики уровня сервиса"""
        total_orders = len(df)
        shortage_events = len(df[df['demand'] > df['stock']])
        service_level = (1 - shortage_events / total_orders) * 100
        
        # Fill Rate (более точная метрика)
        total_demand = df['demand'].sum()
        fulfilled_demand = df[df['demand'] <= df['stock']]['demand'].sum() + \
                          df[df['demand'] > df['stock']]['stock'].sum()
        fill_rate = (fulfilled_demand / total_demand) * 100
        
        return {
            'service_level_percent': round(service_level, 1),
            'fill_rate_percent': round(fill_rate, 1),
            'shortage_events_count': shortage_events,
            'total_orders_count': total_orders
        }
