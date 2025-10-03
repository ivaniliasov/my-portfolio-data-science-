import pandas as pd
import numpy as np
from typing import Dict, Tuple

class BusinessAnalyzer:
    def __init__(self):
        pass
    
    def analyze_business_metrics(self, df: pd.DataFrame) -> Dict:
        """Анализ бизнес-метрик"""
        # ABC анализ
        revenue_analysis = self._perform_abc_analysis(df)
        
        # Анализ проблем с поставками
        supply_issues, service_level = self._analyze_supply_issues(df)
        
        # Анализ аномалий
        anomalies_count = self._count_anomalies(df)
        
        return {
            'revenue_analysis': revenue_analysis,
            'supply_issues': supply_issues,
            'service_level': service_level,
            'anomalies_count': anomalies_count
        }
    
    def _perform_abc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Проведение ABC анализа"""
        revenue_analysis = df.groupby('part_name').agg({
            'demand': 'sum',
            'price': 'mean'
        }).reset_index()
        
        revenue_analysis['revenue'] = revenue_analysis['demand'] * revenue_analysis['price']
        revenue_analysis['revenue_share'] = revenue_analysis['revenue'] / revenue_analysis['revenue'].sum()
        revenue_analysis = revenue_analysis.sort_values('revenue_share', ascending=False)
        revenue_analysis['cumulative_share'] = revenue_analysis['revenue_share'].cumsum()
        
        revenue_analysis['abc_category'] = np.where(
            revenue_analysis['cumulative_share'] <= 0.7, 'A',
            np.where(revenue_analysis['cumulative_share'] <= 0.9, 'B', 'C')
        )
        
        return revenue_analysis
    
    def _analyze_supply_issues(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Анализ проблем с поставками"""
        supply_issues = df[df['demand'] > df['stock']]
        service_level = 1 - (len(supply_issues) / len(df))
        return supply_issues, service_level
    
    def _count_anomalies(self, df: pd.DataFrame) -> int:
        """Подсчет аномалий"""
        anomalies = df[df['is_anomaly'] == True]
        return len(anomalies)
    
    def generate_recommendations(self, df: pd.DataFrame, revenue_df: pd.DataFrame) -> pd.DataFrame:
        """Генерация рекомендаций для бизнеса"""
        recommendations = []
        
        for _, part in revenue_df.iterrows():
            part_data = df[df['part_name'] == part['part_name']]
            avg_demand = part_data['demand'].mean()
            current_stock = part_data['stock'].iloc[-1]
            
            # Рассчет оптимального запаса
            optimal_stock = self._calculate_optimal_stock(avg_demand)
            stock_status = self._evaluate_stock_status(current_stock, optimal_stock)
            
            priority, action = self._get_priority_and_action(part['abc_category'], stock_status)
            
            recommendations.append({
                'Запчасть': part['part_name'],
                'ABC Категория': part['abc_category'],
                'Приоритет': priority,
                'Текущий запас': current_stock,
                'Рекомендуемый запас': int(optimal_stock),
                'Статус': stock_status,
                'Действие': action
            })
        
        return pd.DataFrame(recommendations)
    
    def _calculate_optimal_stock(self, avg_demand: float) -> float:
        """Рассчет оптимального запаса"""
        lead_time_demand = avg_demand * 7  # недельный спрос
        safety_stock = avg_demand * 1.5    # буферный запас
        return lead_time_demand + safety_stock
    
    def _evaluate_stock_status(self, current_stock: int, optimal_stock: float) -> str:
        """Оценка статуса запаса"""
        return 'Оптимальный' if current_stock >= optimal_stock * 0.8 else 'Недостаточный'
    
    def _get_priority_and_action(self, abc_category: str, stock_status: str) -> Tuple[str, str]:
        """Определение приоритета и действия"""
        if abc_category == 'A':
            priority = 'Высокий'
            action = 'Увеличить страховой запас' if stock_status == 'Недостаточный' else 'Поддерживать текущий уровень'
        elif abc_category == 'B':
            priority = 'Средний'
            action = 'Оптимизировать запас' if stock_status == 'Недостаточный' else 'Мониторить'
        else:
            priority = 'Низкий'
            action = 'Минимизировать запас'
        
        return priority, action
