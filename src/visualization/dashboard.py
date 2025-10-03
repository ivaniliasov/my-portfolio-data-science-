import matplotlib.pyplot as plt
import pandas as pd

class DashboardCreator:
    def __init__(self):
        self.colors = ['#2E8B57', '#3CB371', '#20B2AA']
    
    def create_dashboard(self, business_metrics: dict, feature_importance: pd.DataFrame = None):
        """Создание визуализаций"""
        revenue_df = business_metrics['revenue_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # График 1: Выручка по категориям
        self._plot_revenue_by_category(ax1, revenue_df)
        
        # График 2: Распределение ABC
        self._plot_abc_distribution(ax2, revenue_df)
        
        # График 3: Важность признаков (топ-10)
        if feature_importance is not None:
            self._plot_feature_importance(ax3, feature_importance)
        
        # График 4: Уровень сервиса
        self._plot_service_level(ax4, business_metrics['service_level'])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_revenue_by_category(self, ax, revenue_df: pd.DataFrame):
        """График выручки по категориям"""
        abc_colors = revenue_df['abc_category'].map({'A': self.colors[0], 'B': self.colors[1], 'C': self.colors[2]})
        ax.bar(revenue_df['part_name'], revenue_df['revenue'] / 1e6, color=abc_colors)
        ax.set_title('Выручка по запчастям (млн руб)')
        ax.set_ylabel('Млн рублей')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_abc_distribution(self, ax, revenue_df: pd.DataFrame):
        """График распределения ABC категорий"""
        abc_counts = revenue_df['abc_category'].value_counts()
        ax.pie(abc_counts.values, labels=abc_counts.index, autopct='%1.1f%%', colors=self.colors)
        ax.set_title('Распределение по ABC категориям')
    
    def _plot_feature_importance(self, ax, feature_importance: pd.DataFrame):
        """График важности признаков"""
        top_features = feature_importance.head(10)
        ax.barh(top_features['feature'], top_features['importance'])
        ax.set_title('Топ-10 важных признаков для прогноза')
        ax.set_xlabel('Важность')
    
    def _plot_service_level(self, ax, service_level: float):
        """График уровня сервиса"""
        ax.bar(['Уровень сервиса'], [service_level * 100], color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Процент')
        ax.set_title(f'Уровень сервиса: {service_level:.1%}')
