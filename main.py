import pandas as pd
from src.data.generator import DataGenerator
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.analysis.business_analyzer import BusinessAnalyzer
from src.analysis.metrics_calculator import MetricsCalculator
from src.visualization.dashboard import DashboardCreator
import warnings
warnings.filterwarnings('ignore')

class AviationDataAnalyzer:
    def __init__(self):
        self.df = None
        self.model_trainer = None
        self.feature_engineer = FeatureEngineer()
        self.business_analyzer = BusinessAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.dashboard_creator = DashboardCreator()
        
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("=== АНАЛИЗ ДАННЫХ АВИАЗАПЧАСТЕЙ ===\n")
        
        # Генерация данных
        print("1. Генерация бизнес-данных...")
        data_generator = DataGenerator()
        self.df = data_generator.generate_business_data()
        print(f"   Создано записей: {len(self.df):,}")
        
        # ML модель
        print("\n2. Обучение ML модели...")
        df_processed = self.feature_engineer.create_features(self.df)
        X, y = self.feature_engineer.prepare_features_for_training(df_processed)
        
        self.model_trainer = ModelTrainer()
        model_mae = self.model_trainer.train_demand_model(X, y)
        
        # Бизнес-анализ
        print("\n3. Бизнес-анализ...")
        business_metrics = self.business_analyzer.analyze_business_metrics(self.df)
        
        # Рекомендации
        print("\n4. Формирование рекомендаций...")
        recommendations = self.business_analyzer.generate_recommendations(
            self.df, business_metrics['revenue_analysis']
        )
        
        # Расчет метрик
        print("\n5. Расчет метрик...")
        ml_metrics = self.metrics_calculator.calculate_ml_metrics(y, self.model_trainer.predict(X))
        business_metrics_detailed = self.metrics_calculator.calculate_business_metrics(self.df, recommendations)
        
        # Вывод метрик
        print("\n" + "="*50)
        print("ОСНОВНЫЕ МЕТРИКИ ПРОЕКТА")
        print("="*50)
        print(f"MAE модели: {ml_metrics['MAE']}")
        print(f"Уровень сервиса: {business_metrics_detailed['service_level_percent']}%")
        print(f"Потенциальная экономия: {business_metrics_detailed['potential_annual_savings_millions']} млн руб./год")
        
        # Визуализация
        print("\n6. Создание дашборда...")
        feature_importance = self.model_trainer.get_feature_importance()
        self.dashboard_creator.create_dashboard(business_metrics, feature_importance)
        
        return {
            'model_mae': model_mae,
            'ml_metrics': ml_metrics,
            'business_metrics': business_metrics_detailed,
            'recommendations': recommendations
        }

# Запуск анализа
if __name__ == "__main__":
    analyzer = AviationDataAnalyzer()
    results = analyzer.run_full_analysis()
