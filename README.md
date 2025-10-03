# 🛩️ Прогнозирование спроса на авиазапчасти и оптимизация запасов

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine-Learning-orange)
![Supply Chain](https://img.shields.io/badge/Supply-Chain-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📋 О проекте

Комплексное решение для прогнозирования спроса и оптимизации товарных запасов авиационных запчастей с использованием машинного обучения.

**Ключевые бизнес-результаты:**
- 📈 **Точность прогноза**: MAE = 1.193
- 🎯 **Уровень сервиса**: 96.1%
- 💰 **Потенциальная экономия**: 2.1 млн руб./год
- ⚠️ **Обнаружено аномалий**: 356

## 🏗️ Структура проекта
my-portfolio-data-science-/
│
├── src/ # Исходный код
│ ├── data/generator.py # Генерация бизнес-данных
│ ├── features/engineer.py # Feature engineering
│ ├── models/trainer.py # ML модели (Random Forest)
│ ├── analysis/ # Бизнес-анализ
│ │ ├── business_analyzer.py
│ │ └── metrics_calculator.py
│ └── visualization/dashboard.py # Визуализация
│
├── main.py # Основной скрипт
├── requirements.txt # Зависимости
└── README.md # Документация

text

## 🛠️ Технологический стек

- **Python**: pandas, numpy, scikit-learn, matplotlib
- **ML**: Random Forest, Time Series Cross-Validation
- **Анализ**: ABC-анализ, временные ряды, feature engineering
- **Визуализация**: интерактивные дашборды

## 📊 Результаты анализа

### Рекомендации по управлению запасами:

| Запчасть | ABC Категория | Приоритет | Текущий запас | Рекомендуемый | Действие |
|----------|---------------|-----------|---------------|---------------|----------|
| Шасси | A | Высокий | 24 | 198 | Увеличить страховой запас |
| Авионика | A | Высокий | 32 | 126 | Увеличить страховой запас |
| Двигатель | A | Высокий | 50 | 97 | Увеличить страховой запас |
| Гидравлика | B | Средний | 39 | 143 | Оптимизировать запас |
| Электрика | C | Низкий | 25 | 163 | Минимизировать запас |

### Топ-5 важных признаков для прогноза:
1. **stock** (92.6%) - текущий запас
2. **stock_demand_ratio** (2.8%) - соотношение запаса и спроса
3. **demand_lag_7** (2.0%) - спрос недельной давности
4. **demand_rolling_mean_7** (1.6%) - скользящее среднее
5. **demand_rolling_std_7** (0.4%) - волатильность спроса

## 🚀 Быстрый старт

```bash
# Клонирование репозитория
git clone https://github.com/ivaniliasov/my-portfolio-data-science-.git
cd my-portfolio-data-science-

# Установка зависимостей
pip install -r requirements.txt

# Запуск полного анализа
python main.py
