# Прогнозирование спроса и оптимизация запасов авиазапчастей

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Deploy-Docker-green)](https://docker.com)

## Результаты проекта

**MAE**: 1.193 | **Уровень сервиса**: 96.1% | **Экономия**: 28.64 млн руб/год

Полное ML-решение для прогнозирования спроса и оптимизации товарных запасов в авиационной отрасли.

## Быстрый запуск

```bash
git clone https://github.com/ivaniliasov/my-portfolio-data-science-.git
cd my-portfolio-data-science-

# Через Docker
docker build -t aviation-parts .
docker run -it aviation-parts

# Локальный запуск
pip install -r requirements.txt
python main.py
Ключевые метрики

Метрика	Результат
Точность прогноза (MAE)	1.193
Уровень сервиса	96.1%
Годовая экономия	28.64 млн руб
Обнаружено аномалий	356
Технологический стек

ML: scikit-learn, Random Forest, временные ряды
Аналитика: pandas, numpy, feature engineering
Production: Docker, модульное тестирование
Визуализация: matplotlib, аналитические отчеты
Архитектура решения

text
Генерация данных → Feature Engineering → ML модель → 
Бизнес-аналитика → Визуализация → Отчеты
Особенности реализации

Кросс-валидация временных рядов (5 фолдов)
ABC-анализ для категоризации запчастей
Автоматический расчет бизнес-метрик
Готовность к промышленному развертыванию
Пример работы

text
=== АНАЛИЗ ДАННЫХ АВИАЗАПЧАСТЕЙ ===

Создано записей: 3,655
Средний MAE: 1.193 (±0.408)
Уровень сервиса: 96.1%
Потенциальная экономия: 28.64 млн руб./год
Структура проекта

text
src/
├── data/generator.py      # Генерация данных
├── features/engineer.py   # Feature engineering
├── models/trainer.py      # ML модель
├── analysis/              # Бизнес-аналитика
└── visualization/         # Визуализация
tests/                     # Модульные тесты
main.py                    # Основной скрипт
Проект демонстрирует полный цикл разработки ML-решения - от анализа данных до production-готовой системы.

Готов к техническому обсуждению и демонстрации.
