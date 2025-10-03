def test_project_structure():
    """Проверяем что основные файлы существуют"""
    import os
    assert os.path.exists('main.py')
    assert os.path.exists('requirements.txt')
    assert os.path.exists('src/data/generator.py')

def test_imports():
    """Проверяем что основные модули импортируются"""
    try:
        from src.data.generator import DataGenerator
        from src.features.engineer import FeatureEngineer
        assert True
    except ImportError:
        assert False, "Ошибка импорта модулей"
