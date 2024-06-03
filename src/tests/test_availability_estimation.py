import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pytest
from transport_frames.graph_builder.availability_estimation import find_median

@pytest.fixture
def sample_city_points():
    # Создаем GeoDataFrame с двумя точками
    city_points_data = {
        'City': ['City A', 'City B'],
        'geometry': [Point(0, 0), Point(1, 1)]
    }
    sample_city_points = gpd.GeoDataFrame(city_points_data, crs='EPSG:4326')
    return sample_city_points

@pytest.fixture
def sample_adj_mx():
    adj_mx_data = {
        'City A': [0, 2],
        'City B': [2, 0]
    }
    sample_adj_mx = pd.DataFrame(adj_mx_data, index=['City A', 'City B'])
    return sample_adj_mx

def test_find_median(sample_city_points, sample_adj_mx):
    # Запускаем функцию на тестовых данных
    result = find_median(sample_city_points, sample_adj_mx)
    
    # Проверяем, что результат является GeoDataFrame
    assert isinstance(result, gpd.GeoDataFrame)
    
    # Проверяем, что 'to_service' добавлен в DataFrame
    assert 'to_service' in result.columns
    
    # Проверяем, что все значения в 'to_service' являются числами
    assert result['to_service'].dtype == np.float64
    
    # Проверяем, что удалены строки с максимальными значениями
    assert not (result['to_service'] == np.finfo(np.float64).max).any()