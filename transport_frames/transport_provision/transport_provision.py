"""Module for transport provision calculation and visualization"""
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
import networkx as nx
import momepy
import matplotlib.pyplot as plt
import numpy as np
import json


def calculate_transport_provision(
    territory_gdf: gpd.GeoDataFrame, 
    roads: gpd.GeoDataFrame | nx.Graph | nx.DiGraph | nx.MultiDiGraph, 
    railways_gdf: gpd.GeoDataFrame, 
    airport: gpd.GeoDataFrame, 
    towns_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Calculate the transport infrastructure provision index for a given territory.

    Parameters
    ----------
    territory_gdf : gpd.GeoDataFrame
        GeoDataFrame representing the target territory.
    roads : gpd.GeoDataFrame | nx.Graph | nx.DiGraph | nx.MultiDiGraph
        Road network, either as a GeoDataFrame or a NetworkX graph.
    railways_gdf : gpd.GeoDataFrame
        GeoDataFrame containing railway network data.
    airport : gpd.GeoDataFrame
        GeoDataFrame with airport locations.
    towns_gdf : gpd.GeoDataFrame
        GeoDataFrame containing urban area boundaries.

    Returns
    -------
    gpd.GeoDataFrame
        Updated GeoDataFrame with calculated transport provision metrics, including:
        - 'area_km2': Territory area in square kilometers.
        - 'road_length_km': Total road length within the territory (km).
        - 'rail_length_km': Total railway length within the territory (km).
        - 'airport_distance_km': Distance to the nearest airport (km).
        - 'town_area_km2': Total urban area within the territory (km²).
        - 'transport_provision_index': Composite index measuring transport provision.
    """
    if isinstance(roads, nx.Graph) or isinstance(roads, nx.DiGraph) or isinstance(roads, nx.MultiDiGraph):
        road_gdf = momepy.nx_to_gdf(roads)[1]
    if isinstance(roads,gpd.GeoDataFrame):
        road_gdf = roads.copy()
    road_gdf['road_length_km'] = road_gdf['geometry'].length / 1000
    railways_gdf['road_length_km'] = railways_gdf['geometry'].length / 1000
    # Вычисляем площадь территории
    territory_gdf['area_km2'] = territory_gdf.geometry.area / 1e6  # км²
    
    # Вычисляем длину дорог внутри территории с помощью spatial join
    road_sjoin = gpd.sjoin(road_gdf, territory_gdf, predicate="intersects")
    rail_sjoin = gpd.sjoin(railways_gdf, territory_gdf, predicate="intersects")

    road_length = road_sjoin.groupby('index_right').geometry.apply(lambda x: x.length.sum()) / 2000  # км
    rail_length = rail_sjoin.groupby('index_right').geometry.apply(lambda x: x.length.sum()) / 2000  # км
    
    # Добавляем длины дорог в основную таблицу
    territory_gdf['road_length_km'] = territory_gdf.index.map(road_length).fillna(0)
    territory_gdf['rail_length_km'] = territory_gdf.index.map(rail_length).fillna(0)

    # Вычисляем площадь населенных пунктов в каждой территории через spatial join
    towns_sjoin = gpd.sjoin(towns_gdf, territory_gdf, predicate="intersects")
    town_area = towns_sjoin.groupby('index_right').geometry.apply(lambda x: x.area.sum()) / 1e6  # км²
    territory_gdf['town_area_km2'] = territory_gdf.index.map(town_area).fillna(0)

    
    # Вычисление расстояния до ближайшего аэропорта
    if not airport.empty:
        airport_union = airport.unary_union
        territory_gdf['airport_distance_km'] = territory_gdf.geometry.centroid.apply(
            lambda centroid: centroid.distance(nearest_points(centroid, airport_union)[1]) / 1000
        )
    else:
        territory_gdf['airport_distance_km'] = float('inf')
    
    # Вычисляем индекс обеспеченности
    territory_gdf['transport_provision_index'] = (
        0.4 * (territory_gdf['rail_length_km'] / territory_gdf['area_km2']) +
        0.3 * (territory_gdf['road_length_km'] / territory_gdf['area_km2']) +
        0.2 * (1 / territory_gdf['airport_distance_km'].replace(0, 1)) +
        0.1 * (1 - (territory_gdf['town_area_km2'] / territory_gdf['area_km2']))
    )
    
    return territory_gdf[['geometry', 'area_km2', 'road_length_km', 'rail_length_km', 'airport_distance_km', 'town_area_km2', 'transport_provision_index']]



def visualize_transport_provision(
    result: gpd.GeoDataFrame, 
    road_gdf: gpd.GeoDataFrame
) -> None:
    """
    Visualize the transport infrastructure provision index and road network.

    Parameters
    ----------
    result : gpd.GeoDataFrame
        GeoDataFrame containing calculated transport provision metrics.
    road_gdf : gpd.GeoDataFrame
        GeoDataFrame containing road network data.

    Returns
    -------
    None
        Displays a map visualization of transport provision.
    """
    fig, ax = plt.subplots(figsize=(50, 20))
    
    # Ограничиваем влияние выбросов, используя 1-й и 99-й процентиль
    vmin, vmax = np.percentile(result['transport_provision_index'], [1, 90])

    # Отображение карты с улучшенной цветовой шкалой
    result.plot(column='transport_provision_index', cmap='Blues', linewidth=0.5, 
                edgecolor='black', legend=True, ax=ax, vmin=vmin, vmax=vmax)
    
    # Преобразуем properties из строки JSON в словарь, если нужно
    if isinstance(road_gdf['properties'].iloc[0], str):  
        road_gdf['properties'] = road_gdf['properties'].apply(json.loads)

    # Определяем ширину дорог в зависимости от свойства "reg"
    road_gdf['width'] = road_gdf['properties'].apply(lambda x: 1.5 if x.get('reg') == 1 else 0.6)

    # Визуализация дорог с прозрачностью 60%
    road_gdf.plot(ax=ax, color='black', linewidth=road_gdf['width'], alpha=0.7)
    
    # Убираем рамку с координатами
    ax.set_axis_off()
    
    # Добавляем заголовок
    plt.title("Обеспеченность транспортной инфраструктурой", fontsize=14)
    plt.show()