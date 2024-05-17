import re
import momepy
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd

from tqdm import tqdm
from shapely import wkt
from loguru import logger
from shapely.geometry import LineString, Polygon

def replace_empty_geometries(geom):
    if geom.is_empty:
        return None
    return geom


# перевод в геометрию
def convert_geometry_from_wkt(graph):

    """TODO: сравнить с dongraphio!!!

    Convert the geometry in the graph to WKT format.

    Parameters:
    graph: The input graph.

    Returns:
    nx.Graph: The graph with converted geometry.
    """
    G = graph.copy()
    for _, _, data in G.edges(data=True):
        if isinstance(data['geometry'], str):
            geometry_wkt = wkt.loads(data['geometry'])
            data['geometry'] = geometry_wkt
    logger.info('The graph was converted!')
    return G


# Функция для определения значения REG
def determine_reg(value, highway_type=None) -> int:

    """
    Determine the value of REG.

    Parameters:
    value: The input value.
    highway_type: The type of highway.

    Returns:
    int: The value of REG.
    """

    if isinstance(value, list):
        for item in value:
            if re.match(r'^[МАР]', str(item)):
                return 1
            elif re.match(r'^\d.*[A-Za-zА-Яа-я]', str(item)):
                return 2
        return 3
    elif pd.isna(value):
        # Выставление значения по типу дороги, если значение NaN
        if highway_type:
            return highway_type_to_reg(highway_type)
        return 3
    if re.match(r'^[МАР]', str(value)):
        return 1
    elif re.match(r'^\d.*[A-Za-zА-Яа-я]', str(value)):
        return 2
    else:
        return 3


def highway_type_to_reg(highway_type) -> int:

    """
    Convert highway type to REG value.

    Parameters:
    highway_type: The type of highway.

    Returns:
    int: The REG value.
    """

    highway_mapping = {
        'motorway': 1,
        'trunk': 1,
        'primary': 2,
        'secondary': 2,
        'tertiary': 3,
        'unclassified': 3,
        'residential': 3,
        'motorway_link': 1,
        'trunk_link': 1,
        'primary_link': 2,
        'secondary_link': 2,
        'tertiary_link': 3,
        'living_street': 3
    }
    if isinstance(highway_type, list):
        # Выбрать наименьшее значение из списка типов дорог
        reg_values = [highway_mapping.get(ht, 3) for ht in highway_type]
        return min(reg_values)
    return highway_mapping.get(highway_type, 3)


def custom_map(highway_types) -> float:

    """
    Custom mapping for highway types.

    Parameters:
    highway_types: The type(s) of highway.

    Returns:
    float: The maximum speed.
    """

    maxspeed = {
    'motorway':        110 / 3.6, # Автомагистрали 
    'motorway_link':   110 / 3.6, # Съезды на развязках дорог, на которых действуют те же правила движения, что и на (motorway).
    'primary':         80 / 3.6,  # Автомобильные дороги регионального значения
    'primary_link':    80 / 3.6,  # Съезды на развязках дорог с той же важностью в дорожной сети, что и primary.
    'residential':     60 / 3.6,  # Дороги, которые проходят внутри жилых зон, а также используются для подъезда к ним. 
    'secondary':       70 / 3.6,  # Автомобильные дороги областного значения
    'secondary_link':  70 / 3.6,  # Съезды на развязках дорог с той же важностью в дорожной сети, что и secondary.
    'tertiary':        60 / 3.6,  # Более важные автомобильные дороги среди прочих 
                            # автомобильных дорог местного значения, например
                            # соединяющие районные центры с сёлами, а также несколько сёл между собой.
    'tertiary_link':   60 / 3.6,  # Съезды на развязках дорог с той же важностью в дорожной сети, что и tertiary.
    'trunk':           90 / 3.6,  # Важные дороги, не являющиеся автомагистралями
    'trunk_link':      90 / 3.6,  # Съезды на развязках дорог с той же важностью в дорожной сети, что и trunk.
    'unclassified':    60 / 3.6,  # Остальные автомобильные дороги местного значения, образующие соединительную сеть дорог.
    'living_street':   15 / 3.6   # Дорога с приоритетом у пешеходов
    }

    # Проверяем, является ли highway_types списком.
    if isinstance(highway_types, list):
        # Если значение - список, выбираем тип шоссе с наибольшой максимальной скоростью.
        max_type = max(highway_types, key=lambda x: maxspeed.get(x, np.nan))
        return maxspeed.get(max_type, 40 / 3.6)
    else:
        # Если значение - не список, возвращаем значение или 40, если тип не найден.
        return maxspeed.get(highway_types, 40 / 3.6)
    

def get_graph_from_polygon(polygon: gpd.GeoDataFrame, filter:dict=None, crs:int=3857) -> nx.MultiDiGraph:

    """
    Get graph from polygon.

    Parameters:
    polygon (Polygon): The input polygon.
    filter (dict): The filter for the graph.
    crs (int): The coordinate reference system.

    Returns:
    nx.MultiDiGraph: The generated graph.
    """

    buffer_polygon = polygon.to_crs(crs).geometry.buffer(3000).to_crs(4326).unary_union
    if not filter:
        filter = "['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street']"
    graph = ox.graph_from_polygon(buffer_polygon, network_type='drive', custom_filter=filter, truncate_by_edge=True)
    graph.graph["approach"] = "primal"
    nodes, edges = momepy.nx_to_gdf(graph, points=True, lines=True, spatial_weights=False)
    edges['reg'] = edges.apply(lambda row: determine_reg(row['ref'], row['highway']), axis=1)
    nodes = nodes.to_crs(epsg=crs)
    edges = edges.to_crs(epsg=crs)
    return edges
    # Создаем колонку 'exit' и устанавливаем начальные значения
    edges['exit'] = 0
    # Преобразуем координатные системы для корректного пересечения
    city_transformed = polygon.to_crs(epsg=crs)
    # Находим пересечения ребер с границей города
    edges['intersections'] = edges['geometry'].intersection(city_transformed.unary_union)
    # Обновляем геометрию ребер
    edges['geometry'] = edges['intersections']
    # Создаем маску для ребер, пересекающихся с границей города
    mask = edges[edges['reg'].isin([1, 2])].intersects(city_transformed.unary_union.boundary)
    # Устанавливаем значение 'exit' в 1 для ребер, соответствующих маске
    edges.loc[edges['reg'].isin([1, 2]) & mask, 'exit'] = 1
    # Убираем временные колонки
    edges.drop(columns=['intersections'], inplace=True)
    edges['geometry'].apply(replace_empty_geometries)
    return edges
    edges['maxspeed'] = edges['highway'].apply(lambda x: custom_map(x))

    nodes_coord = nodes.geometry.apply(
        lambda p: {"x": p.coords[0][0], "y": p.coords[0][1]}
    ).to_dict()

    graph_type = 'car'
    edges = edges[["highway", "node_start", "node_end", "geometry", 'maxspeed', 'reg', 'ref']]
    edges["type"] = graph_type
    edges["geometry"] = edges["geometry"].apply(
        lambda x: LineString([tuple(round(c, 6) for c in n) for n in x.coords] if x else None)
    )

    travel_type = "walk" if graph_type == "walk" else "car"

    G = nx.MultiDiGraph()
    for _, edge in tqdm(edges.iterrows(), total=len(edges), desc=f"Collecting {graph_type} graph", leave=False):
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geom = (LineString(([(nodes_coord[p1]["x"], nodes_coord[p1]["y"]),
                            (nodes_coord[p2]["x"], nodes_coord[p2]["y"]),])) if not edge.geometry else edge.geometry)
        length=round(geom.length, 3)
        time_sec = round(length/edge.maxspeed, 3)
        G.add_edge(
            p1,
            p2,
            length_meter=length,
            geometry=str(geom),
            type=travel_type,
            time_sec=time_sec,
            time_min=time_sec / 60,
            highway=edge.highway,
            maxspeed=edge.maxspeed,
            reg=edge.reg,
            ref=edge.ref
        )

    # Устанавливаем начальные атрибуты для узлов
    for node in G.nodes:
        G.nodes[node]['reg_1'] = False
        G.nodes[node]['reg_2'] = False

    # Обновляем атрибуты узлов на основе атрибутов ребер
    for u, v, data in G.edges(data=True):
        if data.get('reg') == 1:
            G.nodes[u]['reg_1'] = True
            G.nodes[v]['reg_1'] = True
        elif data.get('reg') == 2:
            G.nodes[u]['reg_2'] = True
            G.nodes[v]['reg_2'] = True

    nx.set_node_attributes(G, nodes_coord)
    G.graph["crs"] = "epsg:" + str(crs)
    G.graph["approach"] = "primal"
    G.graph["graph_type"] = travel_type + " graph"

    return G



def assign_city_names_to_nodes(points, nodes, graph, name_attr='city_name', node_id_attr='nodeID', name_col='name', max_distance=200):

    """
    Присваивает имена точек для проекции узлам графа на основе пространственного ближайшего соседства.

    Parameters:
    points (GeoDataFrame): Геоданные с точками.
    nodes (GeoDataFrame): Геоданные с узлами графа.
    graph (nx.Graph): Граф, к узлам которого будут добавлены имена точек.
    name_attr (str): Название атрибута для имени точек, которое будет добавлено к узлам графа.
    node_id_attr (str): Название атрибута ID узла в графе.
    name_col (str): Название колонки с именами точек в points.

    Returns:
    nx.Graph
    """
    # Копия графа
    G = graph.copy()

    # Присоединяем ближайшие города к узлам
    project_roads_city = gpd.sjoin_nearest(points, nodes, how='left', distance_col='distance', max_distance=max_distance)

    # Присваиваем имена городов узлам графа с отслеживанием прогресса
    for enum, index in enumerate(project_roads_city[node_id_attr].values):
        city_name = project_roads_city.loc[enum, name_col]
        for _, d in G.nodes(data=True):
            if d.get(node_id_attr) == index:
                d[name_attr] = city_name
    return G