import re
import sys
import momepy
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd

from tqdm import tqdm
from shapely import wkt
from loguru import logger
from shapely.geometry import LineString, Polygon, Point


# Удаление стандартных обработчиков
logger.remove()

# Добавление нового обработчика с форматированием и поддержкой уровня INFO
logger.add(
    sys.stdout,
    format="<green>{time:MM-DD HH:mm}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",  # Указываем минимальный уровень логирования
    colorize=True
)

# Константные словари
HIGHWAY_MAPPING = {
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

MAXSPEED = {
    'motorway': 110 / 3.6,
    'motorway_link': 110 / 3.6,
    'primary': 80 / 3.6,
    'primary_link': 80 / 3.6,
    'residential': 60 / 3.6,
    'secondary': 70 / 3.6,
    'secondary_link': 70 / 3.6,
    'tertiary': 60 / 3.6,
    'tertiary_link': 60 / 3.6,
    'trunk': 90 / 3.6,
    'trunk_link': 90 / 3.6,
    'unclassified': 60 / 3.6,
    'living_street': 15 / 3.6
}


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
    logger.info('Starting the conversion of the graph.')
    
    for _, _, data in G.edges(data=True):
        if isinstance(data.get('geometry'), str):
            geometry_wkt = wkt.loads(data['geometry'])
            data['geometry'] = geometry_wkt
    
    logger.info('The graph was converted!')
    return G


# Функция для определения значения REG
def determine_reg(name_roads, highway_type=None) -> int:

    """
    Determine the name_roads of REG.

    Parameters:
    name_roads: The input name_roads.
    highway_type: The type of highway.

    Returns:
    int: The value of REG.
    """

    if isinstance(name_roads, list):
        for item in name_roads:
            if re.match(r'^[МАР]', str(item)):
                return 1
            elif re.match(r'^\d.*[A-Za-zА-Яа-я]', str(item)):
                return 2
        return 3
    elif pd.isna(name_roads):
        # Выставление значения по типу дороги, если значение NaN
        if highway_type:
            return highway_type_to_reg(highway_type)
        return 3
    if re.match(r'^[МАР]', str(name_roads)):
        return 1
    elif re.match(r'^\d.*[A-Za-zА-Яа-я]', str(name_roads)):
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

    if isinstance(highway_type, list):
        reg_values = [HIGHWAY_MAPPING.get(ht, 3) for ht in highway_type]
        return min(reg_values)
    return HIGHWAY_MAPPING.get(highway_type, 3)


def get_max_speed(highway_types) -> float:

    """
    Получение максимальной скорости для типов дорог.

    Parameters:
    highway_types: Тип(ы) дорог.

    Returns:
    float: Максимальная скорость.
    """

    # Проверяем, является ли highway_types списком.
    if isinstance(highway_types, list):
        max_type = max(highway_types, key=lambda x: MAXSPEED.get(x, np.nan))
        return MAXSPEED.get(max_type, 40 / 3.6)
    else:
        return MAXSPEED.get(highway_types, 40 / 3.6)


def add_geometry_to_edges(nodes, edges):
    # Создание словаря координат узлов для быстрого доступа
    node_coords = {node['nodeID']: (node['x'], node['y']) for _, node in nodes.iterrows()}
    
    # Проход по каждому edge
    for idx, edge in edges.iterrows():
        if edge['geometry'] is None:
            start_coords = node_coords[edge['node_start']]
            end_coords = node_coords[edge['node_end']]
            line_geom = LineString([start_coords, end_coords])
            edges.at[idx, 'geometry'] = line_geom
    return edges

def buffer_and_transform_polygon(polygon: gpd.GeoDataFrame, crs: int = 3857):
    """Создание буферной зоны вокруг полигона и преобразование координат."""
    return polygon.to_crs(crs).geometry.buffer(3000).to_crs(4326).unary_union


def update_edges_with_geometry(edges, polygon, crs):
    """Обновление геометрии ребер на основе пересечений с границей города."""
    city_transformed = polygon.to_crs(epsg=crs)
    edges['intersections'] = edges['geometry'].intersection(city_transformed.unary_union)
    edges['geometry'] = edges['intersections']
    mask = edges[edges['reg'].isin([1, 2])].buffer(10).intersects(city_transformed.unary_union.boundary)
    edges.loc[edges['reg'].isin([1, 2]) & mask, 'exit'] = 1
    edges.drop(columns=['intersections'], inplace=True)
    edges = edges.explode(index_parts=True)
    edges = edges[~edges['geometry'].is_empty]
    edges = edges[edges['geometry'].geom_type == 'LineString']
    return edges


def update_nodes_with_geometry(edges, nodes_coord):
    """Обновление координат узлов на основе новой геометрии ребер."""
    for _, row in edges.iterrows():
        start_node = row['node_start']
        end_node = row['node_end']
        if start_node not in nodes_coord:
            nodes_coord[start_node] = {"x": row['geometry'].coords[0][0], "y": row['geometry'].coords[0][1]}
        if end_node not in nodes_coord:
            nodes_coord[end_node] = {"x": row['geometry'].coords[-1][0], "y": row['geometry'].coords[-1][1]}
    return nodes_coord


def create_graph(edges, nodes_coord, graph_type):
    """Создание графа на основе ребер и координат узлов."""
    G = nx.MultiDiGraph()
    travel_type = "walk" if graph_type == "walk" else "car"
    for _, edge in edges.iterrows():
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geom = (LineString(([(nodes_coord[p1]["x"], nodes_coord[p1]["y"]),
                            (nodes_coord[p2]["x"], nodes_coord[p2]["y"]),])) if not edge.geometry else edge.geometry)
        length = round(geom.length, 3)
        time_sec = round(length / edge.maxspeed, 3)
        G.add_edge(
            p1, p2,
            length_meter=length,
            geometry=str(geom),
            type=travel_type,
            time_sec=time_sec,
            time_min=time_sec / 60,
            highway=edge.highway,
            maxspeed=edge.maxspeed,
            reg=edge.reg,
            ref=edge.ref,
            is_exit=edge.exit
        )
    return G


def set_node_attributes(G, nodes_coord, polygon, crs):
    """Установка атрибутов узлов на основе атрибутов ребер."""
    for node in G.nodes:
        G.nodes[node]['reg_1'] = False
        G.nodes[node]['reg_2'] = False

    for u, v, data in G.edges(data=True):
        if data.get('reg') == 1:
            G.nodes[u]['reg_1'] = True
            G.nodes[v]['reg_1'] = True
        elif data.get('reg') == 2:
            G.nodes[u]['reg_2'] = True
            G.nodes[v]['reg_2'] = True

    nx.set_node_attributes(G, nodes_coord)

    city_transformed = polygon.to_crs(epsg=crs)
    for node, d in G.nodes(data=True):
        if d['reg_1'] or d['reg_2']:
            point = Point(G.nodes[node]['x'], G.nodes[node]['y'])
            if point.buffer(0.1).intersects(city_transformed.unary_union.boundary):
                G.nodes[node]['exit'] = 1

    return G


def get_graph_from_polygon(polygon: gpd.GeoDataFrame, filter: str = None, crs: int = 3857, retain_all=False) -> nx.MultiDiGraph:
    """Получение графа на основе полигона."""
    buffer_polygon = buffer_and_transform_polygon(polygon, crs)
    if not filter:
        filter = "['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street']"
    graph = ox.graph_from_polygon(buffer_polygon, network_type='drive', custom_filter=filter, truncate_by_edge=True, retain_all=retain_all)
    graph.graph["approach"] = "primal"
    nodes, edges = momepy.nx_to_gdf(graph, points=True, lines=True, spatial_weights=False)
    edges = add_geometry_to_edges(nodes, edges)
    edges['reg'] = edges.apply(lambda row: determine_reg(row['ref'], row['highway']), axis=1)
    nodes = nodes.to_crs(epsg=crs)
    edges = edges.to_crs(epsg=crs)
    edges['exit'] = 0
    edges = update_edges_with_geometry(edges, polygon, crs)
    edges['maxspeed'] = edges['highway'].apply(lambda x: get_max_speed(x))
    nodes_coord = update_nodes_with_geometry(edges, {})
    edges = edges[["highway", "node_start", "node_end", "geometry", 'maxspeed', 'reg', 'ref', 'exit']]
    edges["type"] = 'car'
    edges["geometry"] = edges["geometry"].apply(lambda x: LineString([tuple(round(c, 6) for c in n) for n in x.coords] if x else None))
    G = create_graph(edges, nodes_coord, 'car')
    G = set_node_attributes(G, nodes_coord, polygon, crs)
    G.graph["crs"] = "epsg:" + str(crs)
    G.graph["approach"] = "primal"
    G.graph["graph_type"] = "car graph"
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


def convert_list_attr_to_str(G):
    """
    Convert list attributes to string format for edges in a directed graph.
    
    Parameters:
        graph (DiGraph): Directed graph.
    
    Returns:
        DiGraph: Directed graph with attributes converted to string format.
    """
    graph = G.copy()
    for u, v, key, data in graph.edges(keys=True, data=True):
        for k, value in data.items():
            if isinstance(value, list):
                graph[u][v][key][k] = ','.join(map(str, value))
    return graph


def convert_list_attr_from_str(G):
    """
    Convert string attributes to list format for edges in a directed graph.
    
    Parameters:
        graph (DiGraph): Directed graph.
    
    Returns:
        DiGraph: Directed graph with attributes converted to list format.
    """
    graph = G.copy()
    for u, v, key, data in graph.edges(keys=True, data=True):
        for k, value in data.items():
            if isinstance(value, str) and ',' in value:
                graph[u][v][key][k] = list(map(str, value.split(',')))
    return graph