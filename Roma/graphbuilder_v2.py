import re
import momepy
import numpy as np
import osmnx as ox
import pandas as pd
import folium
import networkx as nx
from tqdm import tqdm
from shapely.geometry import LineString, Polygon
from shapely import wkt

# перевод в геометрию
def convert_geometry_to_wkt(graph):
    """TODO: сравнить с dongraphio"""
    for _, _, data in graph.edges(data=True):
        if isinstance(data['geometry'], str):
            geometry_wkt = wkt.loads(data['geometry'])
            data['geometry'] = geometry_wkt
    print('The graph was converted!')


# Функция для определения значения REG
def determine_reg(value, highway_type=None):
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


def highway_type_to_reg(highway_type):
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


def custom_map(highway_types):
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
    

def get_graph_polygon(polygon: Polygon, filter:dict=None, crs:int=3857):
    if not filter:
        filter = "['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street']"
    graph = ox.graph_from_polygon(polygon, network_type='drive', custom_filter=filter, truncate_by_edge=True)


    nodes, edges = momepy.nx_to_gdf(graph, points=True, lines=True, spatial_weights=False)
    edges['reg'] = edges.apply(lambda row: determine_reg(row['ref'], row['highway']), axis=1)
    nodes = nodes.to_crs(epsg=crs)
    edges = edges.to_crs(epsg=crs)
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
            time_sec= time_sec,
            highway=edge.highway,
            maxspeed=edge.maxspeed,
            reg=edge.reg,
            ref=edge.ref
        )
    nx.set_node_attributes(G, nodes_coord)
    G.graph["crs"] = "epsg:" + str(crs)
    G.graph["graph_type"] = travel_type + " graph"

    return G


