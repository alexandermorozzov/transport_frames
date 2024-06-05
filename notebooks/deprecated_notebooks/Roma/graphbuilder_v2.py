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
    colorize=True,
)

# Константные словари
HIGHWAY_MAPPING = {
    "motorway": 1,
    "trunk": 1,
    "primary": 2,
    "secondary": 2,
    "tertiary": 3,
    "unclassified": 3,
    "residential": 3,
    "motorway_link": 1,
    "trunk_link": 1,
    "primary_link": 2,
    "secondary_link": 2,
    "tertiary_link": 3,
    "living_street": 3,
}

MAXSPEED = {
    "motorway": 110 / 3.6,
    "motorway_link": 110 / 3.6,
    "primary": 80 / 3.6,
    "primary_link": 80 / 3.6,
    "residential": 60 / 3.6,
    "secondary": 70 / 3.6,
    "secondary_link": 70 / 3.6,
    "tertiary": 60 / 3.6,
    "tertiary_link": 60 / 3.6,
    "trunk": 90 / 3.6,
    "trunk_link": 90 / 3.6,
    "unclassified": 60 / 3.6,
    "living_street": 15 / 3.6,
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
    logger.info("Starting the conversion of the graph.")

    for _, _, data in G.edges(data=True):
        if isinstance(data.get("geometry"), str):
            geometry_wkt = wkt.loads(data["geometry"])
            data["geometry"] = geometry_wkt

    logger.info("The graph was converted!")
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
            if re.match(r"^[МАР]", str(item)):
                return 1
            elif re.match(r"^\d.*[A-Za-zА-Яа-я]", str(item)):
                return 2
        return 3
    elif pd.isna(name_roads):
        # Выставление значения по типу дороги, если значение NaN
        if highway_type:
            return highway_type_to_reg(highway_type)
        return 3
    if re.match(r"^[МАР]", str(name_roads)):
        return 1
    elif re.match(r"^\d.*[A-Za-zА-Яа-я]", str(name_roads)):
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
    node_coords = {
        node["nodeID"]: (node["x"], node["y"]) for _, node in nodes.iterrows()
    }

    # Проход по каждому edge
    for idx, edge in edges.iterrows():
        if edge["geometry"] is None:
            start_coords = node_coords[edge["node_start"]]
            end_coords = node_coords[edge["node_end"]]
            line_geom = LineString([start_coords, end_coords])
            edges.at[idx, "geometry"] = line_geom
    return edges


def buffer_and_transform_polygon(polygon: gpd.GeoDataFrame, crs: int = 3857):
    """Создание буферной зоны вокруг полигона и преобразование координат."""
    return polygon.to_crs(crs).geometry.buffer(3000).to_crs(4326).unary_union


def update_edges_with_geometry(edges, polygon, crs):
    """Обновление геометрии ребер на основе пересечений с границей города."""
    city_transformed = polygon.to_crs(epsg=crs)
    edges["intersections"] = edges["geometry"].intersection(
        city_transformed.unary_union
    )
    edges["geometry"] = edges["intersections"]
    mask = (
        edges[edges["reg"].isin([1, 2])]
        .buffer(10)
        .intersects(city_transformed.unary_union.boundary)
    )
    edges.loc[edges["reg"].isin([1, 2]) & mask, "exit"] = 1
    edges.drop(columns=["intersections"], inplace=True)
    edges = edges.explode(index_parts=True)
    edges = edges[~edges["geometry"].is_empty]
    edges = edges[edges["geometry"].geom_type == "LineString"]
    return edges


def update_nodes_with_geometry(edges, nodes_coord):
    """Обновление координат узлов на основе новой геометрии ребер."""
    for _, row in edges.iterrows():
        start_node = row["node_start"]
        end_node = row["node_end"]
        if start_node not in nodes_coord:
            nodes_coord[start_node] = {
                "x": row["geometry"].coords[0][0],
                "y": row["geometry"].coords[0][1],
            }
        if end_node not in nodes_coord:
            nodes_coord[end_node] = {
                "x": row["geometry"].coords[-1][0],
                "y": row["geometry"].coords[-1][1],
            }
    return nodes_coord


def create_graph(edges, nodes_coord, graph_type):
    """Создание графа на основе ребер и координат узлов."""
    G = nx.MultiDiGraph()
    travel_type = "walk" if graph_type == "walk" else "car"
    for _, edge in edges.iterrows():
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geom = (
            LineString(
                (
                    [
                        (nodes_coord[p1]["x"], nodes_coord[p1]["y"]),
                        (nodes_coord[p2]["x"], nodes_coord[p2]["y"]),
                    ]
                )
            )
            if not edge.geometry
            else edge.geometry
        )
        length = round(geom.length, 3)
        time_sec = round(length / edge.maxspeed, 3)
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
            ref=edge.ref,
            is_exit=edge.exit,
        )
    return G


def set_node_attributes(G, nodes_coord, polygon, crs, country_polygon=None):
    """Установка атрибутов узлов на основе атрибутов ребер."""
    for node in G.nodes:
        G.nodes[node]["reg_1"] = False
        G.nodes[node]["reg_2"] = False

    for u, v, data in G.edges(data=True):
        if data.get("reg") == 1:
            G.nodes[u]["reg_1"] = True
            G.nodes[v]["reg_1"] = True
        elif data.get("reg") == 2:
            G.nodes[u]["reg_2"] = True
            G.nodes[v]["reg_2"] = True

    nx.set_node_attributes(G, nodes_coord)

    city_transformed = polygon.to_crs(epsg=crs)
    if country_polygon is not None:
        country_transformed = country_polygon.to_crs(epsg=crs)
    for node, d in G.nodes(data=True):
        # d['ref'] = None
        if d["reg_1"] or d["reg_2"]:
            point = Point(G.nodes[node]["x"], G.nodes[node]["y"])
            if point.buffer(0.1).intersects(city_transformed.unary_union.boundary):
                G.nodes[node]["exit"] = 1
                get_edges_with_node = lambda G, node: list(
                    G.edges(node, data=True)
                ) + list(G.in_edges(node, data=True))
                # find nodes on country border
                if country_polygon is not None:
                    if point.buffer(0.1).intersects(
                        country_transformed.unary_union.boundary
                    ):
                        G.nodes[node]["exit_country"] = 1

    return G


def get_graph_from_polygon(
    polygon: gpd.GeoDataFrame, filter: str = None, crs: int = 3857, country_polygon=None
) -> nx.MultiDiGraph:
    """Получение графа на основе полигона."""
    buffer_polygon = buffer_and_transform_polygon(polygon, crs)
    if not filter:
        filter = f"['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street']"
    graph = ox.graph_from_polygon(
        buffer_polygon,
        network_type="drive",
        custom_filter=filter,
        truncate_by_edge=True,
    )
    graph.graph["approach"] = "primal"
    nodes, edges = momepy.nx_to_gdf(
        graph, points=True, lines=True, spatial_weights=False
    )
    edges = add_geometry_to_edges(nodes, edges)
    edges["reg"] = edges.apply(
        lambda row: determine_reg(row["ref"], row["highway"]), axis=1
    )
    nodes = nodes.to_crs(epsg=crs)
    edges = edges.to_crs(epsg=crs)
    edges["exit"] = 0
    edges = update_edges_with_geometry(edges, polygon, crs)
    edges["maxspeed"] = edges["highway"].apply(lambda x: get_max_speed(x))
    nodes_coord = update_nodes_with_geometry(edges, {})
    edges = edges[
        [
            "highway",
            "node_start",
            "node_end",
            "geometry",
            "maxspeed",
            "reg",
            "ref",
            "exit",
        ]
    ]
    edges["type"] = "car"
    edges["geometry"] = edges["geometry"].apply(
        lambda x: LineString(
            [tuple(round(c, 6) for c in n) for n in x.coords] if x else None
        )
    )
    G = create_graph(edges, nodes_coord, "car")
    G = set_node_attributes(
        G, nodes_coord, polygon, crs, country_polygon=country_polygon
    )
    G.graph["crs"] = "epsg:" + str(crs)
    G.graph["approach"] = "primal"
    G.graph["graph_type"] = "car graph"
    return G


def assign_city_names_to_nodes(
    points,
    nodes,
    graph,
    name_attr="city_name",
    node_id_attr="nodeID",
    name_col="name",
    max_distance=200,
):

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
    project_roads_city = gpd.sjoin_nearest(
        points, nodes, how="left", distance_col="distance", max_distance=max_distance
    )

    # Присваиваем имена городов узлам графа с отслеживанием прогресса
    for enum, index in enumerate(project_roads_city[node_id_attr].values):
        city_name = project_roads_city.loc[enum, name_col]
        for _, d in G.nodes(data=True):
            if d.get(node_id_attr) == index:
                d[name_attr] = city_name
    return G


def prepare_graph(graph_orig: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Prepare the graph for analysis by converting node names to integers and extract edge geometries from WKT format.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    networkx.MultiDiGraph: The prepared graph with node names as integers and geometries as WKT.
    """
    graph = nx.convert_node_labels_to_integers(graph_orig)
    for _, _, data in graph.edges(data=True):
        if isinstance(data.get("geometry"), str):
            data["geometry"] = wkt.loads(data["geometry"])

    return graph


def determine_ref_type(ref):
    """Converts ref string to numeric type"""
    patterns = {
        1.1: r"М-\d+",
        1.2: r"Р-\d+",
        1.3: r"А-\d+",
        2.1: r"..Р-\d+",
        2.2: r"..А-\d+",
    }
    for value in ref:
        for ref_type, pattern in patterns.items():
            if re.match(pattern, value):
                return ref_type
    return 2.3


def filter_polygons_by_buffer(gdf_polygons, polygon_buffer, crs=3857):
    """Extracts current region polygon from gdf of all regions and take only bordering polygons"""
    gdf_polygons = gdf_polygons.to_crs(crs)
    polygon_buffer = polygon_buffer.to_crs(crs)
    polygon_buffer = gpd.GeoDataFrame(
        {"geometry": polygon_buffer.buffer(0.1)}, crs=polygon_buffer.crs
    ).to_crs(gdf_polygons.crs)
    gdf_polygons = gpd.overlay(
        gdf_polygons[gdf_polygons["type"] == "boundary"], polygon_buffer, "difference"
    )  # cut current region out of all region
    buffer_polygon = polygon_buffer.buffer(5000)
    filtered_gdf = gdf_polygons[gdf_polygons.intersects(buffer_polygon.unary_union)]
    return filtered_gdf


def add_region_attr(n, regions, polygon_buffer, frame, crs=3857):
    """
    Add a 'border_region' attribute to nodes based on intersection with region polygons.

    Parameters:
    - n (GeoDataFrame): Nodes GeoDataFrame with 'exit' indicating border nodes.
    - regions (GeoDataFrame): Regions GeoDataFrame with an 'id' column for region identifiers.
    - polygon_buffer (Polygon): Polygon to create a buffer for filtering regions.
    - frame (NetworkX graph): Graph with nodes to update with 'border_region' attributes.
    - crs (int, optional): Coordinate reference system. Default is 3857.

    Returns:
    - n (GeoDataFrame): Updated nodes GeoDataFrame with 'border_region' attribute.
    """
    exits = n[n["exit"] == 1]
    exits = exits.to_crs(crs)
    exits["buf"] = exits["geometry"].buffer(1000)
    filtered_regions = filter_polygons_by_buffer(regions, polygon_buffer)
    joined_gdf = gpd.sjoin(
        exits.set_geometry("buf"),
        filtered_regions.to_crs(exits.crs),
        how="left",
        op="intersects",
    )
    joined_gdf = joined_gdf.drop_duplicates(subset="geometry")
    exits["border_region"] = joined_gdf["id"]

    n["border_region"] = exits["border_region"]

    for i, (node, data) in enumerate(frame.nodes(data=True)):
        data["border_region"] = n.iloc[i]["border_region"]
    return n


def get_weight(start, end, exit):
    """
    Calculate the weight based on the type of start and end references and exit status.

    Parameters:
    start (float): Reference type of the start node.
    end (float): Reference type of the end node.
    exit (int): Exit status (1 if exit, else 0).

    Returns:
    float: Calculated weight based on the provided matrix.
    """
    dict = {1.1: 0, 1.2: 1, 1.3: 2, 2.1: 3, 2.2: 4, 2.3: 5}
    if exit == 1:
        matrix = [
            [0.12, 0.12, 0.12, 0.12, 0.12, 0.12],  # 2.1.1
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],  # 2.1.2
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08],  # 2.1.3
            [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],  # 2.2.1
            [0.06, 0.06, 0.06, 0.06, 0.06, 0.06],  # 2.2.2
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 2.2.3
        ]
    else:

        matrix = [
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08],  # 2.1.1
            [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],  # 2.1.2
            [0.06, 0.06, 0.06, 0.06, 0.06, 0.06],  # 2.1.3
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 2.2.1
            [0.04, 0.04, 0.04, 0.04, 0.04, 0.04],  # 2.2.2
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # 2.2.3
        ]
    return matrix[dict[end]][dict[start]]


def weigh_roads(frame):
    """
    Calculate and normalize the weights of roads between exits in a road network.

    Parameters:
    frame (networkx.Graph): The road network graph where nodes represent intersections or exits
                             and edges represent road segments with 'time_min' as a weight attribute.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame with frame edges with corresponding weights and normalized weights.
    """
    n, e = momepy.nx_to_gdf(frame)
    e = e.loc[e.groupby(["node_start", "node_end"])["time_sec"].idxmin()]
    e["weight"] = 0
    n["weight"] = 0
    exits = n[n["exit"] == 1]
    for i1, start_node in exits.iterrows():
        for i2, end_node in exits.iterrows():
            if i1 == i2:
                continue
            if (
                pd.notna(start_node["border_region"])
                and start_node["border_region"] == end_node["border_region"]
            ):
                continue
            if start_node.geometry.buffer(15000).intersects(
                end_node.geometry.buffer(15000)
            ) and (
                pd.isna(start_node["exit_country"]) == pd.isna(end_node["exit_country"])
            ):
                continue
            if start_node["exit_country"] == 1 and end_node["exit_country"] == 1:
                continue

            weight = get_weight(
                start_node["ref_type"], end_node["ref_type"], end_node["exit_country"]
            )

            try:
                path = nx.dijkstra_path(frame, i1, i2, weight="time_min")
            except nx.NetworkXNoPath:
                continue
            for j in range(len(path) - 1):
                n.loc[(n["nodeID"] == path[j]), "weight"] += weight
                e.loc[
                    (e["node_start"] == path[j]) & (e["node_end"] == path[j + 1]),
                    "weight",
                ] += weight
            n.loc[(n["nodeID"] == path[j + 1]), "weight"] += weight
    e["normalized_weight"] = round(e["weight"] / e["weight"].max(), 3)
    n["normalized_weight"] = round(n["weight"] / n["weight"].max(), 3)

    for i, (node, data) in enumerate(frame.nodes(data=True)):
        data["weight"] = n.iloc[i]["weight"]

    return frame


def get_frame(graph, regions=None, polygon=None):
    """
    Process the input graph to create a frame subgraph with specific attributes.

    This function prepares the input graph, filters edges based on 'reg' attribute,
    extracts nodes and edges, assigns 'ref' and 'ref_type' attributes to nodes,
    and optionally adds region attributes if regions and city are provided.

    Parameters:
    - graph (NetworkX graph): The input graph to be processed.
    - regions (GeoDataFrame, optional): GeoDataFrame of region polygons to assign region attributes.
    - polygon (GeoDataFrame, optional): GeoDataFrame of a city polygon used for region filtering.

    Returns:
    - frame (NetworkX graph): The processed subgraph with added attributes.
    """
    prepared_graph = prepare_graph(graph)
    e = [
        (u, v, k)
        for u, v, k, d in prepared_graph.edges(data=True, keys=True)
        if d.get("reg") in ([1, 2])
    ]
    frame = prepared_graph.edge_subgraph(e).copy()
    n, e = momepy.nx_to_gdf(frame)
    n["ref"] = None
    ref_edges = e[e["ref"].notna()]

    for idx, node in n.iterrows():

        if node["exit"] == 1:
            point = node.geometry
            distances = ref_edges.geometry.distance(point)
            if not distances.empty:
                nearest_edge = ref_edges.loc[distances.idxmin()]
                ref_value = nearest_edge["ref"]
                if isinstance(ref_value, list):
                    ref_value = tuple(ref_value)
                if isinstance(ref_value, str):
                    ref_value = (ref_value,)
                n.at[idx, "ref"] = ref_value
                n.at[idx, "ref_type"] = determine_ref_type(ref_value)
    n = n.set_index("nodeID")
    if regions is not None and polygon is not None:
        n = add_region_attr(n, regions, polygon, frame)
    for i, (node, data) in enumerate(frame.nodes(data=True)):
        data["ref_type"] = n.iloc[i]["ref_type"]
        data["ref"] = n.iloc[i]["ref"]
    mapping = {node: data["nodeID"] for node, data in frame.nodes(data=True)}
    frame = nx.relabel_nodes(frame, mapping)
    frame = weigh_roads(frame)
    return frame


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
                graph[u][v][key][k] = ",".join(map(str, value))
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
            if isinstance(value, str) and "," in value:
                graph[u][v][key][k] = list(map(str, value.split(",")))
    return graph
