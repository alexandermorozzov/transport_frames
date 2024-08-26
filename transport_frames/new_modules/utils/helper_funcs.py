import geopandas as gpd
import networkx as nx
import osmnx as ox
import logging
from shapely.geometry import LineString
import sys
import momepy
import shapely.wkt as wkt

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
        if isinstance(data.get("geometry"), str):
            geometry_wkt = wkt.loads(data["geometry"])
            data["geometry"] = geometry_wkt

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
            if isinstance(value, str) and "," in value and k != 'geometry':
                graph[u][v][key][k] = list(map(str, value.split(",")))
    return graph

def buffer_and_transform_polygon(polygon: gpd.GeoDataFrame, crs: int = 3857):
    """Создание буферной зоны вокруг полигона и преобразование координат."""
    return polygon.to_crs(crs).geometry.buffer(3000).to_crs(4326).unary_union


def _determine_ref_type(ref):
    """
    Determine the reference type based on the reference list.

    Parameters:
    - ref (tuple): A tuple of reference types.

    Returns:
    - float: Determined reference type.
    """
    if 'A' in ref:
        return 1.1
    elif 'B' in ref:
        return 1.2
    elif 'C' in ref:
        return 2.1
    elif 'D' in ref:
        return 2.2
    elif 'E' in ref:
        return 3.1
    elif 'F' in ref:
        return 3.2
    elif 'G' in ref:
        return 3.3
    return 3.3