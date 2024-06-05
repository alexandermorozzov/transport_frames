# Импорт стандартных библиотек Python для обработки данных
import pandas as pd
import numpy as np

# Импорт библиотек для работы с графами и сетями
import networkx as nx

# Импорт библиотек для работы с геоданными
import geopandas as gpd
from shapely import wkt

# Импорт специализированных библиотек для работы с графами
from dongraphio import DonGraphio, GraphType
import momepy


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


def make_availability_matrix(
    graph: nx.Graph,
    city_points_gdf: gpd.GeoDataFrame,
    service_gdf: gpd.GeoDataFrame = None,
    graph_type=[GraphType.DRIVE],
    weight: str = "time_min",
):
    """
    Compute the availability matrix showing distances between city points and service points.
    If service_gdf is None, adjacency matrix shows connectivity between cities.

    Parameters:
    graph (networkx.Graph): The input graph.
    city_points_gdf (geopandas.GeoDataFrame): GeoDataFrame of city points.
    service_gdf (geopandas.GeoDataFrame, optional): GeoDataFrame of service points. Defaults to None.
    graph_type (list, optional): List of graph types to consider. Defaults to [GraphType.DRIVE].
    weight (str, optional): The edge attribute to use as weight (takes either 'time_min' or 'length_meter'). Defaults to 'time_min'.

    Returns:
    pandas.DataFrame: The adjacency matrix representing distances.
    """
    points = city_points_gdf.copy().to_crs(graph.graph["crs"])
    service_gdf = (
        points if service_gdf is None else service_gdf.to_crs(graph.graph["crs"])
    )

    # Get distances between points and services
    dg = DonGraphio(points.crs.to_epsg())
    dg.set_graph(graph)
    adj_mx = dg.get_adjacency_matrix(
        points, service_gdf, weight=weight, graph_type=graph_type
    )
    return adj_mx


def find_nearest_pois(city_points, adj_mx):
    """
    Find the nearest services from city points using the adjacency matrix.

    Parameters:
    city_points (geopandas.GeoDataFrame): GeoDataFrame of city points.
    adj_mx (pandas.DataFrame): Adjacency matrix representing distances or time.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame of points with the 'to_service' column updated.
    """
    points = city_points.copy()
    # Find the nearest service
    min_values = adj_mx.min(axis=1)
    points["to_service"] = min_values
    if (points["to_service"] == np.finfo(np.float64).max).any():
        print(
            "Some services cannot be reached from some nodes of the graph. The nodes were removed from analysis"
        )
        points = points[points["to_service"] < np.finfo(np.float64).max]
    return points


def find_median(
    city_points: gpd.GeoDataFrame, adj_mx: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Find the median correspondence time from one city to all others.

    Parameters:
    city_points (geopandas.GeoDataFrame): GeoDataFrame of city points.
    adj_mx (pandas.DataFrame): Adjacency matrix representing distances.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame of points with the 'to_service' column updated to median values.
    """
    points = city_points.copy()
    adj_mx_medians = adj_mx.drop(columns=adj_mx.index).apply(np.median, axis=1)
    points["to_service"] = adj_mx_medians

    if (points["to_service"] == np.finfo(np.float64).max).any():
        print(
            "Some services cannot be reached from some nodes of the graph. The nodes were removed from analysis"
        )
        points = points[points["to_service"] < np.finfo(np.float64).max]

    return points


def get_reg(graph: nx.MultiDiGraph, reg: int) -> gpd.GeoDataFrame:
    """
    Extract nodes from edges with REG_STATUS==1 as a GeoDataFrame.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with geometries of REG_STATUS==1 nodes.
    """
    gdf = momepy.nx_to_gdf(graph, points=True, lines=False, spatial_weights=False)
    return gdf[gdf[f"reg_{reg}"] == True]
