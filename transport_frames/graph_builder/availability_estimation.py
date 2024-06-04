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
        if isinstance(data.get('geometry'), str):
            data['geometry'] = wkt.loads(data['geometry'])
    
    return graph


def make_availability_matrix(graph: nx.Graph, city_points_gdf: gpd.GeoDataFrame, service_gdf: gpd.GeoDataFrame=None,
                         graph_type=[GraphType.DRIVE], weight: str='time_min'):
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
    points = city_points_gdf.copy().to_crs(graph.graph['crs'])
    service_gdf = points if service_gdf is None else service_gdf.to_crs(graph.graph['crs'])

    # Get distances between points and services
    dg = DonGraphio(points.crs.to_epsg())
    dg.set_graph(graph)
    adj_mx = dg.get_adjacency_matrix(points, service_gdf, weight=weight, graph_type=graph_type)
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
    points['to_service'] = min_values
    if (points['to_service'] == np.finfo(np.float64).max).any():
        print('Some services cannot be reached from some nodes of the graph. The nodes were removed from analysis')
        points = points[points['to_service'] < np.finfo(np.float64).max]
    return points

def find_median(city_points: gpd.GeoDataFrame, adj_mx: pd.DataFrame) -> gpd.GeoDataFrame:
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
    points['to_service'] = adj_mx_medians

    if (points['to_service'] == np.finfo(np.float64).max).any():
        print('Some services cannot be reached from some nodes of the graph. The nodes were removed from analysis')
        points = points[points['to_service'] < np.finfo(np.float64).max]

    return points


def get_reg(graph: nx.MultiDiGraph, reg:int) -> gpd.GeoDataFrame:
    """
    Extract nodes from edges with REG_STATUS==1 as a GeoDataFrame.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with geometries of REG_STATUS==1 nodes.
    """
    gdf = momepy.nx_to_gdf(graph, points=True, lines=False, spatial_weights=False)
    return gdf[gdf[f'reg_{reg}']==True]



def grade_polygon(row:pd.Series) -> float:
        """
        Determines the grade of a territory based on its distance to features.

        Parameters:
            row (Series): A pandas Series representing a single row of a GeoDataFrame.

        Returns:
            float: The grade of the territory.
        """
        dist_to_reg1 = row['dist_to_reg1']
        dist_to_reg2 = row['dist_to_reg2']
        dist_to_edge = row['dist_to_edge']
        dist_to_railway_stops = row['dist_to_railway_stops']

        # below numbers are represented in meters

        if dist_to_reg1 < 5_000:
            grade = 4.5
        elif dist_to_reg1 < 10_000 and dist_to_reg2 < 5_000:
            grade = 4.0
        elif dist_to_reg1 < 100_000 and dist_to_reg2 < 5_000:
            grade = 3.0
        elif dist_to_reg1 > 100_000 and dist_to_reg2 < 5_000:
            grade = 2.0
        elif dist_to_reg2 > 5_000 and dist_to_reg1 > 100_000 and dist_to_edge < 5_000:
            grade = 1.0
        else:
            grade = 0.0

        if dist_to_railway_stops < 10_000:
            grade += 0.5

        return grade



def grade_territory(gdf_poly: gpd.GeoDataFrame, graph: nx.MultiDiGraph, stops:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Grades territories based on their distances to reg1, reg2 nodes,edges and train stations.

    Parameters:
        gdf_poly (GeoDataFrame): A GeoDataFrame containing the polygons of the territories to be graded.
        graph (networkx.MultiDiGraph): A MultiDiGraph representing the transportation network.
        stops (GeoDataFrame): A GeoDataFrame containing the locations of railway stops.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the graded territories with added 'grade' column.
    """

    # Extract carcas from graph
    e = [(u, v, k) for u, v, k, d in graph.edges(data=True, keys=True) if d.get('reg') in [1, 2]]
    subgraph = graph.edge_subgraph(e).copy()
    nodes, edges = momepy.nx_to_gdf(subgraph, points=True, lines=True, spatial_weights=False)

    poly = gdf_poly.copy().to_crs(nodes.crs)

    reg1_points = nodes[nodes['reg_1'] == 1]
    reg2_points = nodes[nodes['reg_2'] == 1]
    min_distance = lambda polygon, points: points.distance(polygon).min()
    poly['dist_to_reg1'] = poly.geometry.apply(lambda x: min_distance(x, reg1_points.geometry))
    poly['dist_to_reg2'] = poly.geometry.apply(lambda x: min_distance(x, reg2_points.geometry))
    poly['dist_to_railway_stops'] = poly.geometry.apply(lambda x: min_distance(x, stops.to_crs(nodes.crs).geometry))
    poly['dist_to_edge'] = poly.geometry.apply(lambda x: min_distance(x, edges.geometry))

    poly['grade'] = poly.apply(grade_polygon, axis=1)
    return poly