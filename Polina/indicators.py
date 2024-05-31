import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import numpy as np
from dongraphio import DonGraphio, GraphType
import matplotlib.pyplot as plt
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

# плотность дорог
def density_roads(gdf_polygon: gpd.GeoDataFrame, gdf_line: gpd.GeoDataFrame, crs=3857) -> float:
    area = gdf_polygon.to_crs(epsg=crs).unary_union.area / 1000000
    length = gdf_line.to_crs(epsg=crs).geometry.length.sum()
    print(f'Плотность: {length / area:.3f} км/км^2')

    return round(length / area, 3)

#протяженность дорог каждого типа
def calculate_length_sum_by_status(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.to_crs(epsg=3857)
    gdf['REG_STATUS'] = gdf['REG_STATUS'].fillna(3)
    length_sum_by_status = gdf.groupby('REG_STATUS').geometry.apply(lambda x: x.length.sum() / 1000)
    print(length_sum_by_status.reset_index())
    
    return length_sum_by_status.reset_index()

def availability_matrix(graph, city_points_gdf, service_gdf=None, graph_type=[GraphType.DRIVE], weight='time_min'):
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

def visualize_availability(points, polygons, service_gdf=None, median=True, title='Доступность сервиса, мин'):
    """
    Visualize the service availability on a map with bounding polygons.
    Optionally service points and city points are shown.

    Parameters:
    points (geopandas.GeoDataFrame): GeoDataFrame of points with 'to_service' column.
    polygons (geopandas.GeoDataFrame): GeoDataFrame of polygons.
    service_gdf (geopandas.GeoDataFrame, optional): GeoDataFrame of service points. Defaults to None.
    median (bool, optional): Whether to aggregate time by median among cities in the polygon. Defaults to True.
    title (str, optional): Title of the plot. Defaults to 'Доступность сервиса, мин'.
    """
    points = points.to_crs(polygons.crs)
    
    vmax = points['to_service'].max()
    res = gpd.sjoin(points, polygons, how="left", predicate="within").groupby('index_right').median(['to_service'])
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    polygons.boundary.plot(ax=ax, color='black', linewidth=1).set_axis_off()

    if not median:
        merged = points
        merged.to_crs(points.crs).plot(column='to_service', cmap='RdYlGn_r', ax=ax, legend=True, vmax=vmax, markersize=4).set_axis_off()
    else:
        merged = pd.merge(polygons.reset_index(), res, left_on='index', right_on='index_right')
        merged.to_crs(points.crs).plot(column='to_service', cmap='RdYlGn_r', ax=ax, legend=True, vmax=vmax, markersize=4).set_axis_off()
        if service_gdf is not None:
            service_gdf = service_gdf.to_crs(polygons.crs)
            service_gdf.plot(ax=ax, markersize=7, color='white').set_axis_off()

    plt.title(title)
    plt.show()

def find_nearest(city_points, adj_mx):
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

def find_median(city_points, adj_mx):
    """
    Find the median correspondence time from one city to all others.

    Parameters:
    city_points (geopandas.GeoDataFrame): GeoDataFrame of city points.
    adj_mx (pandas.DataFrame): Adjacency matrix representing distances.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame of points with the 'to_service' column updated to median values.
    """
    points = city_points.copy()
    medians = []
    for index, row in adj_mx.iterrows():
        median = np.median(row[row.index != index])
        medians.append(median / 60) #  convert to minutes
    points['to_service'] = medians
    return points

def get_reg(graph,reg):
    """
    Extract nodes from edges with REG_STATUS==1 as a GeoDataFrame.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with geometries of REG_STATUS==1 nodes.
    """
    n= momepy.nx_to_gdf(graph, points=True, lines=False, spatial_weights=False)
    return n[n[f'reg_{reg}']==True]

