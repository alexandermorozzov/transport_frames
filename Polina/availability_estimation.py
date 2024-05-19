import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import numpy as np
from dongraphio import DonGraphio, GraphType
import matplotlib.pyplot as plt

def prepare_graph(graph):
    """
    Prepare the graph for analysis by converting node names to integers and edge geometries to WKT format.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    networkx.MultiDiGraph: The prepared graph with node names as integers and geometries as WKT.
    """
    def convert_node_names_to_int(graph):
        node_mapping = {node: int(node) for node in graph.nodes()}
        G_int = nx.relabel_nodes(graph, node_mapping)
        return G_int

    def convert_geometry_to_wkt(graph):
        for _, _, data in graph.edges(data=True):
            if isinstance(data['geometry'], str):
                geometry_wkt = wkt.loads(data['geometry'])
                data['geometry'] = geometry_wkt
        return graph

    graph = convert_node_names_to_int(graph)
    graph = convert_geometry_to_wkt(graph)

    return graph

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
    if (points['to_service'] > 1e20).any():
        print('Some services cannot be reached, they were removed')
        points = points[points['to_service'] < 1e20]
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
        medians.append(median / 60)

    median_df = pd.DataFrame({'Median': medians})
    points['to_service'] = median_df['Median']
    return points

def get_reg1(graph):
    """
    Extract nodes from edges with REG_STATUS==1 and create a GeoDataFrame of unique start and end points.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with geometries of REG_STATUS==1 nodes.
    """
    unique_points = set()

    # Extract edges with reg_status=1
    for u, v, data in graph.edges(data=True):
        if data.get('REG_STATUS') == 1:
            start_node = graph.nodes[u]
            end_node = graph.nodes[v]
            
            # Ensure nodes have 'x' and 'y' coordinates and convert them to floats
            if 'x' in start_node and 'y' in start_node and 'x' in end_node and 'y' in end_node:
                try:
                    start_point = Point(float(start_node['x']), float(start_node['y']))
                    end_point = Point(float(end_node['x']), float(end_node['y']))
                    unique_points.add(start_point)
                    unique_points.add(end_point)
                except ValueError:
                    print(f"Invalid coordinates for nodes {u} or {v}")

    unique_points_list = list(unique_points)
    gdf = gpd.GeoDataFrame(geometry=unique_points_list, crs=graph.graph['crs'])

    return gdf


