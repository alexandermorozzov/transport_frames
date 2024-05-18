# import momepy
# import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
import ast
from shapely import wkt
from typing import Optional
import numpy as np
# from shapely import LineString, Point, MultiPoint
from dongraphio import DonGraphio,GraphType
import matplotlib.pyplot as plt
import momepy
import math
from shapely import Point


# prepare graph data
def prepare_graph(graph):

    # graph = nx.MultiDiGraph(graph)
    # graph.graph['graph_type'] = 'car graph'
    # graph.graph['car speed']= '283.33'  
    # for node,data in graph.nodes(data=True):
    #     data['x'] = str(ast.literal_eval(node)[0])
    #     data['y'] = str(ast.literal_eval(node)[1])

    # node_mapping = {node: int(i) for i,node in enumerate(graph.nodes())}
    # graph = nx.relabel_nodes(graph, node_mapping)

    # for node,data in enumerate(graph.nodes(data=True)):
    #     data[1]['nodeID'] = node

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

    # использование функции:
    graph = convert_node_names_to_int(graph)
    graph = convert_geometry_to_wkt(graph)

    return graph


# return a gdf of points wth 'to_service' column in min
def availability_matrix(graph,city_points_gdf, service_gdf=None, graph_type=[GraphType.DRIVE],weight='time_min'):
    
    # graph = prepare_graph(graph)
    points = city_points_gdf.copy().to_crs(graph.graph['crs'])
    service_gdf = points if service_gdf is None else service_gdf.to_crs(graph.graph['crs'])


    # get distances between points and services
    dg = DonGraphio(points.crs.to_epsg())
    dg.set_graph(graph)
    adj_mx = dg.get_adjacency_matrix(points, service_gdf, weight=weight, graph_type=graph_type)
    return adj_mx


# visualize availability on a map with bounding polygons
# if median = True, time is aggregated by medians among polygon
# types = dots/gpsp/mo

def visualize_availability(points, polygons, service_gdf = None, median = True,title='Доступность сервиса, мин'):
    points = points.to_crs(polygons.crs)
    
    vmax = points.to_service.max()
    res= gpd.sjoin(points, polygons, how="left", predicate="within").groupby('index_right').median(['to_service'])
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    polygons.boundary.geometry.plot(ax=ax, color='black', linewidth=1).set_axis_off()
    if not median:
        merged = points
        merged.to_crs(points.crs).plot(column = 'to_service',cmap='RdYlGn_r', ax=ax, legend=True,vmax=vmax,markersize=4).set_axis_off()
    else:
        merged = pd.merge(polygons.reset_index(),res,left_on='index',right_on='index_right')
        merged.to_crs(points.crs).plot(column = 'to_service',cmap='RdYlGn_r', ax=ax, legend=True,vmax=vmax,markersize=4).set_axis_off()
        if service_gdf is not None:
            service_gdf = service_gdf.to_crs(polygons.crs)
            service_gdf.plot( ax=ax, markersize=7,color='white').set_axis_off()
    plt.title(title)
    # plt.savefig('Средняя доступность, мин.png', dpi=600)
    plt.show()


# find nearest services from cities and adjacency matrix
def find_nearest(city_points, adj_mx):
    points = city_points.copy()
    # find the nearest service
    min_values = adj_mx.min(axis=1)
    points['to_service'] = min_values
    if (points['to_service'] > 1e20).any():
        print('Some services cannot be reached, they were removed')
        points = points[points['to_service'] < 1e20]
    return points


# find median correspondency time from one city to all of the others
def find_median(city_points,adj_mx):
    points = city_points.copy()
    medians = []
    for index, row in adj_mx.iterrows():
        median = np.median(row[row.index != index])  
        medians.append(median/60)

    median_df = pd.DataFrame({'Median': medians})
    points['to_service'] = median_df
    return points


def get_reg1(graph):
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
    gdf = gpd.GeoDataFrame(geometry=unique_points_list, crs=graph.graph['crs'])  # Adjust CRS as needed

    return gdf