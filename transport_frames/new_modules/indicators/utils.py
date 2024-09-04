import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import sys
sys.path.append('/Users/polina/Desktop/github/transport_frames')

from transport_frames.new_modules.utils.adj_calc import AdjacencyCalculator
import tqdm
import networkx as nx
from dongraphio import GraphType
import shapely.wkt as wkt
from dongraphio import DonGraphio, GraphType
import numpy as np


def get_accessibility(citygraph,territory = None):
    services = ['capital', 'reg_1', 'fuel', 'railway_stops', 'local_aero', 'international_aero', 'ports', ]

    weight_dict = {
        'fuel': 'time_min',
        'railway_stops': 'time_min',
        'local_aero': 'time_min',
        'international_aero': 'time_min',
        'ports': 'time_min',
        'region_capital': 'length_meter',
        'fed_roads': 'length_meter'
    }

    node_distances = { #ноды админ центров
        node: {
            'name': citygraph.nodes[node].get('name'),
            'x': citygraph.nodes[node].get('x'),
            'y': citygraph.nodes[node].get('y'),
            **{service: None for service in services}
        }
        for node in citygraph.nodes if citygraph.nodes[node].get('points') == 1
    }
    if territory: # нода центроиды
        node_distances = {
        node: {
            'name': citygraph.nodes[node].get('name'),
            'x': citygraph.nodes[node].get('x'),
            'y': citygraph.nodes[node].get('y'),
            **{service: None for service in services}
        }
        for node in citygraph.nodes if citygraph.nodes[node].get('neud_center') == 1
    }
    
    for node in tqdm.tqdm(node_distances):
        distances = bfs(citygraph, node, services, weight_dict)
        for service in services:
            node_distances[node][service] = round(distances[service], 3)if (pd.notnull(distances[service]) and distances[service] != float('inf')) else None
    
    # Convert to GeoDataFrame
    df = pd.DataFrame.from_dict(node_distances, orient='index')
    df['geometry'] = df.apply(lambda row: shapely.Point(row['x'], row['y']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf

def density_roads(gdf_polygon: gpd.GeoDataFrame, gdf_line: gpd.GeoDataFrame, crs=3857) -> float:
    """
    This function calculates the density of roads (in km) per square kilometer area.

    Parameters:
    gdf_polygon (gpd.GeoDataFrame): A GeoDataFrame containing the polygons representing the area(s) in which to calculate road density.
    gdf_line (gpd.GeoDataFrame): A GeoDataFrame containing the lines representing the roads.
    crs (int, optional): The Coordinate Reference System to be used for the calculation. Defaults to Web Mercator (EPSG:3857).

    Returns:
    float: The calculated road density in km per square kilometer of the provided polygon areas.
    """
    if not isinstance(gdf_polygon, gpd.GeoDataFrame):
        gdf_polygon = gpd.GeoDataFrame({'geometry': gdf_polygon}, crs=gdf_polygon.crs).to_crs(gdf_polygon.crs)
    area = gdf_polygon.to_crs(epsg=crs).unary_union.area / 1000000
    gdf_line = gpd.overlay(gdf_line.to_crs(epsg=crs), gdf_polygon.to_crs(epsg=crs)).copy()
    length = gdf_line.to_crs(epsg=crs).geometry.length.sum()
    return round(length / area, 3)

def assign_services_names_to_nodes(
        service_dict,
        nodes,
        graph,
        node_id_attr="nodeID",
        max_distance=10000,
        crs=3857
):
    # Копия графа
    G = graph.copy()
    for key, points in service_dict.items():
        if points.size != 0 and key not in ['train_paths', 'oopt', 'water_objects']:
            name_attr = key
            print(name_attr)
            points = points.to_crs(crs)
            nodes = nodes.to_crs(crs)
            # Присоединяем ближайшие города к узлам
            project_roads_city = gpd.sjoin_nearest(
                points, nodes, how="left", distance_col="distancecol", max_distance=max_distance
            )

            # Присваиваем имена городов узлам графа с отслеживанием прогресса
            for enum, index in enumerate(project_roads_city[node_id_attr].values):
                for _, d in G.nodes(data=True):
                    if d.get(node_id_attr) == index:
                        d[name_attr] = 1
                        if name_attr != 'points':
                            d['service'] = 1
        else:
            print('no', key, '(((((((')
    return G

def aggregate_services_by_polygon(services_gdf, polygons_gdf):
    """
    This function counts the services aggregating the number based on the border polygons.

    Parameters:
    services_gdf - GeoDataFrame of service nodes.
    polygons_gdf - GeoDataFrame of polygons representing areas for aggregation.

    Returns:
    GeoDataFrame with 'service_count' column representing number of services for each area.
    """
    joined = gpd.sjoin(services_gdf, polygons_gdf, how="left", predicate='within')
    service_counts = joined.groupby('index_right').size().reset_index(name='service_count')
    result = polygons_gdf.reset_index().merge(service_counts, how='left', left_on='index', right_on='index_right')
    result['service_count'] = result['service_count'].fillna(0)
    result = result.drop(columns=['index_right'])
    result = gpd.GeoDataFrame(result, geometry='geometry')
    return result


def aggregate_routes_by_polygon(routes_gdf, polygons_gdf, route_column='number_of_routes'):
    """
    This function counts the number of routes aggregating them based on the border polygons.

    Parameters:
    routes_gdf - GeoDataFrame of service edges.
    polygons_gdf - GeoDataFrame of polygons representing areas for aggregation.

    Returns:
    GeoDataFrame with 'number_of_routes' column representing number of routes for each area.
    """
    polygons_gdf = polygons_gdf.reset_index().to_crs(routes_gdf.crs)
    routes_intersect = gpd.overlay(routes_gdf, polygons_gdf, how='intersection')
    route_counts = routes_intersect.groupby('index')['desc'].nunique().reset_index(name=route_column)
    result = polygons_gdf.merge(route_counts, how='left', left_on='index', right_on='index')
    result[route_column] = result[route_column].fillna(0)
    return gpd.GeoDataFrame(result, geometry='geometry')


def aggregate_road_lengths(roads_gdf, polygons_gdf, crs, reg=False):
    """
    This function counts the total length of roads aggregating them based on the border polygons and/or attribute agg (usually equal to reg).

    Parameters:
    roads_gdf - GeoDataFrame of road edges.
    polygons_gdf - GeoDataFrame of polygons representing areas for aggregation.
    agg: str - Name of column from roads_gdf for aggregation

    Returns:
    GeoDataFrame with 'number_of_routes' column representing number of routes for each area.
    """
    roads_gdf = roads_gdf.to_crs(crs)
    polygons_gdf = polygons_gdf.to_crs(crs).reset_index(drop=False)
    roads_intersect = gpd.overlay(roads_gdf, polygons_gdf, how='intersection')
    roads_intersect['length_km'] = roads_intersect.geometry.length / 1000

    if reg:
        length_columns = {1: 'reg1_length', 2: 'reg2_length', 3: 'reg3_length'}
        length_sums = roads_intersect.groupby(['index', 'reg'])['length_km'].sum().unstack(fill_value=0).rename(
            columns=length_columns).reset_index()
        result = polygons_gdf.merge(length_sums, how='left', left_on='index', right_on='index').fillna(0)
    else:
        length_sums = roads_intersect.groupby('index')['length_km'].sum().reset_index(name='total_length_km')
        result = polygons_gdf.merge(length_sums, how='left', left_on='index', right_on='index').fillna(0)

    return gpd.GeoDataFrame(result, geometry='geometry')


def new_connectivity(graph, city_nodes, local_crs=3826, inter=False):
    citygraph_copy = graph.copy()
    citygraph_copy.add_nodes_from(citygraph_copy.nodes(data=True))
    citygraph_copy.add_edges_from(citygraph_copy.edges(data=True, keys=True))
    citygraph_copy.graph = graph.graph
    n = city_nodes
    p = n[n['points'] == 1].to_crs(local_crs).copy()
    gdf_buffers = p.to_crs(local_crs).copy()
    
    gdf_buffers['geometry'] = gdf_buffers['geometry'].buffer(100)
    print(gdf_buffers.crs)
    print(citygraph_copy.graph)

    for e1, e2, data in citygraph_copy.edges(data=True):
        data['weight'] = data['time_min']
        
        if inter:
            if data['type'] in ["walk", "drive", "subway", "tram", "bus", "trolleybus"]:
                data['transport_type'] = data['type']
            else:
                data['transport_type'] = 'tram'
        else:
            data['transport_type'] = 'drive'

    ac = AdjacencyCalculator(blocks=gdf_buffers, graph=citygraph_copy)

    adj_mx_old = ac.get_dataframe()
    median_points = find_median(p, adj_mx_old)

    return median_points



def bfs(G, start_node, services, weight_dict):
    distances = {service: float('inf') for service in services}
    visited = set()
    queue = [(start_node, 0)]

    while queue:
        current_node, current_distance = queue.pop(0)

        if current_node in visited:
            continue

        visited.add(current_node)

        for service in services:
            if G.nodes[current_node].get(service) == 1 and distances[service] == float('inf'):
                distances[service] = current_distance

        for neighbor in G.successors(current_node):  # Use successors to account for direction
            for key in G[current_node][neighbor]:
                if neighbor not in visited:
                    weight = weight_dict.get(service, 'time_min')
                    queue.append((neighbor, current_distance + G.edges[current_node, neighbor, key].get(weight, 1)))

    return distances


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
        medians.append(median / 60)  # convert to hours
    points['to_service'] = medians
    return points



def availability_matrix(
        graph,
        city_points_gdf,
        service_gdf=None,
        graph_type=[GraphType.DRIVE],
        weight="time_min",
        check_nearest=None
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
    check_nearest (int, optional): If positive, distance is calculated to n nearet services

    Returns:
    pandas.DataFrame: The adjacency matrix representing distances.
    """
    points = city_points_gdf.copy().to_crs(graph.graph["crs"])
    service_gdf = (
        points.copy() if service_gdf is None else service_gdf.to_crs(graph.graph["crs"]).copy()
    )

    # Get distances between points and services
    dg = DonGraphio(points.crs.to_epsg())
    dg.set_graph(graph)
    if check_nearest:
        service_gdf['dist'] = service_gdf.to_crs(graph.graph['crs']).apply(
            lambda row: city_points_gdf.to_crs(graph.graph['crs']).distance(row.geometry), axis=1)
        service_gdf = service_gdf.nsmallest(check_nearest, 'dist')
        # gpd.sjoin_nearest(service_gdf.to_crs(city_points_gdf.crs),city_points_gdf,distance_col='dist').sort_values('dist').head(10).copy().drop(columns=['index_right'])
    adj_mx = dg.get_adjacency_matrix(points, service_gdf, weight=weight, graph_type=graph_type)
    return adj_mx


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
        medians.append(median / 60)  # convert to hours
    points['to_service'] = medians
    return points

