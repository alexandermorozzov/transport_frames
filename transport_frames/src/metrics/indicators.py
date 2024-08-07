import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import shapely
from shapely import wkt
import numpy as np
from dongraphio import DonGraphio, GraphType
import matplotlib.pyplot as plt
import momepy
import transport_frames.src.graph_builder.graphbuilder as graphbuilder
from transport_frames.src.metrics.imports import AdjacencyCalculator



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


# протяженность дорог каждого типа
def calculate_length_sum_by_status(gdf: gpd.GeoDataFrame, crs=3857) -> gpd.GeoDataFrame:
    """
    This function calculates the length of roads (in km) in the geodataframe grouping by status.

    Parameters:
    gdf (gpd.GeoDataFrame): A GeoDataFrame containing the road geometries.


    Returns:
    gdf (gpd.GeoDataFrame): The calculated roads length in km for each reg_status.
    """
    gdf = gdf.to_crs(epsg=crs)
    gdf['reg'] = gdf['reg'].fillna(3)
    length_sum_by_status = gdf.groupby('reg').geometry.apply(lambda x: x.length.sum() / 1000)
    print(length_sum_by_status.reset_index())

    return length_sum_by_status.reset_index()


def get_intermodal(city_id, utm_crs):
    """
    This function extracts intermodal graph from osm

    Parameters:
    city_osm_id (int): Id of the territory/region.
    crs (int, optional): The Coordinate Reference System to be used for the calculation. Defaults to Web Mercator (EPSG:3857).


    Returns:
    networkx.MultiDiGraph: The prepared intermodal graph with node names as integers.
    """
    dongrph = DonGraphio(city_crs=utm_crs)
    intermodal_graph = dongrph.get_intermodal_graph_from_osm(city_osm_id=city_id)
    intermodal_graph = prepare_graph(intermodal_graph)
    return intermodal_graph


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
        merged.to_crs(points.crs).plot(column='to_service', cmap='RdYlGn_r', ax=ax, legend=True, vmax=vmax,
                                       markersize=4).set_axis_off()
    else:
        merged = pd.merge(polygons.reset_index(), res, left_on='index', right_on='index_right')
        merged.to_crs(points.crs).plot(column='to_service', cmap='RdYlGn_r', ax=ax, legend=True, vmax=vmax,
                                       markersize=4).set_axis_off()
        if service_gdf is not None:
            service_gdf = service_gdf.to_crs(polygons.crs)
            service_gdf.plot(ax=ax, markersize=7, color='white').set_axis_off()

    plt.title(title)
    plt.show()
    return merged


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
        medians.append(median / 60)  # convert to hours
    points['to_service'] = medians
    return points


def get_reg(graph, reg):
    """
    Extract nodes from edges with REG_STATUS==1 as a GeoDataFrame.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with geometries of REG_STATUS==1 nodes.
    """
    n = momepy.nx_to_gdf(graph, points=True, lines=False, spatial_weights=False)
    return n[n[f"reg_{reg}"] == True]


def aggregation(citygraph, points, polygons, service, weight='time_min', check_nearest=None):
    """
    This function calculates the median service availability for each area in a set of polygons,
    based on the nearest service node for each point in a set of points.

    Parameters:
    citygraph - Network graph representing the city.
    points - GeoDataFrame of settlement nodes.
    polygons - GeoDataFrame of polygons representing areas.
    service - gdf representing the service.
    weight - Edge attribute of the citygraph to use for path calculations. ('time_min'/'length_meter')

    Returns:
    GeoDataFrame with 'to_service' column representing service availability for each area.
    """
    points = find_nearest(points,
                          availability_matrix(citygraph, points, service, weight=weight, check_nearest=check_nearest))
    points = points.to_crs(polygons.crs)
    res = gpd.sjoin(points, polygons, how="left", predicate="within").groupby('index_right').median(['to_service'])
    merged = pd.merge(polygons.reset_index(), res, left_on='index', right_on='index_right')
    return merged


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


def get_connectivity(citygraph, points, polygons, graph_type='drive'):
    """
    This function calculates the median connectivity between areas for each area in a set of polygons,
    based on the median service node for each point in a set of points.

    Parameters:
    citygraph - Network graph representing the city.
    points - GeoDataFrame of settlement nodes.
    polygons - GeoDataFrame of polygons representing areas.
    graph_type - The type of graph for which connectivity is calculated: 'intermodal' or 'drive'.

    Returns:
    GeoDataFrame with service availability for each area.
    """
    if graph_type == 'intermodal':
        adj_mx = availability_matrix(citygraph, city_points_gdf=points, weight='time_min',
                                     graph_type=[GraphType.WALK, GraphType.PUBLIC_TRANSPORT])
    else:
        adj_mx = availability_matrix(citygraph, city_points_gdf=points, weight='time_min')
    points = find_median(city_points=points, adj_mx=adj_mx)

    points = points.to_crs(polygons.crs)
    res = gpd.sjoin(points, polygons, how="left", predicate="within").groupby('index_right').median(['to_service'])
    merged = pd.merge(polygons.reset_index(), res, left_on='index', right_on='index_right')
    return merged


# def indicator_area(citygraph,area_polygons,points,region_capital,
#                    fuel, train_stops, international_aero, aero, ports, train_paths, bus_stops, inter = None, crs=3857):
#     """
#     This function calculates the various indicators for a specific place based on its characteristics.

#     Parameters:
#     citygraph - Network graph representing the city.
#     polygon_of_the_region - Polygon of the whole region border.
#     area_polygons - GeoDataFrame of polygons representing areas (regions/districts).
#     points - GeoDataFrame of points of all settlements.
#     inter - Intermodal graph of the territory.
#     fed_center - A special gdf representing the primary federal center.
#     center - A special gdf representing the center of the region.
#     fuel - The gdf representing fuel stations.
#     train_stops - The gdf representing train stops.
#     international_aero - The gdf representing international airports.
#     aero - The gdf representing local airports.
#     ports - The gdf representing ports.
#     train_paths - The gdf representiong train edges
#     bus_stops - The gdf representiong bus stop points

#     Returns:
#     Dictionary of calculated indicators.
#     """
#     d = {}
#     n,e = momepy.nx_to_gdf(graphbuilder.prepare_graph(citygraph))
#     area_polygons2 = area_polygons.copy()
#     area_polygons2['density'] = area_polygons2.apply(lambda row: density_roads(gpd.GeoDataFrame([row], geometry='geometry', crs=area_polygons2.crs), e, crs=crs), axis=1)
#     d['aggregated_density'] = area_polygons2
#     d['road_length_gdf'] = aggregate_road_lengths(e,area_polygons,crs,reg=True)
#     d['connectivity'] = get_connectivity(citygraph,points,area_polygons)
#     if inter is not None:
#         d['connectivity_public_transport'] = get_connectivity(inter,points,area_polygons,graph_type='intermodal')

#         ei = momepy.nx_to_gdf(inter)[1]
#         bus_routes = ei[ei['type']=='bus']
#         d['number_of_bus_routes'] = aggregate_routes_by_polygon(bus_routes,area_polygons)

#     # connnectivity
#     d['to_fed_roads'] = aggregation(citygraph,points,area_polygons,service=get_reg(citygraph,1),weight='length_meter')
#     d['to_fed_roads']['to_service'] = d['to_fed_roads']['to_service']/1000

#     if not region_capital.empty:
#         d['to_region_admin_center'] = aggregation(citygraph,points,area_polygons,service=region_capital,weight='length_meter')
#         d['to_region_admin_center']['to_service'] = d['to_region_admin_center']['to_service']/1000

#     # service availability
#     if not fuel.empty:
#         d['azs_availability'] = aggregation(citygraph,points,area_polygons,service=fuel,weight='time_min')
#         d['azs_availability']['to_service'] = d['azs_availability']['to_service']
#         d['number_of_fuel_stations'] = aggregate_services_by_polygon(fuel,area_polygons)

#     if not train_stops.empty:
#         d['train_stops_availability'] = aggregation(citygraph,points,area_polygons,service=train_stops,weight='time_min')
#         d['train_stops_availability']['to_service'] = d['train_stops_availability']['to_service']
#         d['number_of_train_stops'] = aggregate_services_by_polygon(train_stops,area_polygons)

#     if not international_aero.empty:
#         d['international_aero_availability'] = aggregation(citygraph,points,area_polygons,service=international_aero,weight='time_min')
#         d['international_aero_availability']['to_service'] = d['international_aero_availability']['to_service']
#         d['number_of_international_aero'] = aggregate_services_by_polygon(international_aero,area_polygons)

#     if not aero.empty:
#         d['local_aero_availability'] = aggregation(citygraph,points,area_polygons,service=aero,weight='time_min')
#         d['local_aero_availability']['to_service'] = d['local_aero_availability']['to_service']
#         d['number_of_local_aero'] = aggregate_services_by_polygon(aero,area_polygons)

#     if not ports.empty:
#         d['port_availability'] = aggregation(citygraph,points,area_polygons,service=ports,weight='time_min')
#         d['port_availability']['to_service'] = d['port_availability']['to_service']
#         d['number_of_ports'] = aggregate_services_by_polygon(ports,area_polygons)

#     if not bus_stops.empty:
#         d['number_of_bus_stops'] = aggregate_services_by_polygon(bus_stops,area_polygons)

#     if not train_paths.empty:
#         d['train_paths_length'] = aggregate_road_lengths(train_paths,area_polygons,crs)
#     return d


# def indicator_territory(citygraph, territory, regions_gdf, districts_gdf, region_capital, region_centers,
#                         district_centers, settlement_centers,
#                         fuel, train_stops, international_aero, aero, ports, water_objects, oopt, train_paths,
#                         inter=None, bus_stops=None, bus_routes=None, crs=32636):
#     """
#     This function calculates the various indicators for a specific territory based on its characteristics.

#     Parameters:
#     citygraph - Network graph representing the city.
#     territory - GeoDataFrame of the territory for which indicators are calculated.
#     regions_gdf - GeoDataFrame of regions.
#     districts_gdf - GeoDataFrame of districts.
#     region_centers - A GeoDataFrame representing the centers of regions.
#     district_centers - A GeoDataFrame representing the centers of districts.
#     settlement_centers - A GeoDataFrame representing the centres of settlements.
#     inter - Intermodal graph of the region/territory
#     fuel - The gdf representing fuel stations.
#     train_stops - The gdf representing train stops.
#     international_aero - The gdf representing international airports.
#     aero - The gdf representing other airports.
#     ports - The gdf representing ports.
#     water_objects - The gdf representing water objects.
#     oopt - The gdf representing specially protected natural territories.
#     crs - Coordinate Reference System (default is EPSG:32636).

#     Returns:
#     Dictionary of calculated indicators.
#     """
#     d = dict()
#     n, e = momepy.nx_to_gdf(prepare_graph(citygraph))
#     region_centers = region_centers.copy()
#     territory = territory.to_crs(crs).copy()
#     terr_centroid = gpd.GeoDataFrame({'geometry': shapely.centroid(territory.geometry)}, crs=territory.crs).to_crs(
#         territory.crs)
#     territory['geometry'] = territory['geometry'].buffer(3000)

#     d['to_fed_roads'] = find_nearest(terr_centroid, availability_matrix(citygraph, terr_centroid,
#                                                                         get_reg(citygraph, 1).to_crs(territory.crs),
#                                                                         weight='length_meter', check_nearest=100))
#     d['to_fed_roads']['to_service'] = d['to_fed_roads']['to_service'] / 1000

#     d['to_region_admin_center'] = find_nearest(terr_centroid, availability_matrix(citygraph, terr_centroid,
#                                                                                   region_capital.to_crs(territory.crs),
#                                                                                   weight='length_meter'))
#     d['to_region_admin_center']['to_service'] = d['to_region_admin_center']['to_service'] / 1000
#     regions_gdf.to_crs(territory.crs, inplace=True)
#     region_centers.to_crs(territory.crs, inplace=True)
#     filtered_regions_terr = regions_gdf[regions_gdf.intersects(territory.unary_union)]
#     filtered_region_centers = region_centers[region_centers.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
#     adj_region_centers = availability_matrix(citygraph, terr_centroid, filtered_region_centers, weight='length_meter')
#     filtered_region_centers['to_service'] = adj_region_centers.transpose() / 1000
#     d['connectivity_region_center'] = filtered_region_centers

#     districts_gdf.to_crs(territory.crs, inplace=True)
#     district_centers.to_crs(territory.crs, inplace=True)
#     filtered_districts_terr = districts_gdf[districts_gdf.intersects(territory.unary_union)]
#     filtered_district_centers = district_centers[
#         district_centers.buffer(0.1).intersects(filtered_districts_terr.unary_union)]
#     adj_district_centers = availability_matrix(citygraph, terr_centroid, filtered_district_centers,
#                                                weight='length_meter')
#     filtered_district_centers['to_service'] = adj_district_centers.transpose() / 1000
#     d['connectivity_district_center'] = filtered_district_centers

#     adj_np = availability_matrix(citygraph, terr_centroid, settlement_centers.to_crs(territory.crs))
#     nearest_np = find_nearest(terr_centroid, adj_np)
#     d['connectivity_settlement'] = nearest_np
#     if inter is not None and (bus_routes is None or bus_stops is None):
#         ni, ei = momepy.nx_to_gdf(inter)
#         if bus_stops is None:
#             bus_stops = ni[(ni['desc'] == 'bus') & (ni['stop'] == 'True')]
#         else:
#             bus_routes = ei[ei['type'] == 'bus']

#     d['density'] = density_roads(territory, e.to_crs(territory.crs), crs=crs)

#     if not bus_routes.empty:
#         d['number_of_bus_routes'] = len(set(gpd.overlay(bus_routes.to_crs(crs), territory.to_crs(crs))['desc']))

#     if not bus_stops.empty:
#         d['number_of_bus_stops'] = len(gpd.overlay(bus_stops.to_crs(crs), territory.to_crs(crs)))

#     if not train_paths.empty:
#         d['train_paths_length'] = gpd.overlay(train_paths.to_crs(crs), territory.to_crs(crs)).geometry.length.sum()

#     if not fuel.empty:
#         d['number_of_fuel_stations'] = len(gpd.overlay(fuel.to_crs(territory.crs), territory))
#         d['azs_availability'] = find_nearest(terr_centroid,
#                                              availability_matrix(citygraph, terr_centroid, fuel.to_crs(territory.crs)))
#         if d['number_of_fuel_stations'] != 0:
#             d['azs_availability']['to_service'] = 0

#     if not international_aero.empty:
#         d['number_of_international_aero'] = len(gpd.overlay(international_aero.to_crs(territory.crs), territory))
#         d['international_aero_availability'] = find_nearest(terr_centroid, availability_matrix(citygraph, terr_centroid,
#                                                                                                international_aero.to_crs(
#                                                                                                    territory.crs)))
#         if d['number_of_international_aero'] != 0:
#             d['international_aero_availability']['to_service'] = 0

#     if not aero.empty:
#         d['number_of_local_aero'] = len(gpd.overlay(aero.to_crs(territory.crs), territory))
#         d['local_aero_availability'] = find_nearest(terr_centroid, availability_matrix(citygraph, terr_centroid,
#                                                                                        aero.to_crs(territory.crs)))
#         if d['number_of_local_aero'] != 0:
#             d['local_aero_availability']['to_service'] = 0

#     if not train_stops.empty:
#         d['number_of_train_stops'] = len(gpd.overlay(train_stops.to_crs(territory.crs), territory))
#         d['train_stops_availability'] = find_nearest(terr_centroid, availability_matrix(citygraph, terr_centroid,
#                                                                                         train_stops.to_crs(
#                                                                                             territory.crs)))
#         if d['number_of_train_stops'] != 0:
#             d['train_stops_availability']['to_service'] = 0

#     if not ports.empty:
#         d['number_of_ports'] = len(gpd.overlay(ports.to_crs(territory.crs), territory))
#         d['ports_availability'] = find_nearest(terr_centroid, availability_matrix(citygraph, terr_centroid,
#                                                                                   ports.to_crs(territory.crs)))
#         if d['number_of_ports'] != 0:
#             d['ports_availability']['to_service'] = 0

#     if not oopt.empty:
#         oopt = oopt.to_crs(crs).copy()
#         d['oopt_availability'] = find_nearest(terr_centroid,
#                                               availability_matrix(citygraph, terr_centroid, oopt.to_crs(territory.crs)))
#         if not gpd.overlay(oopt, territory, how='intersection').empty:
#             d['oopt_availability']['to_service'] = 0

#     if not water_objects.empty:
#         d['number_of_water_objects'] = len(gpd.overlay(water_objects.to_crs(territory.crs), territory))
#         water_objects = water_objects.to_crs(crs).copy()
#         a = gpd.sjoin_nearest(terr_centroid.to_crs(crs), water_objects.to_crs(crs), how='inner', distance_col='dist')[
#                 'dist'].min() / 60
#         d['water_objects_availability'] = gpd.GeoDataFrame({'geometry': shapely.centroid(territory.geometry)},
#                                                            crs=territory.crs).to_crs(territory.crs)
#         d['water_objects_availability']['to_service'] = a
#         if not gpd.overlay(water_objects, territory, how='intersection').empty:
#             d['water_objects_availability']['to_service'] = 0

#     return d



import networkx as nx
import tqdm
import geopandas as gpd
from shapely.geometry import Point


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

    node_distances = {
        node: {
            'name': citygraph.nodes[node].get('name'),
            'x': citygraph.nodes[node].get('x'),
            'y': citygraph.nodes[node].get('y'),
            **{service: None for service in services}
        }
        for node in citygraph.nodes if citygraph.nodes[node].get('points') == 1
    }
    if territory:
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


def new_connectivity(graph, city_nodes, local_crs=3826, inter=False):
    citygraph_copy = graph.copy()
    citygraph_copy.add_nodes_from(citygraph_copy.nodes(data=True))
    citygraph_copy.add_edges_from(citygraph_copy.edges(data=True, keys=True))
    citygraph_copy.graph = graph.graph
    n = city_nodes
    p = n[n['points'] == 1].to_crs(local_crs).copy()
    gdf_buffers = p.to_crs(local_crs).copy()
    gdf_buffers['geometry'] = gdf_buffers['geometry'].buffer(100)

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


def indicator_area(citygraph, inter, services, polygonsList, local_crs):
    a = get_accessibility(citygraph)
    at = gpd.GeoDataFrame(a)
    at.crs = local_crs
    n, e = momepy.nx_to_gdf(citygraph)

    # getting uds connectivity
    adj_uds = new_connectivity(citygraph, n, local_crs=local_crs, inter=False)
    # getting intermodal connectivity
    adj_inter = new_connectivity(inter, n, local_crs=local_crs, inter=True)

    merged_gdfs = []
    # aggregating it by polygons
    for polygons in polygonsList:
        res = gpd.sjoin(at, polygons.to_crs(local_crs), how="left", predicate="within").groupby('index_right').median(numeric_only=True)
        for column in ['fuel', 'railway_stops', 'local_aero', 'international_aero', 'ports', 'capital', 'reg_1']:
            if column not in res.columns:
                res[column] = None
        res = res[
            ['fuel', 'railway_stops', 'local_aero', 'international_aero', 'ports', 'capital', 'reg_1']].reset_index()
        merged = pd.merge(polygons[['name','status','layer', 'geometry']].to_crs(local_crs).reset_index(), res, left_on='index',
                          right_on='index_right')

        res = gpd.sjoin(adj_uds, polygons.to_crs(local_crs), how="left", predicate="within").groupby(
            'index_right').median(numeric_only=True)
        res = res[['to_service']].reset_index()
        merged['connectivity'] = res['to_service']

        res = gpd.sjoin(adj_inter, polygons.to_crs(local_crs), how="left", predicate="within").groupby(
            'index_right').median(numeric_only=True)
        res = res[['to_service']].reset_index()
        merged['connectivity_public_transport'] = res['to_service']

        merged['density'] = merged.apply(
            lambda row: density_roads(gpd.GeoDataFrame([row], geometry='geometry', crs=merged.crs), e, crs=local_crs),
            axis=1)
        merged['railway_length'] = aggregate_road_lengths(services['train_paths'], polygons, local_crs)[
            'total_length_km']

        for service in ['fuel', 'railway_stops', 'local_aero', 'international_aero', 'ports', 'bus_stops']:
            if services[service].size != 0:
                merged[f'{service}_number'] = \
                aggregate_services_by_polygon(services[service].to_crs(local_crs), polygons.to_crs(local_crs))[
                    'service_count']
            else:
                merged[f'{service}_number'] = 0
        temp = aggregate_road_lengths(e, polygons, local_crs, reg=True)
        merged['reg1_length'] = temp['reg1_length']
        merged['reg2_length'] = temp['reg2_length']
        merged['reg3_length'] = temp['reg3_length']

        merged['number_of_bus_routes'] = aggregate_routes_by_polygon(services['bus_routes'], polygons)[
            'number_of_routes']
        merged = merged.rename(columns={
            'density': 'road_density',
            'capital': 'to_region_admin_center',
            'fuel': 'fuel_stations_accessibility',
            'fuel_number': 'number_of_fuel_stations',
            'railway_stops': 'train_stops_accessibility',
            'railway_stops_number': 'number_of_train_stops',
            'international_aero': 'international_aero_accessibility',
            'international_aero_number': 'number_of_international_aero',
            'local_aero': 'local_aero_accessibility',
            'local_aero_number': 'number_of_local_aero',
            'bus_stops_number': 'number_of_bus_stops',
            'railway_length': 'train_paths_length',
            'ports': 'ports_accessibility',
            'ports_number': 'number_of_ports',
            'reg_1': 'to_reg1'

            # add more columns as needed
        })
        cols_to_format = [col for col in merged.columns if col not in ['index', 'geometry', 'name','layer','status']]
        merged[cols_to_format] = merged[cols_to_format].applymap(
            lambda x: f"{x:.3f}" if (pd.notnull(x) and x != 'inf') else None)
        merged.replace(['inf'], None, inplace=True)
        merged.drop(columns='index_right', inplace=True)

        merged_gdfs.append(merged.to_crs(4326))

    return merged_gdfs



def dijkstra_nearest_centers(G):
    start_node = momepy.nx_to_gdf(G)[0][momepy.nx_to_gdf(G)[0].neud_center==1].nodeID.reset_index()['nodeID'][0]
    
    # Find nearest settlement center
    settlement_center_nodes = [node for node, data in G.nodes(data=True) if data.get('nearest_settlement_centers') == 1]
    distances = nx.single_source_dijkstra_path_length(G, start_node, weight='length_meter')
    settlement_center_distances = {node: dist for node, dist in distances.items() if node in settlement_center_nodes}
    if settlement_center_distances:
        nearest_settlement_node = min(settlement_center_distances, key=settlement_center_distances.get)
        min_settlement_distance = settlement_center_distances[nearest_settlement_node]
    else:
        nearest_settlement_node, min_settlement_distance = None, None
    
    # Find nearest region center
    region_center_nodes = [node for node, data in G.nodes(data=True) if data.get('nearest_region_centers') == 1]
    region_center_distances = {node: dist for node, dist in distances.items() if node in region_center_nodes}
    if region_center_distances:
        nearest_region_node = min(region_center_distances, key=region_center_distances.get)
        min_region_distance = region_center_distances[nearest_region_node]
    else:
        nearest_region_node, min_region_distance = None, None
    
    return round(min_settlement_distance,3),round(min_region_distance,3)

def indicator_territory(G,territory,local_crs,regions_gdf,points,region_centers,services,G_nodes,G_edges):

    neud_center = territory.geometry.representative_point()
    neud_center = gpd.GeoDataFrame([{'geometry': neud_center.iloc[0]}], crs=territory.crs).to_crs(local_crs)
    merged = get_accessibility(G,territory=True)
    merged.drop(columns=['geometry','x','y'],inplace=True)
    merged['water_objects'] = round(gpd.sjoin_nearest(neud_center.to_crs(local_crs), services['water_objects'].to_crs(local_crs), how='inner', distance_col='dist')[
        'dist'].min()/1000/60,3)
    merged['oopt'] = round(gpd.sjoin_nearest(neud_center.to_crs(local_crs), services['oopt'].to_crs(local_crs), how='inner', distance_col='dist')[
        'dist'].min()/1000/60,3)

    # number of services
    for k,v in services.items():
        v['type_service'] = k 
    point_geoms = pd.concat(
            [services[service].to_crs(local_crs)
            for service in ['railway_stops', 'fuel', 'ports', 'local_aero','bus_stops'] if not services[service].empty]
        ).reset_index(drop=True)
    poly_geoms = pd.concat(
            [services[service].to_crs(local_crs)
            for service in ['water_objects','oopt'] if not services[service].empty]
        ).reset_index(drop=True)
    line_geoms = pd.concat(
            [services[service].to_crs(local_crs)
            for service in ['train_paths','bus_routes'] if not services[service].empty]
        ).reset_index(drop=True)
    
    cut_poly  = gpd.overlay(poly_geoms.to_crs(territory.crs), territory)
    cut_points  = gpd.overlay(point_geoms.to_crs(territory.crs), territory)
    cut_lines = gpd.overlay(line_geoms.to_crs(territory.crs), territory)

    for service in ['railway_stops', 'fuel', 'ports', 'local_aero','international_aero','bus_stops']:
        if len(services[service]) == 0 or not (cut_points['type_service'] == service).any():
            merged[f'number_of_{service}'] = 0
        else:
            merged[f'number_of_{service}'] = len(cut_points[cut_points['type_service']==service])
            if service != 'bus_stops':
                merged[service]=0
    
    for service in ['water_objects','oopt']:
        if len(services[service]) == 0 or not (cut_poly['type_service'] == service).any():
            merged[f'number_of_{service}'] = 0
        else:
            merged[f'number_of_{service}'] = len(cut_poly[cut_poly['type_service']==service])


    merged['density'] = density_roads(territory, G_edges.to_crs(territory.crs), crs=local_crs)
    merged['train_path_length'] = round(cut_lines[cut_lines['type_service']=='train_paths'].to_crs(local_crs).geometry.length.sum()/1000, 3)
    merged['number_of_bus_routes'] = len(set(cut_lines[cut_lines['type_service']=='bus_routes'].to_crs(local_crs)))

    #find the nearest settlements and region centers
    regions_gdf.to_crs(territory.crs, inplace=True)
    region_centers.to_crs(territory.crs, inplace=True)
    territory.to_crs(territory.crs, inplace=True)
    points.to_crs(territory.crs, inplace=True)

    filtered_regions_terr = regions_gdf[regions_gdf.intersects(territory.unary_union)]
    filtered_region_centers = region_centers[region_centers.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
    filtered_settlement_centers = points[points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
    G = assign_services_names_to_nodes({'nearest_region_centers':filtered_region_centers},G_nodes,G)
    G = assign_services_names_to_nodes({'nearest_settlement_centers':filtered_settlement_centers},G_nodes,G)
    merged['to_settlement_center'],merged['to_region_center'] = dijkstra_nearest_centers(G)

    merged = merged.rename(columns={
            'capital': 'to_region_admin_center',
            'reg_1': 'to_reg1',
            'fuel': 'fuel_stations_accessibility',
            'railway_stops': 'train_stops_accessibility',
            'local_aero': 'local_aero_accessibility',
            'international_aero': 'international_aero_accessibility',
            'ports': 'ports_accessibility',
            'water_objects': 'water_objects_accessibility',
            'oopt': 'oopt_accessibility',
            'number_of_railway_stops': 'number_of_train_stops',
            'number_of_fuel': 'number_of_fuel_stations',
            'to_settlement_center': 'to_nearest_settlement_center',
            'to_region_center': 'to_nearest_mo_center',

            # add more columns as needed
        })
    merged['geometry'] = territory.reset_index().geometry[0]
    merged.crs = territory.crs
    merged.to_crs(4326,inplace=True)

    return merged