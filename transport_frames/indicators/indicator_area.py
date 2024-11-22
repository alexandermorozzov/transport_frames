import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
import momepy
from dongraphio import DonGraphio
from transport_frames.utils.helper_funcs import prepare_graph
from transport_frames.indicators.utils import density_roads
from iduedu import get_adj_matrix_gdf_to_gdf
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from transport_frames.models.schema import BaseSchema
from loguru import logger
from tqdm import tqdm
import networkx as nx

def calculate_distances(from_gdf: gpd.GeoDataFrame, 
                        to_gdf: gpd.GeoDataFrame, 
                        graph: nx.MultiDiGraph, 
                        weight: str ='length_meter', 
                        unit_div: int=1000)-> gpd.GeoDataFrame:
            """
            Calculate the minimum distances between two GeoDataFrames using a graph.

            Parameters:
            from_gdf (gpd.GeoDataFrame): The GeoDataFrame containing origin points.
            to_gdf (gpd.GeoDataFrame): The GeoDataFrame containing destination points.
            graph (networkx.MultiDiGraph): The graph representation of the network.
            weight (str): The edge attribute to use for distance calculation. Defaults to 'length_meter'.
            unit_div (float): The divisor to convert distance to desired units. Defaults to 1000 (kilometers).

            Returns:
            gpd.GeoDataFrame: The gdf with minimum distance between the left gdf points 
            and nearest right gdf point in kilometers, rounded to three decimal places.
            """
            if to_gdf is None or to_gdf.empty:
                return None
            return round(get_adj_matrix_gdf_to_gdf(from_gdf, to_gdf, graph, weight=weight, dtype=np.float64).min(axis=1) / unit_div, 3)


def preprocess_service_accessibility(settlement_points: gpd.GeoDataFrame,
                                     services: dict, 
                                     graph: nx.MultiDiGraph, 
                                     local_crs: int) -> gpd.GeoDataFrame:
    """
    Preprocess the accessibility of services for settlement points.

    Parameters:
    settlement_points (gpd.GeoDataFrame): The GeoDataFrame containing settlement points.
    services (dict): A dictionary of GeoDataFrames containing various service points.
    graph (networkx.MultiDiGraph): The graph representation of the network.
    local_crs (int): The coordinate reference system to which the data should be transformed.

    Returns:
    gpd.GeoDataFrame: The extended GeoDataFrame with service accessibility metrics.
    """
    settlement_points_extended = settlement_points.copy()
    for service_name in ['railway_stations', 'fuel_stations', 'ports', 'local_aerodrome', 'international_aerodrome']:
        if services[service_name].empty:
            settlement_points_extended[f'{service_name}_accessibility_min'] = None
        else:
            settlement_points_extended[f'{service_name}_accessibility_min'] = get_adj_matrix_gdf_to_gdf(settlement_points.to_crs(local_crs),services[service_name].to_crs(local_crs),graph,'time_min').min(axis=1)
    return settlement_points_extended

def service_accessibility(preprocessed_settlements_points: gpd.GeoDataFrame, 
                          districts: gpd.GeoDataFrame, 
                          services: dict, 
                          local_crs: int) -> gpd.GeoDataFrame:
    """
    Calculate service accessibility for settlement points within districts.

    Parameters:
    preprocessed_settlements_points (gpd.GeoDataFrame): The GeoDataFrame containing preprocessed settlement points with accessibility data.
    districts (gpd.GeoDataFrame): The GeoDataFrame containing district boundaries.
    services (dict): A dictionary of GeoDataFrames containing various service points.
    local_crs (str): The coordinate reference system to which the data should be transformed.

    Returns:
    gpd.GeoDataFrame: The result GeoDataFrame containing accessibility metrics for each district.
    """    
    res = gpd.sjoin(preprocessed_settlements_points, districts, how="left", predicate="within")
    grouped_median = res.groupby('index_right').median(numeric_only=True)
    districts_with_index = districts[['name','geometry']].to_crs(local_crs).reset_index()
    result = pd.merge(districts_with_index, grouped_median, left_on='index', right_on='index_right')
    for service in ['railway_stations', 'fuel_stations', 'ports', 'local_aerodrome', 'international_aerodrome', 'bus_stops']:
        if services[service].empty:
            result[f'{service}_accessibility_min'] = None
            result[f'number_of_{service}'] = 0
        else:
            joined = gpd.sjoin(services[service], districts.to_crs(local_crs), how="left", predicate='within')
            service_counts = joined.groupby('index_right').size().reset_index(name=f'number_of_{service}')
            result = result.merge(service_counts, how='left', left_on='index', right_on='index_right').drop(columns=['index_right'])
    result = result.drop(columns=[col for col in result.columns if 'index_right' in col])
    numeric_cols = result.select_dtypes(include='number').columns
    result[numeric_cols] = result[numeric_cols].fillna(0)
    result[numeric_cols] = result[numeric_cols].astype(int)
    
    column_order = [
    'name', 'geometry',
    'number_of_railway_stations', 'railway_stations_accessibility_min',
    'number_of_fuel_stations', 'fuel_stations_accessibility_min',
    'number_of_ports', 'ports_accessibility_min',
    'number_of_local_aerodrome', 'local_aerodrome_accessibility_min',
    'number_of_international_aerodrome', 'international_aerodrome_accessibility_min',
    'number_of_bus_stops']
    result = result[column_order]       
    result = result.reset_index(drop=True)

    
    return result

def indicator_area(graph: nx.MultiDiGraph, 
                   areas: list[gpd.GeoDataFrame], 
                   preprocessed_settlement_points: gpd.GeoDataFrame, 
                   services: dict, 
                   local_crs: int, 
                   drive_adj_mx: pd.DataFrame, 
                   inter_adg_mx: pd.DataFrame,
                   region_admin_center: gpd.GeoDataFrame=None) -> list[gpd.GeoDataFrame]:
    """
    Calculate various accessibility and connectivity indicators for given areas.

    Parameters:
    graph (networkx.Graph): The graph representation of the network.
    areas (list): A list of GeoDataFrames representing areas of interest.
    settlement_points (gpd.GeoDataFrame): The GeoDataFrame containing settlement points.
    services (dict): A dictionary of GeoDataFrames containing various service points.
    region_admin_center (gpd.GeoDataFrame): The GeoDataFrame for the region's administrative center.
    local_crs (int): The coordinate reference system to which the data should be transformed.
    drive_adj_mx (pd.DataFrame): The driving adjacency matrix.
    inter_adg_mx (pd.DataFrame): The intermodal adjacency matrix.

    Returns:
    list: A list of GeoDataFrames containing the calculated indicators for each area.
    """
    class CentersSchema(BaseSchema):
        _geom_types = [Point]


    # Convert CRS of inputs
    if region_admin_center is not None:
        region_admin_center = CentersSchema(region_admin_center)
        region_admin_center = region_admin_center.to_crs(local_crs).copy()
    preprocessed_settlement_points = preprocessed_settlement_points.to_crs(local_crs).copy()
    services = {k: v.to_crs(local_crs).copy() if not v.empty else v for k, v in services.items()}
    areas = [area.to_crs(local_crs).copy() for area in areas]
    n, e = momepy.nx_to_gdf(graph)

    # Calculating shortest distances from settlements to services
    for service in ['railway_stations','fuel_stations','ports','local_aerodrome','international_aerodrome']:
        if not services[service].empty:
            preprocessed_settlement_points[f'{service}_accessbility_min'] = get_adj_matrix_gdf_to_gdf(preprocessed_settlement_points,
                                                                services[service],
                                                                graph,weight='time_min',dtype=np.float64).min(axis=1)
            
    results = []

    for area in tqdm(areas, desc="Processing areas\n"):
        logger.info("Calculating service accessibility")
        
        area = area.reset_index()
        # Calculate service availability
        result = service_accessibility(preprocessed_settlement_points, area, services, local_crs)

        # Calculate drive connectivity
        preprocessed_settlement_points['connectivity_drive_min']=drive_adj_mx.median(axis=1)
        res = gpd.sjoin(preprocessed_settlement_points, area, how="left", predicate="within")
        grouped_median = res.groupby('index_right').median(numeric_only=True)   
        result['connectivity_drive_min'] = grouped_median['connectivity_drive_min']

        # Calculate intermodal connectivity
        preprocessed_settlement_points['connectivity_inter_min']=inter_adg_mx.median(axis=1)
        res = gpd.sjoin(preprocessed_settlement_points, area, how="left", predicate="within")
        grouped_median = res.groupby('index_right').median(numeric_only=True)   
        result['connectivity_inter_min'] = grouped_median['connectivity_inter_min']


        # Calculate distances to region admin center and region 1 centers

        logger.info("Calculating distance to region admin center and federal roads")
        result['to_region_admin_center_km'] = calculate_distances(area, region_admin_center,graph)
        result['to_reg_1_km'] = calculate_distances(area, n[n.reg_1 == True],graph)

        # Train paths calculation
        logger.info("Calculating train path lengths")
        result['train_path_length_km'] = 0.0 
        if not services['train_paths'].empty:
            for i, row in result.iterrows():
                row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                train_length = gpd.overlay(services['train_paths'], row_temp).geometry.length.sum()
                result.at[i, 'train_path_length_km'] = round(train_length / 1000, 3)

        # Bus routes calculation
        logger.info("Calculating number of bus routes")
        result['number_of_bus_routes'] = 0 
        if not services['bus_routes'].empty:
            for i, row in result.iterrows():
                row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                bus_routes = set(gpd.overlay(services['bus_routes'], row_temp)['route'])
                result.at[i, 'number_of_bus_routes'] = len(bus_routes)

        # Aggregating road lengths
        for k in [1, 2, 3]:
            logger.info(f"Calculating reg_{k} road lengths")
            result[f'reg{k}_length_km'] = 0.0 
            reg_roads = gpd.GeoDataFrame({'geometry': [e[e.reg==k].unary_union]}, crs=e.crs) if not e[e.reg==k].empty else None
            if reg_roads is not None:
                for i, row in result.iterrows():
                    row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                    road_length = gpd.overlay(reg_roads, row_temp).geometry.length.sum()
                    result.at[i, f'reg{k}_length_km'] = round(road_length / 1000, 3)
            else:
                 result.at[i, f'reg{k}_length_km'] = 0.0

        # Road density calculation
        logger.info("Calculating road density")
        result['road_density_km/km2'] = density_roads(area, e)

        if 'territory_id' in area.columns:
            result['territory_id'] = area['territory_id']
            result=result.set_index('territory_id')
        results.append(result)

    return results


def get_intermodal(city_id: int, utm_crs: int):
    """
    This function extracts intermodal graph from osm

    Parameters:
    city_osm_id (int): Id of the territory/region.
    crs (int): The Coordinate Reference System to be used for the calculation.


    Returns:
    networkx.MultiDiGraph: The prepared intermodal graph with node names as integers.
    """
    dongrph = DonGraphio(city_crs=utm_crs)
    intermodal_graph = dongrph.get_intermodal_graph_from_osm(city_osm_id=city_id)
    intermodal_graph = prepare_graph(intermodal_graph)
    return intermodal_graph