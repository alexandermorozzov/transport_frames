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
import pandera as pa
from pandera.typing import Index
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString
from transport_frames.models.schema import BaseSchema
from loguru import logger
from tqdm import tqdm

def calculate_distances(from_gdf, to_gdf, graph, weight='length_meter', unit_div=1000):
            return round(get_adj_matrix_gdf_to_gdf(from_gdf, to_gdf, graph, weight=weight, dtype=np.float64).min(axis=1) / unit_div, 3)


def preprocess_service_accessibility(settlement_points,services, graph, local_crs):
    settlement_points_extended = settlement_points.copy()
    for service_name in ['railway_stations', 'fuel_stations', 'ports', 'local_aerodrome', 'international_aerodrome']:
        if services[service_name].empty:
            settlement_points_extended[f'{service_name}_accessibility_min'] = None
        else:
            settlement_points_extended[f'{service_name}_accessibility_min'] = get_adj_matrix_gdf_to_gdf(settlement_points.to_crs(local_crs),services[service_name].to_crs(local_crs),graph,'time_min').min(axis=1)
    return settlement_points_extended

def service_accessibility(settlements_points, districts, services, local_crs):
    
    res = gpd.sjoin(settlements_points, districts, how="left", predicate="within")
    grouped_median = res.groupby('index_right').median(numeric_only=True)
    districts_with_index = districts[['name', 'layer','status','geometry']].to_crs(local_crs).reset_index()
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
    
    column_order = [
    'name', 'layer','status', 'geometry',
    'number_of_railway_stations', 'railway_stations_accessibility_min',
    'number_of_fuel_stations', 'fuel_stations_accessibility_min',
    'number_of_ports', 'ports_accessibility_min',
    'number_of_local_aerodrome', 'local_aerodrome_accessibility_min',
    'number_of_international_aerodrome', 'international_aerodrome_accessibility_min',
    'number_of_bus_stops']
    result = result[column_order]       
    result = result.reset_index(drop=True)

    
    return result

def indicator_area(graph, areas, settlement_points, services, region_admin_center, local_crs, drive_adj_mx, inter_adg_mx):

    # Convert CRS of inputs
    region_admin_center = region_admin_center.to_crs(local_crs).copy()
    settlement_points = settlement_points.to_crs(local_crs).copy()
    services = {k: v.to_crs(local_crs).copy() if not v.empty else v for k, v in services.items()}
    areas = [area.to_crs(local_crs).copy() for area in areas]
    n, e = momepy.nx_to_gdf(graph)

    # Calculating shortest distances from settlements to services
    for service in ['railway_stations','fuel_stations','ports','local_aerodrome','international_aerodrome']:
        if not services[service].empty:
            settlement_points[f'{service}_accessbility_min'] = get_adj_matrix_gdf_to_gdf(settlement_points,
                                                                services[service],
                                                                graph,weight='time_min',dtype=np.float64).min(axis=1)
            
    results = []

    for area in tqdm(areas, desc="Processing areas"):
        logger.info("Calculating service accessibility")
        
        # Calculate service availability
        result = service_accessibility(settlement_points, area, services, local_crs)

        # Calculate drive connectivity
        settlement_points['connectivity_drive_min']=drive_adj_mx.median(axis=1)
        res = gpd.sjoin(settlement_points, area, how="left", predicate="within")
        grouped_median = res.groupby('index_right').median(numeric_only=True)   
        result['connectivity_drive_min'] = grouped_median['connectivity_drive_min']

        # Calculate intermodal connectivity
        settlement_points['connectivity_inter_min']=inter_adg_mx.median(axis=1)
        res = gpd.sjoin(settlement_points, area, how="left", predicate="within")
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
        for k in [1, 2]:
            logger.info(f"Calculating reg_{k} road lengths")
            result[f'reg{k}_length_km'] = 0.0 
            reg_roads = gpd.GeoDataFrame({'geometry': [e[e.reg==1].unary_union]}, crs=e.crs)
            for i, row in result.iterrows():
                row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                train_length = gpd.overlay(reg_roads, row_temp).geometry.length.sum()
                result.at[i, f'reg{k}_length_km'] = round(train_length / 1000, 3)

        # Road density calculation
        logger.info("Calculating road density")
        result['road_density_km/km2'] = density_roads(area, e)
        results.append(result)

    return results


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