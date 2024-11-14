import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
import momepy
from transport_frames.indicators.utils import density_roads
from iduedu import get_adj_matrix_gdf_to_gdf
import pandas as pd
import geopandas as gpd
import pandera as pa
from shapely.geometry import Polygon, MultiPolygon, Point
from transport_frames.models.schema import BaseSchema

# Schema for Territory
class TerritorySchema(BaseSchema):
    name: pa.typing.Series[str]
    _geom_types = [Polygon, MultiPolygon]

# Schema for Points (Region Admin Center, District Points, Settlement Points)
class PointsSchema(BaseSchema):
    _geom_types = [Point]

# Schema for Districts
class DistrictsSchema(BaseSchema):
    _geom_types = [Polygon, MultiPolygon]

def validate_crs(crs: int) -> None:
    """
    Validate the coordinate reference system (CRS).

    Parameters:
    crs (int): The CRS to validate.

    Raises:
    ValueError: If the CRS is not an integer.
    """
    if not isinstance(crs, int):
        raise ValueError("CRS must be an integer")

def validate_graph_crs(graph: nx.MultiDiGraph, local_crs: int) -> None:
    """
    Validate that the graph's CRS matches the local CRS.

    Parameters:
    graph (nx.MultiDiGraph): The graph to validate.
    local_crs (int): The local CRS to compare against.

    Raises:
    ValueError: If the graph's CRS does not match the local CRS.
    """
    if graph.graph['crs'] != local_crs:
        raise ValueError(f"Graph CRS ({graph.graph['crs']}) does not match local CRS ({local_crs})")

def validate_dataframes(graph: nx.MultiDiGraph, territory: gpd.GeoDataFrame, services: dict, 
                        region_admin_center: gpd.GeoDataFrame, district_points: gpd.GeoDataFrame, 
                        settlement_points: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, 
                        local_crs: int) -> None:
    """
    Validate the input dataframes for the indicator calculations.

    Parameters:
    graph (nx.MultiDiGraph): The graph representing the transport network.
    territory (gpd.GeoDataFrame): The GeoDataFrame representing the territory.
    services (dict): A dictionary of GeoDataFrames representing service points.
    region_admin_center (gpd.GeoDataFrame): GeoDataFrame of the region's administrative center.
    district_points (gpd.GeoDataFrame): GeoDataFrame of district points.
    settlement_points (gpd.GeoDataFrame): GeoDataFrame of settlement points.
    districts (gpd.GeoDataFrame): GeoDataFrame of districts.
    local_crs (int): The local coordinate reference system.

    Raises:
    ValueError: If any validation checks fail.
    """
    validate_crs(graph.graph['crs'])
    validate_graph_crs(graph, local_crs)
    TerritorySchema.validate(territory)
    if region_admin_center is not None:
        PointsSchema.validate(region_admin_center)
    if district_points is not None:
        PointsSchema.validate(district_points)
    if settlement_points is not None:
        PointsSchema.validate(settlement_points)
    if districts is not None:
        DistrictsSchema.validate(districts)
    validate_crs(local_crs)

def indicator_territory(graph: nx.MultiDiGraph, territory: gpd.GeoDataFrame, services: dict, 
                        region_admin_center: gpd.GeoDataFrame = None, district_points: gpd.GeoDataFrame = None, 
                        settlement_points: gpd.GeoDataFrame = None, districts: gpd.GeoDataFrame = None, 
                        local_crs: int = 3857) -> gpd.GeoDataFrame:
    """
    Calculate various accessibility indicators for a specified territory.

    Parameters:
    graph (nx.MultiDiGraph): The graph representing the transport network.
    territory (gpd.GeoDataFrame): The GeoDataFrame representing the territory.
    services (dict): A dictionary of GeoDataFrames for various service points.
    region_admin_center (gpd.GeoDataFrame): GeoDataFrame of the region's administrative center.
    district_points (gpd.GeoDataFrame): GeoDataFrame of district points.
    settlement_points (gpd.GeoDataFrame): GeoDataFrame of settlement points.
    districts (gpd.GeoDataFrame): GeoDataFrame of districts.
    local_crs (int): The local coordinate reference system.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing accessibility metrics for the territory.
    """
    # Validate CRS of the graph
    validate_crs(graph.graph['crs'])
    validate_dataframes(graph, territory, services, region_admin_center, district_points, settlement_points, districts, local_crs)

    # Standardize CRS for territory and all service points
    territory = territory.to_crs(local_crs).reset_index().copy()
    region_admin_center = region_admin_center.to_crs(local_crs).copy() if region_admin_center is not None else None
    district_points = district_points.to_crs(local_crs).copy() if district_points is not None else None
    settlement_points = settlement_points.to_crs(local_crs).copy() if settlement_points is not None else None
    districts = districts.to_crs(local_crs).copy() if districts is not None else None
    services = {k: v.to_crs(local_crs).copy() if not v.empty else v for k, v in services.items()}

    n, e = momepy.nx_to_gdf(graph)
    
    # Buffer around territory geometries to account for spatial queries
    territory['geometry'] = territory['geometry'].buffer(3000)
    
    result = territory[['name', 'geometry']].copy()

    # Helper function to calculate minimum distances between two GeoDataFrames
    def calculate_distances(from_gdf, to_gdf, weight='length_meter', unit_div=1000):
        if to_gdf is None:
            return None
        return round(get_adj_matrix_gdf_to_gdf(from_gdf, to_gdf, graph, weight=weight, dtype=np.float64).min(axis=1) / unit_div, 3)

    # Calculate distances to region admin center and region 1 centers
    result['to_region_admin_center_km'] = calculate_distances(territory, region_admin_center)
    if n[n.reg_1 == True].empty:
        result['to_reg_1_km'] = None
    else:
        result['to_reg_1_km'] = calculate_distances(territory, n[n.reg_1 == True])

    # Service calculations: Determine accessibility and number of each service type
    for service in ['fuel_stations', 'local_aerodrome', 'international_aerodrome', 'railway_stations', 'ports', 'bus_stops']:
            label = f'{service}_accessibility_min'   
            if services[service].empty:
                result[label] = None   
                result[f'number_of_{service}'] = 0   
            else:
                accessibility = calculate_distances(territory, services[service], weight='time_min', unit_div=1)
                result[label] = accessibility
                # Set accessibility to 0 if any service is inside the territory
                for i,row in result.iterrows():
                    row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                    if not gpd.overlay(services[service], row_temp).empty:
                        result.at[i, label] = 0.0
                # Count the number of services within the territory
                    result.at[i, f'number_of_{service}'] = len(gpd.overlay(services[service], row_temp))

    # Handle water objects and nature reserves
    for service in ['water_objects', 'nature_reserve']:
        
        result[f'number_of_{service}'] = 0  # Initialize the column with 0s
        if services[service].empty:
                result.at[i, f'{service}_accessibility_min'] = None
        else: 
            for i, row in result.iterrows():
                row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                result.at[i, f'number_of_{service}'] = len(gpd.overlay(services[service], row_temp))
                if result.at[i, f'number_of_{service}'] > 0 :
                    result.at[i, f'{service}_accessibility_min'] = 0.0
                else:
                    result.at[i, f'{service}_accessibility_min'] = round(gpd.sjoin_nearest(row_temp, services[service], how='inner', distance_col='dist')['dist'].min()/1000, 3)
    
    # Train paths and bus routes handling: Calculate total length and number of unique routes
    result['train_path_length_km'] = 0.0 
    if not services['train_paths'].empty:
        for i, row in result.iterrows():
            row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
            train_length = gpd.overlay(services['train_paths'], row_temp).geometry.length.sum()
            result.at[i, 'train_path_length_km'] = round(train_length / 1000, 3)  
            
    result['number_of_bus_routes'] = 0 
    if not services['bus_routes'].empty:
        for i, row in result.iterrows():
            row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
            bus_routes = set(gpd.overlay(services['bus_routes'], row_temp)['route'])
            result.at[i, 'number_of_bus_routes'] = len(bus_routes)

    result['to_nearest_district_center_km'] = None
    result['to_nearest_settlement_km'] = None
    # Filter districts and service points that intersect with the territory
    if districts is not None:
        filtered_regions_terr = districts[districts.intersects(territory.unary_union)]
        filtered_district_centers = district_points[district_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)] if district_points is not None else None
        filtered_settlement_centers = settlement_points[settlement_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)] if settlement_points is not None else None
    # Calculate distances to nearest district and settlement centers
        result['to_nearest_district_center_km'] = calculate_distances(territory, filtered_district_centers)
        result['to_nearest_settlement_km'] = calculate_distances(territory, filtered_settlement_centers)
    # Calculate road density (km of road per km²)
    result['road_density_km/km2'] = density_roads(territory, e, crs=local_crs)

    return result