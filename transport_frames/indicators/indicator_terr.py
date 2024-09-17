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
from pandera.typing import Index
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString
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

def validate_crs(crs):
    if not isinstance(crs, int):
        raise ValueError("CRS must be an integer")

def validate_graph_crs(graph, local_crs):
    if graph.graph['crs'] != local_crs:
        raise ValueError(f"Graph CRS ({graph.graph['crs']}) does not match local CRS ({local_crs})")

def validate_dataframes(graph, territory, services, region_admin_center, district_points, settlement_points, districts, local_crs):
    validate_crs(graph.graph['crs'])
    validate_graph_crs(graph, local_crs)
    TerritorySchema.validate(territory)
    PointsSchema.validate(region_admin_center)
    PointsSchema.validate(district_points)
    PointsSchema.validate(settlement_points)
    DistrictsSchema.validate(districts)
    validate_crs(local_crs)

def indicator_territory(graph, territory, services, region_admin_center, district_points, settlement_points, districts, local_crs):
    # Validate CRS of the graph
    validate_crs(graph.graph['crs'])
    validate_dataframes(graph, territory, services, region_admin_center, district_points, settlement_points, districts, local_crs)

    # Standardize CRS for territory and all service points
    territory = territory.to_crs(local_crs).reset_index().copy()
    region_admin_center = region_admin_center.to_crs(local_crs).copy()
    district_points = district_points.to_crs(local_crs).copy()
    settlement_points = settlement_points.to_crs(local_crs).copy()
    districts = districts.to_crs(local_crs).copy()
    services = {k: v.to_crs(local_crs).copy() if not v.empty else v for k, v in services.items()}

    n, e = momepy.nx_to_gdf(graph)
    
    # Buffer around territory geometries to account for spatial queries
    territory['geometry'] = territory['geometry'].buffer(3000)
    
    result = territory[['name', 'geometry']].copy()
    
    # Helper function to calculate minimum distances between two GeoDataFrames
    def calculate_distances(from_gdf, to_gdf, weight='length_meter', unit_div=1000):
        return round(get_adj_matrix_gdf_to_gdf(from_gdf, to_gdf, graph, weight=weight, dtype=np.float64).min(axis=1) / unit_div, 3)

    # Calculate distances to region admin center and region 1 centers
    result['to_region_admin_center_km'] = calculate_distances(territory, region_admin_center)
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


    # Filter districts and service points that intersect with the territory
    filtered_regions_terr = districts[districts.intersects(territory.unary_union)]
    filtered_district_centers = district_points[district_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
    filtered_settlement_centers = settlement_points[settlement_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
    
    # Calculate distances to nearest district and settlement centers
    result['to_nearest_district_center_km'] = calculate_distances(territory, filtered_district_centers)
    result['to_nearest_settlement_km'] = calculate_distances(territory, filtered_settlement_centers)
    
    # Calculate road density (km of road per km²)
    result['road_density_km/km2'] = density_roads(territory, e, crs=local_crs)

    return result


# def indicator_territory(G,territory,local_crs,regions_gdf,points,region_centers,services,G_nodes,G_edges):
#     territory = territory.copy()
#     territory['geometry'] = territory['geometry'].buffer(3000)
#     neud_center = territory.geometry.representative_point()
#     neud_center = gpd.GeoDataFrame([{'geometry': neud_center.iloc[0]}], crs=territory.crs).to_crs(local_crs)
#     merged = get_accessibility(G,territory=True)
#     merged.drop(columns=['geometry','x','y'],inplace=True)
#     merged['water_objects'] = round(gpd.sjoin_nearest(neud_center.to_crs(local_crs), services['water_objects'].to_crs(local_crs), how='inner', distance_col='dist')[
#         'dist'].min()/1000/60, 3)
#     merged['oopt'] = round(gpd.sjoin_nearest(neud_center.to_crs(local_crs), services['oopt'].to_crs(local_crs), how='inner', distance_col='dist')[
#         'dist'].min()/1000/60, 3)

#     # number of services
#     for k,v in services.items():
#         v['type_service'] = k 
#     point_geoms = pd.concat(
#             [services[service].to_crs(local_crs)
#             for service in ['railway_stops', 'fuel', 'ports', 'local_aero','bus_stops'] if not services[service].empty]
#         ).reset_index(drop=True)
#     poly_geoms = pd.concat(
#             [services[service].to_crs(local_crs)
#             for service in ['water_objects','oopt'] if not services[service].empty]
#         ).reset_index(drop=True)
#     line_geoms = pd.concat(
#             [services[service].to_crs(local_crs)
#             for service in ['train_paths','bus_routes'] if not services[service].empty]
#         ).reset_index(drop=True)
    
#     cut_poly  = gpd.overlay(poly_geoms.to_crs(territory.crs), territory)
#     cut_points  = gpd.overlay(point_geoms.to_crs(territory.crs), territory)
#     cut_lines = gpd.overlay(line_geoms.to_crs(territory.crs), territory)

#     for service in ['railway_stops', 'fuel', 'ports', 'local_aero','international_aero','bus_stops']:
#         if len(services[service]) == 0 or not (cut_points['type_service'] == service).any():
#             merged[f'number_of_{service}'] = 0
#         else:
#             merged[f'number_of_{service}'] = len(cut_points[cut_points['type_service']==service])
#             if service != 'bus_stops':
#                 merged[service]=0
    
#     for service in ['water_objects','oopt']:
#         if len(services[service]) == 0 or not (cut_poly['type_service'] == service).any():
#             merged[f'number_of_{service}'] = 0
#         else:
#             merged[f'number_of_{service}'] = len(cut_poly[cut_poly['type_service']==service])


#     merged['density'] = density_roads(territory, G_edges.to_crs(territory.crs), crs=local_crs)
#     merged['train_path_length'] = round(cut_lines[cut_lines['type_service']=='train_paths'].to_crs(local_crs).geometry.length.sum()/1000, 3)
#     merged['number_of_bus_routes'] = len(set(cut_lines[cut_lines['type_service']=='bus_routes'].to_crs(local_crs)['route']))

#     #find the nearest settlements and region centers
#     regions_gdf.to_crs(territory.crs, inplace=True)
#     region_centers.to_crs(territory.crs, inplace=True)
#     territory.to_crs(territory.crs, inplace=True)
#     points.to_crs(territory.crs, inplace=True)

#     filtered_regions_terr = regions_gdf[regions_gdf.intersects(territory.unary_union)]
#     filtered_region_centers = region_centers[region_centers.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
#     filtered_settlement_centers = points[points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
#     G = assign_services_names_to_nodes({'nearest_region_centers':filtered_region_centers},G_nodes,G)
#     G = assign_services_names_to_nodes({'nearest_settlement_centers':filtered_settlement_centers},G_nodes,G)
#     merged['to_settlement_center'],merged['to_region_center'] = dijkstra_nearest_centers(G)

#     merged = merged.rename(columns={
#             'capital': 'to_region_admin_center',
#             'reg_1': 'to_reg1',
#             'fuel': 'fuel_stations_accessibility',
#             'railway_stops': 'train_stops_accessibility',
#             'local_aero': 'local_aero_accessibility',
#             'international_aero': 'international_aero_accessibility',
#             'ports': 'ports_accessibility',
#             'water_objects': 'water_objects_accessibility',
#             'oopt': 'oopt_accessibility',
#             'number_of_railway_stops': 'number_of_train_stops',
#             'number_of_fuel': 'number_of_fuel_stations',
#             'to_settlement_center': 'to_nearest_settlement_center',
#             'to_region_center': 'to_nearest_mo_center',
#         })
#     merged.reset_index(inplace=True,drop=True)
#     merged['name'] = territory['name']
#     merged['geometry'] = territory.reset_index().geometry[0]
#     merged.crs = territory.crs
#     merged.to_crs(local_crs,inplace=True)

#     return merged





# def dijkstra_nearest_centers(G):
#     start_node = momepy.nx_to_gdf(G)[0][momepy.nx_to_gdf(G)[0].neud_center==1].nodeID.reset_index()['nodeID'][0]
    
#     # Find nearest settlement center
#     settlement_center_nodes = [node for node, data in G.nodes(data=True) if data.get('nearest_settlement_centers') == 1]
#     distances = nx.single_source_dijkstra_path_length(G, start_node, weight='length_meter')
#     settlement_center_distances = {node: dist for node, dist in distances.items() if node in settlement_center_nodes}
#     if settlement_center_distances:
#         nearest_settlement_node = min(settlement_center_distances, key=settlement_center_distances.get)
#         min_settlement_distance = settlement_center_distances[nearest_settlement_node]
#     else:
#         nearest_settlement_node, min_settlement_distance = None, None
    
#     # Find nearest region center
#     region_center_nodes = [node for node, data in G.nodes(data=True) if data.get('nearest_region_centers') == 1]
#     region_center_distances = {node: dist for node, dist in distances.items() if node in region_center_nodes}
#     if region_center_distances:
#         nearest_region_node = min(region_center_distances, key=region_center_distances.get)
#         min_region_distance = region_center_distances[nearest_region_node]
#     else:
#         nearest_region_node, min_region_distance = None, None
    
#     return round(min_settlement_distance,3),round(min_region_distance,3)

#validation
#min or hours
