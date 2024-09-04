import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
import momepy
from transport_frames.indicators.utils import get_accessibility, density_roads, assign_services_names_to_nodes


def indicator_territory(G,territory,local_crs,regions_gdf,points,region_centers,services,G_nodes,G_edges):

    neud_center = territory.geometry.representative_point()
    neud_center = gpd.GeoDataFrame([{'geometry': neud_center.iloc[0]}], crs=territory.crs).to_crs(local_crs)
    merged = get_accessibility(G,territory=True)
    merged.drop(columns=['geometry','x','y'],inplace=True)
    merged['water_objects'] = round(gpd.sjoin_nearest(neud_center.to_crs(local_crs), services['water_objects'].to_crs(local_crs), how='inner', distance_col='dist')[
        'dist'].min()/1000/60, 3)
    merged['oopt'] = round(gpd.sjoin_nearest(neud_center.to_crs(local_crs), services['oopt'].to_crs(local_crs), how='inner', distance_col='dist')[
        'dist'].min()/1000/60, 3)

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