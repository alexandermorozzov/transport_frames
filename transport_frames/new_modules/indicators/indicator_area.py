import osmnx as ox
import pandas as pd
import geopandas as gpd
import momepy
from dongraphio import DonGraphio
from transport_frames.new_modules.utils.helper_funcs import prepare_graph
from transport_frames.new_modules.indicators.utils import get_accessibility, density_roads, new_connectivity, aggregate_road_lengths, aggregate_routes_by_polygon, aggregate_services_by_polygon


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