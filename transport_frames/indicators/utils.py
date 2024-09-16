import pandas as pd
import geopandas as gpd
import numpy as np
import sys
sys.path.append('/Users/polina/Desktop/github/transport_frames')
import numpy as np
from iduedu import get_adj_matrix_gdf_to_gdf



def density_roads(gdf_polygon: gpd.GeoDataFrame, gdf_line: gpd.GeoDataFrame, crs=3857):
    """
    Calculate the density of roads (in km) per square kilometer area for each individual polygon.

    Parameters:
    gdf_polygon (gpd.GeoDataFrame): A GeoDataFrame containing polygons representing the areas for road density calculation.
    gdf_line (gpd.GeoDataFrame): A GeoDataFrame containing lines representing the roads.
    crs (int, optional): The Coordinate Reference System to be used for the calculation. Defaults to Web Mercator (EPSG:3857).

    Returns:
    gpd.Series: A series with the calculated road density (in km per square kilometer) for each polygon.
    """

    # Check if the input is a GeoDataFrame and convert if necessary
    if not isinstance(gdf_polygon, gpd.GeoDataFrame):
        gdf_polygon = gpd.GeoDataFrame({'geometry': gdf_polygon}, crs=gdf_polygon.crs).to_crs(crs)
    
    if not isinstance(gdf_line, gpd.GeoDataFrame):
        gdf_line = gpd.GeoDataFrame({'geometry': gdf_line}, crs=gdf_line.crs).to_crs(crs)
    gdf_polygon = gdf_polygon.to_crs(crs)
    gdf_line = gdf_line.to_crs(crs)
    densities = []
    gdf_line = gpd.GeoDataFrame({'geometry': [gdf_line.unary_union]}, crs=crs)
    for idx, polygon in gdf_polygon.iterrows():
        # Create a GeoDataFrame for the current polygon
        polygon_gdf = gpd.GeoDataFrame({'geometry': [polygon.geometry]}, crs=crs)
        area_km2 = polygon_gdf.geometry.area.sum() / 1_000_000
        intersected_lines = gpd.overlay(gdf_line, polygon_gdf, how='intersection')
        road_length_km = intersected_lines.geometry.length.sum() / 1_000
        density = road_length_km / area_km2 
        densities.append(round(density, 3))

    return pd.Series(densities, index=gdf_polygon.index)

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
        gdf_from,
        gdf_to,
        weight="time_min",
        local_crs= 3857):
    """
    Compute the availability matrix showing distances between city points and service points.

    Parameters:
    graph (networkx.Graph): The input graph.
    ...
    pandas.DataFrame: The adjacency matrix representing distances.
    """
    return get_adj_matrix_gdf_to_gdf(gdf_from.to_crs(local_crs),
                                     gdf_to.to_crs(local_crs),
                                     graph,
                                     weight=weight,
                                     dtype=np.float64)


PLACEHOLDER = gpd.GeoDataFrame(geometry=[])

def create_service_dict(railway_stations=None, fuel_stations=None, ferry_terminal=None,
                        local_aerodrome=None, international_aerodrome=None, nature_reserve=None,
                        water_objects=None, railway_paths=None, bus_stops=None, bus_routes=None, local_crs=3857):
    """
    Create a dictionary of services, replacing None values with PLACEHOLDER.
    """
    services = {
        'railway_stations': railway_stations,
        'fuel_stations': fuel_stations,
        'ports': ferry_terminal,
        'local_aerodrome': local_aerodrome,
        'international_aerodrome': international_aerodrome,
        'nature_reserve': nature_reserve,
        'water_objects': water_objects,
        'train_paths': railway_paths,
        'bus_stops': bus_stops,
        'bus_routes': bus_routes,
    }
    
    # Replace None values with PLACEHOLDER
    services = {key: value.to_crs(local_crs) if value is not None else PLACEHOLDER for key, value in services.items()}
    
    return services