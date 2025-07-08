import pandas as pd
import geopandas as gpd
import numpy as np
import sys
import numpy as np
from iduedu import get_adj_matrix_gdf_to_gdf
import networkx as nx
PLACEHOLDER = gpd.GeoDataFrame(geometry=[])
from transport_frames.models.schema import BaseSchema
from shapely import Point, MultiPoint
from pandera.typing import Series
import pandera as pa
import pandera as pa
from pandera.typing import Series
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString


# Define specific schema for each service
class RailwayStationSchema(BaseSchema):
    _geom_types = [Point]

class RailwayPathSchema(BaseSchema):
    _geom_types = [LineString, MultiLineString]

class BusStopSchema(BaseSchema):
    _geom_types = [Point]

class BusRouteSchema(BaseSchema):
    _geom_types = [LineString, MultiLineString]

class FuelStationSchema(BaseSchema):
    _geom_types = [Point]

class AerodromeSchema(BaseSchema):
    _geom_types = [Point] # probably Polygon

class FerryTerminalSchema(BaseSchema):
    _geom_types = [Point]

class WaterObjectSchema(BaseSchema):
    _geom_types = [Polygon, MultiPolygon]

class NatureReserveSchema(BaseSchema):
    _geom_types = [Polygon, MultiPolygon]

# Master schema for services
class ServicesSchema:
    """
    Validates GeoDataFrames in the services dictionary upon initialization.
    """
    
    def __init__(self, services: dict):
        self.services = services
        self.schemas = {
            'railway_stations': RailwayStationSchema,
            'railway_paths': RailwayPathSchema,
            'bus_stops': BusStopSchema,
            'bus_routes': BusRouteSchema,
            'fuel_stations': FuelStationSchema,
            'local_aerodrome': AerodromeSchema,
            'ferry_terminal': FerryTerminalSchema,
            'international_aerodrome': AerodromeSchema,
            'water_objects': WaterObjectSchema,
            'nature_reserve': NatureReserveSchema
        }
        self._validate_services()

    def _validate_services(self):
            """
            Automatically validate each GeoDataFrame in the services dictionary
            when the ServicesSchema object is initialized, and report errors.
            """
            for service_name, schema in self.schemas.items():
                gdf = self.services.get(service_name)
                if gdf is not None:
                    try:
                        schema.validate(gdf)  # Validate the GeoDataFrame
                    except pa.errors.SchemaError as e:
                        # Capture detailed error information
                        raise ValueError(
                            f"Validation failed for '{service_name}' GeoDataFrame. "
                            f"Error details: {e.failure_cases}."
                        )


class GdfSchema(BaseSchema):
    """
    Schema for validating regions defined by polygons and multipolygons.

    Attributes:
    - name (Series[str]): The name associated with the region(s).
    - _geom_types (list): List of allowed geometry types (Polygon, MultiPolygon).
    """
    
    name: Series[str] = pa.Field(nullable=True)
    _geom_types = [Point, MultiPoint]

class PolygonSchema(BaseSchema):
    """
    Schema for validating regions defined by polygons and multipolygons.

    Attributes:
    - name (Series[str]): The name associated with the region(s).
    - _geom_types (list): List of allowed geometry types (Polygon, MultiPolygon).
    """
    
    name: Series[str] = pa.Field(nullable=True)
    _geom_types = [Polygon, MultiPolygon]


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

def find_median(city_points: gpd.GeoDataFrame, adj_mx: pd.DataFrame) -> gpd.GeoDataFrame:
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
        graph: nx.Graph,
        gdf_from: gpd.GeoDataFrame,
        gdf_to: gpd.GeoDataFrame,
        weight: str = "time_min",
        local_crs: int = 3857) -> pd.DataFrame:
    """
    Compute the availability matrix showing distances between city points and service points.

    Parameters:
    graph (networkx.Graph): The input graph.
    gdf_from (gpd.GeoDataFrame): GeoDataFrame of origin points.
    gdf_to (gpd.GeoDataFrame): GeoDataFrame of destination points.
    weight (str): The attribute of the edges to use for distance calculations. Defaults to "time_min".
    local_crs (int): The local coordinate reference system to use for transformation.

    Returns:
    pandas.DataFrame: The adjacency matrix representing distances.
    """
    gdf_to = GdfSchema(gdf_to)
    gdf_from = GdfSchema(gdf_from)
    if weight == 'length_m' or weight=='length_meter':
        return get_adj_matrix_gdf_to_gdf(gdf_from.to_crs(local_crs),
                                       gdf_to.to_crs(local_crs),
                                       graph,
                                       weight=weight,
                                       dtype=np.int64)
    return get_adj_matrix_gdf_to_gdf(gdf_from.to_crs(local_crs),
                                       gdf_to.to_crs(local_crs),
                                       graph,
                                       weight=weight,
                                       dtype=np.float32)


def create_service_dict(railway_stations: gpd.GeoDataFrame = None, 
                        fuel_stations: gpd.GeoDataFrame = None, 
                        ferry_terminal: gpd.GeoDataFrame = None,
                        local_aerodrome: gpd.GeoDataFrame = None, 
                        international_aerodrome: gpd.GeoDataFrame = None, 
                        nature_reserve: gpd.GeoDataFrame = None,
                        water_objects: gpd.GeoDataFrame = None, 
                        railway_paths: gpd.GeoDataFrame = None, 
                        bus_stops: gpd.GeoDataFrame = None, 
                        bus_routes: gpd.GeoDataFrame = None, 
                        local_crs: int = 3857) -> dict:
    """
    Create a dictionary of services, replacing None values with PLACEHOLDER.

    Parameters:
    railway_stations (gpd.GeoDataFrame): GeoDataFrame of railway stations.
    fuel_stations (gpd.GeoDataFrame): GeoDataFrame of fuel stations.
    ferry_terminal (gpd.GeoDataFrame): GeoDataFrame of ferry terminals.
    local_aerodrome (gpd.GeoDataFrame): GeoDataFrame of local aerodromes.
    international_aerodrome (gpd.GeoDataFrame): GeoDataFrame of international aerodromes.
    nature_reserve (gpd.GeoDataFrame): GeoDataFrame of nature reserves.
    water_objects (gpd.GeoDataFrame): GeoDataFrame of water bodies.
    railway_paths (gpd.GeoDataFrame): GeoDataFrame of railway paths.
    bus_stops (gpd.GeoDataFrame): GeoDataFrame of bus stops.
    bus_routes (gpd.GeoDataFrame): GeoDataFrame of bus routes.
    local_crs (int): The local coordinate reference system.

    Returns:
    dict: A dictionary containing GeoDataFrames of services, with None values replaced by PLACEHOLDER.
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
    # services_valid = ServicesSchema(services)
    return services

def availability_matrix_point_fixer(
        graph: nx.Graph,
        gdf_from: gpd.GeoDataFrame,
        gdf_to: gpd.GeoDataFrame,
        polygons: gpd.GeoDataFrame,
        weight: str = "time_min",
        local_crs: int = 3857) -> pd.DataFrame:
    """
    Compute the availability matrix showing distances between city points and service points, adding new points
    for calculation for the regions not containing them

    Parameters:
    graph (networkx.Graph): The input graph.
    gdf_from (gpd.GeoDataFrame): GeoDataFrame of origin points.
    gdf_to (gpd.GeoDataFrame): GeoDataFrame of destination points.
    polygons (gpd.GeoDataFrame): GeoDataFrame of areas which have to contain the points.
    weight (str): The attribute of the edges to use for distance calculations. Defaults to "time_min".
    local_crs (int): The local coordinate reference system to use for transformation.

    Returns:
    pandas.DataFrame: The adjacency matrix representing distances.
    """
    gdf_to = GdfSchema(gdf_to).to_crs(local_crs).copy()
    gdf_from = GdfSchema(gdf_from).to_crs(local_crs).copy()
    polygons = PolygonSchema(polygons).to_crs(local_crs).copy()

    if gdf_to.equals(gdf_from):
        joined = gpd.sjoin(polygons, gdf_to, how='left', predicate='contains')
        missing_polygons = joined[joined['index_right'].isnull()].copy()
        missing_polygons = missing_polygons.drop_duplicates(subset='geometry')
        missing_polygons['rep_point'] = missing_polygons.geometry.representative_point()

        rep_points = gpd.GeoDataFrame(
            missing_polygons[['rep_point']],
            geometry='rep_point',
            crs=polygons.crs
        ).rename(columns={'rep_point': 'geometry'})

        combined_points = pd.concat([gdf_to, rep_points], ignore_index=True)

        combined_points = combined_points.reset_index(drop=True)
        combined_points.crs = polygons.crs


    if weight == 'length_m' or weight=='length_meter':
        return get_adj_matrix_gdf_to_gdf(combined_points.to_crs(local_crs),
                                    combined_points.to_crs(local_crs),
                                    graph,
                                    weight=weight,
                                    dtype=np.int64)
    return get_adj_matrix_gdf_to_gdf(combined_points.to_crs(local_crs),
                                    combined_points.to_crs(local_crs),
                                    graph,
                                    weight=weight,
                                    dtype=np.float32)
   


import numpy as np
import pandas as pd

def fix_points(points: gpd.GeoDataFrame, 
               polygons: gpd.GeoDataFrame):
    """
    Add more points to points gdf so that each polygons contains at least one of them
    points (gpd.GeoDataFrame): GeoDataFrame containing points of interest.
    polygons (gpd.GeoDataFrame): GeoDataFrame containing polygons for spatial joining.
    """
    points =  points.to_crs(polygons.crs).copy()
    joined = gpd.sjoin(polygons, points, how='left', predicate='contains')
    missing_polygons = joined[joined['index_right'].isnull()].copy()
    if len(missing_polygons) > 0:
        missing_polygons = missing_polygons.drop_duplicates(subset='geometry')
        missing_polygons['rep_point'] = missing_polygons.geometry.representative_point()

        rep_points = gpd.GeoDataFrame(
            missing_polygons[['rep_point']],
            geometry='rep_point',
            crs=polygons.crs
        ).rename(columns={'rep_point': 'geometry'})

        combined_points = pd.concat([points, rep_points], ignore_index=True)

        combined_points = combined_points.reset_index(drop=True)
        combined_points.crs = polygons.crs
        return combined_points
    return points

def fix_points_dimension(settlement_points, polygons, adj_drive, adj_inter, graph_drive, graph_inter, weight,local_crs):
    points = fix_points(settlement_points,polygons) 
    if len(points)==len(settlement_points):
         print('Dimensions were correct')
         return points, adj_drive, adj_inter

    else:
        print('Too few settlement points, centroids were added')
        if weight == 'length_m' or weight=='length_meter':
                adj_mx_drive = get_adj_matrix_gdf_to_gdf(points.to_crs(local_crs),
                                            points.to_crs(local_crs),
                                            graph_drive,
                                            weight=weight,
                                            dtype=np.int64)
                
                adj_mx_inter = get_adj_matrix_gdf_to_gdf(points.to_crs(local_crs),
                                            points.to_crs(local_crs),
                                            graph_inter,
                                            weight=weight,
                                            dtype=np.int64)
        else: 
            adj_mx_drive = get_adj_matrix_gdf_to_gdf(points.to_crs(local_crs),
                                        points.to_crs(local_crs),
                                        graph_drive,
                                        weight=weight,
                                        dtype=np.float32)
            adj_mx_inter = get_adj_matrix_gdf_to_gdf(points.to_crs(local_crs),
                                        points.to_crs(local_crs),
                                        graph_inter,
                                        weight=weight,
                                        dtype=np.float32)

        return points, adj_mx_drive, adj_mx_inter
    
