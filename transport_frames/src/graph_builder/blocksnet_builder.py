import shapely
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from blocksnet import BlocksGenerator
from shapely import Polygon, MultiPolygon
from osmnx.features import InsufficientResponseError
from transport_frames.transport_frames.constants.constant_osm_tags import (
    service_tags,
    roads_tags,
    water_tags,
    rail_tags,
)

# service_tags = pd.DataFrame(service_tags)


def fetch_territory(territory_name, by_osmid=False):

    territory = ox.geocode_to_gdf(territory_name, by_osmid=by_osmid)
    territory = territory.set_crs(4326)
    territory = territory["geometry"].reset_index(drop=True)
    territory = gpd.GeoDataFrame(territory.make_valid(), columns=["geometry"])

    return territory


def fetch_buildings(territory, express_mode=True):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    buildings = ox.features_from_polygon(territory, tags={"building": True})
    buildings = buildings.loc[buildings["geometry"].type == "Polygon"]

    if not express_mode:
        buildings_ = ox.features_from_polygon(territory, tags={"building": "yes"})
        buildings_ = buildings_.loc[buildings_["geometry"].type == "Polygon"][
            "geometry"
        ]
        buildings = gpd.GeoSeries(
            pd.concat([buildings, buildings_], ignore_index=True)
        ).drop_duplicates()

    try:
        buildings = (
            buildings[["geometry", "building:levels"]]
            .reset_index(drop=True)
            .rename(columns={"building:levels": "levels"})
        )
    except:
        buildings = buildings["geometry"].reset_index(drop=True)

    buildings = gpd.GeoDataFrame(buildings)

    return buildings


def fetch_long_query(territory, tags, subdivision=3, verbose=True):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    cells = create_grid(territory, n_cells=subdivision)
    res_list = []

    for poly in tqdm(cells["geometry"], leave=False, disable=not verbose):
        try:
            services_in_cell = ox.features_from_polygon(poly, tags)
        except InsufficientResponseError:
            continue
        except:
            services_in_cell = fetch_long_query(poly, tags, subdivision)

        if len(services_in_cell) > 0:
            res_list.append(services_in_cell)

    res = pd.concat(res_list) if res_list else gpd.GeoDataFrame()

    return res


def fetch_roads(territory):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    try:
        roads = ox.features_from_polygon(territory, roads_tags)
    except:
        print("too many roads...")
        roads = fetch_long_query(territory, roads_tags)

    roads = roads.loc[roads.geom_type.isin(["LineString", "MultiLineString"])]
    roads = roads.reset_index(drop=True)["geometry"]

    roads = gpd.GeoDataFrame(roads)

    return roads


def fetch_water(territory):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    try:
        water = ox.features_from_polygon(territory, water_tags)
        water = water.loc[
            water.geom_type.isin(
                ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
            )
        ]

        water = water.reset_index(drop=True)["geometry"].drop_duplicates()
        water = gpd.GeoDataFrame(water)

        return water
    except:
        return


def fetch_railways(territory):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    try:
        railway = ox.features_from_polygon(territory, rail_tags).reset_index(drop=True)

        try:
            railway = railway.query('service not in ["crossover","siding","yard"]')
        except:
            pass

        railway = railway["geometry"]
        railway = gpd.GeoDataFrame(railway)

        return railway
    except:
        return


def create_grid(gdf=None, n_cells=5, crs=4326):

    if type(gdf) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        xmin, ymin, xmax, ymax = gdf.total_bounds
    elif type(gdf) in [Polygon, MultiPolygon]:
        xmin, ymin, xmax, ymax = gdf.bounds

    cell_size = (xmax - xmin) / n_cells
    grid_cells = []

    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(poly)

    cells = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=crs)

    if type(gdf) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        cells = gdf.overlay(cells, keep_geom_type=True)
    elif type(gdf) in [Polygon, MultiPolygon]:
        cells = gdf.intersection(cells)

    cells = cells[~cells.is_empty].set_crs(crs)
    cells = cells[np.logical_or(cells.type == "Polygon", cells.type == "MultiPolygon")]

    return cells


def fetch_long_query(territory, tags, subdivision=3, verbose=True):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    cells = create_grid(territory, n_cells=subdivision)
    res_list = []

    for poly in tqdm(cells["geometry"], leave=False, disable=not verbose):
        try:
            objects_in_cell = ox.features_from_polygon(poly, tags)
        except InsufficientResponseError:
            continue
        except:
            objects_in_cell = fetch_long_query(poly, tags, subdivision)

        if len(objects_in_cell) > 0:
            res_list.append(objects_in_cell)

    res = pd.concat(res_list) if res_list else gpd.GeoDataFrame()

    return res


def fetch_services(territory, service_tags=service_tags, subdivision=3, verbose=True):

    if type(territory) in [gpd.GeoDataFrame, gpd.GeoSeries]:
        territory = territory.unary_union

    res_list = []

    for category in tqdm(service_tags.columns, disable=not verbose):
        tags = dict(service_tags[category].dropna())

        try:
            services_temp = ox.features_from_polygon(territory, tags)
        except InsufficientResponseError:
            continue
        except:
            services_temp = fetch_long_query(territory, tags, subdivision)

        good_keys = list(set(tags.keys()).intersection(services_temp.columns))
        services_temp_tags = (
            services_temp[good_keys]
            .reset_index(drop=True)
            .apply(lambda x: list(x.dropna()), axis=1)
        )
        services_geometry = services_temp[
            ["name", "geometry"] if "name" in services_temp else ["geometry"]
        ].reset_index(drop=True)

        services_temp = pd.concat(
            [services_geometry, services_temp_tags], axis=1
        ).reset_index(drop=True)
        services_temp["category"] = category
        services_temp = services_temp.rename(columns={0: "tags"})
        res_list.append(services_temp)

    res = pd.concat(res_list) if res_list else gpd.GeoDataFrame()
    res["geometry"] = res.to_crs(3857)["geometry"].centroid.to_crs(4326)
    res = res.reset_index(drop=True)

    return res


def fetch(territory_id, by_osmid=True, save=False, local_crs=4326):

    territory = fetch_territory(territory_id, by_osmid=by_osmid)

    water = fetch_water(territory).to_crs(local_crs)
    roads = fetch_roads(territory).to_crs(local_crs)
    railways = fetch_railways(territory).to_crs(local_crs)
    buildings = fetch_buildings(territory).to_crs(local_crs)

    if save:
        water.to_file("water.geojson")
        roads.to_file("roads.geojson")
        railways.to_file("railways.geojson")
        buildings.to_file("buildings.geojson")
        territory.to_file("territory.geojson")

    blocks = BlocksGenerator(
        territory=territory.to_crs(local_crs),
        water=water,
        roads=roads,
        railways=railways,
    ).generate_blocks()

    return blocks
