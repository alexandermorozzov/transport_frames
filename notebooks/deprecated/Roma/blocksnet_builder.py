import shapely
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from blocksnet import BlocksGenerator
from shapely import Polygon,MultiPolygon
from osmnx.features import InsufficientResponseError


service_tags = {
    "leisure":{"amenity":["cinema","public_bath","bbq","nightclub","casino"],"leisure":["bird_hide","picnic_table","dog_park","sauna","fitness_station","swimming_pool","water_park","resort"],"tourism":["zoo","aquarium","theme_park"]},
    "nature": {"leisure": ["park", "beach_resort"], "landuse": ["meadow"], "natural": ["forest", "beach"]},
    "education": {"amenity": ["childcare", "school", "university", "kindergarten", "college", "language_school", "driving_school", "music_school", "dancing_school", "prep_school"]},
    "car": {"amenity": ["fuel","charging_station","car_wash","vehicle_inspection"],"shop":["car","car_repair","car_parts","tyres"]},
    "accomodation": {"tourism": ["alpine_hut", "caravan_site", "motel", "hostel", "wilderness_hut", "camp_site", "chalet", "apartment", "hotel", "camp_pitch", "guest_house"]},
    "culture": {"tourism": ["museum", "galery", "artwork"], "amenity": ["theatre", "library", "public_bookcase", "planetarium", "arts_centre", "studio"]},
    "craft": {"craft": ["basket_maker", "sun_protection", "print_shop", "sailmaker", "carpet_layer", "tailor", "watchmaker", "plasterer", "sweep", "beekeeper", "winery", "key_cutter", "clockmaker", "sculptor", "metal_construction", "plumber", "cabinet_maker", "upholsterer", "scaffolder", "photo_studio", "tiler", "boatbuilder", "mason", "heating_engineer", "brewery", "caterer", "cleaning", "stand_builder", "glass", "painter", "confectionery", "distillery", "furniture", "photographer", "hvac", "pottery", "leather", "window_construction", "saddler", "floorer", "shoemaker", "handicraft", "builder", "roofer", "carpenter", "signmaker", "stonemason", "sawmill", "bookbinder", "dressmaker", "glaziery", "rigger", "tinsmith", "joiner", "blacksmith", "metal_works", "laboratory", "insulation", "gardener", "electrician", "turner", "jeweller", "parquet_layer", "optician", "locksmith", "electronics_repair", "agricultural_engines", "photographic_laboratory"]},
    "food": {"amenity": ["restaurant", "fast_food", "food_court", "cafe", "bubble_tea", "ice_cream", "pub", "biergarten", "bar"]},
    "office": {"office": ["charity", "foundation", "advertising_agency", "energy_supplier", "newspaper", "telecommunication", "estate_agent", "employment_agency", "company", "notary", "tax_advisor", "logistics", "travel_agent", "financial", "diplomatic", "architect", "lawyer", "water_utility", "cooperative", "association", "forestry", "camping", "consulting", "administrative", "religion", "guide", "office", "educational_institution", "accountant", "political_party", "therapist", "research", "government", "it", "vacant", "physician", "parish", "surveyor", "insurance", "quango", "ngo", "publisher", "moving_company", "register"]},
    "shop": {"amenity": ["vending_machine", "marketplace"], "beauty": ["nails"], "shop": ["organic", "outdoor", "cheese", "hearing_aids", "interior_decoration", "anime", "window_blind", "scuba_diving", "deli", "video", "greengrocer", "bakery", "sports", "shoe_repair", "wine", "perfumery", "seafood", "boutique", "cannabis", "computer", "laundry", "nutrition_supplements", "erotic", "bag", "fabric", "mobile_phone", "frozen_food", "electronics", "farm", "art", "houseware", "beverages", "newsagent", "variety_store", "fishing", "craft", "weapons", "department_store", "water", "baby_goods", "tobacco", "fireplace", "herbalist", "wholesale", "coffee", "kiosk", "candles", "beauty", "hairdresser", "shoes", "toys", "appliance", "vacuum_cleaner", "furnace", "butcher", "doors", "party", "pyrotechnics", "health_food", "bathroom_furnishing", "stationery", "carpet", "doityourself", "bookmaker", "kitchen", "massage", "garden_centre", "tiles", "fashion", "water_sports", "security", "alcohol", "tea", "convenience", "games", "books", "funeral_directors", "hardware", "clothes", "hairdresser_supply", "music", "dry_cleaning", "second_hand", "paint", "copyshop", "sewing", "florist", "gift", "hifi", "pet_grooming", "bed", "spices", "ticket", "hunting", "e-cigarette", "pastry", "chocolate", "medical_supply", "fashion_accessories", "photo", "mall", "general", "cosmetics", "tattoo", "chemist", "watches", "electrical", "trade", "travel_agency", "gas", "curtain", "dairy", "video_games", "pet", "frame", "lighting", "jewelry", "storage_rental", "radiotechnics", "antiques", "lottery", "musical_instrument", "supermarket"]}}
service_tags = pd.DataFrame(service_tags)


def fetch_territory(territory_name, by_osmid=False):

    territory = ox.geocode_to_gdf(territory_name, by_osmid=by_osmid)
    territory = territory.set_crs(4326)
    territory = territory["geometry"].reset_index(drop=True)
    territory = gpd.GeoDataFrame(territory.make_valid(),columns=['geometry'])

    return territory


def fetch_buildings(territory, express_mode=True):

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    buildings = ox.features_from_polygon(territory, tags={"building": True})
    buildings = buildings.loc[buildings["geometry"].type == "Polygon"]

    if not express_mode:
        buildings_ = ox.features_from_polygon(territory, tags={"building": "yes"})
        buildings_ = buildings_.loc[buildings_["geometry"].type == "Polygon"]["geometry"]
        buildings = gpd.GeoSeries(pd.concat([buildings, buildings_], ignore_index=True)).drop_duplicates()

    try:
        buildings = buildings[["geometry", "building:levels"]].reset_index(drop=True).rename(columns={"building:levels": "levels"})
    except:
        buildings = buildings["geometry"].reset_index(drop=True)

    buildings = gpd.GeoDataFrame(buildings)

    return buildings


def fetch_long_query(territory, tags, subdivision=3,verbose=True):

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    cells = create_grid(territory,n_cells=subdivision)
    res_list = []

    for poly in tqdm(cells['geometry'],leave=False,disable=not verbose):
        try:
            services_in_cell = ox.features_from_polygon(poly, tags)
        except InsufficientResponseError:
            continue
        except:
            services_in_cell = fetch_long_query(poly,tags,subdivision)

        if len(services_in_cell) > 0: res_list.append(services_in_cell)

    res = pd.concat(res_list) if res_list else gpd.GeoDataFrame()

    return res


def fetch_roads(territory):
    tags = {
        "highway": ["construction","crossing","living_street","motorway","motorway_link","motorway_junction","pedestrian","primary","primary_link","raceway","residential","road","secondary","secondary_link","services","tertiary","tertiary_link","track","trunk","trunk_link","turning_circle","turning_loop","unclassified",],
        "service": ["living_street", "emergency_access"]
    }

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    try:
        roads = ox.features_from_polygon(territory, tags)
    except:
        print('too many roads...')
        roads = fetch_long_query(territory,tags)

    roads = roads.loc[roads.geom_type.isin(['LineString','MultiLineString'])]
    roads = roads.reset_index(drop=True)["geometry"]

    roads = gpd.GeoDataFrame(roads)

    return roads


def fetch_water(territory):

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    try:
        water = ox.features_from_polygon(
            territory, {'riverbank':True,
                        'reservoir':True,
                        'basin':True,
                        'dock':True,
                        'canal':True,
                        'pond':True,
                        'natural':['water','bay'],
                        'waterway':['river','canal','ditch'],
                        'landuse':'basin'})
        water = water.loc[water.geom_type.isin(
            ['Polygon','MultiPolygon','LineString','MultiLineString'])]

        water = water.reset_index(drop=True)["geometry"].drop_duplicates()
        water = gpd.GeoDataFrame(water)

        return water
    except:
        return


def fetch_railways(territory):

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    try:
        railway = ox.features_from_polygon(
            territory, {"railway": "rail"}).reset_index(drop=True)

        try:
            railway = railway.query('service not in ["crossover","siding","yard"]')
        except:
            pass

        railway = railway["geometry"]
        railway  = gpd.GeoDataFrame(railway)

        return railway
    except:
        return


def create_grid(gdf=None, n_cells=5, crs=4326):

    if type(gdf) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        xmin, ymin, xmax, ymax= gdf.total_bounds
    elif type(gdf) in [Polygon,MultiPolygon]:
        xmin, ymin, xmax, ymax= gdf.bounds

    cell_size = (xmax-xmin)/n_cells
    grid_cells = []

    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1 = x0-cell_size
            y1 = y0+cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(poly)

    cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'],crs=crs)

    if type(gdf) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        cells = gdf.overlay(cells,keep_geom_type=True)
    elif type(gdf) in [Polygon,MultiPolygon]:
        cells = gdf.intersection(cells)

    cells = cells[~cells.is_empty].set_crs(crs)
    cells = cells[np.logical_or(
        cells.type=='Polygon',cells.type=='MultiPolygon')]

    return cells


def fetch_long_query(territory, tags, subdivision=3,verbose=True):

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    cells = create_grid(territory,n_cells=subdivision)
    res_list = []

    for poly in tqdm(cells['geometry'],leave=False,disable=not verbose):
        try:
            objects_in_cell = ox.features_from_polygon(poly, tags)
        except InsufficientResponseError:
            continue
        except:
            objects_in_cell = fetch_long_query(poly,tags,subdivision)

        if len(objects_in_cell) > 0: res_list.append(objects_in_cell)

    res = pd.concat(res_list) if res_list else gpd.GeoDataFrame()

    return res


def fetch_services(territory, service_tags=service_tags,subdivision=3,verbose=True):

    if type(territory) in [gpd.GeoDataFrame,gpd.GeoSeries]:
        territory = territory.unary_union

    res_list = []

    for category in tqdm(service_tags.columns, disable=not verbose):
        tags = dict(service_tags[category].dropna())

        try:
            services_temp = ox.features_from_polygon(territory,tags)
        except InsufficientResponseError:
            continue
        except:
            services_temp = fetch_long_query(territory,tags,subdivision)

        good_keys = list(set(tags.keys()).intersection(services_temp.columns))
        services_temp_tags = services_temp[good_keys].reset_index(drop=True).apply(lambda x: list(x.dropna()), axis=1)
        services_geometry = services_temp[['name','geometry'] if 'name' in services_temp else ['geometry']].reset_index(drop=True)

        services_temp = pd.concat([services_geometry,services_temp_tags],axis=1).reset_index(drop=True)
        services_temp["category"] = category
        services_temp = services_temp.rename(columns={0: "tags"})
        res_list.append(services_temp)

    res = pd.concat(res_list) if res_list else gpd.GeoDataFrame()
    res["geometry"] = res.to_crs(3857)["geometry"].centroid.to_crs(4326)
    res = res.reset_index(drop=True)

    return res

def fetch(id, by_osmid=True, save=False, local_crs=4326):

    territory = fetch_territory(id, by_osmid=by_osmid)

    water = fetch_water(territory).to_crs(local_crs)
    roads = fetch_roads(territory).to_crs(local_crs)
    railways = fetch_railways(territory).to_crs(local_crs)
    buildings = fetch_buildings(territory).to_crs(local_crs)

    if save:
        water.to_file('water.geojson')
        roads.to_file('roads.geojson')
        railways.to_file('railways.geojson')
        buildings.to_file('buildings.geojson')
        territory.to_file('territory.geojson')

    blocks = BlocksGenerator(
        territory=territory.to_crs(local_crs),
        water=water,
        roads=roads,
        railways=railways
        ).generate_blocks()
    
    return blocks