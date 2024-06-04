import geopandas as gpd
import pandas as pd


def weight_territory(
    territories: gpd.GeoDataFrame,
    railway_stops: gpd.GeoDataFrame,
    bus_stops: gpd.GeoDataFrame,
    ferry_stops: gpd.GeoDataFrame,
    airports: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Calculates the weight for each territory based on their distances to various transport facilities.

    Parameters:
        territories (GeoDataFrame): A GeoDataFrame containing the polygons of the territories to be graded.
        railway_stops (GeoDataFrame): A GeoDataFrame containing the locations of railway stops.
        bus_stops (GeoDataFrame): A GeoDataFrame containing the locations of bus stops.
        ferry_stops (GeoDataFrame): A GeoDataFrame containing the locations of ferry stops.
        airports (GeoDataFrame): A GeoDataFrame containing the locations of airports.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the graded territories with an added 'weight' column.
    """
    # Ensure all GeoDataFrames have the same CRS
    common_crs = 32636
    territories = territories.to_crs(epsg=common_crs)
    railway_stops = railway_stops.to_crs(epsg=common_crs)
    bus_stops = bus_stops.to_crs(epsg=common_crs)
    ferry_stops = ferry_stops.to_crs(epsg=common_crs)
    airports = airports.to_crs(epsg=common_crs)

    # Define maximum distance
    MAX_DISTANCE = 15000  # 15 km

    # Calculate nearest distances and weights for each type of point
    territories_with_r_stops = territories.sjoin_nearest(
        railway_stops, distance_col="distance_to_railway_stops"
    )
    territories_with_b_stops = territories.sjoin_nearest(
        bus_stops, distance_col="distance_to_bus_stops"
    )
    territories_with_ferry = territories.sjoin_nearest(
        ferry_stops, distance_col="distance_to_ferry_stops"
    )
    territories_with_airports = territories.sjoin_nearest(
        airports, distance_col="distance_to_airports"
    )

    # Initialize weights
    territories["weight"] = 0.0

    # Calculate weights based on distances
    territories["weight"] += territories_with_r_stops[
        "distance_to_railway_stops"
    ].apply(lambda x: 0.35 if x <= MAX_DISTANCE else 0.0)
    territories["weight"] += territories_with_b_stops["distance_to_bus_stops"].apply(
        lambda x: 0.35 if x <= MAX_DISTANCE else 0.0
    )
    territories["weight"] += territories_with_ferry["distance_to_ferry_stops"].apply(
        lambda x: 0.20 if x <= MAX_DISTANCE else 0.0
    )
    territories["weight"] += territories_with_airports["distance_to_airports"].apply(
        lambda x: 0.10 if x <= MAX_DISTANCE else 0.0
    )

    return territories


def calculate_quartiles(df: pd.DataFrame, column: str) -> pd.Series:
    """Calculate quartile ranks (1 to 4) for a given column in a DataFrame."""
    return pd.qcut(df[column], q=4, labels=False) + 1


def assign_grades(
    graded_territories: gpd.GeoDataFrame, accessibility_data: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assign grades to territories based on their accessibility and weight.

    Parameters:
        graded_territories (GeoDataFrame): A GeoDataFrame containing territories with calculated weights.
        accessibility_data (GeoDataFrame): A GeoDataFrame containing accessibility data for areas.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the territories with assigned grades.
    """
    # Ensure both GeoDataFrames have the same CRS
    accessibility_data = accessibility_data.to_crs(epsg=32636)
    accessibility_data["geom"] = accessibility_data["geometry"]
    graded_territories = graded_territories.to_crs(epsg=32636)

    # Calculate quartile ranks for 'in_car' and 'in_inter'
    accessibility_data["car_access_quartile"] = calculate_quartiles(
        accessibility_data, "in_car"
    )
    accessibility_data["public_access_quartile"] = calculate_quartiles(
        accessibility_data, "in_inter"
    )

    # Spatial join to find the matching polygons
    joined = gpd.sjoin(
        graded_territories, accessibility_data, how="left", predicate="intersects"
    )

    # Calculate the area of intersection
    joined["intersection_area"] = joined.apply(
        lambda row: row["geometry"]
        .intersection(accessibility_data.loc[row["index_right"], "geometry"])
        .area,
        axis=1,
    )

    # Sort by intersection area and drop duplicates to keep only the max area
    joined = joined.sort_values("intersection_area", ascending=False).drop_duplicates(
        subset="geometry"
    )
    joined.reset_index(drop=True, inplace=True)

    # Initialize columns for car and public transport grades
    joined["car_grade"] = 0
    joined["public_transport_grade"] = 0

    # Define the grade tables
    car_grade_table = {
        5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4.5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        3.5: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        3: {"Q4": 0, "Q3": 1, "Q2": 3, "Q1": 3},
        2.5: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 3},
        2: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 2},
        1.5: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 2},
        1: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 1},
        0: {"Q4": 0, "Q3": 0, "Q2": 0, "Q1": 1},
    }

    public_transport_grade_table = {
        5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4.5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        3.5: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 5},
        3: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        2.5: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        2: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 4},
        1.5: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 3},
        1: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 3},
        0: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 2},
    }

    # Apply grades based on the quartiles and grade
    for idx, row in joined.iterrows():
        grade = row["grade"]
        car_quartile = row["car_access_quartile"]
        public_transport_quartile = row["public_access_quartile"]
        car_grade = car_grade_table.get(grade, {}).get(f"Q{car_quartile}", 0)
        public_transport_grade = (
            public_transport_grade_table.get(grade, {}).get(
                f"Q{public_transport_quartile}", 0
            )
            * row["weight"]
        )

        joined.at[idx, "car_grade"] = car_grade
        joined.at[idx, "public_transport_grade"] = public_transport_grade
    joined["overall_assessment"] = (
        joined["car_grade"] + joined["public_transport_grade"]
    ) / 2

    return joined[
        [
            "geometry",
            "grade",
            "weight",
            "car_access_quartile",
            "public_access_quartile",
            "car_grade",
            "public_transport_grade",
            "overall_assessment",
        ]
    ]
