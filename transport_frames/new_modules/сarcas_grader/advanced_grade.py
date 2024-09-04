import geopandas as gpd
import pandas as pd
import numpy as np
from dongraphio import GraphType
from transport_frames.new_modules.indicators.utils import availability_matrix,find_median

class AdvancedGrader:
    def __init__(self, local_crs=3857):
        self.local_crs = local_crs
        self.MAX_DISTANCE = 15000 

    def weight_territory(
        self,
        territories: gpd.GeoDataFrame,
        railway_stops: gpd.GeoDataFrame,
        bus_stops: gpd.GeoDataFrame,
        ferry_stops: gpd.GeoDataFrame,
        airports: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        # Ensure all GeoDataFrames have the same CRS
        territories = territories.to_crs(self.local_crs)
        railway_stops = railway_stops.to_crs(self.local_crs)
        bus_stops = bus_stops.to_crs(self.local_crs)
        ferry_stops = ferry_stops.to_crs(self.local_crs)
        airports = airports.to_crs(self.local_crs)

        # Calculate nearest distances and weights for each type of point
        territories_with_r_stops = self._get_nearest_distances(
            territories, railway_stops, "distance_to_railway_stops"
        )
        territories_with_b_stops = self._get_nearest_distances(
            territories, bus_stops, "distance_to_bus_stops"
        )
        territories_with_ferry = self._get_nearest_distances(
            territories, ferry_stops, "distance_to_ferry_stops"
        )
        territories_with_airports = self._get_nearest_distances(
            territories, airports, "distance_to_airports"
        )

        # Initialize weights
        territories["weight"] = 0.0

        territories["weight_r_stops"] = territories_with_r_stops[
            "distance_to_railway_stops"
        ].apply(lambda x: 0.35 if x <= self.MAX_DISTANCE else 0.0)

        territories["weight_b_stops"] = territories_with_b_stops[
            "distance_to_bus_stops"
        ].apply(lambda x: 0.35 if x <= self.MAX_DISTANCE else 0.0)

        territories["weight_ferry"] = territories_with_ferry[
            "distance_to_ferry_stops"
        ].apply(lambda x: 0.2 if x <= self.MAX_DISTANCE else 0.0)

        territories["weight_aero"] = territories_with_airports[
            "distance_to_airports"
        ].apply(lambda x: 0.1 if x <= self.MAX_DISTANCE else 0.0)

        # Calculate weights based on distances
        territories["weight"] += territories["weight_r_stops"]
        territories["weight"] += territories["weight_b_stops"]
        territories["weight"] += territories["weight_ferry"]
        territories["weight"] += territories["weight_aero"]

        return territories

    def _get_nearest_distances(
        self, territories: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, distance_col: str
    ) -> gpd.GeoDataFrame:
        nearest = territories.sjoin_nearest(stops, distance_col=distance_col)
        nearest = nearest.reset_index().loc[
            nearest.groupby(nearest.name_left)[distance_col].idxmin()
        ].reset_index().drop(columns=['index_right'])
        return nearest

    @staticmethod
    def calculate_quartiles(df: pd.DataFrame, column: str) -> pd.Series:
        """Calculate quartile ranks (1 to 4) for a given column in a DataFrame."""
        return pd.qcut(df[column], q=4, labels=False) + 1

    def assign_grades(
        self, graded_territories: gpd.GeoDataFrame, accessibility_data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        # Ensure both GeoDataFrames have the same CRS
        accessibility_data = accessibility_data.to_crs(epsg=32636)
        graded_territories = graded_territories.to_crs(epsg=32636)

        # Calculate quartile ranks for 'in_car' and 'in_inter'
        accessibility_data["car_access_quartile"] = self.calculate_quartiles(
            accessibility_data, "in_car"
        )
        accessibility_data["public_access_quartile"] = self.calculate_quartiles(
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
                "weight_r_stops",
                "weight_b_stops",
                "weight_ferry",
                "weight_aero",
                "car_access_quartile",
                "public_access_quartile",
                "car_grade",
                "public_transport_grade",
                "overall_assessment",
            ]
        ]

    def get_criteria(
        self,
        citygraph,
        graded_terr,
        points,
        polygons,
        inter,
        r_stops,
        b_stops,
        ferry,
        aero
    ) -> gpd.GeoDataFrame:
        adj_mx = availability_matrix(citygraph, points)
        p = find_median(points, adj_mx)
        p_agg = p[p['to_service'] < np.finfo(np.float64).max].copy()
        res = gpd.sjoin(
            p_agg, polygons, how='left', op='within'
        ).groupby('index_right').median(['to_service']).reset_index()
        result_df = pd.merge(
            polygons.reset_index(), res, left_on='index', right_on='index_right', how='left'
        ).rename(columns={'to_service': 'in_car'})
        result_df = result_df.drop(columns=['index_right'])

        adj_mx_inter = availability_matrix(
            inter, points, graph_type=[GraphType.PUBLIC_TRANSPORT, GraphType.WALK]
        )
        p_inter = find_median(points, adj_mx_inter)
        points_inter = p_inter[p_inter['to_service'] < np.finfo(np.float64).max].copy()

        res_inter = gpd.sjoin(
            points_inter, polygons, how="left", predicate="within"
        ).groupby('index_right').median(['to_service']).reset_index()
        result_df_inter = pd.merge(
            result_df, res_inter, left_on='index', right_on='index_right', how='left'
        ).drop(columns=['index_right']).rename(columns={'to_service': 'in_inter'})

        graded_gdf = self.weight_territory(graded_terr, r_stops, b_stops, ferry, aero)
        result = self.assign_grades(
            graded_gdf, result_df_inter[['index', 'name', 'geometry', 'in_car', 'in_inter']]
        )
        return result
