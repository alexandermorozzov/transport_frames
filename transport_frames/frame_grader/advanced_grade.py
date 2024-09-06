import geopandas as gpd
import pandas as pd
import numpy as np
from dongraphio import GraphType
from transport_frames.indicators.utils import availability_matrix,find_median
import networkx as nx

# Глобальные константы
GRADE_DICT = {
    0.0: "Территория находится в удалении (более 1 км) от любых известных дорог.",
    1.0: "Территория находится в непосредственной близости к дороге (попадают в радиус 5 км от границ территории), но нет ни одного узла УДС.",
    2.0: "Территория расположена в непосредственной близости от одной из региональной трассы (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, а ближайший федеральный узел находится в более чем 100км)",
    3.0: "Территория расположена в непосредственной близости от одной из региональных трасс (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, а ближайший федеральный узел находится в не более чем 100км)",
    3.5: "Территория расположена в непосредственной близости от одной из региональных трасс (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, являющейся приоритетной, а ближайший федеральный узел находится в не более чем 100км)",
    4.0: "Территория расположена в непосредственной близости от одной из региональной трассы (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, а ближайший федеральный узел находится в не более чем 10км)",
    4.5: "Территория расположена в непосредственной близости от одной из региональной трассы (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, которая является приоритетной, а ближайший федеральный узел находится в не более чем 10км)",
    5.0: "Территория расположена в непосредственной близости от одной из региональных трасс (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, являющейся приоритетной, а ближайший федеральный узел находится в не более чем 100км)"
}

CAR_ACCESS_QUART_DICT = {
    1: "Территория попадает в I квартиль связности (лучшие 25% МО) на личном транспорте",
    2: "Территория попадает во II квартиль связности (от 50% до 25% МО) на личном транспорте",
    3: "Территория попадает в III квартиль связности (от 75% до 50% МО) на личном транспорте",
    4: "Территория попадает в IV квартиль связности (худшие 25% МО) на личном транспорте",
}

PUBLIC_ACCESS_QUART_DICT = {
    1: "Территория попадает в I квартиль связности (лучшие 25% МО) на общественном транспорте",
    2: "Территория попадает во II квартиль связности (от 50% до 25% МО) на общественном транспорте",
    3: "Территория попадает в III квартиль связности (от 75% до 50% МО) на общественном транспорте",
    4: "Территория попадает в IV квартиль связности (худшие 25% МО) на общественном транспорте",
}

WEIGHT_R_STOPS_DICT = {
    False: "В радиусе 15 км отсутствуют ЖД станции",
    True: "В радиусе 15 км есть ЖД станции"
}

WEIGHT_B_STOPS_DICT = {
    False: "В радиусе 15 км отсутствуют автобусные остановки",
    True: "В радиусе 15 км есть автобусные остановки"
}

WEIGHT_FERRY_DICT = {
    False: "В радиусе 15 км отсутствуют порты/причалы/переправы",
    True: "В радиусе 15 км есть порты/причалы/переправы"
}

WEIGHT_AERO_DICT = {
    False: "В радиусе 15 км отсутствуют аэродромы",
    True: "В радиусе 15 км есть хотя бы 1 аэродром"
}


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
        if railway_stops.empty: 
            territories["weight_r_stops"] = 0.0
        else:
            territories_with_r_stops = self._get_nearest_distances(
                territories, railway_stops, "distance_to_railway_stops"
        )
            territories["weight_r_stops"] = territories_with_r_stops[
                "distance_to_railway_stops"
            ].apply(lambda x: 0.35 if x <= self.MAX_DISTANCE else 0.0)

        if bus_stops.empty: 
            territories["weight_b_stops"] = 0.0
        else:       
            territories_with_b_stops = self._get_nearest_distances(
                territories, bus_stops, "distance_to_bus_stops"
            )
            territories["weight_b_stops"] = territories_with_b_stops[
                "distance_to_bus_stops"
            ].apply(lambda x: 0.35 if x <= self.MAX_DISTANCE else 0.0)

        if ferry_stops.empty: 
            territories["weight_ferry"] = 0.0
        else: 
            territories_with_ferry = self._get_nearest_distances(
                territories, ferry_stops, "distance_to_ferry_stops"
            )
            territories["weight_ferry"] = territories_with_ferry[
                "distance_to_ferry_stops"
            ].apply(lambda x: 0.2 if x <= self.MAX_DISTANCE else 0.0)

        if airports.empty:
            territories['weight_aero'] = 0.0
            
        else:
            territories_with_airports = self._get_nearest_distances(
                territories, airports, "distance_to_airports"
            )
            territories["weight_aero"] = territories_with_airports[
                "distance_to_airports"
            ].apply(lambda x: 0.1 if x <= self.MAX_DISTANCE else 0.0)

        # Initialize weights
        territories["weight"] = 0.0

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
        nearest_reset = nearest.reset_index()
        min_idx = nearest_reset.groupby('name_left')[distance_col].idxmin()
        nearest_min = nearest_reset.loc[min_idx]
        nearest_min = nearest_min.sort_values('index')
        res = nearest_min.reset_index(drop=True).drop(columns=['index_right'])
        return res

    @staticmethod
    def calculate_quartiles(df: pd.DataFrame, column: str) -> pd.Series:
        """Calculate quartile ranks (1 to 4) for a given column in a DataFrame."""
        return pd.qcut(df[column], q=4, labels=False) + 1

    def assign_grades(
        self, graded_territories: gpd.GeoDataFrame, accessibility_data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        # Ensure both GeoDataFrames have the same CRS
        accessibility_data = accessibility_data.to_crs(epsg=self.local_crs)
        graded_territories = graded_territories.to_crs(epsg=self.local_crs)

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
        # joined.reset_index(drop=True, inplace=True)

        # Initialize columns for car and public transport grades
        joined["car_grade"] = 0.0
        joined["public_transport_grade"] = 0.0

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

            joined.at[idx, "car_grade"] = round(car_grade,3)
            joined.at[idx, "public_transport_grade"] = round(public_transport_grade,3)
        joined["overall_assessment"] = (
            joined["car_grade"] + joined["public_transport_grade"]
        ) / 2
        joined.rename(columns={'name_left':'name'},inplace=True)
        joined = joined.sort_index()

        return joined[
            [
                "geometry",
                "name",
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

    placeholder = gpd.GeoDataFrame(geometry=[], crs=3857)
    def get_criteria(
            self,
            graded_terr: gpd.GeoDataFrame,
            points : gpd.GeoDataFrame,
            polygons: gpd.GeoDataFrame,
            citygraph: nx.MultiDiGraph = None,
            inter : nx.MultiDiGraph = None,
            r_stops: gpd.GeoDataFrame = placeholder,
            b_stops: gpd.GeoDataFrame = placeholder,
            ferry: gpd.GeoDataFrame = placeholder,
            aero: gpd.GeoDataFrame = placeholder,
            adj_mx_drive: pd.DataFrame = None,
            adj_mx_inter: pd.DataFrame = None,

        ) -> gpd.GeoDataFrame:
            
            graded_terr.reset_index(drop=True,inplace=True)
            self.adj_mx_drive = availability_matrix(citygraph, points) if adj_mx_drive is None else adj_mx_drive
            p = find_median(points, self.adj_mx_drive)
            p_agg = p[p['to_service'] < np.finfo(np.float64).max].copy()
            res = gpd.sjoin(
                p_agg, polygons, how='left', predicate='within'
            ).groupby('index_right').median(['to_service']).reset_index()
            result_df = pd.merge(
                polygons.reset_index(), res, left_on='index', right_on='index_right', how='left'
            ).rename(columns={'to_service': 'in_car'})

            result_df = result_df.drop(columns=['index_right'])

            self.adj_mx_inter = availability_matrix(
                inter, points, graph_type=[GraphType.PUBLIC_TRANSPORT, GraphType.WALK]
            ) if adj_mx_inter is None else adj_mx_inter
            p_inter = find_median(points, self.adj_mx_inter)
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
            self.criteria = result
            return result

    def interpret_gdf(self) :
        interpretation_list = []
        for i,row in self.criteria.iterrows():

            interpretation_row = interpretation(row['grade'],
                                            row['weight_r_stops'],
                                            row['weight_b_stops'],
                                            row['weight_ferry'],
                                            row['weight_aero'],
                                            row['car_access_quartile'],
                                            row['public_access_quartile'])
            interpretation_list.append((row['name'],interpretation_row))
        return interpretation_list


@staticmethod
def interpretation(
    grade,  
    weight_r_stops, 
    weight_b_stops, 
    weight_ferry, 
    weight_aero,
    car_access_quartile,
    public_access_quartile,
):
    
    texts = []

    # Frame interpretation
    grade_text = GRADE_DICT[grade] + ";"
    texts.append(grade_text)

    # Transport services availability
    if all([weight_r_stops > 0, weight_b_stops > 0, weight_ferry > 0, weight_aero > 0]):
        services_text = "В радиусе 15 км есть инфраструктура наземного, водного, воздушного общественного транспорта."
        normalized_services = 1  # All services available
    else:
        missing_services = []

        if weight_r_stops == 0:
            missing_services.append("ЖД станции")
        if weight_b_stops == 0:
            missing_services.append("автобусные остановки")
        if weight_ferry == 0:
            missing_services.append("порты/причалы/переправы")
        if weight_aero == 0:
            missing_services.append("аэродромы")

        services_text = f"В радиусе 15 км отсутствуют {', '.join(missing_services)}."
        normalized_services = sum([weight_r_stops > 0, weight_b_stops > 0, weight_ferry > 0, weight_aero > 0]) / 4

    # Interpretation by accessibility quartiles
    car_access_text = CAR_ACCESS_QUART_DICT[car_access_quartile] + ";"
    normalized_car_access = (5 - car_access_quartile) / 4  # From 0 to 1 (reversed)

    public_access_text = PUBLIC_ACCESS_QUART_DICT[public_access_quartile] + ";"
    normalized_public_access = (5 - public_access_quartile) / 4  # From 0 to 1 (reversed)

    # Sorting scores by quartiles
    quartile_grades = sorted(
        [(normalized_car_access, car_access_text), (normalized_public_access, public_access_text)], 
        reverse=True, 
        key=lambda x: x[0]
    )

    # Sorting grades by service
    service_grades = [(normalized_services, services_text)]

    
    sorted_grades = quartile_grades + service_grades

    # Final interpretation
    texts.extend([text for _, text in sorted_grades])

    return texts