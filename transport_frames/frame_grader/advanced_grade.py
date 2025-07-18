import geopandas as gpd
import pandas as pd
import numpy as np
from transport_frames.indicators.utils import availability_matrix,find_median
import networkx as nx
from transport_frames.indicators.utils import fix_points

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
    def __init__(self, local_crs: int = 3857) -> None:
        """
        Initialize the AdvancedGrader with a specified local coordinate reference system (CRS).

        Parameters:
        local_crs (int): The local CRS to use for calculations (default is 3857).
        """
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
        """
        Calculate the weights for each territory based on the distances to various types of stops.

        Parameters:
        territories (gpd.GeoDataFrame): GeoDataFrame containing territory geometries.
        railway_stops (gpd.GeoDataFrame): GeoDataFrame containing railway stop geometries.
        bus_stops (gpd.GeoDataFrame): GeoDataFrame containing bus stop geometries.
        ferry_stops (gpd.GeoDataFrame): GeoDataFrame containing ferry stop geometries.
        airports (gpd.GeoDataFrame): GeoDataFrame containing airport geometries.

        Returns:
        gpd.GeoDataFrame: Updated territories with calculated weights.
        """

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
        """
        Get the nearest distances between territories and stops.

        Parameters:
        territories (gpd.GeoDataFrame): GeoDataFrame containing territory geometries.
        stops (gpd.GeoDataFrame): GeoDataFrame containing stop geometries.
        distance_col (str): Column name to store the distance values.

        Returns:
        gpd.GeoDataFrame: GeoDataFrame of territories with nearest distances to stops.
        """
        nearest = territories.sjoin_nearest(stops, distance_col=distance_col)
        nearest_reset = nearest.reset_index()
        min_idx = nearest_reset.groupby('geometry')[distance_col].idxmin()
        nearest_min = nearest_reset.loc[min_idx]
        nearest_min = nearest_min.sort_values('index')
        res = nearest_min.reset_index(drop=True).drop(columns=['index_right'])
        return res

    @staticmethod
    def calculate_quartiles(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Calculate quartile ranks (1 to 4) for a given column in a DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name for which to calculate quartiles.

        Returns:
        pd.Series: Series of quartile ranks.
        """
        return pd.qcut(df[column], q=4, labels=False) + 1

    def assign_grades(
        self, graded_territories: gpd.GeoDataFrame, accessibility_data: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assign grades to territories based on accessibility data.

        Parameters:
        graded_territories (gpd.GeoDataFrame): GeoDataFrame containing territories to grade.
        accessibility_data (gpd.GeoDataFrame): GeoDataFrame containing accessibility data.

        Returns:
        gpd.GeoDataFrame: Updated graded territories with assigned grades.
        """
        # Ensure both GeoDataFrames have the same CRS
        accessibility_data = accessibility_data.to_crs(epsg=self.local_crs)
        graded_territories = graded_territories.to_crs(epsg=self.local_crs)

        accessibility_data['in_car'] = accessibility_data['in_car'].fillna(np.inf)
        accessibility_data['in_inter'] = accessibility_data['in_inter'].fillna(np.inf)

        # Calculate quartile ranks for 'in_car' and 'in_inter'
        accessibility_data["car_access_quartile"] = self.calculate_quartiles(
            accessibility_data, "in_car"
        )
        accessibility_data["public_access_quartile"] = self.calculate_quartiles(
            accessibility_data, "in_inter"
        )

        all_gecs_with_dist = gpd.sjoin_nearest(graded_territories[['grade','weight','geometry','weight_r_stops', 'weight_b_stops', 'weight_ferry', 'weight_aero']],accessibility_data,lsuffix='left',rsuffix='right',distance_col='dist')

        remote_gecs = all_gecs_with_dist[all_gecs_with_dist['dist']>0]
        remote_gecs['public_access_quartile'] = np.minimum(remote_gecs['public_access_quartile'] + 1, 4)
        remote_gecs['car_access_quartile'] = np.minimum(remote_gecs['car_access_quartile'] + 1, 4)
        remote_gecs = remote_gecs.drop_duplicates(subset="geometry")
        # remote_gecs = remote_gecs[['grade','weight','in_car','in_inter','car_access_quartile','public_access_quartile','geometry']]


        # norm_gecs = all_gecs_with_dist[all_gecs_with_dist['dist']==0]
        # norm_gecs["intersection_area"] = norm_gecs.apply(
        #     lambda row: row["geometry"]
        #     .intersection(accessibility_data.loc[row["index_right"], "geometry"])
        #     .unary_union.area  # Ensures a single geometry
        #     if isinstance(row["geometry"].intersection(accessibility_data.loc[row["index_right"], "geometry"]), (gpd.GeoSeries, pd.Series))
        #     else row["geometry"].intersection(accessibility_data.loc[row["index_right"], "geometry"]).area,
        #     axis=1,
        # )
        # norm_gecs = norm_gecs.sort_values("intersection_area", ascending=False).drop_duplicates(
        #             subset="geometry"
        #         )#[['grade','weight','in_car','in_inter','car_access_quartile','public_access_quartile','geometry']]

        # joined = pd.concat([norm_gecs,remote_gecs])
        remote_gecs = remote_gecs[['grade','weight','in_car','in_inter','car_access_quartile','public_access_quartile','geometry']]
        norm_gecs = all_gecs_with_dist[all_gecs_with_dist['dist'] == 0].copy()

        def safe_intersection_area(row):
            try:
                idx = row["index_right"]
                if pd.isna(idx):
                    return 0.0

                # Убедимся, что индекс целочисленный
                idx = int(idx)
                other_geom = accessibility_data.loc[idx, "geometry"]
                intersection = row["geometry"].intersection(other_geom)

                # Если результат — коллекция геометрий
                if isinstance(intersection, (gpd.GeoSeries, pd.Series, list)):
                    return gpd.GeoSeries(intersection).unary_union.area
                # Если результат — одиночная геометрия
                elif intersection is not None:
                    return intersection.area
                else:
                    return 0.0
            except Exception as e:
                print(f" Ошибка при вычислении intersection_area на строке {row.name}: {e}")
                return 0.0

        norm_gecs["intersection_area"] = norm_gecs.apply(safe_intersection_area, axis=1)

        norm_gecs = norm_gecs.sort_values("intersection_area", ascending=False).drop_duplicates(
            subset="geometry"
        )

        joined = pd.concat([norm_gecs, remote_gecs])




        # # Spatial join to find the matching polygons
        # joined = gpd.sjoin(
        #     graded_territories, accessibility_data, how="left", predicate="intersects"
        # )
        
        # # Calculate the area of intersection
        # joined["intersection_area"] = joined.apply(
        #     lambda row: row["geometry"]
        #     .intersection(accessibility_data.loc[row["index_right"], "geometry"])
        #     .area,
        #     axis=1,
        # )

        # # Sort by intersection area and drop duplicates to keep only the max area
        # joined = joined.sort_values("intersection_area", ascending=False).drop_duplicates(
        #     subset="geometry"
        # )
        # # joined = joined[joined['in_car'].notna()].drop_duplicates('geometry')

        # # joined.reset_index(drop=True, inplace=True)

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
            points: gpd.GeoDataFrame,
            polygons: gpd.GeoDataFrame,
            citygraph: nx.MultiDiGraph = None,
            inter: nx.MultiDiGraph = None,
            r_stops: gpd.GeoDataFrame = placeholder,
            b_stops: gpd.GeoDataFrame = placeholder,
            ferry: gpd.GeoDataFrame = placeholder,
            aero: gpd.GeoDataFrame = placeholder,
            adj_mx_drive: pd.DataFrame = None,
            adj_mx_inter: pd.DataFrame = None,
        ) -> gpd.GeoDataFrame:
            """
            Get the criteria for graded territories based on points and polygons.

            Parameters:
            graded_terr (gpd.GeoDataFrame): GeoDataFrame containing graded territories.
            points (gpd.GeoDataFrame): GeoDataFrame containing points of interest.
            polygons (gpd.GeoDataFrame): GeoDataFrame containing polygons for spatial joining.
            citygraph (nx.MultiDiGraph, optional): MultiDiGraph representing the city graph.
            inter (nx.MultiDiGraph, optional): MultiDiGraph representing the interconnection graph.
            r_stops (gpd.GeoDataFrame, optional): GeoDataFrame containing railway stops.
            b_stops (gpd.GeoDataFrame, optional): GeoDataFrame containing bus stops.
            ferry (gpd.GeoDataFrame, optional): GeoDataFrame containing ferry stops.
            aero (gpd.GeoDataFrame, optional): GeoDataFrame containing airports.
            adj_mx_drive (pd.DataFrame, optional): Adjacency matrix for driving.
            adj_mx_inter (pd.DataFrame, optional): Adjacency matrix for public transport.

            Returns:
            gpd.GeoDataFrame: GeoDataFrame with updated criteria based on spatial analysis.
            """
            
            graded_terr.reset_index(drop=True,inplace=True)

            # if there are not enough points for each polygon, we add representative points
            if len(points) != len(adj_mx_drive) or len(points) != len(adj_mx_inter):
                raise ValueError(
                    "Number of points ({}) does not match dimensions of adj_mx_drive ({}) or adj_mx_inter ({}).".format(
                        len(points), len(adj_mx_drive), len(adj_mx_inter)
                    )
                )
            
            self.adj_mx_drive = availability_matrix(citygraph, points,points,local_crs=points.crs) if adj_mx_drive is None else adj_mx_drive
            p = find_median(points, self.adj_mx_drive)
            p_agg = p[p['to_service'] < np.finfo(np.float64).max].copy()
            res = gpd.sjoin(
                p_agg, polygons, how='left', predicate='within'
            ).groupby('index_right').median(['to_service']).reset_index()
            result_df = pd.merge(
                polygons.reset_index(), res, left_index=True, right_on='index_right', how='left'
            ).rename(columns={'to_service': 'in_car'})
            result_df = result_df.drop(columns=['index_right'])

            self.adj_mx_inter = availability_matrix(
                inter, points, points, local_crs=points.crs
            ) if adj_mx_inter is None else adj_mx_inter
            p_inter = find_median(points, self.adj_mx_inter)
            points_inter = p_inter[p_inter['to_service'] < np.finfo(np.float64).max].copy()

            res_inter = gpd.sjoin(
                points_inter, polygons, how="left", predicate="within"
            ).groupby('index_right').median(['to_service']).reset_index()
            
            result_df_inter = pd.merge(
                result_df, res_inter, left_index=True, right_on='index_right', how='left'
            ).drop(columns=['index_right']).rename(columns={'to_service': 'in_inter'})
            graded_gdf = self.weight_territory(graded_terr, r_stops, b_stops, ferry, aero)
            result = self.assign_grades(
                graded_gdf, result_df_inter[[ 'name', 'geometry', 'in_car', 'in_inter']]
            )
            self.criteria = result
            return result
    @staticmethod
    def interpret_gdf(gdf):
        """Interprets geographic accessibility data for each criterion in the criteria DataFrame.

        This method iterates through the criteria DataFrame, extracts relevant weights and quartiles for 
        each criterion, and generates an interpretation of the accessibility based on transport services 
        availability and accessibility quartiles.

        Returns:
            list: A list of tuples, each containing the name of the criterion and its corresponding 
                interpretation as a list of strings.
        """
        interpretation_list = []
        for i,row in gdf.iterrows():

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
    """Generates a textual interpretation of accessibility data based on transport services and 
    accessibility quartiles.

    This method assesses the availability of transport services, as well as the quartiles for car 
    and public access, to create a comprehensive interpretation of accessibility.

    Args:
        grade (int): The grade of the area, used to describe its general quality.
        weight_r_stops (float): Weight indicating the presence of rail stops.
        weight_b_stops (float): Weight indicating the presence of bus stops.
        weight_ferry (float): Weight indicating the presence of ferry services.
        weight_aero (float): Weight indicating the presence of airports.
        car_access_quartile (int): Quartile score for car access (0-4).
        public_access_quartile (int): Quartile score for public access (0-4).

    Returns:
        list: A list of interpretation texts summarizing the accessibility of the area based on 
              transport services and quartiles.
    """
    
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