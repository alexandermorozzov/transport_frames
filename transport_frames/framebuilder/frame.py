import networkx as nx
import geopandas as gpd
import momepy
from shapely.ops import unary_union
from transport_frames.models.graph_validation import ClassifiedEdge
from transport_frames.models.polygon_validation import CountrySchema,CentersSchema,RegionsSchema,PolygonSchema
import osmnx as ox
from transport_frames.utils.helper_funcs import _determine_ref_type
import re
import pandas as pd
import networkit as nk
import numpy as np


class Frame:
    def __init__(self, graph, regions, polygon, centers, max_distance=3000,country_polygon=ox.geocode_to_gdf('RUSSIA')):
        """
        Create and process the frame with the provided graph and spatial data

        Parameters:
        - graph (networkx.Graph): The road network graph to process.
        - regions (GeoDataFrame): GeoDataFrame of regions polygons for assigning region attributes.
        - polygon (GeoDataFrame): GeoDataFrame of a city polygon used for region filtering.
        - centers (GeoDataFrame): GeoDataFrame of region city centers
        - max_distance (int): Maximum distance for assigning city names to nodes. Default is 3000.
        - country_polygon (GeodataFrame): GeoDataFrame of a country polygon used for marking country exits.
        """
        regions = RegionsSchema(regions)
        polygon = PolygonSchema(polygon)
        centers = CentersSchema(centers)
        country_polygon = CountrySchema(country_polygon)
        for d in map(lambda e: e[2], graph.edges(data=True)):
            d = ClassifiedEdge(**d).__dict__
        self.name = polygon.reset_index()['name'][0]
        self.crs = graph.graph['crs']
        self.frame = self.filter_roads(graph)
        self.n, self.e = momepy.nx_to_gdf(self.frame)
        self.n = self.mark_exits(self.n, polygon, regions,country_polygon)  # mark nodes as exits and country_exits
        self.n, self.e, self.frame = self.weigh_roads(self.n, self.e, self.frame) # assign weight to nodes and edges
        self.frame = self.assign_city_names_to_nodes(centers, self.n, self.frame, max_distance=max_distance, local_crs=self.crs) # assign cities to nodes

    def get_geopackage(self):
        n,e = momepy.nx_to_gdf(self.frame)
        combined_gdf = gpd.GeoDataFrame(pd.concat([n, e], ignore_index=True))
        combined_gdf.to_file(f'transport_frame_{self.name}.gpkg', driver='GPKG')
        return  combined_gdf
    
    @staticmethod
    def filter_roads(graph):
        """
        Filter the graph to include only reg_1 and reg_2 roads.

        Parameters:
        - graph (networkx.Graph): The road network graph.

        Returns:
        - networkx.Graph: Filtered graph with only specific roads.
        """
        edges_to_keep = [
            (u, v, k)
            for u, v, k, d in graph.edges(data=True, keys=True)
            if d.get("reg") in ([1, 2])
        ]
        frame = graph.edge_subgraph(edges_to_keep).copy()
        for node, data in frame.nodes(data=True):
            data['nodeID'] = node
        return frame

    def mark_exits(self, gdf_nodes, city_polygon, regions, country_polygon):
            """
            Assign the 'exit' attribute to nodes in a GeoDataFrame based on their intersection with city boundaries.
            
            Parameters:
            - gdf_nodes (GeoDataFrame): GeoDataFrame containing the nodes.
            - city_polygon (GeoDataFrame): GeoDataFrame of city polygons.
            - regions (GeoDataFrame): GeoDataFrame of region polygons.
            
            Returns:
            - GeoDataFrame: Updated GeoDataFrame with the 'exit' attribute.
            """

            city_polygon.sindex
            gdf_nodes.sindex
            regions.sindex
            city_boundary = unary_union(city_polygon.to_crs(gdf_nodes.crs).boundary)
            gdf_nodes.loc[:,'exit'] = gdf_nodes['geometry'].apply(
                lambda point: True if city_boundary.intersects(point.buffer(0.1)) else False
            )

            exits = gdf_nodes[gdf_nodes.exit==1].copy()
            country_boundary = unary_union(country_polygon.to_crs(exits.crs).boundary)
           
            exits.loc[:,'exit_country'] = exits['geometry'].apply(
                lambda point: True if country_boundary.intersects(point.buffer(0.1)) else False)
            gdf_nodes = gdf_nodes.assign(exit_country=exits['exit_country'])
            gdf_nodes['exit_country'] = False
            gdf_nodes.loc[exits.index,'exit_country'] = exits['exit_country'].astype(bool)
            gdf_nodes['exit_country'] = gdf_nodes['exit_country'].fillna(False)
            gdf_nodes = self.add_region_attr(gdf_nodes, regions, city_polygon)
            
            return gdf_nodes
        

    def _filter_polygons_by_buffer(self, gdf_polygons, polygon_buffer):
        """
        Extract and filter region polygons based on a buffer around a given polygon.
        
        Parameters:
        - gdf_polygons (GeoDataFrame): GeoDataFrame of all region polygons.
        - polygon_buffer (GeoDataFrame): GeoDataFrame of the buffer polygon.
        
        Returns:
        - GeoDataFrame: Filtered GeoDataFrame of region polygons.
        """
        gdf_polygons = gdf_polygons.to_crs(self.crs)
        polygon_buffer = polygon_buffer.to_crs(self.crs)
        polygon_buffer = gpd.GeoDataFrame(
            {"geometry": polygon_buffer.buffer(0.1)}, crs=polygon_buffer.crs
        ).to_crs(gdf_polygons.crs)
        gdf_polygons = gpd.overlay(
            gdf_polygons, polygon_buffer, how="difference"
        )
        buffer_polygon = polygon_buffer.buffer(5000)
        filtered_gdf = gdf_polygons[gdf_polygons.intersects(buffer_polygon.unary_union)]
        return filtered_gdf
    

    def add_region_attr(self, n, regions, polygon_buffer):
        """
        Add a 'border_region' attribute to nodes based on their intersection with region polygons.
        
        Parameters:
        - n (GeoDataFrame): Nodes GeoDataFrame with 'exit' attribute.
        - regions (GeoDataFrame): Regions GeoDataFrame with an 'id' column.
        - polygon_buffer (GeoDataFrame): GeoDataFrame of the buffer polygon.
        
        Returns:
        - GeoDataFrame: Updated nodes GeoDataFrame with 'border_region' attribute.
        """
        exits = n[n["exit"] == 1]
        exits = exits.to_crs(self.crs)
        exits.loc[:,"buf"] = exits["geometry"].buffer(1000)
        filtered_regions = self._filter_polygons_by_buffer(regions, polygon_buffer)
        joined_gdf = gpd.sjoin(
            exits.set_geometry("buf"),
            filtered_regions.to_crs(exits.crs),
            how="left",
            predicate="intersects",
        )
        joined_gdf = joined_gdf.drop_duplicates(subset="geometry")
        exits.loc[:, 'border_region'] = joined_gdf['name']
        n.loc[:,"border_region"] = exits["border_region"]
        return n

    @staticmethod
    def _mark_ref_type(n,e,frame):

        n["ref"] = None
        ref_edges = e[e["ref"].notna()]

        for idx, node in n.iterrows():

            if node["exit"] == 1:
                point = node.geometry
                distances = ref_edges.geometry.distance(point)
                if not distances.empty:
                    nearest_edge = ref_edges.loc[distances.idxmin()]
                    ref_value = nearest_edge["ref"]
                    if isinstance(ref_value, list):
                        ref_value = tuple(ref_value)
                    if isinstance(ref_value, str):
                        ref_value = (ref_value,)
                    n.at[idx, "ref"] = ref_value
                    n.at[idx, "ref_type"] = _determine_ref_type(ref_value)
        n = n.set_index("nodeID")
        mapping = {node: data["nodeID"] for node, data in frame.nodes(data=True)}
        n.reset_index(inplace=True)
        frame = nx.relabel_nodes(frame, mapping)
        return n,e,frame


    @staticmethod
    def _get_weight(start, end, exit):
        """
        Calculate the weight based on the type of start and end references and exit status.

        Parameters:
        start (float): Reference type of the start node.
        end (float): Reference type of the end node.
        exit (int): Exit status (1 if exit, else 0).

        Returns:
        float: Calculated weight based on the provided matrix.
        """
        dict = {1.1: 0, 1.2: 1, 1.3: 2, 2.1: 3, 2.2: 4, 2.3: 5}
        if exit == 1:
            matrix = [
                [0.12, 0.12, 0.12, 0.12, 0.12, 0.12],  # 2.1.1
                [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],  # 2.1.2
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08],  # 2.1.3
                [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],  # 2.2.1
                [0.06, 0.06, 0.06, 0.06, 0.06, 0.06],  # 2.2.2
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 2.2.3
            ]
        else:

            matrix = [
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08],  # 2.1.1
                [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],  # 2.1.2
                [0.06, 0.06, 0.06, 0.06, 0.06, 0.06],  # 2.1.3
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 2.2.1
                [0.04, 0.04, 0.04, 0.04, 0.04, 0.04],  # 2.2.2
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # 2.2.3
            ]
        return matrix[dict[end]][dict[start]]


    @staticmethod
    def weigh_roads(n,e,frame):
        """
        Calculate and normalize the weights of roads between exits in a road network.

        Parameters:
        frame (networkx.Graph): The road network graph where nodes represent intersections or exits
                                and edges represent road segments with 'time_min' as a weight attribute.

        Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with frame edges with corresponding weights and normalized weights.
        """
        n,e,frame = Frame._mark_ref_type(n, e, frame) 
        e = e.loc[e.groupby(["node_start", "node_end"])["time_min"].idxmin()]
        e["weight"] = 0.0
        n["weight"] = 0.0
        exits = n[n["exit"] == 1]
        for i1, start_node in exits.iterrows():
            for i2, end_node in exits.iterrows():
                if i1 == i2:
                    continue
                if (
                    pd.notna(start_node["border_region"])
                    and start_node["border_region"] == end_node["border_region"]
                ):
                    continue
                if start_node.geometry.buffer(15000).intersects(
                    end_node.geometry.buffer(15000)
                ) and (
                    pd.isna(start_node["exit_country"]) == pd.isna(end_node["exit_country"])
                ):
                    continue
                if start_node["exit_country"] == 1 and end_node["exit_country"] == 1:
                    continue

                weight = Frame._get_weight(
                    start_node["ref_type"], end_node["ref_type"], end_node["exit_country"]
                )

                try:
                    path = nx.astar_path(frame, i1, i2, weight='time_min')
                except nx.NetworkXNoPath:
                    continue

                for j in range(len(path) - 1):
                    n.loc[(n["nodeID"] == path[j]), "weight"] += weight
                    e.loc[
                        (e["node_start"] == path[j]) & (e["node_end"] == path[j + 1]),
                        "weight",
                    ] += weight
                n.loc[(n["nodeID"] == path[j + 1]), "weight"] += weight

        n['weight'] = round(n.weight,3)
        n.drop(columns=['ref','ref_type','border_region'], inplace=True)
        e.drop(columns=['ref','weight','highway','maxspeed'], inplace=True)
        for u, v, key, data in frame.edges(keys=True, data=True):  
            if 'ref' in data:
                del data['ref']
            if 'highway' in data:
                del data['highway']
            if 'maxspeed' in data:
                del data['maxspeed']

        for i, (node, data) in enumerate(frame.nodes(data=True)):
            data['exit'] = n.iloc[i]["exit"]
            data['exit_country'] = n.iloc[i]["exit_country"]
            data["weight"] = n.iloc[i]["weight"]
                
        return n,e,frame


    @staticmethod
    def assign_city_names_to_nodes(
        points,
        nodes,
        graph,
        name_attr="city_name",
        node_id_attr="nodeID",
        name_col="name",
        max_distance=3000,
        local_crs=3857
    ):

        """
        Присваивает имена точек для проекции узлам графа на основе пространственного ближайшего соседства.

        Parameters:
        points (GeoDataFrame): Геоданные с точками.
        nodes (GeoDataFrame): Геоданные с узлами графа.
        graph (nx.Graph): Граф, к узлам которого будут добавлены имена точек.
        name_attr (str): Название атрибута для имени точек, которое будет добавлено к узлам графа.
        node_id_attr (str): Название атрибута ID узла в графе.
        name_col (str): Название колонки с именами точек в points.

        Returns:
        nx.Graph
        """
        G = graph.copy()
        nodes.to_crs(local_crs,inplace=True)
        points.to_crs(local_crs,inplace=True)
        project_roads_city = gpd.sjoin_nearest(
            points, nodes, how="left", distance_col="distance", max_distance=max_distance
        ).to_crs(graph.graph['crs'])
        flag = 0
        missed_cities = set()
        for enum, index in enumerate(project_roads_city[node_id_attr].values):
            city_name = project_roads_city.loc[enum, name_col]
            for _, d in G.nodes(data=True):
                if d.get(node_id_attr) == index:
                    d[name_attr] = city_name
                    flag = 1
                else:
                    missed_cities.add(city_name)
        if flag == 0:
            print('no cities can be assigned nodes')

                    
        return G

    


    def grade_territory(self, gdf_poly, include_priority=True):
        """
        Grades territories based on their distances to reg1, reg2 nodes,edges and train stations.

        Parameters:
            gdf_poly (GeoDataFrame): A GeoDataFrame containing the polygons of the territories to be graded.
            frame
        (networkx.MultiDiGraph): A MultiDiGraph representing the transportation network.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the graded territories with added 'grade' column.
        """

        nodes, edges = momepy.nx_to_gdf(
            self.frame, points=True, lines=True, spatial_weights=False
        )

        poly = gdf_poly.copy().to_crs(nodes.crs)

        reg1_points = nodes[nodes["reg_1"] == 1]
        reg2_points = nodes[nodes["reg_2"] == 1]
        priority_reg1_points = nodes[
            (nodes["weight"] > np.percentile(nodes[nodes["weight"] != 0]["weight"], 60))
            & (nodes["reg_1"] == 1)
        ]
        priority_reg2_points = nodes[
            (nodes["weight"] > np.percentile(nodes[nodes["weight"] != 0]["weight"], 60))
            & (nodes["reg_2"] == 1)
        ]

        min_distance = lambda polygon, points: points.distance(polygon).min()
        poly["dist_to_reg1"] = poly.geometry.apply(
            lambda x: min_distance(x, reg1_points.geometry)
        )
        poly["dist_to_reg2"] = poly.geometry.apply(
            lambda x: min_distance(x, reg2_points.geometry)
        )
        poly["dist_to_edge"] = poly.geometry.apply(
            lambda x: min_distance(x, edges.geometry)
        )
        poly["dist_to_priority_reg1"] = poly.geometry.apply(
            lambda x: min_distance(x, priority_reg1_points.geometry)
        )
        poly["dist_to_priority_reg2"] = poly.geometry.apply(
            lambda x: min_distance(x, priority_reg2_points.geometry)
        )

        poly["grade"] = poly.apply(grade_polygon, axis=1, args=(include_priority,))
        output = poly[['name', 'layer', 'status', 'geometry', 'grade']].copy()
        return output

def _determine_ref_type(ref):
    """Converts ref string to numeric type"""
    patterns = {
        1.1: r"М-\d+",
        1.2: r"Р-\d+",
        1.3: r"А-\d+",
        2.1: r"..Р-\d+",
        2.2: r"..А-\d+",
    }
    for value in ref:
        for ref_type, pattern in patterns.items():
            if re.match(pattern, value):
                return ref_type
    return 2.3

@staticmethod
def grade_polygon(row, include_priority=False):
        """
        Determines the grade of a territory based on its distance to features.

        Parameters:
            row (Series): A pandas Series representing a single row of a GeoDataFrame.

        Returns:
            float: The grade of the territory.
        """
        dist_to_reg1 = row["dist_to_reg1"]
        dist_to_reg2 = row["dist_to_reg2"]
        dist_to_edge = row["dist_to_edge"]
        dist_to_priority1 = row["dist_to_priority_reg1"]
        dist_to_priority2 = row["dist_to_priority_reg2"]

        # below numbers measured in thousands are representes in meters eg 5_000 meters ie 5km
        if include_priority and dist_to_priority1 < 5000:
            grade = 5
        elif (
            include_priority
            and dist_to_priority1 < 10000
            and dist_to_priority2 < 5000
            or dist_to_reg1 < 5000
        ):
            grade = 4.5
        elif dist_to_reg1 < 10000 and dist_to_reg2 < 5000:
            grade = 4.0
        elif include_priority and dist_to_priority1 < 100000 and dist_to_priority2 < 5000:
            grade = 3.5
        elif dist_to_reg1 < 100000 and dist_to_reg2 < 5000:
            grade = 3.0
        elif dist_to_reg1 > 100000 and dist_to_reg2 < 5000:
            grade = 2.0
        elif dist_to_reg2 > 5000 and dist_to_reg1 > 100000 and dist_to_edge < 5000:
            grade = 1.0
        else:
            grade = 0.0

        return grade