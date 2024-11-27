import geopandas as gpd
import networkx as nx
import osmnx as ox
from loguru import logger
from shapely import Polygon
from shapely.geometry import LineString
import momepy
import pandas as pd
import numpy as np
from transport_frames.utils.helper_funcs import buffer_and_transform_polygon, convert_geometry_from_wkt
from transport_frames.graphbuilder.road_classifier import RoadClassifier
from transport_frames.models.graph_validation import GraphNode, GraphEdge, GraphMetadata
from transport_frames.models.polygon_validation import PolygonSchema
import warnings
import iduedu

# Suppress UserWarnings
warnings.simplefilter("ignore", UserWarning)

class Graph:
    """
    A class to represent and manipulate a graph of road networks.
    
    Attributes:
    graph (nx.MultiDiGraph): The road network graph.
    crs (int): Coordinate reference system, default is 3857.
    polygon (gpd.GeoDataFrame): The geographic boundary for the graph.
    """

    def __init__(self, nx_graph: nx.MultiDiGraph, crs: int = 3857, polygon= None):
        """
        Initialize the Graph object.
        
        Parameters:
        nx_graph (nx.MultiDiGraph): The networkx graph object representing the road network.
        crs (int): Coordinate reference system, default is 3857.
        """
        self.graph = nx_graph
        self.crs = crs
        self.polygon = polygon
        self._prepare_attrs()
        self.validate_graph(self.graph)
        GraphMetadata(**self.graph.graph)
        self.classify_roads()
        

    @classmethod
    def from_polygon(cls, polygon: gpd.GeoDataFrame, crs: int = 3857):
        """
        Create a Graph instance by downloading data within a specified polygon.

        Parameters:
        polygon (gpd.GeoDataFrame): The polygon representing the area of interest.
        crs (int): Coordinate reference system, default is 3857.

        Returns:
        Graph: A Graph object created from the polygon.
        """
        
        
        polygon = PolygonSchema(polygon)
        buffer_polygon = buffer_and_transform_polygon(polygon, crs)

        custom_filter = ("['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|"
                         "residential|motorway_link|trunk_link|primary_link|secondary_link|"
                         "tertiary_link|living_street']")

        logger.info("Downloading the graph from OSM...")
        nx_graph = ox.graph_from_polygon(
            buffer_polygon,
            network_type="drive",
            custom_filter=custom_filter,
            truncate_by_edge=True,
        )
        graph_instance = cls(nx_graph, crs, polygon)
        graph_instance.polygon = polygon
        return graph_instance

    @classmethod
    def from_polygon_iduedu(cls,polygon:gpd.GeoDataFrame,local_crs: int, buffer=3000):
        polygon.to_crs(local_crs,inplace=True)
        polygon_with_buf = gpd.GeoDataFrame([{'geometry': polygon.loc[0].geometry.buffer(buffer)}], crs=local_crs)
        lo_polygon_geometry_with_buf = polygon_with_buf.to_crs(4326).geometry.unary_union
        g_don = iduedu.get_drive_graph(polygon = lo_polygon_geometry_with_buf, additional_edgedata=['highway', 'maxspeed', 'reg', 'ref', 'name'])
        
        graph_instance = cls(g_don, local_crs, polygon)
        graph_instance.polygon = polygon
        return graph_instance

    @classmethod
    def from_geocode(cls, geocode: str, by_osmid=False, crs: int = 3857):
        """
        Create a Graph instance by downloading data for a region based on its name or ID.

        Parameters:
        geocode (str): The name or ID of the region.
        by_osmid (bool): Whether to use the OpenStreetMap ID for geocoding. Default is False.
        crs (int): Coordinate reference system, default is 3857.

        Returns:
        Graph: A Graph object created from the geocode.
        """
        polygon = ox.geocode_to_gdf(geocode, by_osmid=by_osmid)
        return cls.from_polygon(polygon, crs)
    

    @staticmethod
    def validate_graph(graph) -> nx.MultiDiGraph:
        """Validates copy graph, according to ``GraphEdge`` and ``GraphNode`` classes"""
        for d in map(lambda e: e[2], graph.edges(data=True)):
            d = GraphEdge(**d).__dict__
        for d in map(lambda n: n[1], graph.nodes(data=True)):
            d = GraphNode(**d).__dict__
        return graph

        

    def classify_roads(self):
        """
        Classify roads in the graph based on their type and assign appropriate attributes.
        """
        for _, _, data in self.graph.edges(data=True):
            data['reg'] = RoadClassifier.determine_reg(data.get('ref'), data.get('highway'))
            data['maxspeed'] = RoadClassifier.get_max_speed(data.get('highway'))
            data['time_min'] = round(data['length_meter']/ data['maxspeed']/60, 3)
            data['type'] = 'car'

        for node in self.graph.nodes:
            self.graph.nodes[node]["reg_1"] = False
            self.graph.nodes[node]["reg_2"] = False

        for u, v, data in self.graph.edges(data=True):
            if data.get("reg") == 1:
                self.graph.nodes[u]["reg_1"] = True
                self.graph.nodes[v]["reg_1"] = True
            elif data.get("reg") == 2:
                self.graph.nodes[u]["reg_2"] = True
                self.graph.nodes[v]["reg_2"] = True

    def _prepare_attrs(self):
        """
        Prepare or update attributes of the graph after it has been created.
        This method converts node labels, updates geometries, and sets up necessary attributes.
        """
        logger.info("Preparing the graph...")

        self.graph.graph["approach"] = "primal"
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        nodes, edges = momepy.nx_to_gdf(self.graph, points=True, lines=True, spatial_weights=False)
        edges = _add_geometry_to_edges(nodes, edges)
        nodes = nodes.to_crs(epsg=self.crs)
        edges = edges.to_crs(epsg=self.crs)
        self.edges = edges
        if self.polygon is not None:
            edges = _update_edges_with_geometry(edges, self.polygon, self.crs)

        nodes_coord = _update_nodes_with_geometry(edges, {})

        edges = edges[
            [
                "highway",
                "node_start",
                "node_end",
                "geometry",
                "ref"            ]
        ]
        edges.loc[:, "geometry"] = edges["geometry"].apply(
            lambda x: LineString(
                [tuple(round(c, 6) for c in n) for n in x.coords] if x else None
            )
        )

        # Apply the transformation on the 'ref' column and assign it back using .loc[]
        edges.loc[:, 'ref'] = edges['ref'].apply(lambda x: np.nan if not isinstance(x, list) and pd.isna(x) else x)
        self.graph = _create_graph(edges, nodes_coord)
        nx.set_node_attributes(self.graph, nodes_coord)
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.graph = convert_geometry_from_wkt(self.graph)
        self.graph.graph["crs"] = self.crs
        self.graph.graph["approach"] = "primal"
        self.graph.graph["graph_type"] = "car graph"

        logger.info("Graph is ready!")

def _update_edges_with_geometry(edges: gpd.GeoDataFrame, polygon: Polygon, crs):
    """
    Update edge geometries based on intersections with the city boundary.

    Parameters:
    edges (gpd.GeoDataFrame): The edges GeoDataFrame with existing geometries.
    polygon (gpd.GeoDataFrame): The polygon representing the city boundary.
    crs (int): The coordinate reference system to use.

    Returns:
    gpd.GeoDataFrame: Updated edges with geometries trimmed to the city boundary.
    """
    city_transformed = polygon.to_crs(edges.crs)
    edges["intersections"] = edges["geometry"].intersection(
        city_transformed.unary_union
    )
    edges["geometry"] = edges["intersections"]

    edges.drop(columns=["intersections"], inplace=True)
    edges = edges.explode(index_parts=True)
    edges = edges[~edges["geometry"].is_empty]
    edges = edges[edges["geometry"].geom_type == "LineString"]
    return edges





def _update_nodes_with_geometry(edges, nodes_coord):
    """
    Update node coordinates based on new edge geometries.

    Parameters:
    edges (gpd.GeoDataFrame): The edges GeoDataFrame.
    nodes_coord (dict): A dictionary of node coordinates.

    Returns:
    dict: Updated dictionary of node coordinates.
    """
    for _, row in edges.iterrows():
        start_node = row["node_start"]
        end_node = row["node_end"]
        if start_node not in nodes_coord:
            nodes_coord[start_node] = {
                "x": row["geometry"].coords[0][0],
                "y": row["geometry"].coords[0][1],
            }
        if end_node not in nodes_coord:
            nodes_coord[end_node] = {
                "x": row["geometry"].coords[-1][0],
                "y": row["geometry"].coords[-1][1],
            }
    return nodes_coord

def _create_graph(edges, nodes_coord):
    """
    Create a graph based on edges and node coordinates.

    Parameters:
    edges (gpd.GeoDataFrame): The edges with their attributes and geometries.
    nodes_coord (dict): A dictionary containing node coordinates.

    Returns:
    nx.MultiDiGraph: The constructed graph.
    """
    G = nx.MultiDiGraph()
    for _, edge in edges.iterrows():
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geom = (
            LineString(
                (
                    [
                        (nodes_coord[p1]["x"], nodes_coord[p1]["y"]),
                        (nodes_coord[p2]["x"], nodes_coord[p2]["y"]),
                    ]
                )
            )
            if not edge.geometry
            else edge.geometry
        )
        length = round(geom.length, 3)
        G.add_edge(
            p1,
            p2,
            length_meter=length,
            geometry=str(geom),
            highway=edge.highway,
            ref=edge.ref,
        )
    return G

def _add_geometry_to_edges(nodes, edges):
    """
    Add geometry to edges based on node coordinates.

    Parameters:
    nodes (gpd.GeoDataFrame): The nodes with their coordinates.
    edges (gpd.GeoDataFrame): The edges with their start and end nodes.

    Returns:
    gpd.GeoDataFrame: The edges with added geometries.
    """
    node_coords = {
        node["nodeID"]: (node["x"], node["y"]) for _, node in nodes.iterrows()
    }

    for idx, edge in edges.iterrows():
        if edge["geometry"] is None:
            start_coords = node_coords[edge["node_start"]]
            end_coords = node_coords[edge["node_end"]]
            line_geom = LineString([start_coords, end_coords])
            edges.at[idx, "geometry"] = line_geom
    return edges