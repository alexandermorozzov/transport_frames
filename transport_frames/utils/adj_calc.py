from abc import ABC
from typing import Generic, TypeVar
import geopandas as gpd
from pydantic import BaseModel, ConfigDict
from shapely.geometry.base import BaseGeometry


T = TypeVar("T")


class BaseRow(BaseModel, ABC):
    """Provides an abstract for data validation in GeoDataFrame.
    Generics must be inherited from this base class.

    The inherited class also can be configured to provide default column values to avoid None and NaN"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    geometry: BaseGeometry
    index: int
    """Index can be override but should not be set by default"""


class GeoDataFrame(gpd.GeoDataFrame, BaseModel, Generic[T]):
    """Basically a geopandas GeoDataFrame, but with Generic[T].
    Provide a BaseRow inherited class to automatically validate data on init."""

    @property
    def generic(self):
        # pydantic is only needed to access generic class
        return self.__pydantic_generic_metadata__["args"][0]

    def __init__(self, data, *args, **kwargs):
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_
        """
        generic_class = self.generic
        assert issubclass(generic_class, BaseRow), "Generic should be inherited from BaseRow"
        # if data is not a GeoDataFrame, we create it ourselves
        if not isinstance(data, gpd.GeoDataFrame):
            data = gpd.GeoDataFrame(data, *args, **kwargs)
        # next we create list of dicts from BaseRow inherited class
        rows: list[dict] = [generic_class(index=i, **data.loc[i].to_dict()).__dict__ for i in data.index]
        super().__init__(rows)
        # and finally return index to where it belongs
        if "index" in self.columns:
            self.index = self["index"]
            self.drop(columns=["index"], inplace=True)
        index_name = data.index.name
        self.index.name = index_name
        self.set_geometry("geometry", inplace=True)
        # and also set crs
        self.crs = kwargs["crs"] if "crs" in kwargs else data.crs





from typing import Literal

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from pydantic import BaseModel, Field, InstanceOf, field_validator
from shapely import LineString, MultiPolygon, Point, Polygon, line_locate_point
from shapely.ops import linemerge, nearest_points, split



OX_CRS = 4326
METERS_IN_KILOMETER = 1000
MINUTES_IN_HOUR = 60


class GraphNode(BaseModel):
    x: float
    y: float


class GraphEdge(BaseModel):
    geometry: InstanceOf[LineString] = None
    weight: float = Field(ge=0)
    transport_type: Literal["walk", "drive", "subway", "tram", "bus", "trolleybus"]


class CityRow(BaseRow):
    geometry: Polygon | MultiPolygon


class GraphGenerator(BaseModel):
    territory: InstanceOf[GeoDataFrame[CityRow]]
    """City geometry or geometries, may contain blocks or boundaries of the city"""
    overpass_url: str = "http://lz4.overpass-api.de/api/interpreter"
    """Overpass url used in OSM queries"""
    speed: dict[str, int] = {"walk": 4, "drive": 25, "subway": 12, "tram": 15, "trolleybus": 12, "bus": 17}
    """Average transport type speed in km/h"""
    waiting_time: dict[str, int] = {
        "subway": 5,
        "tram": 5,
        "trolleybus": 5,
        "bus": 5,
    }
    """Average waiting time in min"""

    @field_validator("territory", mode="before")
    def cast_territory(gdf):
        if not isinstance(gdf, GeoDataFrame[CityRow]):
            gdf = GeoDataFrame[CityRow](gdf)
        return gdf

    @field_validator("territory", mode="after")
    def union_territory(gdf):
        return GeoDataFrame[CityRow]([{"geometry": gdf.geometry.unary_union.convex_hull}]).set_crs(gdf.crs)

    @staticmethod
    def to_graphml(graph: nx.MultiDiGraph, file_path: str):
        """Save graph as OX .graphml"""
        ox.save_graphml(graph, file_path)

    @staticmethod
    def from_graphml(file_path: str):
        """Load graph from OX .graphml"""
        return ox.load_graphml(file_path)

    @property
    def local_crs(self):
        return self.territory.crs

    @classmethod
    def plot(cls, graph: nx.MultiDiGraph):
        _, edges = ox.graph_to_gdfs(graph)
        edges.plot(column="transport_type", legend=True).set_axis_off()

    def _get_speed(self, transport_type: str):
        """Return transport type speed in meters per minute"""
        return METERS_IN_KILOMETER * self.speed[transport_type] / MINUTES_IN_HOUR

    def _get_basic_graph(self, network_type: Literal["walk", "drive"]):
        """Returns walk or drive graph for the city geometry"""
        speed = self._get_speed(network_type)
        G = ox.graph_from_polygon(polygon=self.territory.to_crs(OX_CRS).unary_union, network_type=network_type)
        G = ox.project_graph(G, to_crs=self.local_crs)
        for edge in G.edges(data=True):
            _, _, data = edge
            length = data["length"]
            data["weight"] = length / speed
            data["transport_type"] = network_type
        G = ox.project_graph(G, self.local_crs)
        print(f"Graph made for '{network_type}' network type")
        return G

    def _get_routes(
        self, bounds: pd.DataFrame, public_transport_type: Literal["subway", "tram", "trolleybus", "bus"]
    ) -> pd.DataFrame:
        """Returns OSM routes for the given geometry shapely geometry bounds and given transport type"""
        bbox = f"{bounds.loc[0,'miny']},{bounds.loc[0,'minx']},{bounds.loc[0,'maxy']},{bounds.loc[0,'maxx']}"
        tags = f"'route'='{public_transport_type}'"
        overpass_query = f"""
    [out:json];
            (
                relation({bbox})[{tags}];
            );
    out geom;
    """
        result = requests.get(self.overpass_url, params={"data": overpass_query})
        json_result = result.json()["elements"]
        return pd.DataFrame(json_result)

    @staticmethod
    def _coordinates_to_linestring(coordinates: list[dict[str, float]]) -> LineString:
        """For given route coordinates dicts returns a concated linestring"""
        points = []
        for point in coordinates:
            points.append(Point(point["lon"], point["lat"]))
        linestring = LineString(points)
        return linestring

    def _ways_to_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Returns GeoDataFrame for the given route ways, converting way's coordinates to linestring"""
        copy = df.copy()
        copy["geometry"] = df["coordinates"].apply(lambda x: self._coordinates_to_linestring(x))
        return gpd.GeoDataFrame(copy, geometry=copy["geometry"]).set_crs(epsg=OX_CRS).to_crs(self.local_crs)

    def _nodes_to_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Returns GeoDataFrame for the given route nodes, converting lon and lat columns to geometry column and local CRS"""
        return (
            gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))
            .set_crs(epsg=OX_CRS)
            .to_crs(self.local_crs)
        )

    def _graph_from_route(self, nodes: pd.DataFrame, ways: pd.DataFrame, transport_type: str) -> list[nx.MultiGraph]:
        """Get graph for the given route nodes and ways"""
        linestring = linemerge(list(ways["geometry"]))
        nodes = nodes.copy()
        nodes["geometry"] = nodes["geometry"].apply(lambda x: nearest_points(linestring, x)[0])
        nodes["distance"] = nodes["geometry"].apply(lambda x: line_locate_point(linestring, x))
        nodes = nodes.loc[nodes["geometry"].within(self.territory.unary_union)]
        sorted_nodes = nodes.sort_values(by="distance").reset_index()
        sorted_nodes["hash"] = sorted_nodes["geometry"].apply(lambda x: f"{transport_type}_{hash(x)}")
        G = nx.MultiDiGraph()
        for index in list(sorted_nodes.index)[:-1]:
            n1 = sorted_nodes.loc[index]
            n2 = sorted_nodes.loc[index + 1]
            d = n2["distance"] - n1["distance"]
            id1 = n1["hash"]  # hash(n1['geometry'])
            id2 = n2["hash"]  # hash(n1['geometry'])
            speed = self._get_speed(transport_type)
            G.add_edge(id1, id2, weight=d / speed, transport_type=transport_type)
            G.nodes[id1]["x"] = n1["geometry"].x
            G.nodes[id1]["y"] = n1["geometry"].y
            G.nodes[id2]["x"] = n2["geometry"].x
            G.nodes[id2]["y"] = n2["geometry"].y
        return G

    def _get_pt_graph(self, pt_type: Literal["subway", "tram", "trolleybus", "bus"]) -> list[nx.MultiGraph]:
        """Get public transport routes graphs for the given transport_type"""
        routes: pd.DataFrame = self._get_routes(self.territory.to_crs(OX_CRS).bounds, pt_type)
        graphs = []
        for i in routes.index:
            df = pd.DataFrame(routes.loc[i, "members"])
            nodes_df = df.loc[lambda x: x["type"] == "node"].copy()
            ways_df = df.loc[lambda x: x["type"] == "way"].copy().rename(columns={"geometry": "coordinates"})
            if len(nodes_df) == 0 or len(ways_df) == 0:
                continue
            nodes_gdf = self._nodes_to_gdf(nodes_df)
            ways_gdf = self._ways_to_gdf(ways_df)
            graphs.append(self._graph_from_route(nodes_gdf, ways_gdf, pt_type))
        graph = None
        if len(graphs) > 0:
            graph = nx.compose_all(graphs)
            graph.graph["crs"] = self.local_crs
        print(f"Graph made for '{pt_type}'")
        return graph

    @staticmethod
    def validate_graph(graph) -> nx.MultiDiGraph:
        """Returns validated copy of the graph, according to ```GraphEdge``` and ```GraphNode``` classes"""
        graph = graph.copy()
        for d in map(lambda e: e[2], graph.edges(data=True)):
            d = GraphEdge(**d).__dict__
        for d in map(lambda n: n[1], graph.nodes(data=True)):
            d = GraphNode(**d).__dict__
        return graph

    def get_graph(self, graph_type: Literal["intermodal", "walk", "drive"]):
        """Returns intermodal graph for the city geometry bounds"""
        if graph_type != "intermodal":
            graph = self._get_basic_graph(graph_type)
            return self.validate_graph(graph)

        walk_graph: nx.MultiDiGraph = self._get_basic_graph("walk")
        walk_nodes, _ = ox.graph_to_gdfs(walk_graph)

        pt_types: list[str] = ["bus", "trolleybus", "tram", "subway"]
        pt_graphs: list[nx.MultiDiGraph] = list(map(lambda t: self._get_pt_graph(t), pt_types))
        pt_graphs = list(filter(lambda g: g is not None, pt_graphs))
        pt_graph = nx.compose_all(pt_graphs)
        pt_graph.crs = self.local_crs
        pt_nodes, _ = ox.graph_to_gdfs(pt_graph)

        intermodal_graph = nx.compose(walk_graph, pt_graph)
        pt_to_walk = pt_nodes.sjoin_nearest(walk_nodes, how="left", distance_col="distance")
        for i in pt_to_walk.index:
            gs = pt_to_walk.loc[i]
            transport_node = i
            walk_node = gs["index_right"]
            distance = gs["distance"]
            speed = self._get_speed("walk")
            weight = distance / speed
            intermodal_graph.add_edge(transport_node, walk_node, weight=weight, transport_type="walk")
            intermodal_graph.add_edge(walk_node, transport_node, weight=weight + 5, transport_type="walk")
        intermodal_graph.graph["crs"] = self.local_crs

        return self.validate_graph(intermodal_graph)



from typing import Any

import geopandas as gpd
import networkit as nk
import networkx as nx
import pandas as pd
from pydantic import BaseModel, InstanceOf, field_validator
from shapely import Polygon
from enum import Enum




class AdjacencyMethod(Enum):
    SPSP = "SPSP"
    OPTIMIZED_SPSP = "Optimized SPSP"


class BlockRow(BaseRow):
    geometry: Polygon


class AdjacencyCalculator(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Class Accessibility calculates accessibility matrix between city blocks.
    It takes a lot of RAM to calculate one since we have thousands of city blocks.

    Methods
    -------
    get_matrix
    """

    blocks: GeoDataFrame[BlockRow]
    graph: InstanceOf[nx.MultiDiGraph]

    @field_validator("blocks", mode="before")
    def validate_blocks(gdf):
        if not isinstance(gdf, GeoDataFrame[BlockRow]):
            gdf = GeoDataFrame[BlockRow](gdf)
        return gdf

    @field_validator("graph", mode="before")
    def validate_graph(graph):
        assert "crs" in graph.graph, 'Graph should contain "crs" property similar to GeoDataFrame'
        return GraphGenerator.validate_graph(graph)

    def model_post_init(self, __context: Any) -> None:
        assert self.blocks.crs == self.graph.graph["crs"], "Blocks CRS should match graph CRS"
        return super().model_post_init(__context)

    @staticmethod
    def _get_nx2nk_idmap(graph: nx.Graph) -> dict:  # TODO: add typing for the dict
        """
        This method gets ids from nx graph to place as attribute in nk graph

        Attributes
        ----------
        graph: networkx graph

        Returns
        -------
        idmap: dict
            map of old and new ids
        """

        idmap = dict((id, u) for (id, u) in zip(graph.nodes(), range(graph.number_of_nodes())))
        return idmap

    @staticmethod
    def _get_nk_attrs(graph: nx.Graph) -> dict:  # TODO: add typing for the dict
        """
        This method gets attributes from nx graph to set as attributes in nk graph

        Attributes
        ----------
        graph: networkx graph

        Returns
        -------
        idmap: dict
            map of old and new attributes
        """

        attrs = dict(
            (u, {"x": d[-1]["x"], "y": d[-1]["y"]})
            for (d, u) in zip(graph.nodes(data=True), range(graph.number_of_nodes()))
        )
        return attrs

    @classmethod
    def _convert_nx2nk(  # pylint: disable=too-many-locals,invalid-name
        cls, graph_nx: nx.MultiDiGraph, idmap: dict | None = None, weight: str = "weight"
    ) -> nk.Graph:
        """
        This method converts `networkx` graph to `networkit` graph to fasten calculations.

        Attributes
        ----------
        graph_nx: networkx graph
        idmap: dict
            map of ids in old nx and new nk graphs
        weight: str
            value to be used as a edge's weight

        Returns
        -------
        graph_nk: nk.Graph
            the same graph but now presented in is `networkit` package Graph class.

        """

        if not idmap:
            idmap = cls._get_nx2nk_idmap(graph_nx)
        n = max(idmap.values()) + 1
        edges = list(graph_nx.edges())

        graph_nk = nk.Graph(n, directed=graph_nx.is_directed(), weighted=True)
        for u_, v_ in edges:
            u, v = idmap[u_], idmap[v_]
            d = dict(graph_nx[u_][v_])
            if len(d) > 1:
                for d_ in d.values():
                    v__ = graph_nk.addNodes(2)
                    u__ = v__ - 1
                    w = round(d[weight], 1) if weight in d else 1
                    graph_nk.addEdge(u, v, w)
                    graph_nk.addEdge(u_, u__, 0)
                    graph_nk.addEdge(v_, v__, 0)
            else:
                d_ = list(d.values())[0]
                w = round(d_[weight], 1) if weight in d_ else 1
                graph_nk.addEdge(u, v, w)

        return graph_nk

    def _get_nk_distances(
        self, nk_dists: nk.base.Algorithm, loc: pd.Series  # pylint: disable=c-extension-no-member
    ) -> pd.Series:
        """
        This method calculates distances between blocks using nk SPSP algorithm.
        The function is called inside apply function.

        Attributes
        ----------
        nk_dists: nk.base.Algorithm
            Compressed nk graph to compute distances between nodes using SPSP algorithm
        loc: pd.Series
            Row in the df

        Returns
        -------
        pd.Series with computed distances
        """

        target_nodes = loc.index
        source_node = loc.name
        distances = [nk_dists.getDistance(source_node, node) for node in target_nodes]

        return pd.Series(data=distances, index=target_nodes)

    @staticmethod
    def get_distances(graph, df):
        source_nodes = df.index
        target_nodes = df.columns
        spsp = nk.distance.SPSP(graph, source_nodes, target_nodes)
        spsp.run()
        return {(sn, tn): spsp.getDistance(sn, tn) for sn in source_nodes for tn in target_nodes}

    def get_dataframe(self, method: AdjacencyMethod = AdjacencyMethod.OPTIMIZED_SPSP) -> pd.DataFrame:
        """
        This methods runs graph to matrix calculations

        Returns
        -------
        accs_matrix: pd.DataFrame
            An accessibility matrix that contains time between all blocks in the city
        """

        graph_nx = nx.convert_node_labels_to_integers(self.graph)
        graph_nk = self._convert_nx2nk(graph_nx)

        graph_df = pd.DataFrame.from_dict(dict(graph_nx.nodes(data=True)), orient="index")
        graph_gdf = gpd.GeoDataFrame(
            graph_df, geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"]), crs=self.blocks.crs.to_epsg()
        )

        blocks = self.blocks.copy()
        blocks.geometry = blocks.geometry.representative_point()
        from_blocks = graph_gdf["geometry"].sindex.nearest(blocks["geometry"], return_distance=False, return_all=False)
        accs_matrix = pd.DataFrame(0, index=from_blocks[1], columns=from_blocks[1])

        if method == AdjacencyMethod.SPSP:
            nk_dists = nk.distance.SPSP(  # pylint: disable=c-extension-no-member
                graph_nk, sources=accs_matrix.index.values
            ).run()

            accs_matrix = accs_matrix.apply(lambda x: self._get_nk_distances(nk_dists, x), axis=1)

        if method == AdjacencyMethod.OPTIMIZED_SPSP:
            k_rows = 100
            distances = {}

            for i in range(0, len(accs_matrix.index), k_rows):
                sub_df = accs_matrix.iloc[i : i + k_rows]
                distances.update(self.get_distances(graph_nk, sub_df))

            accs_matrix = accs_matrix.apply(
                lambda x: pd.Series(data=[distances[x.name, i] for i in x.index], index=x.index), axis=1
            )

        accs_matrix.index = blocks.index
        accs_matrix.columns = blocks.index

        # bug fix in city block's closest node is no connecte to actual transport infrastructure
        print('new version')
        # accs_matrix[accs_matrix > 500] = accs_matrix[accs_matrix < 500].max().max()

        return accs_matrix
