import geopandas as gpd
import networkx as nx
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import split, unary_union, nearest_points
from shapely.geometry import MultiPoint



def create_nodeID() -> int:
    """Generate a new unique node ID.

    Returns:
        int: The new node ID.
    """
    global node_id # holds the id of last created node
    node_id += 1
    return node_id # returns an id for new node

def find_nearest_node(point: Point, nodes_gdf: gpd.GeoDataFrame, node_buffer: float) -> int:
    """Find the nearest node within a specified buffer distance.

    Args:
        point (Point): The point to find the nearest node for.
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes.
        node_buffer (float): The buffer distance within which to search for the nearest node.

    Returns:
        int: The ID of the nearest node, or None if no node is found.
    """
    node_buffer_geom = point.buffer(node_buffer)
    nearby_nodes = nodes_gdf[nodes_gdf.intersects(node_buffer_geom)]
    if not nearby_nodes.empty:
        nearest_node = nearby_nodes.distance(point).idxmin()
        nearest_node_id = nearby_nodes.loc[nearest_node]
        return nearest_node_id
    
    return None

def find_nearest_road_point(point: Point, roads_gdf: gpd.GeoDataFrame, road_buffer: float) -> tuple:
    """Find the nearest road and the closest point on that road within a specified buffer distance.

    Args:
        point (Point): The point to find the nearest road for.
        roads_gdf (gpd.GeoDataFrame): GeoDataFrame containing roads.
        road_buffer (float): The buffer distance within which to search for the nearest road.

    Returns:
        tuple: A tuple containing the nearest roads GeoDataFrame and the nearest point on the road, or (None, None) if no road is found.
    """
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs=roads_gdf.crs)
    
    # Perform a spatial join to find the nearest road within the buffer distance
    nearest_roads_all = gpd.sjoin_nearest(
        point_gdf, 
        roads_gdf, 
        how="inner", 
        max_distance=road_buffer, 
    )
    if not nearest_roads_all.empty:
        nearest_roads = roads_gdf.loc[nearest_roads_all['index_right'].to_list()]
        nearest_point_on_road = nearest_points(point, nearest_roads.geometry)[1].to_list()[0]
        return nearest_roads, nearest_point_on_road
    return None, None

def make_edge_data(edge_geom: LineString, maxspeed: float, reg: int, local_crs: str, start: int = None, end: int = None, nodes_gdf: gpd.GeoDataFrame = None, bidirectional: bool = True, flag: str = None) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with edge data based on geometry and other attributes.

    Args:
        edge_geom (LineString): The geometry of the edge.
        maxspeed (float): The maximum speed on the edge.
        reg (int): The region identifier.
        local_crs (str): The coordinate reference system.
        start (int, optional): The starting node ID. Defaults to None.
        end (int, optional): The ending node ID. Defaults to None.
        nodes_gdf (gpd.GeoDataFrame, optional): GeoDataFrame of nodes. Defaults to None.
        bidirectional (bool, optional): If True, creates a bidirectional edge. Defaults to True.
        flag (str, optional): Additional information for the edge. Defaults to None.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the edge data.
    """
    edge_data = {
                'length_meter': edge_geom.length,
                'geometry': edge_geom,
                'type': 'car',
                'time_sec': edge_geom.length / (maxspeed),
                'time_min': edge_geom.length/ (maxspeed * 60),
                'highway': 'unclassified',
                'maxspeed': maxspeed,
                'reg': reg,
                'ref': np.nan,
                # 'is_exit': 'RECOUNT',
                'node_start' : start if start else find_end_node(edge_geom.coords[0],nodes_gdf,local_crs).reset_index().nodeID[0],
                'node_end' : end if end else find_end_node(edge_geom.coords[-1],nodes_gdf,local_crs).reset_index().nodeID[0],
                'flag': flag
            }
    
    gdf = gpd.GeoDataFrame([edge_data], crs=local_crs)
    if bidirectional:
        reverse_data = make_edge_data(edge_geom.reverse(), maxspeed, reg, local_crs, end, start, nodes_gdf, bidirectional=False,flag=flag)
        gdf = pd.concat([gdf,reverse_data])
    return gdf
    
def make_node_data(point: Point, local_crs: int, reg_1: bool = False, reg_2: bool = False, flag: str = 'intersection') -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with node data based on geometry and other attributes.

    Args:
        point (Point): The geometry of the node.
        local_crs (str): The coordinate reference system.
        reg_1 (bool, optional): Indicator for region 1. Defaults to False.
        reg_2 (bool, optional): Indicator for region 2. Defaults to False.
        exit (str, optional): Exit status. Defaults to 'RECOUNT'.
        exit_country (str, optional): Exit country. Defaults to 'RECOUNT'.
        flag (str, optional): Additional information for the node. Defaults to 'intersection'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the node data.
    """
    node_data = {
            'reg_1': reg_1,
            'reg_2': reg_2,
            'x': point.x,
            'y': point.y,
            'nodeID': create_nodeID(),
            # 'exit': exit,
            # 'exit_country': exit_country, #POLINA
            'geometry': point,
            'flag':flag
        }
    gdf = gpd.GeoDataFrame([node_data], crs=local_crs)
    return gdf

def find_end_node(end: tuple, nodes_gdf: gpd.GeoDataFrame, local_crs: int) -> gpd.GeoDataFrame:
    """Find the node corresponding to a LineString end using the nearest node.

    Args:
        end (tuple): The coordinates of the end point.
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing nodes.
        local_crs (int): The coordinate reference system.

    Returns:
        gpd.GeoDataFrame: The nearest node as a GeoDataFrame.
    """    
    end_point = gpd.GeoDataFrame(geometry=[Point(end)], crs=local_crs)
    nearest_points = gpd.sjoin_nearest(nodes_gdf,end_point,distance_col='dist').reset_index()
    return nearest_points.loc[[nearest_points['dist'].idxmin()]]

def connect_line_to_roads(each_line: gpd.GeoSeries, nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame, node_buffer: float, edge_buffer: float, local_crs: str) -> tuple:
    """Connect a line to nearby roads by adding nodes and edges as needed.

    Args:
        each_line (gpd.GeoSeries): The line to connect.
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing existing nodes.
        edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing existing edges.
        node_buffer (float): Buffer distance to search for existing nodes.
        edge_buffer (float): Buffer distance to search for existing edges.
        local_crs (int): The coordinate reference system.

    Returns:
        tuple: A tuple containing GeoDataFrames of added nodes, added edges, a list of deleted edges, and a list of deleted edge ends.
    """
    added_nodes  = gpd.GeoDataFrame(columns=nodes_gdf.columns).set_geometry('geometry').set_crs(local_crs) 
    added_edges = gpd.GeoDataFrame(columns=edges_gdf.columns).set_geometry('geometry').set_crs(local_crs)
    del_list_ends = []
    del_list = []
    # global node_id
    for end in [each_line.geometry.coords[0],each_line.geometry.coords[-1]]:    
        new_end_node = make_node_data(Point(end),local_crs,
                                reg_1 = True if each_line['reg']== 1 else False, 
                                reg_2 = True if each_line['reg']==2 else False )
        added_nodes = pd.concat([added_nodes, new_end_node]) # adding ends

        nearest_node = find_nearest_node(Point(end),nodes_gdf,node_buffer)
        if nearest_node is not None:
            bridge = LineString([end, nearest_node.geometry])
            bridge_edge = make_edge_data(bridge, each_line['maxspeed'],each_line['reg'],local_crs=local_crs, start=new_end_node.reset_index()['nodeID'][0],end=nearest_node.nodeID, flag='bridge')
            added_edges = pd.concat([added_edges, bridge_edge]) # if we can connect end to a node, only the bridge is added

        else: # if there is no node nearby
            nearest_roads, nearest_point_on_road = find_nearest_road_point(Point(end),edges_gdf,edge_buffer)
            if nearest_roads is not None:
                bridge_connector_node = make_node_data(nearest_point_on_road, local_crs,
                                        reg_1 = True if 1 in nearest_roads['reg'].to_list() else False, 
                                        reg_2 = True if 2 in nearest_roads['reg'].to_list() else False,
                                        # exit=False,
                                        # exit_country=False
                                         )
                added_nodes = pd.concat([added_nodes, bridge_connector_node]) # adding bridge connector node
                reg=max(nearest_roads['reg'].to_list())
                speeds ={ 1: 110/3.6, 2: 90/3.6, 3: 60/3.6}
                bridge_edge = make_edge_data(LineString([end,nearest_point_on_road]), 
                                        maxspeed = speeds[reg], 
                                        reg=reg, 
                                        local_crs=local_crs,
                                        start=new_end_node.reset_index()['nodeID'][0], 
                                        end=bridge_connector_node.reset_index()['nodeID'][0],
                                        flag='bridge')
                added_edges = pd.concat([added_edges, bridge_edge]) # adding bridge

                # splitting the road
                intersections = nearest_roads[nearest_roads.intersects(nearest_point_on_road.buffer(0.001))] # POLINA
                for idx, row in intersections.iterrows():
                    split_geometries = split(row.geometry, nearest_point_on_road.buffer(0.001)) #POLINA CHECK THE OUTPUT OF SPLITTING 
                    split_lines = [geom for geom in split_geometries.geoms]
                    first_geom = LineString([*split_lines[0].coords] + [*split_lines[1].coords[1:]])
                    first_piece = make_edge_data(first_geom,
                                                row.maxspeed, 
                                                row.reg, 
                                                local_crs=local_crs,
                                                start=row.node_start, 
                                                end=bridge_connector_node.reset_index()['nodeID'][0], 
                                                bidirectional=False,
                                                flag='road_segment')
                    if len(split_lines) >2:
                        second_geom = LineString([*split_lines[2].coords])
                        second_piece = make_edge_data(second_geom,
                                                row.maxspeed, 
                                                row.reg, 
                                                local_crs=local_crs,
                                                start=bridge_connector_node.reset_index()['nodeID'][0], 
                                                end=row.node_end, 
                                                bidirectional=False,
                                                flag='road_segment')
                        
                        added_edges = pd.concat([added_edges, first_piece, second_piece]) # adding road pieces
                    else:
                        added_edges = pd.concat([added_edges, first_piece]) # adding road pieces

                del_list.extend(intersections.index.to_list()) 
                del_list_ends.extend(list(zip(intersections['node_start'].to_list(),intersections['node_end'].to_list())))
  

    return added_nodes, added_edges, del_list, del_list_ends
    



def split_roads(each_line: gpd.GeoDataFrame, 
                intermediate_nodes: gpd.GeoDataFrame, 
                intermediate_edges: gpd.GeoDataFrame, 
                local_crs: str) -> tuple:
    """
    Splits roads at intersection points and adds nodes and edges for new road segments.

    Args:
        each_line (gpd.GeoDataFrame): The new road geometry being processed.
        intermediate_nodes (gpd.GeoDataFrame): GeoDataFrame of nodes for the existing road network.
        intermediate_edges (gpd.GeoDataFrame): GeoDataFrame of edges for the existing road network.
        local_crs (str): Coordinate reference system for spatial data.

    Returns:
        tuple: A tuple containing:
            - added_intersections_nodes (gpd.GeoDataFrame): GeoDataFrame of new intersection nodes.
            - added_intersections_edges (gpd.GeoDataFrame): GeoDataFrame of new road segments.
            - del_list (list): List of indices of deleted roads.
            - del_list_ends (list): List of node start-end pairs of deleted roads.
    """ 
    del_list = []
    del_list_ends =[]
    added_intersections_nodes = gpd.GeoDataFrame(columns=intermediate_nodes.columns).set_geometry('geometry').set_crs(local_crs) 
    added_intersections_edges = gpd.GeoDataFrame(columns=intermediate_edges.columns).set_geometry('geometry').set_crs(local_crs) 

    road_pieces = intermediate_edges[~intermediate_edges.flag.isin(['bridge'])]
    intersected_roads = road_pieces[road_pieces.intersects(each_line.geometry)] # find pieces of road that should be split
    
    if len(intersected_roads) != 0:
        intersection_points = each_line.geometry.intersection(unary_union(intersected_roads.geometry)) # points of intersections
        if isinstance(intersection_points,Point):
                point = intersection_points
                regs = find_nearest_road_point(point,intersected_roads,0.1)[0]['reg'].to_list()+ [each_line.reg]
                new_intersection = make_node_data(point,local_crs=local_crs,reg_1= 1 in regs, reg_2 = 2 in regs)
                added_intersections_nodes = pd.concat([added_intersections_nodes,new_intersection]) # adding new no
        if isinstance(intersection_points,MultiPoint):  
            for point in intersection_points.geoms:
                regs = find_nearest_road_point(point,intersected_roads,0.1)[0]['reg'].to_list()+ [each_line.reg]
                new_intersection = make_node_data(point,local_crs=local_crs,reg_1= 1 in regs, reg_2 = 2 in regs)
                added_intersections_nodes = pd.concat([added_intersections_nodes,new_intersection]) # adding new nodes
            
        for idx, road in intersected_roads.iterrows():
            for piece in split(road.geometry, each_line.geometry).geoms:
                global ra
                ra = pd.concat([intermediate_nodes,added_intersections_nodes])
                new_piece = make_edge_data(piece,road.maxspeed,local_crs=local_crs, reg=road.reg, nodes_gdf=pd.concat([intermediate_nodes,added_intersections_nodes]),bidirectional=False, flag='road_segment')
                added_intersections_edges = pd.concat([added_intersections_edges,new_piece]) # adding road pieces
            del_list.extend(intersected_roads.index.to_list()) 
            del_list_ends.extend(list(zip(intersected_roads['node_start'].to_list(),intersected_roads['node_end'].to_list())))
            
        for piece in split(each_line.geometry, unary_union(intersected_roads.geometry)).geoms:
            new_piece = make_edge_data(piece, each_line.maxspeed, local_crs=local_crs,reg=each_line.reg, nodes_gdf=pd.concat([intermediate_nodes,added_intersections_nodes]),bidirectional=True, flag='line_segment')
            added_intersections_edges = pd.concat([added_intersections_edges,new_piece]) #adding line pieces
            
    return added_intersections_nodes, added_intersections_edges, del_list, del_list_ends

def modify_graph(citygraph: nx.MultiDiGraph, 
                 del_list_ends: list, 
                 nodes_to_add: gpd.GeoDataFrame, 
                 edges_to_add: gpd.GeoDataFrame) -> nx.Graph:
    """
    Modifies the city graph by removing edges and adding new nodes and edges.

    Args:
        citygraph (nx.MultiDiGraph): The existing city graph.
        del_list_ends (list): List of node start-end pairs for edges to remove.
        nodes_to_add (gpd.GeoDataFrame): GeoDataFrame of nodes to add to the graph.
        edges_to_add (gpd.GeoDataFrame): GeoDataFrame of edges to add to the graph.

    Returns:
        nx.Graph: The modified city graph.
    """    
    graph = citygraph.copy()
    graph.remove_edges_from(del_list_ends)
    to_add = []
    for idx,row in nodes_to_add.iterrows():
        to_add.append((row.nodeID,row.to_dict()))
    graph.add_nodes_from(to_add)

    to_add=[]
    for idx,row in edges_to_add.iterrows():
        to_add.append((row.node_start, row.node_end,row.to_dict()))
    graph.add_edges_from(to_add);
    return graph


def add_roads(citygraph: nx.MultiDiGraph, 
              line_gdf: gpd.GeoDataFrame, 
              nodes_gdf: gpd.GeoDataFrame, 
              edges_gdf: gpd.GeoDataFrame, 
              local_crs: int, 
              node_buffer: int = 300, 
              road_buffer: int = 5000) -> nx.Graph:
    """
    Adds new roads to the city graph by processing each line in the road GeoDataFrame.

    Args:
        citygraph (nx.MultiDiGraph): The existing city graph.
        line_gdf (gpd.GeoDataFrame): GeoDataFrame of the new road lines.
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame of the existing road network nodes.
        edges_gdf (gpd.GeoDataFrame): GeoDataFrame of the existing road network edges.
        local_crs (int): Coordinate reference system for spatial data.
        node_buffer (int): Buffer distance for finding nearby nodes. Defaults to 300.
        road_buffer (int): Buffer distance for finding nearby roads. Defaults to 5000.

    Returns:
        nx.MultiDiGraph: The modified city graph with new roads, nodes, and edges.
    """
    global node_id
    line_gdf['maxspeed'] = line_gdf['reg'].apply(lambda reg: 110/3.6 if reg == 1 else 90/3.6 if reg == 2 else 60/3.6 if reg == 3 else None)

    node_id = nodes_gdf.index.max()
    nodes_to_add = gpd.GeoDataFrame(columns=nodes_gdf.columns).set_geometry('geometry').set_crs(local_crs) 
    edges_to_add = gpd.GeoDataFrame(columns=edges_gdf.columns).set_geometry('geometry').set_crs(local_crs) 
    del_list_ends = []
    for idx, each_line in line_gdf.iterrows():
        added_nodes, added_edges, del_list, del_list_ends1 = connect_line_to_roads(each_line,nodes_gdf,edges_gdf,node_buffer,road_buffer,local_crs)
        intermediate_nodes = pd.concat([nodes_gdf,added_nodes])
        intermediate_edges= pd.concat([edges_gdf,added_edges]).drop(del_list).reset_index()

        added_intersections_nodes, added_intersections_edges, del_list, del_list_ends2 = split_roads(each_line,intermediate_nodes, intermediate_edges,local_crs)
        

        if len(added_edges) == 0 and len(added_intersections_edges)==0:
            print('not connected')
            continue
        else:
            nodes_to_add = pd.concat([nodes_to_add, added_nodes,added_intersections_nodes])
            edges_to_add = pd.concat([edges_to_add, added_edges,added_intersections_edges])
            del_list_ends.extend(del_list_ends1+del_list_ends2)
            
    graph = modify_graph(citygraph,del_list_ends1+ del_list_ends2, nodes_to_add, edges_to_add)
    return graph

