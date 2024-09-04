import geopandas as gpd
import networkx as nx
import momepy
import sys
sys.path.append('../')

import momepy
import osmnx as ox
import geopandas as gpd
import shapely
import pandas as pd
import networkx as nx
import numpy as np
import json
from shapely.geometry import Point, LineString
from shapely.ops import split, unary_union, nearest_points
from shapely.geometry import GeometryCollection, MultiPoint



def create_nodeID():
    global node_id # holds the id of last created node
    node_id += 1
    return node_id # returns an id for new node

def find_nearest_node(point, nodes_gdf, node_buffer):
    # finds nearest node in some buffer, returns the node is no None
    node_buffer_geom = point.buffer(node_buffer)
    nearby_nodes = nodes_gdf[nodes_gdf.intersects(node_buffer_geom)]
    if not nearby_nodes.empty:
        nearest_node = nearby_nodes.distance(point).idxmin()
        nearest_node_id = nearby_nodes.loc[nearest_node]
        return nearest_node_id
    
    return None

def find_nearest_road_point(point, roads_gdf, road_buffer):
    # finds nearest road in buffer, return the point on the road else None
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

def make_edge_data(edge_geom, maxspeed, reg, local_crs,start=None, end=None, nodes_gdf=None, bidirectional=True, flag = None):
    # creates a gdf with new edge data based on its geometry. can make gdf with 2 rows - bidirectional
    ra = start if start else find_end_node(edge_geom.coords[0],nodes_gdf,local_crs).reset_index().nodeID[0],
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
                'is_exit': 'RECOUNT',
                'node_start' : start if start else find_end_node(edge_geom.coords[0],nodes_gdf,local_crs).reset_index().nodeID[0],
                'node_end' : end if end else find_end_node(edge_geom.coords[-1],nodes_gdf,local_crs).reset_index().nodeID[0],
                'flag': flag
            }
    
    gdf = gpd.GeoDataFrame([edge_data], crs=local_crs)
    if bidirectional:
        reverse_data = make_edge_data(edge_geom.reverse(), maxspeed, reg, local_crs, end, start, nodes_gdf, bidirectional=False,flag=flag)
        gdf = pd.concat([gdf,reverse_data])
    return gdf
    
def make_node_data(point,local_crs, reg_1=False, reg_2=False, exit='RECOUNT', exit_country='RECOUNT',flag='intersection'):
    node_data = {
            'reg_1': reg_1,
            'reg_2': reg_2,
            'x': point.x,
            'y': point.y,
            'nodeID': create_nodeID(),
            'exit': exit,
            'exit_country': exit_country, #POLINA
            'geometry': point,
            'flag':flag
        }
    gdf = gpd.GeoDataFrame([node_data], crs=local_crs)
    return gdf

def find_end_node(end, nodes_gdf,local_crs):
    # find the node corresponding to a linestring end using nearest of the nodes
    end_point = gpd.GeoDataFrame(geometry=[Point(end)], crs=local_crs)
    nearest_points = gpd.sjoin_nearest(nodes_gdf,end_point,distance_col='dist').reset_index()
    return nearest_points.loc[[nearest_points['dist'].idxmin()]]

def connect_line_to_roads(each_line,nodes_gdf,edges_gdf,node_buffer,edge_buffer, local_crs):

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
                                        exit=False,
                                        exit_country=False )
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
    



def split_roads(each_line,intermediate_nodes, intermediate_edges,local_crs):
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

def modify_graph(citygraph,del_list_ends,nodes_to_add, edges_to_add):
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


def add_roads(citygraph,line_gdf, nodes_gdf,edges_gdf,local_crs,node_buffer = 300,road_buffer = 5000):
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
            
    graf = modify_graph(citygraph,del_list_ends1+ del_list_ends2, nodes_to_add, edges_to_add)
    return graf

