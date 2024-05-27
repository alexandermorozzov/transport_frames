import pandas as pd
import geopandas as gpd
import networkx as nx
import momepy

def get_weight(start,end,exit):
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
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]   # 2.2.3
            ]
    else:
        
        matrix = [
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08],  # 2.1.1
                [0.07, 0.07, 0.07, 0.07, 0.07, 0.07],  # 2.1.2
                [0.06, 0.06, 0.06, 0.06, 0.06, 0.06],  # 2.1.3
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # 2.2.1
                [0.04, 0.04, 0.04, 0.04, 0.04, 0.04],  # 2.2.2
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]   # 2.2.3
            ]
    return matrix[dict[end]][dict[start]]




def weigh_roads(carcas):
    """
    Calculate and normalize the weights of roads between exits in a road network.

    Parameters:
    carcas (networkx.Graph): The road network graph where nodes represent intersections or exits
                             and edges represent road segments with 'time_min' as a weight attribute.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame with road segments and their corresponding normalized weights.
                            The geometry of the segments is converted to EPSG:4326.
    """
    n,e = momepy.nx_to_gdf(carcas)
    exits = n[n['exit']==1] 
    data = []
    for i1, start_node in exits.iterrows():
        for i2, end_node in exits.iterrows():
            if i1 == i2:
                continue
            if pd.notna(start_node['border_region']) and start_node['border_region'] == end_node['border_region']:
                continue
            if start_node.geometry.buffer(15000).intersects(end_node.geometry.buffer(15000)) and (pd.isna(start_node['exit_country']) == pd.isna(end_node['exit_country'])):
                continue
            if start_node['exit_country'] == 1 and end_node['exit_country'] == 1:
                continue

            weight = get_weight(start_node['ref_type'], end_node['ref_type'], end_node['exit_country'])

            try:
                # Find the shortest path using Dijkstra's algorithm
                path = nx.dijkstra_path(carcas, i1, i2, weight='time_min')
            except nx.NetworkXNoPath:
                continue

            # Extract the path coordinates and concatenate geometries
            for j in range(len(path) - 1):
                edge_data = carcas.get_edge_data(path[j], path[j + 1])
                for key, value in edge_data.items():
                    g = value['geometry']
                    data.append({
                        'start_node': i1,
                        'end_node': i2,
                        'geometry': g,
                        'weight': weight
                    })

    # Create GeoDataFrame from the data
    roads_gdf = gpd.GeoDataFrame(data, crs=exits.crs)
    
    # Group by geometry and sum the weights
    roads_gdf = roads_gdf.groupby('geometry').agg({'weight': 'sum'}).reset_index().set_geometry('geometry').to_crs(epsg=4326)

    # Normalize weights for plotting
    max_weight = roads_gdf['weight'].max()
    roads_gdf['normalized_weight'] = roads_gdf['weight'] / max_weight

    return roads_gdf

