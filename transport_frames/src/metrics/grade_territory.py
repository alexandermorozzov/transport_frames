import momepy
import numpy as np
import pandas as pd

def grade_polygon(row,include_priority = False):
        """
        Determines the grade of a territory based on its distance to features.

        Parameters:
            row (Series): A pandas Series representing a single row of a GeoDataFrame.

        Returns:
            float: The grade of the territory.
        """
        dist_to_reg1 = row['dist_to_reg1']
        dist_to_reg2 = row['dist_to_reg2']
        dist_to_edge = row['dist_to_edge']
        dist_to_priority1 = row['dist_to_priority_reg1']
        dist_to_priority2 = row['dist_to_priority_reg2']


        # below numbers measured in thousands are representes in meters eg 5_000 meters ie 5km
        if include_priority and dist_to_priority1 < 5000:
            grade = 5
        elif include_priority and dist_to_priority1 < 10000 and dist_to_priority2 < 5000 or dist_to_reg1 < 5000:
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



def grade_territory(gdf_poly, frame, include_priority=False):
    """
    Grades territories based on their distances to reg1, reg2 nodes,edges and train stations.

    Parameters:
        gdf_poly (GeoDataFrame): A GeoDataFrame containing the polygons of the territories to be graded.
        frame
     (networkx.MultiDiGraph): A MultiDiGraph representing the transportation network.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the graded territories with added 'grade' column.
    """

    nodes, edges = momepy.nx_to_gdf(frame, points=True, lines=True, spatial_weights=False)
    
    poly = gdf_poly.copy().to_crs(nodes.crs)

    reg1_points = nodes[nodes['reg_1'] == 1]
    reg2_points = nodes[nodes['reg_2'] == 1]
    priority_reg1_points = nodes[(nodes['weight'] > np.percentile(nodes[nodes['weight'] != 0]['weight'], 60)) & (nodes['reg_1']==1)]
    priority_reg2_points = nodes[(nodes['weight'] > np.percentile(nodes[nodes['weight'] != 0]['weight'], 60)) & (nodes['reg_2']==1)]

    min_distance = lambda polygon, points: points.distance(polygon).min()
    poly['dist_to_reg1'] = poly.geometry.apply(lambda x: min_distance(x, reg1_points.geometry))
    poly['dist_to_reg2'] = poly.geometry.apply(lambda x: min_distance(x, reg2_points.geometry))
    poly['dist_to_edge'] = poly.geometry.apply(lambda x: min_distance(x, edges.geometry))
    poly['dist_to_priority_reg1'] = poly.geometry.apply(lambda x: min_distance(x, priority_reg1_points.geometry))
    poly['dist_to_priority_reg2'] = poly.geometry.apply(lambda x: min_distance(x, priority_reg2_points.geometry))

    poly['grade'] = poly.apply(grade_polygon,axis=1,args =(include_priority,))

    return poly


def create_buffered_gdf(graded_gdf, frame):
    """
    Creates a GeoDataFrame by buffering the geometries, converting CRS, and concatenating with filtered edges.

    Parameters:
        graded_gdf (GeoDataFrame): A GeoDataFrame containing graded geometries.
        frame
     (networkx.MultiDiGraph): A MultiDiGraph representing the transportation network.
        geojson_path (str): Path to save the output GeoDataFrame as a GeoJSON file.

    Returns:
        GeoDataFrame: The resulting concatenated GeoDataFrame.
    """
    graded_gdf = graded_gdf.to_crs(3857)
    _, edges = momepy.nx_to_gdf(frame)

    edges_filtered = edges[edges['reg'].isin([1])]
    buffer_gdf = graded_gdf.copy()
    buffer_gdf['geometry'] = graded_gdf['geometry'].buffer(5000)
    graded_gdf_4326 = graded_gdf.to_crs(4326)
    edges_filtered_4326 = edges_filtered.to_crs(4326)
    buffer_gdf_4326 = buffer_gdf.to_crs(4326)
    gdf = pd.concat([graded_gdf_4326, edges_filtered_4326, buffer_gdf_4326])

    return gdf
