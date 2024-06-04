"""
module is aimed to group all scripts related to visualisations
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def visualize_availability(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    service_gdf: gpd.GeoDataFrame = None,
    median: bool = True,
    title="Доступность сервиса, мин",
) -> None:
    """
    Visualize the service availability on a map with bounding polygons.
    Optionally service points and city points are shown.

    Parameters:
    points (geopandas.GeoDataFrame): GeoDataFrame of points with 'to_service' column.
    polygons (geopandas.GeoDataFrame): GeoDataFrame of polygons.
    service_gdf (geopandas.GeoDataFrame, optional): GeoDataFrame of service points. Defaults to None.
    median (bool, optional): Whether to aggregate time by median among cities in the polygon. Defaults to True.
    title (str, optional): Title of the plot. Defaults to 'Доступность сервиса, мин'.
    """
    points = points.to_crs(polygons.crs)

    vmax = points["to_service"].max()
    res = (
        gpd.sjoin(points, polygons, how="left", predicate="within")
        .groupby("index_right")
        .median(["to_service"])
    )
    _, ax = plt.subplots(1, 1, figsize=(16, 8))
    polygons.boundary.plot(ax=ax, color="black", linewidth=1).set_axis_off()

    if not median:
        merged = points
        merged.to_crs(points.crs).plot(
            column="to_service",
            cmap="RdYlGn_r",
            ax=ax,
            legend=True,
            vmax=vmax,
            markersize=4,
        ).set_axis_off()
    else:
        merged = pd.merge(
            polygons.reset_index(), res, left_on="index", right_on="index_right"
        )
        merged.to_crs(points.crs).plot(
            column="to_service",
            cmap="RdYlGn_r",
            ax=ax,
            legend=True,
            vmax=vmax,
            markersize=4,
        ).set_axis_off()
        if service_gdf is not None:
            service_gdf = service_gdf.to_crs(polygons.crs)
            service_gdf.plot(ax=ax, markersize=7, color="white").set_axis_off()

    plt.title(title)
    plt.show()
