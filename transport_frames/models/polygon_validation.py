from transport_frames.models.schema import BaseSchema
from pandera.typing import Series
from shapely import Polygon, MultiPolygon, Point

class PolygonSchema(BaseSchema):
    """
    Schema for validating polygons and multipolygons.

    Attributes:
    - name (Series[str]): The name associated with the polygon(s).
    - _geom_types (list): List of allowed geometry types (Polygon, MultiPolygon).
    """
    
    name: Series[str]
    _geom_types = [Polygon, MultiPolygon]


class RegionsSchema(BaseSchema):
    """
    Schema for validating regions defined by polygons and multipolygons.

    Attributes:
    - name (Series[str]): The name associated with the region(s).
    - _geom_types (list): List of allowed geometry types (Polygon, MultiPolygon).
    """
    
    name: Series[str]
    _geom_types = [Polygon, MultiPolygon]


class CentersSchema(BaseSchema):
    """
    Schema for validating point geometries representing centers.

    Attributes:
    - name (Series[str]): The name associated with the center(s).
    - _geom_types (list): List of allowed geometry types (Point).
    """
    
    name: Series[str]
    _geom_types = [Point]


class CountrySchema(BaseSchema):
    """
    Schema for validating countries defined by polygons and multipolygons.

    Attributes:
    - _geom_types (list): List of allowed geometry types (Polygon, MultiPolygon).
    """
    
    _geom_types = [Polygon, MultiPolygon]
