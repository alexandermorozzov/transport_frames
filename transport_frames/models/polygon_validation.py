from  transport_frames.models.schema import BaseSchema
from pandera.typing import Series
from shapely import  Polygon, MultiPolygon, Point

class PolygonSchema(BaseSchema):
    
    name: Series[str]
    _geom_types = [Polygon, MultiPolygon]

class RegionsSchema(BaseSchema):
    
    name: Series[str]
    _geom_types = [Polygon, MultiPolygon]

class CentersSchema(BaseSchema):
    
    name: Series[str]
    _geom_types = [Point]

class CountrySchema(BaseSchema):
    
    _geom_types = [Polygon, MultiPolygon]