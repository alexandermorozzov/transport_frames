from  transport_frames.new_modules.models.schema import BaseSchema
from pandera.typing import Series
from shapely import  Polygon, MultiPolygon

class PolygonSchema(BaseSchema):
    
    name: Series[str]
    _geom_types = [Polygon, MultiPolygon]