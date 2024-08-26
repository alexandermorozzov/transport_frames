from pydantic import BaseModel, Field, field_validator, model_validator, conint
from typing import Union, Optional, List
from shapely.geometry import LineString
import re
import numpy as np
import pandas as pd

class GraphMetadata(BaseModel):
    crs: str


    @field_validator('crs', mode='before')
    def validate_crs(cls, value):
        pattern = r'^epsg:\d+$'
        if not re.match(pattern, value):
            raise ValueError(f"Invalid CRS format: {value}. Expected format: 'epsg:XXXX'")
        return value

class GraphNode(BaseModel):
    x: float
    y: float

class GraphEdge(BaseModel):
    length_meter: float = Field(..., ge=0)
    geometry: LineString
    highway: Union[str, List[str]]
    ref: Optional[Union[str, List[str], float, None]]

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    @model_validator(mode='before')
    def validate_highway(cls, values):
        highway = values.get('highway')
        valid_highways = {
            'living_street', 'motorway', 'primary', 'primary_link', 'residential',
            'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk',
            'trunk_link', 'unclassified'
        }

        if isinstance(highway, str):
            highway = [highway]
        elif not isinstance(highway, list):
            raise ValueError('Highway should be a string or a list of strings.')

        for h in highway:
            if h not in valid_highways:
                raise ValueError(f'Invalid highway type: {h}')
        
        values['highway'] = highway  # Ensure the highway field is always a list of strings
        return values


    @model_validator(mode='before')
    def check_ref(cls, values):
        ref = values.get('ref')

        if not isinstance(ref, list) and pd.isna(ref):
            ref = np.nan 
        
        if ref is np.nan:
            values['ref'] = np.nan
            return values

        if isinstance(ref, str):
            values['ref'] = ref
            return values
        
        if isinstance(ref, list):
            if all(isinstance(item, str) for item in ref):
                values['ref'] = ref
                return values
        
        raise ValueError('Ref should be either a string, a list of strings, None, or float("nan").')


class ClassifiedEdge(GraphEdge):
    time_min: float = Field(..., ge=0.0)
    reg: int = Field(ge=1, le=3)  # reg should be an integer (1, 2, or 3)

    @model_validator(mode='before')
    def validate_reg(cls, values):
        reg = values.get('reg')
        if reg not in {1, 2, 3}:
            raise ValueError(f"Invalid reg value: {reg}. It should be 1, 2, or 3.")
        return values

