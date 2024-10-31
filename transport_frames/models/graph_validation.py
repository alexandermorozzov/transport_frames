from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Union, Optional, List
from shapely.geometry import LineString
import re
import numpy as np
import pandas as pd

class GraphMetadata(BaseModel):
    """
    Represents metadata for a graph, specifically the coordinate reference system (CRS).

    Attributes:
    - crs (str): The coordinate reference system in EPSG format.
    """

    crs: int

    # @field_validator('crs', mode='before')
    # def validate_crs(cls, value: str) -> str:
    #     """
    #     Validates the CRS format to ensure it follows the 'epsg:XXXX' pattern.

    #     Parameters:
    #     - value (str): The CRS value to validate.

    #     Returns:
    #     - str: The validated CRS value.

    #     Raises:
    #     - ValueError: If the CRS format is invalid.
    #     """
    #     pattern = r'^epsg:\d+$'
    #     if not re.match(pattern, value):
    #         raise ValueError(f"Invalid CRS format: {value}. Expected format: 'epsg:XXXX'")
    #     return value


class GraphNode(BaseModel):
    """
    Represents a node in the graph with its coordinates.

    Attributes:
    - x (float): The x-coordinate of the node.
    - y (float): The y-coordinate of the node.
    """

    x: float
    y: float


class GraphEdge(BaseModel):
    """
    Represents an edge in the graph with its properties.

    Attributes:
    - length_meter (float): The length of the edge in meters. Must be non-negative.
    - geometry (LineString): The geometric representation of the edge.
    - highway (Union[str, List[str]]): The type(s) of highway represented by this edge.
    - ref (Optional[Union[str, List[str], float, None]]): A reference attribute that can be a string, list of strings, or NaN.
    """

    length_meter: float = Field(..., ge=0)
    geometry: LineString
    highway: Union[str, List[str]]
    ref: Optional[Union[str, List[str], float, None]]

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types

    @model_validator(mode='before')
    def validate_highway(cls, values) -> dict:
        """
        Validates the highway attribute to ensure it is a valid type.

        Parameters:
        - values (dict): The attributes of the edge being validated.

        Returns:
        - dict: The validated attributes.

        Raises:
        - ValueError: If the highway attribute is invalid.
        """
        highway = values.get('highway')
        valid_highways = {
            'living_street', 'motorway', 'motorway_link', 'primary', 'primary_link', 'residential',
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
    def check_ref(cls, values) -> dict:
        """
        Validates the ref attribute to ensure it meets expected types.

        Parameters:
        - values (dict): The attributes of the edge being validated.

        Returns:
        - dict: The validated attributes.

        Raises:
        - ValueError: If the ref attribute is invalid.
        """
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
    """
    Represents a classified edge in the graph with additional time and region attributes.

    Attributes:
    - time_min (float): The estimated travel time in minutes. Must be non-negative.
    - reg (int): The region classification (1, 2, or 3).
    """

    time_min: float = Field(..., ge=0.0)
    reg: int = Field(ge=1, le=3)  # reg should be an integer (1, 2, or 3)

    @model_validator(mode='before')
    def validate_reg(cls, values) -> dict:
        """
        Validates the reg attribute to ensure it is within the allowed range.

        Parameters:
        - values (dict): The attributes of the classified edge being validated.

        Returns:
        - dict: The validated attributes.

        Raises:
        - ValueError: If the reg value is invalid.
        """
        reg = values.get('reg')
        if reg not in {1, 2, 3}:
            raise ValueError(f"Invalid reg value: {reg}. It should be 1, 2, or 3.")
        return values
