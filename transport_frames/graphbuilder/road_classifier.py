import re
import pandas as pd
from transport_frames.utils.constant_road_vars import HIGHWAY_MAPPING, MAX_SPEEDS


class RoadClassifier:
    
    @staticmethod
    def determine_reg(name_roads, highway_type=None) -> int:
        """
        Determine the REG value based on road names and highway type.

        Parameters:
        name_roads: The input road names.
        highway_type: The type of highway.

        Returns:
        int: The REG value.
        """
        if isinstance(name_roads, list):
            for item in name_roads:
                if re.match(r"^[МАР]", str(item)):
                    return 1
                elif re.match(r"^\d.*[A-Za-zА-Яа-я]", str(item)):
                    return 2
            return 3
        elif pd.isna(name_roads):
            # Default REG value based on highway type if name_roads is NaN
            if highway_type:
                return RoadClassifier.highway_type_to_reg(highway_type)
            return 3
        if re.match(r"^[МАР]", str(name_roads)):
            return 1
        elif re.match(r"^\d.*[A-Za-zА-Яа-я]", str(name_roads)):
            return 2
        else:
            return 3

    @staticmethod
    def highway_type_to_reg(highway_type) -> int:
        """
        Convert highway type to REG value.

        Parameters:
        highway_type: The type of highway.

        Returns:
        int: The REG value.
        """
        if isinstance(highway_type, list):
            reg_values = [HIGHWAY_MAPPING.get(ht, 3) for ht in highway_type]
            return min(reg_values)
        return HIGHWAY_MAPPING.get(highway_type, 3)

    @staticmethod
    def get_max_speed(highway_types) -> float:
        """
        Get the maximum speed for road types.

        Parameters:
        highway_types: Type(s) of roads.

        Returns:
        float: Maximum speed.
        """
        if isinstance(highway_types, list):
            max_type = max(highway_types, key=lambda x: MAX_SPEEDS.get(x, float('nan')))
            return MAX_SPEEDS.get(max_type, 40 / 3.6)
        else:
            return MAX_SPEEDS.get(highway_types, 40 / 3.6)
