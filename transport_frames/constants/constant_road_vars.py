# Константные словари
HIGHWAY_MAPPING = {
    "motorway": 1,
    "trunk": 1,
    "primary": 2,
    "secondary": 2,
    "tertiary": 3,
    "unclassified": 3,
    "residential": 3,
    "motorway_link": 1,
    "trunk_link": 1,
    "primary_link": 2,
    "secondary_link": 2,
    "tertiary_link": 3,
    "living_street": 3,
}

MAX_SPEEDS = {
    "motorway": 110 / 3.6,
    "motorway_link": 110 / 3.6,
    "primary": 80 / 3.6,
    "primary_link": 80 / 3.6,
    "residential": 60 / 3.6,
    "secondary": 70 / 3.6,
    "secondary_link": 70 / 3.6,
    "tertiary": 60 / 3.6,
    "tertiary_link": 60 / 3.6,
    "trunk": 90 / 3.6,
    "trunk_link": 90 / 3.6,
    "unclassified": 60 / 3.6,
    "living_street": 15 / 3.6,
}
