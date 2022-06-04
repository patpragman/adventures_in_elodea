import pandas as pd
from dataclasses import dataclass
import re
import folium
from folium.plugins import MarkerCluster


def convert_coord(coord_str):
    """
    convert lat-longs of the format
    deg-min-sec
    with a direction the last character to decimal
    """
    deg, minutes, seconds = coord_str.split("-")
    direction = seconds[-1]
    seconds = seconds[0:-1]
    return (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60)) * (1 if direction in ['N', 'E'] else -1)


@dataclass
class Airport:
    type: str
    name: str
    identifier: str
    latitude: str
    longitude: str


# iterate through the airports, make Airport objects for later
airports = []
df = pd.read_csv("ak_airports.csv")
for index, row in df.iterrows():
    airport = Airport(type=row['Type'],
                      name=row['FacilityName'],
                      identifier=row['LocationID'],
                      latitude=convert_coord(row['ARPLatitude']),
                      longitude=convert_coord(row['ARPLongitude'])
                      )

    airports.append(airport)

m = folium.Map()
seaplane_bases_fg = folium.FeatureGroup("Seaplane Base")
airports_fg = folium.FeatureGroup("Airports")

for airport in airports:
    color = 'blue' if 'SEA' in airport.type else 'green'
    feature_group = seaplane_bases_fg if 'SEA' in airport.type else airports_fg

    folium.CircleMarker(location=[airport.latitude, airport.longitude],
                        popup=airport.name,
                        fill=True,
                        fill_opacity=1.0,
                        color=color,
                        radius=10,
                        fill_color=color).add_to(feature_group)

airports_fg.add_to(m)
seaplane_bases_fg.add_to(m)
folium.LayerControl().add_to(m)


m.save('map.html')
