import json
from pprint import pprint
import googlemaps

with open("gmap_api_config.json","r") as openfile:
    API_KEY = json.load(openfile)


gmaps_client = googlemaps.Client(key = API_KEY["API_KEY"])





"""

work_place_adress = "1 Market st, San Francisco, CA"
response = gmaps_client.geocode(work_place_adress)

pprint(response)

#speed = gmaps_client.speed_limits(work_place_adress)


"""

