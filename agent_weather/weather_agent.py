import requests
import os
from dotenv import load_dotenv

load_dotenv()
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")
AZURE_MAPS_ENDPOINT = os.getenv("AZURE_MAPS_ENDPOINT")

def get_coordinates_from_location(location: str):
    url = f"{AZURE_MAPS_ENDPOINT}/search/address/json"
    params = {
        'api-version': '1.0',
        'subscription-key': AZURE_MAPS_KEY,
        'query': location
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['results']:
        coords = data['results'][0]['position']
        return coords['lat'], coords['lon']
    return None

def get_weather_forecast(lat: float, lon: float):
    url = f"{AZURE_MAPS_ENDPOINT}/weather/forecast/daily/json"
    params = {
        'api-version': '1.1',
        'subscription-key': AZURE_MAPS_KEY,
        'query': f"{lat},{lon}",
        'duration': 5
    }
    response = requests.get(url, params=params)
    return response.json()

# Exemple d'utilisation :
location = "Paris, France"
coords = get_coordinates_from_location(location)
if coords:
    forecast = get_weather_forecast(*coords)
    print(forecast)