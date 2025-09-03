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






















# weather_agent_langchain.py
import os
import re
import json
import logging
from typing import Optional, Tuple, Dict, Any
import requests
from dotenv import load_dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain, PromptTemplate

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Azure Maps config
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")
AZURE_MAPS_ENDPOINT = os.getenv("AZURE_MAPS_ENDPOINT").rstrip("/") if os.getenv("AZURE_MAPS_ENDPOINT") else None

# LangChain / Azure LLM config
LLM = AzureChatOpenAI(
    deployment_name=os.getenv("AI_MODEL_DEPLOYMENT_NAME") or os.getenv("AI_MODEL_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_AI_KEY"),
    openai_api_base=os.getenv("AZURE_AI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
)

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["coords", "user_query", "forecast_snippet"],
    template=(
        "Tu es un assistant météo concis et factuel.\n\n"
        "Contexte coordonnées: {coords}\n"
        "Question utilisateur: {user_query}\n\n"
        "Voici un extrait JSON des données météo brutes (format API Azure Maps):\n"
        "{forecast_snippet}\n\n"
        "1) Donne une réponse courte à la question de l'utilisateur, basée uniquement sur ces données.\n"
        "2) Fournis un résumé structuré par jour (date si disponible, min/max si disponibles, probabilite de pluie si disponible).\n"
        "3) Si les données ne permettent pas de répondre précisément, indique les limites et propose une action simple (ex: vérifier coords, demander plus de précision).\n\n"
        "Répond en français, pas plus de 250 mots."
    ),
)
SUMMARIZE_CHAIN = LLMChain(llm=LLM, prompt=SUMMARIZE_PROMPT)


COORDS_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[,;\\s]\s*(-?\d+(?:\.\d+)?)")


def _parse_coords(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    m = COORDS_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def get_coordinates_from_location(location: str) -> Optional[Tuple[float, float]]:
    """Résout un nom d'adresse/lieu en coordonnées via Azure Maps."""
    # si le message contient déjà des coords
    coords = _parse_coords(location)
    if coords:
        return coords

    if not AZURE_MAPS_ENDPOINT or not AZURE_MAPS_KEY:
        logging.error("AZURE_MAPS_ENDPOINT ou AZURE_MAPS_KEY manquante.")
        return None

    url = f"{AZURE_MAPS_ENDPOINT}/search/address/json"
    params = {
        "api-version": "1.0",
        "subscription-key": AZURE_MAPS_KEY,
        "query": location,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            pos = data["results"][0].get("position")
            if pos and "lat" in pos and "lon" in pos:
                return float(pos["lat"]), float(pos["lon"])
    except Exception as e:
        logging.exception("Erreur get_coordinates_from_location pour '%s': %s", location, e)
    return None


def get_weather_forecast(lat: float, lon: float, duration: int = 5) -> Optional[Dict[str, Any]]:
    """Récupère la prévision quotidienne via Azure Maps Weather API."""
    if not AZURE_MAPS_ENDPOINT or not AZURE_MAPS_KEY:
        logging.error("AZURE_MAPS_ENDPOINT ou AZURE_MAPS_KEY manquante.")
        return None

    url = f"{AZURE_MAPS_ENDPOINT}/weather/forecast/daily/json"
    params = {
        "api-version": "1.1",
        "subscription-key": AZURE_MAPS_KEY,
        "query": f"{lat},{lon}",
        "duration": duration,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.exception("Erreur get_weather_forecast pour %s,%s: %s", lat, lon, e)
        return None


def _prepare_forecast_snippet(forecast: Dict[str, Any], max_chars: int = 3000) -> str:
    """Séri alise partiellement le JSON pour l'envoyer au LLM. Tronque si nécessaire."""
    raw = json.dumps(forecast, ensure_ascii=False, indent=2)
    if len(raw) <= max_chars:
        return raw
    # essayer d'extraire seulement le noeud 'forecasts' ou 'dailyForecasts' si présent
    for key in ("forecasts", "dailyForecasts", "data"):
        if key in forecast:
            snippet = json.dumps({key: forecast[key][:10]}, ensure_ascii=False, indent=2)
            if len(snippet) <= max_chars:
                return snippet
    # fallback: tronquer le raw
    return raw[: max_chars - 50] + "\n... (truncated)\n"


def call_weather_agent(user_message: str, duration: int = 5, use_llm_summary: bool = True) -> Dict[str, Any]:
    """
    Interface principale pour l'orchestrateur.
    - Accepte un message contenant un nom de lieu, une question météo ou des coords 'lat,lon'.
    - Retourne dict structuré { input, resolved_query, coords, forecast_raw, summary, error? }.
    """
    out: Dict[str, Any] = {
        "input": user_message,
        "resolved_query": None,
        "coords": None,
        "forecast_raw": None,
        "summary": None,
    }

    if not user_message or not user_message.strip():
        out["error"] = "Message utilisateur vide."
        return out

    resolved_query = user_message.strip()
    coords = _parse_coords(user_message)
    if coords:
        lat, lon = coords
        resolved_query = f"{lat},{lon}"
    out["resolved_query"] = resolved_query

    if not coords:
        found = get_coordinates_from_location(resolved_query)
        if not found:
            out["error"] = f"Impossible de résoudre la localisation '{resolved_query}'."
            return out
        lat, lon = found
    out["coords"] = {"lat": lat, "lon": lon}

    forecast = get_weather_forecast(lat, lon, duration=duration)
    if forecast is None:
        out["error"] = "Erreur lors de la récupération des données météo."
        return out
    out["forecast_raw"] = forecast

    # résumé par LLM ou résumé déterministe simple si LLM désactivé
    if use_llm_summary:
        try:
            snippet = _prepare_forecast_snippet(forecast)
            llm_input = {
                "coords": f"{lat},{lon}",
                "user_query": user_message,
                "forecast_snippet": snippet,
            }
            summary = SUMMARIZE_CHAIN.run(llm_input)
            out["summary"] = summary.strip()
        except Exception as e:
            logging.exception("Erreur synthèse LLM: %s", e)
            out["summary"] = "Erreur lors de la génération du résumé par LLM."
    else:
        # fallback basique: lister premiers jours
        days = forecast.get("forecasts") or forecast.get("dailyForecasts") or forecast.get("data") or []
        lines = []
        for i, d in enumerate(days[:duration]):
            date = d.get("date") or d.get("validTimeLocal") or f"Jour {i+1}"
            mn = d.get("temperatureMin") or d.get("minTemp") or d.get("temperature", {}).get("min")
            mx = d.get("temperatureMax") or d.get("maxTemp") or d.get("temperature", {}).get("max")
            pop = d.get("precipitationProbability") or d.get("pop")
            lines.append(f"{date}: {mn or '?'}° / {mx or '?'}° — Prob pluie {pop or '?'}%")
        out["summary"] = "\n".join(lines) if lines else "Aucune donnée quotidienne disponible."

    return out


# usage rapide
if __name__ == "__main__":
    q = input("Lieu ou coordonnées (lat,lon) et question (ex: 'Paris, pluie demain?'): ")
    res = call_weather_agent(q, duration=5, use_llm_summary=True)
    print(json.dumps(res, ensure_ascii=False, indent=2))
