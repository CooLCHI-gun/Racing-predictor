"""
HKJC 賽馬預測系統 - 天氣及場地狀況抓取模組
Fetches track condition and weather from HKJC / Hong Kong Observatory API.
"""
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

HKO_CURRENT_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc"
HKO_FORECAST_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=tc"


@dataclass
class WeatherInfo:
    """Weather and track condition information."""
    going: str              # "GOOD" | "GOOD TO YIELDING" | "YIELDING" | "SOFT" | "HEAVY" | "FAST"
    track_condition: str    # Raw going string
    temperature: float      # Celsius
    humidity: int           # Percentage (0-100)
    rainfall_mm: float      # Rainfall in mm (last hour)
    wind_speed: int         # km/h
    wind_direction: str     # e.g. "Northeast"
    weather_desc: str       # Human-readable description
    venue: str              # "ST" | "HV"
    is_raining: bool

    @property
    def going_code(self) -> int:
        """Numeric encoding of going for model features. Firmer = lower."""
        mapping = {
            "FAST": 0,
            "GOOD": 1,
            "GOOD TO YIELDING": 2,
            "YIELDING": 3,
            "SOFT": 4,
            "HEAVY": 5,
        }
        return mapping.get(self.going.upper(), 1)


def _fetch_json(url: str, max_retries: int = 3) -> Optional[dict]:
    """Fetch JSON from URL with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Attempt {attempt}/{max_retries} for {url}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def _rainfall_to_going(rainfall_mm: float, humidity: int) -> str:
    """
    Estimate track going from rainfall and humidity.

    This is an approximation; actual going is set by HKJC officials.
    """
    if rainfall_mm >= 10:
        return "HEAVY"
    elif rainfall_mm >= 5:
        return "SOFT"
    elif rainfall_mm >= 1:
        return "YIELDING"
    elif rainfall_mm > 0 or humidity > 90:
        return "GOOD TO YIELDING"
    else:
        return "GOOD"


def fetch_weather(venue: str = "ST") -> WeatherInfo:
    """
    Fetch current weather and infer track going condition.

    Combines HKO current weather API with HKJC track condition scraping.

    Args:
        venue: "ST" (Sha Tin) or "HV" (Happy Valley)

    Returns:
        WeatherInfo object
    """
    from config import config

    if config.DEMO_MODE:
        return mock_weather(venue)

    data = _fetch_json(HKO_CURRENT_URL)
    if not data:
        logger.warning("HKO API unavailable; using mock weather")
        return mock_weather(venue)

    try:
        # Parse HKO JSON structure
        # NOTE: HKO API returns Chinese place names, NOT English.
        # ST station: "沙田"  |  HV station: "跑馬地"  |  fallback: "香港天文台"
        temp = 22.0
        humidity = 75
        rainfall = 0.0
        wind_speed = 10
        wind_dir = "East"

        # Temperature — match Chinese station names
        ST_TEMP_STATIONS = {"沙田", "大埔"}
        HV_TEMP_STATIONS = {"跑馬地", "黃竹坑", "九龍城"}
        FALLBACK_TEMP_STATIONS = {"香港天文台", "京士柏"}

        if "temperature" in data:
            temps = data["temperature"].get("data", [])
            # Build lookup dict
            temp_map = {s.get("place", ""): s.get("value") for s in temps if s.get("value") is not None}
            # Try venue-specific stations first
            target_stations = ST_TEMP_STATIONS if venue == "ST" else HV_TEMP_STATIONS
            matched = next((temp_map[k] for k in target_stations if k in temp_map), None)
            if matched is None:
                matched = next((temp_map[k] for k in FALLBACK_TEMP_STATIONS if k in temp_map), None)
            if matched is not None:
                temp = float(matched)

        # Humidity — use HKO Observatory station (only one returned)
        if "humidity" in data:
            hum_data = data["humidity"].get("data", [])
            if hum_data:
                humidity = int(hum_data[0].get("value", 75))

        # Rainfall — match Chinese district names near racecourses
        # ST ≈ "沙田"  |  HV ≈ "灣仔" / "中西區" (nearest)
        ST_RAIN_DISTRICTS = {"沙田"}
        HV_RAIN_DISTRICTS = {"灣仔", "中西區", "九龍城", "東區"}

        if "rainfall" in data:
            rain_data = data["rainfall"].get("data", [])
            rain_map = {s.get("place", ""): s.get("max", 0) for s in rain_data}
            target_rain = ST_RAIN_DISTRICTS if venue == "ST" else HV_RAIN_DISTRICTS
            rain_val = next((rain_map[k] for k in target_rain if k in rain_map), None)
            if rain_val is None:
                rain_val = next(iter(rain_map.values()), 0)
            try:
                rainfall = float(rain_val) if rain_val is not None else 0.0
            except (TypeError, ValueError):
                rainfall = 0.0

        # Wind
        if "wind" in data:
            wind_data = data["wind"].get("data", [])
            if wind_data:
                wind_speed = int(wind_data[0].get("speed", 10))
                wind_dir = wind_data[0].get("direction", "East")

        # Going: use HKJC official going from wind tracker, fall back to rainfall estimate
        try:
            from scraper.racecard import fetch_going as _fetch_going_official
            official_going = _fetch_going_official()
            going = official_going if official_going else _rainfall_to_going(rainfall, humidity)
        except Exception:
            going = _rainfall_to_going(rainfall, humidity)

        desc_list = data.get("weatherForecast", {}).get("forecastDesc", "") if isinstance(data, dict) else ""
        weather_desc = str(desc_list)[:100] if desc_list else "Fine"

        return WeatherInfo(
            going=going,
            track_condition=going,
            temperature=temp,
            humidity=humidity,
            rainfall_mm=rainfall,
            wind_speed=wind_speed,
            wind_direction=str(wind_dir),
            weather_desc=weather_desc,
            venue=venue,
            is_raining=rainfall > 0,
        )

    except Exception as e:
        logger.error(f"Error parsing weather data: {e}")
        return mock_weather(venue)


def mock_weather(venue: str = "ST") -> WeatherInfo:
    """
    Return realistic mock weather for offline testing.

    Args:
        venue: "ST" or "HV"

    Returns:
        Mock WeatherInfo
    """
    scenarios = [
        dict(going="GOOD", temperature=24.0, humidity=65, rainfall_mm=0.0,
             wind_speed=12, wind_direction="Northeast", weather_desc="晴天", is_raining=False),
        dict(going="GOOD TO YIELDING", temperature=22.0, humidity=80, rainfall_mm=0.2,
             wind_speed=18, wind_direction="East", weather_desc="多雲", is_raining=False),
        dict(going="YIELDING", temperature=20.0, humidity=88, rainfall_mm=1.5,
             wind_speed=22, wind_direction="Southeast", weather_desc="有雨", is_raining=True),
        dict(going="GOOD", temperature=26.5, humidity=60, rainfall_mm=0.0,
             wind_speed=8, wind_direction="South", weather_desc="天晴", is_raining=False),
        dict(going="FAST", temperature=25.0, humidity=55, rainfall_mm=0.0,
             wind_speed=5, wind_direction="Southwest", weather_desc="天晴（全天候賽道）", is_raining=False),
    ]

    rng = random.Random(hash(venue + str(random.randint(0, 4))))
    chosen = rng.choice(scenarios)

    return WeatherInfo(
        going=chosen["going"],
        track_condition=chosen["going"],
        temperature=chosen["temperature"],
        humidity=chosen["humidity"],
        rainfall_mm=chosen["rainfall_mm"],
        wind_speed=chosen["wind_speed"],
        wind_direction=chosen["wind_direction"],
        weather_desc=chosen["weather_desc"],
        venue=venue,
        is_raining=chosen["is_raining"],
    )
