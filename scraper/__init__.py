"""
HKJC 賽馬預測系統 - 數據抓取模組
Scraper package for fetching HKJC race data.
"""
from scraper.racecard import fetch_racecard, mock_racecard, Race, HorseEntry
from scraper.horse_profile import fetch_horse_profile, mock_horse_profile, HorseProfile, PastRace
from scraper.odds import fetch_odds, mock_odds, calculate_implied_probability, calculate_expected_value
from scraper.results_fetcher import fetch_day_results, fetch_historical_results, RaceResult
from scraper.weather import fetch_weather, mock_weather, WeatherInfo

__all__ = [
    "fetch_racecard", "mock_racecard", "Race", "HorseEntry",
    "fetch_horse_profile", "mock_horse_profile", "HorseProfile", "PastRace",
    "fetch_odds", "mock_odds", "calculate_implied_probability", "calculate_expected_value",
    "fetch_day_results", "fetch_historical_results", "RaceResult",
    "fetch_weather", "mock_weather", "WeatherInfo",
]
