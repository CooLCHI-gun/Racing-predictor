"""
HKJC 賽馬預測系統 - 歷史賽果抓取模組
Fetches actual race results from HKJC for post-race analysis and model training.

Results URL:
    https://racing.hkjc.com/en-us/local/information/localresults
  ?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=1

Also fetches:
  - Win/Place dividends
  - Trio dividends
  - Full finishing order

Used by:
  1. ELO auto-update (after each race day)
  2. Backtest result recording
  3. Historical data accumulation for model training
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RESULTS_URL = "https://racing.hkjc.com/en-us/local/information/localresults"
RESULTS_BASE_URL = "https://racing.hkjc.com/en-us/local/information/localresults"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://racing.hkjc.com/",
}

HISTORICAL_DATA_FILE = "data/processed/historical_results.json"


@dataclass
class RaceResult:
    """Full result for a single race including dividends."""
    race_id: str                       # "{date}_{venue}_{race_num}"
    race_date: str                     # "YYYY-MM-DD"
    race_number: int
    venue_code: str                    # "ST" | "HV"
    distance: int
    track_type: str
    race_class: str
    going: str

    # Finishing order — list of horse numbers in finishing position order
    finishing_order: List[int] = field(default_factory=list)
    # Winner info
    winner_horse_num: int = 0
    winner_horse_name: str = ""
    winner_jockey: str = ""
    winner_trainer: str = ""

    # Dividends (per $10 stake)
    win_dividend: float = 0.0          # Win on horse #1
    place_dividends: Dict[str, float] = field(default_factory=dict)  # {horse_num_str: amount}
    trio_dividend: float = 0.0
    first_second_dividend: float = 0.0  # 連贏 (Quinella)
    quinella_place_dividend: float = 0.0

    # Full details for model training
    all_horses: List[Dict] = field(default_factory=list)  # [{horse_num, finish, time, ...}]
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _fetch_html(url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[str]:
    session = requests.Session()
    session.headers.update(HEADERS)
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt}/{max_retries} for {url}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_dividend(text: str) -> float:
    """Parse dividend text like '$123.50' or '123.5' to float."""
    clean = re.sub(r"[^\d.]", "", text)
    try:
        return float(clean)
    except ValueError:
        return 0.0


def _parse_results_page(html: str, race_number: int, race_date: str, venue_code: str) -> Optional[RaceResult]:
    """Parse a single race results page."""
    soup = BeautifulSoup(html, "lxml")

    if "No information" in html or "no information" in html.lower():
        return None

    race_id = f"{race_date}_{venue_code}_{race_number}"
    result = RaceResult(
        race_id=race_id,
        race_date=race_date,
        race_number=race_number,
        venue_code=venue_code,
        distance=0,
        track_type="Turf",
        race_class="",
        going="GOOD",
    )

    try:
        # ── Race header info ──────────────────────────────────────────────
        header_text = soup.get_text()

        dist_match = re.search(r"(\d{3,4})[mM]", header_text)
        if dist_match:
            result.distance = int(dist_match.group(1))

        if "All Weather" in header_text:
            result.track_type = "AWT"

        class_match = re.search(r"Class (\d)", header_text)
        if class_match:
            result.race_class = f"Class {class_match.group(1)}"

        # Going
        for going_kw in ["GOOD TO YIELDING", "GOOD TO FIRM", "YIELDING", "SOFT", "HEAVY", "FAST", "GOOD"]:
            if going_kw in header_text.upper():
                result.going = going_kw
                break

        # ── Finishing order table ─────────────────────────────────────────
        # Strictly target the official finishing table by header signature.
        parsed_rows: List[Tuple[int, int, str, str, str]] = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            header_cells = rows[0].find_all(["th", "td"])
            header_text = " ".join(c.get_text(" ", strip=True).lower() for c in header_cells)
            if not any(k in header_text for k in ["pla", "place"]):
                continue
            if not any(k in header_text for k in ["no", "horse no", "horse"]):
                continue

            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                texts = [c.get_text(strip=True) for c in cells]
                pos_text = texts[0] if len(texts) > 0 else ""
                horse_num_text = texts[1] if len(texts) > 1 else ""
                if not (pos_text.isdigit() and horse_num_text.isdigit()):
                    continue

                pos = int(pos_text)
                horse_num = int(horse_num_text)
                horse_name = texts[2] if len(texts) > 2 else ""
                jockey = texts[3] if len(texts) > 3 else ""
                trainer = texts[4] if len(texts) > 4 else ""
                parsed_rows.append((pos, horse_num, horse_name, jockey, trainer))

            if parsed_rows:
                break

        parsed_rows.sort(key=lambda x: x[0])
        result.finishing_order = [horse_num for _, horse_num, _, _, _ in parsed_rows]

        for pos, horse_num, horse_name, jockey, trainer in parsed_rows:
            result.all_horses.append({
                "horse_num": horse_num,
                "horse_name": horse_name,
                "finish_pos": pos,
                "jockey": jockey,
                "trainer": trainer,
            })
            if pos == 1:
                result.winner_horse_num = horse_num
                result.winner_horse_name = horse_name
                result.winner_jockey = jockey
                result.winner_trainer = trainer

        # ── Dividends table ───────────────────────────────────────────────
        # Look for dividend amounts in the page
        # Win dividend
        win_match = re.search(r"Win\s*\$?([\d,.]+)", header_text)
        if win_match:
            result.win_dividend = _parse_dividend(win_match.group(1))

        # Trio dividend
        trio_match = re.search(r"Trio\s*\$?([\d,.]+)", header_text)
        if trio_match:
            result.trio_dividend = _parse_dividend(trio_match.group(1))

        logger.debug(
            f"Parsed R{race_number}: winner=#{result.winner_horse_num} "
            f"({result.winner_horse_name}), order={result.finishing_order[:5]}"
        )
        return result if result.finishing_order else None

    except Exception as e:
        logger.warning(f"Error parsing results for race {race_number}: {e}")
        return None


# ── Main public functions ─────────────────────────────────────────────────────

def fetch_day_results(race_date: str, venue_code: str = "ST") -> List[RaceResult]:
    """
    Fetch all race results for a given race day.

    Args:
        race_date: "YYYY-MM-DD"
        venue_code: "ST" | "HV"

    Returns:
        List of RaceResult objects for each completed race
    """
    from config import config
    if config.DEMO_MODE:
        return []

    date_param = race_date.replace("-", "/")
    results: List[RaceResult] = []

    logger.info(f"Fetching results for {race_date} at {venue_code}")

    for race_num in range(1, 13):
        params = {
            "racedate": date_param,
            "Racecourse": venue_code,
            "RaceNo": str(race_num),
        }
        html = _fetch_html(RESULTS_URL, params=params)

        if not html or "No information" in html:
            logger.info(f"No more results after race {race_num - 1}")
            break

        result = _parse_results_page(html, race_num, race_date, venue_code)
        if result and result.finishing_order:
            results.append(result)
            logger.info(
                f"R{race_num}: #{result.winner_horse_num} {result.winner_horse_name} wins"
            )
        else:
            logger.warning(f"Could not parse results for R{race_num}")

        time.sleep(0.4)  # polite delay

    logger.info(f"Fetched {len(results)} race results for {race_date}")
    return results


def fetch_historical_results(
    start_date: str,
    end_date: Optional[str] = None,
    venue_code: str = "ST",
) -> List[RaceResult]:
    """
    Fetch historical results for a date range (for model training).

    Args:
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD" (defaults to today)
        venue_code: "ST" | "HV"

    Returns:
        List of all RaceResult objects in date range
    """
    from config import config
    if config.DEMO_MODE:
        return []

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    all_results: List[RaceResult] = []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    current = start_dt

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        day_results = fetch_day_results(date_str, venue_code)
        all_results.extend(day_results)

        if day_results:
            logger.info(f"{date_str}: fetched {len(day_results)} races")
            # Polite delay between race days
            time.sleep(1.0)

        current += timedelta(days=1)

    return all_results


# ── Historical data storage ───────────────────────────────────────────────────

def save_results_to_history(results: List[RaceResult], filepath: str = HISTORICAL_DATA_FILE) -> None:
    """Append race results to the historical JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Load existing
    existing: Dict[str, Dict] = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing history: {e}")

    # Merge new results
    for r in results:
        existing[r.race_id] = asdict(r)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(existing)} total results to {filepath}")


def load_results_history(filepath: str = HISTORICAL_DATA_FILE) -> List[RaceResult]:
    """Load all historical race results from JSON file."""
    if not os.path.exists(filepath):
        logger.info("No historical results file found")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for item in data.values():
            try:
                results.append(RaceResult(**item))
            except Exception:
                pass
        logger.info(f"Loaded {len(results)} historical results")
        return results
    except Exception as e:
        logger.error(f"Failed to load historical results: {e}")
        return []


def get_race_days_in_range(start_date: str, end_date: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Get list of (date, venue_code) for known HKJC race days in a range.
    HKJC races Wednesday/Saturday at Sha Tin, and mid-week Happy Valley nights.
    This is an approximation — real race schedule should be fetched from HKJC.

    Returns list of (date_str, venue_code) tuples.
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    race_days = []
    current = start_dt

    while current <= end_dt:
        weekday = current.weekday()  # 0=Mon, 2=Wed, 5=Sat, 6=Sun
        if weekday == 5:   # Saturday → Sha Tin
            race_days.append((current.strftime("%Y-%m-%d"), "ST"))
        elif weekday == 2:  # Wednesday → alternate ST/HV (rough approximation)
            race_days.append((current.strftime("%Y-%m-%d"), "ST"))
        elif weekday == 6:  # Sunday → sometimes Sha Tin (international meets)
            pass
        current += timedelta(days=1)

    return race_days
