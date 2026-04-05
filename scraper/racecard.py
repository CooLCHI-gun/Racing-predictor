"""
HKJC 賽馬預測系統 - 賽事表抓取模組
Fetches today's race card from HKJC website.

Real URL: https://racing.hkjc.com/en-us/local/information/racecard
         ?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=1

Structure confirmed via browser inspection (2026-04-02):
- Modern SPA URL, server-side rendered tables
- No meaningful id/class on main table — find by 'MY Race Card LIST' header
- Horse profile URL: https://racing.hkjc.com/en-us/local/information/horse?horseid=HK_{YEAR}_{CODE}
"""
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
HKT = ZoneInfo("Asia/Hong_Kong")

BASE_URL = "https://racing.hkjc.com"
RACECARD_URL = "https://racing.hkjc.com/zh-hk/local/information/racecard"
RACECARD_FALLBACK_URL = "https://racing.hkjc.com/en-us/local/information/racecard"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    # Note: Do NOT set Accept-Encoding — let requests handle decompression automatically
    "Connection": "keep-alive",
    "Referer": "https://racing.hkjc.com/",
}


@dataclass
class HorseEntry:
    """Single horse entry in a race."""
    horse_number: int
    horse_name: str
    horse_name_ch: str
    horse_code: str          # e.g. HK_2025_L126
    jockey: str
    jockey_code: str         # e.g. PZ
    jockey_allowance: int    # weight allowance in lbs (3, 5, 7, or 0)
    trainer: str
    trainer_code: str        # e.g. MKL
    draw: int                # Gate / barrier number
    weight: int              # Weight carried in lbs
    handicap_rating: int     # Official handicap rating
    rating_change: int       # Rtg.+/- relative to declaration weight
    last_6_runs: str         # e.g. "1/3/5/2/4/7"
    gear: str                # Equipment codes e.g. "TT", "B", "XB"
    age: int = 0
    country: str = "HK"
    priority: str = ""       # "+" Trump, "*" priority, "1"/"2" trainer pref


@dataclass
class Race:
    """A single race on the race card."""
    race_number: int
    race_name: str           # e.g. "LUGARD HANDICAP"
    venue: str               # "Sha Tin" | "Happy Valley"
    venue_code: str          # "ST" | "HV"
    distance: int            # In metres
    track_type: str          # "Turf" | "All Weather Track"
    track_config: str        # "A" | "B" | "B+2" | "C" | "C+3"
    race_class: str          # "Class 1" ... "Class 5" | "Griffin"
    class_rating_upper: int  # e.g. 100
    class_rating_lower: int  # e.g. 80
    going: str               # "GOOD" | "GOOD TO YIELDING" etc.
    prize: int               # Prize money in HKD
    start_time: str          # "HH:MM" HKT
    race_date: str           # "YYYY-MM-DD"
    horses: List[HorseEntry] = field(default_factory=list)


# ── Session helpers ─────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def _fetch_html(url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[str]:
    """
    Fetch a URL with retry + exponential backoff.
    Returns HTML string or None on failure.
    """
    session = _make_session()
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Fetching {url} (attempt {attempt}/{max_retries})")
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    logger.error(f"All {max_retries} attempts failed for {url}")
    return None


def _fetch_racecard_html(params: Optional[Dict] = None, max_retries: int = 3) -> Optional[str]:
    """Fetch racecard HTML with locale fallback (zh-hk -> en-us)."""
    html = _fetch_html(RACECARD_URL, params=params, max_retries=max_retries)
    if html:
        return html
    return _fetch_html(RACECARD_FALLBACK_URL, params=params, max_retries=max_retries)


def _meeting_exists(race_date: str, venue_code: str) -> bool:
    """Return True if HKJC has racecard info for the given date/venue race 1."""
    params = {
        "racedate": race_date.replace("-", "/"),
        "Racecourse": venue_code,
        "RaceNo": "1",
    }
    html = _fetch_racecard_html(params=params, max_retries=2)
    if not html:
        return False
    low = html.lower()
    if "no information" in low or "沒有資料" in html or "暫無資料" in html:
        return False
    return ("MY Race Card LIST" in html) or ("我 的 排 位 表" in html) or ("我的排位表" in html)


def find_next_meeting_date(start_date: str, max_days: int = 7) -> Optional[Tuple[str, str]]:
    """
    Find next available meeting date and venue by probing racecard pages.

    Returns:
        (YYYY-MM-DD, venue_code) or None when no meeting detected.
    """
    try:
        base = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        return None

    for offset in range(1, max_days + 1):
        d = (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        for venue in ("ST", "HV"):
            if _meeting_exists(d, venue):
                return d, venue
    return None


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_jockey_allowance(jockey_text: str) -> Tuple[str, int]:
    """
    Parse jockey name and weight allowance from combined text like "E C W Wong (-3)".
    Returns (clean_name, allowance_lbs).
    """
    match = re.search(r"\((-?\d+)\)", jockey_text)
    allowance = abs(int(match.group(1))) if match else 0
    clean = re.sub(r"\s*\(-?\d+\)\s*", "", jockey_text).strip()
    return clean, allowance


def _extract_code_from_href(href: str, param: str) -> str:
    """Extract query param value from a URL string."""
    match = re.search(rf"{param}=([^&]+)", href or "")
    return match.group(1) if match else ""


def _parse_race_header(header_text: str) -> Dict:
    """
    Parse race metadata from the text block above the main starter table.
    Expected format: 'Race 1 - LUGARD HANDICAP\nMonday, April 06, 2026, Sha Tin, 12:30\nTurf, "B+2" Course, 1000M\nPrize Money: $1,170,000, Rating: 60-40, Class 4'
    """
    result = {
        "race_name": "",
        "venue": "沙田",
        "venue_code": "ST",
        "start_time": "12:00",
        "distance": 1200,
        "track_type": "Turf",
        "track_config": "A",
        "race_class": "Class 4",
        "class_rating_upper": 60,
        "class_rating_lower": 40,
        "prize": 1_000_000,
        "going": "",          # filled below if found in header text
    }

    # Race name — extract only the race title, not the rest of the header
    name_match = re.search(r"Race \d+ - ([A-Z][A-Z'\s-]+?)(?:\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|$)", header_text)
    if name_match:
        result["race_name"] = name_match.group(1).strip()
    elif "Race" in header_text:
        # Fallback: take text after "Race N - " up to first comma/newline
        fb = re.search(r"Race \d+\s*[-–]\s*(.+?)(?:[,\n]|$)", header_text)
        if fb:
            result["race_name"] = fb.group(1).strip()[:50]

    # Venue
    if "Happy Valley" in header_text or "HV" in header_text or "跑馬地" in header_text:
        result["venue"] = "跑馬地"
        result["venue_code"] = "HV"
    elif "Sha Tin" in header_text or "ST" in header_text or "沙田" in header_text:
        result["venue"] = "沙田"
        result["venue_code"] = "ST"

    # Start time — find HH:MM pattern
    time_match = re.search(r"(\d{1,2}:\d{2})", header_text)
    if time_match:
        result["start_time"] = time_match.group(1)

    # Distance — find NNNNm or NNNNM
    dist_match = re.search(r"(\d{3,4})[mM]", header_text)
    if dist_match:
        result["distance"] = int(dist_match.group(1))

    # Track type
    if "All Weather" in header_text or "AWT" in header_text or "全天候" in header_text:
        result["track_type"] = "全天候跑道"
    elif "Turf" in header_text or "草地" in header_text:
        result["track_type"] = "草地"

    # Course config e.g. "A+3", "B+2", "C"
    course_match = re.search(r'"([A-C][+\d]*)"', header_text)
    if course_match:
        result["track_config"] = course_match.group(1)

    # Class
    class_match = re.search(r"Class (\d)", header_text)
    if class_match:
        result["race_class"] = f"Class {class_match.group(1)}"
    class_match_ch = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*班", header_text)
    if class_match_ch:
        result["race_class"] = f"第{class_match_ch.group(1)}班"

    # Rating range e.g. "Rating: 60-40"
    rating_match = re.search(r"Rating[:\s]+(\d+)-(\d+)", header_text)
    if rating_match:
        result["class_rating_upper"] = int(rating_match.group(1))
        result["class_rating_lower"] = int(rating_match.group(2))
    rating_match_ch = re.search(r"評分[:：]\s*(\d+)-(\d+)", header_text)
    if rating_match_ch:
        result["class_rating_upper"] = int(rating_match_ch.group(1))
        result["class_rating_lower"] = int(rating_match_ch.group(2))

    # Prize money
    prize_match = re.search(r"\$([\d,]+)", header_text)
    if prize_match:
        result["prize"] = int(prize_match.group(1).replace(",", ""))

    # Going — scan header text for official going declaration
    # HKJC racecard shows e.g. "GOING : GOOD TO YIELDING" or just "GOOD"
    # Search longest keywords first to avoid "GOOD" matching before "GOOD TO YIELDING"
    for going_kw in ["GOOD TO YIELDING", "GOOD TO FIRM", "YIELDING", "SOFT", "HEAVY", "FAST", "GOOD"]:
        if going_kw in header_text.upper():
            result["going"] = going_kw
            break

    going_map_ch = {
        "好地": "好地",
        "好黏地": "好黏地",
        "黏地": "黏地",
        "軟地": "軟地",
        "大爛地": "大爛地",
        "快地": "快地",
    }
    for ch_text, mapped in going_map_ch.items():
        if ch_text in header_text:
            result["going"] = mapped
            break

    return result


def _parse_horse_row(row: Any) -> Optional[HorseEntry]:
    """
    Parse a single horse row from the 'MY Race Card LIST' inner table.

    HKJC column order (confirmed 2026-04-06):
    0: Horse No.
    1: Last 6 Runs  (e.g. "1/3/5/2/8/4" or "-" for debut)
    2: Colour/silks image  (skip)
    3: Horse name  (with <a href="?horseid=HK_...">)
    4: Wt. (weight in lbs)
    5: Jockey  (with allowance in brackets if applicable)
    6: Draw
    7: Trainer
    8: Rtg.
    9: Rtg.+/-
    10: Horse Wt. (declared)
    11: Priority
    12: Gear
    """
    try:
        cells = row.find_all("td")
        if len(cells) < 8:
            return None

        texts = [c.get_text(strip=True) for c in cells]

        # Horse number must be numeric
        horse_num_text = texts[0].strip()
        if not horse_num_text.isdigit():
            return None
        horse_number = int(horse_num_text)

        # Horse name & code from link
        horse_name = ""
        horse_code = ""
        horse_name_ch = ""
        if len(cells) > 3:
            name_cell = cells[3]
            name_link = name_cell.find("a")
            if name_link:
                href = name_link.get("href", "")
                horse_code = _extract_code_from_href(href, "horseid")
                # Name may have English + Chinese on separate lines
                name_texts = [t.strip() for t in name_link.stripped_strings]
                horse_name = name_texts[0] if name_texts else ""
                horse_name_ch = name_texts[1] if len(name_texts) > 1 else ""
                if horse_name and any("\u4e00" <= ch <= "\u9fff" for ch in horse_name):
                    horse_name_ch = horse_name
            else:
                horse_name = texts[3]
                if horse_name and any("\u4e00" <= ch <= "\u9fff" for ch in horse_name):
                    horse_name_ch = horse_name

        # Jockey (col 6) + allowance — CONFIRMED live page 2026-04-06
        jockey_raw = texts[6] if len(texts) > 6 else ""
        jockey_name, allowance = _parse_jockey_allowance(jockey_raw)
        jockey_code = ""
        if len(cells) > 6:
            jockey_link = cells[6].find("a")
            if jockey_link:
                href = jockey_link.get("href", "")
                jockey_code = _extract_code_from_href(href, "jockeyid")

        # Trainer (col 9)
        trainer_name = texts[9] if len(texts) > 9 else ""
        trainer_code = ""
        if len(cells) > 9:
            trainer_link = cells[9].find("a")
            if trainer_link:
                href = trainer_link.get("href", "")
                trainer_code = _extract_code_from_href(href, "trainerid")

        # Draw (col 8)
        draw_text = texts[8] if len(texts) > 8 else "0"
        draw = int(draw_text) if draw_text.isdigit() else 0

        # Weight (col 5)
        weight_text = texts[5] if len(texts) > 5 else "126"
        weight = int(weight_text) if weight_text.isdigit() else 126

        # Rating (col 11)
        rating_text = texts[11] if len(texts) > 11 else "0"
        rating = int(rating_text) if rating_text.lstrip("-").isdigit() else 0

        # Rating change (col 12)
        rtg_change_text = texts[12] if len(texts) > 12 else "0"
        rtg_change = int(rtg_change_text) if rtg_change_text.lstrip("+-").isdigit() else 0

        # Priority (col 20)
        priority = texts[20] if len(texts) > 20 else ""

        # Gear (col 22)
        gear = texts[22] if len(texts) > 22 else ""

        # Age (col 16), Days since last run (col 21)
        age_text = texts[16] if len(texts) > 16 else "0"
        age = int(age_text) if age_text.isdigit() else 0

        # Sire (col 24)
        sire = texts[24] if len(texts) > 24 else ""

        # Last 6 runs (col 1)
        last6 = texts[1] if len(texts) > 1 else "-"

        return HorseEntry(
            horse_number=horse_number,
            horse_name=horse_name,
            horse_name_ch=horse_name_ch,
            horse_code=horse_code,
            jockey=jockey_name,
            jockey_code=jockey_code,
            jockey_allowance=allowance,
            trainer=trainer_name,
            trainer_code=trainer_code,
            draw=draw,
            weight=weight,
            handicap_rating=rating,
            rating_change=rtg_change,
            last_6_runs=last6,
            gear=gear,
            age=age,
            priority=priority,
        )
    except (IndexError, ValueError) as e:
        logger.debug(f"Failed to parse horse row: {e}")
        return None


def _find_main_table(soup: BeautifulSoup):
    """
    Find the 'MY Race Card LIST' outer table, return the inner data table.
    Strategy: look for a table containing text 'MY Race Card LIST' in any descendant.
    """
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if (
            "MY Race Card LIST" in text
            or "我 的 排 位 表" in text
            or "我的排位表" in text
        ):
            inner = table.find("table", class_=re.compile(r"\bstarter\b"))
            if inner:
                return inner
            inner = table.find("table")
            return inner or table

    starter = soup.find("table", class_=re.compile(r"\bstarter\b"))
    if starter:
        return starter

    for table in soup.find_all("table"):
        headers = " ".join(th.get_text(" ", strip=True) for th in table.find_all(["th", "td"])[:30])
        if (
            ("馬名" in headers and "騎師" in headers and "練馬師" in headers)
            or ("Horse" in headers and "Jockey" in headers and "Trainer" in headers)
        ):
            return table
    return None


def _parse_single_race_page(html: str, race_number: int, race_date: str, going: str) -> Optional[Race]:
    """Parse one race card page (one race at a time)."""
    soup = BeautifulSoup(html, "lxml")

    # ── Extract race header ─────────────────────────────────────────────────
    # The header block is a text region before the main table
    header_text = ""
    # Look for text containing race title in both EN and ZH formats.
    for el in soup.find_all(string=re.compile(r"Race\s+\d+|第\s*\d+\s*場")):
        block = el.find_parent()
        if block:
            header_text = block.get_text(" ", strip=True)
            if len(header_text) > 20:
                break

    meta = _parse_race_header(header_text)

    # ── Extract horse entries ───────────────────────────────────────────────
    inner_table = _find_main_table(soup)
    horses: List[HorseEntry] = []

    if inner_table:
        rows = inner_table.find_all("tr")
        # Skip header rows (those that only contain <th> elements)
        for row in rows:
            if row.find("th"):
                continue
            entry = _parse_horse_row(row)
            if entry:
                horses.append(entry)

    if not horses:
        logger.warning(f"No horses parsed for race {race_number} on {race_date}")
        return None

    return Race(
        race_number=race_number,
        race_name=meta["race_name"],
        venue=meta["venue"],
        venue_code=meta["venue_code"],
        distance=meta["distance"],
        track_type=meta["track_type"],
        track_config=meta["track_config"],
        race_class=meta["race_class"],
        class_rating_upper=meta["class_rating_upper"],
        class_rating_lower=meta["class_rating_lower"],
        going=meta.get("going") or going,   # per-race HTML value preferred; fallback to windtracker
        prize=meta["prize"],
        start_time=meta["start_time"],
        race_date=race_date,
        horses=horses,
    )


# ── Going detection ──────────────────────────────────────────────────────────

def fetch_going(race_date: Optional[str] = None) -> str:
    """
    Fetch going condition from HKJC Wind Tracker page.
    Returns Traditional Chinese going text.
    Falls back to '好地' on failure.
    """
    url = "https://racing.hkjc.com/zh-hk/local/info/windtracker"
    html = _fetch_html(url)
    if not html:
        return "好地"
    try:
        soup = BeautifulSoup(html, "lxml")
        # Look for going value near 'Going' label
        going_keywords_ch = ["好黏地", "好快地", "黏地", "軟地", "大爛地", "快地", "好地"]
        going_keywords_en = ["GOOD TO YIELDING", "GOOD TO FIRM", "YIELDING", "SOFT", "HEAVY", "FAST", "GOOD"]
        text = soup.get_text()
        for kw in going_keywords_ch:
            if kw in text:
                return kw
        for kw in going_keywords_en:
            if kw in text.upper():
                return {
                    "GOOD TO YIELDING": "好黏地",
                    "GOOD TO FIRM": "好快地",
                    "YIELDING": "黏地",
                    "SOFT": "軟地",
                    "HEAVY": "大爛地",
                    "FAST": "快地",
                    "GOOD": "好地",
                }.get(kw, "好地")
    except Exception as e:
        logger.warning(f"Failed to parse going: {e}")
    return "好地"


# ── Next race date detection ─────────────────────────────────────────────────

def fetch_next_race_date() -> Optional[Tuple[str, str]]:
    """
    Fetch the next upcoming race date and venue from HKJC racecard page.
    Returns (date_str "YYYY-MM-DD", venue_code "ST"|"HV") or None.
    """
    html = _fetch_html(RACECARD_URL)
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
        # Look for "DD Apr - Sha Tin" or similar pattern
        text = soup.get_text()
        date_match = re.search(r"(\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+-\s+(Sha Tin|Happy Valley)", text)
        if date_match:
            day = date_match.group(1)
            month = date_match.group(2)
            venue_name = date_match.group(3)
            year = date.today().year
            dt = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
            venue_code = "HV" if "Happy Valley" in venue_name else "ST"
            return dt.strftime("%Y-%m-%d"), venue_code
    except Exception as e:
        logger.warning(f"Failed to detect next race date: {e}")
    return None


# ── Main public function ─────────────────────────────────────────────────────

def fetch_racecard(
    race_date: Optional[str] = None,
    venue_code: Optional[str] = None,
    auto_next_if_empty: bool = True,
) -> List[Race]:
    """
    Fetch the race card from HKJC for a given date.

    Args:
        race_date: Date in "YYYY-MM-DD" format. Defaults to today.
        venue_code: "ST" or "HV". Auto-detected if None.

    Returns:
        List of Race objects with horse entries.
    """
    from config import config

    if race_date is None:
        race_date = datetime.now(HKT).strftime("%Y-%m-%d")

    if config.DEMO_MODE:
        logger.info("DEMO_MODE enabled — returning mock racecard")
        return mock_racecard(race_date)

    # Convert YYYY-MM-DD → YYYY/MM/DD for HKJC URL
    date_param = race_date.replace("-", "/")

    # Auto-detect venue if not provided
    if venue_code is None:
        detected = fetch_next_race_date()
        venue_code = detected[1] if detected else "ST"

    # Fetch going condition once
    going = fetch_going(race_date)
    logger.info(f"Going condition: {going}")

    races: List[Race] = []
    race_num = 1

    while True:
        params = {
            "racedate": date_param,
            "Racecourse": venue_code,
            "RaceNo": str(race_num),
        }
        html = _fetch_racecard_html(params=params)
        if not html:
            logger.warning(f"Failed to fetch race {race_num}; stopping")
            break

        # Check if this race exists (page shows "No information" if not)
        if "No information" in html or "no information" in html.lower():
            logger.info(f"No more races after race {race_num - 1}")
            break

        race = _parse_single_race_page(html, race_num, race_date, going)
        if race is None:
            logger.warning(f"Could not parse race {race_num}; stopping")
            break

        races.append(race)
        logger.info(f"Parsed Race {race_num}: {race.race_name} ({len(race.horses)} runners)")
        race_num += 1

        # HKJC typically has max 11 races per day
        if race_num > 12:
            break

        time.sleep(0.5)  # polite delay

    if not races:
        if auto_next_if_empty:
            next_meeting = find_next_meeting_date(race_date)
            if next_meeting:
                next_date, next_venue = next_meeting
                logger.info(
                    f"No races on {race_date}; retrying next meeting {next_date} ({next_venue})"
                )
                return fetch_racecard(next_date, next_venue, auto_next_if_empty=False)

        if config.REAL_DATA_ONLY and not config.DEMO_MODE:
            logger.error("No races fetched from HKJC and REAL_DATA_ONLY=True; returning empty race list")
            return []
        logger.warning("No races fetched from HKJC; falling back to mock data")
        return mock_racecard(race_date)

    logger.info(f"Fetched {len(races)} races for {race_date} at {venue_code}")
    return races


# ── Results fetcher ──────────────────────────────────────────────────────────

def fetch_results(race_date: str, venue_code: str = "ST") -> Dict[int, List[int]]:
    """
    Fetch actual race results from HKJC results page.

    Args:
        race_date: "YYYY-MM-DD"
        venue_code: "ST" | "HV"

    Returns:
        Dict mapping race_number -> list of finishing positions as horse numbers
        e.g. {1: [3, 7, 1, 5, ...], 2: [...]}
    """
    from config import config
    if config.DEMO_MODE:
        return {}

    date_param = race_date.replace("-", "/")
    results_url = "https://racing.hkjc.com/en-us/local/information/localresults"
    results: Dict[int, List[int]] = {}

    for race_num in range(1, 13):
        params = {
            "racedate": date_param,
            "Racecourse": venue_code,
            "RaceNo": str(race_num),
        }
        html = _fetch_html(results_url, params=params)
        if not html or "No information" in html:
            break

        try:
            soup = BeautifulSoup(html, "lxml")
            # Results table: strictly match header to avoid parsing payout/summary tables.
            # Expected columns include Pla. | No. | Horse
            result_rows = []
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
                    if len(cells) < 2:
                        continue
                    pos_text = cells[0].get_text(strip=True)
                    horse_num_text = cells[1].get_text(strip=True)
                    if not (pos_text.isdigit() and horse_num_text.isdigit()):
                        continue
                    pos = int(pos_text)
                    horse_num = int(horse_num_text)
                    result_rows.append((pos, horse_num))

            result_rows.sort(key=lambda x: x[0])
            result_list = [horse for _, horse in result_rows]

            if result_list:
                results[race_num] = result_list
                logger.debug(f"Results R{race_num}: {result_list[:5]}")
        except Exception as e:
            logger.warning(f"Failed to parse results for race {race_num}: {e}")

        time.sleep(0.3)

    return results


# ── Mock data ────────────────────────────────────────────────────────────────

def mock_racecard(race_date: Optional[str] = None) -> List[Race]:
    """
    Return realistic mock racecard for offline testing.
    Includes real HKJC horse names, jockeys, and trainers (2025-26 season).
    """
    if race_date is None:
        race_date = date.today().strftime("%Y-%m-%d")

    import random
    race_templates = [
        dict(race_number=1, race_name="LUGARD HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=1000, track_type="Turf", track_config="C+3", race_class="Class 4",
             class_rating_upper=60, class_rating_lower=40, going="GOOD", prize=1_000_000, start_time="12:30"),
        dict(race_number=2, race_name="BOWEN HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=1200, track_type="Turf", track_config="B", race_class="Class 4",
             class_rating_upper=60, class_rating_lower=40, going="GOOD", prize=1_000_000, start_time="13:00"),
        dict(race_number=3, race_name="JARDINE HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=1400, track_type="Turf", track_config="A", race_class="Class 3",
             class_rating_upper=80, class_rating_lower=60, going="GOOD TO YIELDING", prize=1_200_000, start_time="13:30"),
        dict(race_number=4, race_name="HONG KONG CLASSIC CUP", venue="Sha Tin", venue_code="ST",
             distance=1800, track_type="Turf", track_config="B+2", race_class="Class 1",
             class_rating_upper=115, class_rating_lower=90, going="GOOD", prize=3_000_000, start_time="14:05"),
        dict(race_number=5, race_name="CLEARWATER BAY HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=1600, track_type="Turf", track_config="A", race_class="Class 2",
             class_rating_upper=100, class_rating_lower=80, going="GOOD", prize=1_500_000, start_time="14:40"),
        dict(race_number=6, race_name="TUEN MUN HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=1200, track_type="All Weather Track", track_config="A", race_class="Class 4",
             class_rating_upper=60, class_rating_lower=40, going="FAST", prize=1_000_000, start_time="15:15"),
        dict(race_number=7, race_name="SHEK O HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=1400, track_type="Turf", track_config="B", race_class="Class 3",
             class_rating_upper=80, class_rating_lower=60, going="GOOD", prize=1_200_000, start_time="15:50"),
        dict(race_number=8, race_name="REPULSE BAY HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=2000, track_type="Turf", track_config="B+2", race_class="Class 2",
             class_rating_upper=100, class_rating_lower=80, going="GOOD", prize=1_600_000, start_time="16:25"),
        dict(race_number=9, race_name="TAI TAM HANDICAP", venue="Sha Tin", venue_code="ST",
             distance=2400, track_type="Turf", track_config="A", race_class="Class 1",
             class_rating_upper=115, class_rating_lower=85, going="GOOD", prize=2_500_000, start_time="17:00"),
    ]

    # Real HKJC 2025-26 season horses/jockeys/trainers
    horses_pool = [
        ("ALMIGHTY LIGHTNING", "萬里神駒", "HK_2025_L126", "Z Purton",    "PZ",  "K L Man",      "MKL",  60),
        ("SUPER SIXTY",        "超級六十", "HK_2023_J071", "E C W Wong",   "WEC", "C Fownes",     "FC",   54),
        ("CASA BUDDY",         "家居好友", "HK_2024_K207", "M Chadwick",   "CML", "C W Chang",    "CCW",  52),
        ("GRACEFUL HEART",     "優雅之心", "HK_2024_K312", "L Hewitson",   "HEL", "K W Lui",      "LKW",  52),
        ("STAR PERFORMER",     "星級表演", "HK_2023_J201", "V Borges",     "BV",  "D Whyte",      "WD",   78),
        ("EAGLE WAY",          "雄鷹路",   "HK_2022_H155", "C Y Ho",       "HCY", "F C Lor",      "LFC",  85),
        ("GOOD SHOT",          "好一箭",   "HK_2023_J088", "B Prebble",    "PRE", "J Size",       "SJ",   70),
        ("FAIRY FLOSS",        "仙女棉花", "HK_2024_K401", "K Leung",      "LKH", "P O Sullivan", "OS",   62),
        ("LUCKY PATCH",        "幸運補丁", "HK_2022_H033", "N Callan",     "CN",  "C H Yip",      "YCH",  75),
        ("SOLAR WIND",         "太陽風",   "HK_2023_J145", "V Duric",      "DV",  "P F Yiu",      "YPF",  68),
        ("KING OF HEARTS",     "紅心之王", "HK_2021_G088", "A Hamelin",    "HA",  "R Gibson",     "GR",   92),
        ("THUNDER SOUND",      "雷聲",     "HK_2022_H199", "H T Mo",       "MHT", "A T Millard",  "MAT",  80),
        ("GOLD WELCOME",       "黃金歡迎", "HK_2024_K055", "U Rispoli",    "RU",  "D Hall",       "HD",   58),
        ("BRILLIANT KNIGHT",   "明亮騎士", "HK_2023_J333", "T H So",       "STH", "L Ho",         "HL",   65),
        ("IRON FIST",          "鐵拳",     "HK_2021_G201", "P Cosgrave",   "CP",  "K W Lui",      "LKW",  88),
        ("MISTER POTENTIAL",   "潛力先生", "HK_2022_H088", "Z Purton",     "PZ",  "C Fownes",     "FC",   95),
        ("PERFECT STORM",      "完美風暴", "HK_2020_F112", "J Moreira",    "MJ",  "J Size",       "SJ",   108),
        ("JADE TIGER",         "玉虎",     "HK_2023_J277", "M Chadwick",   "CML", "F C Lor",      "LFC",  72),
        ("FLYING BANNER",      "飛旗",     "HK_2024_K188", "L Hewitson",   "HEL", "D Whyte",      "WD",   55),
        ("CLASSIC HERO",       "經典英雄", "HK_2021_G155", "V Borges",     "BV",  "P O Sullivan", "OS",   100),
    ]

    races = []
    for tmpl in race_templates:
        random.seed(tmpl["race_number"] * 7919)
        n_horses = random.randint(10, 14)
        selected = random.sample(horses_pool, min(n_horses, len(horses_pool)))
        draws = list(range(1, len(selected) + 1))
        random.shuffle(draws)

        entries = []
        for i, (horse, draw) in enumerate(zip(selected, draws), start=1):
            name_en, name_ch, code, jockey, jk_code, trainer, tr_code, base_rating = horse
            class_adj = {"Class 5": -20, "Class 4": -10, "Class 3": 0,
                         "Class 2": 10, "Class 1": 20}.get(tmpl["race_class"], 0)
            rating = max(0, min(140, base_rating + class_adj + random.randint(-5, 5)))
            weight = max(113, min(133, 133 - (i - 1) * 2 + random.randint(-2, 2)))

            # Simulate last 6 runs
            last6 = "/".join([
                str(random.randint(1, 10)) if random.random() > 0.1 else random.choice(["WV", "DNF"])
                for _ in range(6)
            ])

            entries.append(HorseEntry(
                horse_number=i,
                horse_name=name_en,
                horse_name_ch=name_ch,
                horse_code=code,
                jockey=jockey,
                jockey_code=jk_code,
                jockey_allowance=random.choice([0, 0, 0, 3, 5]),
                trainer=trainer,
                trainer_code=tr_code,
                draw=draw,
                weight=weight,
                handicap_rating=rating,
                rating_change=random.randint(-5, 5),
                last_6_runs=last6,
                gear=random.choice(["", "TT", "B", "TT/B", "E/B", "XB", ""]),
                age=random.randint(3, 8),
                country=random.choice(["HK", "HK", "HK", "AUS", "IRE", "GB"]),
                priority=random.choice(["", "", "", "+", "*"]),
            ))

        races.append(Race(**tmpl, race_date=race_date, horses=entries))

    logger.info(f"Generated {len(races)} mock races for {race_date}")
    return races
