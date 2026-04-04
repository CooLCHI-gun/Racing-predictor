"""
HKJC 賽馬預測系統 - 馬匹資料抓取模組
Fetches horse past performance data from HKJC.

Real URL (confirmed 2026-04-02, server-side rendered, no JS needed):
  https://racing.hkjc.com/en-us/local/information/horse?horseid=HK_2025_L126

Past performance table column order (19 cols):
  0: Race Index (link)
  1: Pla. (finish position)
  2: Date (DD/MM/YY)
  3: RC/Track/Course   e.g. "ST / Turf / A+3"
  4: Dist. (metres)
  5: G (going code: G=Good, Y=Yielding, SY=Soft-Yielding, etc.)
  6: Race Class
  7: Dr. (draw)
  8: Rtg. (rating)
  9: Trainer
  10: Jockey
  11: LBW (lengths behind winner)
  12: Win Odds
  13: Act. Wt. (weight carried)
  14: Running Position (sectional)
  15: Finish Time
  16: Declar. Horse Wt.
  17: Gear
  18: Video Replay (skip)
"""
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HORSE_PROFILE_URL = "https://racing.hkjc.com/en-us/local/information/horse"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://racing.hkjc.com/",
}

# Going code mapping (HKJC single-letter codes)
GOING_MAP = {
    "G": "GOOD",
    "GY": "GOOD TO YIELDING",
    "GF": "GOOD TO FIRM",
    "Y": "YIELDING",
    "SY": "SOFT-YIELDING",
    "S": "SOFT",
    "H": "HEAVY",
    "F": "FAST",          # AWT only
    "TF": "TURFFAST",
}


@dataclass
class PastRace:
    """A single past race record for a horse."""
    race_date: str            # "YYYY-MM-DD"
    race_number: int
    venue: str                # "ST" | "HV"
    distance: int             # metres
    track_type: str           # "Turf" | "AWT"
    track_config: str         # "A", "B+2" etc.
    going: str                # "GOOD", "GOOD TO YIELDING" etc.
    going_code: str           # raw HKJC code e.g. "GY"
    race_class: str
    draw: int
    finish_position: int      # 1 = won, 99 = retired/WV/DNF
    runners: int              # estimated from field (not always available)
    jockey: str
    trainer: str
    weight: int               # actual weight carried (lbs)
    rating_at_race: int       # official rating on race day
    starting_price: float     # win odds at off
    winning_margin: float     # lengths behind winner (0.0 if won)
    finish_time: str          # "m:ss.ss"
    running_position: str     # e.g. "6-7-1" sectional positions


@dataclass
class HorseProfile:
    """Full profile and past performance of a horse."""
    horse_code: str
    horse_name: str
    horse_name_ch: str
    age: int
    country: str
    colour: str
    sex: str               # "G" | "M" | "F"
    sire: str
    dam: str
    current_rating: int
    best_rating: int
    past_races: List[PastRace] = field(default_factory=list)

    # Computed stats (call compute_stats() after construction)
    win_rate: float = 0.0
    place_rate: float = 0.0         # Top-3 rate
    distance_preference: Dict[str, float] = field(default_factory=dict)
    going_preference: Dict[str, float] = field(default_factory=dict)
    venue_preference: Dict[str, float] = field(default_factory=dict)
    days_since_last_run: int = 999

    def compute_stats(self) -> None:
        """Compute win/place rates and preferences from past_races."""
        finished = [r for r in self.past_races if r.finish_position < 99]
        if not finished:
            return

        n = len(finished)
        wins = sum(1 for r in finished if r.finish_position == 1)
        places = sum(1 for r in finished if r.finish_position <= 3)
        self.win_rate = round(wins / n, 4)
        self.place_rate = round(places / n, 4)

        # Distance preference: group into 200m buckets
        dist_groups: Dict[str, List[bool]] = {}
        for r in finished:
            bucket = str(round(r.distance / 200) * 200)
            dist_groups.setdefault(bucket, []).append(r.finish_position <= 3)
        self.distance_preference = {k: round(sum(v) / len(v), 4) for k, v in dist_groups.items()}

        # Going preference
        going_groups: Dict[str, List[bool]] = {}
        for r in finished:
            going_groups.setdefault(r.going, []).append(r.finish_position <= 3)
        self.going_preference = {k: round(sum(v) / len(v), 4) for k, v in going_groups.items()}

        # Venue preference
        venue_groups: Dict[str, List[bool]] = {}
        for r in finished:
            venue_groups.setdefault(r.venue, []).append(r.finish_position <= 3)
        self.venue_preference = {k: round(sum(v) / len(v), 4) for k, v in venue_groups.items()}

        # Days since last run
        if finished:
            last_race_dates = sorted([r.race_date for r in finished], reverse=True)
            try:
                last_dt = datetime.strptime(last_race_dates[0], "%Y-%m-%d").date()
                self.days_since_last_run = (date.today() - last_dt).days
            except Exception:
                self.days_since_last_run = 999


# ── Fetch helpers ─────────────────────────────────────────────────────────────

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

def _parse_going(going_raw: str) -> Tuple[str, str]:
    """Convert HKJC going code to full name. Returns (full_name, code)."""
    code = going_raw.strip().upper()
    return GOING_MAP.get(code, going_raw), code


def _parse_date_hkjc(date_str: str) -> str:
    """Convert HKJC date format DD/MM/YY to YYYY-MM-DD."""
    try:
        dt = datetime.strptime(date_str.strip(), "%d/%m/%y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        try:
            dt = datetime.strptime(date_str.strip(), "%d/%m/%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return date_str


def _parse_rc_track(rc_track: str) -> Tuple[str, str, str]:
    """
    Parse 'RC/Track/Course' cell like 'ST / Turf / "A+3"' or 'HV / Turf / "A"'.
    Returns (venue_code, track_type, config).
    """
    parts = [p.strip().strip('"') for p in rc_track.split("/")]
    venue = parts[0] if parts else "ST"
    track = parts[1] if len(parts) > 1 else "Turf"
    config = parts[2].strip('"') if len(parts) > 2 else "A"
    return venue, track, config


def _parse_finish_position(pos_str: str) -> int:
    """Parse finish position string. Returns 99 for WV/DNF/PU/DSQ etc."""
    pos_str = pos_str.strip()
    if pos_str.isdigit():
        return int(pos_str)
    # Special codes
    special = {"WV", "WX", "DNF", "PU", "RTD", "DSQ", "DISQ", "UR", "FE", "BD"}
    if pos_str.upper() in special or not pos_str:
        return 99
    # Sometimes has suffix like "1DH" (dead heat)
    match = re.match(r"(\d+)", pos_str)
    return int(match.group(1)) if match else 99


def _parse_past_race_row(cells: List) -> Optional[PastRace]:
    """
    Parse one row of past performance table.

    Column indices (confirmed 2026-04-02):
    0: race index link
    1: Pla. (finish)
    2: Date (DD/MM/YY)
    3: RC/Track/Course
    4: Dist.
    5: G (going code)
    6: Race Class
    7: Dr. (draw)
    8: Rtg.
    9: Trainer
    10: Jockey
    11: LBW
    12: Win Odds
    13: Act. Wt.
    14: Running Position
    15: Finish Time
    16: Declar. Horse Wt.
    17: Gear
    """
    if len(cells) < 12:
        return None
    texts = [c.get_text(strip=True) for c in cells]

    try:
        finish_pos = _parse_finish_position(texts[1])
        race_date = _parse_date_hkjc(texts[2])
        venue, track_type, config = _parse_rc_track(texts[3])
        distance = int(re.sub(r"\D", "", texts[4])) if texts[4] else 1200
        going_full, going_code = _parse_going(texts[5])
        race_class = texts[6]
        draw = int(texts[7]) if texts[7].isdigit() else 0
        rating = int(texts[8]) if texts[8].lstrip("-").isdigit() else 0
        trainer = texts[9]
        jockey = texts[10]

        # LBW (lengths behind winner) — can be "0" if won, or e.g. "1.5"
        lbw_str = texts[11].replace("SH", "0.1").replace("HD", "0.2").replace("NK", "0.3")
        try:
            winning_margin = float(lbw_str)
        except ValueError:
            winning_margin = 0.0 if finish_pos == 1 else 5.0

        # Win odds
        try:
            sp = float(texts[12].replace(",", "")) if texts[12] else 0.0
        except ValueError:
            sp = 0.0

        # Weight
        weight = int(texts[13]) if len(texts) > 13 and texts[13].isdigit() else 126

        # Running position
        running_pos = texts[14] if len(texts) > 14 else ""

        # Finish time
        finish_time = texts[15] if len(texts) > 15 else "0:00.0"

        # Race number from link in cell[0]
        race_num = 0
        link = cells[0].find("a")
        if link:
            href = link.get("href", "")
            rno_match = re.search(r"RaceNo=(\d+)", href, re.IGNORECASE)
            race_num = int(rno_match.group(1)) if rno_match else 0

        return PastRace(
            race_date=race_date,
            race_number=race_num,
            venue=venue,
            distance=distance,
            track_type=track_type,
            track_config=config,
            going=going_full,
            going_code=going_code,
            race_class=race_class,
            draw=draw,
            finish_position=finish_pos,
            runners=14,  # not always available from this table
            jockey=jockey,
            trainer=trainer,
            weight=weight,
            rating_at_race=rating,
            starting_price=sp,
            winning_margin=winning_margin,
            finish_time=finish_time,
            running_position=running_pos,
        )
    except (IndexError, ValueError) as e:
        logger.debug(f"Row parse error: {e}")
        return None


def _parse_horse_info(soup: BeautifulSoup, horse_code: str) -> Dict:
    """
    Extract basic horse info (name, age, sire, dam, rating etc.)
    
    HKJC page structure (confirmed 2026-04-02):
    - Horse name in first table cell: 'SUPER SIXTY (J071)'
    - Info table has rows: Country of Origin/Age | Colour/Sex | Trainer | Current Rating...
    """
    info = {
        "horse_name": horse_code,
        "horse_name_ch": "",
        "age": 4,
        "country": "HK",
        "colour": "B",
        "sex": "G",
        "sire": "",
        "dam": "",
        "current_rating": 80,
        "best_rating": 85,
    }

    try:
        tables = soup.find_all("table")
        text = soup.get_text(" ")

        # Horse name from first large text (e.g. 'SUPER SIXTY (J071)')
        for table in tables[:3]:
            cell_text = table.get_text(strip=True)
            # Pattern: ALL CAPS NAME (CODE)
            name_match = re.match(r"([A-Z][A-Z ']+?)\s*\(", cell_text)
            if name_match:
                info["horse_name"] = name_match.group(1).strip()
                break

        # Age from 'Country of Origin / Age : AUS / 3'
        age_match = re.search(r"Country of Origin\s*/\s*Age\s*[:|]\s*([A-Z]+)\s*/\s*(\d+)", text)
        if age_match:
            info["country"] = age_match.group(1)
            info["age"] = int(age_match.group(2))

        # Colour / Sex : Bay / Gelding
        sex_match = re.search(r"Colour\s*/\s*Sex\s*[:|]\s*([A-Za-z ]+?)\s*/\s*(Gelding|Mare|Colt|Filly|Horse|Rig)", text)
        if sex_match:
            colour_map = {"Bay": "B", "Brown": "Br", "Grey": "Gr", "Chestnut": "Ch", "Black": "Blk"}
            colour_raw = sex_match.group(1).strip()
            info["colour"] = colour_map.get(colour_raw, colour_raw[:2])
            sex_map = {"Gelding": "G", "Mare": "F", "Colt": "C", "Filly": "F", "Horse": "H"}
            info["sex"] = sex_map.get(sex_match.group(2), "G")

        # Current rating from text
        rating_match = re.search(r"Current Rating\s*[:|]\s*(\d+)", text, re.IGNORECASE)
        if rating_match:
            info["current_rating"] = int(rating_match.group(1))
            info["best_rating"] = info["current_rating"] + 5
        
        # Best rating from 'No. of 1-2-3-Starts' context or rating history
        best_match = re.search(r"Best Rating:\s*(\d+)", text, re.IGNORECASE)
        if best_match:
            info["best_rating"] = int(best_match.group(1))

    except Exception as e:
        logger.debug(f"Horse info parse error: {e}")

    return info


# ── Main public function ──────────────────────────────────────────────────────

def fetch_horse_profile(horse_code: str, last_n: int = 20) -> HorseProfile:
    """
    Fetch a horse's past performance profile from HKJC.

    Args:
        horse_code: HKJC horse code e.g. "HK_2025_L126"
        last_n: Number of past races to retrieve (up to ~30 shown)

    Returns:
        HorseProfile with computed statistics
    """
    from config import config

    if config.DEMO_MODE:
        return mock_horse_profile(horse_code)

    params = {"horseid": horse_code, "Option": "1"}  # Option=1 = Show All seasons
    html = _fetch_html(HORSE_PROFILE_URL, params=params)

    if not html:
        if config.REAL_DATA_ONLY and not config.DEMO_MODE:
            logger.error(f"Could not fetch profile for {horse_code}; REAL_DATA_ONLY=True so mock fallback is disabled")
            raise RuntimeError(f"Missing real horse profile for {horse_code}")
        logger.warning(f"Could not fetch profile for {horse_code}; using mock")
        return mock_horse_profile(horse_code)

    try:
        soup = BeautifulSoup(html, "lxml")
        horse_info = _parse_horse_info(soup, horse_code)

        # Past performance table — confirmed structure (2026-04-02):
        # Find table with header row: RaceIndex | Pla. | Date | RC/Track/Course | ...
        past_races: List[PastRace] = []
        tables = soup.find_all("table")
        perf_table = None
        for t in tables:
            rows = t.find_all("tr")
            if not rows:
                continue
            header_texts = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            # Check for the known column headers
            if "Pla." in header_texts and "Date" in header_texts and "G" in header_texts:
                perf_table = t
                break

        if perf_table:
            rows = perf_table.find_all("tr")
            for row in rows[1:]:  # skip header
                cells = row.find_all("td")
                if len(cells) < 12:
                    continue
                # Skip season separator rows (e.g. '25/26Season')
                first_text = cells[0].get_text(strip=True) if cells else ""
                if "Season" in first_text or not first_text:
                    continue
                pr = _parse_past_race_row(cells)
                if pr:
                    past_races.append(pr)
                    if len(past_races) >= last_n:
                        break

        profile = HorseProfile(
            horse_code=horse_code,
            horse_name=horse_info["horse_name"],
            horse_name_ch=horse_info["horse_name_ch"],
            age=horse_info["age"],
            country=horse_info["country"],
            colour=horse_info["colour"],
            sex=horse_info["sex"],
            sire=horse_info["sire"],
            dam=horse_info["dam"],
            current_rating=horse_info["current_rating"],
            best_rating=horse_info["best_rating"],
            past_races=past_races,
        )
        profile.compute_stats()
        logger.info(f"Fetched profile for {horse_code}: {len(past_races)} past races")
        return profile

    except Exception as e:
        logger.error(f"Error parsing horse profile for {horse_code}: {e}")
        if config.REAL_DATA_ONLY and not config.DEMO_MODE:
            raise
        return mock_horse_profile(horse_code)


def fetch_multiple_profiles(horse_codes: List[str], last_n: int = 20, delay: float = 0.5) -> Dict[str, HorseProfile]:
    """
    Fetch profiles for multiple horses with polite delays.

    Args:
        horse_codes: List of HKJC horse codes
        last_n: Past races per horse
        delay: Seconds between requests

    Returns:
        Dict mapping horse_code -> HorseProfile
    """
    profiles: Dict[str, HorseProfile] = {}
    for i, code in enumerate(horse_codes):
        logger.debug(f"Fetching profile {i+1}/{len(horse_codes)}: {code}")
        profiles[code] = fetch_horse_profile(code, last_n)
        if i < len(horse_codes) - 1:
            time.sleep(delay)
    return profiles


# ── Mock data ─────────────────────────────────────────────────────────────────

def mock_horse_profile(horse_code: str, last_n: int = 12) -> HorseProfile:
    """
    Generate realistic mock horse profile for offline testing.
    Uses a stable random seed per horse_code for reproducibility.
    """
    rng = random.Random(hash(horse_code) & 0xFFFFFF)

    goings = ["GOOD", "GOOD", "GOOD TO YIELDING", "YIELDING", "SOFT", "FAST"]
    going_codes = ["G", "G", "GY", "Y", "S", "F"]
    venues = ["ST", "HV"]
    jockeys = ["Z Purton", "J Moreira", "V Borges", "M Chadwick", "L Hewitson",
               "B Prebble", "C Y Ho", "K Leung", "N Callan", "H T Mo"]
    trainers = ["F C Lor", "D Whyte", "C Fownes", "J Size", "R Gibson",
                "A T Millard", "P F Yiu", "K W Lui", "L Ho", "C H Yip"]
    distances_st = [1000, 1200, 1400, 1600, 1800, 2000, 2400]
    distances_hv = [1000, 1200, 1650, 2200]
    classes = ["Class 5", "Class 4", "Class 3", "Class 2", "Class 1"]
    configs_st = ["A", "B", "B+2", "C", "C+3", "A+3"]
    configs_hv = ["A", "B", "C"]

    base_rating = 70 + rng.randint(-20, 40)
    best_rating = base_rating + rng.randint(0, 15)
    base_win_prob = rng.uniform(0.05, 0.28)

    past_races: List[PastRace] = []
    today = date.today()

    for i in range(last_n):
        days_ago = sum(rng.randint(14, 35) for _ in range(i + 1))
        race_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        venue = rng.choice(venues)
        distance = rng.choice(distances_st if venue == "ST" else distances_hv)
        going_idx = rng.randint(0, len(goings) - 1)
        going = goings[going_idx]
        going_code = going_codes[going_idx]
        config = rng.choice(configs_st if venue == "ST" else configs_hv)
        draw = rng.randint(1, 14)

        rand_val = rng.random()
        if rand_val < base_win_prob:
            pos, margin = 1, 0.0
        elif rand_val < base_win_prob * 2.5:
            pos, margin = 2, round(rng.uniform(0.1, 2.0), 1)
        elif rand_val < base_win_prob * 4.0:
            pos, margin = 3, round(rng.uniform(0.5, 3.0), 1)
        elif rand_val > 0.93:
            pos, margin = 99, 0.0
        else:
            pos = rng.randint(4, 12)
            margin = round(rng.uniform(1.0, 10.0), 1)

        sp = round(rng.uniform(2.5, 35.0), 1)
        rating = max(40, min(130, base_rating + rng.randint(-5, 5)))
        run_positions = f"{rng.randint(1,14)}-{rng.randint(1,14)}-{rng.randint(1,pos+2)}"

        past_races.append(PastRace(
            race_date=race_date,
            race_number=rng.randint(1, 10),
            venue=venue,
            distance=distance,
            track_type=rng.choice(["Turf", "Turf", "Turf", "AWT"]),
            track_config=config,
            going=going,
            going_code=going_code,
            race_class=rng.choice(classes),
            draw=draw,
            finish_position=pos,
            runners=rng.randint(10, 14),
            jockey=rng.choice(jockeys),
            trainer=rng.choice(trainers),
            weight=rng.randint(113, 133),
            rating_at_race=rating,
            starting_price=sp,
            winning_margin=margin,
            finish_time=f"1:{rng.randint(8, 20):02d}.{rng.randint(0, 9)}",
            running_position=run_positions,
        ))

    # Name lookup for known mock horses
    known_names = {
        "HK_2025_L126": ("ALMIGHTY LIGHTNING", "萬里神駒"),
        "HK_2023_J071": ("SUPER SIXTY", "超級六十"),
        "HK_2024_K207": ("CASA BUDDY", "家居好友"),
        "HK_2022_H155": ("EAGLE WAY", "雄鷹路"),
        "HK_2021_G201": ("IRON FIST", "鐵拳"),
        "HK_2020_F112": ("PERFECT STORM", "完美風暴"),
    }
    en, ch = known_names.get(horse_code, (horse_code, horse_code))

    sires = ["Frankel", "Deep Impact", "Kingman", "Galileo", "Dubawi",
             "Exceed And Excel", "Zoustar", "Shalaa", "Invincible Spirit"]

    profile = HorseProfile(
        horse_code=horse_code,
        horse_name=en,
        horse_name_ch=ch,
        age=rng.randint(3, 8),
        country=rng.choice(["HK", "HK", "HK", "AUS", "IRE", "GB", "NZ"]),
        colour=rng.choice(["B", "Br", "Gr", "Ch", "B"]),
        sex=rng.choice(["G", "G", "G", "M", "F"]),
        sire=rng.choice(sires),
        dam="Dam Name",
        current_rating=base_rating,
        best_rating=best_rating,
        past_races=past_races,
    )
    profile.compute_stats()
    return profile
