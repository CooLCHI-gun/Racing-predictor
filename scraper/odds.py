"""
HKJC 賽馬預測系統 - 賠率抓取模組
Fetches real-time win/place odds from HKJC GraphQL API.

CONFIRMED (2026-04-02 via Playwright network intercept):
  Endpoint  : https://info.cld.hkjc.com/graphql/base/
  Operation : "racing"
  Variables : {date, venueCode, raceNo, oddsTypes: ["WIN","PLA"]}
  Odds field: oddsNodes[].oddsValue   ← NOT "odds"!

Pool status lifecycle:
  INITIAL    → Not yet open (days before race)
  STOP_SELL  → Temporarily closed
  SELLING    → Open for betting (odds visible)
  RESULTED   → Race over, dividends declared

Odds are ONLY available on race day when sellStatus = "SELLING".
Before race day, oddsNodes is always empty [].
"""
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

HK_TZ = ZoneInfo("Asia/Hong_Kong")


def _convert_to_hk_timezone(ts: Optional[str]) -> Optional[datetime]:
    """Convert timestamps to aware Asia/Hong_Kong datetime."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # Try fallback ISO formats
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(ts, fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Invalid timestamp format: {ts}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(HK_TZ)

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://info.cld.hkjc.com/graphql/base/"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
    "Accept": "application/json",
}

# ── Confirmed working GraphQL query (from real network intercept) ─────────────
ODDS_QUERY = """
query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id
      status
      sellStatus
      oddsType
      lastUpdateTime
      guarantee
      minTicketCost
      name_en
      name_ch
      leg {
        number
        races
      }
      cWinSelections {
        composite
        name_ch
        name_en
        starters
      }
      oddsNodes {
        combString
        oddsValue
        hotFavourite
        oddsDropValue
        bankerOdds {
          combString
          oddsValue
        }
      }
    }
  }
}
"""

# Query for full runners + basic odds (available pre-race)
RUNNERS_QUERY = """
query racing($date: String, $venueCode: String) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    status
    currentNumberOfRace
    totalInvestment
    races {
      no
      status
      go_en
      distance
      raceClass_en
      postTime
      raceName_en
      runners {
        id
        no
        standbyNo
        status
        name_ch
        name_en
        horse { id code }
        color
        barrierDrawNumber
        handicapWeight
        currentWeight
        currentRating
        internationalRating
        gearInfo
        allowance
        trainerPreference
        last6run
        jockey { code name_en }
        trainer { code name_en }
      }
    }
  }
}
"""

# Pool sizes query
POOL_QUERY = """
query racing($date: String, $venueCode: String, $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    totalInvestment
    poolInvs: pmPools(oddsTypes: [WIN, PLA, TRI, TT, QIN, QPL, FCT], raceNo: $raceNo) {
      oddsType
      status
      sellStatus
      investment
      guarantee
    }
  }
}
"""


# ── Maths helpers ─────────────────────────────────────────────────────────────

def calculate_implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability. e.g. 4.0 → 0.25"""
    if odds <= 1.0:
        return 1.0
    return round(1.0 / odds, 6)


def calculate_expected_value(predicted_prob: float, odds: float, stake: float = 10.0) -> float:
    """EV = (p_win × profit_if_win) - (p_lose × stake). Positive = value bet."""
    if odds <= 0 or predicted_prob < 0:
        return -stake
    profit = (odds - 1.0) * stake
    return round((predicted_prob * profit) - ((1.0 - predicted_prob) * stake), 4)


def calculate_overlay(model_prob: float, implied_prob: float) -> float:
    """Edge = (model_prob - implied_prob) / implied_prob. 0.10 = 10% edge."""
    if implied_prob <= 0:
        return 0.0
    return round((model_prob - implied_prob) / implied_prob, 4)


# ── GraphQL helpers ───────────────────────────────────────────────────────────

def _graphql(query: str, variables: Dict, operation_name: str = "racing",
             max_retries: int = 3) -> Optional[Dict]:
    """Execute a GraphQL query against the HKJC API."""
    payload = {
        "operationName": operation_name,
        "query": query,
        "variables": variables,
    }
    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(GRAPHQL_URL, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                # WHITELIST_ERROR is expected for custom queries — use Playwright instead
                err_codes = [e.get("extensions", {}).get("code", "") for e in data["errors"]]
                if "WHITELIST_ERROR" in err_codes:
                    logger.debug("GraphQL whitelist restriction — query must match page's exact format")
                    return None
                logger.warning(f"GraphQL errors: {data['errors']}")
                return None
            return data.get("data")
        except Exception as e:
            logger.warning(f"GraphQL attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)
    return None


def _parse_odds_from_nodes(odds_nodes: List[Dict]) -> Dict[int, float]:
    """Parse oddsNodes [{combString, oddsValue}] into {horse_number: odds}."""
    result = {}
    for node in odds_nodes:
        comb = str(node.get("combString", "")).strip()
        odds_str = str(node.get("oddsValue", "0")).strip()
        try:
            horse_num = int(comb)
            odds_val = float(odds_str)
            if odds_val > 0:
                result[horse_num] = odds_val
        except (ValueError, TypeError):
            continue
    return result


# ── Playwright-based fetch (for JS-rendered odds on race day) ─────────────────

async def _fetch_odds_playwright_async(race_date: str, venue_code: str,
                                       race_number: int) -> Optional[Dict[int, Dict]]:
    """
    Fetch real odds via Playwright by:
    1. Loading the bet.hkjc.com page (to establish session + cookies)
    2. Intercepting the GraphQL response for WIN/PLA pools
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed: pip install playwright && playwright install chromium")
        return None

    target_url = f"https://bet.hkjc.com/en/racing/wp/{race_date}/{venue_code.lower()}/{race_number}"

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            win_nodes: Dict[int, float] = {}
            pla_nodes: Dict[int, float] = {}
            odds_timestamp = None
            captured = {"done": False}

            async def on_response(response):
                if "graphql" in response.url and response.status == 200 and not captured["done"]:
                    try:
                        req = response.request
                        req_body = req.post_data or ""
                        if req_body:
                            req_data = json.loads(req_body)
                            variables = req_data.get("variables", {})
                            # Only process the odds query (has raceNo + oddsTypes)
                            if "oddsTypes" in variables and variables.get("raceNo") == race_number:
                                body = await response.text()
                                data = json.loads(body)
                                meetings = (data.get("data") or {}).get("raceMeetings") or []
                                if meetings:
                                    for pool in meetings[0].get("pmPools") or []:
                                        nodes = pool.get("oddsNodes") or []
                                        if pool.get("oddsType") == "WIN" and nodes:
                                            nonlocal win_nodes
                                            nonlocal odds_timestamp
                                            win_nodes = _parse_odds_from_nodes(nodes)
                                            # lastUpdateTime likely in ISO 8601 format.
                                            odds_timestamp = _convert_to_hk_timezone(pool.get("lastUpdateTime"))
                                        elif pool.get("oddsType") == "PLA" and nodes:
                                            nonlocal pla_nodes
                                            pla_nodes = _parse_odds_from_nodes(nodes)
                                    if win_nodes:
                                        captured["done"] = True
                    except Exception as e:
                        logger.debug(f"Response intercept error: {e}")

            page.on("response", on_response)
            await page.goto(target_url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(4000)  # Wait for lazy-loaded odds
            await browser.close()

        if not win_nodes:
            logger.info(f"No live odds for R{race_number} on {race_date} (betting not open yet)")
            return None

        result: Dict[int, Dict] = {}
        for hnum in set(win_nodes) | set(pla_nodes):
            w = win_nodes.get(hnum, 0.0)
            p = pla_nodes.get(hnum, 0.0)
            if w > 0:
                if p <= 0:
                    div = 3.0 if w < 4 else (3.5 if w < 10 else 4.0)
                    p = max(1.05, round(w / div, 1))

                result[hnum] = {
                    "win_odds": w,
                    "place_odds": p,
                    "implied_win_prob": calculate_implied_probability(w),
                    "implied_place_prob": calculate_implied_probability(p),
                    "odds_ts": odds_timestamp.isoformat() if odds_timestamp else None,
                }

        logger.info(f"Playwright odds: {len(result)} runners for R{race_number} ({race_date})")
        return result if result else None

    except Exception as e:
        logger.warning(f"Playwright odds fetch failed: {e}")
        return None


def fetch_odds_playwright_sync(race_date: str, venue_code: str,
                               race_number: int) -> Optional[Dict[int, Dict]]:
    """Synchronous wrapper for Playwright odds fetch."""
    import asyncio
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    _fetch_odds_playwright_async(race_date, venue_code, race_number)
                )
                return future.result(timeout=45)
        except RuntimeError:
            return asyncio.run(_fetch_odds_playwright_async(race_date, venue_code, race_number))
    except Exception as e:
        logger.warning(f"Playwright sync wrapper failed: {e}")
        return None


# ── Main public functions ─────────────────────────────────────────────────────

def fetch_odds(
    race_date: str,
    venue_code: str,
    race_number: int,
    horse_numbers: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Fetch real-time win and place odds for a race.

    Strategy:
      1. Playwright (captures real browser GraphQL response)
      2. Mock fallback

    Note: Odds are only available on race day when betting pools open (~2h before race).
          Before race day, this returns mock odds for prediction purposes.

    Args:
        race_date: "YYYY-MM-DD"
        venue_code: "ST" or "HV"
        race_number: 1-12
        horse_numbers: Optional runner count hint for mock generation

    Returns:
        Dict: {horse_number: {win_odds, place_odds, implied_win_prob, implied_place_prob}}
    """
    from config import config

    if config.DEMO_MODE:
        n = len(horse_numbers) if horse_numbers else 12
        return mock_odds(race_number, n)

    logger.info(f"Fetching odds: {race_date} {venue_code} R{race_number}")

    # Strategy 1: Playwright (most reliable for live odds)
    result = fetch_odds_playwright_sync(race_date, venue_code, race_number)
    if result:
        for hnum, val in result.items():
            if not val.get("odds_ts"):
                val["odds_ts"] = datetime.now(tz=HK_TZ).isoformat()
            else:
                try:
                    _convert_to_hk_timezone(val["odds_ts"])
                except Exception as ex:
                    raise ValueError(f"Invalid odds_ts for horse {hnum}: {val['odds_ts']}") from ex
        return result

    # Strategy 2: Mock (pre-race or fallback)
    if config.REAL_DATA_ONLY and not config.DEMO_MODE:
        logger.error(
            "Live odds unavailable and REAL_DATA_ONLY=True; returning empty odds instead of mock data"
        )
        return {}

    logger.info(f"Using estimated odds for R{race_number} (live odds not yet available)")
    n = len(horse_numbers) if horse_numbers else 12
    return mock_odds(race_number, n)


def fetch_trio_pool(race_date: str, venue_code: str, race_number: int) -> Dict:
    """Fetch pool sizes (WIN/PLA/TRI/TT) for a race. Returns investment amounts."""
    from config import config
    if config.DEMO_MODE:
        return _mock_trio_pool(race_number)

    data = _graphql(POOL_QUERY, {
        "date": race_date,
        "venueCode": venue_code,
        "raceNo": race_number,
    })

    result = {"pool_size": 0, "jackpot_amount": 0, "carryover": 0, "tt_investment": 0}
    if data:
        meetings = data.get("raceMeetings") or []
        if meetings:
            for pool in meetings[0].get("poolInvs") or []:
                ptype = pool.get("oddsType", "")
                try:
                    inv = int(float(str(pool.get("investment") or "0").replace(",", "")))
                    guar = int(float(str(pool.get("guarantee") or "0").replace(",", "")))
                except (ValueError, TypeError):
                    inv, guar = 0, 0
                if ptype == "TT":
                    result["tt_investment"] = inv
                    result["jackpot_amount"] = guar
                elif ptype == "TRI":
                    result["pool_size"] = inv

    return result


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_trio_pool(race_number: int) -> Dict:
    rng = random.Random(race_number * 997)
    return {
        "pool_size": rng.randint(2_000_000, 15_000_000),
        "jackpot_amount": rng.randint(0, 5_000_000),
        "carryover": rng.randint(0, 3_000_000),
        "tt_investment": rng.randint(1_000_000, 8_000_000),
    }


def mock_odds(race_number: int, n_horses: int = 12) -> Dict[int, Dict[str, float]]:
    """
    Generate realistic mock odds (HKJC-style overround ~117%).
    Used pre-race or as fallback when live odds unavailable.
    """
    rng = random.Random(race_number * 137 + n_horses)
    raw_probs = []
    for i in range(n_horses):
        if i == 0:
            raw_probs.append(rng.uniform(0.20, 0.35))
        elif i < 3:
            raw_probs.append(rng.uniform(0.10, 0.20))
        else:
            raw_probs.append(rng.uniform(0.03, 0.10))

    total = sum(raw_probs)
    overround = rng.uniform(1.15, 1.20)
    horse_order = list(range(1, n_horses + 1))
    rng.shuffle(horse_order)

    result: Dict[int, Dict[str, float]] = {}
    for horse_num, prob in zip(horse_order, raw_probs):
        ip = prob / total / overround
        w = round(1.0 / max(ip, 0.01), 1)
        if w > 50:
            w = round(w / 5) * 5
        elif w > 20:
            w = round(w / 2) * 2
        elif w > 10:
            w = round(w * 2) / 2
        w = max(1.2, w)
        div = 3.0 if w < 4 else (3.5 if w < 10 else 4.0)
        p = max(1.05, round(w / div, 1))
        result[horse_num] = {
            "win_odds": w,
            "place_odds": p,
            "implied_win_prob": calculate_implied_probability(w),
            "implied_place_prob": calculate_implied_probability(p),
            "odds_ts": datetime.now(tz=HK_TZ).isoformat(),
        }
    return result


def generate_mock_odds_update(
    current_odds: Dict[int, Dict[str, float]],
    drift_pct: float = 0.04,
) -> Dict[int, Dict[str, float]]:
    """Simulate small odds drift for live UI testing."""
    updated = {}
    for horse_num, odds in current_odds.items():
        drift = random.uniform(-drift_pct, drift_pct)
        new_win = max(1.2, round(odds["win_odds"] * (1 + drift), 1))
        div = 3.0 if new_win < 4 else (3.5 if new_win < 10 else 4.0)
        new_place = max(1.05, round(new_win / div, 1))
        updated[horse_num] = {
            "win_odds": new_win,
            "place_odds": new_place,
            "implied_win_prob": calculate_implied_probability(new_win),
            "implied_place_prob": calculate_implied_probability(new_place),
            "odds_ts": datetime.now(tz=HK_TZ).isoformat(),
        }
    return updated
