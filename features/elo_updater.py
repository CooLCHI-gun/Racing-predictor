"""
HKJC 賽馬預測系統 - ELO 自動更新模組
Post-race ELO auto-update pipeline.

Called automatically by scheduler after each race day.
Can also be run manually:
    python -c "from features.elo_updater import run_post_race_elo_update; run_post_race_elo_update('2026-04-06', 'ST')"
"""
import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ELO_RATINGS_FILE = "data/processed/elo_ratings.json"
ELO_UPDATE_LOG_FILE = "data/processed/elo_update_log.json"


def run_post_race_elo_update(
    race_date: str,
    venue_code: str,
    racecard_horses: Optional[Dict[int, Dict[str, str]]] = None,
) -> Dict:
    """
    Full post-race ELO update pipeline:
    1. Fetch actual results from HKJC
    2. Map horse numbers → horse codes (from racecard data)
    3. Update ELO ratings for all participants
    4. Save updated ratings
    5. Log the update

    Args:
        race_date: "YYYY-MM-DD"
        venue_code: "ST" | "HV"
        racecard_horses: Optional dict {race_num: {horse_num_str: horse_code}}
                         If provided, enables accurate horse_code mapping.
                         If None, uses horse_num as fallback identifier.

    Returns:
        Summary dict with update statistics
    """
    from features.elo import ELOSystem
    from scraper.results_fetcher import fetch_day_results
    from model.backtest import Backtester

    logger.info(f"Starting post-race ELO update for {race_date} {venue_code}")

    # Load existing ELO system
    elo = ELOSystem(k_factor=32, base_rating=1500)
    elo.load_ratings(ELO_RATINGS_FILE)
    prev_ratings = elo.get_all_ratings().copy()

    # Fetch today's results
    results = fetch_day_results(race_date, venue_code)
    if not results:
        logger.warning(f"No results fetched for {race_date} — ELO not updated")
        return {"status": "no_results", "races_updated": 0}

    update_summary = {
        "date": race_date,
        "venue": venue_code,
        "races_updated": 0,
        "horses_updated": 0,
        "timestamp": datetime.now().isoformat(),
        "race_updates": [],
    }

    backtester = Backtester()

    for result in results:
        if not result.finishing_order:
            continue

        race_id = result.race_id
        race_num = result.race_number

        # Build finishing order as horse codes
        # Priority: use racecard_horses mapping if provided
        if racecard_horses and race_num in racecard_horses:
            horse_map = racecard_horses[race_num]
            horse_codes_order = []
            for horse_num in result.finishing_order:
                code = horse_map.get(str(horse_num), f"UNKNOWN_{race_date}_{venue_code}_{race_num}_{horse_num}")
                horse_codes_order.append(code)
        else:
            # Fallback: use synthetic code from horse number
            # Not ideal but preserves relative order for ELO purposes
            horse_codes_order = [
                f"RACEDAY_{race_date}_{venue_code}_R{race_num}_H{h}"
                for h in result.finishing_order
            ]

        # Update ELO
        new_ratings = elo.update_ratings(horse_codes_order)

        # Record in backtest system
        predicted_race_id = f"{race_date}_{venue_code}_{race_num}"
        backtester.record_result(
            race_id=predicted_race_id,
            actual_result=result.finishing_order,
            trio_dividend=result.trio_dividend,
            win_dividend=result.win_dividend,
        )

        # Track changes
        top3_codes = horse_codes_order[:3]
        winner_code = horse_codes_order[0] if horse_codes_order else ""
        winner_delta = 0.0
        if winner_code and winner_code in prev_ratings:
            winner_delta = new_ratings.get(winner_code, prev_ratings[winner_code]) - prev_ratings[winner_code]

        update_summary["race_updates"].append({
            "race_num": race_num,
            "winner_horse_num": result.winner_horse_num,
            "winner_name": result.winner_horse_name,
            "elo_winner_delta": round(winner_delta, 2),
            "horses_updated": len(new_ratings),
        })
        update_summary["races_updated"] += 1
        update_summary["horses_updated"] += len(new_ratings)

        logger.info(
            f"R{race_num}: #{result.winner_horse_num} {result.winner_horse_name} wins. "
            f"ELO updated for {len(new_ratings)} horses."
        )

    # Save updated ratings
    elo.save_ratings(ELO_RATINGS_FILE)

    # Save update log
    _append_update_log(update_summary)

    logger.info(
        f"ELO update complete: {update_summary['races_updated']} races, "
        f"{update_summary['horses_updated']} horse updates saved"
    )
    return update_summary


def run_elo_bootstrap_from_history(filepath: str = "data/processed/historical_results.json") -> int:
    """
    Bootstrap ELO from accumulated historical results.
    Called once when sufficient history is available.

    Returns number of horses rated.
    """
    from features.elo import ELOSystem
    from scraper.results_fetcher import load_results_history

    logger.info("Bootstrapping ELO from historical results...")
    results = load_results_history(filepath)

    if not results:
        logger.warning("No historical results found for ELO bootstrap")
        return 0

    elo = ELOSystem(k_factor=32, base_rating=1500)

    # Sort by date for chronological processing
    sorted_results = sorted(results, key=lambda r: (r.race_date, r.race_number))

    for result in sorted_results:
        if not result.finishing_order:
            continue
        # Use synthetic horse codes (horse_num based) for bootstrap
        codes = [
            f"RACEDAY_{result.race_date}_{result.venue_code}_R{result.race_number}_H{h}"
            for h in result.finishing_order
        ]
        elo.update_ratings(codes)

    elo.save_ratings(ELO_RATINGS_FILE)
    n = len(elo.get_all_ratings())
    logger.info(f"ELO bootstrap complete: {n} horses rated from {len(results)} races")
    return n


def get_elo_update_history(last_n: int = 10) -> List[Dict]:
    """Return last N ELO update log entries."""
    if not os.path.exists(ELO_UPDATE_LOG_FILE):
        return []
    try:
        with open(ELO_UPDATE_LOG_FILE, "r") as f:
            log = json.load(f)
        return log[-last_n:] if isinstance(log, list) else []
    except Exception:
        return []


def _append_update_log(summary: Dict) -> None:
    """Append a summary entry to the ELO update log."""
    os.makedirs(os.path.dirname(ELO_UPDATE_LOG_FILE), exist_ok=True)
    log = []
    if os.path.exists(ELO_UPDATE_LOG_FILE):
        try:
            with open(ELO_UPDATE_LOG_FILE, "r") as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append(summary)
    # Keep last 100 entries
    log = log[-100:]

    with open(ELO_UPDATE_LOG_FILE, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
