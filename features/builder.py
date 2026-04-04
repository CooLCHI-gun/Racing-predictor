"""
HKJC 賽馬預測系統 - 特徵建構管線
Main feature engineering pipeline.

Combines ELO ratings, jockey/trainer stats, draw stats, horse form,
and market odds into a single feature DataFrame for model training/prediction.
"""
import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HK_TZ = ZoneInfo("Asia/Hong_Kong")


def _parse_race_datetime(race_date: str, start_time: str) -> datetime:
    """Convert race date+start time to timezone-aware datetime (Asia/Hong_Kong)."""
    if not race_date or not start_time:
        raise ValueError("race_date and start_time required to build race_datetime")
    try:
        target = datetime.strptime(f"{race_date} {start_time}", "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError(f"Invalid race_date/start_time format: {race_date} {start_time}") from exc

    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return target.astimezone(HK_TZ)

from features.elo import ELOSystem
from features.jockey_trainer import JockeyTrainerStats
from features.draw import DrawStats
from scraper.racecard import Race, HorseEntry
from scraper.horse_profile import HorseProfile
from scraper.odds import calculate_implied_probability

logger = logging.getLogger(__name__)

# Form weighting: 1st place = 6, 2nd = 5, ..., 6th = 1, 7th+ = 0
FORM_WEIGHTS = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}


def _days_since(date_str: str) -> int:
    """Calculate days elapsed since a date string (YYYY-MM-DD)."""
    try:
        past = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - past).days
    except (ValueError, TypeError):
        return 999


def _form_score(past_races: list, n: int = 6) -> float:
    """
    Calculate weighted recent form score.

    Args:
        past_races: List of PastRace objects (most recent first)
        n: Number of races to consider

    Returns:
        Weighted form score normalised to [0, 1]
    """
    if not past_races:
        return 0.3  # neutral for debutants
    recent = [r for r in past_races if r.finish_position < 99][:n]
    if not recent:
        return 0.2
    max_score = sum(FORM_WEIGHTS.get(1, 6) for _ in recent)
    raw_score = sum(FORM_WEIGHTS.get(r.finish_position, 0) for r in recent)
    return raw_score / max(max_score, 1)


def _distance_aptitude(
    past_races: list, target_distance: int, tolerance: int = 200
) -> float:
    """
    Calculate performance at similar distances.

    Args:
        past_races: List of PastRace objects
        target_distance: Today's race distance in metres
        tolerance: ±metres to include

    Returns:
        Win/place rate at similar distances [0, 1]
    """
    similar = [
        r for r in past_races
        if r.finish_position < 99 and abs(r.distance - target_distance) <= tolerance
    ]
    if not similar:
        return 0.3
    place_count = sum(1 for r in similar if r.finish_position <= 3)
    return place_count / len(similar)


def _going_aptitude(past_races: list, target_going: str) -> float:
    """
    Calculate performance on similar going.

    Groups going into firm/good/soft/heavy buckets for comparison.
    """
    GOING_GROUPS = {
        "FAST": "firm", "GOOD": "good", "GOOD TO YIELDING": "good",
        "YIELDING": "soft", "SOFT": "soft", "HEAVY": "heavy",
    }
    target_group = GOING_GROUPS.get(target_going.upper(), "good")

    relevant = [
        r for r in past_races
        if r.finish_position < 99 and GOING_GROUPS.get(r.going.upper(), "good") == target_group
    ]
    if not relevant:
        return 0.3
    place_count = sum(1 for r in relevant if r.finish_position <= 3)
    return place_count / len(relevant)


def _rating_trend(past_races: list) -> float:
    """
    Calculate rating trend (last 3 minus previous 3).

    Returns:
        Positive = improving, negative = declining
    """
    if len(past_races) < 4:
        return 0.0
    recent = [r.rating_at_race for r in past_races[:3] if r.finish_position < 99]
    older = [r.rating_at_race for r in past_races[3:6] if r.finish_position < 99]
    if not recent or not older:
        return 0.0
    return np.mean(recent) - np.mean(older)


def _weight_vs_avg(weight: int, all_weights: List[int]) -> float:
    """Normalised weight carried vs field average."""
    if not all_weights:
        return 0.0
    avg = np.mean(all_weights)
    return (weight - avg) / max(avg, 1)


def build_features(
    race: Race,
    horse_profiles: Dict[str, HorseProfile],
    odds: Dict[int, Dict],
    elo: ELOSystem,
    jt_stats: JockeyTrainerStats,
    draw_stats: DrawStats,
) -> pd.DataFrame:
    """
    Build the full feature DataFrame for a race.

    Args:
        race: Race object with horse entries
        horse_profiles: Dict of horse_code -> HorseProfile
        odds: Dict of horse_number -> {win_odds, place_odds, ...}
        elo: ELO system instance
        jt_stats: Jockey/trainer stats instance
        draw_stats: Draw stats instance

    Returns:
        DataFrame where each row is a horse with feature columns.
        Also contains metadata columns: horse_number, horse_name, horse_code.
    """
    rows = []
    n_runners = len(race.horses)
    all_weights = [h.weight for h in race.horses]
    all_codes = [h.horse_code for h in race.horses]
    field_avg_elo = elo.get_field_average_rating(all_codes)

    for horse in race.horses:
        profile = horse_profiles.get(horse.horse_code)
        past = profile.past_races if profile else []
        h_odds = odds.get(horse.horse_number, {})

        win_odds = h_odds.get("win_odds", 20.0)
        place_odds = h_odds.get("place_odds", 6.0)
        imp_win_prob = calculate_implied_probability(win_odds)
        imp_place_prob = calculate_implied_probability(place_odds)

        elo_rating = elo.get_rating(horse.horse_code)
        elo_vs_field = elo_rating - field_avg_elo
        elo_delta = elo.get_rating_delta(horse.horse_code, last_n=3)

        jk_win_30d = jt_stats.get_jockey_win_rate(horse.jockey_code, "last_30d")
        jk_place_30d = jt_stats.get_jockey_place_rate(horse.jockey_code, "last_30d")
        jk_win_overall = jt_stats.get_jockey_win_rate(horse.jockey_code, "overall")
        tr_win_30d = jt_stats.get_trainer_win_rate(horse.trainer_code, "last_30d")
        tr_place_30d = jt_stats.get_trainer_place_rate(horse.trainer_code, "last_30d")
        tr_win_overall = jt_stats.get_trainer_win_rate(horse.trainer_code, "overall")
        combo_win = jt_stats.get_combo_win_rate(horse.jockey_code, horse.trainer_code)
        combo_place = jt_stats.get_combo_place_rate(horse.jockey_code, horse.trainer_code)
        jk_venue_wr = jt_stats.get_jockey_venue_win_rate(horse.jockey_code, race.venue_code)
        jk_dist_wr = jt_stats.get_jockey_distance_win_rate(horse.jockey_code, race.distance)

        draw_adv = draw_stats.get_draw_advantage_score(
            race.venue_code, race.distance, horse.draw, n_runners
        )

        form = _form_score(past)
        dist_apt = _distance_aptitude(past, race.distance)
        going_apt = _going_aptitude(past, race.going)
        rating_trend = _rating_trend(past)
        weight_vs_avg = _weight_vs_avg(horse.weight, all_weights)

        days_since_last = _days_since(past[0].race_date) if past else 999

        # Win rate & place rate from profile
        win_rate = profile.win_rate if profile else 0.0
        place_rate = profile.place_rate if profile else 0.0

        rows.append({
            # ── Metadata (not model features) ─────────────────────────────
            "horse_number": horse.horse_number,
            "horse_name": horse.horse_name,
            "horse_name_ch": horse.horse_name_ch,
            "horse_code": horse.horse_code,
            "jockey": horse.jockey,
            "trainer": horse.trainer,
            "draw": horse.draw,
            "weight": horse.weight,

            # ── ELO features ───────────────────────────────────────────────
            "elo_rating": round(elo_rating, 2),
            "elo_vs_field": round(elo_vs_field, 2),
            "elo_delta_last3": round(elo_delta, 2),

            # ── Jockey features ────────────────────────────────────────────
            "jockey_win_rate_30d": round(jk_win_30d, 4),
            "jockey_place_rate_30d": round(jk_place_30d, 4),
            "jockey_win_rate_overall": round(jk_win_overall, 4),
            "jockey_venue_win_rate": round(jk_venue_wr, 4),
            "jockey_distance_win_rate": round(jk_dist_wr, 4),

            # ── Trainer features ───────────────────────────────────────────
            "trainer_win_rate_30d": round(tr_win_30d, 4),
            "trainer_place_rate_30d": round(tr_place_30d, 4),
            "trainer_win_rate_overall": round(tr_win_overall, 4),

            # ── Combination features ───────────────────────────────────────
            "combo_win_rate": round(combo_win, 4),
            "combo_place_rate": round(combo_place, 4),

            # ── Draw features ──────────────────────────────────────────────
            "draw_advantage_score": round(draw_adv, 4),
            "draw_position": horse.draw,
            "draw_normalised": horse.draw / max(n_runners, 1),

            # ── Form features ──────────────────────────────────────────────
            "recent_form_score": round(form, 4),
            "win_rate": round(win_rate, 4),
            "place_rate": round(place_rate, 4),

            # ── Distance/going features ────────────────────────────────────
            "distance_aptitude": round(dist_apt, 4),
            "going_aptitude": round(going_apt, 4),

            # ── Rating features ────────────────────────────────────────────
            "handicap_rating": horse.handicap_rating,
            "rating_trend": round(float(rating_trend), 4),
            "rating_vs_avg": round(horse.handicap_rating - np.mean([h.handicap_rating for h in race.horses]), 4),

            # ── Physical/fitness features ─────────────────────────────────
            "weight_carried": horse.weight,
            "weight_vs_avg": round(weight_vs_avg, 4),
            "days_since_last_run": days_since_last,

            # ── Market features ────────────────────────────────────────────
            "win_odds": win_odds,
            "place_odds": place_odds,
            "implied_win_probability": round(imp_win_prob, 4),
            "implied_place_probability": round(imp_place_prob, 4),

            # ── Race context ───────────────────────────────────────────────
            "race_date": race.race_date,
            "race_start_time": race.start_time,
            "race_datetime": _parse_race_datetime(race.race_date, race.start_time),
            "race_distance": race.distance,
            "n_runners": n_runners,
            "venue_code": race.venue_code,  # will be encoded
        })

    df = pd.DataFrame(rows)

    # Encode venue as binary (ST=1, HV=0)
    df["venue_is_st"] = (df["venue_code"] == "ST").astype(int)
    df = df.drop(columns=["venue_code"])

    # Data validity checks
    if not df["horse_number"].is_unique:
        raise ValueError(f"Duplicate horse_number in race {race.race_number}")
    if not df["implied_win_probability"].between(0, 1).all():
        raise ValueError("implied_win_probability out of bounds [0,1]")
    null_pct = df.isna().mean()
    if (null_pct > 0).any():
        logger.warning("Features contain missing values: %s", null_pct[null_pct > 0].to_dict())

    logger.debug(
        f"Built features for race {race.race_number}: {len(df)} horses, {len(df.columns)} features, "
        f"null_pct={null_pct.to_dict()}"
    )

    return df


# ── Feature column names (for model training) ─────────────────────────────────

MODEL_FEATURE_COLS = [
    "elo_rating", "elo_vs_field", "elo_delta_last3",
    "jockey_win_rate_30d", "jockey_place_rate_30d", "jockey_win_rate_overall",
    "jockey_venue_win_rate", "jockey_distance_win_rate",
    "trainer_win_rate_30d", "trainer_place_rate_30d", "trainer_win_rate_overall",
    "combo_win_rate", "combo_place_rate",
    "draw_advantage_score", "draw_position", "draw_normalised",
    "recent_form_score", "win_rate", "place_rate",
    "distance_aptitude", "going_aptitude",
    "handicap_rating", "rating_trend", "rating_vs_avg",
    "weight_carried", "weight_vs_avg", "days_since_last_run",
    "implied_win_probability", "implied_place_probability",
    "race_date", "race_start_time", "race_datetime", "race_distance", "n_runners", "venue_is_st",
]
