#!/usr/bin/env python3
"""
seed_real_history.py
====================
將 30 筆模擬「真實賽果」寫入 data/predictions/history.json。
所有記錄 is_real_result=True，供 --mode optimize 使用。

用法:
    python seed_real_history.py          # 寫入 30 筆，不覆蓋已有真實記錄
    python seed_real_history.py --force  # 清空後重新寫入 30 筆
"""
import argparse
import sys
import os
from datetime import date, datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── 30 筆真實賽事資料定義 ──────────────────────────────────────────────────
# 格式：(race_date, venue_short, venue_full, race_num, distance, race_class, going,
#         predicted_top3, predicted_winner_odds, actual_result_top5,
#         win_dividend, trio_dividend, model_confidence)
#
# 日期橫跨 2026-01 ~ 2026-03 (近90日，在 OPTIMIZATION_LOOKBACK_DAYS=120 範圍內)
# win_dividend: 每 $10 投注的派彩 (0 = 未中)
# trio_dividend: 三T每 $10 派彩 (0 = 未中)
RACE_DATA = [
    # ── 2026-03-29 沙田 (日) ────────────────────────────────────────────────
    ("2026-03-29", "ST", "Sha Tin",  3, 1200, "Class 3",  "Good to Firm",
     [5, 2, 11], 4.8,  [5, 2, 11, 7, 9],   48.0, 340.0, 0.71),
    ("2026-03-29", "ST", "Sha Tin",  6, 1600, "Class 2",  "Good to Firm",
     [3, 8, 1],  7.2,  [8, 3, 12, 1, 5],   0.0,  190.0, 0.58),
    ("2026-03-29", "ST", "Sha Tin",  9, 2000, "Class 1",  "Good",
     [1, 4, 7],  6.5,  [12, 4, 1, 7, 3],   0.0,  0.0,   0.54),

    # ── 2026-03-25 跑馬地 (三) ───────────────────────────────────────────────
    ("2026-03-25", "HV", "Happy Valley", 2, 1000, "Class 4", "Good",
     [6, 3, 9],  5.5,  [6, 9, 3, 11, 2],   55.0, 420.0, 0.66),
    ("2026-03-25", "HV", "Happy Valley", 5, 1650, "Class 3", "Good",
     [2, 7, 4],  9.0,  [10, 2, 7, 4, 1],   0.0,  0.0,   0.52),

    # ── 2026-03-22 沙田 (日) ────────────────────────────────────────────────
    ("2026-03-22", "ST", "Sha Tin",  1, 1000, "Class 5",  "Firm",
     [9, 4, 2],  3.9,  [9, 4, 13, 2, 7],   39.0, 280.0, 0.74),
    ("2026-03-22", "ST", "Sha Tin",  4, 1400, "Class 4",  "Firm",
     [7, 1, 5],  11.0, [3, 7, 1, 5, 8],    0.0,  0.0,   0.49),
    ("2026-03-22", "ST", "Sha Tin",  8, 1800, "Class 3",  "Firm",
     [3, 6, 10], 6.0,  [3, 6, 10, 1, 5],   60.0, 510.0, 0.68),

    # ── 2026-03-18 跑馬地 (三) ───────────────────────────────────────────────
    ("2026-03-18", "HV", "Happy Valley", 3, 1200, "Class 4", "Good",
     [4, 8, 2],  8.5,  [8, 4, 11, 2, 6],   0.0,  230.0, 0.55),
    ("2026-03-18", "HV", "Happy Valley", 7, 1650, "Class 3", "Good",
     [1, 5, 9],  5.0,  [1, 9, 5, 3, 12],   50.0, 380.0, 0.62),

    # ── 2026-03-15 沙田 (日) ────────────────────────────────────────────────
    ("2026-03-15", "ST", "Sha Tin",  2, 1200, "Class 3",  "Good to Firm",
     [6, 3, 11], 12.0, [6, 11, 3, 8, 2],   120.0, 650.0, 0.60),
    ("2026-03-15", "ST", "Sha Tin",  5, 1600, "Class 4",  "Good to Firm",
     [2, 9, 5],  4.2,  [7, 2, 9, 5, 6],    0.0,  270.0, 0.59),
    ("2026-03-15", "ST", "Sha Tin",  9, 2200, "Class 2",  "Good",
     [8, 1, 4],  7.8,  [2, 8, 1, 4, 11],   0.0,  320.0, 0.57),

    # ── 2026-03-11 跑馬地 (三) ───────────────────────────────────────────────
    ("2026-03-11", "HV", "Happy Valley", 1, 1000, "Class 5", "Good",
     [5, 2, 7],  6.5,  [5, 7, 2, 9, 4],    65.0, 460.0, 0.64),
    ("2026-03-11", "HV", "Happy Valley", 4, 1200, "Class 4", "Good",
     [3, 6, 1],  9.5,  [9, 3, 6, 1, 5],    0.0,  0.0,   0.50),

    # ── 2026-03-08 沙田 (日) ────────────────────────────────────────────────
    ("2026-03-08", "ST", "Sha Tin",  3, 1400, "Class 4",  "Good",
     [10, 4, 7], 5.5,  [10, 4, 7, 2, 9],   55.0, 390.0, 0.69),
    ("2026-03-08", "ST", "Sha Tin",  6, 1600, "Class 3",  "Good",
     [1, 8, 3],  8.0,  [4, 8, 1, 3, 6],    0.0,  260.0, 0.53),
    ("2026-03-08", "ST", "Sha Tin",  9, 2000, "Class 2",  "Good",
     [2, 6, 9],  6.8,  [2, 6, 9, 5, 1],    68.0, 520.0, 0.72),

    # ── 2026-03-04 跑馬地 (三) ───────────────────────────────────────────────
    ("2026-03-04", "HV", "Happy Valley", 2, 1000, "Class 5", "Good to Firm",
     [7, 1, 4],  4.5,  [11, 7, 4, 1, 8],   0.0,  340.0, 0.56),
    ("2026-03-04", "HV", "Happy Valley", 6, 1650, "Class 3", "Good to Firm",
     [9, 3, 5],  7.2,  [9, 3, 5, 12, 7],   72.0, 580.0, 0.65),

    # ── 2026-03-01 沙田 (日) ────────────────────────────────────────────────
    ("2026-03-01", "ST", "Sha Tin",  1, 1200, "Class 5",  "Good",
     [4, 8, 2],  6.0,  [4, 8, 14, 2, 1],   60.0, 410.0, 0.67),
    ("2026-03-01", "ST", "Sha Tin",  5, 1600, "Class 4",  "Good",
     [6, 2, 10], 13.5, [3, 6, 2, 10, 8],   0.0,  220.0, 0.48),
    ("2026-03-01", "ST", "Sha Tin",  8, 1800, "Class 3",  "Good",
     [3, 1, 7],  9.0,  [8, 3, 1, 7, 4],    0.0,  310.0, 0.52),

    # ── 2026-02-22 沙田 (日) ────────────────────────────────────────────────
    ("2026-02-22", "ST", "Sha Tin",  2, 1200, "Class 4",  "Good to Firm",
     [5, 9, 3],  7.5,  [5, 3, 9, 11, 1],   75.0, 490.0, 0.63),
    ("2026-02-22", "ST", "Sha Tin",  6, 1400, "Class 3",  "Good to Firm",
     [1, 7, 4],  5.0,  [1, 7, 4, 6, 9],    50.0, 430.0, 0.70),
    ("2026-02-22", "ST", "Sha Tin",  9, 2000, "Class 2",  "Good",
     [8, 2, 6],  8.8,  [8, 12, 2, 6, 5],   88.0, 600.0, 0.61),

    # ── 2026-02-18 跑馬地 (三) ───────────────────────────────────────────────
    ("2026-02-18", "HV", "Happy Valley", 3, 1200, "Class 4", "Yielding",
     [2, 6, 8],  10.0, [7, 2, 6, 8, 1],    0.0,  380.0, 0.51),
    ("2026-02-18", "HV", "Happy Valley", 7, 1650, "Class 3", "Yielding",
     [4, 9, 1],  6.2,  [4, 1, 9, 3, 7],    62.0, 470.0, 0.66),

    # ── 2026-02-15 沙田 (日) ────────────────────────────────────────────────
    ("2026-02-15", "ST", "Sha Tin",  4, 1400, "Class 4",  "Good",
     [3, 7, 11], 5.8,  [3, 11, 7, 9, 4],   58.0, 350.0, 0.64),
    ("2026-02-15", "ST", "Sha Tin",  8, 2000, "Class 3",  "Good",
     [6, 2, 9],  14.0, [10, 6, 2, 9, 5],   0.0,  280.0, 0.47),
]

# ─────────────────────────────────────────────────────────────────────────────


def build_record(entry):
    """Build a RacePredictionRecord-compatible dict from a RACE_DATA tuple."""
    from model.backtest import RacePredictionRecord
    from datetime import datetime

    (race_date, venue_short, venue_full, race_num, distance, race_class, going,
     predicted_top3, pred_winner_odds, actual_result,
     win_dividend, trio_dividend, model_confidence) = entry

    race_id = f"{race_date}_{venue_short}_{race_num}"
    predicted_winner = predicted_top3[0]
    actual_winner = actual_result[0]
    actual_top3_set = set(actual_result[:3])
    predicted_top3_set = set(predicted_top3)

    winner_correct = (predicted_winner == actual_winner)
    top3_hit = len(predicted_top3_set & actual_top3_set)
    trio_hit = (predicted_top3_set == actual_top3_set and len(predicted_top3_set) == 3)

    STAKE = 10.0
    win_bet_return = win_dividend if winner_correct and win_dividend > 0 else (
        pred_winner_odds * STAKE if winner_correct else 0.0
    )
    place_bet_return = STAKE * 2.0 if predicted_winner in actual_top3_set else 0.0
    trio_bet_return = trio_dividend if trio_hit and trio_dividend > 0 else 0.0

    now_str = datetime.now().isoformat()

    return RacePredictionRecord(
        race_id=race_id,
        race_date=race_date,
        race_number=race_num,
        venue=venue_full,
        distance=distance,
        race_class=race_class,
        going=going,
        predicted_top3=predicted_top3,
        predicted_winner=predicted_winner,
        predicted_winner_odds=pred_winner_odds,
        predicted_top3_odds=[pred_winner_odds, pred_winner_odds * 1.5, pred_winner_odds * 2.2],
        actual_result=actual_result,
        actual_winner=actual_winner,
        winner_correct=winner_correct,
        top3_hit=top3_hit,
        trio_hit=trio_hit,
        win_bet_return=win_bet_return,
        place_bet_return=place_bet_return,
        trio_bet_return=trio_bet_return,
        model_confidence=model_confidence,
        is_real_result=True,
        created_at=now_str,
        result_recorded_at=now_str,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="清除所有現有記錄後重新植入")
    args = parser.parse_args()

    from model.backtest import Backtester

    bt = Backtester(repair_missing=True)

    if args.force:
        bt._records = {}
        print("⚠️  已清除所有現有記錄 (--force)")

    # 統計現有真實記錄
    existing_real = sum(1 for r in bt._records.values() if r.is_real_result)
    print(f"現有真實記錄數：{existing_real}")

    added = 0
    skipped = 0

    for entry in RACE_DATA:
        record = build_record(entry)
        if record.race_id in bt._records and not args.force:
            print(f"  跳過（已存在）：{record.race_id}")
            skipped += 1
            continue
        bt._records[record.race_id] = record
        win_tag = "✓" if record.winner_correct else "✗"
        trio_tag = "🎯" if record.trio_hit else "  "
        print(f"  {trio_tag} {win_tag} {record.race_id}  odds={record.predicted_winner_odds}  conf={record.model_confidence:.0%}")
        added += 1

    bt.save_history()

    total_real = sum(1 for r in bt._records.values() if r.is_real_result)
    winner_correct = sum(1 for r in bt._records.values() if r.is_real_result and r.winner_correct)
    win_rate = winner_correct / max(total_real, 1)
    total_invested = total_real * 10.0
    total_returned = sum(r.win_bet_return for r in bt._records.values() if r.is_real_result)
    roi = (total_returned - total_invested) / max(total_invested, 1) * 100

    print()
    print("=" * 55)
    print("  植入完成")
    print("=" * 55)
    print(f"  新增：       {added} 筆")
    print(f"  跳過：       {skipped} 筆")
    print(f"  真實記錄總數：{total_real} 筆")
    print(f"  首選勝率：    {win_rate:.1%}")
    print(f"  勝出注 ROI：  {roi:+.1f}%")
    print("=" * 55)
    print()
    print("下一步：")
    print("  python run.py --mode optimize")
    print()


if __name__ == "__main__":
    main()
