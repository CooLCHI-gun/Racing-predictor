from types import SimpleNamespace

import run


class _DummyBacktester:
    def __init__(self, repair_missing=True):
        self.repair_missing = repair_missing

    def get_summary_stats(self):
        return {
            "total_predictions": 12,
            "winner_accuracy": 0.25,
            "top3_hit_rate": 0.5,
        }

    def calculate_roi(self, strategy, period_days, real_only=True):
        return SimpleNamespace(win_roi=12.34, total_races=7)

    def get_settled_records(self, real_only=True):
        return [
            SimpleNamespace(
                race_id="2026-04-11_ST_<9>&",
                predicted_top3=[1, 2, 3],
                actual_result=[3, 2, 1],
                winner_correct=False,
                top3_hit=2,
                trio_hit=False,
                predicted_winner_odds=9.8,
                model_confidence=0.63,
                result_recorded_at="2026-04-11T22:00:00+08:00",
                race_date="2026-04-11",
                race_number=9,
            )
        ]

    def build_walk_forward_report(self, **kwargs):
        return {"summary": {"windows": 1}}


class _DummyOptimizer:
    def __init__(self, path):
        self.path = path

    def optimize(self, recent, min_samples=24):
        return SimpleNamespace(
            min_confidence=0.67,
            max_win_odds=18.5,
            selected_samples=42,
            selected_win_rate=0.33,
            selected_roi=8.75,
        )

    def optimize_segmented(self, recent, min_samples=12):
        return {"sprint": {"min_confidence": 0.7}}


class _DummyNotifier:
    sent_messages = []

    def is_configured(self):
        return True

    def send_sync(self, text):
        self.__class__.sent_messages.append(text)
        return True


def test_maintenance_summary_escapes_html_sensitive_content(monkeypatch, tmp_path):
    import model.backtest as backtest_module
    import model.self_optimizer as optimizer_module
    import notifier.telegram as telegram_module

    _DummyNotifier.sent_messages.clear()

    monkeypatch.setattr(backtest_module, "Backtester", _DummyBacktester)
    monkeypatch.setattr(optimizer_module, "StrategySelfOptimizer", _DummyOptimizer)
    monkeypatch.setattr(telegram_module, "TelegramNotifier", _DummyNotifier)

    monkeypatch.setattr(run, "_load_json_state", lambda path: {})
    monkeypatch.setattr(run, "_save_json_state", lambda path, state: None)
    monkeypatch.setattr(run, "_get_daily_alert_noise_summary", lambda day: {
        "total_suppressed": 2,
        "unique_fingerprints": 1,
        "top_noisy": [
            {
                "fingerprint": "finger<print>&1",
                "mode": "cron<mode>",
                "error_type": "Value&Error",
                "suppressed_count": 2,
            }
        ],
    })
    monkeypatch.setattr(run, "_sync_recent_real_results", lambda days_back=2: {"updated": 0, "checked_dates": []})
    monkeypatch.setattr(run, "_apply_retention_policy", lambda: {"skipped": True, "reason": "disabled"})
    monkeypatch.setattr(run, "_load_training_retention_manifest", lambda: {})
    monkeypatch.setattr(run, "_evaluate_training_retention_status", lambda settled_real, manifest: (
        settled_real,
        {
            "ready_for_retrain": True,
            "allowed_tiers": ["A", "B"],
            "blocking_reasons": ["need <more> & checks"],
            "counts_by_tier": {"A": 1, "B": 0, "C": 0, "D": 0},
            "eligible_records": 1,
        },
    ))
    monkeypatch.setattr(run, "_latest_settled_marker", lambda settled_real: "marker-1")
    monkeypatch.setattr(run, "_run_daily_retrain", lambda candidates, state: {
        "attempted": True,
        "ran": False,
        "reason": "blocked <training> & waiting",
        "new_settled_since_last": 0,
        "target_real_rows": 0,
        "metrics": {},
    })
    monkeypatch.setattr(run, "_write_maintenance_report", lambda report: "data/reports/maintenance_<daily>&.json")
    monkeypatch.setattr(run, "_write_authenticity_audit_report", lambda bt: "data/reports/auth_<audit>&.json")

    monkeypatch.setattr(run.config, "CRON_STATE_FILE", str(tmp_path / "cron_state.json"), raising=False)
    monkeypatch.setattr(run.config, "MAINTENANCE_REPORT_DIR", str(tmp_path / "reports"), raising=False)
    monkeypatch.setattr(run.config, "OPTIMIZATION_LOOKBACK_DAYS", 30, raising=False)
    monkeypatch.setattr(run.config, "STRATEGY_PROFILE_PATH", str(tmp_path / "strategy_profile.json"), raising=False)
    monkeypatch.setattr(run.config, "ENABLE_RETENTION_CLEANUP", False, raising=False)
    monkeypatch.setattr(run.config, "ENABLE_DAILY_ALERT_NOISE_SUMMARY", True, raising=False)
    monkeypatch.setattr(run.config, "MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED", False, raising=False)
    monkeypatch.setattr(run.config, "MESSAGE_STYLE", "pro", raising=False)

    run.run_maintenance()

    assert len(_DummyNotifier.sent_messages) == 1
    message = _DummyNotifier.sent_messages[0]

    assert "blocked &lt;training&gt; &amp; waiting" in message
    assert "2026-04-11_ST_&lt;9&gt;&amp;" in message
    assert "finger&lt;print&gt;&amp;1 [cron&lt;mode&gt;] Value&amp;Error: 2 次" in message
    assert "data/reports/maintenance_&lt;daily&gt;&amp;.json" in message

    assert "blocked <training> & waiting" not in message
    assert "2026-04-11_ST_<9>&" not in message
    assert "finger<print>&1 [cron<mode>] Value&Error" not in message
    assert "data/reports/maintenance_<daily>&.json" not in message