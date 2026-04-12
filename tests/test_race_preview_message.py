import asyncio
from types import SimpleNamespace

from notifier.telegram import TelegramNotifier


class _DummyHorse:
    def __init__(self, horse_number, horse_name, confidence, is_value_bet=False):
        self.horse_number = horse_number
        self.horse_name = horse_name
        self.confidence = confidence
        self.is_value_bet = is_value_bet


class _DummyPrediction:
    def __init__(self):
        self.top5 = [
            _DummyHorse(5, "龍騰", 0.72, True),
            _DummyHorse(2, "金駿", 0.66, False),
            _DummyHorse(9, "雷霆", 0.61, False),
        ]
        self.trio_suggestions = [(5, 2, 9), (5, 9, 2)]
        self.best_ev_horse = SimpleNamespace(horse_number=5, win_prob=0.31, implied_win_prob=0.22)
        self.confidence = 0.68


class _DummyRace:
    race_number = 7
    venue_code = "ST"
    venue = "Sha Tin"
    start_time = "20:15"
    distance = 1400
    race_class = "Class 3"
    track_type = "Turf"
    going = "GOOD"
    horses = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_race_preview_contains_independent_trio_and_quinella_sections(monkeypatch):
    notifier = TelegramNotifier(token="dummy", chat_id="dummy")
    captured = {}

    async def _fake_send(text):
        captured["text"] = text
        return True

    monkeypatch.setattr(notifier, "_send", _fake_send)

    ok = asyncio.run(notifier.send_race_preview(_DummyRace(), _DummyPrediction()))

    assert ok is True
    text = captured["text"]
    assert "🎰🔥 <b>三T推薦（進取）</b>" in text
    assert "• 主推組合: <b>5/2/9</b>" in text
    assert "• 備選組合: <b>5/9/2</b>" in text
    assert "• 星級評分: ★★★☆☆" in text
    assert "• 風險標籤: 🟡 中風險" in text
    assert "• 單注建議: 💵 <b>1.0u</b>" in text
    assert "🥈⚡ <b>孖T推薦（穩陣）</b>" in text
    assert "• 主線組合: <b>5/2</b>" in text
    assert "• 拖膽組合: <b>5/9</b>" in text
