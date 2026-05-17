"""
HKJC 賽馬預測系統 - Telegram 通知模組
Sends race previews, results, and daily summaries via Telegram Bot API.

All messages are in Traditional Chinese with emoji formatting.
"""
import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends Telegram notifications for race predictions and results.

    Requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in environment variables.
    Uses python-telegram-bot v20+ (async API).
    """

    def __init__(self, token: str = "", chat_id: str = ""):
        """
        Initialise Telegram notifier.

        Args:
            token: Telegram Bot API token
            chat_id: Target chat or group chat ID
        """
        from config import config
        self.token = token or config.TELEGRAM_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.message_style = (config.MESSAGE_STYLE or "pro").strip().lower()
        self.language_guard_enabled = bool(getattr(config, "TELEGRAM_LANGUAGE_GUARD", True))
        self._batch_blocked = False
        if self.message_style not in {"casual", "pro"}:
            self.message_style = "pro"
        self._bot: Any = None
        self._templates: Dict[str, str] = {}
        if not self.token or not self.chat_id:
            logger.warning(
                "Telegram token or chat_id not configured. "
                "Set TELEGRAM_TOKEN (or TELEGRAM_BOT_TOKEN) and TELEGRAM_CHAT_ID environment variables."
            )

    def _load_template(self, name: str) -> str:
        """Load template from disk with caching."""
        if name in self._templates:
            return self._templates[name]
        tpl_path = os.path.join(os.path.dirname(__file__), "templates", f"{name}_{self.message_style}.txt")
        fallback_path = os.path.join(os.path.dirname(__file__), "templates", f"{name}_pro.txt")
        path = tpl_path if os.path.exists(tpl_path) else fallback_path
        try:
            with open(path, encoding="utf-8") as f:
                self._templates[name] = f.read()
        except FileNotFoundError:
            self._templates[name] = "[{name}] template missing"
        return self._templates[name]

    def _get_bot(self) -> Any:
        """Create a fresh Telegram bot instance.

        IMPORTANT: Do NOT cache the Bot instance. python-telegram-bot v20+
        uses httpx internally which binds to the event loop active during
        the first HTTP request. When used across multiple asyncio.run()
        calls (each creating a new event loop), a cached Bot causes
        "Event loop is closed" errors because httpx still references the
        original (closed) loop. Creating a fresh Bot per send avoids this.
        """
        try:
            from telegram import Bot
            return Bot(token=self.token)
        except ImportError:
            logger.error("python-telegram-bot not installed: pip install python-telegram-bot==20.6")
        except Exception as e:
            logger.error(f"Failed to initialise Telegram bot: {e}")
        return None

    def is_configured(self) -> bool:
        """Check if Telegram credentials are configured."""
        return bool(self.token and self.chat_id)

    async def _send(self, text: str) -> bool:
        """
        Send a message to the configured chat.

        Args:
            text: Message text (supports HTML formatting)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured():
            logger.info(f"[TELEGRAM MOCK] Would send:\n{text}")
            return False

        bot = self._get_bot()
        if not bot:
            return False

        try:
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            logger.info(f"Telegram message sent ({len(text)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def _contains_english_horse_name(self, text: str) -> bool:
        """Detect likely English horse names in outgoing message text."""
        patterns = [
            r"\[[^\]]*[A-Za-z][^\]]*\]",  # e.g. [GOLDEN ARROW]
            r"#\d+\s+[A-Za-z][A-Za-z0-9' .\-]{1,}",  # e.g. #5 GOLDEN ARROW
        ]
        return any(re.search(p, text) for p in patterns)

    def _run_async_sync(self, coro) -> bool:
        """Run async coroutine from sync context safely.

        When an event loop is already running (e.g. after Playwright),
        schedule the coroutine on that loop via run_coroutine_threadsafe
        instead of trying to create a new loop in a separate thread.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to run directly
            try:
                return asyncio.run(coro)
            except Exception as e:
                logger.error(f"Error running async coroutine: {e}")
                return False

        # There IS a running loop — schedule on it from a helper thread
        import concurrent.futures

        def _run_on_loop():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            try:
                future = pool.submit(_run_on_loop)
                return future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.error("Timeout waiting for async coroutine")
                return False
            except Exception as e:
                logger.error(f"Error running async coroutine on existing loop: {e}")
                return False

    def send_sync(self, text: str) -> bool:
        """Synchronous wrapper for _send."""
        return self._run_async_sync(self._send(text))

    def _is_pro(self) -> bool:
        return self.message_style != "casual"

    def _risk_profile(self, score: float, high_threshold: float = 0.75, mid_threshold: float = 0.60) -> tuple[str, str]:
        """Return risk tag and unit stake suggestion from a 0-1 confidence score."""
        if score >= high_threshold:
            return "🟢 低風險", "2.0u"
        if score >= mid_threshold:
            return "🟡 中風險", "1.0u"
        return "🔴 高風險", "0.5u"

    def _star_rating(self, score: float) -> str:
        """Render a 5-star string from a 0-1 confidence score."""
        filled = max(1, min(5, int(round(score * 5))))
        return "★" * filled + "☆" * (5 - filled)

    async def send_test(self, mode: str = "preview") -> str:
        """
        Generate a test message without sending to Telegram.
        Returns the formatted text for local preview.

        Args:
            mode: "preview" | "summary" | "result" | "alert"
        """
        from model.backtest import Backtester
        from model.self_optimizer import StrategySelfOptimizer
        from model.trainer import EnsembleTrainer
        import io
        import logging

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        if mode == "preview":
            text = self._build_test_preview()
        elif mode == "summary":
            text = self._build_test_summary()
        elif mode == "result":
            text = self._build_test_result()
        else:
            text = self._build_test_alert()

        root_logger.removeHandler(handler)
        return text

    def _build_test_preview(self) -> str:
        """Build a demo race preview message (no Telegram send)."""
        now = datetime.now().strftime("%H:%M")
        return (
            f"🏇✨ <b>賽前雷達｜第5場</b> — 沙田 {now}\n"
            f"\n"
            f"📌 1200m | Class 3 | 草地 | GOOD\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>焦點三甲</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"1️⃣ 馬號3 [GOLDEN ACE] — 信心 78% ⭐值博\n"
            f"2️⃣ 馬號7 [LUCKY STAR] — 信心 65%\n"
            f"3️⃣ 馬號1 [SUPER FAST] — 信心 58%\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎰 <b>三T（進取）</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 主推: <b>3/7/1</b>\n"
            f"🔹 備選: <b>3/7/5</b>\n"
            f"⭐ ★★★★☆\n"
            f"🏷️ 🟢 低風險 | 💵 <b>2.0u</b>\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🥈 <b>孖T（穩陣）</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 主線: <b>3/7</b>\n"
            f"🔹 拖膽: <b>3/1</b>\n"
            f"⭐ ★★★★☆\n"
            f"🏷️ 🟢 低風險 | 💵 <b>2.0u</b>\n"
            f"\n"
            f"📊 <b>EV 亮點</b>: 馬號3 模型52% vs 賠率隱含28%\n"
            f"\n"
            f"🔢 共 12 匹 | 信心 72%\n"
            f"📈 <b>優化門檻</b>: min_conf=50% max_odds=18\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>🧪 僅供研究參考，不構成投注建議</i>"
        )

    def _build_test_summary(self) -> str:
        """Build a demo daily summary message."""
        return (
            f"📊🌙 <b>每日戰報</b> — {datetime.now().strftime('%Y-%m-%d')}\n"
            f"\n"
            f"🏇 今日賽事完成 ✅\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>核心指標（近30日）</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 首選命中: 38.5%\n"
            f"🏆 頭三到位: 62.3%\n"
            f"📊 總預測場次: 100 場\n"
            f"\n"
            f"📈 <b>ROI 分析</b>\n"
            f"• 30日 ROI: <b>+12.4%</b>\n"
            f"• 7日 ROI: +5.2%\n"
            f"• 勝出 ROI: +15.8%\n"
            f"• 位置 ROI: +8.3%\n"
            f"💰 30日淨利潤: <b>+$124.00</b>\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ <b>策略設定</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• 信心門檻: 50%\n"
            f"• 最大賠率: 18.0\n"
            f"• 優化分數: 0.4614\n"
            f"\n"
            f"📈 優化歷史 (最近3次):\n"
            f"  05/16 conf=0.50 odds=18 score=0.461\n"
            f"  05/15 conf=0.55 odds=18 score=0.423\n"
            f"  05/14 conf=0.55 odds=25 score=0.398\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📉 <b>校準分析</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• 中低 (50%): 樣本28 實際勝率0% ⚠️ 過度高估\n"
            f"• 中 (58%):  樣本31 實際勝率0% ⚠️\n"
            f"• 高 (70%):  樣本41 實際勝率100% ✅\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🏆 <b>Top 5 重要特徵</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"1. elo_vs_field\n"
            f"2. win_rate_x_odds\n"
            f"3. rating_trend\n"
            f"4. margin_x_form\n"
            f"5. elo_rating\n"
            f"\n"
            f"<i>🧪 僅供研究參考，不構成投注建議</i>"
        )

    def _build_test_result(self) -> str:
        """Build a demo race result message."""
        return (
            f"🏁 <b>賽果速報｜第3場</b>\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>預測 vs 實際</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🔮 預測TOP3: #5 / #2 / #11\n"
            f"🏁 實際TOP3: #5 / #2 / #11\n"
            f"\n"
            f"✅ <b>三甲全中！</b> 🎉\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>模擬回報</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• 勝出注 (首選 #5): <b>+$48.00</b> ✅\n"
            f"• 位置注 (#5): <b>+$11.50</b> ✅\n"
            f"• 三T注 (5/2/11): <b>+$340.00</b> 🎰\n"
            f"\n"
            f"<i>🧪 模擬計算，僅供研究參考</i>"
        )

    def _build_test_alert(self) -> str:
        """Build a demo system alert message."""
        return (
            f"🚨 <b>系統警報</b>\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <b>數據源異常</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"• 源: HKJC GraphQL API\n"
            f"• 錯誤: 400 Bad Request\n"
            f"• 影響: 已自動降級至 HTML 模式\n"
            f"• 後續: 15分鐘後自動重試\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📋 <b>系統狀態</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• 模型: ✅ 已訓練\n"
            f"• Scraper: ⚠️ 降級中\n"
            f"• Telegram: ✅ 正常\n"
            f"• 最後同步: {datetime.now().strftime('%H:%M:%S')}\n"
            f"\n"
            f"<i>系統自動通知，無需操作</i>"
        )

    def send_test_sync(self, mode: str = "preview") -> None:
        """Print test message to stdout (for local verification)."""
        text = self._run_async_sync(self.send_test(mode))
        print(text)

    async def send_race_preview(self, race, prediction) -> bool:
        """
        Send a race preview notification 30 minutes before the race.

        Args:
            race: Race object
            prediction: PredictionResult from predictor

        Returns:
            True if sent successfully
        """
        if not prediction or not prediction.top5:
            return False

        top3 = prediction.top5[:3]
        venue_map = {"ST": "沙田", "HV": "跑馬地"}
        venue_ch = venue_map.get(race.venue_code, race.venue)

        # Build top-3 lines
        rank_emojis = ["1️⃣", "2️⃣", "3️⃣"]
        horse_lines = []
        for i, horse in enumerate(top3):
            value_tag = " ⭐值博" if horse.is_value_bet else ""
            confidence_pct = int(horse.confidence * 100)
            line = (
                f"{rank_emojis[i]} 馬號{horse.horse_number} "
                f"[{horse.horse_name}] - 信心度 {confidence_pct}%{value_tag}"
            )
            horse_lines.append(line)

        # Trio suggestion
        trio_recommendations = []
        for trio in list(getattr(prediction, "trio_suggestions", []) or [])[:2]:
            if len(trio) >= 3:
                trio_recommendations.append(f"{int(trio[0])}/{int(trio[1])}/{int(trio[2])}")

        primary_trio = trio_recommendations[0] if trio_recommendations else "N/A"
        backup_trio = trio_recommendations[1] if len(trio_recommendations) > 1 else "N/A"

        quinella_main = "N/A"
        quinella_backup = "N/A"
        if top3 and len(top3) >= 2:
            quinella_main = f"{int(top3[0].horse_number)}/{int(top3[1].horse_number)}"
        if top3 and len(top3) >= 3:
            quinella_backup = f"{int(top3[0].horse_number)}/{int(top3[2].horse_number)}"

        trio_risk_tag, trio_stake = self._risk_profile(float(prediction.confidence or 0.0))
        trio_stars = self._star_rating(float(prediction.confidence or 0.0))

        quinella_score = 0.0
        if len(top3) >= 2:
            quinella_score = (float(top3[0].confidence or 0.0) + float(top3[1].confidence or 0.0)) / 2.0
        quinella_risk_tag, quinella_stake = self._risk_profile(quinella_score, high_threshold=0.72, mid_threshold=0.58)
        quinella_stars = self._star_rating(quinella_score)

        # Best EV
        ev_str = "N/A"
        if prediction.best_ev_horse:
            bev = prediction.best_ev_horse
            bev_model_pct = int(bev.win_prob * 100)
            bev_implied_pct = int(bev.implied_win_prob * 100)
            ev_str = f"馬號{bev.horse_number} (模型{bev_model_pct}% vs 賠率隱含{bev_implied_pct}%)"

        title = "🏇✨ <b>賽前雷達｜第{race}場</b>" if self._is_pro() else "🏇 <b>開跑前提你｜第{race}場</b>"
        top3_title = "🎯 <b>焦點三甲預測 (不分先後)</b>" if self._is_pro() else "🎯 <b>我睇好呢三隻 (不分先後)</b>"
        trio_label = "🎰🔥 <b>三T推薦（進取）</b>" if self._is_pro() else "🎰 <b>三T推介（進取）</b>"
        quinella_label = "🥈⚡ <b>孖T推薦（穩陣）</b>" if self._is_pro() else "🥈 <b>孖T推介（穩陣）</b>"
        ev_label = "📊 EV亮點" if self._is_pro() else "📊 值博重點"
        confidence_label = "📈 全場模型信心度" if self._is_pro() else "📈 今場整體信心"

        text = (
            f"{title.format(race=race.race_number)} - {venue_ch} {race.start_time}\n"
            f"📍 距離: {race.distance}m | 班次: {race.race_class} | "
            f"場地: {'草地' if race.track_type == 'Turf' else '全天候'}\n"
            f"⛅ 場地狀況: {race.going}\n"
            f"\n"
            f"{top3_title}\n"
            + "\n".join(horse_lines) +
            f"\n\n"
            f"{trio_label}\n"
            f"• 主推組合: <b>{primary_trio}</b>\n"
            f"• 備選組合: <b>{backup_trio}</b>\n"
            f"• 星級評分: {trio_stars}\n"
            f"• 風險標籤: {trio_risk_tag}\n"
            f"• 單注建議: 💵 <b>{trio_stake}</b>\n"
            f"\n"
            f"{quinella_label}\n"
            f"• 主線組合: <b>{quinella_main}</b>\n"
            f"• 拖膽組合: <b>{quinella_backup}</b>\n"
            f"• 星級評分: {quinella_stars}\n"
            f"• 風險標籤: {quinella_risk_tag}\n"
            f"• 單注建議: 💵 <b>{quinella_stake}</b>\n"
            f"{ev_label}: {ev_str}\n"
            f"\n"
            f"🔢 共 {len(race.horses)} 匹參賽\n"
            f"{confidence_label}: {int(prediction.confidence * 100)}%\n"
            f"\n"
            f"<i>{'🧪 僅供研究參考，不構成投注建議' if self._is_pro() else '🧪 研究用途，唔係投注建議'}</i>"
        )

        return await self._send(text)

    async def send_race_result(self, race, actual_result: List[int], prediction) -> bool:
        """
        Send race result and prediction comparison after the race.

        Args:
            race: Race object
            actual_result: Finishing order (list of horse numbers)
            prediction: PredictionResult used for this race

        Returns:
            True if sent successfully
        """
        if not actual_result:
            return False

        venue_map = {"ST": "沙田", "HV": "跑馬地"}
        venue_ch = venue_map.get(race.venue_code, race.venue)

        actual_top3 = actual_result[:3]
        predicted_nums = [h.horse_number for h in prediction.top5[:3]] if prediction and prediction.top5 else []

        # Count hits
        hits = len(set(predicted_nums) & set(actual_top3))
        trio_hit = set(predicted_nums) == set(actual_top3) and len(predicted_nums) == 3

        # Hit indicator
        hit_emojis = {0: "❌", 1: "🟡", 2: "🟠", 3: "✅"}
        hit_emoji = hit_emojis.get(hits, "❓")

        # Winner info
        winner_num = actual_top3[0] if actual_top3 else 0
        winner_correct = winner_num in predicted_nums[:1]

        predicted_str = " / ".join(f"#{n}" for n in predicted_nums) if predicted_nums else "N/A"
        actual_str = " / ".join(f"#{n}" for n in actual_top3)

        tpl = self._load_template("race_result")
        winner_status = "✅ 首選命中，節奏漂亮！" if winner_correct else "❌ 首選未中，下場再追！"
        trio_status = "🎉 三T完整命中！" if trio_hit else "📌 三T未齊，持續追蹤變化。"
        text = tpl.format(
            race=race.race_number,
            venue_ch=venue_ch,
            actual_str=actual_str,
            predicted_str=predicted_str,
            hit_emoji=hit_emoji,
            hits=hits,
            winner_status=winner_status,
            trio_status=trio_status,
        )

        return await self._send(text)

    async def send_daily_summary(self, race_date: str, stats: Dict) -> bool:
        """
        Send end-of-day performance summary.

        Args:
            race_date: Date in "YYYY-MM-DD" format
            stats: Summary stats dict from Backtester.get_summary_stats()

        Returns:
            True if sent successfully
        """
        total = stats.get("races_30d", 0)
        win_acc = stats.get("winner_accuracy", 0)
        top3_rate = stats.get("top3_hit_rate", 0)
        roi_7d = stats.get("roi_7d", 0)
        roi_30d = stats.get("roi_30d", 0)
        net_30d = stats.get("net_profit_30d", 0)

        roi_emoji = "📈" if roi_30d >= 0 else "📉"

        title = "📊🌙 <b>每日戰報</b>" if self._is_pro() else "📊 <b>今日總結</b>"
        kpi_title = "<b>近30日核心指標</b>" if self._is_pro() else "<b>近30日表現</b>"

        text = (
            f"{title} - {race_date}\n"
            f"\n"
            f"🏇 今日賽事: 已完成\n"
            f"\n"
            f"{kpi_title}\n"
            f"🎯 首選準確率: {win_acc:.1%}\n"
            f"🏆 頭三到位率: {top3_rate:.1%}\n"
            f"{roi_emoji} 30日 ROI: {roi_30d:+.1f}%\n"
            f"7日 ROI: {roi_7d:+.1f}%\n"
            f"💰 30日淨利潤: {'+'if net_30d >= 0 else ''}${net_30d:.0f}\n"
            f"\n"
            f"<i>{'🧪 本系統僅供研究參考，不構成投注建議' if self._is_pro() else '🧪 研究用途參考，唔係投注建議'}</i>"
        )

        return await self._send(text)

    async def send_system_alert(self, message: str, level: str = "INFO") -> bool:
        """
        Send a system alert (e.g., data fetch failure).

        Args:
            message: Alert message
            level: "INFO" | "WARNING" | "ERROR"

        Returns:
            True if sent
        """
        level_emojis = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨"}
        emoji = level_emojis.get(level, "📢")
        tpl = self._load_template("system_alert")
        text = tpl.format(
            emoji=emoji,
            level=level,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message=message,
        )
        return await self._send(text)

    def send_race_preview_sync(self, race, prediction) -> bool:
        """Synchronous wrapper for send_race_preview."""
        return self._run_async_sync(self.send_race_preview(race, prediction))
