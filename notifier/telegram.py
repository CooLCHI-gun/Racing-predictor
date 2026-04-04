"""
HKJC 賽馬預測系統 - Telegram 通知模組
Sends race previews, results, and daily summaries via Telegram Bot API.

All messages are in Traditional Chinese with emoji formatting.
"""
import asyncio
import logging
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
        self._bot: Any = None

        if not self.token or not self.chat_id:
            logger.warning(
                "Telegram token or chat_id not configured. "
                "Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables."
            )

    def _get_bot(self) -> Any:
        """Create a Telegram bot bound to the current event loop."""
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

    def _run_async_sync(self, coro) -> bool:
        """Run async coroutine from sync context safely."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)
        except Exception as e:
            logger.error(f"Error running async coroutine: {e}")
            return False

    def send_sync(self, text: str) -> bool:
        """Synchronous wrapper for _send."""
        return self._run_async_sync(self._send(text))

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
        trio_str = "N/A"
        if prediction.trio_suggestions:
            t = prediction.trio_suggestions[0]
            trio_str = f"{t[0]}/{t[1]}/{t[2]}"

        # Best EV
        ev_str = "N/A"
        if prediction.best_ev_horse:
            bev = prediction.best_ev_horse
            bev_model_pct = int(bev.win_prob * 100)
            bev_implied_pct = int(bev.implied_win_prob * 100)
            ev_str = f"馬號{bev.horse_number} (模型{bev_model_pct}% vs 賠率隱含{bev_implied_pct}%)"

        text = (
            f"🏇 <b>第{race.race_number}場</b> - {venue_ch} {race.start_time}\n"
            f"📍 距離: {race.distance}m | 班次: {race.race_class} | "
            f"場地: {'草地' if race.track_type == 'Turf' else '全天候'}\n"
            f"⛅ 場地狀況: {race.going}\n"
            f"\n"
            f"🎯 <b>頭三名預測 (不分先後):</b>\n"
            + "\n".join(horse_lines) +
            f"\n\n"
            f"💰 三T建議: <b>{trio_str}</b>\n"
            f"📊 EV最佳: {ev_str}\n"
            f"\n"
            f"🔢 共 {len(race.horses)} 匹參賽\n"
            f"📈 模型信心度: {int(prediction.confidence * 100)}%\n"
            f"\n"
            f"<i>⚠️ 僅供研究參考，不構成投注建議</i>"
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

        text = (
            f"🏁 <b>第{race.race_number}場結果</b> - {venue_ch}\n"
            f"\n"
            f"🥇 實際頭三: {actual_str}\n"
            f"🎯 預測頭三: {predicted_str}\n"
            f"\n"
            f"{hit_emoji} 命中率: {hits}/3\n"
            f"{'✅ 首選馬勝出！' if winner_correct else '❌ 首選馬落敗'}\n"
            f"{'🎉 三T完整命中！' if trio_hit else ''}\n"
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

        text = (
            f"📊 <b>每日賽事總結</b> - {race_date}\n"
            f"\n"
            f"🏇 今日賽事: 完成\n"
            f"\n"
            f"<b>近30日成績:</b>\n"
            f"🎯 首選準確率: {win_acc:.1%}\n"
            f"🏆 頭三到位率: {top3_rate:.1%}\n"
            f"{roi_emoji} 30日 ROI: {roi_30d:+.1f}%\n"
            f"7日 ROI: {roi_7d:+.1f}%\n"
            f"💰 30日淨利潤: {'+'if net_30d >= 0 else ''}${net_30d:.0f}\n"
            f"\n"
            f"<i>⚠️ 本系統僅供研究參考，不構成投注建議</i>"
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

        text = (
            f"{emoji} <b>系統通知 [{level}]</b>\n"
            f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\n"
            f"{message}"
        )
        return await self._send(text)

    def send_race_preview_sync(self, race, prediction) -> bool:
        """Synchronous wrapper for send_race_preview."""
        return self._run_async_sync(self.send_race_preview(race, prediction))
