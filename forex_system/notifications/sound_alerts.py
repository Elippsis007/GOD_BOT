# notifications/sound_alerts.py
import threading
import pytz
from datetime import datetime
from monitoring.logger import logger
from config.settings   import CONFIG

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


class SoundAlerts:
    """
    Windows sound alerts for trading signals.
    Different tones for different alert types.
    Runs in background thread — never blocks trading.

    Quiet hours are respected automatically — sounds are suppressed
    during the same window configured for Telegram quiet hours
    (CONFIG.TELEGRAM_QUIET_START to CONFIG.TELEGRAM_QUIET_END).
    Danger alerts bypass quiet hours, matching Telegram behaviour.
    """

    def __init__(self):
        # Auto-disable if winsound is not available (non-Windows platform)
        self.enabled  = SOUND_AVAILABLE
        self.timezone = pytz.timezone("Europe/Madrid")

    # ── Enable / Disable ──────────────────────────────────────────────────────
    # Fix 2 – Add explicit enable() and disable() methods so alert_manager.py
    # can call them directly without relying solely on toggle().
    def enable(self) -> None:
        """Enable sound alerts (no-op on non-Windows platforms)."""
        if not SOUND_AVAILABLE:
            logger.info(
                "🔊 Sound enable requested but winsound not available "
                "on this platform"
            )
            return
        self.enabled = True
        logger.info("🔊 Sound alerts: ON")

    def disable(self) -> None:
        """Disable sound alerts."""
        self.enabled = False
        logger.info("🔊 Sound alerts: OFF")

    # ── is_enabled property ───────────────────────────────────────────────────
    # Fix 1 – Expose is_enabled as a property so alert_manager.py references
    # to self._sound.is_enabled resolve correctly instead of AttributeError.
    @property
    def is_enabled(self) -> bool:
        return self.enabled and SOUND_AVAILABLE

    # ── Quiet hours check ─────────────────────────────────────────────────────
    def _is_quiet_hours(self) -> bool:
        """
        Returns True if the current time falls within the configured
        quiet hours window. Mirrors the logic in TelegramAlerts so
        both notification channels are suppressed consistently.
        """
        try:
            if not CONFIG.TELEGRAM_QUIET_ON:
                return False
            hour  = datetime.now(self.timezone).hour
            start = CONFIG.TELEGRAM_QUIET_START
            end   = CONFIG.TELEGRAM_QUIET_END
            if start > end:
                return hour >= start or hour < end
            return start <= hour < end
        except Exception:
            return False

    # ── Guard helper ──────────────────────────────────────────────────────────
    def _guard(self, bypass_quiet: bool = False) -> bool:
        """
        Returns True if sound can play.

        Parameters
        ----------
        bypass_quiet : If True, quiet hours are ignored.
                       Used for danger_exit() which is always urgent.
        """
        if not self.enabled or not SOUND_AVAILABLE:
            return False
        if not bypass_quiet and self._is_quiet_hours():
            logger.debug("Sound suppressed — quiet hours active")
            return False
        return True

    # ── Sequence player ───────────────────────────────────────────────────────
    def _play_sequence(
        self,
        tones:        list,
        bypass_quiet: bool = False,
    ) -> None:
        """
        Play a sequence of (frequency_hz, duration_ms) tuples in a
        background daemon thread. Never blocks the trading loop.

        Parameters
        ----------
        tones        : list of (freq: int, duration: int) tuples
        bypass_quiet : passed through to _guard() for danger alerts
        """
        if not self._guard(bypass_quiet=bypass_quiet):
            return

        def _beep_sequence():
            for freq, duration in tones:
                try:
                    winsound.Beep(freq, duration)
                except Exception as e:
                    logger.debug(f"Sound error: {e}")
                    break

        threading.Thread(target=_beep_sequence, daemon=True).start()

    # ── Alert types ───────────────────────────────────────────────────────────
    def buy_signal(self) -> None:
        """Ascending tone — opportunity detected."""
        self._play_sequence([(400, 150), (600, 150), (800, 300)])
        logger.debug("🔊 BUY sound played")

    def sell_signal(self) -> None:
        """Descending tone — opportunity detected."""
        self._play_sequence([(800, 150), (600, 150), (400, 300)])
        logger.debug("🔊 SELL sound played")

    def hold_alert(self) -> None:
        """Single steady tone — stay in trade."""
        self._play_sequence([(600, 200)])
        logger.debug("🔊 HOLD sound played")

    def potential_exit(self) -> None:
        """Double tone — consider taking profit."""
        self._play_sequence([(700, 200), (700, 200)])
        logger.debug("🔊 POTENTIAL EXIT sound played")

    def danger_exit(self) -> None:
        """
        Urgent alarm — get out now.
        Bypasses quiet hours — danger is always audible.
        """
        self._play_sequence(
            [(1000, 150), (500, 150)] * 5,
            bypass_quiet=True,
        )
        logger.debug("🔊 DANGER sound played")

    def news_warning(self) -> None:
        """Warning beeps — news event coming."""
        self._play_sequence([(800, 100), (800, 100), (800, 100)])
        logger.debug("🔊 NEWS WARNING sound played")

    def trade_opened(self) -> None:
        """Confirmation tone — trade placed."""
        self._play_sequence([(500, 100), (1000, 300)])
        logger.debug("🔊 TRADE OPENED sound played")

    def trade_closed_profit(self) -> None:
        """Happy tone — trade closed in profit."""
        self._play_sequence(
            [(600, 100), (800, 100), (1000, 100), (1200, 300)]
        )
        logger.debug("🔊 PROFIT sound played")

    def trade_closed_loss(self) -> None:
        """Sad tone — trade closed at loss."""
        self._play_sequence([(600, 200), (400, 400)])
        logger.debug("🔊 LOSS sound played")

    def system_ready(self) -> None:
        """Startup confirmation — system online."""
        self._play_sequence(
            [(400, 100), (600, 100), (800, 100), (1000, 100), (1200, 300)]
        )
        logger.debug("🔊 SYSTEM READY sound played")

    def system_error(self) -> None:
        """Error tone — something went wrong."""
        self._play_sequence([(300, 500), (300, 500)])
        logger.debug("🔊 ERROR sound played")

    def toggle(self) -> bool:
        """
        Toggle sounds on/off.
        Logs a platform note when winsound is unavailable so the
        operator knows sounds will not play even if toggled ON.
        """
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        if self.enabled and not SOUND_AVAILABLE:
            logger.info(
                "🔊 Sound alerts: ON (unavailable — winsound not found on "
                "this platform; sounds will not play)"
            )
        else:
            logger.info(f"🔊 Sound alerts: {status}")
        return self.enabled
