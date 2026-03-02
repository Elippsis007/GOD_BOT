# notifications/sound_alerts.py
import threading
from monitoring.logger import get_logger

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

logger = get_logger("SoundAlerts")


class SoundAlerts:
    """
    Windows sound alerts for trading signals.
    Different tones for different alert types.
    Runs in background thread — never blocks trading.
    """

    def __init__(self):
        self.enabled = SOUND_AVAILABLE  # auto-disable if no winsound

    # ── Guard helper ──────────────────────────────────────
    def _guard(self) -> bool:
        """Returns True if sound can play."""
        return self.enabled and SOUND_AVAILABLE

    # ── Play sound in background ──────────────────────────
    def _play(self, freq: int, duration: int, times: int = 1):
        if not self._guard():
            return
        def _beep():
            for _ in range(times):
                try:
                    winsound.Beep(freq, duration)
                except Exception as e:
                    logger.debug(f"Sound error: {e}")
        threading.Thread(target=_beep, daemon=True).start()

    # ── Alert Types ───────────────────────────────────────
    def buy_signal(self):
        """Ascending tone — opportunity detected."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(400, 150)
            winsound.Beep(600, 150)
            winsound.Beep(800, 300)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 BUY sound played")

    def sell_signal(self):
        """Descending tone — opportunity detected."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(800, 150)
            winsound.Beep(600, 150)
            winsound.Beep(400, 300)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 SELL sound played")

    def hold_alert(self):
        """Single steady tone — stay in trade."""
        self._play(600, 200, 1)
        logger.debug("🔊 HOLD sound played")

    def potential_exit(self):
        """Double tone — consider taking profit."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(700, 200)
            winsound.Beep(700, 200)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 POTENTIAL EXIT sound played")

    def danger_exit(self):
        """Urgent alarm — get out now."""
        if not self._guard():
            return
        def _play():
            for _ in range(5):
                winsound.Beep(1000, 150)
                winsound.Beep(500,  150)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 DANGER sound played")

    def news_warning(self):
        """Warning beeps — news event coming."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(800, 100)
            winsound.Beep(800, 100)
            winsound.Beep(800, 100)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 NEWS WARNING sound played")

    def trade_opened(self):
        """Confirmation tone — trade placed."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(500,  100)
            winsound.Beep(1000, 300)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 TRADE OPENED sound played")

    def trade_closed_profit(self):
        """Happy tone — trade closed in profit."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(600,  100)
            winsound.Beep(800,  100)
            winsound.Beep(1000, 100)
            winsound.Beep(1200, 300)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 PROFIT sound played")

    def trade_closed_loss(self):
        """Sad tone — trade closed at loss."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(600, 200)
            winsound.Beep(400, 400)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 LOSS sound played")

    def system_ready(self):
        """Startup confirmation — system online."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(400,  100)
            winsound.Beep(600,  100)
            winsound.Beep(800,  100)
            winsound.Beep(1000, 100)
            winsound.Beep(1200, 300)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 SYSTEM READY sound played")

    def system_error(self):
        """Error tone — something went wrong."""
        if not self._guard():
            return
        def _play():
            winsound.Beep(300, 500)
            winsound.Beep(300, 500)
        threading.Thread(target=_play, daemon=True).start()
        logger.debug("🔊 ERROR sound played")

    def toggle(self):
        """Toggle sounds on/off."""
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        logger.info(f"🔊 Sound alerts: {status}")
        return self.enabled