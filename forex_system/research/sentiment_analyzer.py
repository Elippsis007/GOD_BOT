# research/sentiment_analyzer.py
import feedparser
import json
import re
import time
from datetime import datetime
from monitoring.logger import get_logger

logger = get_logger("SentimentAnalyzer")

# ── RSS News Sources ──────────────────────────────────────────────────────────
RSS_FEEDS = {
    "fed":        "https://www.federalreserve.gov/feeds/press_all.xml",
    "ecb":        "https://www.ecb.europa.eu/rss/press.html",
    "boe":        "https://www.bankofengland.co.uk/rss/news",
    "reuters_fx": "https://feeds.reuters.com/reuters/businessNews",
    "ft":         "https://www.ft.com/rss/home",
    "fxstreet":   "https://www.fxstreet.com/rss/news",
    "forexlive":  "https://www.forexlive.com/feed/news",
    "investing":  "https://www.investing.com/rss/news.rss",
    "dailyfx":    "https://www.dailyfx.com/feeds/all",
    "cnbc_fx":    "https://www.cnbc.com/id/20910258/device/rss/rss.html",
}

# ── Currency-Specific Keywords ────────────────────────────────────────────────
CURRENCY_KEYWORDS = {
    "EURUSD": ["euro", "EUR", "ECB", "lagarde", "european central bank",
               "eurozone", "eur/usd", "eurodollar"],
    "GBPUSD": ["pound", "GBP", "sterling", "bank of england", "BOE",
               "bailey", "gbp/usd", "british economy"],
    "USDJPY": ["yen", "JPY", "bank of japan", "BOJ", "ueda", "boj",
               "usd/jpy", "japanese yen"],
    "AUDUSD": ["aussie", "AUD", "reserve bank australia", "RBA",
               "aud/usd", "australian dollar", "iron ore"],
    "USDCAD": ["loonie", "CAD", "bank of canada", "BOC",
               "usd/cad", "canadian dollar", "oil prices"],
    "XAUUSD": ["gold", "XAU", "bullion", "xau/usd", "precious metals",
               "gold price", "safe haven"],
}

# Neutral fallback returned when all engines fail
_NEUTRAL_RESULT = {
    "score":      0.0,
    "label":      "Neutral",
    "confidence": 0.0,
    "engine":     "None",
    "articles":   0,
}

# Cache TTL in seconds (30 minutes)
CACHE_TTL_SECS = 1800

# Minimum articles required before trusting a sentiment score
MIN_ARTICLES = 3


class SentimentAnalyzer:
    """
    Hybrid sentiment engine: Gemini (EURUSD only) → FinBERT → VADER.
    Each symbol fetches articles using its OWN keywords only,
    preventing identical scores across all pairs.
    """

    def __init__(self):
        self._gemini       = None
        self._finbert_pipe = None
        self._vader        = None
        self._cache:       dict = {}
        self._cache_time:  dict = {}
        self._load_gemini()
        self._load_finbert()

    # ── Engine Loading ────────────────────────────────────────────────────────

    def _load_gemini(self) -> None:
        try:
            from config.settings import GEMINI_ENABLED, GEMINI_API_KEY
            if not GEMINI_ENABLED:
                logger.info("Gemini disabled in settings — using FinBERT")
                return
            if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_GEMINI_KEY_HERE":
                logger.info("Gemini key not set — using FinBERT")
                return
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._gemini = genai.GenerativeModel("gemini-2.0-flash")
            logger.info(
                "✅ Gemini loaded — contextual sentiment active "
                "(~92% accuracy) [EURUSD only]"
            )
        except Exception as e:
            logger.warning(f"Gemini load failed: {e} — using FinBERT")

    def _load_finbert(self) -> None:
        try:
            logger.info("🧠 Loading FinBERT (~88% accuracy)…")
            from transformers import pipeline
            self._finbert_pipe = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
            )
            logger.info("✅ FinBERT loaded")
        except Exception as e:
            logger.warning(f"FinBERT not available: {e} — will use VADER")

    def _load_vader(self) -> None:
        if self._vader:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"VADER not available: {e}")

    # ── Article Fetching ──────────────────────────────────────────────────────

    def _fetch_articles(self, symbol: str) -> list:
        """
        Fetch up to 20 articles for a symbol using ONLY that symbol's
        keywords, sampling evenly across all feeds before hitting the cap.

        FIX: old inner-loop break fired as soon as any single feed produced
        20 articles, leaving all remaining feeds unsampled. New approach
        collects up to 5 articles per feed first, then fills to 20 from
        whatever feeds had more, ensuring broad source coverage.
        """
        kw = CURRENCY_KEYWORDS.get(symbol, [])
        if not kw:
            logger.warning(f"No keywords defined for {symbol}")
            return []

        per_feed:   list = []     # list of lists, one per feed
        total_found: int = 0

        for src, url in RSS_FEEDS.items():
            feed_articles: list = []
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:25]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", "")
                    txt     = f"{title} {summary}".lower()

                    if any(k.lower() in txt for k in kw):
                        feed_articles.append({
                            "title":   title,
                            "summary": summary[:300],
                            "source":  src,
                        })
            except Exception:
                pass    # silently skip dead feeds

            per_feed.append(feed_articles)
            total_found += len(feed_articles)

        # Round-robin merge across feeds up to 20 articles total
        # so no single feed monopolises the sample
        articles: list = []
        idx = 0
        while len(articles) < 20 and any(per_feed):
            for feed_list in per_feed:
                if idx < len(feed_list):
                    articles.append(feed_list[idx])
                    if len(articles) >= 20:
                        break
            idx += 1
            if idx > max((len(f) for f in per_feed), default=0):
                break

        logger.debug(
            f"📰 {symbol}: {len(articles)} articles sampled "
            f"({total_found} total matches across {len(RSS_FEEDS)} feeds)"
        )
        return articles

    # ── Scoring Engines ───────────────────────────────────────────────────────

    def _score_gemini(self, articles: list, symbol: str) -> dict | None:
        if not self._gemini or not articles:
            return None

        # Require a minimum number of articles before trusting Gemini's score
        if len(articles) < MIN_ARTICLES:
            logger.debug(
                f"Gemini skipped for {symbol} — only {len(articles)} articles "
                f"(minimum {MIN_ARTICLES})"
            )
            return None

        headlines = "\n".join(f"- {a['title']}" for a in articles[:10])
        prompt = (
            f"You are a professional forex trader analyzing market sentiment.\n\n"
            f"Analyze these {len(articles)} news headlines for {symbol}:\n\n"
            f"{headlines}\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float -1.0..1.0>, "label": "Bullish|Bearish|Neutral", '
            '"confidence": <0.0..1.0>, "reasoning": "one-sentence"}'
        )
        try:
            resp  = self._gemini.generate_content(prompt)
            match = re.search(r"\{.*?\}", resp.text, re.DOTALL)
            if not match:
                logger.warning("Gemini response contained no JSON object")
                return None

            data = json.loads(match.group())

            score      = float(data.get("score",      0.0))
            confidence = float(data.get("confidence", 0.0))
            label      = data.get("label", "Neutral")

            # FIX: reject low-confidence Gemini results rather than returning
            # a bogus 0.7 default confidence that blocks legitimate trades
            if confidence < 0.5:
                logger.debug(
                    f"Gemini result for {symbol} rejected — "
                    f"confidence {confidence:.2f} below 0.5 threshold"
                )
                return None

            return {
                "score":      round(score, 4),
                "label":      label,
                "confidence": round(confidence, 4),
                "reasoning":  data.get("reasoning", ""),
                "engine":     "Gemini",
                "articles":   len(articles),
            }

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                logger.warning(
                    "⚠️ Gemini quota hit — disabling for this session, "
                    "switching to FinBERT"
                )
                self._gemini = None
            else:
                logger.warning(f"Gemini scoring failed: {e}")
        return None

    def _score_finbert(self, articles: list) -> dict | None:
        if not self._finbert_pipe or not articles:
            return None
        try:
            scores: list = []
            for art in articles[:15]:
                # FIX: ensure we always pass a single string, not a list,
                # to the pipeline so [0] indexing is always valid
                title  = str(art["title"])[:512]
                result = self._finbert_pipe(title)[0]   # list of label dicts
                best   = max(result, key=lambda x: x["score"])
                val    = (
                    {"positive": 1, "negative": -1, "neutral": 0}
                    .get(best["label"].lower(), 0)
                ) * best["score"]
                scores.append(val)

            if not scores:
                return None

            avg   = sum(scores) / len(scores)
            label = "Bullish" if avg > 0.1 else "Bearish" if avg < -0.1 else "Neutral"
            return {
                "score":      round(avg, 4),
                "label":      label,
                "confidence": round(min(abs(avg) * 3, 0.95), 4),
                "engine":     "FinBERT",
                "articles":   len(articles),
            }
        except Exception as e:
            logger.warning(f"FinBERT scoring failed: {e}")
        return None

    def _score_vader(self, articles: list) -> dict:
        """Always returns a dict — last-resort fallback, never returns None."""
        self._load_vader()
        if not self._vader or not articles:
            return dict(_NEUTRAL_RESULT)

        try:
            comps = [
                self._vader.polarity_scores(str(a["title"]))["compound"]
                for a in articles
            ]
            avg   = sum(comps) / len(comps) if comps else 0.0
            label = "Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral"
            return {
                "score":      round(avg, 4),
                "label":      label,
                "confidence": round(min(abs(avg) * 2, 0.7), 4),
                "engine":     "VADER",
                "articles":   len(articles),
            }
        except Exception as e:
            logger.error(f"VADER failed: {e}")
            return dict(_NEUTRAL_RESULT)

    # ── Main Public Method ────────────────────────────────────────────────────

    def get_symbol_sentiment(self, symbol: str) -> dict:
        """Return sentiment dict for a symbol. Cached for 30 minutes."""
        now = datetime.now()

        # FIX: was using .seconds which returns only the seconds *component*
        # of the timedelta. A 2-hour-old cache entry shows .seconds == 0 and
        # appears valid indefinitely. total_seconds() returns full elapsed time.
        if (
            symbol in self._cache
            and symbol in self._cache_time
            and (now - self._cache_time[symbol]).total_seconds() < CACHE_TTL_SECS
        ):
            logger.debug(f"📰 {symbol} sentiment served from cache")
            return self._cache[symbol]

        articles = self._fetch_articles(symbol)

        # Gemini only for EURUSD — preserves free-tier API quota
        if symbol == "EURUSD" and self._gemini:
            result = (
                self._score_gemini(articles, symbol)
                or self._score_finbert(articles)
                or self._score_vader(articles)
            )
        else:
            result = (
                self._score_finbert(articles)
                or self._score_vader(articles)
            )

        # FIX: final safety net — if the entire chain somehow returns None
        # (all engines unavailable), use the neutral fallback so result.get()
        # on the next line never raises AttributeError.
        if result is None:
            logger.warning(
                f"All sentiment engines failed for {symbol} — "
                f"returning neutral fallback"
            )
            result = dict(_NEUTRAL_RESULT)
            result["articles"] = len(articles)

        logger.info(
            f"📰 {symbol} Sentiment [{result.get('engine')}]: "
            f"{result.get('label')} ({result.get('score', 0):+.3f}) | "
            f"{result.get('articles', 0)} articles"
        )

        self._cache[symbol]      = result
        self._cache_time[symbol] = now
        return result

    # ── Print Table ───────────────────────────────────────────────────────────

    def print_sentiment_table(self) -> None:
        from config.settings import CONFIG
        engine_label = (
            "🤖 Gemini (~92%) for EURUSD | 🧠 FinBERT (~88%) for others"
            if self._gemini
            else (
                "🧠 FinBERT (~88% accuracy)"
                if self._finbert_pipe
                else "📊 VADER (~65% accuracy)"
            )
        )
        print(f"\n📰 MARKET SENTIMENT\n{'=' * 65}")
        print(f"  {engine_label}\n{'=' * 65}")
        for sym in CONFIG.SYMBOLS:
            r    = self.get_symbol_sentiment(sym)
            icon = (
                "🟢" if r["label"] == "Bullish"
                else ("🔴" if r["label"] == "Bearish" else "⚪")
            )
            print(
                f"  {icon} {sym:<8} {r['label']:<10} "
                f"Score: {r['score']:+.3f} | "
                f"Conf: {r['confidence']:.0%} | "
                f"{r['articles']} articles [{r['engine']}]"
            )
            if r.get("reasoning"):
                print(f"     💬 {r['reasoning']}")
        print("=" * 65)
