# research/sentiment_analyzer.py
import feedparser
import re
import time
from datetime import datetime
from monitoring.logger import get_logger

logger = get_logger("SentimentAnalyzer")

# ── RSS News Sources ──────────────────────────────────────────
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

# ── Currency-Specific Keywords ────────────────────────────────
# Each symbol has UNIQUE keywords so articles don't bleed across pairs
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


class SentimentAnalyzer:
    """
    Hybrid sentiment engine: Gemini (EURUSD only) → FinBERT → VADER.
    BUG FIX: Each symbol now fetches articles using its OWN keywords only,
    preventing identical scores across all pairs.
    """

    def __init__(self):
        self._gemini       = None
        self._finbert_pipe = None
        self._vader        = None
        self._cache        = {}
        self._cache_time   = {}
        self._load_gemini()
        self._load_finbert()

    # ── Engine Loading ────────────────────────────────────────

    def _load_gemini(self):                                        # ← 4-space indent — inside class
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

    def _load_finbert(self):
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

    def _load_vader(self):
        if self._vader:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"VADER not available: {e}")

    # ── Article Fetching ──────────────────────────────────────

    def _fetch_articles(self, symbol: str) -> list:
        """
        Fetch up to 20 articles for a symbol using ONLY that symbol's
        keywords. This prevents all pairs returning the same USD/Fed articles.
        """
        kw = CURRENCY_KEYWORDS.get(symbol, [])
        if not kw:
            logger.warning(f"No keywords defined for {symbol}")
            return []

        articles = []

        for src, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:25]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", "")
                    txt     = f"{title} {summary}".lower()

                    # Only match this symbol's own keywords
                    if any(k.lower() in txt for k in kw):
                        articles.append({
                            "title":   title,
                            "summary": summary[:300],
                            "source":  src,
                        })

                    if len(articles) >= 20:
                        break

            except Exception:
                continue  # silently skip dead feeds

            if len(articles) >= 20:
                break

        logger.debug(f"📰 {symbol}: {len(articles)} articles found across RSS feeds")
        return articles[:20]

    # ── Scoring Engines ───────────────────────────────────────

    def _score_gemini(self, articles: list, symbol: str) -> dict | None:
        if not self._gemini or not articles:
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
            match = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if match:
                import json
                data = json.loads(match.group())
                return {
                    "score":      float(data.get("score", 0)),
                    "label":      data.get("label", "Neutral"),
                    "confidence": float(data.get("confidence", 0.7)),
                    "reasoning":  data.get("reasoning", ""),
                    "engine":     "Gemini",
                    "articles":   len(articles),
                }
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                logger.warning(
                    "⚠️ Gemini quota hit — disabling for this session, switching to FinBERT"
                )
                self._gemini = None  # stop retrying this session
            else:
                logger.warning(f"Gemini scoring failed: {e}")
        return None

    def _score_finbert(self, articles: list) -> dict | None:
        if not self._finbert_pipe or not articles:
            return None
        try:
            scores = []
            for art in articles[:15]:
                result = self._finbert_pipe(art["title"][:512])[0]
                best   = max(result, key=lambda x: x["score"])
                val    = {"positive": 1, "negative": -1, "neutral": 0}.get(
                    best["label"].lower(), 0
                ) * best["score"]
                scores.append(val)

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
        self._load_vader()
        if not self._vader or not articles:
            return {
                "score": 0, "label": "Neutral",
                "confidence": 0, "engine": "None", "articles": 0,
            }
        try:
            comps = [
                self._vader.polarity_scores(a["title"])["compound"]
                for a in articles
            ]
            avg   = sum(comps) / len(comps) if comps else 0
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
            return {
                "score": 0, "label": "Neutral",
                "confidence": 0, "engine": "None", "articles": 0,
            }

    # ── Main Public Method ────────────────────────────────────

    def get_symbol_sentiment(self, symbol: str) -> dict:
        """Return sentiment dict for a symbol. Cached for 30 min."""
        now = datetime.now()
        if (
            symbol in self._cache
            and symbol in self._cache_time
            and (now - self._cache_time[symbol]).seconds < 1800
        ):
            return self._cache[symbol]

        articles = self._fetch_articles(symbol)

        # Gemini only for EURUSD (preserves free-tier quota)
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

        logger.info(
            f"📰 {symbol} Sentiment [{result.get('engine')}]: "
            f"{result.get('label')} ({result.get('score'):+.3f}) | "
            f"{result.get('articles')} articles"
        )

        self._cache[symbol]      = result
        self._cache_time[symbol] = now
        return result

    # ── Print Table ───────────────────────────────────────────

    def print_sentiment_table(self):
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
        print(f"\n📰 MARKET SENTIMENT\n{'='*65}")
        print(f"  {engine_label}\n{'='*65}")
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
