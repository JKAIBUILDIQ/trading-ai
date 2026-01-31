#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NIGHTLY RESEARCH BOT
═══════════════════════════════════════════════════════════════════════════════

Runs every night at 10 PM EST to:
1. Analyze the day's price action
2. Track agent recommendations vs actual outcomes
3. Fetch news/article sentiment
4. Update market context for tomorrow
5. Generate daily intelligence brief
6. Feed insights to knowledge base

Cron: 0 22 * * * cd /home/jbot/trading_ai && python3 cron/nightly_research.py
"""

import asyncio
import httpx
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("NightlyResearch")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_API = "http://localhost:8890/knowledge"
NEWS_SOURCES = {
    "gold": [
        "gold price forecast",
        "XAUUSD analysis today",
        "gold technical analysis",
        "gold news today"
    ],
    "macro": [
        "Federal Reserve news",
        "DXY dollar index forecast",
        "US treasury yields",
        "inflation expectations"
    ],
    "crypto_miners": [
        "IREN stock news",
        "CLSK Cleanspark news",
        "CIFR Cipher Mining news",
        "Bitcoin mining stocks"
    ]
}

REPORTS_DIR = Path("/home/jbot/trading_ai/neo/reports")
REPORTS_DIR.mkdir(exist_ok=True)

DAILY_DATA_DIR = Path("/home/jbot/trading_ai/neo/daily_data")
DAILY_DATA_DIR.mkdir(exist_ok=True)

# API Keys
SERP_API_KEY = os.getenv("SERP_API_KEY", "")  # For news search
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_news_sentiment(query: str, num_results: int = 5) -> List[Dict]:
    """Fetch news articles for a query using web search."""
    articles = []
    
    # Try SerpAPI if available
    if SERP_API_KEY:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "tbm": "nws",  # News search
                        "api_key": SERP_API_KEY,
                        "num": num_results
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("news_results", [])[:num_results]:
                        articles.append({
                            "title": item.get("title", ""),
                            "source": item.get("source", ""),
                            "snippet": item.get("snippet", ""),
                            "date": item.get("date", ""),
                            "link": item.get("link", "")
                        })
        except Exception as e:
            logger.warning(f"SerpAPI failed: {e}")
    
    # Fallback: use DuckDuckGo news (no API key needed)
    if not articles:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://duckduckgo.com/",
                    params={"q": f"{query} site:reuters.com OR site:bloomberg.com OR site:cnbc.com", "format": "json"},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15.0
                )
                # DuckDuckGo doesn't have a proper API, so this is limited
                # In production, you'd use a proper news API
        except:
            pass
    
    return articles


async def gather_all_news() -> Dict[str, List[Dict]]:
    """Gather news from all categories."""
    all_news = {}
    
    for category, queries in NEWS_SOURCES.items():
        category_news = []
        for query in queries:
            articles = await fetch_news_sentiment(query, num_results=3)
            category_news.extend(articles)
        all_news[category] = category_news[:10]  # Limit per category
        logger.info(f"Gathered {len(category_news)} articles for {category}")
    
    return all_news


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY ACTION TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

async def get_todays_analyses() -> List[Dict]:
    """Get all agent analyses from today."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KNOWLEDGE_API}/recent_analyses",
                params={"hours": 24},
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("analyses", [])
    except Exception as e:
        logger.error(f"Failed to get analyses: {e}")
    return []


async def get_todays_journal() -> List[Dict]:
    """Get today's trading journal entries."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KNOWLEDGE_API}/journal",
                params={"days": 1},
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("entries", [])
    except Exception as e:
        logger.error(f"Failed to get journal: {e}")
    return []


async def get_agent_accuracies() -> Dict[str, Dict]:
    """Get accuracy stats for all agents."""
    agents = ["ghost", "casper", "neo", "fomo", "chart", "sequence"]
    accuracies = {}
    
    async with httpx.AsyncClient() as client:
        for agent in agents:
            try:
                response = await client.get(
                    f"{KNOWLEDGE_API}/agent_accuracy",
                    params={"agent": agent, "days": 7},
                    timeout=10.0
                )
                if response.status_code == 200:
                    accuracies[agent] = response.json()
            except:
                pass
    
    return accuracies


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════════

async def get_market_prices() -> Dict[str, float]:
    """Get current market prices from various sources."""
    prices = {}
    
    # Try to get from existing APIs
    try:
        async with httpx.AsyncClient() as client:
            # Gold from gold-forex-api
            response = await client.get("http://localhost:8037/api/gold/price", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                prices["xauusd"] = data.get("price", 0)
    except:
        pass
    
    # Default values if APIs fail
    prices.setdefault("xauusd", 0)
    prices.setdefault("dxy", 0)
    prices.setdefault("usdjpy", 0)
    prices.setdefault("btc", 0)
    prices.setdefault("vix", 0)
    
    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

async def analyze_sentiment_with_ai(news: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Use AI to analyze overall sentiment from news."""
    
    if not ANTHROPIC_API_KEY:
        return {"overall": "neutral", "confidence": 50, "note": "No AI analysis available"}
    
    # Build news summary for AI
    news_text = ""
    for category, articles in news.items():
        news_text += f"\n## {category.upper()} NEWS:\n"
        for art in articles[:5]:
            news_text += f"- {art.get('title', 'No title')}\n"
    
    if not news_text.strip():
        return {"overall": "neutral", "confidence": 50, "note": "No news to analyze"}
    
    prompt = f"""Analyze this financial news and determine the overall sentiment for gold/XAUUSD trading.

{news_text}

Respond in JSON format:
{{
    "gold_sentiment": "bullish" | "bearish" | "neutral",
    "gold_confidence": 0-100,
    "gold_reasoning": "brief explanation",
    "dxy_outlook": "up" | "down" | "neutral",
    "risk_events": ["list of upcoming risk events mentioned"],
    "key_themes": ["list of main themes"],
    "trading_bias": "buy_dips" | "sell_rallies" | "wait" | "range_trade"
}}"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data["content"][0]["text"]
                # Parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
    except Exception as e:
        logger.error(f"AI sentiment analysis failed: {e}")
    
    return {"overall": "neutral", "confidence": 50, "note": "Analysis failed"}


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE UPDATES
# ═══════════════════════════════════════════════════════════════════════════════

async def update_market_context(prices: Dict, sentiment: Dict, news: Dict):
    """Update market context in knowledge base."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Determine DEFCON based on sentiment and market conditions
    defcon = 3  # Default
    if sentiment.get("gold_sentiment") == "bearish" and sentiment.get("gold_confidence", 0) > 70:
        defcon = 2
    elif sentiment.get("trading_bias") == "wait":
        defcon = 3
    elif sentiment.get("gold_sentiment") == "bullish" and sentiment.get("gold_confidence", 0) > 70:
        defcon = 4
    
    context = {
        "date": today,
        "xauusd": prices.get("xauusd", 0),
        "dxy": prices.get("dxy", 0),
        "usdjpy": prices.get("usdjpy", 0),
        "btc": prices.get("btc", 0),
        "vix": prices.get("vix", 0),
        "defcon": defcon,
        "bias": sentiment.get("gold_sentiment", "neutral"),
        "key_events": sentiment.get("risk_events", []),
        "notes": f"Auto-generated. AI confidence: {sentiment.get('gold_confidence', 'N/A')}%"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{KNOWLEDGE_API}/save_market_context",
                json=context,
                timeout=30.0
            )
            logger.info(f"Updated market context for {today}")
    except Exception as e:
        logger.error(f"Failed to update market context: {e}")


async def feed_news_insights(sentiment: Dict):
    """Feed news-derived insights to knowledge base."""
    
    insights = []
    
    # Gold sentiment insight
    if sentiment.get("gold_sentiment") and sentiment.get("gold_reasoning"):
        insights.append({
            "analysis_type": "macro",
            "content": f"News sentiment: {sentiment['gold_sentiment'].upper()}. {sentiment['gold_reasoning']}",
            "symbol": "XAUUSD",
            "importance": 7
        })
    
    # DXY outlook
    if sentiment.get("dxy_outlook"):
        direction = sentiment["dxy_outlook"]
        gold_impact = "bearish" if direction == "up" else "bullish" if direction == "down" else "neutral"
        insights.append({
            "analysis_type": "correlation",
            "content": f"DXY expected to move {direction}. Gold impact: {gold_impact}.",
            "symbol": "XAUUSD",
            "related_assets": ["DXY"],
            "importance": 7
        })
    
    # Key themes
    for theme in sentiment.get("key_themes", [])[:3]:
        insights.append({
            "analysis_type": "macro",
            "content": f"Market theme: {theme}",
            "symbol": "XAUUSD",
            "importance": 5
        })
    
    # Feed to knowledge base
    async with httpx.AsyncClient() as client:
        for insight in insights:
            try:
                await client.post(
                    f"{KNOWLEDGE_API}/feed/neo",
                    json=insight,
                    timeout=10.0
                )
            except:
                pass
    
    logger.info(f"Fed {len(insights)} insights to knowledge base")


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY BRIEF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_daily_brief(
    analyses: List[Dict],
    journal: List[Dict],
    accuracies: Dict,
    news: Dict,
    sentiment: Dict,
    prices: Dict
) -> str:
    """Generate a daily intelligence brief."""
    
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    brief = f"""# DAILY INTELLIGENCE BRIEF
## {today.strftime('%A, %B %d, %Y')}

---

## MARKET SNAPSHOT

| Asset | Price | Status |
|-------|-------|--------|
| XAUUSD | ${prices.get('xauusd', 'N/A')} | {sentiment.get('gold_sentiment', 'N/A').upper()} |
| DXY | {prices.get('dxy', 'N/A')} | {sentiment.get('dxy_outlook', 'N/A').upper()} |
| BTC | ${prices.get('btc', 'N/A'):,.0f} | - |

---

## AI SENTIMENT ANALYSIS

**Gold Outlook:** {sentiment.get('gold_sentiment', 'N/A').upper()}
**Confidence:** {sentiment.get('gold_confidence', 'N/A')}%
**Trading Bias:** {sentiment.get('trading_bias', 'N/A').upper()}

**Reasoning:** {sentiment.get('gold_reasoning', 'No analysis available')}

---

## TODAY'S AGENT ACTIVITY

**Analyses Made:** {len(analyses)}
**Journal Entries:** {len(journal)}

### Agent Accuracy (7 Days)
"""

    for agent, acc in accuracies.items():
        if acc.get('total', 0) > 0:
            brief += f"- **{agent.upper()}**: {acc.get('accuracy_pct', 0)}% ({acc.get('total', 0)} calls)\n"
        else:
            brief += f"- **{agent.upper()}**: No data yet\n"

    brief += f"""
---

## NEWS HIGHLIGHTS

### Gold News
"""
    for art in news.get('gold', [])[:5]:
        brief += f"- {art.get('title', 'No title')} ({art.get('source', 'Unknown')})\n"

    brief += f"""
### Macro News
"""
    for art in news.get('macro', [])[:5]:
        brief += f"- {art.get('title', 'No title')} ({art.get('source', 'Unknown')})\n"

    if news.get('crypto_miners'):
        brief += f"""
### Crypto Miners
"""
        for art in news.get('crypto_miners', [])[:3]:
            brief += f"- {art.get('title', 'No title')} ({art.get('source', 'Unknown')})\n"

    brief += f"""
---

## KEY EVENTS TO WATCH

"""
    for event in sentiment.get('risk_events', ['No events identified']):
        brief += f"- {event}\n"

    brief += f"""
---

## TOMORROW'S FOCUS

**Date:** {tomorrow.strftime('%A, %B %d, %Y')}
**Recommended DEFCON:** {3 if sentiment.get('trading_bias') == 'wait' else 4}
**Primary Strategy:** {sentiment.get('trading_bias', 'wait').replace('_', ' ').title()}

### Key Levels to Watch
- Support: TBD based on tonight's close
- Resistance: TBD based on tonight's close

---

*Generated: {today.strftime('%Y-%m-%d %H:%M:%S')} EST*
*Source: Nightly Research Bot*
"""

    return brief


async def save_daily_brief(brief: str):
    """Save daily brief to file and knowledge base."""
    today = datetime.now()
    
    # Save to file
    filename = REPORTS_DIR / f"DAILY_BRIEF_{today.strftime('%Y%m%d')}.md"
    with open(filename, 'w') as f:
        f.write(brief)
    logger.info(f"Saved daily brief to {filename}")
    
    # Also save to daily_data as JSON
    data_file = DAILY_DATA_DIR / f"brief_{today.strftime('%Y%m%d')}.json"
    with open(data_file, 'w') as f:
        json.dump({
            "date": today.strftime('%Y-%m-%d'),
            "brief": brief,
            "generated_at": today.isoformat()
        }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def run_nightly_research():
    """Main function to run nightly research."""
    logger.info("=" * 60)
    logger.info("STARTING NIGHTLY RESEARCH")
    logger.info("=" * 60)
    
    # 1. Gather news
    logger.info("Gathering news...")
    news = await gather_all_news()
    
    # 2. Get market prices
    logger.info("Fetching market prices...")
    prices = await get_market_prices()
    
    # 3. Analyze sentiment with AI
    logger.info("Analyzing sentiment...")
    sentiment = await analyze_sentiment_with_ai(news)
    
    # 4. Get today's analyses and journal
    logger.info("Getting today's activity...")
    analyses = await get_todays_analyses()
    journal = await get_todays_journal()
    
    # 5. Get agent accuracies
    logger.info("Calculating agent accuracies...")
    accuracies = await get_agent_accuracies()
    
    # 6. Update knowledge base
    logger.info("Updating knowledge base...")
    await update_market_context(prices, sentiment, news)
    await feed_news_insights(sentiment)
    
    # 7. Generate and save daily brief
    logger.info("Generating daily brief...")
    brief = await generate_daily_brief(
        analyses, journal, accuracies, news, sentiment, prices
    )
    await save_daily_brief(brief)
    
    logger.info("=" * 60)
    logger.info("NIGHTLY RESEARCH COMPLETE")
    logger.info("=" * 60)
    
    return {
        "status": "complete",
        "news_articles": sum(len(v) for v in news.values()),
        "sentiment": sentiment.get("gold_sentiment"),
        "analyses_today": len(analyses),
        "journal_entries": len(journal)
    }


if __name__ == "__main__":
    result = asyncio.run(run_nightly_research())
    print(json.dumps(result, indent=2))
