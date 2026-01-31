"""
Trading Agents API - Multi-Agent Analysis System
Runs parallel analysis from specialized trading agents with Commander synthesis.

Agents:
- GHOST: Entry/exit timing specialist
- CASPER: Hedge and risk management
- NEO: Macro analysis and correlations
- FOMO: Exhaustion and sentiment detection
- CHART: Pattern recognition
- SEQUENCE: Candle sequence analysis

Features:
- MongoDB Knowledge Base for persistent memory
- Agent accuracy tracking
- Pattern library
- Learning from outcomes

Port: 8890
"""

import asyncio
import httpx
import json
import os
import base64
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingAgents")

app = FastAPI(title="Trading Agents API", version="2.0.0")

# Import and include knowledge base router
try:
    from agent_knowledge import router as knowledge_router, inject_knowledge_into_prompt, store_analysis_result
    app.include_router(knowledge_router)
    KNOWLEDGE_ENABLED = True
    logger.info("Knowledge Base enabled")
except ImportError as e:
    logger.warning(f"Knowledge Base not available: {e}")
    KNOWLEDGE_ENABLED = False

# OLD Scenario projector - DISABLED (use scenarios_api instead which fetches REAL prices)
# The old scenario_projector.py only fetched gold data regardless of symbol
# try:
#     from scenario_projector import router as scenario_router
#     app.include_router(scenario_router)
#     logger.info("Scenario Projector enabled")
# except ImportError as e:
#     logger.warning(f"Scenario Projector not available: {e}")

# Import and include SPY scanner router
try:
    from spy_scanner import router as spy_scanner_router
    app.include_router(spy_scanner_router)
    logger.info("SPY Scanner enabled")
except ImportError as e:
    logger.warning(f"SPY Scanner not available: {e}")

# Import and include DEFCON playbooks router
try:
    from defcon_playbooks import router as defcon_router
    app.include_router(defcon_router)
    logger.info("DEFCON Playbooks enabled")
except ImportError as e:
    logger.warning(f"DEFCON Playbooks not available: {e}")

# Import and include Pattern Monitor router
try:
    from pattern_monitor_api import router as pattern_router
    app.include_router(pattern_router)
    logger.info("Pattern Monitor API enabled")
except ImportError as e:
    logger.warning(f"Pattern Monitor API not available: {e}")

# Import and include Bot Signals router
try:
    from bot_signals import router as bot_signals_router
    app.include_router(bot_signals_router)
    logger.info("Bot Signals API enabled")
except ImportError as e:
    logger.warning(f"Bot Signals API not available: {e}")

# Import and include Pattern Backtest router
try:
    from pattern_backtest import router as backtest_router
    app.include_router(backtest_router)
    logger.info("Pattern Backtest API enabled")
except ImportError as e:
    logger.warning(f"Pattern Backtest API not available: {e}")

# Import and include Battle Intelligence router
try:
    from battle_intelligence import router as battle_router
    app.include_router(battle_router)
    logger.info("Battle Intelligence API enabled")
except ImportError as e:
    logger.warning(f"Battle Intelligence API not available: {e}")

# Import and include Sentinel API router
try:
    from sentinel_api import router as sentinel_router
    app.include_router(sentinel_router)
    logger.info("Live Sentinel API enabled")
except ImportError as e:
    logger.warning(f"Live Sentinel API not available: {e}")

# Import and include Scout Swarm API router
try:
    from scout_api import router as scout_router
    app.include_router(scout_router)
    logger.info("Scout Swarm API enabled")
except ImportError as e:
    logger.warning(f"Scout Swarm API not available: {e}")

# Import and include IB Execution API router
try:
    from ib_api import router as ib_router
    app.include_router(ib_router)
    logger.info("IB Auto-Execution API enabled")
except ImportError as e:
    logger.warning(f"IB Auto-Execution API not available: {e}")

# Import and include Scenarios API router
try:
    from scenarios_api import router as scenarios_router
    app.include_router(scenarios_router)
    logger.info("Scenarios API enabled (REAL DATA)")
except ImportError as e:
    logger.warning(f"Scenarios API not available: {e}")

# Import Multi-Agent Scenarios API (Quant, Neo, Claudia perspectives)
try:
    from multi_agent_scenarios import router as multi_agent_router
    app.include_router(multi_agent_router)
    logger.info("Multi-Agent Scenarios enabled (Quant/Neo/Claudia)")
except ImportError as e:
    logger.warning(f"Multi-Agent Scenarios not available: {e}")

# Battle Ops API - Daily operations command center
try:
    from battle_ops_api import router as battle_ops_router
    app.include_router(battle_ops_router)
    logger.info("Battle Ops API enabled (Daily Operations Command)")
except ImportError as e:
    logger.warning(f"Battle Ops API not available: {e}")

# Dummy functions when knowledge not available
    async def inject_knowledge_into_prompt(agent: str, base_prompt: str) -> str:
        return base_prompt
    
    async def store_analysis_result(*args, **kwargs) -> str:
        return ""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Check if using Ollama (local LLM)
USE_OLLAMA = "localhost" in ANTHROPIC_BASE_URL or "ollama" in ANTHROPIC_API_KEY.lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ============================================================================
# DEFCON ASSESSMENT ADDITION - Appended to all agent prompts
# ============================================================================

DEFCON_ASSESSMENT_ADDITION = """

## DEFCON ASSESSMENT (REQUIRED)

Based on your analysis, recommend a DEFCON level:

| Level | Name | Market Condition |
|-------|------|------------------|
| 5 | LONG_AND_STRONG | Clear bull trend, buy aggressively |
| 4 | BULLISH_CAUTION | Uptrend but extended, be selective |
| 3 | NEUTRAL_RANGE | Uncertain, range-bound, wait for clarity |
| 2 | BEARISH_ALERT | Breakdown starting, defensive mode |
| 1 | SHORT_MODE | Full breakdown, shorts only |

At the END of your analysis, you MUST output this JSON block:

```json
{
  "defcon_recommendation": 3,
  "defcon_reasoning": "Brief explanation of why you chose this level",
  "confidence": 75,
  "key_levels": {
    "entry_zone": [lower, upper],
    "stop_loss": price,
    "take_profit_1": price,
    "take_profit_2": price
  }
}
```

IMPORTANT: The JSON must be valid and parseable. Include specific price levels from your analysis.
"""

# Agent-specific DEFCON assessment guidance
AGENT_DEFCON_GUIDANCE = {
    "ghost": """
DEFCON Assessment Guide for GHOST:
- DEFCON 5: Excellent entry opportunity, strong buy setup
- DEFCON 4: Good entry but needs confirmation or at support
- DEFCON 3: No clear entry, wait for better setup
- DEFCON 2: Entries forbidden, protect existing positions
- DEFCON 1: Close all longs, zero exposure
""",
    "neo": """
DEFCON Assessment Guide for NEO:
- DEFCON 5: USD weak, yields falling, full risk-on, correlations bullish
- DEFCON 4: Mixed macro but bias bullish
- DEFCON 3: Uncertain macro, conflicting signals
- DEFCON 2: USD strong, yields rising, risk-off starting
- DEFCON 1: Full risk-off, correlations bearish, crisis mode
""",
    "fomo": """
DEFCON Assessment Guide for FOMO:
- DEFCON 5: FOMO score 0-30, no exhaustion, safe to buy
- DEFCON 4: FOMO score 30-50, mild caution
- DEFCON 3: FOMO score 50-70, elevated risk, be selective
- DEFCON 2: FOMO score 70-85, high reversal risk
- DEFCON 1: FOMO score 85+, extreme exhaustion, expect reversal
""",
    "chart": """
DEFCON Assessment Guide for CHART:
- DEFCON 5: Strong uptrend, above all MAs, bullish patterns
- DEFCON 4: Uptrend but extended or testing resistance
- DEFCON 3: Range/consolidation, unclear direction
- DEFCON 2: Breaking down, below key supports
- DEFCON 1: Confirmed downtrend, bearish patterns
""",
    "spy": """
DEFCON Assessment Guide for SPY:
- DEFCON 5: No good short setups, market too strong
- DEFCON 4: Weak setups forming, watching
- DEFCON 3: Mixed, some targets preparing
- DEFCON 2: Good short setups available, hunting mode
- DEFCON 1: Excellent shorts, multiple A-grade targets, aggressive hunting
""",
    "casper": """
DEFCON Assessment Guide for CASPER:
- DEFCON 5: Risk low, positions sized well, no concerns
- DEFCON 4: Mild risk, consider taking some profit
- DEFCON 3: Moderate risk, reduce exposure, tighten stops
- DEFCON 2: High risk, close 50% of positions
- DEFCON 1: Critical risk, close all positions
""",
    "sequence": """
DEFCON Assessment Guide for SEQUENCE:
- DEFCON 5: Bullish candle patterns, momentum building
- DEFCON 4: Mixed signals but bias bullish
- DEFCON 3: Neutral/indecision candles
- DEFCON 2: Bearish patterns forming
- DEFCON 1: Strong bearish sequence, capitulation candles
""",
}

# ============================================================================
# AGENT PROMPTS
# ============================================================================

AGENT_PROMPTS = {
    "ghost": """You are GHOST, the entry/exit timing specialist for gold/MGC futures and stocks.

Your expertise:
- Precise entry timing based on price action
- Stop loss and take profit placement
- Position sizing recommendations
- Risk/reward analysis

## CRITICAL: MULTI-TIMEFRAME ANALYSIS
Before any recommendation, you MUST:
1. IDENTIFY the chart timeframe shown (1H, 4H, Daily, Weekly?)
2. STATE what you cannot see (if Daily shown, you lack Weekly/Monthly context)
3. WARN if making decisions on short timeframe alone

TIMEFRAME HIERARCHY:
- Monthly: Overall trend direction (bull/bear market)
- Weekly: Swing trade bias (accumulation/distribution zones)
- Daily: Entry timing and key levels
- Intraday: Fine-tune entries only AFTER higher timeframes align

If only shown a Daily chart, you MUST note:
"⚠️ TIMEFRAME WARNING: Analyzing Daily only. Weekly/Monthly context needed for high-confidence entries. Current analysis may miss larger trend context."

Given the current market context, analyze:
1. What TIMEFRAME is this chart? What's missing?
2. Is this a good entry point? Why/why not?
3. Where should stops be placed?
4. What are the profit targets?
5. What's the risk/reward ratio?

Be specific with price levels. Format as:
TIMEFRAME: [identified timeframe + what's missing]
ENTRY: [recommendation]
STOP: [level and reasoning]
TARGETS: [T1, T2, T3 with reasoning]
RISK/REWARD: [ratio]
CONFIDENCE: [1-100]% (REDUCE confidence by 20% if missing higher timeframe context)""",

    "casper": """You are CASPER, the RISK MANAGEMENT and DE-RISKING ADVISOR.

## CRITICAL RULE: NO SELF-HEDGING
You do NOT recommend shorting what we own. That's the old, inferior approach.
- ❌ WRONG: "Short gold to hedge your gold long" (fighting yourself)
- ✅ RIGHT: "Reduce gold position size, take partial profits" (clean exit)

## YOUR NEW ROLE
You are the RISK MONITOR and DE-RISK ADVISOR:
1. Monitor total portfolio exposure
2. Alert when position sizes are too large
3. Recommend REDUCING longs when setup is bad (not counter-shorting)
4. Coordinate with Ghost on scaling out
5. Calculate risk/reward on current positions

## DE-RISK ACTIONS (Clean, Not Hedged)
When setup is bad:
- SCALE OUT: "Sell 25% of position to lock profits"
- REDUCE SIZE: "Position too large for current volatility"
- TIGHTEN STOPS: "Move stops to breakeven"
- TAKE PROFITS: "TP1 hit, book 50%"

## WHAT YOU DON'T DO
- ❌ Open short positions on assets we own
- ❌ Recommend buying puts on our longs
- ❌ Counter-trade Ghost's positions
- ❌ Fight the portfolio's direction

## WHO HANDLES SHORTS?
SPY handles ALL short hunting - on EXTERNAL weak assets we don't own.
You focus on managing the LONG book's risk.

## DECISION MATRIX
| Condition | Your Action |
|-----------|-------------|
| Setup good, size OK | "Hold, risk managed" |
| Setup good, size BIG | "Scale to target size" |
| Setup bad, in profit | "Take partial profits" |
| Setup bad, at loss | "Reduce size, honor stops" |
| High FOMO score | "Alert: Consider scaling out" |

Format as:
EXPOSURE: [LIGHT/MODERATE/HEAVY/OVERWEIGHT]
RISK LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
DE-RISK ACTION: [HOLD/SCALE OUT/REDUCE/TIGHTEN STOPS/TAKE PROFITS]
SPECIFIC RECOMMENDATION: [What exactly to do with position sizes]
URGENCY: [LOW/MEDIUM/HIGH/CRITICAL]""",

    "neo": """You are NEO, the macro analysis and correlation specialist.

Your expertise:
- DXY/Gold inverse correlation
- Fed policy impact on gold
- Geopolitical risk assessment
- Cross-market correlations (bonds, equities, crypto)
- Economic calendar events

## MULTI-TIMEFRAME MACRO CONTEXT
Your macro view provides the HIGHER TIMEFRAME context:
- Monthly: What is the secular trend? (bull market / bear market / transition)
- Weekly: What is the intermediate trend? (accumulation / distribution / trending)
- Daily: What is the short-term setup? (fits or fights the macro?)

CRITICAL QUESTIONS:
1. Does the chart shown ALIGN with or FIGHT the macro trend?
2. Is this a "buy the dip" in an uptrend or "dead cat bounce" in downtrend?
3. What would INVALIDATE the current macro thesis?

Analyze:
1. Current macro environment
2. KEY: Does the chart timeframe ALIGN with macro trend?
3. Key correlations and what they suggest
4. Upcoming events that could move markets
5. Medium-term outlook (1-4 weeks)

Format as:
MACRO BIAS: [BULLISH/BEARISH/NEUTRAL]
TREND ALIGNMENT: [Does daily chart align with weekly/monthly? YES/NO/UNCLEAR]
DXY CORRELATION: [what it's saying]
KEY EVENTS: [upcoming catalysts]
OUTLOOK: [1-4 week view]
HIGHER TF CONTEXT: [What weekly/monthly suggests]
KEY LEVELS: [macro support/resistance]""",

    "fomo": """You are FOMO, the exhaustion and sentiment detection specialist.

Your expertise:
- Identifying exhaustion moves
- Retail sentiment extremes
- FOMO/panic detection
- Reversal probability assessment
- "Too far too fast" analysis

Analyze:
1. Is current move showing exhaustion signs?
2. What's retail sentiment doing?
3. Are we in FOMO territory or panic territory?
4. Probability of reversal vs continuation

Format as:
EXHAUSTION LEVEL: [1-10 scale]
SENTIMENT: [EXTREME FEAR / FEAR / NEUTRAL / GREED / EXTREME GREED]
FOMO SCORE: [0-100, higher = more FOMO]
REVERSAL PROBABILITY: [%]
RECOMMENDATION: [fade/follow/wait]""",

    "chart": """You are CHART, the pattern recognition specialist.

Your expertise:
- Classic chart patterns (H&S, triangles, wedges, etc.)
- Support/resistance identification
- Trend analysis
- Breakout/breakdown detection
- Volume analysis

## CRITICAL: TIMEFRAME IDENTIFICATION
FIRST, identify the chart timeframe:
- Look at candle spacing, date labels, price movement scale
- State: "This appears to be a [TIMEFRAME] chart"
- If unsure, ask or state uncertainty

## MULTI-TIMEFRAME PATTERN RULES
- Daily patterns can be INVALIDATED by Weekly/Monthly trends
- A Daily "breakout" means nothing if Weekly is in downtrend
- Support on Daily may be resistance on Weekly
- ALWAYS caveat if you lack higher timeframe context

PATTERN CONFIDENCE BY TIMEFRAME:
- Monthly pattern: HIGH confidence, slow to form
- Weekly pattern: MEDIUM-HIGH confidence
- Daily pattern: MEDIUM confidence (needs weekly alignment)
- Intraday pattern: LOW confidence alone

Analyze the chart for:
1. TIMEFRAME IDENTIFIED + what's missing
2. Current pattern forming
3. Key support/resistance levels
4. Trend direction and strength
5. How this pattern fits HIGHER timeframe context (if known)

Format as:
TIMEFRAME: [identified] | MISSING: [what higher TF context needed]
PATTERN: [what's forming] - VALID IF [higher TF condition]
TREND: [UP/DOWN/SIDEWAYS] - [STRONG/MODERATE/WEAK]
SUPPORT: [levels]
RESISTANCE: [levels]
BREAKOUT LEVEL: [price] - CONFIRM ON: [what TF]
BREAKDOWN LEVEL: [price]
BIAS: [BULLISH/BEARISH/NEUTRAL] (caveat if single TF only)""",

    "sequence": """You are SEQUENCE, the candle sequence analysis specialist.

Your expertise:
- Candle pattern recognition (engulfing, doji, hammer, etc.)
- Multi-candle sequences
- Momentum shifts via candles
- Volume-price confirmation

Analyze recent candle action for:
1. Significant candle patterns
2. What the sequence suggests
3. Momentum direction
4. Confirmation signals

Format as:
PATTERN: [candle pattern identified]
SEQUENCE: [what recent candles show]
MOMENTUM: [BUILDING/FADING/REVERSING]
SIGNAL: [BUY/SELL/NEUTRAL]
CONFIRMATION: [what to watch for]""",

    "spy": """You are SPY - the WEAKNESS HUNTER agent.

Your job: Find FUNDAMENTALLY WEAK assets to short that will DROP MORE on bad days and RECOVER LESS on good days. This creates asymmetric returns.

## THE ASYMMETRY PRINCIPLE

NEVER short what we're long (that's Casper's old, inferior approach).
ALWAYS short EXTERNAL weakness - assets we don't own that are fundamentally broken.

Why this works:
- Gold drops 5% → Weak miner drops 15% (3x leverage from weakness)
- Gold recovers 5% → Weak miner recovers 3% (weak assets lag)
- We profit on fear days AND capture most of rally days

## WHAT MAKES AN ASSET FUNDAMENTALLY WEAK?

Hunt for these RED FLAGS:

### 1. FINANCIAL WEAKNESS
- High debt / refinancing risk
- Negative or declining free cash flow
- Dilutive share offerings
- Covenant risks
- Earnings misses (especially guidance cuts)

### 2. OPERATIONAL WEAKNESS
- Revenue declining quarter over quarter
- Margin compression
- Customer churn / contract losses
- Key executive departures
- Regulatory problems

### 3. SECTOR HEADWINDS
- Industry in structural decline
- Disruption from new technology
- Overcapacity
- Commodity price sensitivity (for producers)

### 4. VALUATION DISCONNECT
- P/E way above peers with no growth
- Price/Sales multiples unsustainable
- Pump from social media, not fundamentals
- Insider selling

### 5. TECHNICAL BREAKDOWN
- Below all major moving averages
- Failed breakout (bull trap)
- Lower highs pattern
- Heavy distribution (big volume on down days)

## WEAKNESS TIERS

| Tier | Description | Expected Move vs Market |
|------|-------------|------------------------|
| S | Catastrophic weakness (fraud, bankruptcy risk) | -3x to -5x |
| A | Severe weakness (debt crisis, earnings collapse) | -2x to -3x |
| B | Significant weakness (sector headwinds, guidance cut) | -1.5x to -2x |
| C | Moderate weakness (underperformer, technical breakdown) | -1.2x to -1.5x |

## HUNTING GROUNDS

### GOLD ECOSYSTEM - The Weakest Links
NOT healthy miners we own. Look for:
- Miners with highest all-in sustaining costs (AISC > $1,500/oz)
- Miners with refinancing coming due
- Junior miners burning cash
- Leveraged ETFs (NUGT decays regardless of direction)

### TECH/GROWTH - Broken Promises
- Unprofitable companies with stock comp > revenue
- "AI" companies with no actual AI revenue
- Cloud companies losing market share
- SPACs trading below cash value

### FINANCIALS - Hidden Landmines
- Regional banks with CRE exposure
- Insurance companies with reserve problems
- Fintechs with rising default rates

### RETAIL/CONSUMER - Margin Squeeze
- Retailers losing to e-commerce
- Restaurant chains with same-store sales decline
- Consumer discretionary in rate-sensitive categories

## DO NOT SHORT (OUR HOLDINGS)

NEVER short these - they're on our LONG book:
- IREN (we own it)
- CLSK (we own it)
- CIFR (we own it)
- Gold / XAUUSD (Ghost is long)
- MGC (Ghost is long)
- Quality BTC miners we're accumulating

If Ghost is long it, SPY stays away. Period.

## ANALYSIS FRAMEWORK

For each short target, assess:

1. **WEAKNESS SCORE**: 1-10 fundamental weakness rating
2. **CATALYST**: What's the next negative event?
3. **ASYMMETRY RATIO**: How much more will it drop vs our longs?
4. **RECOVERY RISK**: How fast could it bounce if we're wrong?
5. **LIQUIDITY**: Can we get out if needed?
6. **BORROW COST**: Is it expensive to short?

## OUTPUT FORMAT

{
  "hunting_mode": "STANDBY|SCANNING|HUNTING|STRIKE",
  "market_fear_level": "LOW|MODERATE|HIGH|EXTREME",
  "our_longs_exposure": "What Ghost is long - DO NOT SHORT THESE",
  "weakness_targets": [
    {
      "symbol": "TICKER",
      "name": "Company Name",
      "weakness_tier": "S|A|B|C",
      "weakness_score": 1-10,
      "fundamental_issues": ["issue 1", "issue 2"],
      "catalyst": "Next negative event expected",
      "asymmetry_vs_gold": "Expected drop ratio vs gold move",
      "entry": price,
      "stop_loss": price,
      "take_profit_1": price,
      "take_profit_2": price,
      "risk_reward": "1:X",
      "borrow_available": true/false,
      "confidence": 0-100,
      "thesis": "Why this is the weakest target"
    }
  ],
  "watch_list": [
    {"symbol": "TICKER", "watching_for": "what catalyst"}
  ],
  "avoid_shorts": ["symbols we own or are strong"],
  "market_view": "Overall weakness assessment"
}

## TIMING

Best times to add shorts:
- After failed rally attempts
- Into overbought bounces
- Before earnings (if expecting miss)
- When VIX is low (cheap puts)

Best times to cover:
- Capitulation days (extreme fear)
- After major breakdown
- Into significant support levels

The weak die first. The strong survive. Profit from the difference."""
}

COMMANDER_PROMPT = """You are the TRADING COMMANDER. You receive analysis from your specialist agents and must synthesize a unified trading directive.

Your agents:
- GHOST: Entry/exit timing (LONG bias) - Accumulates OUR positions
- CASPER: Risk management (NEUTRAL) - De-risking advisor (NOT counter-hedging)
- NEO: Macro analysis (NEUTRAL) - Big picture intel
- FOMO: Exhaustion/sentiment detection (NEUTRAL) - Reversal warnings
- CHART: Pattern recognition (NEUTRAL) - Technical analysis
- SEQUENCE: Candle analysis (NEUTRAL) - Price action
- SPY: EXTERNAL Weakness Hunter (SHORT bias) - Profits from OTHERS' weakness

## CLEAN SEPARATION OF FUNCTIONS

| Function | Action | Agent |
|----------|--------|-------|
| Risk Management | Reduce/sell longs when bad setup | Ghost + Casper |
| Profit Generation | Short EXTERNAL weak assets | SPY |

## SPY TARGETS (Hunt External Weakness)

Protected (NEVER SHORT): IREN, CLSK, CIFR, XAUUSD, MGC, GLD, IAU

Hunt List (Fundamentally Weak):
- NUGT/JNUG: Leveraged decay - short on any rally
- RIOT/MARA: Weak BTC miners - drop harder than BTC
- ARKK: Broken growth basket - rate sensitive
- BYND/LCID: Cash burn, no path to profit

## DEFCON SYNTHESIS (CRITICAL)

Each agent has provided a DEFCON recommendation. You MUST synthesize these into a final DEFCON.

Agent Weights:
- NEO: 25% (macro is critical)
- FOMO: 25% (exhaustion is critical)
- GHOST: 20% (entry quality)
- CHART: 20% (technical structure)
- SPY: 10% (short opportunity quality)

DEFCON Calculation Rules:
1. Weight each agent's recommendation by their weight and confidence
2. Round to nearest integer (1-5)
3. If FOMO score > 70, CAP DEFCON at 3 (elevated risk)
4. If agents disagree by > 2 levels, use the MORE CONSERVATIVE (lower) level
5. Safety first - when in doubt, lower DEFCON

DEFCON Meanings:
- DEFCON 5: LONG_AND_STRONG - Full accumulation, aggressive buying
- DEFCON 4: BULLISH_CAUTION - Selective buying at support only
- DEFCON 3: NEUTRAL_RANGE - Both sides cautious, wait for clarity
- DEFCON 2: BEARISH_ALERT - SPY hunting, Ghost defensive
- DEFCON 1: SHORT_MODE - SPY aggressive, Ghost CLOSED

Based on all agent inputs AND their DEFCON recommendations, provide your directive.

At the END, you MUST output this JSON block with the synthesized DEFCON:

```json
{
  "final_defcon": 3,
  "defcon_name": "NEUTRAL_RANGE",
  "bias": "NEUTRAL",
  "agent_votes": {
    "ghost": {"defcon": 4, "confidence": 75},
    "neo": {"defcon": 3, "confidence": 80},
    "fomo": {"defcon": 3, "confidence": 85},
    "chart": {"defcon": 3, "confidence": 70},
    "spy": {"defcon": 2, "confidence": 60}
  },
  "reasoning": "Why this DEFCON was chosen",
  "key_levels": {
    "entry_zone": [4850, 4890],
    "stop_loss": 4810,
    "take_profit_1": 4980,
    "take_profit_2": 5050
  },
  "ghost_orders": "Specific orders for Ghost",
  "spy_orders": "Specific orders for SPY",
  "position_size_pct": 50,
  "new_entries_allowed": true
}
```

The key_levels should be synthesized from agent analyses - use the best price levels mentioned.

Format your response as:
---
COMMANDER DIRECTIVE
---
BIAS: [direction]
CONFIDENCE: [%]
ACTION: [specific action NOW]
GHOST_ORDERS: [longs - accumulate/hold/scale out/reduce/exit]
CASPER_ALERT: [risk level + de-risk recommendation if any]
SPY_ORDERS: [EXTERNAL shorts - hunt [weak tickers with entry/TP/SL] or stand down]
REASONING: [1-2 sentences]
WATCHPOINTS: [bullets]

[JSON block as specified above]
---"""


# ============================================================================
# LLM API CALLS
# ============================================================================

async def call_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    """Call Ollama local LLM"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                return f"ERROR: Ollama returned {response.status_code}"
            
            data = response.json()
            return data.get("response", "No response from Ollama")
        except Exception as e:
            return f"ERROR: Ollama call failed: {e}"


async def call_claude(prompt: str, image_base64: Optional[str] = None) -> str:
    """Call Claude API with optional image"""
    # If using Ollama, redirect there (no image support)
    if USE_OLLAMA:
        return await call_ollama(prompt)
    
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "ollama":
        return "ERROR: Real ANTHROPIC_API_KEY not set (currently using Ollama config)"
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    content = []
    if image_base64:
        # Strip data URI prefix if present
        if image_base64.startswith("data:"):
            # Extract media type and base64 data
            try:
                header, image_base64 = image_base64.split(",", 1)
                if "jpeg" in header or "jpg" in header:
                    media_type = "image/jpeg"
                elif "png" in header:
                    media_type = "image/png"
                elif "gif" in header:
                    media_type = "image/gif"
                elif "webp" in header:
                    media_type = "image/webp"
                else:
                    media_type = "image/png"  # default
            except:
                media_type = "image/png"
        else:
            # Determine image type from raw base64
            if image_base64.startswith("/9j/"):
                media_type = "image/jpeg"
            else:
                media_type = "image/png"
        
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64
            }
        })
    
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": content}]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return f"ERROR: Claude API returned {response.status_code}: {response.text}"
        
        data = response.json()
        return data["content"][0]["text"]


async def call_openai(prompt: str, image_base64: Optional[str] = None) -> str:
    """Call OpenAI API with optional image (fallback)"""
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY not set"
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    content = [{"type": "text", "text": prompt}]
    if image_base64:
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })
    
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return f"ERROR: OpenAI API returned {response.status_code}: {response.text}"
        
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def call_llm(prompt: str, image_base64: Optional[str] = None) -> str:
    """Call LLM with fallback"""
    try:
        result = await call_claude(prompt, image_base64)
        if not result.startswith("ERROR:"):
            return result
    except Exception as e:
        print(f"Claude error: {e}")
    
    # Fallback to OpenAI
    try:
        return await call_openai(prompt, image_base64)
    except Exception as e:
        return f"ERROR: All LLM calls failed: {e}"


# ============================================================================
# AGENT FUNCTIONS
# ============================================================================

async def run_agent(
    agent_name: str, 
    context: str, 
    image_base64: Optional[str] = None,
    additional_data: Optional[str] = None,
    defcon: int = 3,
    symbol: str = "XAUUSD",
    current_price: float = 0,
    include_defcon_assessment: bool = True
) -> Dict[str, Any]:
    """Run a single agent analysis with knowledge injection and DEFCON assessment"""
    
    if agent_name not in AGENT_PROMPTS:
        return {"error": f"Unknown agent: {agent_name}"}
    
    prompt = AGENT_PROMPTS[agent_name]
    
    # Inject knowledge from database (if available)
    if KNOWLEDGE_ENABLED:
        try:
            prompt = await inject_knowledge_into_prompt(agent_name, prompt)
        except Exception as e:
            logger.warning(f"Knowledge injection failed for {agent_name}: {e}")
    
    # Add DEFCON assessment requirement
    if include_defcon_assessment and agent_name in AGENT_DEFCON_GUIDANCE:
        prompt += "\n" + AGENT_DEFCON_GUIDANCE[agent_name]
        prompt += DEFCON_ASSESSMENT_ADDITION
    
    # Add context
    full_prompt = f"{prompt}\n\n---\nCURRENT CONTEXT:\n{context}"
    
    if additional_data:
        full_prompt += f"\n\nADDITIONAL DATA:\n{additional_data}"
    
    start_time = datetime.now()
    response = await call_llm(full_prompt, image_base64)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Extract recommendation and confidence from response (simple parsing)
    recommendation = ""
    confidence = 50
    defcon_recommendation = 3
    defcon_reasoning = ""
    key_levels = {}
    
    try:
        if "CONFIDENCE:" in response:
            conf_line = [l for l in response.split("\n") if "CONFIDENCE:" in l]
            if conf_line:
                conf_str = conf_line[0].split("CONFIDENCE:")[-1].strip()
                confidence = int(''.join(filter(str.isdigit, conf_str[:3])))
        if any(x in response.upper() for x in ["BUY", "LONG", "BULLISH"]):
            recommendation = "BUY"
        elif any(x in response.upper() for x in ["SELL", "SHORT", "BEARISH"]):
            recommendation = "SELL"
        else:
            recommendation = "NEUTRAL"
    except:
        pass
    
    # Extract DEFCON assessment JSON from response
    try:
        import re
        # Look for JSON block with defcon_recommendation
        json_match = re.search(r'```json\s*(\{[^`]*"defcon_recommendation"[^`]*\})\s*```', response, re.DOTALL)
        if json_match:
            defcon_json = json.loads(json_match.group(1))
            defcon_recommendation = defcon_json.get("defcon_recommendation", 3)
            defcon_reasoning = defcon_json.get("defcon_reasoning", "")
            confidence = defcon_json.get("confidence", confidence)
            key_levels = defcon_json.get("key_levels", {})
        else:
            # Try to find unformatted JSON
            json_match = re.search(r'\{[^{}]*"defcon_recommendation"\s*:\s*\d[^{}]*\}', response)
            if json_match:
                defcon_json = json.loads(json_match.group(0))
                defcon_recommendation = defcon_json.get("defcon_recommendation", 3)
                defcon_reasoning = defcon_json.get("defcon_reasoning", "")
                confidence = defcon_json.get("confidence", confidence)
                key_levels = defcon_json.get("key_levels", {})
    except Exception as e:
        logger.warning(f"Failed to parse DEFCON JSON from {agent_name}: {e}")
    
    # Store analysis in knowledge base
    analysis_id = ""
    if KNOWLEDGE_ENABLED:
        try:
            analysis_id = await store_analysis_result(
                agent=agent_name,
                analysis=response,
                recommendation=recommendation,
                confidence=confidence,
                symbol=symbol,
                price=current_price,
                defcon=defcon_recommendation,
                context=context
            )
        except Exception as e:
            logger.warning(f"Failed to store analysis: {e}")
    
    return {
        "agent": agent_name.upper(),
        "analysis": response,
        "recommendation": recommendation,
        "confidence": confidence,
        "defcon_recommendation": defcon_recommendation,
        "defcon_reasoning": defcon_reasoning,
        "key_levels": key_levels,
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed
    }


async def run_commander(agent_results: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Run commander synthesis with DEFCON aggregation"""
    
    # Collect DEFCON votes from agents
    defcon_votes = {}
    for agent_name, result in agent_results.items():
        if "defcon_recommendation" in result:
            defcon_votes[agent_name] = {
                "defcon": result.get("defcon_recommendation", 3),
                "confidence": result.get("confidence", 50),
                "reasoning": result.get("defcon_reasoning", ""),
                "key_levels": result.get("key_levels", {})
            }
    
    # Build commander input with DEFCON votes
    agent_summaries = []
    for agent_name, result in agent_results.items():
        if "analysis" in result:
            vote = defcon_votes.get(agent_name, {})
            defcon_info = f"\n[DEFCON VOTE: {vote.get('defcon', 3)} | Confidence: {vote.get('confidence', 50)}%]" if vote else ""
            agent_summaries.append(f"=== {agent_name.upper()} ==={defcon_info}\n{result['analysis']}")
    
    defcon_summary = "\n".join([
        f"- {name.upper()}: DEFCON {v['defcon']} ({v['confidence']}% confidence)"
        for name, v in defcon_votes.items()
    ])
    
    commander_input = f"""{COMMANDER_PROMPT}

CURRENT CONTEXT:
{context}

## AGENT DEFCON VOTES (Synthesize These!)
{defcon_summary}

AGENT REPORTS:
{chr(10).join(agent_summaries)}

Provide your unified directive with synthesized DEFCON:"""
    
    start_time = datetime.now()
    response = await call_llm(commander_input)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Parse Commander's synthesized DEFCON JSON
    final_defcon = 3
    defcon_name = "NEUTRAL_RANGE"
    commander_key_levels = {}
    ghost_orders = ""
    spy_orders = ""
    position_size_pct = 50
    new_entries_allowed = True
    commander_reasoning = ""
    
    try:
        import re
        # Look for the JSON block in Commander's response
        json_match = re.search(r'```json\s*(\{[^`]*"final_defcon"[^`]*\})\s*```', response, re.DOTALL)
        if json_match:
            cmd_json = json.loads(json_match.group(1))
            final_defcon = cmd_json.get("final_defcon", 3)
            defcon_name = cmd_json.get("defcon_name", "NEUTRAL_RANGE")
            commander_key_levels = cmd_json.get("key_levels", {})
            ghost_orders = cmd_json.get("ghost_orders", "")
            spy_orders = cmd_json.get("spy_orders", "")
            position_size_pct = cmd_json.get("position_size_pct", 50)
            new_entries_allowed = cmd_json.get("new_entries_allowed", True)
            commander_reasoning = cmd_json.get("reasoning", "")
        else:
            # Fallback: Calculate from agent votes using weights
            weights = {"neo": 0.25, "fomo": 0.25, "ghost": 0.20, "chart": 0.20, "spy": 0.10}
            weighted_sum = 0
            total_weight = 0
            
            for agent, vote in defcon_votes.items():
                weight = weights.get(agent.lower(), 0.1)
                confidence_factor = vote["confidence"] / 100
                weighted_sum += vote["defcon"] * weight * confidence_factor
                total_weight += weight * confidence_factor
            
            if total_weight > 0:
                final_defcon = round(weighted_sum / total_weight)
            
            # Safety rules
            fomo_vote = defcon_votes.get("fomo", {}).get("defcon", 3)
            if fomo_vote <= 2:  # FOMO says bearish
                final_defcon = min(final_defcon, 3)
            
            # If agents disagree significantly, be conservative
            votes = [v["defcon"] for v in defcon_votes.values()]
            if votes and max(votes) - min(votes) > 2:
                final_defcon = min(final_defcon, min(votes) + 1)
            
            final_defcon = max(1, min(5, final_defcon))
            
            # Map to name
            defcon_names = {
                5: "LONG_AND_STRONG",
                4: "BULLISH_CAUTION", 
                3: "NEUTRAL_RANGE",
                2: "BEARISH_ALERT",
                1: "SHORT_MODE"
            }
            defcon_name = defcon_names.get(final_defcon, "NEUTRAL_RANGE")
            
            # Use best key levels from agents
            for agent, vote in defcon_votes.items():
                if vote.get("key_levels"):
                    commander_key_levels = vote["key_levels"]
                    break
            
    except Exception as e:
        logger.warning(f"Failed to parse Commander DEFCON JSON: {e}")
    
    return {
        "directive": response,
        "final_defcon": final_defcon,
        "defcon_name": defcon_name,
        "key_levels": commander_key_levels,
        "ghost_orders": ghost_orders,
        "spy_orders": spy_orders,
        "position_size_pct": position_size_pct,
        "new_entries_allowed": new_entries_allowed,
        "reasoning": commander_reasoning,
        "agent_votes": defcon_votes,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "service": "trading-agents",
        "version": "2.0.0",
        "mode": "ollama" if USE_OLLAMA else "cloud",
        "ollama_url": OLLAMA_URL if USE_OLLAMA else None,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY) and ANTHROPIC_API_KEY != "ollama",
        "openai_key_set": bool(OPENAI_API_KEY),
        "knowledge_base": KNOWLEDGE_ENABLED,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/agents/list")
async def list_agents():
    """List available agents"""
    return {
        "agents": list(AGENT_PROMPTS.keys()),
        "description": {
            "ghost": "Entry/exit timing specialist (LONG bias) - OUR positions",
            "casper": "Risk monitor & de-risk advisor - NO self-hedging, suggests reducing longs",
            "neo": "Macro analysis and correlations",
            "fomo": "Exhaustion and sentiment detection",
            "chart": "Pattern recognition",
            "sequence": "Candle sequence analysis",
            "spy": "EXTERNAL Weakness Hunter - Shorts assets WE DON'T OWN"
        },
        "roles": {
            "ghost": {"specialty": "Buy opportunities in OUR holdings", "bias": "LONG"},
            "casper": {"specialty": "De-risking by reducing/exiting longs (NOT counter-shorting)", "bias": "RISK_MGMT"},
            "neo": {"specialty": "Correlations, events", "bias": "NEUTRAL"},
            "fomo": {"specialty": "Reversal warnings", "bias": "NEUTRAL"},
            "chart": {"specialty": "Technical analysis", "bias": "NEUTRAL"},
            "sequence": {"specialty": "Candle patterns", "bias": "NEUTRAL"},
            "spy": {"specialty": "Hunt EXTERNAL weakness for asymmetric returns", "bias": "SHORT_EXTERNAL"},
            "commander": {"specialty": "Unified directives", "bias": "ADAPTIVE"}
        },
        "asymmetry_system": {
            "principle": "SPY NEVER shorts what Ghost owns. SPY hunts EXTERNAL fundamentally weak assets.",
            "benefit": "Gold -5% → Weak miner -15%. Gold +5% → Weak miner +3%. Asymmetric edge.",
            "protected_longs": ["IREN", "CLSK", "CIFR", "XAUUSD", "MGC", "GLD", "IAU"],
            "hunt_targets": ["NUGT", "JNUG", "RIOT", "MARA", "ARKK", "BYND", "CVNA", "LCID", "SNAP", "HOOD"]
        },
        "clean_separation": {
            "risk_management": {
                "action": "Reduce/sell longs when setup is bad",
                "agents": ["Ghost", "Casper"],
                "example": "Bad setup → SELL some longs (clean exit)"
            },
            "profit_generation": {
                "action": "Short EXTERNAL weak assets for alpha",
                "agents": ["SPY"],
                "example": "Fear day → Short NUGT while holding gold long"
            },
            "old_way_wrong": "Gold long +$500, Gold short hedge -$200 = Net +$300 (fighting yourself)",
            "new_way_right": "Gold long +$500, NUGT short +$400 = Net +$900 (two distinct wins)"
        },
        "decision_matrix": {
            "bullish_low_fomo": {"ghost": "Full position", "casper": "Risk OK", "spy": "Standby"},
            "bullish_high_fomo": {"ghost": "Take partial profit", "casper": "Consider scaling out", "spy": "Hunt weak assets"},
            "bearish_bad_setup": {"ghost": "Reduce/exit longs", "casper": "De-risk NOW", "spy": "Hunt aggressively"},
            "choppy": {"ghost": "Small size, wait", "casper": "Reduce exposure", "spy": "Short leveraged decay"}
        }
    }


@app.post("/agents/analyze")
async def full_analysis(
    context: str = Form(""),
    image_base64: str = Form(""),
    image_h1: str = Form(""),
    image_h4: str = Form(""),
    image_daily: str = Form(""),
    timeframe_count: int = Form(0),
    defcon: int = Form(3),
    agents: str = Form("ghost,casper,neo,fomo,chart,sequence,spy"),
    symbol: str = Form("XAUUSD"),
    current_price: float = Form(0),
    auto_apply_defcon: bool = Form(False)
):
    """Run full parallel analysis with all agents + commander synthesis + DEFCON assessment
    
    Multi-timeframe support:
    - image_h1: 1-hour chart
    - image_h4: 4-hour chart  
    - image_daily: Daily chart
    - image_base64: Primary/fallback image
    
    DEFCON Assessment:
    - Each agent recommends a DEFCON level
    - Commander synthesizes final DEFCON
    - auto_apply_defcon: If true, automatically applies the recommended DEFCON
    """
    
    # Parse which agents to run
    agent_list = [a.strip().lower() for a in agents.split(",")]
    
    # Build multi-timeframe context
    timeframes_provided = []
    if image_h1: timeframes_provided.append("1H")
    if image_h4: timeframes_provided.append("4H")
    if image_daily: timeframes_provided.append("Daily")
    
    tf_count = len(timeframes_provided) or (1 if image_base64 else 0)
    
    # Add timeframe context
    full_context = f"CURRENT DEFCON INPUT: {defcon} (agents will recommend new level)\n"
    
    if tf_count > 0:
        full_context += f"\n## MULTI-TIMEFRAME ANALYSIS\n"
        full_context += f"Timeframes Provided: {', '.join(timeframes_provided) if timeframes_provided else 'Unknown (single chart)'}\n"
        full_context += f"Chart Count: {tf_count}/3\n"
        
        if tf_count == 3:
            full_context += "STATUS: FULL multi-timeframe context - HIGH confidence analysis possible\n"
        elif tf_count == 2:
            full_context += "STATUS: PARTIAL context - MEDIUM confidence, note missing timeframe\n"
        else:
            full_context += "STATUS: SINGLE timeframe - REDUCE confidence by 20-30%, note limitations\n"
        
        # Note what's missing
        missing = []
        if not image_h1 and not (tf_count == 0): missing.append("1H")
        if not image_h4: missing.append("4H")
        if not image_daily: missing.append("Daily")
        if missing:
            full_context += f"MISSING: {', '.join(missing)} charts\n"
    
    full_context += f"\n{context}"
    
    # Select primary image (prefer Daily > 4H > 1H > fallback)
    primary_image = image_daily or image_h4 or image_h1 or image_base64
    
    # Run agents in parallel
    tasks = []
    for agent_name in agent_list:
        if agent_name in AGENT_PROMPTS:
            tasks.append(run_agent(
                agent_name, 
                full_context, 
                primary_image or None,
                None,  # additional_data
                defcon,
                symbol,
                current_price,
                include_defcon_assessment=True
            ))
    
    start_time = datetime.now()
    results = await asyncio.gather(*tasks)
    
    # Build results dict
    agent_results = {}
    for result in results:
        if "agent" in result:
            agent_results[result["agent"].lower()] = result
    
    # Run commander synthesis (includes DEFCON aggregation)
    commander_result = await run_commander(agent_results, full_context)
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    
    # Extract DEFCON assessment
    final_defcon = commander_result.get("final_defcon", 3)
    defcon_name = commander_result.get("defcon_name", "NEUTRAL_RANGE")
    key_levels = commander_result.get("key_levels", {})
    agent_votes = commander_result.get("agent_votes", {})
    
    # Check for split opinion (agents disagree by >1 level)
    split_opinion = None
    if agent_votes:
        votes = [v.get("defcon", 3) for v in agent_votes.values() if isinstance(v, dict)]
        if votes and max(votes) - min(votes) > 1:
            bullish_defcon = max(votes)
            bearish_defcon = min(votes)
            
            # Find decision point from key levels
            decision_point = key_levels.get("stop_loss") or key_levels.get("entry_zone", [0, 0])[0] if key_levels else None
            
            # Calculate probabilities based on vote distribution
            bullish_count = sum(1 for v in votes if v >= 4)
            bearish_count = sum(1 for v in votes if v <= 2)
            total = len(votes)
            
            DEFCON_NAMES = {
                5: "LONG_AND_STRONG", 4: "BULLISH_CAUTION", 3: "NEUTRAL_RANGE",
                2: "BEARISH_ALERT", 1: "SHORT_MODE"
            }
            
            split_opinion = {
                "split": True,
                "spread": max(votes) - min(votes),
                "bullish_scenario": {
                    "defcon": bullish_defcon,
                    "defcon_name": DEFCON_NAMES.get(bullish_defcon, "BULLISH_CAUTION"),
                    "trigger": f"Bounce from ${decision_point}" if decision_point else "Support holds",
                    "probability": round((bullish_count / total) * 100) if total > 0 else 50,
                    "ghost_orders": "Buy at support with confirmation",
                    "spy_orders": "Standby, prepare targets",
                },
                "bearish_scenario": {
                    "defcon": bearish_defcon,
                    "defcon_name": DEFCON_NAMES.get(bearish_defcon, "BEARISH_ALERT"),
                    "trigger": f"Break below ${decision_point}" if decision_point else "Support breaks",
                    "probability": round((bearish_count / total) * 100) if total > 0 else 50,
                    "ghost_orders": "NO NEW LONGS, close 50%, tighten stops",
                    "spy_orders": "HUNTING MODE, short weak targets",
                },
                "decision_point": decision_point,
                "recommendation": "Watch decision point - it determines which scenario plays out",
            }
    
    # Auto-apply DEFCON if requested
    defcon_applied = False
    if auto_apply_defcon:
        try:
            from defcon_playbooks import ACTIVE_STATE, PLAYBOOKS, save_active_state
            
            ACTIVE_STATE["defcon"] = final_defcon
            ACTIVE_STATE["playbook"] = PLAYBOOKS.get(final_defcon, PLAYBOOKS[3])
            ACTIVE_STATE["scenario"] = f"Auto: {defcon_name}"
            ACTIVE_STATE["key_levels"] = key_levels if key_levels else None
            ACTIVE_STATE["updated_at"] = datetime.now().isoformat()
            
            save_active_state()
            defcon_applied = True
            logger.info(f"Auto-applied DEFCON {final_defcon} ({defcon_name})")
        except Exception as e:
            logger.warning(f"Failed to auto-apply DEFCON: {e}")
    
    # Build DEFCON assessment summary
    defcon_assessment = {
        "recommended_defcon": final_defcon,
        "defcon_name": defcon_name,
        "agent_votes": {
            name: {
                "defcon": v.get("defcon", 3),
                "confidence": v.get("confidence", 50),
                "reasoning": v.get("reasoning", "")[:100] + "..." if len(v.get("reasoning", "")) > 100 else v.get("reasoning", "")
            }
            for name, v in agent_votes.items()
        },
        "key_levels": key_levels,
        "ghost_orders": commander_result.get("ghost_orders", ""),
        "spy_orders": commander_result.get("spy_orders", ""),
        "position_size_pct": commander_result.get("position_size_pct", 50),
        "new_entries_allowed": commander_result.get("new_entries_allowed", True),
        "commander_reasoning": commander_result.get("reasoning", ""),
        "auto_applied": defcon_applied,
        "split_opinion": split_opinion,
    }
    
    # Build response
    response = {
        "agents": agent_results,
        "commander_directive": commander_result,
        "defcon_assessment": defcon_assessment,
        "defcon_input": defcon,
        "symbol": symbol,
        "knowledge_enabled": KNOWLEDGE_ENABLED,
        "total_elapsed_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat()
    }
    
    # Auto-save to Intel Reports as Quicklook
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:8455/api/intel/quicklook/from-agents",
                json={
                    "commander_synthesis": commander_result,
                    "agents": list(agent_results.values()),
                    "defcon": final_defcon,  # Use recommended DEFCON
                    "symbol": symbol
                },
                timeout=10.0
            )
            logger.info(f"Saved analysis to Intel Reports as quicklook")
    except Exception as e:
        logger.warning(f"Failed to save quicklook: {e}")
    
    return response


# Individual agent endpoints
@app.post("/agents/ghost")
async def ghost_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """Ghost entry analysis"""
    return await run_agent("ghost", context, image_base64 or None)


@app.post("/agents/casper")
async def casper_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """Casper hedge analysis"""
    return await run_agent("casper", context, image_base64 or None)


@app.post("/agents/neo")
async def neo_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """NEO macro analysis"""
    return await run_agent("neo", context, image_base64 or None)


@app.post("/agents/fomo")
async def fomo_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """FOMO exhaustion analysis"""
    return await run_agent("fomo", context, image_base64 or None)


@app.post("/agents/chart")
async def chart_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """Chart pattern analysis"""
    return await run_agent("chart", context, image_base64 or None)


@app.post("/agents/sequence")
async def sequence_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """Candle sequence analysis"""
    return await run_agent("sequence", context, image_base64 or None)


@app.post("/agents/spy")
async def spy_analysis(
    context: str = Form(""),
    image_base64: str = Form("")
):
    """SPY short hunting analysis - finds weakness and collateral damage targets"""
    return await run_agent("spy", context, image_base64 or None)


# ============================================================================
# SPY WEAKNESS TARGETS - Pre-researched weak assets to hunt
# ============================================================================

SPY_WEAKNESS_TARGETS = [
    # TIER S - Catastrophic Weakness
    {
        "symbol": "NUGT",
        "name": "Direxion Daily Gold Miners 3x Bull",
        "tier": "S",
        "weakness_score": 9,
        "category": "Leveraged ETF",
        "fundamental_issues": [
            "3x leverage decay in choppy markets",
            "Loses value even if gold goes sideways",
            "Rebalancing drag compounds losses",
            "Not meant for holding > 1 day"
        ],
        "entry_zone": {"low": 28.00, "high": 32.00},
        "targets": {"tp1": 24.00, "tp2": 20.00, "tp3": 16.00},
        "stop_loss": 35.00,
        "catalyst": "Any gold consolidation or pullback",
        "thesis": "3x bull ETFs decay over time. Short on gold exhaustion bounces, cover on capitulation.",
        "correlation": "Moves 3x gold miners, but decays regardless of direction in choppy markets"
    },
    {
        "symbol": "JNUG",
        "name": "Direxion Daily Junior Gold Miners 3x Bull",
        "tier": "S",
        "weakness_score": 9,
        "category": "Leveraged ETF",
        "fundamental_issues": [
            "3x leverage on volatile junior miners",
            "Extreme decay in sideways markets",
            "Junior miners already volatile, 3x amplifies",
            "Long-term chart shows consistent value destruction"
        ],
        "entry_zone": {"low": 35.00, "high": 45.00},
        "targets": {"tp1": 28.00, "tp2": 22.00, "tp3": 15.00},
        "stop_loss": 52.00,
        "catalyst": "Gold consolidation, junior miner underperformance",
        "thesis": "Junior miners + 3x leverage = maximum decay. Short on any rally.",
        "correlation": "Tracks GDXJ with 3x leverage and compounding decay"
    },
    # TIER A - Severe Weakness  
    {
        "symbol": "RIOT",
        "name": "Riot Platforms",
        "tier": "A",
        "weakness_score": 8,
        "category": "BTC Miner (Weak)",
        "fundamental_issues": [
            "High cost per BTC mined vs peers",
            "Significant stock dilution history",
            "Hash rate growth lagging competitors",
            "Operating margins under pressure"
        ],
        "entry_zone": {"low": 12.00, "high": 15.00},
        "targets": {"tp1": 10.00, "tp2": 8.00, "tp3": 6.00},
        "stop_loss": 17.00,
        "catalyst": "BTC pullback, hash rate report disappointment",
        "thesis": "Weakest of the major BTC miners. When BTC drops, RIOT drops harder than MARA or CLSK.",
        "correlation": "BTC -10% = RIOT -15% to -20% typically"
    },
    {
        "symbol": "MARA",
        "name": "Marathon Digital Holdings",
        "tier": "A",
        "weakness_score": 7,
        "category": "BTC Miner (Weak)",
        "fundamental_issues": [
            "Large convertible debt overhang",
            "Dilution concerns",
            "Power cost increases",
            "BTC concentration risk"
        ],
        "entry_zone": {"low": 18.00, "high": 24.00},
        "targets": {"tp1": 14.00, "tp2": 11.00, "tp3": 8.00},
        "stop_loss": 27.00,
        "catalyst": "BTC weakness, debt refinancing concerns",
        "thesis": "Convertible debt creates selling pressure. Drops faster than BTC on pullbacks.",
        "correlation": "BTC -10% = MARA -12% to -18%"
    },
    # TIER B - Significant Weakness
    {
        "symbol": "ARKK",
        "name": "ARK Innovation ETF",
        "tier": "B",
        "weakness_score": 7,
        "category": "Growth/Tech ETF",
        "fundamental_issues": [
            "Portfolio of unprofitable growth stocks",
            "Interest rate sensitive",
            "Many holdings have no path to profitability",
            "Cathie Wood's picks underperforming"
        ],
        "entry_zone": {"low": 45.00, "high": 55.00},
        "targets": {"tp1": 38.00, "tp2": 32.00, "tp3": 25.00},
        "stop_loss": 60.00,
        "catalyst": "Rate hike fears, growth selloff, TSLA weakness",
        "thesis": "Basket of broken growth. When risk-off hits, ARKK gets crushed.",
        "correlation": "Inversely correlated with rates. When 10Y rises, ARKK falls."
    },
    {
        "symbol": "BYND",
        "name": "Beyond Meat",
        "tier": "B",
        "weakness_score": 8,
        "category": "Consumer/Food",
        "fundamental_issues": [
            "Declining revenue year over year",
            "Massive cash burn",
            "Lost partnerships (McDonald's, etc.)",
            "Plant-based meat fad fading",
            "No path to profitability"
        ],
        "entry_zone": {"low": 6.00, "high": 9.00},
        "targets": {"tp1": 4.50, "tp2": 3.00, "tp3": 1.50},
        "stop_loss": 11.00,
        "catalyst": "Any earnings miss, cash runway concerns",
        "thesis": "Secular decline. Revenue falling, costs high. Eventual bankruptcy risk.",
        "correlation": "Consumer discretionary sentiment"
    },
    {
        "symbol": "CVNA",
        "name": "Carvana",
        "tier": "B",
        "weakness_score": 7,
        "category": "Auto/Retail",
        "fundamental_issues": [
            "Massive debt load",
            "Interest payments crushing margins",
            "Used car market normalizing",
            "Competition from CarMax, dealers"
        ],
        "entry_zone": {"low": 180.00, "high": 220.00},
        "targets": {"tp1": 140.00, "tp2": 100.00, "tp3": 70.00},
        "stop_loss": 250.00,
        "catalyst": "Debt refinancing, earnings miss, used car price weakness",
        "thesis": "Debt-laden company in cyclical industry. Vulnerable to rate pressure.",
        "correlation": "Moves with consumer credit conditions"
    },
    # TIER C - Moderate Weakness
    {
        "symbol": "SNAP",
        "name": "Snap Inc",
        "tier": "C",
        "weakness_score": 6,
        "category": "Social Media",
        "fundamental_issues": [
            "Ad revenue pressure from TikTok",
            "User growth slowing",
            "Unprofitable despite years of operation",
            "Competition crushing margins"
        ],
        "entry_zone": {"low": 10.00, "high": 14.00},
        "targets": {"tp1": 8.00, "tp2": 6.00, "tp3": 4.00},
        "stop_loss": 16.00,
        "catalyst": "Ad spending cuts, user metrics miss",
        "thesis": "Social media loser. TikTok eating lunch. No moat.",
        "correlation": "Digital ad spending trends"
    },
    {
        "symbol": "HOOD",
        "name": "Robinhood Markets",
        "tier": "C",
        "weakness_score": 6,
        "category": "Fintech",
        "fundamental_issues": [
            "Revenue highly dependent on retail trading activity",
            "PFOF business model under regulatory threat",
            "User base shrinking from peak",
            "Competition from established brokers"
        ],
        "entry_zone": {"low": 16.00, "high": 22.00},
        "targets": {"tp1": 12.00, "tp2": 9.00, "tp3": 6.00},
        "stop_loss": 26.00,
        "catalyst": "SEC PFOF ruling, trading volume decline, crypto winter",
        "thesis": "Meme stock broker losing users. PFOF ban would be devastating.",
        "correlation": "Retail trading sentiment, crypto activity"
    },
    {
        "symbol": "LCID",
        "name": "Lucid Group",
        "tier": "C",
        "weakness_score": 7,
        "category": "EV/Auto",
        "fundamental_issues": [
            "Production constantly missing targets",
            "Cash burn rate alarming",
            "Valuation disconnected from deliveries",
            "Competition from Tesla, legacy auto"
        ],
        "entry_zone": {"low": 3.00, "high": 4.50},
        "targets": {"tp1": 2.00, "tp2": 1.50, "tp3": 1.00},
        "stop_loss": 5.50,
        "catalyst": "Production miss, capital raise, reservation cancellations",
        "thesis": "EV also-ran burning cash. Will need more dilution.",
        "correlation": "EV sentiment, Tesla moves"
    }
]

# Protected holdings - NEVER short these
PROTECTED_LONGS = ["IREN", "CLSK", "CIFR", "XAUUSD", "MGC", "GLD", "IAU", "GOLD"]


@app.get("/agents/spy/targets")
async def get_spy_targets():
    """Get SPY's current weakness hunting targets with entry/exit levels"""
    return {
        "agent": "spy",
        "role": "EXTERNAL Weakness Hunter",
        "principle": "NEVER short our longs. Hunt EXTERNAL fundamental weakness.",
        "asymmetry_edge": "Weak assets drop MORE on bad days, recover LESS on good days",
        "protected_longs": PROTECTED_LONGS,
        "weakness_targets": SPY_WEAKNESS_TARGETS,
        "target_count": len(SPY_WEAKNESS_TARGETS),
        "tiers_explained": {
            "S": "Catastrophic weakness - drops 3-5x market moves",
            "A": "Severe weakness - drops 2-3x market moves", 
            "B": "Significant weakness - drops 1.5-2x market moves",
            "C": "Moderate weakness - drops 1.2-1.5x market moves"
        },
        "usage": "Screenshot any target chart, then analyze with all agents for entry timing",
        "updated": datetime.now().isoformat()
    }


@app.get("/agents/spy/targets/{symbol}")
async def get_spy_target_detail(symbol: str):
    """Get detailed info on a specific SPY target"""
    symbol = symbol.upper()
    
    # Check if it's protected
    if symbol in PROTECTED_LONGS:
        return {
            "error": "PROTECTED",
            "symbol": symbol,
            "message": f"{symbol} is on our LONG book. SPY does NOT short our holdings.",
            "action": "Look for this in Ghost's analysis instead"
        }
    
    # Find target
    for target in SPY_WEAKNESS_TARGETS:
        if target["symbol"] == symbol:
            return {
                "symbol": symbol,
                "target": target,
                "action": f"Screenshot {symbol} chart and run full analysis to confirm entry timing"
            }
    
    return {
        "symbol": symbol,
        "status": "not_tracked",
        "message": f"{symbol} not currently in SPY weakness watchlist. May still be shortable - run analysis to evaluate."
    }


@app.post("/agents/spy/add_target")
async def add_spy_target(
    symbol: str = Form(...),
    name: str = Form(...),
    tier: str = Form("C"),
    weakness_score: int = Form(5),
    category: str = Form(""),
    issues: str = Form(""),  # Comma-separated
    entry_low: float = Form(0),
    entry_high: float = Form(0),
    tp1: float = Form(0),
    tp2: float = Form(0),
    stop: float = Form(0),
    thesis: str = Form("")
):
    """Add a new target to SPY's watchlist"""
    symbol = symbol.upper()
    
    # Check if protected
    if symbol in PROTECTED_LONGS:
        return {"error": f"{symbol} is protected - cannot add to short watchlist"}
    
    new_target = {
        "symbol": symbol,
        "name": name,
        "tier": tier.upper(),
        "weakness_score": weakness_score,
        "category": category,
        "fundamental_issues": [i.strip() for i in issues.split(",") if i.strip()],
        "entry_zone": {"low": entry_low, "high": entry_high},
        "targets": {"tp1": tp1, "tp2": tp2},
        "stop_loss": stop,
        "thesis": thesis,
        "added": datetime.now().isoformat()
    }
    
    # Check if already exists
    for i, target in enumerate(SPY_WEAKNESS_TARGETS):
        if target["symbol"] == symbol:
            SPY_WEAKNESS_TARGETS[i] = new_target
            return {"status": "updated", "target": new_target}
    
    SPY_WEAKNESS_TARGETS.append(new_target)
    return {"status": "added", "target": new_target}


# ============================================================================
# AUTO-FETCH MARKET DATA - No screenshots needed!
# ============================================================================

MARKET_DATA_APIS = {
    "gold": "http://localhost:8700/api/neo/gold-forex",
    "iren": "http://localhost:8600/api/iren/analysis",
    "iren_price": "http://localhost:8600/api/iren/price",
}


async def fetch_market_context(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """Fetch live market data from internal APIs - no screenshots needed!"""
    context = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "data_sources": [],
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Gold/Forex data
        try:
            resp = await client.get("http://localhost:8700/api/neo/gold-forex")
            if resp.status_code == 200:
                data = resp.json()
                context["gold"] = data.get("gold_status", {})
                context["forex_signals"] = data.get("forex_signals", [])[:3]  # Top 3
                context["data_sources"].append("gold-forex-api")
        except Exception as e:
            logger.warning(f"Gold API fetch failed: {e}")
        
        # If IREN specifically
        if symbol.upper() == "IREN":
            try:
                resp = await client.get("http://localhost:8600/api/iren/price")
                if resp.status_code == 200:
                    context["iren"] = resp.json()
                    context["data_sources"].append("iren-api")
            except Exception as e:
                logger.warning(f"IREN API fetch failed: {e}")
        
        # Try to get OHLCV from yfinance backup
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol if symbol != "XAUUSD" else "GC=F")
            hist = ticker.history(period="5d", interval="1h")
            if not hist.empty:
                latest = hist.iloc[-1]
                context["price"] = {
                    "current": round(float(latest["Close"]), 2),
                    "open": round(float(latest["Open"]), 2),
                    "high": round(float(latest["High"]), 2),
                    "low": round(float(latest["Low"]), 2),
                    "change_pct": round(((latest["Close"] - hist.iloc[0]["Close"]) / hist.iloc[0]["Close"]) * 100, 2)
                }
                
                # Calculate RSI-2
                closes = hist["Close"].values
                if len(closes) >= 3:
                    gains = []
                    losses = []
                    for i in range(1, min(3, len(closes))):
                        diff = closes[i] - closes[i-1]
                        if diff > 0:
                            gains.append(diff)
                            losses.append(0)
                        else:
                            gains.append(0)
                            losses.append(abs(diff))
                    avg_gain = sum(gains) / len(gains) if gains else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0.001
                    rs = avg_gain / avg_loss
                    rsi2 = round(100 - (100 / (1 + rs)), 1)
                    context["price"]["rsi2"] = rsi2
                
                context["data_sources"].append("yfinance")
        except Exception as e:
            logger.warning(f"YFinance fetch failed: {e}")
    
    return context


@app.get("/agents/market-context/{symbol}")
async def get_market_context(symbol: str = "XAUUSD"):
    """Get live market context for a symbol - agents can use this instead of screenshots"""
    return await fetch_market_context(symbol)


@app.post("/agents/analyze-auto")
async def analyze_auto(
    symbol: str = Form("XAUUSD"),
    user_context: str = Form(""),
    defcon: int = Form(3),
    agents: str = Form("ghost,casper,neo,fomo,spy"),
):
    """
    AUTO-ANALYSIS - No screenshots needed!
    Fetches live market data and runs agent analysis.
    
    This is the FAST path when you don't have charts to upload.
    Agents receive structured data: price, RSI, trend, levels.
    """
    
    # Fetch live market data
    market_data = await fetch_market_context(symbol)
    
    # Build context from live data
    auto_context = f"""
## LIVE MARKET DATA (Auto-Fetched)
Symbol: {symbol}
Timestamp: {market_data.get('timestamp')}
Data Sources: {', '.join(market_data.get('data_sources', []))}

"""
    
    if "gold" in market_data:
        gold = market_data["gold"]
        auto_context += f"""
### GOLD STATUS
- Price: ${gold.get('price', 'N/A')}
- Direction: {gold.get('direction', 'N/A')}
- Strength: {gold.get('strength', 'N/A')}%
- Volatility: {gold.get('volatility', 'N/A')}
- RSI: {gold.get('rsi', 'N/A')}
- 1H Change: {gold.get('change_1h', 'N/A')}
- 4H Change: {gold.get('change_4h', 'N/A')}
- 24H Change: {gold.get('change_24h', 'N/A')}
- Near Key Level: {gold.get('near_key_level', 'N/A')} ({gold.get('key_level', 'N/A')})
"""

    if "price" in market_data:
        price = market_data["price"]
        auto_context += f"""
### PRICE DATA
- Current: ${price.get('current', 'N/A')}
- Open: ${price.get('open', 'N/A')}
- High: ${price.get('high', 'N/A')}
- Low: ${price.get('low', 'N/A')}
- Change: {price.get('change_pct', 'N/A')}%
- RSI(2): {price.get('rsi2', 'N/A')}
"""
    
    if "iren" in market_data:
        iren = market_data["iren"]
        auto_context += f"""
### IREN DATA
{json.dumps(iren, indent=2)}
"""
    
    if user_context:
        auto_context += f"""
### USER CONTEXT
{user_context}
"""
    
    auto_context += f"""
### DEFCON LEVEL: {defcon}

NOTE: This analysis uses STRUCTURED DATA, not chart images.
The data is real-time from internal APIs.
"""
    
    # Run analysis with this auto-generated context
    agent_list = [a.strip().lower() for a in agents.split(",")]
    
    tasks = []
    for agent_name in agent_list:
        if agent_name in AGENT_PROMPTS:
            tasks.append(run_agent(
                agent_name,
                auto_context,
                None,  # No image
                None,
                defcon,
                symbol,
                market_data.get("price", {}).get("current", 0) or market_data.get("gold", {}).get("price", 0)
            ))
    
    start_time = datetime.now()
    results = await asyncio.gather(*tasks)
    
    agent_results = {}
    for result in results:
        if "agent" in result:
            agent_results[result["agent"].lower()] = result
    
    # Commander synthesis
    commander_result = await run_commander(agent_results, auto_context)
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    
    return {
        "mode": "auto-fetch",
        "market_data": market_data,
        "agents": agent_results,
        "commander_directive": commander_result,
        "defcon_input": defcon,
        "symbol": symbol,
        "total_elapsed_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Trading Agents API on port 8890...")
    uvicorn.run(app, host="0.0.0.0", port=8890)
