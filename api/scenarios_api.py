"""
Scenario Projector API
======================
Generates real-time market scenarios using LIVE data from yfinance.
Uses Claude for analysis - NO PLACEHOLDER DATA.
"""

import os
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import APIRouter, Form, HTTPException
import yfinance as yf
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScenariosAPI")

router = APIRouter(prefix="/scenarios", tags=["scenarios"])

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# If using Ollama as a proxy, use a compatible model
logger.info(f"Scenarios API: ANTHROPIC_API_KEY={ANTHROPIC_API_KEY}, ANTHROPIC_BASE_URL={ANTHROPIC_BASE_URL}")
if ANTHROPIC_BASE_URL == "http://localhost:11434" or ANTHROPIC_API_KEY == "ollama":
    CLAUDE_MODEL = "llama3.1:8b"  # Ollama model
    USE_OLLAMA = True
    logger.info(f"Scenarios API: Using OLLAMA with model {CLAUDE_MODEL}")
else:
    CLAUDE_MODEL = "claude-sonnet-4-20250514"
    USE_OLLAMA = False
    logger.info(f"Scenarios API: Using Anthropic API with model {CLAUDE_MODEL}")


async def get_real_market_data(symbol: str) -> Dict:
    """
    Fetch REAL market data from yfinance.
    Returns current price, support, resistance, and recent price action.
    """
    
    # Normalize symbol for yfinance
    yf_symbol = symbol.upper()
    if yf_symbol == "XAUUSD":
        yf_symbol = "GC=F"  # Gold futures
    elif yf_symbol == "XAGUSD":
        yf_symbol = "SI=F"  # Silver futures
    elif yf_symbol == "BTCUSD":
        yf_symbol = "BTC-USD"
    
    try:
        ticker = yf.Ticker(yf_symbol)
        
        # Get daily data for support/resistance
        daily = ticker.history(period="3mo", interval="1d")
        
        # Get recent intraday data
        hourly = ticker.history(period="5d", interval="1h")
        
        if daily.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        current_price = float(daily['Close'].iloc[-1])
        
        # Calculate support and resistance from recent data
        recent_highs = daily['High'].tail(20)
        recent_lows = daily['Low'].tail(20)
        
        # Simple S/R based on recent swing points
        resistance = float(recent_highs.max())
        support = float(recent_lows.min())
        
        # Find more precise levels
        pivot = (daily['High'].iloc[-1] + daily['Low'].iloc[-1] + daily['Close'].iloc[-1]) / 3
        r1 = (2 * pivot) - daily['Low'].iloc[-1]
        s1 = (2 * pivot) - daily['High'].iloc[-1]
        
        # Price changes
        change_1d = ((current_price - daily['Close'].iloc[-2]) / daily['Close'].iloc[-2]) * 100 if len(daily) > 1 else 0
        change_5d = ((current_price - daily['Close'].iloc[-6]) / daily['Close'].iloc[-6]) * 100 if len(daily) > 5 else 0
        change_20d = ((current_price - daily['Close'].iloc[-21]) / daily['Close'].iloc[-21]) * 100 if len(daily) > 20 else 0
        
        # Volume analysis
        avg_volume = daily['Volume'].tail(20).mean()
        latest_volume = daily['Volume'].iloc[-1]
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Recent candle patterns (basic)
        last_candle = daily.iloc[-1]
        prev_candle = daily.iloc[-2] if len(daily) > 1 else last_candle
        
        body = last_candle['Close'] - last_candle['Open']
        upper_wick = last_candle['High'] - max(last_candle['Close'], last_candle['Open'])
        lower_wick = min(last_candle['Close'], last_candle['Open']) - last_candle['Low']
        
        candle_type = "BULLISH" if body > 0 else "BEARISH" if body < 0 else "DOJI"
        
        # Determine bias based on data
        if change_5d > 3 and current_price > float(r1):
            bias = "BULLISH"
        elif change_5d < -3 and current_price < float(s1):
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        
        return {
            'symbol': symbol,
            'yf_symbol': yf_symbol,
            'current_price': round(current_price, 2),
            'support': round(float(s1), 2),
            'resistance': round(float(r1), 2),
            'swing_low': round(support, 2),
            'swing_high': round(resistance, 2),
            'pivot': round(float(pivot), 2),
            'change_1d': round(change_1d, 2),
            'change_5d': round(change_5d, 2),
            'change_20d': round(change_20d, 2),
            'volume_ratio': round(volume_ratio, 2),
            'last_candle_type': candle_type,
            'bias': bias,
            'fetched_at': datetime.now().isoformat(),
            'data_source': 'yfinance',
        }
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


async def generate_scenarios_with_claude(market_data: Dict, context: str = "", chart_base64: str = None) -> Dict:
    """
    Use Claude to generate realistic scenarios based on REAL market data.
    """
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")
    
    symbol = market_data['symbol']
    current_price = market_data['current_price']
    support = market_data['support']
    resistance = market_data['resistance']
    swing_low = market_data['swing_low']
    swing_high = market_data['swing_high']
    
    # Calculate realistic price ranges based on asset type
    # IREN ~$50, Gold ~$5000, SPY ~$500, BTC ~$100000
    if current_price > 1000:
        # Gold/BTC - use 1-2% ranges
        price_range_pct = 0.015
    else:
        # Stocks - use 2-5% ranges
        price_range_pct = 0.03
    
    # Pre-calculate realistic price levels based on ACTUAL current price
    entry_low = round(current_price * 0.99, 2)
    entry_high = round(current_price * 1.01, 2)
    stop_loss = round(current_price * 0.97, 2)
    tp1 = round(current_price * 1.03, 2)
    tp2 = round(current_price * 1.05, 2)
    
    logger.info(f"Generating scenarios for {symbol} at ${current_price:.2f} (S: ${support:.2f}, R: ${resistance:.2f})")
    
    prompt = f"""You are an expert technical analyst. Generate realistic trading scenarios for {symbol}.

=== CRITICAL: USE THESE EXACT PRICES ===
CURRENT PRICE: ${current_price:.2f}
SUPPORT LEVEL: ${support:.2f}  
RESISTANCE LEVEL: ${resistance:.2f}
SWING LOW: ${swing_low:.2f}
SWING HIGH: ${swing_high:.2f}

This is a {"stock trading around $" + str(int(current_price)) if current_price < 500 else "commodity/crypto trading around $" + str(int(current_price))}.

RECENT PERFORMANCE:
- 1-Day Change: {market_data['change_1d']:+.2f}%
- 5-Day Change: {market_data['change_5d']:+.2f}%
- 20-Day Change: {market_data['change_20d']:+.2f}%
- Volume Ratio: {market_data['volume_ratio']:.2f}x average
- Last Candle: {market_data['last_candle_type']}
- Current Bias: {market_data['bias']}

{f"CONTEXT: {context}" if context else ""}

STRICT RULES:
1. ALL price targets MUST be within 5% of ${current_price:.2f}
2. For {symbol}, valid price range is ${current_price * 0.95:.2f} to ${current_price * 1.05:.2f}
3. DO NOT use prices from other assets (e.g., don't use gold prices for stocks)
4. Probabilities must add up to 100%

Generate exactly 4 scenarios. Return ONLY valid JSON, no extra text:

{{
  "current_price": {current_price},
  "current_bias": "{market_data['bias']}",
  "analysis_timestamp": "{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}",
  "key_support": {support},
  "key_resistance": {resistance},
  "scenarios": [
    {{
      "name": "Scenario Name",
      "probability": 30,
      "bias": "BULLISH|BEARISH|NEUTRAL",
      "description": "Description referencing ${current_price:.2f}",
      "key_trigger": "Price action around ${current_price:.2f}",
      "invalidation": "Break below/above specific level",
      "price_path": [
        {{"session": "Asia", "price_low": {round(current_price * 0.99, 2)}, "price_high": {round(current_price * 1.01, 2)}}},
        {{"session": "London", "price_low": {round(current_price * 0.98, 2)}, "price_high": {round(current_price * 1.02, 2)}}},
        {{"session": "NY", "price_low": {round(current_price * 0.97, 2)}, "price_high": {round(current_price * 1.03, 2)}}}
      ],
      "key_levels": {{
        "entry_zone": [{entry_low}, {entry_high}],
        "stop_loss": {stop_loss},
        "take_profit_1": {tp1},
        "take_profit_2": {tp2}
      }},
      "action_plan": "Action based on ${current_price:.2f}",
      "risk_reward": "1:2"
    }}
  ]
}}

Remember: {symbol} is currently at ${current_price:.2f} - ALL prices in your response must be close to this value."""

    try:
        async with httpx.AsyncClient() as client:
            if USE_OLLAMA:
                # Use Ollama API format
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": CLAUDE_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                    timeout=120,
                )
            else:
                # Use Anthropic API format
                messages = [{"role": "user", "content": prompt}]
                
                # If chart image provided, include it
                if chart_base64:
                    messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": chart_base64.split(',')[1] if ',' in chart_base64 else chart_base64
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                
                response = await client.post(
                    f"{ANTHROPIC_BASE_URL}/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": CLAUDE_MODEL,
                        "max_tokens": 4000,
                        "messages": messages,
                    },
                    timeout=60,
                )
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=500, detail="Failed to generate scenarios")
            
            result = response.json()
            
            # Extract content based on API type
            if USE_OLLAMA:
                content = result.get('response', '')
            else:
                content = result['content'][0]['text']
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                scenarios_data = json.loads(json_match.group())
                
                # VALIDATION: Force correct prices if Claude hallucinated
                real_price = market_data['current_price']
                returned_price = scenarios_data.get('current_price', 0)
                
                # Check if Claude's price is WAY off (more than 50% different)
                if returned_price > 0 and abs(returned_price - real_price) / real_price > 0.5:
                    logger.warning(f"Claude hallucinated price ${returned_price} for {symbol}, forcing to ${real_price}")
                    
                    # Fix all prices in the response
                    price_ratio = real_price / returned_price if returned_price > 0 else 1
                    
                    scenarios_data['current_price'] = real_price
                    scenarios_data['key_support'] = market_data['support']
                    scenarios_data['key_resistance'] = market_data['resistance']
                    
                    # Fix each scenario's prices
                    for scenario in scenarios_data.get('scenarios', []):
                        # Fix price path
                        for path in scenario.get('price_path', []):
                            if 'price_low' in path:
                                path['price_low'] = round(path['price_low'] * price_ratio, 2)
                            if 'price_high' in path:
                                path['price_high'] = round(path['price_high'] * price_ratio, 2)
                        
                        # Fix key levels
                        levels = scenario.get('key_levels', {})
                        if 'entry_zone' in levels and isinstance(levels['entry_zone'], list):
                            levels['entry_zone'] = [round(p * price_ratio, 2) for p in levels['entry_zone']]
                        if 'stop_loss' in levels:
                            levels['stop_loss'] = round(levels['stop_loss'] * price_ratio, 2)
                        if 'take_profit_1' in levels:
                            levels['take_profit_1'] = round(levels['take_profit_1'] * price_ratio, 2)
                        if 'take_profit_2' in levels:
                            levels['take_profit_2'] = round(levels['take_profit_2'] * price_ratio, 2)
                    
                    logger.info(f"Fixed hallucinated prices for {symbol}")
                
                return scenarios_data
            else:
                logger.error(f"Could not parse JSON from response: {content[:500]}")
                raise HTTPException(status_code=500, detail="Failed to parse scenarios")
                
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse scenario data")
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_scenarios_from_chart(
    symbol: str = Form("XAUUSD"),
    context: str = Form(""),
    image_base64: str = Form(None),
):
    """
    Generate scenarios from uploaded chart image + REAL market data.
    
    - Fetches live prices from yfinance
    - Analyzes chart with Claude Vision
    - Returns realistic scenarios
    """
    
    # Get REAL market data
    market_data = await get_real_market_data(symbol)
    
    # Generate scenarios with Claude
    scenarios = await generate_scenarios_with_claude(
        market_data=market_data,
        context=context,
        chart_base64=image_base64,
    )
    
    return {
        "status": "success",
        "data": scenarios,
        "market_data": market_data,
    }


@router.post("/generate-auto")
async def generate_scenarios_auto(
    symbol: str = Form("XAUUSD"),
    context: str = Form(""),
):
    """
    Generate scenarios using ONLY real market data (no chart upload needed).
    
    - Fetches live prices from yfinance
    - Generates scenarios based on current market conditions
    """
    
    # Get REAL market data
    market_data = await get_real_market_data(symbol)
    
    # Generate scenarios with Claude (no image)
    scenarios = await generate_scenarios_with_claude(
        market_data=market_data,
        context=context,
    )
    
    return {
        "status": "success",
        "data": scenarios,
        "market_data": market_data,
    }


@router.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """
    Get real-time market data for a symbol.
    """
    
    market_data = await get_real_market_data(symbol)
    return market_data


@router.get("/health")
async def scenarios_health():
    """Health check for scenarios API."""
    
    return {
        "status": "healthy",
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "data_source": "yfinance (REAL DATA)",
        "timestamp": datetime.now().isoformat(),
    }
