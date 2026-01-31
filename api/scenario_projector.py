"""
Scenario Projector API - Pre-visualize tomorrow's battlefield
Generates 3-4 probable scenarios for next trading session with entry/exit levels.

Port: Integrated with trading_agents.py (8890)
"""

import asyncio
import httpx
import json
import base64
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScenarioProjector")

router = APIRouter(prefix="/scenarios", tags=["scenarios"])

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# Detect Ollama mode
USE_OLLAMA = (ANTHROPIC_BASE_URL == "http://localhost:11434" or ANTHROPIC_API_KEY == "ollama")
OLLAMA_MODEL = "llama3.1:8b" if USE_OLLAMA else "claude-sonnet-4-20250514"
logger.info(f"Scenario Projector: USE_OLLAMA={USE_OLLAMA}, model={OLLAMA_MODEL}")

SCENARIO_PROMPT = """You are a professional technical analyst creating scenario projections for traders.

Given the current chart, create 3-4 probable scenarios for the NEXT trading session.

For EACH scenario, provide:

1. **name**: Short descriptive name (e.g., "V-Recovery", "Breakdown", "Bull Flag Breakout")
2. **probability**: Your estimate as integer (all should sum to ~100%)
3. **bias**: "BULLISH", "BEARISH", or "NEUTRAL"
4. **description**: 1-2 sentence description of what happens
5. **key_trigger**: What confirms this scenario is playing out
6. **invalidation**: What proves this scenario wrong
7. **price_path**: Array of expected price progression
   - Format: [{"session": "Asia", "price_low": X, "price_high": Y}, ...]
   - Sessions: "Asia", "London", "NY"
8. **key_levels**:
   - entry_zone: [low, high] array
   - stop_loss: number
   - take_profit_1: number
   - take_profit_2: number (optional)
9. **action_plan**: Specific trading instructions
10. **risk_reward**: Approximate R:R ratio

CRITICAL RULES:
- Be SPECIFIC with price levels based on actual chart structure
- Scenarios should cover bullish, bearish, AND sideways possibilities
- Probabilities must be realistic based on chart context
- Each scenario must have a clear trigger and invalidation
- Focus on the NEXT 24 hours of trading

Output as valid JSON with this exact structure:
{
  "current_price": number,
  "current_bias": "string",
  "analysis_timestamp": "ISO string",
  "key_support": number,
  "key_resistance": number,
  "scenarios": [
    {
      "name": "string",
      "probability": number,
      "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
      "description": "string",
      "key_trigger": "string",
      "invalidation": "string",
      "price_path": [
        {"session": "Asia", "price_low": number, "price_high": number},
        {"session": "London", "price_low": number, "price_high": number},
        {"session": "NY", "price_low": number, "price_high": number}
      ],
      "key_levels": {
        "entry_zone": [number, number],
        "stop_loss": number,
        "take_profit_1": number,
        "take_profit_2": number
      },
      "action_plan": "string",
      "risk_reward": "string"
    }
  ]
}"""


class ScenarioRequest(BaseModel):
    symbol: str = "XAUUSD"
    current_price: Optional[float] = None
    context: str = ""


@router.post("/generate")
async def generate_scenarios(
    file: UploadFile = File(None),
    image_base64: str = Form(None),
    symbol: str = Form("XAUUSD"),
    current_price: float = Form(None),
    context: str = Form(""),
):
    """Generate probable scenarios for next trading session from chart image."""
    
    # Get image
    if file:
        image_data = await file.read()
        img_b64 = base64.b64encode(image_data).decode()
    elif image_base64:
        # Handle data URL format
        if image_base64.startswith("data:"):
            img_b64 = image_base64.split(",")[1]
        else:
            img_b64 = image_base64
    else:
        return {"error": "No chart provided. Upload a chart image for scenario analysis."}
    
    # Build request content
    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64}
        },
        {
            "type": "text",
            "text": f"""Analyze this {symbol} chart and generate scenarios for the NEXT trading session.

Current price: {current_price or 'determine from chart'}
Additional context: {context or 'None provided'}
Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC

Generate 3-4 probable scenarios. Be specific with price levels based on the chart structure you see.
Focus on actionable trading plans for each scenario."""
        }
    ]
    
    # Call Claude
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                    "system": SCENARIO_PROMPT,
                    "messages": [{"role": "user", "content": user_content}],
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["content"][0]["text"]
                
                # Parse JSON from response
                try:
                    if "```json" in text:
                        json_str = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        json_str = text.split("```")[1].split("```")[0]
                    elif "{" in text:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        json_str = text[start:end]
                    else:
                        json_str = text
                    
                    scenarios_data = json.loads(json_str)
                    
                    # Add metadata
                    scenarios_data["symbol"] = symbol
                    scenarios_data["generated_at"] = datetime.now().isoformat()
                    
                    return {
                        "status": "success",
                        "data": scenarios_data
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}")
                    return {
                        "status": "success",
                        "raw_text": text,
                        "parse_error": str(e)
                    }
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": f"AI API returned {response.status_code}"}
                
    except Exception as e:
        logger.error(f"Scenario generation failed: {e}")
        return {"error": str(e)}


@router.post("/generate-auto")
async def generate_scenarios_auto(
    symbol: str = Form("XAUUSD"),
    context: str = Form(""),
):
    """Generate scenarios using live market data (no chart needed)."""
    
    # Fetch live market data
    market_context = ""
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get gold data
            resp = await client.get("http://localhost:8700/api/neo/gold-forex")
            if resp.status_code == 200:
                data = resp.json()
                gold = data.get("gold_status", {})
                market_context += f"""
LIVE MARKET DATA:
- Current Price: ${gold.get('price', 'N/A')}
- Direction: {gold.get('direction', 'N/A')}
- Strength: {gold.get('strength', 'N/A')}%
- Volatility: {gold.get('volatility', 'N/A')}
- RSI: {gold.get('rsi', 'N/A')}
- 1H Change: {gold.get('change_1h', 'N/A')}
- 4H Change: {gold.get('change_4h', 'N/A')}
- 24H Change: {gold.get('change_24h', 'N/A')}
- Key Level: ${gold.get('key_level', 'N/A')}
"""
                current_price = gold.get('price')
    except Exception as e:
        logger.warning(f"Failed to fetch market data: {e}")
        current_price = None
    
    # Generate scenarios from data (text-only, no image)
    user_content = f"""Based on the following market data, generate 3-4 probable scenarios for the NEXT trading session of {symbol}.

{market_context}

Additional context: {context or 'None'}
Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC

Generate realistic scenarios with specific price levels. Cover bullish, bearish, and consolidation possibilities."""
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if USE_OLLAMA:
                # Use Ollama API
                full_prompt = f"{SCENARIO_PROMPT}\n\n{user_content}"
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": full_prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
            else:
                # Use Anthropic API
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 4000,
                        "system": SCENARIO_PROMPT,
                        "messages": [{"role": "user", "content": user_content}],
                    },
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response") if USE_OLLAMA else result["content"][0]["text"]
                
                try:
                    if "```json" in text:
                        json_str = text.split("```json")[1].split("```")[0]
                    elif "{" in text:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        json_str = text[start:end]
                    else:
                        json_str = text
                    
                    scenarios_data = json.loads(json_str)
                    scenarios_data["symbol"] = symbol
                    scenarios_data["generated_at"] = datetime.now().isoformat()
                    scenarios_data["mode"] = "auto-fetch"
                    
                    return {"status": "success", "data": scenarios_data}
                except:
                    return {"status": "success", "raw_text": text}
            
            return {"error": f"API returned {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}


@router.get("/symbols")
async def get_supported_symbols():
    """Get list of supported symbols for scenario generation."""
    return {
        "symbols": [
            {"id": "XAUUSD", "name": "Gold Spot", "type": "commodity"},
            {"id": "IREN", "name": "Iris Energy", "type": "stock"},
            {"id": "BTCUSD", "name": "Bitcoin", "type": "crypto"},
            {"id": "SPY", "name": "S&P 500 ETF", "type": "etf"},
            {"id": "GC=F", "name": "Gold Futures", "type": "futures"},
        ]
    }
