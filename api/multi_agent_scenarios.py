"""
Multi-Agent Scenario Projector
==============================
Three distinct perspectives for scenario analysis:
1. QUANT - Pure technical, price action only
2. NEO - Pattern recognition, institutional flow, proven strategies
3. CLAUDIA - Fundamental research, macro context, catalysts

Each agent uses different knowledge sources and reasoning.
Predictions are logged for accuracy tracking.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException
import yfinance as yf
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgentScenarios")

router = APIRouter(prefix="/scenarios/multi", tags=["multi-agent-scenarios"])

# Paths to knowledge bases
KNOWLEDGE_BASE = Path("/home/jbot/trading_ai")
NEO_KNOWLEDGE = KNOWLEDGE_BASE / "neo" / "knowledge"
NEO_SIGNALS = KNOWLEDGE_BASE / "neo" / "signals"
CLAUDIA_RESEARCH = KNOWLEDGE_BASE / "claudia" / "research"
PREDICTION_LOG = KNOWLEDGE_BASE / "neo" / "prediction_data" / "scenario_predictions.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


async def get_market_data(symbol: str) -> Dict:
    """Fetch real market data from yfinance."""
    yf_symbol = symbol.upper()
    if yf_symbol == "XAUUSD":
        yf_symbol = "GC=F"
    elif yf_symbol == "BTCUSD":
        yf_symbol = "BTC-USD"
    
    try:
        ticker = yf.Ticker(yf_symbol)
        daily = ticker.history(period="3mo", interval="1d")
        
        if daily.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        current_price = float(daily['Close'].iloc[-1])
        
        # Calculate levels
        recent_highs = daily['High'].tail(20)
        recent_lows = daily['Low'].tail(20)
        
        pivot = (daily['High'].iloc[-1] + daily['Low'].iloc[-1] + daily['Close'].iloc[-1]) / 3
        r1 = (2 * pivot) - daily['Low'].iloc[-1]
        s1 = (2 * pivot) - daily['High'].iloc[-1]
        
        # RSI calculation
        delta = daily['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Changes
        change_1d = ((current_price - daily['Close'].iloc[-2]) / daily['Close'].iloc[-2]) * 100 if len(daily) > 1 else 0
        change_5d = ((current_price - daily['Close'].iloc[-6]) / daily['Close'].iloc[-6]) * 100 if len(daily) > 5 else 0
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'support': round(float(s1), 2),
            'resistance': round(float(r1), 2),
            'swing_low': round(float(recent_lows.min()), 2),
            'swing_high': round(float(recent_highs.max()), 2),
            'rsi': round(rsi, 1),
            'change_1d': round(change_1d, 2),
            'change_5d': round(change_5d, 2),
            'volume_ratio': round(daily['Volume'].iloc[-1] / daily['Volume'].tail(20).mean(), 2),
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_neo_knowledge(symbol: str) -> Dict:
    """Load Neo's knowledge base - proven strategies and patterns."""
    knowledge = {
        'proven_strategies': {},
        'recent_patterns': [],
        'institutional_rules': []
    }
    
    # Load proven strategies
    strategies_file = NEO_KNOWLEDGE / "proven_strategies.json"
    if strategies_file.exists():
        with open(strategies_file) as f:
            strategies = json.load(f)
            # Extract relevant strategies
            knowledge['proven_strategies'] = {
                'rsi2': strategies.get('rsi2_connors', {}),
                'turtle': strategies.get('turtle_traders', {}),
                'stop_hunt': strategies.get('stop_hunt_fade', {}),
            }
    
    # Load recent pattern alerts
    alerts_file = NEO_SIGNALS / "pattern_alerts.json"
    if alerts_file.exists():
        with open(alerts_file) as f:
            alerts = json.load(f)
            knowledge['recent_patterns'] = alerts.get('alerts', [])[:5]  # Last 5
    
    # Institutional rules
    knowledge['institutional_rules'] = [
        "Stop hunts occur at obvious S/R levels",
        "First moves after news are usually wrong",
        "Volume confirms breakouts, lack of volume suggests fakeout",
        "Round numbers ($50, $100) attract stop clusters",
    ]
    
    return knowledge


def load_claudia_knowledge(symbol: str) -> Dict:
    """Load Claudia's research knowledge for the symbol."""
    knowledge = {
        'research_summary': '',
        'key_catalysts': [],
        'price_targets': {},
        'fundamental_thesis': ''
    }
    
    # Try to find symbol-specific research
    symbol_upper = symbol.upper()
    
    # Check for IREN-specific research
    if symbol_upper == "IREN":
        # Load institutional report
        report_files = list(CLAUDIA_RESEARCH.glob("IREN_*.md"))
        if report_files:
            latest = sorted(report_files)[-1]
            with open(latest) as f:
                content = f.read()
                # Extract key sections
                knowledge['research_summary'] = content[:2000]  # First 2000 chars
                
                # Extract key info
                if "Microsoft" in content:
                    knowledge['key_catalysts'].append("Microsoft AI Cloud $9.7B contract")
                if "GPU" in content:
                    knowledge['key_catalysts'].append("140,000 GPU deployment target")
                if "ARR" in content:
                    knowledge['key_catalysts'].append("$3.4B AI Cloud ARR target by end 2026")
                    
        # Load BTC miners research
        btc_research = CLAUDIA_RESEARCH / "btc_miners_hyperscaling" / "RESEARCH_REPORT.md"
        if btc_research.exists():
            with open(btc_research) as f:
                knowledge['fundamental_thesis'] = f.read()[:1500]
    
    # Load general portfolio strategy
    portfolio_file = CLAUDIA_RESEARCH / "complete_portfolio_strategy.json"
    if portfolio_file.exists():
        with open(portfolio_file) as f:
            portfolio = json.load(f)
            if symbol_upper in str(portfolio):
                knowledge['portfolio_context'] = "Symbol in active watchlist"
    
    return knowledge


async def generate_agent_scenario(
    agent: str,
    market_data: Dict,
    knowledge: Dict,
    context: str = ""
) -> Dict:
    """Generate scenario from specific agent's perspective."""
    
    symbol = market_data['symbol']
    price = market_data['current_price']
    
    # Build agent-specific prompts
    if agent == "quant":
        system_prompt = """You are QUANT, a pure technical analyst. You ONLY look at price action, 
support/resistance levels, and technical indicators. You ignore news, fundamentals, and emotions.
Your analysis is cold, mathematical, and probability-based."""
        
        data_context = f"""
PURE TECHNICAL DATA for {symbol}:
- Price: ${price}
- Support (S1): ${market_data['support']}
- Resistance (R1): ${market_data['resistance']}
- Swing Low: ${market_data['swing_low']}
- Swing High: ${market_data['swing_high']}
- RSI(14): {market_data['rsi']}
- 1D Change: {market_data['change_1d']}%
- 5D Change: {market_data['change_5d']}%
- Volume Ratio: {market_data['volume_ratio']}x avg

Analyze ONLY the numbers. No fundamentals, no news, pure price action."""

    elif agent == "neo":
        strategies = knowledge.get('proven_strategies', {})
        patterns = knowledge.get('recent_patterns', [])
        rules = knowledge.get('institutional_rules', [])
        
        system_prompt = """You are NEO, a pattern recognition specialist focused on institutional flow.
You look for stop hunts, liquidity sweeps, and proven trading patterns. You think like a market maker,
anticipating where retail stops are clustered and how smart money will exploit them."""
        
        data_context = f"""
MARKET DATA for {symbol}:
- Price: ${price}
- Support: ${market_data['support']}
- Resistance: ${market_data['resistance']}
- RSI(14): {market_data['rsi']}
- Volume Ratio: {market_data['volume_ratio']}x

PROVEN STRATEGY SIGNALS:
- RSI2 Strategy: RSI(2) under 10 = oversold bounce setup (88% win rate historically)
- Turtle System: 20-day breakout signals (38% win rate, 2.1 profit factor)
- Stop Hunt Pattern: Watch for spike through S/R then reversal (65% win rate)

INSTITUTIONAL RULES:
{chr(10).join(f'- {rule}' for rule in rules)}

RECENT PATTERN ALERTS:
{json.dumps(patterns[:3], indent=2) if patterns else 'No recent alerts'}

Think like a market maker. Where are the stops? What would smart money do?"""

    elif agent == "claudia":
        research = knowledge.get('research_summary', '')
        catalysts = knowledge.get('key_catalysts', [])
        thesis = knowledge.get('fundamental_thesis', '')
        
        system_prompt = """You are CLAUDIA, a fundamental research analyst. You focus on catalysts,
earnings, macro trends, and company-specific news. You understand that price follows fundamentals
in the medium term, even if technicals dominate short-term moves."""
        
        data_context = f"""
MARKET DATA for {symbol}:
- Price: ${price}
- 1D Change: {market_data['change_1d']}%
- 5D Change: {market_data['change_5d']}%

KEY CATALYSTS:
{chr(10).join(f'- {cat}' for cat in catalysts) if catalysts else '- No specific catalysts loaded'}

RESEARCH SUMMARY:
{research[:1500] if research else 'No specific research available for this symbol.'}

FUNDAMENTAL THESIS:
{thesis[:1000] if thesis else 'General market conditions apply.'}

Additional context: {context}

Focus on catalysts, news impact, and fundamental value. How should fundamentals drive price?"""

    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    
    # Generate scenarios with Ollama
    full_prompt = f"""{system_prompt}

{data_context}

Generate 3 scenarios for tomorrow's trading session. For each scenario:
1. Name (e.g., "Bullish Breakout", "Support Test", "Consolidation")
2. Probability (must sum to 100%)
3. Direction: BULLISH, BEARISH, or NEUTRAL
4. Your reasoning based on your expertise
5. Entry zone, stop loss, and target
6. What would confirm or invalidate this scenario

CRITICAL: Use prices close to ${price} (within 5%). Return ONLY valid JSON:

{{
  "agent": "{agent}",
  "symbol": "{symbol}",
  "current_price": {price},
  "conviction": "HIGH/MEDIUM/LOW",
  "primary_bias": "BULLISH/BEARISH/NEUTRAL",
  "reasoning": "Your main argument in 1-2 sentences",
  "scenarios": [
    {{
      "name": "Scenario Name",
      "probability": 40,
      "direction": "BULLISH",
      "reasoning": "Agent-specific reasoning",
      "entry_zone": [{round(price*0.99, 2)}, {round(price*1.01, 2)}],
      "stop_loss": {round(price*0.97, 2)},
      "target": {round(price*1.05, 2)},
      "confirmation": "What confirms this plays out",
      "invalidation": "What proves this wrong"
    }}
  ]
}}"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Ollama error: {response.status_code}")
            
            result = response.json()
            text = result.get('response', '{}')
            
            # Parse JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                scenarios = json.loads(json_match.group())
                scenarios['generated_at'] = datetime.now().isoformat()
                scenarios['knowledge_sources'] = list(knowledge.keys()) if knowledge else ['market_data_only']
                return scenarios
            else:
                logger.error(f"Failed to parse {agent} response: {text[:500]}")
                raise HTTPException(status_code=500, detail="Failed to parse agent response")
                
    except json.JSONDecodeError as e:
        logger.error(f"JSON error for {agent}: {e}")
        raise HTTPException(status_code=500, detail="JSON parse error")


def log_prediction(predictions: Dict):
    """Log predictions for accuracy tracking."""
    try:
        existing = []
        if PREDICTION_LOG.exists():
            with open(PREDICTION_LOG) as f:
                existing = json.load(f)
        
        predictions['logged_at'] = datetime.now().isoformat()
        predictions['verified'] = False
        existing.append(predictions)
        
        # Keep last 100 predictions
        existing = existing[-100:]
        
        with open(PREDICTION_LOG, 'w') as f:
            json.dump(existing, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


@router.post("/generate")
async def generate_multi_agent_scenarios(
    symbol: str = Form("IREN"),
    context: str = Form(""),
    agents: str = Form("quant,neo,claudia"),  # Comma-separated
):
    """
    Generate scenarios from multiple agent perspectives.
    
    Each agent analyzes the same market data but with different:
    - Knowledge sources
    - Reasoning frameworks
    - Biases and focus areas
    """
    
    # Get market data
    market_data = await get_market_data(symbol)
    
    agent_list = [a.strip().lower() for a in agents.split(",")]
    results = {
        'symbol': symbol,
        'market_data': market_data,
        'agents': {},
        'generated_at': datetime.now().isoformat(),
    }
    
    for agent in agent_list:
        try:
            # Load agent-specific knowledge
            if agent == "neo":
                knowledge = load_neo_knowledge(symbol)
            elif agent == "claudia":
                knowledge = load_claudia_knowledge(symbol)
            else:
                knowledge = {}
            
            # Generate scenario
            scenario = await generate_agent_scenario(
                agent=agent,
                market_data=market_data,
                knowledge=knowledge,
                context=context,
            )
            results['agents'][agent] = scenario
            
        except Exception as e:
            logger.error(f"Agent {agent} failed: {e}")
            results['agents'][agent] = {'error': str(e)}
    
    # Calculate consensus
    biases = []
    for agent, data in results['agents'].items():
        if 'primary_bias' in data:
            biases.append(data['primary_bias'])
    
    if biases:
        bullish = biases.count('BULLISH')
        bearish = biases.count('BEARISH')
        if bullish > bearish:
            results['consensus'] = 'BULLISH'
            results['consensus_strength'] = f"{bullish}/{len(biases)} agents"
        elif bearish > bullish:
            results['consensus'] = 'BEARISH'
            results['consensus_strength'] = f"{bearish}/{len(biases)} agents"
        else:
            results['consensus'] = 'MIXED'
            results['consensus_strength'] = 'No clear agreement'
    
    # Log for tracking
    log_prediction(results)
    
    return {'status': 'success', 'data': results}


@router.get("/agent/{agent}/{symbol}")
async def get_single_agent_scenario(agent: str, symbol: str, context: str = ""):
    """Get scenario from a specific agent."""
    
    market_data = await get_market_data(symbol)
    
    if agent == "neo":
        knowledge = load_neo_knowledge(symbol)
    elif agent == "claudia":
        knowledge = load_claudia_knowledge(symbol)
    else:
        knowledge = {}
    
    scenario = await generate_agent_scenario(
        agent=agent,
        market_data=market_data,
        knowledge=knowledge,
        context=context,
    )
    
    return {
        'status': 'success',
        'agent': agent,
        'symbol': symbol,
        'market_data': market_data,
        'scenario': scenario,
    }


@router.get("/predictions")
async def get_prediction_history(limit: int = 20):
    """Get logged predictions for accuracy tracking."""
    
    if not PREDICTION_LOG.exists():
        return {'predictions': [], 'count': 0}
    
    with open(PREDICTION_LOG) as f:
        predictions = json.load(f)
    
    return {
        'predictions': predictions[-limit:],
        'count': len(predictions),
        'verified_count': sum(1 for p in predictions if p.get('verified')),
    }


@router.post("/verify/{prediction_index}")
async def verify_prediction(
    prediction_index: int,
    actual_outcome: str = Form(...),
    actual_price: float = Form(...),
):
    """Verify a past prediction with actual outcome."""
    
    if not PREDICTION_LOG.exists():
        raise HTTPException(status_code=404, detail="No predictions logged")
    
    with open(PREDICTION_LOG) as f:
        predictions = json.load(f)
    
    if prediction_index >= len(predictions):
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    predictions[prediction_index]['verified'] = True
    predictions[prediction_index]['actual_outcome'] = actual_outcome
    predictions[prediction_index]['actual_price'] = actual_price
    predictions[prediction_index]['verified_at'] = datetime.now().isoformat()
    
    # Calculate which agent was closest
    agents_data = predictions[prediction_index].get('agents', {})
    for agent, data in agents_data.items():
        if data.get('primary_bias') == actual_outcome:
            data['was_correct'] = True
        else:
            data['was_correct'] = False
    
    with open(PREDICTION_LOG, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    return {'status': 'verified', 'prediction_index': prediction_index}


@router.get("/accuracy")
async def get_agent_accuracy():
    """Get accuracy statistics for each agent."""
    
    if not PREDICTION_LOG.exists():
        return {'message': 'No predictions to analyze'}
    
    with open(PREDICTION_LOG) as f:
        predictions = json.load(f)
    
    verified = [p for p in predictions if p.get('verified')]
    
    if not verified:
        return {'message': 'No verified predictions yet', 'total_predictions': len(predictions)}
    
    stats = {}
    for pred in verified:
        for agent, data in pred.get('agents', {}).items():
            if agent not in stats:
                stats[agent] = {'correct': 0, 'total': 0}
            stats[agent]['total'] += 1
            if data.get('was_correct'):
                stats[agent]['correct'] += 1
    
    # Calculate accuracy
    for agent in stats:
        stats[agent]['accuracy'] = round(
            stats[agent]['correct'] / stats[agent]['total'] * 100, 1
        ) if stats[agent]['total'] > 0 else 0
    
    return {
        'agent_accuracy': stats,
        'verified_predictions': len(verified),
        'total_predictions': len(predictions),
    }


@router.get("/knowledge/{agent}/{symbol}")
async def preview_agent_knowledge(agent: str, symbol: str):
    """Preview what knowledge an agent will use for a symbol."""
    
    if agent == "neo":
        knowledge = load_neo_knowledge(symbol)
    elif agent == "claudia":
        knowledge = load_claudia_knowledge(symbol)
    else:
        knowledge = {'info': 'Quant agent uses only market data, no additional knowledge'}
    
    return {
        'agent': agent,
        'symbol': symbol,
        'knowledge_loaded': knowledge,
    }
