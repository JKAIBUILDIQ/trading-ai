#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
DAILY FEEDBACK COLLECTOR
═══════════════════════════════════════════════════════════════════════════════

Tracks real-time feedback throughout the day:
1. What positions were opened/closed
2. What agents recommended
3. Actual price movements
4. Updates agent accuracy based on outcomes

Can be run:
- Manually: python3 daily_feedback.py
- Hourly: 0 * * * * python3 daily_feedback.py --hourly
- At market close: 30 16 * * 1-5 python3 daily_feedback.py --close

Also provides endpoints for the War Room to submit feedback.
"""

import asyncio
import httpx
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("DailyFeedback")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_API = "http://localhost:8890/knowledge"
PORTFOLIO_STATE_FILE = Path("/home/jbot/trading_ai/neo/portfolio_state.json")
FEEDBACK_LOG = Path("/home/jbot/trading_ai/neo/daily_data/feedback_log.json")

# Price movement thresholds
CORRECT_THRESHOLD = 20  # Pips in predicted direction = correct
PARTIAL_THRESHOLD = 10  # Pips = partially correct


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def load_portfolio_state() -> Dict:
    """Load current portfolio state."""
    if PORTFOLIO_STATE_FILE.exists():
        try:
            with open(PORTFOLIO_STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"positions": [], "total_pnl": 0}


async def get_current_price(symbol: str = "XAUUSD") -> float:
    """Get current price for a symbol."""
    try:
        async with httpx.AsyncClient() as client:
            # Try gold-forex-api
            response = await client.get("http://localhost:8037/api/gold/price", timeout=10.0)
            if response.status_code == 200:
                return response.json().get("price", 0)
    except:
        pass
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS OUTCOME TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

async def get_pending_analyses() -> List[Dict]:
    """Get analyses that haven't had outcomes recorded yet."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KNOWLEDGE_API}/recent_analyses",
                params={"hours": 48},
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                # Filter to those without outcomes
                return [a for a in data.get("analyses", []) if not a.get("outcome")]
    except Exception as e:
        logger.error(f"Failed to get analyses: {e}")
    return []


async def evaluate_analysis_outcome(
    analysis: Dict,
    current_price: float
) -> Optional[Dict]:
    """
    Evaluate if an analysis was correct based on price movement.
    
    Returns outcome dict or None if not enough time has passed.
    """
    # Get analysis details
    analysis_id = analysis.get("_id")
    agent = analysis.get("agent", "unknown")
    price_at_analysis = analysis.get("price_at_analysis", 0)
    recommendation = analysis.get("recommendation", "").upper()
    timestamp = analysis.get("timestamp", "")
    
    if not price_at_analysis or not recommendation:
        return None
    
    # Check if enough time has passed (at least 2 hours)
    try:
        analysis_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if datetime.now(analysis_time.tzinfo) - analysis_time < timedelta(hours=2):
            return None
    except:
        pass
    
    # Calculate price movement
    price_diff = current_price - price_at_analysis
    pips = abs(price_diff) * 10  # For gold, 1 pip = $0.10
    
    # Determine outcome based on recommendation
    outcome = "incorrect"
    
    if recommendation in ["BUY", "LONG", "BULLISH"]:
        if price_diff > 0:
            if pips >= CORRECT_THRESHOLD:
                outcome = "correct"
            elif pips >= PARTIAL_THRESHOLD:
                outcome = "partially_correct"
    elif recommendation in ["SELL", "SHORT", "BEARISH"]:
        if price_diff < 0:
            if pips >= CORRECT_THRESHOLD:
                outcome = "correct"
            elif pips >= PARTIAL_THRESHOLD:
                outcome = "partially_correct"
    elif recommendation == "NEUTRAL":
        if pips < PARTIAL_THRESHOLD:
            outcome = "correct"  # Neutral was right if price didn't move much
    
    return {
        "analysis_id": analysis_id,
        "agent": agent,
        "recommendation": recommendation,
        "price_at_analysis": price_at_analysis,
        "price_after": current_price,
        "pips_moved": round(pips, 1),
        "outcome": outcome
    }


async def record_outcome(outcome: Dict):
    """Record an outcome to the knowledge base."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{KNOWLEDGE_API}/record_outcome",
                json={
                    "analysis_id": outcome["analysis_id"],
                    "actual_outcome": outcome["outcome"],
                    "price_after": outcome["price_after"],
                    "pnl_result": 0,  # Could calculate based on position size
                    "notes": f"Auto-evaluated. {outcome['pips_moved']} pips from {outcome['recommendation']}"
                },
                timeout=30.0
            )
            logger.info(f"Recorded {outcome['outcome']} for {outcome['agent']}")
            return True
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

async def collect_hourly_feedback():
    """Collect feedback on an hourly basis."""
    logger.info("Collecting hourly feedback...")
    
    # Get current price
    current_price = await get_current_price()
    if not current_price:
        logger.warning("Could not get current price, skipping feedback")
        return
    
    logger.info(f"Current XAUUSD price: ${current_price}")
    
    # Get pending analyses
    pending = await get_pending_analyses()
    logger.info(f"Found {len(pending)} pending analyses to evaluate")
    
    # Evaluate each
    outcomes_recorded = 0
    for analysis in pending:
        outcome = await evaluate_analysis_outcome(analysis, current_price)
        if outcome:
            success = await record_outcome(outcome)
            if success:
                outcomes_recorded += 1
    
    logger.info(f"Recorded {outcomes_recorded} outcomes")
    
    # Log to feedback file
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "hourly",
        "current_price": current_price,
        "pending_analyses": len(pending),
        "outcomes_recorded": outcomes_recorded
    }
    append_to_feedback_log(log_entry)
    
    return outcomes_recorded


async def collect_close_feedback():
    """Collect feedback at market close - more comprehensive."""
    logger.info("Collecting end-of-day feedback...")
    
    current_price = await get_current_price()
    
    # Get all today's analyses
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KNOWLEDGE_API}/recent_analyses",
                params={"hours": 24},
                timeout=30.0
            )
            if response.status_code == 200:
                analyses = response.json().get("analyses", [])
    except:
        analyses = []
    
    # Get portfolio state
    portfolio = load_portfolio_state()
    
    # Calculate daily P&L
    daily_pnl = portfolio.get("daily_pnl", 0)
    
    # Create summary
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "close_price": current_price,
        "analyses_count": len(analyses),
        "daily_pnl": daily_pnl,
        "positions_held": len(portfolio.get("positions", [])),
        "agents": {}
    }
    
    # Summarize by agent
    for analysis in analyses:
        agent = analysis.get("agent", "unknown")
        if agent not in summary["agents"]:
            summary["agents"][agent] = {
                "count": 0,
                "recommendations": []
            }
        summary["agents"][agent]["count"] += 1
        summary["agents"][agent]["recommendations"].append(
            analysis.get("recommendation", "N/A")
        )
    
    # Save summary
    summary_file = Path(f"/home/jbot/trading_ai/neo/daily_data/summary_{datetime.now().strftime('%Y%m%d')}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved daily summary to {summary_file}")
    
    # Also run hourly collection
    await collect_hourly_feedback()
    
    return summary


def append_to_feedback_log(entry: Dict):
    """Append entry to feedback log."""
    log = []
    if FEEDBACK_LOG.exists():
        try:
            with open(FEEDBACK_LOG) as f:
                log = json.load(f)
        except:
            log = []
    
    log.append(entry)
    
    # Keep last 500 entries
    log = log[-500:]
    
    with open(FEEDBACK_LOG, 'w') as f:
        json.dump(log, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL FEEDBACK SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

async def submit_manual_feedback(
    analysis_id: str,
    outcome: str,
    price_after: float = 0,
    pnl: float = 0,
    notes: str = ""
):
    """Submit manual feedback for an analysis."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{KNOWLEDGE_API}/record_outcome",
                json={
                    "analysis_id": analysis_id,
                    "actual_outcome": outcome,
                    "price_after": price_after,
                    "pnl_result": pnl,
                    "notes": notes or "Manual submission"
                },
                timeout=30.0
            )
            logger.info(f"Submitted manual feedback: {outcome}")
            return True
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
    return False


async def submit_trade_to_journal(
    action: str,
    symbol: str = "XAUUSD",
    size: float = 0,
    price: float = 0,
    reason: str = "",
    agent: str = ""
):
    """Submit a trade to the journal."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{KNOWLEDGE_API}/journal",
                json={
                    "action": action,
                    "symbol": symbol,
                    "size": size,
                    "price": price,
                    "reason": reason,
                    "agent_recommendation": agent,
                    "defcon": 3,
                    "tags": ["manual"]
                },
                timeout=30.0
            )
            logger.info(f"Logged trade: {action} {size} {symbol} @ {price}")
            return True
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Main function."""
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if "--hourly" in args:
        await collect_hourly_feedback()
    elif "--close" in args:
        await collect_close_feedback()
    elif "--manual" in args:
        # Example: python daily_feedback.py --manual analysis123 correct 4950 250 "Good call"
        if len(args) >= 3:
            analysis_id = args[1]
            outcome = args[2]
            price = float(args[3]) if len(args) > 3 else 0
            pnl = float(args[4]) if len(args) > 4 else 0
            notes = args[5] if len(args) > 5 else ""
            await submit_manual_feedback(analysis_id, outcome, price, pnl, notes)
    elif "--trade" in args:
        # Example: python daily_feedback.py --trade BUY XAUUSD 1.0 4900 "DCA entry" ghost
        if len(args) >= 5:
            action = args[1]
            symbol = args[2]
            size = float(args[3])
            price = float(args[4])
            reason = args[5] if len(args) > 5 else ""
            agent = args[6] if len(args) > 6 else ""
            await submit_trade_to_journal(action, symbol, size, price, reason, agent)
    else:
        # Default: run hourly feedback
        await collect_hourly_feedback()


if __name__ == "__main__":
    asyncio.run(main())
