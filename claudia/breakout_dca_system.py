#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
BREAKOUT DCA OPTIONS SYSTEM
═══════════════════════════════════════════════════════════════════════════════

Automated options buying system for top breakout stocks:
- Buy on dips (5% drops from entry)
- DCA (Dollar Cost Average) on downturns
- Stop buying when SuperTrend turns BEARISH
- Track all positions across multiple symbols

Created: 2026-01-26
By: Claudia & The Swarm
═══════════════════════════════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("BreakoutDCA")

# Configuration
CONFIG = {
    "watchlist": [
        {"symbol": "NVDA", "target": 253.19, "preferred_strike_delta": 0.05},
        {"symbol": "AMD", "target": 287.38, "preferred_strike_delta": 0.05},
        {"symbol": "DOCN", "target": 70.00, "preferred_strike_delta": 0.10},
        {"symbol": "IREN", "target": 150.00, "preferred_strike_delta": 0.05},
        {"symbol": "HUT", "target": 80.00, "preferred_strike_delta": 0.10},
        {"symbol": "TSM", "target": 419.81, "preferred_strike_delta": 0.05},
        {"symbol": "ASML", "target": 1500.00, "preferred_strike_delta": 0.03},
        {"symbol": "LRCX", "target": 280.00, "preferred_strike_delta": 0.05},
        {"symbol": "KLAC", "target": 1700.00, "preferred_strike_delta": 0.03},
        {"symbol": "AMAT", "target": 380.00, "preferred_strike_delta": 0.05},
    ],
    "dca_drop_pct": 5.0,  # Buy on every 5% drop
    "max_positions_per_symbol": 5,  # Max DCA entries per stock
    "contracts_per_entry": 2,  # Contracts to buy each time
    "preferred_dte_days": 45,  # Days to expiration (theta friendly)
    "supertrend_period": 10,
    "supertrend_multiplier": 3.0,
    "paper_trading_url": "http://localhost:8500",
}

# State file
STATE_FILE = Path("/home/jbot/trading_ai/claudia/research/breakout_dca_state.json")


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate SuperTrend indicator
    Returns DataFrame with supertrend values and direction
    """
    df = df.copy()
    
    # Calculate ATR
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(period).mean()
    
    # Calculate basic bands
    hl2 = (df['High'] + df['Low']) / 2
    df['basic_ub'] = hl2 + (multiplier * df['atr'])
    df['basic_lb'] = hl2 - (multiplier * df['atr'])
    
    # Initialize SuperTrend columns
    df['supertrend'] = 0.0
    df['direction'] = 1  # 1 = bullish, -1 = bearish
    
    for i in range(period, len(df)):
        # Upper band
        if df['basic_ub'].iloc[i] < df['supertrend'].iloc[i-1] or df['Close'].iloc[i-1] > df['supertrend'].iloc[i-1]:
            df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
        else:
            df.loc[df.index[i], 'final_ub'] = df['supertrend'].iloc[i-1] if df['supertrend'].iloc[i-1] == df.get('final_ub', df['basic_ub']).iloc[i-1] else df['basic_ub'].iloc[i]
        
        # Lower band
        if df['basic_lb'].iloc[i] > df['supertrend'].iloc[i-1] or df['Close'].iloc[i-1] < df['supertrend'].iloc[i-1]:
            df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lb'] = df['supertrend'].iloc[i-1] if df['supertrend'].iloc[i-1] == df.get('final_lb', df['basic_lb']).iloc[i-1] else df['basic_lb'].iloc[i]
        
        # SuperTrend
        if df['direction'].iloc[i-1] == 1:  # Was bullish
            if df['Close'].iloc[i] < df.get('final_lb', df['basic_lb']).iloc[i]:
                df.loc[df.index[i], 'direction'] = -1
                df.loc[df.index[i], 'supertrend'] = df.get('final_ub', df['basic_ub']).iloc[i]
            else:
                df.loc[df.index[i], 'direction'] = 1
                df.loc[df.index[i], 'supertrend'] = df.get('final_lb', df['basic_lb']).iloc[i]
        else:  # Was bearish
            if df['Close'].iloc[i] > df.get('final_ub', df['basic_ub']).iloc[i]:
                df.loc[df.index[i], 'direction'] = 1
                df.loc[df.index[i], 'supertrend'] = df.get('final_lb', df['basic_lb']).iloc[i]
            else:
                df.loc[df.index[i], 'direction'] = -1
                df.loc[df.index[i], 'supertrend'] = df.get('final_ub', df['basic_ub']).iloc[i]
    
    return df


def get_supertrend_signal(symbol: str) -> Dict:
    """Get SuperTrend signal for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo", interval="1d")
        
        if df.empty or len(df) < 20:
            return {"error": "Insufficient data"}
        
        # Calculate SuperTrend
        df = calculate_supertrend(
            df, 
            period=CONFIG["supertrend_period"],
            multiplier=CONFIG["supertrend_multiplier"]
        )
        
        current_price = df['Close'].iloc[-1]
        supertrend = df['supertrend'].iloc[-1]
        direction = df['direction'].iloc[-1]
        prev_direction = df['direction'].iloc[-2] if len(df) > 1 else direction
        
        # Detect trend change
        trend_changed = direction != prev_direction
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "supertrend": round(supertrend, 2),
            "direction": "BULLISH" if direction == 1 else "BEARISH",
            "trend_changed": trend_changed,
            "buy_allowed": direction == 1,  # Only buy when bullish
            "distance_to_st": round((current_price - supertrend) / current_price * 100, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"SuperTrend error for {symbol}: {e}")
        return {"error": str(e)}


def load_state() -> Dict:
    """Load DCA state from file"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    
    # Initialize state for all symbols
    state = {
        "created_at": datetime.utcnow().isoformat(),
        "positions": {},
        "signals_history": []
    }
    
    for stock in CONFIG["watchlist"]:
        state["positions"][stock["symbol"]] = {
            "symbol": stock["symbol"],
            "target_price": stock["target"],
            "reference_price": None,
            "last_buy_price": None,
            "dca_count": 0,
            "total_contracts": 0,
            "entries": [],
            "supertrend_status": "UNKNOWN",
            "buying_enabled": True
        }
    
    return state


def save_state(state: Dict):
    """Save DCA state to file"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_best_option(symbol: str, stock_price: float, dte_target: int = 45) -> Optional[Dict]:
    """Find the best call option for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get expiration dates
        expirations = ticker.options
        if not expirations:
            return None
        
        # Find expiration closest to target DTE
        today = datetime.now()
        best_exp = None
        best_dte_diff = float('inf')
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - today).days
            if dte >= 21 and abs(dte - dte_target) < best_dte_diff:  # Min 3 weeks out
                best_dte_diff = abs(dte - dte_target)
                best_exp = exp
        
        if not best_exp:
            return None
        
        # Get option chain
        chain = ticker.option_chain(best_exp)
        calls = chain.calls
        
        # Find ATM or slightly OTM call
        target_strike = stock_price * 1.05  # 5% OTM
        calls['strike_diff'] = abs(calls['strike'] - target_strike)
        best_call = calls.loc[calls['strike_diff'].idxmin()]
        
        return {
            "symbol": symbol,
            "expiration": best_exp,
            "strike": float(best_call['strike']),
            "bid": float(best_call['bid']),
            "ask": float(best_call['ask']),
            "mid": round((float(best_call['bid']) + float(best_call['ask'])) / 2, 2),
            "volume": int(best_call['volume']) if not pd.isna(best_call['volume']) else 0,
            "open_interest": int(best_call['openInterest']) if not pd.isna(best_call['openInterest']) else 0,
            "implied_volatility": round(float(best_call['impliedVolatility']) * 100, 1) if not pd.isna(best_call['impliedVolatility']) else 0,
            "dte": (datetime.strptime(best_exp, '%Y-%m-%d') - today).days
        }
        
    except Exception as e:
        logger.error(f"Option chain error for {symbol}: {e}")
        return None


def check_buy_signal(symbol: str, state: Dict) -> Dict:
    """Check if we should buy options for a symbol"""
    position = state["positions"].get(symbol, {})
    
    # Get current price and SuperTrend
    st_signal = get_supertrend_signal(symbol)
    if "error" in st_signal:
        return {"should_buy": False, "reason": st_signal["error"]}
    
    current_price = st_signal["current_price"]
    
    # Update state with SuperTrend status
    position["supertrend_status"] = st_signal["direction"]
    
    # Check if buying is allowed (SuperTrend must be bullish)
    if st_signal["direction"] == "BEARISH":
        position["buying_enabled"] = False
        return {
            "should_buy": False,
            "reason": f"SuperTrend BEARISH - buying paused",
            "supertrend": st_signal,
            "current_price": current_price
        }
    
    position["buying_enabled"] = True
    
    # Check position limits
    if position.get("dca_count", 0) >= CONFIG["max_positions_per_symbol"]:
        return {
            "should_buy": False,
            "reason": f"Max positions reached ({CONFIG['max_positions_per_symbol']})",
            "supertrend": st_signal,
            "current_price": current_price
        }
    
    # Initialize reference price if not set
    if not position.get("reference_price"):
        position["reference_price"] = current_price
        return {
            "should_buy": True,
            "reason": "Initial entry - opening first position",
            "supertrend": st_signal,
            "current_price": current_price,
            "is_initial": True
        }
    
    # Check for DCA opportunity (5% drop from last buy)
    last_buy = position.get("last_buy_price") or position.get("reference_price")
    drop_pct = (last_buy - current_price) / last_buy * 100
    
    if drop_pct >= CONFIG["dca_drop_pct"]:
        return {
            "should_buy": True,
            "reason": f"DCA triggered: {drop_pct:.1f}% drop from ${last_buy:.2f}",
            "supertrend": st_signal,
            "current_price": current_price,
            "drop_pct": drop_pct,
            "is_dca": True
        }
    
    return {
        "should_buy": False,
        "reason": f"Waiting for {CONFIG['dca_drop_pct']}% drop (current: {drop_pct:.1f}%)",
        "supertrend": st_signal,
        "current_price": current_price,
        "drop_needed": CONFIG["dca_drop_pct"] - drop_pct,
        "next_buy_price": round(last_buy * (1 - CONFIG["dca_drop_pct"]/100), 2)
    }


def execute_buy(symbol: str, state: Dict, signal: Dict) -> bool:
    """Execute a buy order on paper trading"""
    try:
        current_price = signal["current_price"]
        
        # Get best option
        option = get_best_option(symbol, current_price)
        if not option:
            logger.warning(f"Could not find option for {symbol}")
            return False
        
        # Prepare order
        payload = {
            "symbol": symbol,
            "type": "BUY_CALL",
            "size": CONFIG["contracts_per_entry"],
            "entry_price": option["ask"],
            "strike": option["strike"],
            "expiry": option["expiration"],
            "is_option": True,
            "option_type": "call",
            "source": "BREAKOUT_DCA",
            "confidence": 75,
            "notes": f"DCA #{state['positions'][symbol]['dca_count'] + 1} | ST: {signal['supertrend']['direction']} | Drop: {signal.get('drop_pct', 0):.1f}%"
        }
        
        # Execute on paper trading
        response = requests.post(
            f"{CONFIG['paper_trading_url']}/paper-trading/open",
            json=payload,
            timeout=10
        )
        
        if response.ok:
            result = response.json()
            logger.info(f"✅ Bought {CONFIG['contracts_per_entry']}x {symbol} ${option['strike']} calls @ ${option['ask']}")
            
            # Update state
            position = state["positions"][symbol]
            position["dca_count"] += 1
            position["total_contracts"] += CONFIG["contracts_per_entry"]
            position["last_buy_price"] = current_price
            position["entries"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "stock_price": current_price,
                "option_price": option["ask"],
                "strike": option["strike"],
                "expiry": option["expiration"],
                "contracts": CONFIG["contracts_per_entry"],
                "reason": signal["reason"]
            })
            
            save_state(state)
            return True
        else:
            logger.error(f"Buy failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Execute buy error: {e}")
        return False


def run_dca_check():
    """Run DCA check for all symbols"""
    logger.info("=" * 70)
    logger.info("🔄 BREAKOUT DCA SYSTEM CHECK")
    logger.info("=" * 70)
    
    state = load_state()
    results = []
    
    for stock in CONFIG["watchlist"]:
        symbol = stock["symbol"]
        logger.info(f"\n📊 Checking {symbol}...")
        
        signal = check_buy_signal(symbol, state)
        signal["symbol"] = symbol
        results.append(signal)
        
        st = signal.get("supertrend", {})
        logger.info(f"   Price: ${signal.get('current_price', 0):.2f}")
        logger.info(f"   SuperTrend: {st.get('direction', 'N/A')} (ST: ${st.get('supertrend', 0):.2f})")
        logger.info(f"   Signal: {'🟢 BUY' if signal['should_buy'] else '⏳ WAIT'}")
        logger.info(f"   Reason: {signal['reason']}")
        
        if signal["should_buy"]:
            logger.info(f"   🎯 Executing buy...")
            success = execute_buy(symbol, state, signal)
            signal["executed"] = success
    
    # Save updated state
    save_state(state)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 SUMMARY")
    logger.info("=" * 70)
    
    bullish = [r for r in results if r.get("supertrend", {}).get("direction") == "BULLISH"]
    bearish = [r for r in results if r.get("supertrend", {}).get("direction") == "BEARISH"]
    buys = [r for r in results if r.get("should_buy")]
    
    logger.info(f"   🟢 Bullish: {len(bullish)}/{len(results)}")
    logger.info(f"   🔴 Bearish: {len(bearish)}/{len(results)}")
    logger.info(f"   🎯 Buy Signals: {len(buys)}")
    
    return results


def get_system_status() -> Dict:
    """Get full system status for API"""
    state = load_state()
    
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "dca_drop_pct": CONFIG["dca_drop_pct"],
            "max_positions": CONFIG["max_positions_per_symbol"],
            "contracts_per_entry": CONFIG["contracts_per_entry"]
        },
        "symbols": []
    }
    
    for stock in CONFIG["watchlist"]:
        symbol = stock["symbol"]
        st_signal = get_supertrend_signal(symbol)
        position = state["positions"].get(symbol, {})
        
        last_buy = position.get("last_buy_price") or position.get("reference_price")
        current_price = st_signal.get("current_price", 0)
        
        if last_buy and current_price:
            drop_pct = (last_buy - current_price) / last_buy * 100
            next_buy = round(last_buy * (1 - CONFIG["dca_drop_pct"]/100), 2)
        else:
            drop_pct = 0
            next_buy = current_price * 0.95 if current_price else 0
        
        status["symbols"].append({
            "symbol": symbol,
            "target": stock["target"],
            "current_price": current_price,
            "supertrend": st_signal.get("direction", "UNKNOWN"),
            "supertrend_value": st_signal.get("supertrend", 0),
            "buying_enabled": st_signal.get("direction") == "BULLISH",
            "dca_count": position.get("dca_count", 0),
            "total_contracts": position.get("total_contracts", 0),
            "last_buy_price": last_buy,
            "current_drop_pct": round(drop_pct, 1),
            "next_buy_price": next_buy,
            "entries": position.get("entries", [])[-3:]  # Last 3 entries
        })
    
    # Summary stats
    status["summary"] = {
        "total_symbols": len(CONFIG["watchlist"]),
        "bullish_count": len([s for s in status["symbols"] if s["supertrend"] == "BULLISH"]),
        "bearish_count": len([s for s in status["symbols"] if s["supertrend"] == "BEARISH"]),
        "total_contracts": sum(s["total_contracts"] for s in status["symbols"]),
        "symbols_with_positions": len([s for s in status["symbols"] if s["dca_count"] > 0])
    }
    
    return status


def initialize_positions():
    """Initialize first positions for all bullish stocks"""
    logger.info("=" * 70)
    logger.info("🚀 INITIALIZING BREAKOUT DCA POSITIONS")
    logger.info("=" * 70)
    
    state = load_state()
    initialized = []
    
    for stock in CONFIG["watchlist"]:
        symbol = stock["symbol"]
        position = state["positions"].get(symbol, {})
        
        # Skip if already has positions
        if position.get("dca_count", 0) > 0:
            logger.info(f"⏭️ {symbol}: Already has {position['dca_count']} positions")
            continue
        
        # Check SuperTrend
        st_signal = get_supertrend_signal(symbol)
        if st_signal.get("direction") != "BULLISH":
            logger.info(f"⏸️ {symbol}: SuperTrend BEARISH - skipping initial buy")
            continue
        
        current_price = st_signal["current_price"]
        
        # Get option and execute
        option = get_best_option(symbol, current_price)
        if not option:
            logger.warning(f"❌ {symbol}: Could not find suitable option")
            continue
        
        logger.info(f"\n🎯 {symbol} @ ${current_price:.2f}")
        logger.info(f"   Option: ${option['strike']} {option['expiration']} @ ${option['ask']:.2f}")
        logger.info(f"   Executing initial buy...")
        
        # Set reference price
        position["reference_price"] = current_price
        position["supertrend_status"] = "BULLISH"
        position["buying_enabled"] = True
        state["positions"][symbol] = position
        
        signal = {
            "current_price": current_price,
            "supertrend": st_signal,
            "reason": "Initial position - SuperTrend bullish"
        }
        
        success = execute_buy(symbol, state, signal)
        if success:
            initialized.append(symbol)
    
    save_state(state)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Initialized {len(initialized)} positions: {', '.join(initialized)}")
    logger.info("=" * 70)
    
    return initialized


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        # Initialize all positions
        initialize_positions()
    else:
        # Regular check
        run_dca_check()
