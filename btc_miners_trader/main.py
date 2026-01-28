"""
BTC Miners Auto-Trader
======================
Automated trading system for IREN, CIFR, CLSK positions.
Monitors TP/SL targets, executes trades based on NEO + Meta signals.

Designed for paper trading now, ready for live trading later.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import httpx
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import json
import os

# Configuration
PAPER_TRADING_API = os.getenv("PAPER_TRADING_API", "http://127.0.0.1:8500")
NEO_API = os.getenv("NEO_API", "http://127.0.0.1:8036")
META_BOT_API = os.getenv("META_BOT_API", "http://127.0.0.1:8035")
# Telegram Configuration - Paul's notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6776619257")  # Paul's chat ID

# Trading settings
MINERS = ["IREN", "CIFR", "CLSK"]
CHECK_INTERVAL_MINUTES = 15
TP_MULTIPLIER_OPTIONS = 1.30  # 30% profit for options
TP_MULTIPLIER_STOCKS = 1.15   # 15% profit for stocks (or use $150 target for IREN)
SL_MULTIPLIER_OPTIONS = 0.70  # 30% loss for options
SL_MULTIPLIER_STOCKS = 0.92   # 8% loss for stocks

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("btc-miners-trader")

app = FastAPI(title="BTC Miners Auto-Trader", version="1.0.0")

# State
trader_state = {
    "running": False,
    "last_check": None,
    "positions_monitored": 0,
    "trades_executed": 0,
    "last_error": None,
    "trade_log": []
}


class TradeAction(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    reason: str
    position_id: Optional[int] = None
    current_price: float
    target_price: Optional[float] = None
    confidence: int = 0
    source: str = "AUTO_TRADER"


async def send_telegram(message: str):
    """Send notification to Telegram"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("No Telegram token configured")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
    except Exception as e:
        logger.error(f"Telegram error: {e}")


async def get_paper_positions() -> Dict[str, Any]:
    """Fetch all positions from paper trading API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PAPER_TRADING_API}/paper-trading/positions",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
    return {"positions": [], "closed_positions": []}


async def close_position(position_id: int, reason: str) -> bool:
    """Close a paper trading position"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PAPER_TRADING_API}/paper-trading/close/{position_id}",
                json={"reason": reason},
                timeout=10
            )
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Error closing position {position_id}: {e}")
    return False


async def get_neo_signal(symbol: str) -> Optional[Dict]:
    """Get NEO's signal for a symbol"""
    try:
        async with httpx.AsyncClient() as client:
            # Try IREN-specific endpoint
            if symbol == "IREN":
                response = await client.get(
                    f"{NEO_API}/api/neo/iren/signal",
                    timeout=10
                )
            else:
                # Generic endpoint
                response = await client.get(
                    f"{NEO_API}/api/neo/{symbol.lower()}/signal",
                    timeout=10
                )
            
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.debug(f"NEO signal not available for {symbol}: {e}")
    return None


async def get_meta_signal(symbol: str) -> Optional[Dict]:
    """Get Meta Bot's signal for a symbol"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{META_BOT_API}/api/meta/{symbol.lower()}/signal",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.debug(f"Meta signal not available for {symbol}: {e}")
    return None


def calculate_tp_sl(position: Dict) -> tuple:
    """Calculate TP and SL targets for a position"""
    entry_price = position.get("entry_price", 0)
    is_option = position.get("is_option", False) or "CALL" in position.get("type", "") or "PUT" in position.get("type", "")
    symbol = position.get("symbol", "")
    
    # Use stored TP/SL if available
    stored_tp = position.get("take_profit")
    stored_sl = position.get("stop_loss")
    
    if stored_tp and stored_sl:
        return stored_tp, stored_sl
    
    # Calculate based on position type
    if is_option:
        tp = entry_price * TP_MULTIPLIER_OPTIONS
        sl = entry_price * SL_MULTIPLIER_OPTIONS
    else:
        # For IREN long-term holds, use $150 target
        if symbol == "IREN" and "CORE" in position.get("source", ""):
            tp = 150.0  # Long-term target
            sl = entry_price * 0.85  # Wider stop for core position
        else:
            tp = entry_price * TP_MULTIPLIER_STOCKS
            sl = entry_price * SL_MULTIPLIER_STOCKS
    
    return tp, sl


async def check_position(position: Dict) -> Optional[TradeAction]:
    """Check a single position for TP/SL or signal-based exit"""
    symbol = position.get("symbol")
    position_id = position.get("id")
    current_price = position.get("current_price", 0)
    entry_price = position.get("entry_price", 0)
    pos_type = position.get("type", "LONG")
    source = position.get("signal_source", position.get("source", ""))
    
    if not current_price or not entry_price:
        return None
    
    tp, sl = calculate_tp_sl(position)
    pnl_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    
    # Check TP hit
    if current_price >= tp:
        return TradeAction(
            symbol=symbol,
            action="SELL",
            reason=f"TP_HIT: Price ${current_price:.2f} >= Target ${tp:.2f} (+{pnl_percent:.1f}%)",
            position_id=position_id,
            current_price=current_price,
            target_price=tp,
            confidence=95,
            source="TP_AUTO"
        )
    
    # Check SL hit
    if current_price <= sl:
        return TradeAction(
            symbol=symbol,
            action="SELL",
            reason=f"SL_HIT: Price ${current_price:.2f} <= Stop ${sl:.2f} ({pnl_percent:.1f}%)",
            position_id=position_id,
            current_price=current_price,
            target_price=sl,
            confidence=95,
            source="SL_AUTO"
        )
    
    # Check signals for exit (only for scalp positions, not core holds)
    if "CORE" not in source and "ACCUMULATION" not in source:
        neo_signal = await get_neo_signal(symbol)
        meta_signal = await get_meta_signal(symbol)
        
        # Both agree on SELL with high confidence
        neo_sell = neo_signal and neo_signal.get("action") == "SELL" and neo_signal.get("confidence", 0) >= 70
        meta_sell = meta_signal and meta_signal.get("action") == "SELL" and meta_signal.get("confidence", 0) >= 70
        
        if neo_sell and meta_sell:
            return TradeAction(
                symbol=symbol,
                action="SELL",
                reason=f"SIGNAL_EXIT: NEO ({neo_signal.get('confidence')}%) + Meta ({meta_signal.get('confidence')}%) both SELL",
                position_id=position_id,
                current_price=current_price,
                confidence=min(neo_signal.get("confidence", 0), meta_signal.get("confidence", 0)),
                source="SIGNAL_CONSENSUS"
            )
    
    # Near TP - consider partial exit for options - SEND ALERT
    is_option = "CALL" in pos_type or "PUT" in pos_type
    if is_option and pnl_percent >= 25 and pnl_percent < 30:
        # Alert but don't auto-sell
        logger.info(f"⚠️ {symbol} option at +{pnl_percent:.1f}% - consider taking profits")
        await send_telegram(
            f"⚠️ <b>PROFIT ALERT - {symbol}</b>\n\n"
            f"Option at <b>+{pnl_percent:.1f}%</b> profit!\n"
            f"Entry: ${entry_price:.2f}\n"
            f"Current: ${current_price:.2f}\n"
            f"TP Target: ${tp:.2f}\n\n"
            f"<i>Consider taking partial profits (TP Ladder)</i>"
        )
    
    return None


async def execute_trade(action: TradeAction) -> bool:
    """Execute a trade action"""
    if action.action == "SELL" and action.position_id:
        success = await close_position(action.position_id, action.reason)
        
        if success:
            # Log the trade
            trade_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": action.symbol,
                "action": action.action,
                "reason": action.reason,
                "price": action.current_price,
                "position_id": action.position_id,
                "source": action.source
            }
            trader_state["trade_log"].append(trade_record)
            trader_state["trades_executed"] += 1
            
            # Save to file
            log_file = "/home/jbot/trading_ai/btc_miners_trader/trade_log.json"
            try:
                existing = []
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        existing = json.load(f)
                existing.append(trade_record)
                with open(log_file, "w") as f:
                    json.dump(existing, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving trade log: {e}")
            
            # Send Telegram notification
            emoji = "💰" if "TP" in action.reason else "🛑" if "SL" in action.reason else "📊"
            await send_telegram(
                f"{emoji} <b>BTC MINERS TRADE EXECUTED</b>\n\n"
                f"Symbol: <b>{action.symbol}</b>\n"
                f"Action: <b>{action.action}</b>\n"
                f"Price: ${action.current_price:.2f}\n"
                f"Reason: {action.reason}\n"
                f"Source: {action.source}\n"
                f"Time: {datetime.utcnow().strftime('%H:%M:%S UTC')}"
            )
            
            logger.info(f"✅ Trade executed: {action.symbol} {action.action} - {action.reason}")
            return True
        else:
            logger.error(f"❌ Failed to execute trade: {action.symbol}")
            await send_telegram(
                f"❌ <b>TRADE FAILED</b>\n\n"
                f"Symbol: {action.symbol}\n"
                f"Action: {action.action}\n"
                f"Reason: {action.reason}\n"
                f"Error: Failed to close position"
            )
    
    return False


async def check_entry_opportunities():
    """Check for buying opportunities on dips"""
    import yfinance as yf
    
    opportunities = []
    
    for symbol in MINERS:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            if hist.empty:
                continue
            
            current_price = float(hist['Close'].iloc[-1])
            high_5d = float(hist['High'].max())
            
            # Calculate RSI
            close = hist['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            
            # Calculate dip from recent high
            dip_pct = ((current_price - high_5d) / high_5d) * 100
            
            # Entry signals
            entry_signal = None
            
            if dip_pct <= -10:
                entry_signal = f"🟢 STRONG BUY - {abs(dip_pct):.1f}% dip from 5D high!"
            elif dip_pct <= -5:
                entry_signal = f"🟢 BUY DIP - {abs(dip_pct):.1f}% pullback"
            elif rsi < 35:
                entry_signal = f"🟢 OVERSOLD - RSI {rsi:.0f}"
            
            if entry_signal:
                opportunities.append({
                    "symbol": symbol,
                    "signal": entry_signal,
                    "price": current_price,
                    "rsi": rsi,
                    "dip_pct": dip_pct
                })
        except Exception as e:
            logger.debug(f"Error checking {symbol}: {e}")
    
    # Send alert if opportunities found
    if opportunities:
        msg = "🎯 <b>ENTRY OPPORTUNITY ALERT</b>\n\n"
        for opp in opportunities:
            msg += f"<b>{opp['symbol']}</b> {opp['signal']}\n"
            msg += f"├── Price: ${opp['price']:.2f}\n"
            msg += f"├── RSI: {opp['rsi']:.0f}\n"
            msg += f"└── From High: {opp['dip_pct']:.1f}%\n\n"
        
        msg += "<i>DCA: 1-1-2-2-4 (test the water first!)</i>"
        await send_telegram(msg)
        logger.info(f"📩 Sent entry opportunity alert for {len(opportunities)} stocks")


async def send_market_status_update():
    """Send periodic status update during market hours"""
    import yfinance as yf
    from datetime import datetime
    
    # Check if market hours (9:30 AM - 4:00 PM ET)
    # Simplified: just check if it's a reasonable time
    hour_utc = datetime.utcnow().hour
    if hour_utc < 14 or hour_utc > 21:  # Roughly 9 AM - 4 PM ET
        return
    
    status_lines = ["📊 <b>BTC MINERS STATUS UPDATE</b>\n"]
    
    for symbol in MINERS:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            current = info.last_price
            prev_close = info.previous_close
            change_pct = ((current - prev_close) / prev_close) * 100 if prev_close else 0
            
            emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
            status_lines.append(f"{emoji} <b>{symbol}</b>: ${current:.2f} ({change_pct:+.1f}%)")
        except:
            pass
    
    if len(status_lines) > 1:
        status_lines.append(f"\n<i>Updated: {datetime.utcnow().strftime('%H:%M UTC')}</i>")
        # Only send every 2 hours to avoid spam
        # This is called from trading_loop which runs every 15 min
        # So we only send on specific intervals
        if datetime.utcnow().hour in [14, 16, 18, 20]:  # ~10AM, 12PM, 2PM, 4PM ET
            if datetime.utcnow().minute < 20:  # First 20 min of those hours
                await send_telegram("\n".join(status_lines))


async def trading_loop():
    """Main trading loop - runs every CHECK_INTERVAL_MINUTES"""
    while trader_state["running"]:
        try:
            logger.info("🔍 Checking BTC Miners positions...")
            
            # Check for entry opportunities (dips)
            await check_entry_opportunities()
            
            # Get all positions
            data = await get_paper_positions()
            positions = data.get("positions", [])
            
            # Filter for our miners only
            miner_positions = [p for p in positions if p.get("symbol") in MINERS]
            trader_state["positions_monitored"] = len(miner_positions)
            
            if not miner_positions:
                logger.info("No BTC Miner positions to monitor")
            else:
                logger.info(f"Monitoring {len(miner_positions)} positions")
                
                # Check each position
                for position in miner_positions:
                    action = await check_position(position)
                    
                    if action and action.action == "SELL":
                        logger.info(f"🎯 Action triggered: {action.symbol} - {action.reason}")
                        await execute_trade(action)
            
            trader_state["last_check"] = datetime.utcnow().isoformat()
            trader_state["last_error"] = None
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            trader_state["last_error"] = str(e)
        
        # Wait for next check
        await asyncio.sleep(CHECK_INTERVAL_MINUTES * 60)


@app.on_event("startup")
async def startup():
    """Start the trading loop on startup"""
    trader_state["running"] = True
    asyncio.create_task(trading_loop())
    logger.info("🚀 BTC Miners Auto-Trader started")
    await send_telegram(
        "🚀 <b>BTC Miners Auto-Trader Started</b>\n\n"
        f"Monitoring: {', '.join(MINERS)}\n"
        f"Check Interval: {CHECK_INTERVAL_MINUTES} minutes\n"
        f"TP Options: +{(TP_MULTIPLIER_OPTIONS-1)*100:.0f}%\n"
        f"SL Options: -{(1-SL_MULTIPLIER_OPTIONS)*100:.0f}%"
    )


@app.on_event("shutdown")
async def shutdown():
    """Stop the trading loop"""
    trader_state["running"] = False
    logger.info("🛑 BTC Miners Auto-Trader stopped")


@app.get("/")
async def root():
    return {
        "service": "BTC Miners Auto-Trader",
        "status": "running" if trader_state["running"] else "stopped",
        "miners": MINERS,
        "check_interval_minutes": CHECK_INTERVAL_MINUTES,
        "last_check": trader_state["last_check"],
        "positions_monitored": trader_state["positions_monitored"],
        "trades_executed": trader_state["trades_executed"],
        "last_error": trader_state["last_error"]
    }


@app.get("/status")
async def get_status():
    """Get detailed status"""
    data = await get_paper_positions()
    positions = [p for p in data.get("positions", []) if p.get("symbol") in MINERS]
    
    position_status = []
    for pos in positions:
        tp, sl = calculate_tp_sl(pos)
        current = pos.get("current_price", 0)
        entry = pos.get("entry_price", 0)
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        
        position_status.append({
            "id": pos.get("id"),
            "symbol": pos.get("symbol"),
            "type": pos.get("type"),
            "entry": entry,
            "current": current,
            "pnl_percent": round(pnl_pct, 1),
            "tp_target": round(tp, 2),
            "sl_target": round(sl, 2),
            "tp_distance_pct": round((tp - current) / current * 100, 1) if current > 0 else 0,
            "sl_distance_pct": round((current - sl) / current * 100, 1) if current > 0 else 0,
            "status": "TP_HIT" if current >= tp else "SL_HIT" if current <= sl else "ACTIVE"
        })
    
    return {
        "running": trader_state["running"],
        "last_check": trader_state["last_check"],
        "next_check_in_minutes": CHECK_INTERVAL_MINUTES,
        "positions": position_status,
        "recent_trades": trader_state["trade_log"][-10:]
    }


@app.post("/check-now")
async def check_now(background_tasks: BackgroundTasks):
    """Manually trigger a position check"""
    async def run_check():
        logger.info("🔍 Manual check triggered...")
        data = await get_paper_positions()
        positions = [p for p in data.get("positions", []) if p.get("symbol") in MINERS]
        
        actions_taken = []
        for position in positions:
            action = await check_position(position)
            if action and action.action == "SELL":
                success = await execute_trade(action)
                actions_taken.append({
                    "symbol": action.symbol,
                    "action": action.action,
                    "reason": action.reason,
                    "executed": success
                })
        
        trader_state["last_check"] = datetime.utcnow().isoformat()
        return actions_taken
    
    background_tasks.add_task(run_check)
    return {"message": "Check triggered", "status": "running"}


@app.post("/pause")
async def pause_trader():
    """Pause the auto-trader"""
    trader_state["running"] = False
    await send_telegram("⏸️ <b>BTC Miners Auto-Trader PAUSED</b>")
    return {"status": "paused"}


@app.post("/resume")
async def resume_trader():
    """Resume the auto-trader"""
    if not trader_state["running"]:
        trader_state["running"] = True
        asyncio.create_task(trading_loop())
        await send_telegram("▶️ <b>BTC Miners Auto-Trader RESUMED</b>")
    return {"status": "running"}


@app.get("/trade-log")
async def get_trade_log():
    """Get the trade log"""
    log_file = "/home/jbot/trading_ai/btc_miners_trader/trade_log.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                return {"trades": json.load(f)}
    except Exception as e:
        logger.error(f"Error reading trade log: {e}")
    return {"trades": trader_state["trade_log"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8037)
