"""
NEO Daily Options Research Bot
==============================
Runs before market open to research BTC miners and recommend options strategies.

Schedule: 8:30 AM ET on weekdays
Sends report via Telegram

Stocks: IREN, CLSK, CIFR
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import httpx
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NEO_DailyOptionsResearch")

# Configuration
BTC_MINERS = ["IREN", "CLSK", "CIFR"]
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6776619257")
META_BOT_API = os.getenv("META_BOT_API", "http://127.0.0.1:8035")
NEO_API = os.getenv("NEO_API", "http://127.0.0.1:8036")

REPORT_FILE = "/home/jbot/trading_ai/neo/reports/daily_options_report.json"


@dataclass
class StockAnalysis:
    """Daily analysis for a single stock"""
    symbol: str
    current_price: float
    prev_close: float
    change_pct: float
    high_5d: float
    low_5d: float
    dip_from_high_pct: float
    rsi: float
    volume_ratio: float  # vs 20-day avg
    meta_signal: str
    meta_confidence: int
    outlook: str  # BULLISH, BEARISH, NEUTRAL
    options_recommendation: str
    strategy_details: Dict
    key_levels: Dict


@dataclass
class OptionsStrategy:
    """Recommended options strategy"""
    name: str
    action: str
    strike: float
    expiration: str
    premium: float
    max_risk: float
    target_profit: float
    rationale: str


def is_market_hours_soon() -> bool:
    """Check if we're within 2 hours of market open"""
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    # Skip weekends
    if now_et.weekday() >= 5:
        return False
    
    # Market opens at 9:30 AM ET, we want to run around 8:00-8:30 AM
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    time_to_open = (market_open - now_et).total_seconds() / 60  # minutes
    
    return -30 <= time_to_open <= 120  # 30 min after to 2 hours before


async def send_telegram(message: str, parse_mode: str = "HTML"):
    """Send message via Telegram"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("No Telegram token configured")
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": parse_mode
                },
                timeout=10
            )
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


async def get_meta_signal(symbol: str) -> Dict:
    """Get Meta Bot signal for a symbol"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{META_BOT_API}/api/meta/{symbol.lower()}/signal",
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"Meta signal unavailable for {symbol}: {e}")
    return {"action": "UNKNOWN", "confidence": 0}


def fetch_stock_data(symbol: str) -> Dict:
    """Fetch stock data from Yahoo Finance"""
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get recent history
        hist = ticker.history(period="20d", interval="1d")
        if hist.empty:
            return {}
        
        current = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
        high_5d = float(hist['High'].tail(5).max())
        low_5d = float(hist['Low'].tail(5).min())
        
        # Volume ratio
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        today_volume = hist['Volume'].iloc[-1]
        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0
        
        # RSI
        close = hist['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])
        
        return {
            "current": current,
            "prev_close": prev_close,
            "change_pct": ((current - prev_close) / prev_close) * 100,
            "high_5d": high_5d,
            "low_5d": low_5d,
            "dip_from_high": ((current - high_5d) / high_5d) * 100,
            "rsi": rsi,
            "volume_ratio": volume_ratio
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return {}


def fetch_options_chain(symbol: str) -> Dict:
    """Fetch options chain data"""
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            return {"available": False}
        
        # Get nearest expiration (preferably 2-4 weeks out)
        target_exp = None
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            days_out = (exp_date - datetime.now()).days
            if 14 <= days_out <= 45:
                target_exp = exp
                break
        
        if not target_exp:
            target_exp = expirations[0] if expirations else None
        
        if not target_exp:
            return {"available": False}
        
        # Get options chain
        chain = ticker.option_chain(target_exp)
        spot = ticker.history(period='1d')['Close'].iloc[-1]
        
        # Find ATM options
        calls = chain.calls
        puts = chain.puts
        
        atm_strike = round(spot / 5) * 5  # Round to nearest $5
        
        atm_call = calls[calls['strike'] == atm_strike]
        atm_put = puts[puts['strike'] == atm_strike]
        
        # OTM options (1 strike out)
        otm_call_strike = atm_strike + 5
        otm_put_strike = atm_strike - 5
        
        otm_call = calls[calls['strike'] == otm_call_strike]
        otm_put = puts[puts['strike'] == otm_put_strike]
        
        # Calculate IV
        avg_iv = calls['impliedVolatility'].mean()
        
        # Put/Call ratio
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0
        
        return {
            "available": True,
            "expiration": target_exp,
            "spot": float(spot),
            "atm_strike": atm_strike,
            "atm_call_price": float(atm_call['lastPrice'].iloc[0]) if not atm_call.empty else 0,
            "atm_put_price": float(atm_put['lastPrice'].iloc[0]) if not atm_put.empty else 0,
            "otm_call_strike": otm_call_strike,
            "otm_call_price": float(otm_call['lastPrice'].iloc[0]) if not otm_call.empty else 0,
            "otm_put_strike": otm_put_strike,
            "otm_put_price": float(otm_put['lastPrice'].iloc[0]) if not otm_put.empty else 0,
            "avg_iv": float(avg_iv) if avg_iv else 0.5,
            "put_call_ratio": float(pc_ratio),
            "expirations_available": len(expirations)
        }
    except Exception as e:
        logger.error(f"Error fetching options for {symbol}: {e}")
        return {"available": False, "error": str(e)}


def determine_outlook(stock_data: Dict, meta_signal: Dict, options_data: Dict) -> str:
    """Determine overall outlook based on multiple factors"""
    bullish_score = 0
    bearish_score = 0
    
    # RSI
    rsi = stock_data.get('rsi', 50)
    if rsi < 30:
        bullish_score += 2  # Oversold = bullish setup
    elif rsi < 40:
        bullish_score += 1
    elif rsi > 70:
        bearish_score += 2  # Overbought = bearish setup
    elif rsi > 60:
        bearish_score += 1
    
    # Price action
    change = stock_data.get('change_pct', 0)
    if change > 3:
        bullish_score += 1
    elif change < -3:
        bearish_score += 1
    
    # Dip from high
    dip = stock_data.get('dip_from_high', 0)
    if dip < -10:
        bullish_score += 2  # Big dip = buying opportunity
    elif dip < -5:
        bullish_score += 1
    
    # Meta signal
    meta_action = meta_signal.get('action', 'UNKNOWN')
    meta_conf = meta_signal.get('confidence', 0)
    
    if meta_action in ['BUY', 'STRONG_BUY'] and meta_conf >= 60:
        bullish_score += 2
    elif meta_action in ['SELL', 'STRONG_SELL'] and meta_conf >= 60:
        bearish_score += 2
    
    # Put/Call ratio
    pc_ratio = options_data.get('put_call_ratio', 1.0)
    if pc_ratio > 1.5:
        bullish_score += 1  # High put buying = contrarian bullish
    elif pc_ratio < 0.5:
        bearish_score += 1  # High call buying = contrarian bearish
    
    # Determine outlook
    if bullish_score >= bearish_score + 2:
        return "BULLISH"
    elif bearish_score >= bullish_score + 2:
        return "BEARISH"
    else:
        return "NEUTRAL"


def recommend_strategy(symbol: str, outlook: str, stock_data: Dict, options_data: Dict) -> Dict:
    """Recommend options strategy based on outlook"""
    
    if not options_data.get('available'):
        return {
            "name": "NO_OPTIONS",
            "action": "WAIT",
            "reason": "Options data not available"
        }
    
    spot = options_data.get('spot', 0)
    expiration = options_data.get('expiration', '')
    atm_strike = options_data.get('atm_strike', 0)
    iv = options_data.get('avg_iv', 0.5)
    
    if outlook == "BULLISH":
        # Check if IV is high (>80%) - prefer selling premium
        if iv > 0.8:
            return {
                "name": "Cash-Secured Put",
                "action": f"SELL {options_data.get('otm_put_strike', 0)} PUT",
                "strike": options_data.get('otm_put_strike', 0),
                "expiration": expiration,
                "premium": options_data.get('otm_put_price', 0) * 100,
                "max_risk": (options_data.get('otm_put_strike', 0) - options_data.get('otm_put_price', 0)) * 100,
                "target_profit": options_data.get('otm_put_price', 0) * 100,
                "reason": f"High IV ({iv*100:.0f}%), sell premium on pullback support"
            }
        else:
            # Buy calls or bull spread
            dip = stock_data.get('dip_from_high', 0)
            if dip < -7:
                # Big dip - aggressive call
                return {
                    "name": "Long Call (Dip Buy)",
                    "action": f"BUY {atm_strike} CALL",
                    "strike": atm_strike,
                    "expiration": expiration,
                    "premium": options_data.get('atm_call_price', 0) * 100,
                    "max_risk": options_data.get('atm_call_price', 0) * 100,
                    "target_profit": options_data.get('atm_call_price', 0) * 200,  # 100% target
                    "reason": f"Stock down {dip:.1f}% from high - bounce expected"
                }
            else:
                # Moderate bullish - spread
                return {
                    "name": "Bull Call Spread",
                    "action": f"BUY {atm_strike} / SELL {options_data.get('otm_call_strike', 0)} CALL",
                    "strike": atm_strike,
                    "expiration": expiration,
                    "premium": (options_data.get('atm_call_price', 0) - options_data.get('otm_call_price', 0)) * 100,
                    "max_risk": (options_data.get('atm_call_price', 0) - options_data.get('otm_call_price', 0)) * 100,
                    "target_profit": (options_data.get('otm_call_strike', 0) - atm_strike) * 100,
                    "reason": f"Moderate bullish outlook, defined risk"
                }
    
    elif outlook == "BEARISH":
        return {
            "name": "Long Put or Bear Spread",
            "action": f"BUY {atm_strike} PUT",
            "strike": atm_strike,
            "expiration": expiration,
            "premium": options_data.get('atm_put_price', 0) * 100,
            "max_risk": options_data.get('atm_put_price', 0) * 100,
            "target_profit": options_data.get('atm_put_price', 0) * 150,
            "reason": "Bearish signals - protect or profit from decline"
        }
    
    else:  # NEUTRAL
        return {
            "name": "Wait / Iron Condor",
            "action": "WAIT for direction",
            "strike": atm_strike,
            "expiration": expiration,
            "premium": 0,
            "max_risk": 0,
            "target_profit": 0,
            "reason": "Mixed signals - wait for clearer setup or sell iron condor"
        }


async def analyze_stock(symbol: str) -> StockAnalysis:
    """Complete analysis for a single stock"""
    logger.info(f"Analyzing {symbol}...")
    
    # Fetch data
    stock_data = fetch_stock_data(symbol)
    options_data = fetch_options_chain(symbol)
    meta_signal = await get_meta_signal(symbol)
    
    if not stock_data:
        logger.warning(f"No data available for {symbol}")
        return None
    
    # Determine outlook
    outlook = determine_outlook(stock_data, meta_signal, options_data)
    
    # Get strategy recommendation
    strategy = recommend_strategy(symbol, outlook, stock_data, options_data)
    
    # Key levels
    key_levels = {
        "support": round(stock_data.get('low_5d', 0), 2),
        "resistance": round(stock_data.get('high_5d', 0), 2),
        "current": round(stock_data.get('current', 0), 2)
    }
    
    return StockAnalysis(
        symbol=symbol,
        current_price=round(stock_data.get('current', 0), 2),
        prev_close=round(stock_data.get('prev_close', 0), 2),
        change_pct=round(stock_data.get('change_pct', 0), 2),
        high_5d=round(stock_data.get('high_5d', 0), 2),
        low_5d=round(stock_data.get('low_5d', 0), 2),
        dip_from_high_pct=round(stock_data.get('dip_from_high', 0), 2),
        rsi=round(stock_data.get('rsi', 50), 1),
        volume_ratio=round(stock_data.get('volume_ratio', 1), 2),
        meta_signal=meta_signal.get('action', 'UNKNOWN'),
        meta_confidence=meta_signal.get('confidence', 0),
        outlook=outlook,
        options_recommendation=strategy.get('name', 'WAIT'),
        strategy_details=strategy,
        key_levels=key_levels
    )


def format_telegram_report(analyses: List[StockAnalysis]) -> str:
    """Format analysis into Telegram message"""
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    msg = f"🎯 <b>NEO DAILY OPTIONS BRIEF</b>\n"
    msg += f"📅 {now_et.strftime('%A, %B %d, %Y')}\n"
    msg += f"⏰ {now_et.strftime('%I:%M %p ET')}\n"
    msg += "━" * 30 + "\n\n"
    
    for analysis in analyses:
        if analysis is None:
            continue
        
        # Emoji based on outlook
        outlook_emoji = {
            "BULLISH": "🟢",
            "BEARISH": "🔴",
            "NEUTRAL": "⚪"
        }.get(analysis.outlook, "⚪")
        
        # Change emoji
        change_emoji = "📈" if analysis.change_pct > 0 else "📉" if analysis.change_pct < 0 else "➡️"
        
        msg += f"<b>{analysis.symbol}</b> {outlook_emoji} {analysis.outlook}\n"
        msg += f"├── Price: ${analysis.current_price:.2f} {change_emoji} {analysis.change_pct:+.1f}%\n"
        msg += f"├── RSI: {analysis.rsi:.0f} | Vol: {analysis.volume_ratio:.1f}x\n"
        msg += f"├── 5D Range: ${analysis.low_5d:.2f} - ${analysis.high_5d:.2f}\n"
        msg += f"├── From High: {analysis.dip_from_high_pct:.1f}%\n"
        msg += f"├── Meta: {analysis.meta_signal} ({analysis.meta_confidence}%)\n"
        
        # Strategy
        strat = analysis.strategy_details
        msg += f"└── <b>Strategy: {strat.get('name', 'WAIT')}</b>\n"
        if strat.get('action') and strat.get('action') != 'WAIT':
            msg += f"    📋 {strat.get('action')}\n"
            if strat.get('expiration'):
                msg += f"    📆 Exp: {strat.get('expiration')}\n"
            if strat.get('reason'):
                msg += f"    💡 {strat.get('reason')}\n"
        msg += "\n"
    
    # Summary
    bullish_count = sum(1 for a in analyses if a and a.outlook == "BULLISH")
    bearish_count = sum(1 for a in analyses if a and a.outlook == "BEARISH")
    
    msg += "━" * 30 + "\n"
    msg += f"📊 <b>SUMMARY:</b> {bullish_count} Bullish | {bearish_count} Bearish\n"
    
    if bullish_count >= 2:
        msg += "🎯 <i>Sector looking strong - consider calls on dips</i>\n"
    elif bearish_count >= 2:
        msg += "⚠️ <i>Sector weakness - be cautious, consider puts for protection</i>\n"
    else:
        msg += "📊 <i>Mixed signals - wait for clearer direction</i>\n"
    
    msg += "\n<i>DCA: 1-1-2-2-4 (test the water first!)</i>"
    
    return msg


async def run_daily_research():
    """Main function to run daily options research"""
    logger.info("=" * 60)
    logger.info("🔬 NEO DAILY OPTIONS RESEARCH")
    logger.info("=" * 60)
    
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    # Skip weekends
    if now_et.weekday() >= 5:
        logger.info("Weekend - skipping research")
        return {"status": "skipped", "reason": "weekend"}
    
    analyses = []
    
    for symbol in BTC_MINERS:
        try:
            analysis = await analyze_stock(symbol)
            if analysis:
                analyses.append(analysis)
                logger.info(f"✅ {symbol}: {analysis.outlook} - {analysis.options_recommendation}")
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    if not analyses:
        logger.warning("No analyses completed")
        return {"status": "error", "reason": "no_data"}
    
    # Format and send report
    report = format_telegram_report(analyses)
    
    # Send to Telegram
    success = await send_telegram(report)
    
    if success:
        logger.info("📨 Report sent to Telegram")
    else:
        logger.error("Failed to send Telegram report")
    
    # Save report to file
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "date": now_et.strftime("%Y-%m-%d"),
        "analyses": [asdict(a) for a in analyses if a],
        "telegram_sent": success
    }
    
    try:
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, 'w') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"📁 Report saved to {REPORT_FILE}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")
    
    return report_data


def main():
    """Entry point"""
    return asyncio.run(run_daily_research())


if __name__ == "__main__":
    result = main()
    print("\n" + json.dumps(result, indent=2, default=str))
