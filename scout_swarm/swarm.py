"""
SCOUT SWARM - H100 Powered Market Intelligence
Scans for bounce and fade setups using local Ollama models.

Models:
- deepseek-r1:70b - Coordinator & Final Ranker
- claudiaArgo-vl - Chart/Vision Analysis  
- qwen3:32b - Fundamental Analysis
- mistral:7b - Fast Technical Scans

Runs: Morning (6am) + Evening (6pm)
Output: Top 10 setups → JSON + Telegram
"""

import asyncio
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import yfinance as yf
import numpy as np
import requests

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6776619257")


def send_telegram(message: str, parse_mode: str = None) -> bool:
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        print("⚠️  No Telegram token configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message[:4000],  # Telegram limit
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("📱 Telegram sent!")
            return True
        else:
            print(f"❌ Telegram error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def format_telegram_report(report: Dict) -> str:
    """Format report for Telegram (compact)."""
    lines = []
    lines.append(f"🔍 SCOUT REPORT - {report['scan_date']}")
    lines.append(f"Scanned: {report['stocks_scanned']} stocks")
    lines.append("")
    
    lines.append("🟢 BOUNCE SETUPS:")
    for i, s in enumerate(report['top_bounces'][:5], 1):
        lines.append(f"#{i} {s['symbol']} ${s['price']:.2f} ({s['change_3d']:+.1f}%)")
        lines.append(f"   Score: {s['combined_score']:.0f} | {s['confidence']}")
        lines.append(f"   → Entry: ${s['entry_zone'][0]:.2f}-${s['entry_zone'][1]:.2f}")
        lines.append(f"   → Target: ${s['target']:.2f} | Stop: ${s['stop']:.2f}")
    
    lines.append("")
    lines.append("🔴 FADE SETUPS:")
    for i, s in enumerate(report['top_fades'][:5], 1):
        lines.append(f"#{i} {s['symbol']} ${s['price']:.2f} ({s['change_3d']:+.1f}%)")
        lines.append(f"   Score: {s['combined_score']:.0f} | {s['confidence']}")
        lines.append(f"   → Entry: ${s['entry_zone'][0]:.2f}-${s['entry_zone'][1]:.2f}")
        lines.append(f"   → Target: ${s['target']:.2f} | Stop: ${s['stop']:.2f}")
    
    return "\n".join(lines)

# Model assignments
MODELS = {
    "coordinator": "deepseek-r1:70b",
    "technical": "mistral:7b",
    "fundamental": "qwen3:32b",
    "news": "llama3.1:8b",  # Fast for news summaries
    "ranker": "deepseek-r1:70b",
}

@dataclass
class StockData:
    symbol: str
    price: float
    change_1d: float
    change_3d: float
    change_5d: float
    volume_ratio: float  # vs 20d avg
    rsi: float
    distance_from_52w_high: float
    distance_from_52w_low: float
    gap_percent: float  # Today's gap
    sector: str
    market_cap: float
    pe_ratio: Optional[float]
    
@dataclass
class ScoutScore:
    technical_score: float  # 0-100
    fundamental_score: float  # 0-100
    news_score: float  # 0-100 (positive = bullish news)
    combined_score: float
    direction: str  # BOUNCE or FADE
    confidence: str  # HIGH, MEDIUM, LOW
    thesis: str
    entry_zone: Tuple[float, float]
    target: float
    stop: float


class MarketDataFetcher:
    """Fetches market data for scanning."""
    
    def __init__(self):
        self.cache = {}
    
    def get_biggest_movers(self, min_change: float = 5.0, limit: int = 50) -> List[str]:
        """Get stocks with biggest moves (up or down)."""
        # Use a watchlist + scan for movers
        # In production, this would use a screener API
        
        base_watchlist = [
            # Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC",
            "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR", "NET",
            # BTC Miners / Crypto
            "IREN", "CLSK", "MARA", "RIOT", "CIFR", "COIN", "MSTR",
            # AI/Semis
            "SMCI", "ARM", "AVGO", "MU", "QCOM", "TSM",
            # High Beta
            "TSLA", "SQ", "SHOP", "ROKU", "SNAP", "PINS",
            # Biotech
            "MRNA", "BNTX", "REGN", "VRTX",
            # Financials
            "JPM", "GS", "MS", "BAC", "C",
            # Energy
            "XOM", "CVX", "OXY", "SLB",
            # ETFs
            "SPY", "QQQ", "IWM", "GLD", "SLV",
        ]
        
        movers = []
        for symbol in base_watchlist:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if len(hist) >= 2:
                    change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                    if abs(change) >= min_change:
                        movers.append((symbol, change))
            except Exception as e:
                continue
        
        # Sort by absolute change
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        return [m[0] for m in movers[:limit]]
    
    def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Get comprehensive data for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")
            
            if len(hist) < 20:
                return None
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            
            # Price changes
            change_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
            change_3d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-4]) - 1) * 100 if len(hist) >= 4 else change_1d
            change_5d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-6]) - 1) * 100 if len(hist) >= 6 else change_3d
            
            # Volume ratio
            avg_volume = hist['Volume'].iloc[-20:].mean()
            today_volume = hist['Volume'].iloc[-1]
            volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0
            
            # RSI (14-period)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50
            
            # 52-week high/low distance
            high_52w = info.get('fiftyTwoWeekHigh', current_price)
            low_52w = info.get('fiftyTwoWeekLow', current_price)
            distance_from_high = ((current_price - high_52w) / high_52w) * 100
            distance_from_low = ((current_price - low_52w) / low_52w) * 100
            
            # Gap
            prev_close = hist['Close'].iloc[-2]
            today_open = hist['Open'].iloc[-1]
            gap_percent = ((today_open - prev_close) / prev_close) * 100
            
            return StockData(
                symbol=symbol,
                price=current_price,
                change_1d=change_1d,
                change_3d=change_3d,
                change_5d=change_5d,
                volume_ratio=volume_ratio,
                rsi=rsi if not np.isnan(rsi) else 50,
                distance_from_52w_high=distance_from_high,
                distance_from_52w_low=distance_from_low,
                gap_percent=gap_percent,
                sector=info.get('sector', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE')
            )
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None


class OllamaScout:
    """Base class for Ollama-powered scouts."""
    
    def __init__(self, model: str):
        self.model = model
    
    async def query(self, prompt: str, timeout: int = 120) -> str:
        """Query Ollama model."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1000,
                }
            }
            try:
                async with session.post(
                    OLLAMA_URL, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('response', '')
                    else:
                        return f"Error: {resp.status}"
            except asyncio.TimeoutError:
                return "Error: Timeout"
            except Exception as e:
                return f"Error: {e}"


class TechnicalScout(OllamaScout):
    """Fast technical analysis scout."""
    
    def __init__(self):
        super().__init__(MODELS["technical"])
    
    async def analyze(self, data: StockData) -> Dict:
        """Analyze technicals and return score."""
        prompt = f"""Analyze this stock technically and score 0-100 for mean reversion potential.

STOCK: {data.symbol}
Price: ${data.price:.2f}
1-Day Change: {data.change_1d:+.1f}%
3-Day Change: {data.change_3d:+.1f}%
5-Day Change: {data.change_5d:+.1f}%
RSI(14): {data.rsi:.1f}
Volume Ratio (vs 20d avg): {data.volume_ratio:.1f}x
Distance from 52W High: {data.distance_from_52w_high:.1f}%
Distance from 52W Low: {data.distance_from_52w_low:.1f}%
Today's Gap: {data.gap_percent:+.1f}%

SCORING CRITERIA:
- RSI < 30 or > 70 = oversold/overbought (high score)
- High volume on move = capitulation/exhaustion (high score)  
- At 52W extremes = potential reversal (high score)
- Gap = potential gap fill (high score)

Respond in JSON format ONLY:
{{"score": 0-100, "direction": "BOUNCE" or "FADE", "reason": "brief reason"}}
"""
        response = await self.query(prompt, timeout=30)
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        return {"score": 50, "direction": "NEUTRAL", "reason": "Parse error"}


class FundamentalScout(OllamaScout):
    """Fundamental analysis scout."""
    
    def __init__(self):
        super().__init__(MODELS["fundamental"])
    
    async def analyze(self, data: StockData) -> Dict:
        """Analyze fundamentals and return score."""
        pe_str = f"{data.pe_ratio:.1f}" if data.pe_ratio else "N/A"
        mcap_b = data.market_cap / 1e9 if data.market_cap else 0
        
        prompt = f"""Analyze if this stock's move is justified fundamentally. Score 0-100 for mean reversion potential.

STOCK: {data.symbol}
Sector: {data.sector}
Market Cap: ${mcap_b:.1f}B
P/E Ratio: {pe_str}
Recent Move: {data.change_3d:+.1f}% (3 days)

HIGH SCORE IF:
- Quality company with temporary selloff
- No fundamental reason for the move
- Sector rotation, not company-specific issue
- Oversold blue chip

LOW SCORE IF:
- Earnings miss/guidance cut
- Company-specific bad news
- Structural problem
- Meme stock pump with no substance

Respond in JSON format ONLY:
{{"score": 0-100, "quality": "HIGH/MEDIUM/LOW", "reason": "brief reason"}}
"""
        response = await self.query(prompt, timeout=60)
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        return {"score": 50, "quality": "MEDIUM", "reason": "Parse error"}


class ScoutCoordinator:
    """Coordinates all scouts and produces final rankings."""
    
    def __init__(self):
        self.data_fetcher = MarketDataFetcher()
        self.technical_scout = TechnicalScout()
        self.fundamental_scout = FundamentalScout()
        self.ranker = OllamaScout(MODELS["ranker"])
    
    async def scan_stock(self, symbol: str) -> Optional[Dict]:
        """Run all scouts on a single stock."""
        print(f"  📊 Scanning {symbol}...")
        
        # Get data
        data = self.data_fetcher.get_stock_data(symbol)
        if not data:
            return None
        
        # Run scouts in parallel
        tech_task = self.technical_scout.analyze(data)
        fund_task = self.fundamental_scout.analyze(data)
        
        tech_result, fund_result = await asyncio.gather(tech_task, fund_task)
        
        # Combine scores
        tech_score = tech_result.get('score', 50)
        fund_score = fund_result.get('score', 50)
        combined = (tech_score * 0.6) + (fund_score * 0.4)  # Weight technical higher
        
        # Determine direction
        direction = tech_result.get('direction', 'NEUTRAL')
        if direction == 'NEUTRAL':
            direction = 'BOUNCE' if data.change_3d < -3 else 'FADE' if data.change_3d > 3 else 'NEUTRAL'
        
        # Calculate entry/target/stop
        if direction == 'BOUNCE':
            entry_low = data.price * 0.98
            entry_high = data.price * 1.01
            target = data.price * 1.10  # 10% bounce target
            stop = data.price * 0.95
        else:  # FADE
            entry_low = data.price * 0.99
            entry_high = data.price * 1.02
            target = data.price * 0.90  # 10% fade target
            stop = data.price * 1.05
        
        # Confidence
        if combined >= 75:
            confidence = "HIGH"
        elif combined >= 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            "symbol": symbol,
            "price": data.price,
            "change_1d": data.change_1d,
            "change_3d": data.change_3d,
            "rsi": data.rsi,
            "volume_ratio": data.volume_ratio,
            "technical_score": tech_score,
            "fundamental_score": fund_score,
            "combined_score": combined,
            "direction": direction,
            "confidence": confidence,
            "technical_reason": tech_result.get('reason', ''),
            "fundamental_reason": fund_result.get('reason', ''),
            "fundamental_quality": fund_result.get('quality', 'MEDIUM'),
            "entry_zone": [round(entry_low, 2), round(entry_high, 2)],
            "target": round(target, 2),
            "stop": round(stop, 2),
        }
    
    async def run_scan(self, min_change: float = 3.0) -> Dict:
        """Run full market scan."""
        print("\n" + "="*60)
        print("🔍 SCOUT SWARM - MARKET SCAN")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Get movers
        print("\n📈 Finding biggest movers...")
        movers = self.data_fetcher.get_biggest_movers(min_change=min_change, limit=30)
        print(f"   Found {len(movers)} stocks with >{min_change}% moves")
        
        # Scan each
        print("\n🔬 Running scouts...")
        results = []
        for symbol in movers:
            result = await self.scan_stock(symbol)
            if result and result['direction'] != 'NEUTRAL':
                results.append(result)
        
        # Separate bounce and fade
        bounce_setups = [r for r in results if r['direction'] == 'BOUNCE']
        fade_setups = [r for r in results if r['direction'] == 'FADE']
        
        # Sort by combined score
        bounce_setups.sort(key=lambda x: x['combined_score'], reverse=True)
        fade_setups.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Top 5 each
        top_bounces = bounce_setups[:5]
        top_fades = fade_setups[:5]
        
        report = {
            "scan_time": datetime.now().isoformat(),
            "scan_date": date.today().isoformat(),
            "stocks_scanned": len(movers),
            "bounce_candidates": len(bounce_setups),
            "fade_candidates": len(fade_setups),
            "top_bounces": top_bounces,
            "top_fades": top_fades,
        }
        
        return report
    
    def format_report(self, report: Dict) -> str:
        """Format report for display/Telegram."""
        lines = []
        lines.append("═" * 50)
        lines.append(f"🔍 SCOUT REPORT - {report['scan_date']}")
        lines.append(f"   Scanned: {report['stocks_scanned']} stocks")
        lines.append("═" * 50)
        
        lines.append("\n🟢 TOP BOUNCE SETUPS (Buy the dip):\n")
        for i, setup in enumerate(report['top_bounces'], 1):
            lines.append(f"#{i} {setup['symbol']} @ ${setup['price']:.2f}")
            lines.append(f"   Move: {setup['change_3d']:+.1f}% (3d) | RSI: {setup['rsi']:.0f}")
            lines.append(f"   Score: {setup['combined_score']:.0f}/100 ({setup['confidence']})")
            lines.append(f"   Tech: {setup['technical_reason']}")
            lines.append(f"   Fund: {setup['fundamental_reason']}")
            lines.append(f"   Entry: ${setup['entry_zone'][0]:.2f}-${setup['entry_zone'][1]:.2f}")
            lines.append(f"   Target: ${setup['target']:.2f} | Stop: ${setup['stop']:.2f}")
            lines.append("")
        
        lines.append("\n🔴 TOP FADE SETUPS (Short the rip):\n")
        for i, setup in enumerate(report['top_fades'], 1):
            lines.append(f"#{i} {setup['symbol']} @ ${setup['price']:.2f}")
            lines.append(f"   Move: {setup['change_3d']:+.1f}% (3d) | RSI: {setup['rsi']:.0f}")
            lines.append(f"   Score: {setup['combined_score']:.0f}/100 ({setup['confidence']})")
            lines.append(f"   Tech: {setup['technical_reason']}")
            lines.append(f"   Fund: {setup['fundamental_reason']}")
            lines.append(f"   Entry: ${setup['entry_zone'][0]:.2f}-${setup['entry_zone'][1]:.2f}")
            lines.append(f"   Target: ${setup['target']:.2f} | Stop: ${setup['stop']:.2f}")
            lines.append("")
        
        lines.append("═" * 50)
        return "\n".join(lines)


async def run_morning_scan():
    """Run morning scan (pre-market)."""
    coordinator = ScoutCoordinator()
    report = await coordinator.run_scan(min_change=3.0)
    
    # Save report
    save_path = f"/home/jbot/trading_ai/scout_swarm/reports/scout_report_{date.today().isoformat()}_morning.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print formatted report
    formatted = coordinator.format_report(report)
    print(formatted)
    
    # Save formatted
    txt_path = save_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write(formatted)
    
    # Send to Telegram
    telegram_msg = "🌅 MORNING " + format_telegram_report(report)
    send_telegram(telegram_msg)
    
    print(f"\n✅ Report saved: {save_path}")
    return report


async def run_evening_scan():
    """Run evening scan (after close)."""
    coordinator = ScoutCoordinator()
    report = await coordinator.run_scan(min_change=3.0)
    
    # Save report
    save_path = f"/home/jbot/trading_ai/scout_swarm/reports/scout_report_{date.today().isoformat()}_evening.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    formatted = coordinator.format_report(report)
    print(formatted)
    
    txt_path = save_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write(formatted)
    
    # Send to Telegram
    telegram_msg = "🌆 EVENING " + format_telegram_report(report)
    send_telegram(telegram_msg)
    
    print(f"\n✅ Report saved: {save_path}")
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "morning":
            asyncio.run(run_morning_scan())
        elif sys.argv[1] == "evening":
            asyncio.run(run_evening_scan())
        else:
            print("Usage: python swarm.py [morning|evening]")
    else:
        # Default: run scan now
        asyncio.run(run_morning_scan())
