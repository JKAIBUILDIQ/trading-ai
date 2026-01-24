"""
IREN Daily Brief Generator - Claudia's Morning Report

Generates comprehensive daily intelligence brief for Paul.
Scheduled to run at 6 AM UTC (before market open).

Author: Claudia (Research Lead)
For: Paul (Investment Partner)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np

from news_feed import IRENNewsFeed, get_iren_news

logger = logging.getLogger(__name__)

# Output directory for reports
REPORTS_DIR = Path('/home/jbot/trading_ai/reports/iren')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class IRENDailyBrief:
    """
    Generate daily IREN intelligence brief.
    
    Contents:
    1. Price action summary
    2. Key news (last 24 hours)
    3. Technical analysis
    4. BTC correlation update
    5. Options flow summary
    6. Upcoming catalysts
    7. Risk alerts
    8. Claudia's take
    """
    
    def __init__(self):
        self.ticker = yf.Ticker('IREN')
        self.btc = yf.Ticker('BTC-USD')
        self.news_feed = IRENNewsFeed()
        
    def get_price_summary(self) -> Dict[str, Any]:
        """Get price action summary"""
        try:
            # Get recent history
            hist = self.ticker.history(period='5d')
            if hist.empty:
                return {'error': 'No price data'}
            
            today = hist.iloc[-1]
            yesterday = hist.iloc[-2] if len(hist) > 1 else today
            
            # Get current quote
            info = self.ticker.info or {}
            
            return {
                'current_price': float(today['Close']),
                'previous_close': float(yesterday['Close']),
                'change': float(today['Close'] - yesterday['Close']),
                'change_pct': ((today['Close'] / yesterday['Close']) - 1) * 100,
                'open': float(today['Open']),
                'high': float(today['High']),
                'low': float(today['Low']),
                'volume': int(today['Volume']),
                'avg_volume': int(hist['Volume'].mean()),
                'volume_ratio': today['Volume'] / hist['Volume'].mean(),
                '52wk_high': info.get('fiftyTwoWeekHigh', 0),
                '52wk_low': info.get('fiftyTwoWeekLow', 0),
                'market_cap': info.get('marketCap', 0)
            }
        except Exception as e:
            logger.error(f"Price summary failed: {e}")
            return {'error': str(e)}
    
    def get_technical_analysis(self) -> Dict[str, Any]:
        """Get technical indicators"""
        try:
            hist = self.ticker.history(period='30d')
            if hist.empty:
                return {'error': 'No data'}
            
            close = hist['Close']
            
            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            # Moving averages
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            ema_9 = float(close.ewm(span=9).mean().iloc[-1])
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - signal_line
            
            # Trend determination
            current_price = float(close.iloc[-1])
            
            if current_price > sma_20 and current_rsi > 50:
                trend = 'BULLISH'
                trend_strength = min(100, int((current_rsi - 50) * 2 + 50))
            elif current_price < sma_20 and current_rsi < 50:
                trend = 'BEARISH'
                trend_strength = min(100, int((50 - current_rsi) * 2 + 50))
            else:
                trend = 'NEUTRAL'
                trend_strength = 50
            
            return {
                'rsi_14': current_rsi,
                'rsi_status': 'OVERBOUGHT' if current_rsi > 70 else ('OVERSOLD' if current_rsi < 30 else 'NEUTRAL'),
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_9': ema_9,
                'macd': float(macd_line.iloc[-1]),
                'macd_signal': float(signal_line.iloc[-1]),
                'macd_histogram': float(macd_histogram.iloc[-1]),
                'macd_status': 'BULLISH' if macd_histogram.iloc[-1] > 0 else 'BEARISH',
                'trend': trend,
                'trend_strength': trend_strength,
                'above_sma20': current_price > sma_20,
                'above_sma50': current_price > sma_50 if sma_50 else None
            }
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {'error': str(e)}
    
    def get_btc_correlation(self) -> Dict[str, Any]:
        """Calculate BTC correlation and status"""
        try:
            # Get 30-day history for both
            iren_hist = self.ticker.history(period='30d')
            btc_hist = self.btc.history(period='30d')
            
            if iren_hist.empty or btc_hist.empty:
                return {'error': 'Insufficient data'}
            
            # Calculate returns
            iren_returns = iren_hist['Close'].pct_change().dropna()
            btc_returns = btc_hist['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = iren_returns.index.intersection(btc_returns.index)
            iren_returns = iren_returns[common_dates]
            btc_returns = btc_returns[common_dates]
            
            # Calculate correlation
            correlation_30d = float(iren_returns.corr(btc_returns))
            
            # Calculate 7-day correlation
            iren_7d = iren_returns[-7:]
            btc_7d = btc_returns[-7:]
            correlation_7d = float(iren_7d.corr(btc_7d)) if len(iren_7d) >= 5 else None
            
            # BTC current status
            btc_price = float(btc_hist['Close'].iloc[-1])
            btc_change = ((btc_hist['Close'].iloc[-1] / btc_hist['Close'].iloc[-2]) - 1) * 100
            
            # Determine coupling status
            if abs(correlation_30d) > 0.7:
                coupling_status = 'COUPLED'
            elif abs(correlation_30d) > 0.5:
                coupling_status = 'TRANSITIONING'
            else:
                coupling_status = 'DECOUPLED'
            
            return {
                'correlation_30d': correlation_30d,
                'correlation_7d': correlation_7d,
                'coupling_status': coupling_status,
                'btc_price': btc_price,
                'btc_change_pct': float(btc_change),
                'btc_trend': 'BULLISH' if btc_change > 1 else ('BEARISH' if btc_change < -1 else 'NEUTRAL'),
                'decoupling_thesis': 'CONFIRMED' if coupling_status in ['DECOUPLED', 'TRANSITIONING'] else 'MONITORING'
            }
        except Exception as e:
            logger.error(f"BTC correlation failed: {e}")
            return {'error': str(e)}
    
    def get_upcoming_catalysts(self) -> List[Dict]:
        """Get upcoming catalysts and events"""
        catalysts = [
            {
                'date': '2026-02-05',
                'event': 'Q4 2025 Earnings',
                'impact': 'HIGH',
                'days_away': (datetime(2026, 2, 5) - datetime.now()).days,
                'notes': 'Watch for AI datacenter revenue growth'
            },
            {
                'date': '2026-Q4',
                'event': "Paul's Land Deal Completion",
                'impact': 'MAJOR',
                'days_away': None,
                'notes': 'Texas land + utility rights'
            }
        ]
        
        # Filter to upcoming only
        return [c for c in catalysts if c['days_away'] is None or c['days_away'] > 0]
    
    def get_risk_alerts(self, technicals: Dict, btc: Dict) -> List[Dict]:
        """Generate risk alerts based on current conditions"""
        alerts = []
        
        # RSI overbought
        if technicals.get('rsi_14', 50) > 70:
            alerts.append({
                'level': 'WARNING',
                'type': 'OVERBOUGHT',
                'message': f"RSI at {technicals['rsi_14']:.1f} - approaching overbought territory"
            })
        
        # Earnings proximity
        days_to_earnings = (datetime(2026, 2, 5) - datetime.now()).days
        if days_to_earnings <= 14:
            alerts.append({
                'level': 'INFO',
                'type': 'EARNINGS',
                'message': f"Earnings in {days_to_earnings} days - avoid short-term options!"
            })
        
        # BTC divergence
        if btc.get('coupling_status') == 'DECOUPLED' and btc.get('btc_trend') == 'BEARISH':
            alerts.append({
                'level': 'INFO',
                'type': 'DECOUPLING',
                'message': "BTC bearish but IREN decoupled - trading on own fundamentals"
            })
        
        return alerts
    
    def generate_claudia_take(self, price: Dict, technicals: Dict, btc: Dict, news: Dict) -> str:
        """Generate Claudia's analysis summary"""
        
        # Sentiment from news
        sentiment = news.get('sentiment_summary', {})
        news_sentiment = 'positive' if sentiment.get('positive', 0) > sentiment.get('negative', 0) else 'neutral'
        
        # Build the take
        take_parts = []
        
        # Trend assessment
        trend = technicals.get('trend', 'NEUTRAL')
        if trend == 'BULLISH':
            take_parts.append(f"IREN showing {technicals.get('trend_strength', 50)}% bullish trend strength")
        elif trend == 'BEARISH':
            take_parts.append(f"IREN in bearish territory, exercise caution")
        else:
            take_parts.append("IREN in consolidation mode")
        
        # BTC relationship
        coupling = btc.get('coupling_status', 'UNKNOWN')
        if coupling == 'DECOUPLED':
            take_parts.append("Decoupling from BTC confirmed - trade on IREN fundamentals")
        elif coupling == 'TRANSITIONING':
            take_parts.append("BTC correlation declining - decoupling thesis progressing")
        
        # Earnings
        days_to_earnings = (datetime(2026, 2, 5) - datetime.now()).days
        if days_to_earnings <= 14:
            take_parts.append(f"⚠️ Earnings in {days_to_earnings} days - position carefully")
        
        # News sentiment
        if news_sentiment == 'positive':
            take_parts.append("News flow is positive")
        
        # Final recommendation
        if trend == 'BULLISH' and coupling in ['DECOUPLED', 'TRANSITIONING']:
            take_parts.append("📈 Favorable setup for LONG positions (Paul's thesis)")
        
        return ". ".join(take_parts) + "."
    
    def generate_brief(self) -> Dict[str, Any]:
        """Generate the complete daily brief"""
        
        logger.info("Generating IREN daily brief...")
        
        # Gather all data
        price = self.get_price_summary()
        technicals = self.get_technical_analysis()
        btc = self.get_btc_correlation()
        news = get_iren_news(limit=10)
        catalysts = self.get_upcoming_catalysts()
        
        # Generate insights
        alerts = self.get_risk_alerts(technicals, btc)
        claudia_take = self.generate_claudia_take(price, technicals, btc, news)
        
        brief = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'ticker': 'IREN',
            'company': 'IREN Limited',
            
            'price_summary': price,
            'technical_analysis': technicals,
            'btc_correlation': btc,
            'news_summary': {
                'total_items': news.get('total_items', 0),
                'sentiment': news.get('sentiment_summary', {}),
                'top_headlines': [n['title'] for n in news.get('news', [])[:5]]
            },
            'catalysts': catalysts,
            'risk_alerts': alerts,
            'claudia_take': claudia_take,
            
            # Trading recommendation
            'recommendation': {
                'action': 'BUY' if technicals.get('trend') == 'BULLISH' else 'HOLD',
                'confidence': technicals.get('trend_strength', 50),
                'strikes': ['$60 (ATM)', '$70 (OTM)', '$80 (Far OTM)'],
                'preferred_expiry': 'Feb 20 or Feb 27 (post-earnings)',
                'reasoning': claudia_take
            }
        }
        
        # Save the brief
        report_file = REPORTS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}-iren-brief.json"
        with open(report_file, 'w') as f:
            json.dump(brief, f, indent=2, default=str)
        
        # Also save as markdown
        self._save_markdown_brief(brief)
        
        logger.info(f"Brief saved to {report_file}")
        return brief
    
    def _save_markdown_brief(self, brief: Dict):
        """Save brief as readable markdown"""
        
        md_file = REPORTS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}-iren-brief.md"
        
        price = brief.get('price_summary', {})
        tech = brief.get('technical_analysis', {})
        btc = brief.get('btc_correlation', {})
        
        # Safe getters with defaults
        def safe_get(d, key, default=0):
            val = d.get(key, default)
            return default if val is None else val
        
        content = f"""# IREN Daily Intelligence Brief

**Date:** {brief.get('report_date', 'N/A')}
**Generated:** {brief.get('generated_at', 'N/A')}
**Analyst:** Claudia (Research Lead)
**For:** Paul (Investment Partner)

---

## 📈 Price Summary

| Metric | Value |
|--------|-------|
| **Current Price** | ${safe_get(price, 'current_price', 0):.2f} |
| **Change** | {safe_get(price, 'change_pct', 0):+.2f}% |
| **Open** | ${safe_get(price, 'open', 0):.2f} |
| **High** | ${safe_get(price, 'high', 0):.2f} |
| **Low** | ${safe_get(price, 'low', 0):.2f} |
| **Volume** | {safe_get(price, 'volume', 0):,} |
| **Vol Ratio** | {safe_get(price, 'volume_ratio', 1):.2f}x avg |

---

## 📊 Technical Analysis

| Indicator | Value | Status |
|-----------|-------|--------|
| **Trend** | {tech.get('trend', 'N/A')} | {safe_get(tech, 'trend_strength', 50)}% strength |
| **RSI(14)** | {safe_get(tech, 'rsi_14', 0):.1f} | {tech.get('rsi_status', 'N/A')} |
| **MACD** | {safe_get(tech, 'macd', 0):.3f} | {tech.get('macd_status', 'N/A')} |
| **SMA(20)** | ${safe_get(tech, 'sma_20', 0):.2f} | {'Above ✅' if tech.get('above_sma20') else 'Below ❌'} |

---

## ₿ BTC Correlation

| Metric | Value |
|--------|-------|
| **30-Day Correlation** | {safe_get(btc, 'correlation_30d', 0):.2f} |
| **7-Day Correlation** | {f"{btc.get('correlation_7d'):.2f}" if btc.get('correlation_7d') else 'N/A'} |
| **Coupling Status** | **{btc.get('coupling_status', 'N/A')}** |
| **BTC Price** | ${safe_get(btc, 'btc_price', 0):,.0f} |
| **BTC Trend** | {btc.get('btc_trend', 'N/A')} |

---

## 📰 News Summary

**Sentiment:** {brief['news_summary']['sentiment']}

**Top Headlines:**
"""
        for headline in brief['news_summary'].get('top_headlines', []):
            content += f"- {headline}\n"
        
        content += f"""

---

## 🎯 Upcoming Catalysts

"""
        for cat in brief.get('catalysts', []):
            days = f"({cat['days_away']} days)" if cat.get('days_away') else ""
            content += f"- **{cat['date']}** - {cat['event']} {days} - Impact: {cat['impact']}\n"
        
        content += f"""

---

## ⚠️ Risk Alerts

"""
        for alert in brief.get('risk_alerts', []):
            content += f"- **[{alert['level']}]** {alert['message']}\n"
        
        content += f"""

---

## 💡 Claudia's Take

{brief['claudia_take']}

---

## 🎯 Trading Recommendation

| Field | Value |
|-------|-------|
| **Action** | **{brief['recommendation']['action']}** |
| **Confidence** | {brief['recommendation']['confidence']}% |
| **Strikes** | {', '.join(brief['recommendation']['strikes'])} |
| **Expiry** | {brief['recommendation']['preferred_expiry']} |

---

*Generated by Claudia IREN Intelligence System*
*For Paul's eyes only - IN-HOUSE USE*
"""
        
        with open(md_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Markdown brief saved to {md_file}")


# Convenience function
def generate_iren_daily_brief() -> Dict[str, Any]:
    """Generate and return IREN daily brief"""
    briefer = IRENDailyBrief()
    return briefer.generate_brief()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Generating IREN Daily Brief...")
    brief = generate_iren_daily_brief()
    
    print("\n" + "="*60)
    print("IREN DAILY BRIEF SUMMARY")
    print("="*60)
    print(f"Price: ${brief['price_summary'].get('current_price', 0):.2f}")
    print(f"Trend: {brief['technical_analysis'].get('trend', 'N/A')} ({brief['technical_analysis'].get('trend_strength', 0)}%)")
    print(f"BTC Coupling: {brief['btc_correlation'].get('coupling_status', 'N/A')}")
    print(f"\nClaudia's Take: {brief['claudia_take']}")
    print(f"\nRecommendation: {brief['recommendation']['action']} ({brief['recommendation']['confidence']}%)")
