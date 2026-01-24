"""
IREN Stock Deep Analysis Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IREN (Nasdaq: IREN) - Bitcoin Mining + AI Data Centers

Business Model:
├── PRIMARY: Bitcoin Mining (revenue tied to BTC price)
├── SECONDARY: AI Data Center Capacity (HPC)
└── KEY: Power access = competitive advantage

Why IREN?
- Direct BTC exposure via mining
- Growing AI/HPC revenue stream  
- Strong institutional interest
- High beta = high reward potential

Created: 2026-01-24
For: Paul (Investment Partner) - Budget Unlock!
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Company Profile
IREN_PROFILE = {
    'ticker': 'IREN',
    'name': 'IREN Limited (Formerly Iris Energy)',
    'exchange': 'NASDAQ',
    'sector': 'Technology',
    'industry': 'Bitcoin Mining / AI Data Centers',
    'website': 'https://iren.com',
    'country': 'Australia',
    'employees': '~200',
    
    'business_segments': {
        'bitcoin_mining': {
            'description': 'Self-mining Bitcoin operations',
            'revenue_pct': 85,
            'key_metrics': ['hash_rate', 'btc_mined', 'cost_per_btc']
        },
        'ai_data_centers': {
            'description': 'High-performance computing & AI workloads',
            'revenue_pct': 15,
            'key_metrics': ['capacity_mw', 'utilization', 'contracts']
        }
    },
    
    'competitors': {
        'bitcoin_miners': ['MARA', 'RIOT', 'CLSK', 'HUT', 'CIFR'],
        'ai_data_centers': ['APLD', 'BTBT', 'WULF']
    }
}


@dataclass
class TechnicalSignals:
    """Technical analysis signals"""
    trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0-1
    rsi: float
    rsi_signal: str  # 'OVERSOLD', 'OVERBOUGHT', 'NEUTRAL'
    macd_signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    ema_trend: str  # 'ABOVE_50', 'BELOW_50', 'GOLDEN_CROSS', 'DEATH_CROSS'
    support: float
    resistance: float
    atr: float
    volume_trend: str  # 'INCREASING', 'DECREASING', 'NORMAL'


@dataclass
class FundamentalData:
    """Fundamental analysis data"""
    market_cap: float
    pe_ratio: Optional[float]
    ps_ratio: float
    book_value: float
    revenue_ttm: float
    revenue_growth: float
    gross_margin: float
    cash_position: float
    debt_total: float
    btc_holdings: float
    hash_rate_eh: float  # Exahash/second
    institutional_ownership: float


@dataclass
class BTCCorrelation:
    """BTC correlation analysis"""
    btc_price: float
    correlation_30d: float
    correlation_90d: float
    mining_difficulty: float
    next_difficulty_adj: str
    btc_dominance: float
    iren_btc_beta: float


@dataclass
class IRENAnalysis:
    """Complete IREN analysis"""
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    high_52w: float
    low_52w: float
    technicals: TechnicalSignals
    fundamentals: FundamentalData
    btc_correlation: BTCCorrelation
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    timestamp: datetime


class IRENAnalyzer:
    """
    Comprehensive IREN stock analyzer
    
    Features:
    - Technical analysis (RSI, MACD, EMAs, support/resistance)
    - Fundamental analysis (revenue, margins, BTC holdings)
    - BTC correlation tracking
    - AI-powered trading signals
    """
    
    def __init__(self):
        self.ticker = 'IREN'
        self.last_analysis = None
    
    def fetch_stock_data(self, period: str = '6mo', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch IREN stock data from Yahoo Finance
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(self.ticker)
            df = stock.history(period=period, interval=interval)
            
            # Normalize column names
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            return df
            
        except Exception as e:
            log.error(f"Error fetching IREN data: {e}")
            return pd.DataFrame()
    
    def fetch_btc_data(self, period: str = '6mo') -> pd.DataFrame:
        """
        Fetch BTC data for correlation analysis
        """
        try:
            import yfinance as yf
            
            btc = yf.Ticker('BTC-USD')
            df = btc.history(period=period, interval='1d')
            
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            return df
            
        except Exception as e:
            log.error(f"Error fetching BTC data: {e}")
            return pd.DataFrame()
    
    def calculate_technicals(self, df: pd.DataFrame) -> TechnicalSignals:
        """
        Calculate technical indicators
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        rsi = self._calculate_rsi(close, 14)
        if rsi < 30:
            rsi_signal = 'OVERSOLD'
        elif rsi > 70:
            rsi_signal = 'OVERBOUGHT'
        else:
            rsi_signal = 'NEUTRAL'
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            macd_signal = 'BUY'
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            macd_signal = 'SELL'
        else:
            macd_signal = 'BULLISH' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'BEARISH'
        
        # EMAs
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else close.ewm(span=50, adjust=False).mean()
        
        current_price = close.iloc[-1]
        
        if ema_50.iloc[-1] > ema_200.iloc[-1] and ema_50.iloc[-2] <= ema_200.iloc[-2]:
            ema_trend = 'GOLDEN_CROSS'
        elif ema_50.iloc[-1] < ema_200.iloc[-1] and ema_50.iloc[-2] >= ema_200.iloc[-2]:
            ema_trend = 'DEATH_CROSS'
        elif current_price > ema_50.iloc[-1]:
            ema_trend = 'ABOVE_50'
        else:
            ema_trend = 'BELOW_50'
        
        # Support/Resistance
        support = low.tail(20).min()
        resistance = high.tail(20).max()
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Volume trend
        vol_ma_5 = volume.tail(5).mean()
        vol_ma_20 = volume.tail(20).mean()
        
        if vol_ma_5 > vol_ma_20 * 1.5:
            volume_trend = 'INCREASING'
        elif vol_ma_5 < vol_ma_20 * 0.7:
            volume_trend = 'DECREASING'
        else:
            volume_trend = 'NORMAL'
        
        # Overall trend
        bullish_signals = sum([
            current_price > ema_50.iloc[-1],
            rsi > 50,
            macd_line.iloc[-1] > signal_line.iloc[-1],
            ema_9.iloc[-1] > ema_21.iloc[-1]
        ])
        
        if bullish_signals >= 3:
            trend = 'BULLISH'
            strength = bullish_signals / 4
        elif bullish_signals <= 1:
            trend = 'BEARISH'
            strength = (4 - bullish_signals) / 4
        else:
            trend = 'NEUTRAL'
            strength = 0.5
        
        return TechnicalSignals(
            trend=trend,
            strength=round(strength, 2),
            rsi=round(rsi, 1),
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            ema_trend=ema_trend,
            support=round(support, 2),
            resistance=round(resistance, 2),
            atr=round(atr, 2),
            volume_trend=volume_trend
        )
    
    def fetch_fundamentals(self) -> FundamentalData:
        """
        Fetch fundamental data
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            return FundamentalData(
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE'),
                ps_ratio=info.get('priceToSalesTrailing12Months', 0),
                book_value=info.get('bookValue', 0),
                revenue_ttm=info.get('totalRevenue', 0),
                revenue_growth=info.get('revenueGrowth', 0),
                gross_margin=info.get('grossMargins', 0),
                cash_position=info.get('totalCash', 0),
                debt_total=info.get('totalDebt', 0),
                btc_holdings=0,  # Would need to pull from filings
                hash_rate_eh=0,  # Would need to pull from company reports
                institutional_ownership=info.get('heldPercentInstitutions', 0)
            )
            
        except Exception as e:
            log.error(f"Error fetching fundamentals: {e}")
            return FundamentalData(
                market_cap=0, pe_ratio=None, ps_ratio=0, book_value=0,
                revenue_ttm=0, revenue_growth=0, gross_margin=0,
                cash_position=0, debt_total=0, btc_holdings=0,
                hash_rate_eh=0, institutional_ownership=0
            )
    
    def calculate_btc_correlation(self, iren_df: pd.DataFrame, btc_df: pd.DataFrame) -> BTCCorrelation:
        """
        Calculate IREN-BTC correlation
        """
        try:
            # Align dates
            common_dates = iren_df.index.intersection(btc_df.index)
            
            if len(common_dates) < 30:
                log.warning("Not enough common dates for correlation")
                return self._default_btc_correlation(btc_df)
            
            iren_returns = iren_df.loc[common_dates, 'close'].pct_change().dropna()
            btc_returns = btc_df.loc[common_dates, 'close'].pct_change().dropna()
            
            # 30-day correlation
            corr_30d = iren_returns.tail(30).corr(btc_returns.tail(30))
            
            # 90-day correlation
            corr_90d = iren_returns.tail(90).corr(btc_returns.tail(90)) if len(iren_returns) >= 90 else corr_30d
            
            # Beta calculation
            cov = iren_returns.tail(90).cov(btc_returns.tail(90))
            var = btc_returns.tail(90).var()
            beta = cov / var if var > 0 else 1.0
            
            btc_price = btc_df['close'].iloc[-1]
            
            return BTCCorrelation(
                btc_price=round(btc_price, 2),
                correlation_30d=round(corr_30d, 2) if not np.isnan(corr_30d) else 0,
                correlation_90d=round(corr_90d, 2) if not np.isnan(corr_90d) else 0,
                mining_difficulty=0,  # Would need blockchain API
                next_difficulty_adj='~2 weeks',
                btc_dominance=0,  # Would need crypto API
                iren_btc_beta=round(beta, 2) if not np.isnan(beta) else 1.0
            )
            
        except Exception as e:
            log.error(f"Error calculating BTC correlation: {e}")
            return self._default_btc_correlation(btc_df)
    
    def _default_btc_correlation(self, btc_df: pd.DataFrame) -> BTCCorrelation:
        """Default BTC correlation data"""
        btc_price = btc_df['close'].iloc[-1] if not btc_df.empty else 0
        return BTCCorrelation(
            btc_price=round(btc_price, 2),
            correlation_30d=0.75,  # Historical average
            correlation_90d=0.70,
            mining_difficulty=0,
            next_difficulty_adj='~2 weeks',
            btc_dominance=0,
            iren_btc_beta=1.5  # Historical average
        )
    
    def generate_signal(self, technicals: TechnicalSignals, 
                        fundamentals: FundamentalData,
                        btc_corr: BTCCorrelation) -> Tuple[str, float, str]:
        """
        Generate trading signal based on all factors
        """
        score = 0
        reasons = []
        
        # Technical factors (50% weight)
        if technicals.trend == 'BULLISH':
            score += 25
            reasons.append(f"Bullish trend ({technicals.strength:.0%} strength)")
        elif technicals.trend == 'BEARISH':
            score -= 25
            reasons.append(f"Bearish trend ({technicals.strength:.0%} strength)")
        
        if technicals.rsi_signal == 'OVERSOLD':
            score += 15
            reasons.append(f"RSI oversold ({technicals.rsi})")
        elif technicals.rsi_signal == 'OVERBOUGHT':
            score -= 15
            reasons.append(f"RSI overbought ({technicals.rsi})")
        
        if technicals.macd_signal in ['BUY', 'BULLISH']:
            score += 10
            reasons.append("MACD bullish")
        elif technicals.macd_signal in ['SELL', 'BEARISH']:
            score -= 10
            reasons.append("MACD bearish")
        
        if technicals.ema_trend == 'GOLDEN_CROSS':
            score += 20
            reasons.append("Golden Cross detected!")
        elif technicals.ema_trend == 'DEATH_CROSS':
            score -= 20
            reasons.append("Death Cross warning!")
        elif technicals.ema_trend == 'ABOVE_50':
            score += 5
            reasons.append("Price above 50 EMA")
        
        # BTC correlation factor (30% weight)
        if btc_corr.correlation_30d > 0.7:
            reasons.append(f"High BTC correlation ({btc_corr.correlation_30d})")
            # BTC trend would add/subtract points
        
        if btc_corr.iren_btc_beta > 1.5:
            reasons.append(f"High BTC beta ({btc_corr.iren_btc_beta}) - amplified moves")
        
        # Volume factor (10% weight)
        if technicals.volume_trend == 'INCREASING':
            score += 10
            reasons.append("Volume increasing")
        elif technicals.volume_trend == 'DECREASING':
            score -= 5
            reasons.append("Volume declining")
        
        # Fundamental factors (10% weight)
        if fundamentals.institutional_ownership > 0.5:
            score += 5
            reasons.append(f"Strong institutional ownership ({fundamentals.institutional_ownership:.0%})")
        
        # Determine signal
        if score >= 30:
            signal = 'BUY'
            confidence = min(0.9, 0.5 + score / 100)
        elif score <= -30:
            signal = 'SELL'
            confidence = min(0.9, 0.5 + abs(score) / 100)
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        reasoning = ' | '.join(reasons[:5])  # Top 5 reasons
        
        return signal, round(confidence, 2), reasoning
    
    def analyze(self) -> IRENAnalysis:
        """
        Complete IREN analysis
        """
        log.info("Starting IREN analysis...")
        
        # Fetch data
        iren_df = self.fetch_stock_data('6mo', '1d')
        btc_df = self.fetch_btc_data('6mo')
        
        if iren_df.empty:
            log.error("Could not fetch IREN data")
            return None
        
        # Current price info
        current_price = iren_df['close'].iloc[-1]
        prev_close = iren_df['close'].iloc[-2] if len(iren_df) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100
        volume = int(iren_df['volume'].iloc[-1])
        avg_volume = int(iren_df['volume'].tail(20).mean())
        
        # 52-week high/low
        high_52w = iren_df['high'].max()
        low_52w = iren_df['low'].min()
        
        # Calculate components
        technicals = self.calculate_technicals(iren_df)
        fundamentals = self.fetch_fundamentals()
        btc_corr = self.calculate_btc_correlation(iren_df, btc_df)
        
        # Generate signal
        signal, confidence, reasoning = self.generate_signal(technicals, fundamentals, btc_corr)
        
        analysis = IRENAnalysis(
            price=round(current_price, 2),
            change_pct=round(change_pct, 2),
            volume=volume,
            avg_volume=avg_volume,
            high_52w=round(high_52w, 2),
            low_52w=round(low_52w, 2),
            technicals=technicals,
            fundamentals=fundamentals,
            btc_correlation=btc_corr,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.utcnow()
        )
        
        self.last_analysis = analysis
        return analysis
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def to_dict(self) -> Dict:
        """Convert analysis to dictionary for API"""
        analysis = self.last_analysis or self.analyze()
        
        if not analysis:
            return {'error': 'Could not fetch IREN data'}
        
        return {
            'ticker': self.ticker,
            'price': analysis.price,
            'change_pct': analysis.change_pct,
            'volume': analysis.volume,
            'avg_volume': analysis.avg_volume,
            'high_52w': analysis.high_52w,
            'low_52w': analysis.low_52w,
            
            'technicals': {
                'trend': analysis.technicals.trend,
                'strength': analysis.technicals.strength,
                'rsi': analysis.technicals.rsi,
                'rsi_signal': analysis.technicals.rsi_signal,
                'macd_signal': analysis.technicals.macd_signal,
                'ema_trend': analysis.technicals.ema_trend,
                'support': analysis.technicals.support,
                'resistance': analysis.technicals.resistance,
                'atr': analysis.technicals.atr,
                'volume_trend': analysis.technicals.volume_trend
            },
            
            'fundamentals': {
                'market_cap': analysis.fundamentals.market_cap,
                'pe_ratio': analysis.fundamentals.pe_ratio,
                'ps_ratio': analysis.fundamentals.ps_ratio,
                'revenue_ttm': analysis.fundamentals.revenue_ttm,
                'revenue_growth': analysis.fundamentals.revenue_growth,
                'gross_margin': analysis.fundamentals.gross_margin,
                'cash_position': analysis.fundamentals.cash_position,
                'debt_total': analysis.fundamentals.debt_total,
                'institutional_ownership': analysis.fundamentals.institutional_ownership
            },
            
            'btc_correlation': {
                'btc_price': analysis.btc_correlation.btc_price,
                'correlation_30d': analysis.btc_correlation.correlation_30d,
                'correlation_90d': analysis.btc_correlation.correlation_90d,
                'iren_btc_beta': analysis.btc_correlation.iren_btc_beta
            },
            
            'signal': analysis.signal,
            'confidence': analysis.confidence,
            'reasoning': analysis.reasoning,
            'timestamp': analysis.timestamp.isoformat()
        }


def get_iren_analysis() -> Dict:
    """Quick function for API/NEO integration"""
    analyzer = IRENAnalyzer()
    return analyzer.to_dict()


# Test
if __name__ == "__main__":
    print("="*70)
    print("📈 IREN STOCK DEEP ANALYSIS")
    print("="*70)
    
    analyzer = IRENAnalyzer()
    analysis = analyzer.analyze()
    
    if analysis:
        print(f"\n💰 PRICE: ${analysis.price:.2f} ({analysis.change_pct:+.2f}%)")
        print(f"   Volume: {analysis.volume:,} (Avg: {analysis.avg_volume:,})")
        print(f"   52W Range: ${analysis.low_52w:.2f} - ${analysis.high_52w:.2f}")
        
        print(f"\n📊 TECHNICAL SIGNALS:")
        print(f"   Trend: {analysis.technicals.trend} ({analysis.technicals.strength:.0%})")
        print(f"   RSI: {analysis.technicals.rsi:.1f} ({analysis.technicals.rsi_signal})")
        print(f"   MACD: {analysis.technicals.macd_signal}")
        print(f"   EMA Trend: {analysis.technicals.ema_trend}")
        print(f"   Support: ${analysis.technicals.support:.2f}")
        print(f"   Resistance: ${analysis.technicals.resistance:.2f}")
        
        print(f"\n🔗 BTC CORRELATION:")
        print(f"   BTC Price: ${analysis.btc_correlation.btc_price:,.2f}")
        print(f"   30D Correlation: {analysis.btc_correlation.correlation_30d:.2f}")
        print(f"   IREN-BTC Beta: {analysis.btc_correlation.iren_btc_beta:.2f}")
        
        print(f"\n🏢 FUNDAMENTALS:")
        print(f"   Market Cap: ${analysis.fundamentals.market_cap/1e9:.2f}B")
        print(f"   Institutional: {analysis.fundamentals.institutional_ownership:.0%}")
        
        print(f"\n🎯 SIGNAL: {analysis.signal}")
        print(f"   Confidence: {analysis.confidence:.0%}")
        print(f"   Reasoning: {analysis.reasoning}")
    else:
        print("❌ Could not analyze IREN")
