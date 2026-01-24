"""
BTC Coupling Analyzer - Track IREN's correlation/decoupling from BTC

Paul's Thesis:
- IREN was built on BTC mining infrastructure
- Now pivoting to AI datacenters with MASSIVE power demand
- Legacy power infrastructure = 3-5 year competitive moat
- AI datacenter revenue growing >> BTC mining
- IREN gradually DECOUPLING from BTC price action

This module tracks and quantifies the decoupling phenomenon.
"""
import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BTCCouplingAnalyzer:
    """
    Track if IREN is coupled or decoupling from BTC.
    
    Paul's thesis: AI datacenter demand > BTC mining, so IREN
    should gradually decouple from BTC price movements.
    """
    
    # Thresholds for coupling status
    COUPLED_THRESHOLD = 0.70      # Above = strongly coupled
    TRANSITIONING_THRESHOLD = 0.50  # Between = transitioning
    DECOUPLED_THRESHOLD = 0.50    # Below = decoupled
    
    # Historical beta range for IREN vs BTC
    HISTORICAL_BETA = 1.5  # IREN typically moves 1.5x BTC
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes
        self._cache = {}
    
    def _get_btc_prices(self, days: int = 90) -> List[float]:
        """Get BTC daily closing prices"""
        try:
            # Use Yahoo Finance for BTC
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(period=f"{days}d")
            if hist.empty:
                raise ValueError("No BTC data")
            return hist['Close'].tolist()
        except Exception as e:
            logger.error(f"Failed to get BTC prices: {e}")
            # Fallback to CoinGecko
            try:
                url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
                params = {'vs_currency': 'usd', 'days': days}
                response = requests.get(url, params=params, timeout=15)
                data = response.json()
                prices = [p[1] for p in data['prices']]
                return prices
            except:
                return []
    
    def _get_iren_prices(self, days: int = 90) -> List[float]:
        """Get IREN daily closing prices"""
        try:
            iren = yf.Ticker("IREN")
            hist = iren.history(period=f"{days}d")
            if hist.empty:
                raise ValueError("No IREN data")
            return hist['Close'].tolist()
        except Exception as e:
            logger.error(f"Failed to get IREN prices: {e}")
            return []
    
    def calculate_correlation(self, days: int = 30) -> float:
        """
        Calculate rolling correlation between IREN and BTC.
        
        Returns correlation coefficient (-1 to 1).
        Higher = more coupled.
        """
        btc_prices = self._get_btc_prices(days + 5)  # Extra buffer
        iren_prices = self._get_iren_prices(days + 5)
        
        if len(btc_prices) < days or len(iren_prices) < days:
            logger.warning("Insufficient data for correlation")
            return 0.75  # Default assumption
        
        # Align lengths
        min_len = min(len(btc_prices), len(iren_prices), days)
        btc_prices = btc_prices[-min_len:]
        iren_prices = iren_prices[-min_len:]
        
        # Calculate daily returns
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        iren_returns = np.diff(iren_prices) / iren_prices[:-1]
        
        # Calculate correlation
        if len(btc_returns) > 2:
            correlation = np.corrcoef(btc_returns, iren_returns)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return float(correlation)
        
        return 0.75
    
    def calculate_beta(self, days: int = 30) -> float:
        """
        Calculate IREN's beta to BTC.
        
        Beta > 1 = IREN moves more than BTC
        Beta < 1 = IREN moves less than BTC
        """
        btc_prices = self._get_btc_prices(days + 5)
        iren_prices = self._get_iren_prices(days + 5)
        
        if len(btc_prices) < 10 or len(iren_prices) < 10:
            return self.HISTORICAL_BETA
        
        min_len = min(len(btc_prices), len(iren_prices), days)
        btc_prices = btc_prices[-min_len:]
        iren_prices = iren_prices[-min_len:]
        
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        iren_returns = np.diff(iren_prices) / iren_prices[:-1]
        
        # Beta = Cov(IREN, BTC) / Var(BTC)
        cov = np.cov(iren_returns, btc_returns)[0, 1]
        var = np.var(btc_returns)
        
        if var > 0:
            beta = cov / var
            return float(np.clip(beta, 0, 5))  # Reasonable bounds
        
        return self.HISTORICAL_BETA
    
    def get_coupling_trend(self, windows: List[int] = [7, 14, 30, 60]) -> Dict[str, float]:
        """
        Get correlation over multiple time windows to detect trend.
        
        If shorter windows have lower correlation than longer windows,
        IREN is decoupling from BTC.
        """
        correlations = {}
        for window in windows:
            try:
                corr = self.calculate_correlation(days=window)
                correlations[f"{window}d"] = round(corr, 3)
            except:
                correlations[f"{window}d"] = None
        
        return correlations
    
    def get_coupling_status(self) -> Dict[str, Any]:
        """
        Main analysis method - returns complete coupling analysis.
        
        Returns:
        {
            'correlation_30d': 0.65,
            'correlation_7d': 0.45,
            'trend': 'DECREASING',  # INCREASING, STABLE, DECREASING
            'status': 'DECOUPLING',  # COUPLED, TRANSITIONING, DECOUPLED
            'beta': 1.35,
            'analysis': {...},
            'driver': 'AI datacenter revenue growing vs BTC mining',
            'recommendation': 'Trade IREN on fundamentals, use BTC as secondary'
        }
        """
        # Get correlations for different time windows
        corr_7d = self.calculate_correlation(7)
        corr_14d = self.calculate_correlation(14)
        corr_30d = self.calculate_correlation(30)
        corr_60d = self.calculate_correlation(60)
        
        # Get beta
        beta = self.calculate_beta(30)
        
        # Determine trend
        if corr_7d < corr_30d - 0.1:
            trend = "DECREASING"  # Decoupling accelerating
        elif corr_7d > corr_30d + 0.1:
            trend = "INCREASING"  # Re-coupling
        else:
            trend = "STABLE"
        
        # Determine status
        avg_recent_corr = (corr_7d + corr_14d) / 2
        if avg_recent_corr >= self.COUPLED_THRESHOLD:
            status = "COUPLED"
            recommendation = "Use BTC as leading indicator for IREN entries"
        elif avg_recent_corr >= self.TRANSITIONING_THRESHOLD:
            status = "TRANSITIONING"
            recommendation = "Watch both BTC and IREN fundamentals"
        else:
            status = "DECOUPLED"
            recommendation = "Trade IREN on its own merit - AI datacenter thesis"
        
        # Get current day's analysis
        btc_data = self._get_realtime_btc()
        iren_data = self._get_realtime_iren()
        
        # Calculate expected vs actual move
        if btc_data and iren_data:
            btc_change = btc_data.get('change_24h', 0)
            iren_change = iren_data.get('change_pct', 0)
            expected_iren = btc_change * beta
            actual_vs_expected = iren_change - expected_iren
            
            if actual_vs_expected > 2:
                move_explanation = "IREN outperforming BTC - AI catalyst?"
            elif actual_vs_expected < -2:
                move_explanation = "IREN underperforming BTC - possible decoupling"
            else:
                move_explanation = "IREN tracking BTC normally"
        else:
            expected_iren = 0
            actual_vs_expected = 0
            move_explanation = "Insufficient data"
        
        # Determine driver based on trend
        if status == "COUPLED":
            driver = "BTC mining still primary revenue driver"
        elif trend == "DECREASING":
            driver = "AI datacenter contracts growing, reducing BTC dependency"
        else:
            driver = "Mixed revenue: BTC mining + AI datacenter"
        
        return {
            'correlation_7d': round(corr_7d, 3),
            'correlation_14d': round(corr_14d, 3),
            'correlation_30d': round(corr_30d, 3),
            'correlation_60d': round(corr_60d, 3),
            'correlation_trend': self.get_coupling_trend(),
            'trend': trend,
            'status': status,
            'beta': round(beta, 2),
            'analysis': {
                'btc_price': btc_data.get('price') if btc_data else None,
                'btc_change_24h': btc_data.get('change_24h') if btc_data else None,
                'iren_price': iren_data.get('price') if iren_data else None,
                'iren_change_pct': iren_data.get('change_pct') if iren_data else None,
                'expected_iren_move': round(expected_iren, 2) if expected_iren else None,
                'actual_vs_expected': round(actual_vs_expected, 2) if actual_vs_expected else None,
                'move_explanation': move_explanation
            },
            'driver': driver,
            'recommendation': recommendation,
            'pauls_thesis': {
                'core_insight': "Legacy BTC infrastructure being repurposed for AI datacenters",
                'competitive_moat': "3-5 year power infrastructure advantage",
                'trend_direction': "IREN transitioning from BTC miner to AI datacenter operator",
                'trading_implication': "Less BTC correlation over time, more fundamental-driven"
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def _get_realtime_btc(self) -> Optional[Dict]:
        """Get real-time BTC data"""
        try:
            url = 'https://api.coingecko.com/api/v3/coins/bitcoin'
            params = {'localization': 'false'}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            return {
                'price': data['market_data']['current_price']['usd'],
                'change_24h': data['market_data']['price_change_percentage_24h']
            }
        except:
            return None
    
    def _get_realtime_iren(self) -> Optional[Dict]:
        """Get real-time IREN data"""
        try:
            iren = yf.Ticker("IREN")
            hist = iren.history(period='2d')
            if hist.empty:
                return None
            
            current = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
            
            return {
                'price': current,
                'change_pct': ((current - prev) / prev) * 100
            }
        except:
            return None
    
    def should_use_btc_signal(self) -> Dict[str, Any]:
        """
        Should we use BTC as a leading indicator for IREN trades?
        
        Returns recommendation with reasoning.
        """
        status = self.get_coupling_status()
        
        use_btc = status['status'] == 'COUPLED'
        confidence = status['correlation_7d'] * 100
        
        return {
            'use_btc_signal': use_btc,
            'confidence': round(confidence, 1),
            'coupling_status': status['status'],
            'correlation': status['correlation_7d'],
            'recommendation': status['recommendation'],
            'reason': (
                f"7-day correlation: {status['correlation_7d']:.2f} | "
                f"Status: {status['status']} | "
                f"Trend: {status['trend']}"
            )
        }
    
    def get_trading_mode(self) -> str:
        """
        Get the appropriate trading mode for IREN.
        
        Returns:
        - "BTC_LED": BTC drives IREN, use BTC signals
        - "HYBRID": Watch both BTC and IREN fundamentals
        - "FUNDAMENTAL": Trade IREN on its own AI datacenter thesis
        """
        status = self.get_coupling_status()
        
        if status['status'] == 'COUPLED':
            return "BTC_LED"
        elif status['status'] == 'TRANSITIONING':
            return "HYBRID"
        else:
            return "FUNDAMENTAL"


# Singleton instance
_analyzer_instance = None

def get_coupling_analyzer() -> BTCCouplingAnalyzer:
    """Get singleton analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = BTCCouplingAnalyzer()
    return _analyzer_instance


if __name__ == "__main__":
    analyzer = BTCCouplingAnalyzer()
    
    print("=" * 60)
    print("BTC-IREN COUPLING ANALYSIS")
    print("=" * 60)
    
    status = analyzer.get_coupling_status()
    
    print(f"\nCorrelation (7d):  {status['correlation_7d']:.3f}")
    print(f"Correlation (30d): {status['correlation_30d']:.3f}")
    print(f"Correlation (60d): {status['correlation_60d']:.3f}")
    print(f"\nBeta: {status['beta']:.2f}x")
    print(f"Trend: {status['trend']}")
    print(f"Status: {status['status']}")
    print(f"\nDriver: {status['driver']}")
    print(f"Recommendation: {status['recommendation']}")
    
    print(f"\n{'='*60}")
    print("PAUL'S THESIS:")
    print(f"{'='*60}")
    for key, value in status['pauls_thesis'].items():
        print(f"  {key}: {value}")
