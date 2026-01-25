"""
Real-Time Correlation Monitor
Tracks rolling correlations between Gold and Forex pairs
Alerts on correlation breakdowns and regime changes
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
from scipy import stats

# Data directory
DATA_DIR = Path(__file__).parent.parent / "neo_gold"
DATA_DIR.mkdir(exist_ok=True)
CORRELATION_HISTORY_FILE = DATA_DIR / "correlation_history.json"


class CorrelationMonitor:
    """
    Monitor real-time correlations between Gold and Forex pairs
    Alert on significant correlation changes
    """
    
    def __init__(self):
        # Pairs to monitor with their expected correlations
        self.monitored_pairs = {
            'AUDUSD': {'expected': 0.80, 'yf_symbol': 'AUDUSD=X'},
            'NZDUSD': {'expected': 0.65, 'yf_symbol': 'NZDUSD=X'},
            'EURUSD': {'expected': 0.50, 'yf_symbol': 'EURUSD=X'},
            'USDCHF': {'expected': -0.70, 'yf_symbol': 'USDCHF=X'},
            'USDJPY': {'expected': -0.60, 'yf_symbol': 'USDJPY=X'},
            'USDCAD': {'expected': -0.40, 'yf_symbol': 'USDCAD=X'},
        }
        
        # Alert thresholds
        self.alert_threshold = 0.20  # Alert if correlation shifts by 20%
        self.breakdown_threshold = 0.30  # Major breakdown if shift > 30%
        
        # Historical data
        self.historical_correlations = self._load_history()
        
    def _load_history(self) -> Dict:
        """Load historical correlation data"""
        if CORRELATION_HISTORY_FILE.exists():
            with open(CORRELATION_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return {'correlations': [], 'last_update': None}
    
    def _save_history(self):
        """Save correlation history"""
        with open(CORRELATION_HISTORY_FILE, 'w') as f:
            json.dump(self.historical_correlations, f, indent=2)
    
    def get_gold_data(self, period: str = '1mo', interval: str = '1h') -> pd.Series:
        """Fetch Gold price data"""
        try:
            gold = yf.Ticker("GC=F")
            hist = gold.history(period=period, interval=interval)
            if hist.empty:
                gold = yf.Ticker("XAUUSD=X")
                hist = gold.history(period=period, interval=interval)
            return hist['Close']
        except Exception as e:
            print(f"Error fetching Gold data: {e}")
            return pd.Series()
    
    def get_forex_data(self, pair: str, period: str = '1mo', interval: str = '1h') -> pd.Series:
        """Fetch Forex pair data"""
        try:
            config = self.monitored_pairs.get(pair)
            if not config:
                return pd.Series()
            
            ticker = yf.Ticker(config['yf_symbol'])
            hist = ticker.history(period=period, interval=interval)
            return hist['Close']
        except Exception as e:
            print(f"Error fetching {pair} data: {e}")
            return pd.Series()
    
    def calculate_rolling_correlation(self, pair: str, window: int = 30) -> Dict:
        """
        Calculate rolling correlation between Gold and a forex pair
        
        Args:
            pair: Forex pair code (e.g., 'AUDUSD')
            window: Number of periods for rolling correlation
            
        Returns:
            Dict with current correlation, trend, and statistics
        """
        gold_data = self.get_gold_data(period='1mo', interval='1h')
        forex_data = self.get_forex_data(pair, period='1mo', interval='1h')
        
        if gold_data.empty or forex_data.empty:
            return {
                'pair': pair,
                'correlation': None,
                'error': 'Insufficient data'
            }
        
        # Align data
        combined = pd.DataFrame({
            'gold': gold_data,
            'forex': forex_data
        }).dropna()
        
        if len(combined) < window:
            return {
                'pair': pair,
                'correlation': None,
                'error': f'Need at least {window} data points'
            }
        
        # Calculate returns
        combined['gold_returns'] = combined['gold'].pct_change()
        combined['forex_returns'] = combined['forex'].pct_change()
        combined = combined.dropna()
        
        # Current correlation (last N periods)
        current_corr = combined['gold_returns'].iloc[-window:].corr(
            combined['forex_returns'].iloc[-window:]
        )
        
        # Previous correlation (N periods before)
        if len(combined) >= window * 2:
            prev_corr = combined['gold_returns'].iloc[-window*2:-window].corr(
                combined['forex_returns'].iloc[-window*2:-window]
            )
        else:
            prev_corr = current_corr
        
        # Calculate rolling correlation series
        rolling_corr = combined['gold_returns'].rolling(window).corr(
            combined['forex_returns']
        ).dropna()
        
        # Statistics
        corr_mean = rolling_corr.mean()
        corr_std = rolling_corr.std()
        corr_min = rolling_corr.min()
        corr_max = rolling_corr.max()
        
        # Trend detection
        if len(rolling_corr) >= 10:
            recent_trend = rolling_corr.iloc[-10:].values
            slope, _, r_value, _, _ = stats.linregress(range(len(recent_trend)), recent_trend)
            
            if slope > 0.01:
                trend = 'STRENGTHENING'
            elif slope < -0.01:
                trend = 'WEAKENING'
            else:
                trend = 'STABLE'
        else:
            trend = 'UNKNOWN'
            slope = 0
            r_value = 0
        
        # Compare to expected
        expected = self.monitored_pairs[pair]['expected']
        deviation = current_corr - expected
        deviation_pct = abs(deviation / expected) * 100 if expected != 0 else 0
        
        return {
            'pair': pair,
            'correlation': round(current_corr, 3),
            'previous': round(prev_corr, 3),
            'change': round(current_corr - prev_corr, 3),
            'expected': expected,
            'deviation': round(deviation, 3),
            'deviation_pct': round(deviation_pct, 1),
            'trend': trend,
            'trend_slope': round(slope, 4),
            'statistics': {
                'mean': round(corr_mean, 3),
                'std': round(corr_std, 3),
                'min': round(corr_min, 3),
                'max': round(corr_max, 3)
            },
            'window': window,
            'data_points': len(combined),
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_correlation_breakdown(self) -> List[Dict]:
        """
        Alert when historical correlations break down
        Returns list of pairs with significant correlation changes
        """
        alerts = []
        
        for pair, config in self.monitored_pairs.items():
            corr_data = self.calculate_rolling_correlation(pair)
            
            if corr_data.get('correlation') is None:
                continue
            
            current = corr_data['correlation']
            expected = config['expected']
            deviation = abs(current - expected)
            
            # Check for breakdown
            if deviation >= self.breakdown_threshold:
                alert_type = 'BREAKDOWN'
                severity = 'HIGH'
            elif deviation >= self.alert_threshold:
                alert_type = 'DEVIATION'
                severity = 'MEDIUM'
            else:
                continue  # No alert needed
            
            alerts.append({
                'pair': pair,
                'alert_type': alert_type,
                'severity': severity,
                'current_correlation': current,
                'expected_correlation': expected,
                'deviation': round(deviation, 3),
                'trend': corr_data['trend'],
                'message': self._generate_alert_message(pair, current, expected, alert_type),
                'trading_implication': self._get_trading_implication(pair, current, expected),
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _generate_alert_message(self, pair: str, current: float, expected: float, alert_type: str) -> str:
        """Generate human-readable alert message"""
        direction = "higher" if current > expected else "lower"
        
        if alert_type == 'BREAKDOWN':
            return f"⚠️ CORRELATION BREAKDOWN: {pair}-Gold correlation is {abs(current):.2f}, " \
                   f"significantly {direction} than expected {abs(expected):.2f}. " \
                   f"Historical relationship may have changed!"
        else:
            return f"📊 CORRELATION SHIFT: {pair}-Gold correlation drifting {direction}. " \
                   f"Current: {current:.2f}, Expected: {expected:.2f}"
    
    def _get_trading_implication(self, pair: str, current: float, expected: float) -> str:
        """Get trading implication of correlation change"""
        if abs(current) < abs(expected) * 0.5:
            return f"{pair} no longer following Gold reliably. Reduce correlated position sizes or trade independently."
        elif abs(current) > abs(expected):
            return f"{pair} showing stronger than expected Gold correlation. Can increase position sizes on correlated trades."
        elif (current > 0 and expected < 0) or (current < 0 and expected > 0):
            return f"⚠️ CORRELATION FLIP: {pair} has reversed its relationship with Gold! Review all correlated positions immediately."
        else:
            return f"{pair} correlation weakening but still directionally correct. Monitor closely."
    
    def get_correlation_heatmap(self) -> Dict:
        """
        Return current correlations for all monitored pairs
        Suitable for dashboard visualization
        """
        heatmap = {}
        details = {}
        
        for pair in self.monitored_pairs.keys():
            corr_data = self.calculate_rolling_correlation(pair)
            
            if corr_data.get('correlation') is not None:
                heatmap[f'XAUUSD_{pair}'] = corr_data['correlation']
                details[pair] = {
                    'correlation': corr_data['correlation'],
                    'expected': corr_data['expected'],
                    'deviation': corr_data['deviation'],
                    'trend': corr_data['trend'],
                    'status': self._get_correlation_status(corr_data)
                }
            else:
                heatmap[f'XAUUSD_{pair}'] = None
                details[pair] = {'error': corr_data.get('error', 'Unknown error')}
        
        return {
            'heatmap': heatmap,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_correlation_status(self, corr_data: Dict) -> str:
        """Get status label for correlation"""
        deviation_pct = corr_data.get('deviation_pct', 0)
        
        if deviation_pct <= 10:
            return 'NORMAL'
        elif deviation_pct <= 25:
            return 'SHIFTED'
        elif deviation_pct <= 50:
            return 'WEAK'
        else:
            return 'BROKEN'
    
    def get_regime_analysis(self) -> Dict:
        """
        Analyze current market regime based on correlation patterns
        """
        correlations = {}
        for pair in self.monitored_pairs.keys():
            corr_data = self.calculate_rolling_correlation(pair)
            if corr_data.get('correlation') is not None:
                correlations[pair] = corr_data['correlation']
        
        # Analyze regime
        positive_corrs = [v for k, v in correlations.items() 
                        if self.monitored_pairs[k]['expected'] > 0]
        negative_corrs = [v for k, v in correlations.items() 
                        if self.monitored_pairs[k]['expected'] < 0]
        
        avg_positive = np.mean(positive_corrs) if positive_corrs else 0
        avg_negative = np.mean(negative_corrs) if negative_corrs else 0
        
        # Determine regime
        if avg_positive > 0.6 and avg_negative < -0.5:
            regime = 'RISK_OFF'
            description = 'Strong safe-haven flows. Gold up, USD down, risk currencies following Gold.'
        elif avg_positive < 0.3 and avg_negative > -0.3:
            regime = 'RISK_ON'
            description = 'Risk appetite high. Correlations weakening as markets diverge.'
        elif abs(avg_positive - avg_negative) < 0.2:
            regime = 'TRANSITIONAL'
            description = 'Market regime changing. Correlations in flux.'
        else:
            regime = 'NORMAL'
            description = 'Standard correlation patterns. Trade normal strategies.'
        
        return {
            'regime': regime,
            'description': description,
            'avg_positive_correlation': round(avg_positive, 3),
            'avg_negative_correlation': round(avg_negative, 3),
            'correlations': correlations,
            'recommendation': self._get_regime_recommendation(regime),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_regime_recommendation(self, regime: str) -> str:
        """Get trading recommendation for current regime"""
        recommendations = {
            'RISK_OFF': 'Favor Gold longs with AUD/USD longs and USD/JPY shorts. High confidence in correlations.',
            'RISK_ON': 'Correlations less reliable. Trade each pair on its own merits. Reduce correlated position sizes.',
            'TRANSITIONAL': 'Wait for clearer signals. Reduce position sizes. Watch for correlation stabilization.',
            'NORMAL': 'Standard correlation trading active. Use full position sizes on correlated trades.'
        }
        return recommendations.get(regime, 'Monitor market conditions.')
    
    def update_history(self):
        """Update historical correlation records"""
        heatmap = self.get_correlation_heatmap()
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'correlations': heatmap['heatmap']
        }
        
        self.historical_correlations['correlations'].append(record)
        self.historical_correlations['last_update'] = datetime.now().isoformat()
        
        # Keep only last 1000 records
        if len(self.historical_correlations['correlations']) > 1000:
            self.historical_correlations['correlations'] = \
                self.historical_correlations['correlations'][-1000:]
        
        self._save_history()
        
        return record
    
    def get_full_report(self) -> Dict:
        """
        Generate comprehensive correlation report
        """
        heatmap_data = self.get_correlation_heatmap()
        breakdown_alerts = self.detect_correlation_breakdown()
        regime = self.get_regime_analysis()
        
        # Individual pair analysis
        pair_analysis = {}
        for pair in self.monitored_pairs.keys():
            pair_analysis[pair] = self.calculate_rolling_correlation(pair)
        
        return {
            'summary': {
                'regime': regime['regime'],
                'regime_description': regime['description'],
                'alerts_count': len(breakdown_alerts),
                'recommendation': regime['recommendation']
            },
            'heatmap': heatmap_data['heatmap'],
            'pair_details': heatmap_data['details'],
            'pair_analysis': pair_analysis,
            'alerts': breakdown_alerts,
            'regime_analysis': regime,
            'timestamp': datetime.now().isoformat()
        }


# Test
if __name__ == "__main__":
    monitor = CorrelationMonitor()
    
    print("=" * 70)
    print("🔗 GOLD-FOREX CORRELATION MONITOR")
    print("=" * 70)
    
    report = monitor.get_full_report()
    
    print(f"""
MARKET REGIME: {report['summary']['regime']}
{report['summary']['regime_description']}

RECOMMENDATION: {report['summary']['recommendation']}

CORRELATION HEATMAP:
""")
    
    for pair_key, corr in report['heatmap'].items():
        pair = pair_key.replace('XAUUSD_', '')
        details = report['pair_details'].get(pair, {})
        expected = monitor.monitored_pairs.get(pair, {}).get('expected', 0)
        status = details.get('status', 'N/A')
        
        if corr is not None:
            bar = '█' * int(abs(corr) * 10)
            sign = '+' if corr > 0 else ''
            print(f"  {pair:8} {sign}{corr:+.3f} {bar:10} (exp: {expected:+.2f}) [{status}]")
        else:
            print(f"  {pair:8} N/A")
    
    print(f"\nALERTS ({report['summary']['alerts_count']}):")
    if report['alerts']:
        for alert in report['alerts']:
            print(f"  ⚠️ {alert['pair']}: {alert['message']}")
            print(f"     → {alert['trading_implication']}")
    else:
        print("  ✅ All correlations within expected ranges")
