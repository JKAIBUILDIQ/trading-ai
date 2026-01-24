"""
Greek Analyzer - Fight the Greeks for Large Options Positions

Essential for Paul's high-volume options trading.
At 50-200+ contracts, Greeks can make or break profits.

Key Greeks:
- Delta: Directional exposure ($ move per $1 stock move)
- Theta: Time decay (daily bleed)
- Gamma: Delta acceleration (how fast delta changes)
- Vega: Volatility exposure (IV crush risk)
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import scipy for better calculations
try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed - using simplified Greeks")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class GreekAnalyzer:
    """
    Fight the Greeks - essential for large positions.
    
    At Paul's scale (50-200 contracts), Greeks matter:
    - Theta of -$450/day = -$3,150/week
    - Vega of +$850 = $8,500 loss on 10% IV drop
    - Gamma can accelerate profits OR losses
    """
    
    def __init__(self):
        self.risk_free_rate = 0.05  # ~5% risk-free rate
        
    def analyze_position(self, contracts: int, strike: float, 
                         expiry: str, current_price: float,
                         option_type: str = 'call',
                         premium: float = None,
                         iv: float = None) -> Dict:
        """
        Full Greeks analysis for a position.
        
        Args:
            contracts: Number of contracts (each = 100 shares)
            strike: Strike price
            expiry: Expiration date (YYYY-MM-DD)
            current_price: Current stock price
            option_type: 'call' or 'put'
            premium: Current option premium (optional, will estimate)
            iv: Implied volatility (optional, will estimate)
        
        Returns comprehensive Greeks analysis with recommendations.
        """
        # Calculate days to expiry
        try:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        except:
            expiry_date = datetime.strptime(expiry, '%m/%d/%Y')
        
        dte = (expiry_date - datetime.now()).days
        dte = max(1, dte)  # At least 1 day
        
        # Estimate IV if not provided (use historical or default)
        if iv is None:
            iv = self._estimate_iv(current_price, strike, dte)
        
        # Calculate Black-Scholes Greeks
        greeks = self._calculate_greeks(
            S=current_price,
            K=strike,
            T=dte / 365,
            r=self.risk_free_rate,
            sigma=iv,
            option_type=option_type
        )
        
        # Scale to position size (contracts * 100)
        multiplier = contracts * 100
        
        # Calculate position-level Greeks
        total_delta = greeks['delta'] * multiplier
        total_gamma = greeks['gamma'] * multiplier
        total_theta = greeks['theta'] * multiplier  # Daily
        total_vega = greeks['vega'] * multiplier
        
        # Dollar values
        dollar_delta = total_delta * current_price * 0.01  # $ per 1% move
        
        # Estimate premium if not provided
        if premium is None:
            premium = self._estimate_premium(greeks, current_price, strike, option_type)
        
        position_value = premium * multiplier
        
        # Calculate breakeven
        if option_type.lower() == 'call':
            breakeven = strike + premium
        else:
            breakeven = strike - premium
        
        breakeven_pct = ((breakeven - current_price) / current_price) * 100
        
        # Days until theta eats all extrinsic value
        extrinsic = premium - max(0, (current_price - strike) if option_type.lower() == 'call' else (strike - current_price))
        days_to_zero = extrinsic / abs(greeks['theta']) if greeks['theta'] != 0 else dte
        
        # Build recommendations
        recommendations = self._build_recommendations(
            dte=dte,
            theta=total_theta,
            vega=total_vega,
            iv=iv,
            position_value=position_value,
            delta=total_delta
        )
        
        return {
            'position': {
                'contracts': contracts,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'option_type': option_type,
                'current_price': current_price,
                'premium': round(premium, 2),
                'position_value': round(position_value, 2)
            },
            'delta': {
                'per_contract': round(greeks['delta'], 4),
                'total': round(total_delta, 2),
                'dollar_delta': round(dollar_delta, 2),
                'equivalent_shares': int(total_delta * 100),
                'interpretation': f"${dollar_delta:,.0f} P&L per 1% stock move"
            },
            'theta': {
                'per_contract': round(greeks['theta'], 4),
                'daily_decay': round(total_theta, 2),
                'weekly_decay': round(total_theta * 5, 2),
                'weekend_decay': round(total_theta * 3, 2),
                'days_to_worthless': round(days_to_zero, 1),
                'interpretation': f"Losing ${abs(total_theta):,.0f}/day to time decay"
            },
            'gamma': {
                'per_contract': round(greeks['gamma'], 4),
                'total': round(total_gamma, 2),
                'acceleration': round(total_gamma * 100, 2),
                'risk_level': 'HIGH' if total_gamma > 50 else 'MODERATE' if total_gamma > 20 else 'LOW',
                'interpretation': f"Delta changes by {total_gamma:.1f} per $1 stock move"
            },
            'vega': {
                'per_contract': round(greeks['vega'], 4),
                'total': round(total_vega, 2),
                'iv_current': round(iv * 100, 1),
                'iv_percentile': self._estimate_iv_percentile(iv),
                'risk_level': 'HIGH' if iv > 0.8 else 'MODERATE' if iv > 0.5 else 'LOW',
                'interpretation': f"${total_vega:,.0f} loss per 1% IV drop"
            },
            'breakeven': {
                'price': round(breakeven, 2),
                'percent_move_needed': round(breakeven_pct, 2),
                'days_until_theta_eats_profit': round(days_to_zero, 1)
            },
            'risk_metrics': {
                'max_loss': round(position_value, 2),
                'theta_vs_value': round((abs(total_theta) / position_value) * 100, 2),  # % of position lost daily
                'time_risk': 'CRITICAL' if dte < 7 else 'HIGH' if dte < 14 else 'MODERATE'
            },
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_greeks(self, S: float, K: float, T: float, 
                          r: float, sigma: float, option_type: str) -> Dict:
        """
        Calculate Black-Scholes Greeks.
        
        S: Stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility (decimal)
        """
        if T <= 0:
            T = 0.001  # Avoid division by zero
        
        if HAS_SCIPY:
            # Use scipy for accurate normal CDF
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - 
                         r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                delta = norm.cdf(d1) - 1
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + 
                         r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% IV change
            
        else:
            # Simplified approximation without scipy
            moneyness = S / K
            
            if option_type.lower() == 'call':
                if moneyness > 1.1:
                    delta = 0.85
                elif moneyness > 1.0:
                    delta = 0.55 + (moneyness - 1.0) * 3
                elif moneyness > 0.9:
                    delta = 0.35 + (moneyness - 0.9) * 2
                else:
                    delta = 0.15
            else:
                if moneyness < 0.9:
                    delta = -0.85
                elif moneyness < 1.0:
                    delta = -0.55 - (1.0 - moneyness) * 3
                elif moneyness < 1.1:
                    delta = -0.35 - (1.1 - moneyness) * 2
                else:
                    delta = -0.15
            
            # Simplified gamma - highest ATM
            gamma = 0.05 * math.exp(-((moneyness - 1.0) ** 2) * 10)
            
            # Simplified theta - accelerates near expiry
            base_theta = sigma * S * 0.01 / math.sqrt(T * 365)
            theta = -base_theta / 365
            
            # Simplified vega
            vega = S * 0.01 * math.sqrt(T)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    def _estimate_iv(self, price: float, strike: float, dte: int) -> float:
        """Estimate IV based on moneyness and time"""
        moneyness = abs(price - strike) / price
        
        # Base IV around 50-80% for growth stocks like IREN
        base_iv = 0.65
        
        # Skew - OTM options have higher IV
        skew = moneyness * 0.5
        
        # Term structure - shorter dated = higher IV
        term_adj = 0.1 * (30 / max(dte, 7))
        
        return min(1.5, base_iv + skew + term_adj)
    
    def _estimate_iv_percentile(self, iv: float) -> int:
        """Estimate IV percentile (rough approximation)"""
        # IREN typically has high IV due to BTC correlation
        if iv > 1.0:
            return 95
        elif iv > 0.8:
            return 85
        elif iv > 0.6:
            return 65
        elif iv > 0.4:
            return 40
        else:
            return 20
    
    def _estimate_premium(self, greeks: Dict, price: float, 
                          strike: float, option_type: str) -> float:
        """Estimate option premium"""
        # Intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(0, price - strike)
        else:
            intrinsic = max(0, strike - price)
        
        # Rough extrinsic estimate based on delta
        delta = abs(greeks['delta'])
        extrinsic = price * 0.05 * delta  # ~5% of price weighted by delta
        
        return intrinsic + extrinsic
    
    def _build_recommendations(self, dte: int, theta: float, vega: float,
                               iv: float, position_value: float, delta: float) -> List[str]:
        """Build actionable recommendations"""
        recs = []
        
        # Theta warnings
        theta_pct = abs(theta) / position_value * 100 if position_value > 0 else 0
        if theta_pct > 3:
            recs.append(f"⚠️ CRITICAL: Losing {theta_pct:.1f}% of position daily to theta")
        elif theta_pct > 1:
            recs.append(f"Theta eating ${abs(theta):,.0f}/day - need move soon")
        
        # DTE warnings
        if dte < 7:
            recs.append("⚠️ < 7 DTE - Gamma risk HIGH, consider rolling")
        elif dte < 14:
            recs.append("14 DTE sweet spot ending - monitor closely")
        
        # IV warnings
        iv_pct = iv * 100
        if iv_pct > 80:
            recs.append(f"⚠️ IV at {iv_pct:.0f}% - HIGH crush risk, don't hold through catalyst")
        elif iv_pct > 60:
            recs.append(f"IV elevated at {iv_pct:.0f}% - watch for crush")
        
        # Vega exposure
        if abs(vega) > position_value * 0.1:
            recs.append(f"Vega exposure HIGH - 10% IV drop = ${abs(vega):.0f} loss")
        
        # General advice based on position
        if delta > 0:
            recs.append("Long delta - profits when IREN rises")
        else:
            recs.append("Short delta - profits when IREN falls")
        
        # Weekend warning
        today = datetime.now().weekday()
        if today >= 3:  # Thursday or later
            weekend_theta = abs(theta) * 3
            recs.append(f"Weekend coming - ${weekend_theta:,.0f} theta decay over weekend")
        
        return recs
    
    def optimal_entry_time(self, target_expiry: str) -> Dict:
        """
        When to enter to minimize Greek damage.
        
        Best: Monday-Tuesday (avoid paying for weekend theta)
        Worst: Thursday-Friday (weekend theta priced in)
        """
        try:
            expiry_date = datetime.strptime(target_expiry, '%Y-%m-%d')
        except:
            expiry_date = datetime.strptime(target_expiry, '%m/%d/%Y')
        
        dte = (expiry_date - datetime.now()).days
        
        return {
            'best_days': ['Monday', 'Tuesday'],
            'acceptable_days': ['Wednesday'],
            'avoid': ['Thursday', 'Friday'],
            'reasoning': [
                'Monday-Tuesday: Don\'t pay for weekend theta',
                'Wednesday: Acceptable if strong signal',
                'Thursday-Friday: Weekend decay priced in, avoid entry'
            ],
            'optimal_dte_range': '14-21 days',
            'dte_warning': 'CRITICAL' if dte < 7 else 'CAUTION' if dte < 10 else 'OK',
            'current_dte': dte,
            'recommendations': [
                f"Current DTE: {dte} days",
                "Sweet spot: 14-21 DTE for best theta/gamma balance",
                "< 7 DTE: Only for quick scalps, gamma can hurt you",
                "> 30 DTE: Moves too slow, paying for excess time"
            ]
        }
    
    def theta_decay_schedule(self, contracts: int, strike: float,
                             expiry: str, current_price: float,
                             premium: float = None) -> List[Dict]:
        """
        Day-by-day theta decay projection.
        Shows how position value erodes over time.
        """
        try:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        except:
            expiry_date = datetime.strptime(expiry, '%m/%d/%Y')
        
        dte = (expiry_date - datetime.now()).days
        schedule = []
        
        # Initial position analysis
        position = self.analyze_position(
            contracts=contracts,
            strike=strike,
            expiry=expiry,
            current_price=current_price,
            premium=premium
        )
        
        current_value = position['position']['position_value']
        daily_theta = abs(position['theta']['daily_decay'])
        
        for day in range(min(dte, 14)):  # Project up to 14 days
            date = datetime.now() + timedelta(days=day)
            day_name = date.strftime('%A')
            
            # Weekend decay
            if day_name == 'Saturday':
                decay = 0  # Priced in Friday
            elif day_name == 'Sunday':
                decay = 0  # Priced in Friday
            elif day_name == 'Friday':
                decay = daily_theta * 3  # Weekend decay
            else:
                decay = daily_theta
            
            # Theta accelerates as expiry approaches
            days_left = dte - day
            if days_left < 7:
                decay *= 1.5  # Acceleration
            elif days_left < 14:
                decay *= 1.2
            
            current_value -= decay
            current_value = max(0, current_value)
            
            schedule.append({
                'day': day,
                'date': date.strftime('%Y-%m-%d'),
                'day_name': day_name,
                'days_to_expiry': days_left,
                'decay': round(decay, 2),
                'remaining_value': round(current_value, 2),
                'pct_remaining': round((current_value / position['position']['position_value']) * 100, 1)
            })
        
        return schedule
    
    def iv_crush_risk(self, symbol: str = 'IREN', expiry: str = None) -> Dict:
        """
        Assess IV crush risk for position.
        """
        # Check for upcoming events
        events = self._check_upcoming_events(symbol)
        
        # Current IV assessment
        iv_assessment = {
            'current_iv_estimate': 65,  # Default for IREN
            'iv_percentile': 75,
            'historical_avg_iv': 55,
            'iv_premium': 10,  # Current vs historical
            'crush_risk': 'MODERATE'
        }
        
        # Adjust for events
        if events['earnings_soon']:
            iv_assessment['crush_risk'] = 'HIGH'
            iv_assessment['expected_crush'] = 15  # Expected IV drop %
        
        return {
            'symbol': symbol,
            'iv_assessment': iv_assessment,
            'events': events,
            'recommendations': [
                'IV elevated - consider selling premium or hedging',
                'If holding calls, exit before earnings to avoid crush',
                'BTC volatility also affects IREN IV'
            ]
        }
    
    def _check_upcoming_events(self, symbol: str) -> Dict:
        """Check for upcoming catalysts"""
        # In production, would check earnings calendar, etc.
        return {
            'earnings_soon': False,
            'next_earnings': 'TBD',
            'ex_dividend': None,
            'btc_halving': False,
            'other_catalysts': []
        }


# Singleton instance
_greek_analyzer = None

def get_greek_analyzer() -> GreekAnalyzer:
    """Get or create GreekAnalyzer singleton"""
    global _greek_analyzer
    if _greek_analyzer is None:
        _greek_analyzer = GreekAnalyzer()
    return _greek_analyzer


# CLI Testing
if __name__ == '__main__':
    print("=" * 60)
    print("📊 GREEK ANALYZER TEST - Paul's Position")
    print("=" * 60)
    
    analyzer = GreekAnalyzer()
    
    # Test Paul's typical position
    print("\n🎯 Analyzing: 150 contracts IREN $58 Calls, Feb 7 expiry")
    
    result = analyzer.analyze_position(
        contracts=150,
        strike=58,
        expiry='2026-02-07',
        current_price=56.68,
        option_type='call'
    )
    
    print(f"\n📈 POSITION:")
    print(f"   {result['position']['contracts']}x ${result['position']['strike']} {result['position']['option_type']}")
    print(f"   DTE: {result['position']['dte']} days")
    print(f"   Premium: ${result['position']['premium']}")
    print(f"   Position Value: ${result['position']['position_value']:,.2f}")
    
    print(f"\n📊 DELTA:")
    print(f"   Per Contract: {result['delta']['per_contract']}")
    print(f"   Total: {result['delta']['total']}")
    print(f"   Equivalent Shares: {result['delta']['equivalent_shares']}")
    print(f"   → {result['delta']['interpretation']}")
    
    print(f"\n⏰ THETA:")
    print(f"   Daily Decay: ${result['theta']['daily_decay']:,.2f}")
    print(f"   Weekend Decay: ${result['theta']['weekend_decay']:,.2f}")
    print(f"   Days to Worthless: {result['theta']['days_to_worthless']}")
    print(f"   → {result['theta']['interpretation']}")
    
    print(f"\n📈 GAMMA:")
    print(f"   Total: {result['gamma']['total']}")
    print(f"   Risk Level: {result['gamma']['risk_level']}")
    
    print(f"\n📉 VEGA:")
    print(f"   Total: ${result['vega']['total']:,.2f}")
    print(f"   IV Current: {result['vega']['iv_current']}%")
    print(f"   Risk Level: {result['vega']['risk_level']}")
    print(f"   → {result['vega']['interpretation']}")
    
    print(f"\n🎯 BREAKEVEN:")
    print(f"   Price: ${result['breakeven']['price']}")
    print(f"   Move Needed: {result['breakeven']['percent_move_needed']}%")
    
    print(f"\n⚠️ RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"   • {rec}")
    
    print("\n" + "=" * 60)
    print("✅ Greek Analyzer Ready!")
