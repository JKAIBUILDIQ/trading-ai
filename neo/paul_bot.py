"""
PaulBot - Real-Time Trading Intelligence for Paul's IREN Options

IN-HOUSE USE ONLY - Not for outside distribution

This is Paul's dedicated trading assistant that combines:
- NEO's BTC intelligence
- IREN correlation analysis
- Options Greeks analysis
- Liquidity/volume analysis
- Position sizing recommendations

Paul's Scale: 50-200+ contracts, making $50K-130K/week
At this level, we need INSTITUTIONAL-GRADE tools.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Import our analysis modules
try:
    from btc_analyzer import BTCAnalyzer, get_btc_analyzer
    from iren_analyzer import IRENAnalyzer, get_iren_analysis
    from greek_analyzer import GreekAnalyzer, get_greek_analyzer
    from liquidity_analyzer import LiquidityAnalyzer, get_liquidity_analyzer
except ImportError:
    # Relative imports for package usage
    from .btc_analyzer import BTCAnalyzer, get_btc_analyzer
    from .iren_analyzer import IRENAnalyzer, get_iren_analysis
    from .greek_analyzer import GreekAnalyzer, get_greek_analyzer
    from .liquidity_analyzer import LiquidityAnalyzer, get_liquidity_analyzer


class PaulBot:
    """
    Real-time trading intelligence for Paul's IREN options.
    
    Designed for high-volume trading where:
    - Greeks can destroy profits
    - Liquidity is critical
    - BTC correlation drives IREN
    - Clear, actionable signals needed
    """
    
    def __init__(self):
        self.btc = get_btc_analyzer()
        self.greeks = get_greek_analyzer()
        self.liquidity = get_liquidity_analyzer()
        
        # Paul's default preferences
        self.default_contracts = 100
        self.preferred_dte = (14, 21)  # 14-21 days
        self.max_spread_pct = 5.0
        self.min_volume = 2000
        
    def chat(self, query: str) -> Dict:
        """
        Natural language interface for Paul.
        
        Examples:
        - "What's the play today?"
        - "Volume check on $60 calls"
        - "Greeks on 150 contracts $58 calls"
        - "Can I exit 200 contracts?"
        - "Daily strategy"
        - "Weekly outlook"
        """
        query_lower = query.lower()
        
        # Route to appropriate handler
        if any(word in query_lower for word in ['play', 'strategy', 'signal', 'recommendation', 'today']):
            return self.get_daily_strategy()
        
        elif any(word in query_lower for word in ['weekly', 'outlook', 'week']):
            return self.get_weekly_strategy()
        
        elif 'volume' in query_lower or 'liquidity' in query_lower:
            # Extract strike if mentioned
            strike = self._extract_strike(query)
            contracts = self._extract_contracts(query) or 100
            return self.check_volume(strike or 60, contracts)
        
        elif 'greek' in query_lower:
            strike = self._extract_strike(query) or 58
            contracts = self._extract_contracts(query) or 100
            return self.analyze_greeks(contracts, strike)
        
        elif any(word in query_lower for word in ['exit', 'sell', 'close']):
            strike = self._extract_strike(query) or 60
            contracts = self._extract_contracts(query) or 100
            return self.get_exit_strategy(contracts, strike)
        
        elif 'size' in query_lower or 'position' in query_lower:
            strike = self._extract_strike(query) or 60
            return self.recommend_size(strike)
        
        elif 'btc' in query_lower or 'bitcoin' in query_lower:
            return self.get_btc_outlook()
        
        else:
            # Default to daily strategy
            return self.get_daily_strategy()
    
    def _extract_strike(self, query: str) -> Optional[float]:
        """Extract strike price from query"""
        # Look for patterns like $60, 60 calls, strike 60
        patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*call',
            r'(\d+\.?\d*)\s*put',
            r'strike\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return float(match.group(1))
        return None
    
    def _extract_contracts(self, query: str) -> Optional[int]:
        """Extract contract count from query"""
        patterns = [
            r'(\d+)\s*contract',
            r'(\d+)x',
            r'size\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None
    
    def get_daily_strategy(self) -> Dict:
        """
        Morning brief with complete daily strategy.
        """
        # Get all market intelligence
        btc_data = self.btc.get_btc_signal()
        btc_price = self.btc.get_btc_price()
        correlation = self.btc.get_btc_iren_correlation()
        iren_data = get_iren_analysis()
        
        # Find liquid strikes
        liquid_strikes = self.liquidity.find_liquid_strikes(
            symbol='IREN',
            expiry=self._get_optimal_expiry(),
            option_type='call',
            min_volume=self.min_volume
        )
        
        # Determine signal
        if btc_data['signal'] == 'BUY' and btc_data['confidence'] >= 70:
            action = 'BUY CALLS'
            bias = 'BULLISH'
        elif btc_data['signal'] == 'SELL' and btc_data['confidence'] >= 70:
            action = 'BUY PUTS'
            bias = 'BEARISH'
        else:
            action = 'HOLD'
            bias = 'NEUTRAL'
        
        # Get recommended strike
        iren_price = iren_data.get('price', 56.68)
        
        if action == 'BUY CALLS':
            # Slightly OTM call
            recommended_strike = round(iren_price * 1.03 / 2.5) * 2.5  # Round to 2.5
        elif action == 'BUY PUTS':
            recommended_strike = round(iren_price * 0.97 / 2.5) * 2.5
        else:
            recommended_strike = round(iren_price / 2.5) * 2.5
        
        # Check liquidity on recommended strike
        strike_liquidity = self.liquidity.check_liquidity(
            symbol='IREN',
            strike=recommended_strike,
            expiry=self._get_optimal_expiry(),
            option_type='call' if action != 'BUY PUTS' else 'put',
            contracts_needed=100
        )
        
        # Greeks warning
        greeks_analysis = self.greeks.analyze_position(
            contracts=100,
            strike=recommended_strike,
            expiry=self._get_optimal_expiry(),
            current_price=iren_price,
            option_type='call' if action != 'BUY PUTS' else 'put'
        )
        
        return {
            'type': 'daily_strategy',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            
            'signal': {
                'action': action,
                'bias': bias,
                'confidence': btc_data['confidence']
            },
            
            'market_context': {
                'btc_price': btc_price['price'],
                'btc_signal': btc_data['signal'],
                'btc_confidence': btc_data['confidence'],
                'iren_price': iren_price,
                'iren_change': iren_data.get('change_pct', 0),
                'correlation': correlation['correlation_30d'],
                'beta': correlation['beta']
            },
            
            'recommendation': {
                'strike': recommended_strike,
                'expiry': self._get_optimal_expiry(),
                'option_type': 'call' if action != 'BUY PUTS' else 'put',
                'entry_range': f"${greeks_analysis['position']['premium'] * 0.95:.2f} - ${greeks_analysis['position']['premium'] * 1.05:.2f}",
                'max_contracts': strike_liquidity['max_clean_exit']['contracts']
            },
            
            'suggested_size': {
                'conservative': 50,
                'standard': 100,
                'aggressive': 150
            },
            
            'volume_hot_spots': [
                {'strike': s['strike'], 'volume': s['volume'], 'grade': s['liquidity_grade']}
                for s in liquid_strikes[:5]
            ],
            
            'greek_warning': {
                'daily_theta': greeks_analysis['theta']['daily_decay'],
                'weekend_theta': greeks_analysis['theta']['weekend_decay'],
                'iv_rank': greeks_analysis['vega']['iv_percentile'],
                'iv_risk': greeks_analysis['vega']['risk_level']
            },
            
            'targets': {
                'tp1': {'pct': 30, 'action': 'Exit 50%'},
                'tp2': {'pct': 50, 'action': 'Exit 30%'},
                'stop': {'pct': -25, 'action': 'Exit all'}
            },
            
            'btc_reasoning': btc_data['reasoning'],
            
            'warnings': greeks_analysis['recommendations'][:3],
            
            'formatted': self._format_daily_strategy(
                action=action,
                btc_price=btc_price['price'],
                btc_signal=btc_data['signal'],
                btc_confidence=btc_data['confidence'],
                iren_price=iren_price,
                correlation=correlation['correlation_30d'],
                strike=recommended_strike,
                expiry=self._get_optimal_expiry(),
                max_contracts=strike_liquidity['max_clean_exit']['contracts'],
                liquid_strikes=liquid_strikes[:3],
                theta=greeks_analysis['theta']['daily_decay'],
                iv_rank=greeks_analysis['vega']['iv_percentile']
            )
        }
    
    def get_weekly_strategy(self) -> Dict:
        """Weekly outlook with position building strategy"""
        btc_signal = self.btc.get_btc_signal()
        correlation = self.btc.get_btc_iren_correlation()
        iren_data = get_iren_analysis()
        
        # Weekly bias
        if btc_signal['signal'] == 'BUY' and btc_signal['confidence'] >= 60:
            weekly_bias = 'BULLISH'
            strategy = 'Accumulate calls on dips'
        elif btc_signal['signal'] == 'SELL' and btc_signal['confidence'] >= 60:
            weekly_bias = 'BEARISH'
            strategy = 'Accumulate puts on rallies'
        else:
            weekly_bias = 'NEUTRAL'
            strategy = 'Wait for clearer signal or trade both sides'
        
        return {
            'type': 'weekly_strategy',
            'week_of': datetime.now().strftime('%Y-%m-%d'),
            
            'weekly_bias': weekly_bias,
            'strategy': strategy,
            
            'btc_trend': {
                'signal': btc_signal['signal'],
                'confidence': btc_signal['confidence'],
                'key_levels': {
                    'support': btc_signal.get('stop_loss', 85000),
                    'resistance': btc_signal.get('take_profit', [95000])[0]
                }
            },
            
            'iren_correlation': {
                'correlation_30d': correlation['correlation_30d'],
                'beta': correlation['beta'],
                'implied_moves': correlation['implied_iren_move']
            },
            
            'position_building': {
                'total_target': 200,
                'tranches': [
                    {'day': 'Monday', 'size': 50, 'condition': 'On any pullback'},
                    {'day': 'Tuesday', 'size': 50, 'condition': 'If still bullish'},
                    {'day': 'Wednesday', 'size': 50, 'condition': 'Add on strength'},
                    {'day': 'Thursday', 'size': 50, 'condition': 'Final tranche if trend intact'}
                ]
            },
            
            'key_dates': {
                'earnings': 'Check calendar',
                'btc_events': 'Monitor',
                'fed_meeting': 'Check calendar'
            },
            
            'expiry_recommendation': {
                'preferred': self._get_optimal_expiry(),
                'reasoning': '14-21 DTE optimal for theta/gamma balance'
            }
        }
    
    def check_volume(self, strike: float, contracts: int = 100) -> Dict:
        """Check volume/liquidity for a specific strike"""
        expiry = self._get_optimal_expiry()
        
        result = self.liquidity.check_liquidity(
            symbol='IREN',
            strike=strike,
            expiry=expiry,
            option_type='call',
            contracts_needed=contracts
        )
        
        # Add formatted output
        result['formatted'] = self._format_volume_check(result, contracts)
        
        return result
    
    def analyze_greeks(self, contracts: int, strike: float) -> Dict:
        """Full Greeks analysis for a position"""
        iren_data = get_iren_analysis()
        current_price = iren_data.get('price', 56.68)
        
        result = self.greeks.analyze_position(
            contracts=contracts,
            strike=strike,
            expiry=self._get_optimal_expiry(),
            current_price=current_price,
            option_type='call'
        )
        
        # Add formatted output
        result['formatted'] = self._format_greeks(result)
        
        return result
    
    def get_exit_strategy(self, contracts: int, strike: float) -> Dict:
        """How to exit a position"""
        expiry = self._get_optimal_expiry()
        
        result = self.liquidity.optimal_exit_strategy(
            contracts=contracts,
            symbol='IREN',
            strike=strike,
            expiry=expiry,
            option_type='call',
            urgency='normal'
        )
        
        # Add formatted output
        result['formatted'] = self._format_exit_strategy(result, contracts, strike)
        
        return result
    
    def recommend_size(self, strike: float) -> Dict:
        """Recommend position size based on liquidity"""
        liquidity = self.liquidity.check_liquidity(
            symbol='IREN',
            strike=strike,
            expiry=self._get_optimal_expiry(),
            option_type='call',
            contracts_needed=200
        )
        
        max_clean = liquidity['max_clean_exit']['contracts']
        
        return {
            'strike': strike,
            'recommended_sizes': {
                'conservative': min(50, max_clean),
                'standard': min(100, max_clean),
                'aggressive': min(150, max_clean),
                'max_clean': max_clean
            },
            'liquidity_grade': liquidity['liquidity_grade'],
            'reasoning': f"Based on {liquidity['market_data']['volume']:,} volume and {liquidity['market_data']['open_interest']:,} OI",
            'warnings': liquidity['recommendations']
        }
    
    def get_btc_outlook(self) -> Dict:
        """BTC-specific outlook"""
        price = self.btc.get_btc_price()
        signal = self.btc.get_btc_signal()
        technicals = self.btc.get_btc_technicals()
        correlation = self.btc.get_btc_iren_correlation()
        
        return {
            'btc_price': price['price'],
            'btc_change_24h': price['change_24h'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'technicals': {
                'rsi': technicals['rsi'],
                'macd': technicals['macd']['trend'],
                'trend': technicals['trend']
            },
            'iren_correlation': correlation['correlation_30d'],
            'iren_beta': correlation['beta'],
            'implied_iren_move': correlation['implied_iren_move'],
            'reasoning': signal['reasoning']
        }
    
    def _get_optimal_expiry(self) -> str:
        """Get optimal expiry date (14-21 DTE)"""
        today = datetime.now()
        target_dte = 14  # 2 weeks out
        
        expiry = today + timedelta(days=target_dte)
        
        # Move to Friday
        while expiry.weekday() != 4:  # Friday
            expiry += timedelta(days=1)
        
        return expiry.strftime('%Y-%m-%d')
    
    def _format_daily_strategy(self, action: str, btc_price: float, btc_signal: str,
                                btc_confidence: int, iren_price: float, correlation: float,
                                strike: float, expiry: str, max_contracts: int,
                                liquid_strikes: List, theta: float, iv_rank: float) -> str:
        """Format daily strategy as text"""
        lines = [
            "━" * 40,
            f"🎯 IREN SIGNAL - {datetime.now().strftime('%b %d, %Y %H:%M')}",
            "━" * 40,
            "",
            f"ACTION: {action}",
            "",
            "📈 RECOMMENDATION:",
            f"   Strike: ${strike}",
            f"   Expiry: {expiry}",
            f"   Size: Up to {max_contracts} contracts (good liquidity)",
            "",
            f"📡 BTC STATUS: ${btc_price:,.0f} ({btc_signal} {btc_confidence}%)",
            f"   Correlation: {correlation:.2f}",
            "",
            f"💰 IREN: ${iren_price:.2f}",
            "",
            "📊 VOLUME HOT SPOTS:"
        ]
        
        for s in liquid_strikes[:3]:
            lines.append(f"   ${s['strike']}: {s['volume']:,} vol ({s['liquidity_grade']})")
        
        lines.extend([
            "",
            "⚠️ GREEK WARNING:",
            f"   Theta: ${abs(theta):,.0f}/day per 100 contracts",
            f"   IV Rank: {iv_rank}% (elevated - crush risk)",
            "",
            "🎯 TARGETS:",
            "   TP1: +30% → Exit 50%",
            "   TP2: +50% → Exit 30%",
            "   Stop: -25% → Exit all",
            "━" * 40
        ])
        
        return "\n".join(lines)
    
    def _format_volume_check(self, result: Dict, contracts: int) -> str:
        """Format volume check as text"""
        can_trade = "✅ YES" if result['can_trade_size'] else "❌ NO"
        
        lines = [
            "━" * 40,
            f"📊 LIQUIDITY CHECK: IREN ${result['strike']} Call",
            "━" * 40,
            "",
            f"{can_trade} - {'GOOD' if result['can_trade_size'] else 'RISKY'} LIQUIDITY FOR {contracts} CONTRACTS",
            "",
            "📈 Current Stats:",
            f"   Volume: {result['market_data']['volume']:,}",
            f"   Open Interest: {result['market_data']['open_interest']:,}",
            f"   Bid: ${result['market_data']['bid']} | Ask: ${result['market_data']['ask']}",
            f"   Spread: ${result['market_data']['spread']} ({result['market_data']['spread_percent']:.1f}%)",
            "",
            f"💰 {contracts} Contract Analysis:",
            f"   Slippage Est: ${result['slippage_estimate']['total_cost']:,.0f}",
            f"   Market Impact: {result['slippage_estimate']['market_impact']}",
            "",
            f"🚪 Max Clean Exit: {result['max_clean_exit']['contracts']} contracts",
            "",
            "📋 RECOMMENDATIONS:"
        ]
        
        for rec in result['recommendations'][:3]:
            lines.append(f"   {rec}")
        
        lines.append("━" * 40)
        return "\n".join(lines)
    
    def _format_greeks(self, result: Dict) -> str:
        """Format Greeks as text"""
        lines = [
            "━" * 40,
            f"📊 GREEKS: {result['position']['contracts']}x IREN ${result['position']['strike']} Call",
            "━" * 40,
            "",
            "📈 POSITION GREEKS:",
            "",
            f"   DELTA: +{result['delta']['total']:.1f} ({result['delta']['equivalent_shares']:,} share equiv)",
            f"   └── {result['delta']['interpretation']}",
            "",
            f"   THETA: ${result['theta']['daily_decay']:,.0f}/day 😰",
            f"   └── Losing ${abs(result['theta']['daily_decay']):,.0f} daily to time decay",
            f"   └── Weekend = ${abs(result['theta']['weekend_decay']):,.0f}",
            "",
            f"   GAMMA: +{result['gamma']['total']:.2f}",
            f"   └── {result['gamma']['risk_level']} gamma risk",
            "",
            f"   VEGA: ${result['vega']['total']:,.0f}",
            f"   └── IV at {result['vega']['iv_current']}% ({result['vega']['risk_level']} crush risk)",
            "",
            "⚠️ WARNINGS:"
        ]
        
        for rec in result['recommendations'][:4]:
            lines.append(f"   • {rec}")
        
        lines.extend([
            "",
            f"🎯 BREAKEVEN: ${result['breakeven']['price']} ({result['breakeven']['percent_move_needed']:+.1f}%)",
            "━" * 40
        ])
        
        return "\n".join(lines)
    
    def _format_exit_strategy(self, result: Dict, contracts: int, strike: float) -> str:
        """Format exit strategy as text"""
        lines = [
            "━" * 40,
            f"🚪 EXIT STRATEGY: {contracts}x IREN ${strike} Calls",
            "━" * 40,
            "",
            f"STRATEGY: {result['strategy']}",
            "",
            "📊 TRANCHES:"
        ]
        
        for t in result['tranches']:
            lines.append(f"   {t['size']} contracts @ {t['timing']} → ~${t['expected_fill']}")
        
        lines.extend([
            "",
            "💰 COST COMPARISON:",
            f"   Market Order Cost: ${result['vs_market_order']['market_order_cost']:,.0f}",
            f"   Tranche Cost: ${result['vs_market_order']['tranche_cost']:,.0f}",
            f"   Savings: ${result['vs_market_order']['savings']:,.0f}",
            "",
            "📋 RECOMMENDATIONS:"
        ])
        
        for rec in result['recommendations']:
            lines.append(f"   • {rec}")
        
        lines.append("━" * 40)
        return "\n".join(lines)


# Singleton
_paul_bot = None

def get_paul_bot() -> PaulBot:
    global _paul_bot
    if _paul_bot is None:
        _paul_bot = PaulBot()
    return _paul_bot


def paul_chat(query: str) -> Dict:
    """Quick function for Paul's chat queries"""
    return get_paul_bot().chat(query)


def paul_daily() -> Dict:
    """Get daily strategy"""
    return get_paul_bot().get_daily_strategy()


def paul_weekly() -> Dict:
    """Get weekly strategy"""
    return get_paul_bot().get_weekly_strategy()


# CLI Testing
if __name__ == '__main__':
    print("=" * 60)
    print("🎯 PAUL BOT TEST")
    print("=" * 60)
    
    bot = PaulBot()
    
    print("\n📊 Testing: 'What's the play today?'")
    result = bot.chat("What's the play today?")
    print(result['formatted'])
    
    print("\n" + "=" * 60)
    print("📊 Testing: 'Volume check on $60 calls 200 contracts'")
    result = bot.chat("Volume check on $60 calls 200 contracts")
    print(result['formatted'])
    
    print("\n" + "=" * 60)
    print("📊 Testing: 'Greeks on 150 contracts $58 calls'")
    result = bot.chat("Greeks on 150 contracts $58 calls")
    print(result['formatted'])
    
    print("\n" + "=" * 60)
    print("✅ Paul Bot Ready!")
