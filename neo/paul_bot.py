"""
PaulBot - Real-Time Trading Intelligence for Paul's IREN Options

IN-HOUSE USE ONLY - Not for outside distribution

Paul's Requirements (Jan 24, 2026):
1. LONG ONLY - No shorts, no puts. Paul owns 100K shares, wants to ADD on dips
2. BUY CALLS ONLY - He's bullish IREN for AI datacenter thesis
3. AVOID NEAR-TERM EXPIRIES - Especially earnings (Feb 5, 2026)
4. TRACK BTC DECOUPLING - IREN transitioning from BTC miner to AI datacenter

Paul's Scale: 50-200+ contracts, making $50K-130K/week
At this level, we need INSTITUTIONAL-GRADE tools.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

logger = logging.getLogger(__name__)

# Import our analysis modules
try:
    from btc_analyzer import BTCAnalyzer, get_btc_analyzer
    from iren_analyzer import IRENAnalyzer, get_iren_analysis
    from greek_analyzer import GreekAnalyzer, get_greek_analyzer
    from liquidity_analyzer import LiquidityAnalyzer, get_liquidity_analyzer
    from btc_coupling_analyzer import BTCCouplingAnalyzer, get_coupling_analyzer
    from core_position_bot import IRENCorePosisionBot, get_core_position_bot
except ImportError:
    # Relative imports for package usage
    from .btc_analyzer import BTCAnalyzer, get_btc_analyzer
    from .iren_analyzer import IRENAnalyzer, get_iren_analysis
    from .greek_analyzer import GreekAnalyzer, get_greek_analyzer
    from .liquidity_analyzer import LiquidityAnalyzer, get_liquidity_analyzer
    from .btc_coupling_analyzer import BTCCouplingAnalyzer, get_coupling_analyzer
    from .core_position_bot import IRENCorePosisionBot, get_core_position_bot


# IREN Earnings Date - CRITICAL
IREN_EARNINGS_DATE = datetime(2026, 2, 5)


class PaulBot:
    """
    Real-time trading intelligence for Paul's IREN options.
    
    PAUL'S RULES:
    1. LONG ONLY - He owns shares, wants to add via calls
    2. NO SHORTS, NO PUTS - He's bullish IREN
    3. Prefer 14-35 DTE expirations
    4. AVOID earnings window (Feb 5, 2026)
    5. Watch BTC but note decoupling thesis
    """
    
    def __init__(self):
        self.btc = get_btc_analyzer()
        self.greeks = get_greek_analyzer()
        self.liquidity = get_liquidity_analyzer()
        self.coupling = get_coupling_analyzer()
        self.core_position = get_core_position_bot()
        
        # Paul's preferences
        self.default_contracts = 100
        self.preferred_dte_range = (14, 35)  # 2-5 weeks, NOT close to earnings
        self.min_dte = 14  # Minimum 14 days
        self.max_spread_pct = 5.0
        self.min_volume = 2000
        
        # Paul's thesis
        self.thesis = {
            'target_price': 150.00,
            'entry_price': 56.68,
            'core_shares': 100_000,
            'thesis_summary': "AI datacenter demand + legacy BTC infrastructure = $150 target"
        }
    
    def chat(self, query: str) -> Dict:
        """
        Natural language interface for Paul.
        
        LONG-ONLY responses - no shorts or puts suggested.
        """
        query_lower = query.lower()
        
        # Route to appropriate handler
        if any(word in query_lower for word in ['play', 'strategy', 'signal', 'recommendation', 'today']):
            return self.get_daily_strategy()
        
        elif any(word in query_lower for word in ['weekly', 'outlook', 'week']):
            return self.get_weekly_strategy()
        
        elif 'volume' in query_lower or 'liquidity' in query_lower:
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
        
        elif any(word in query_lower for word in ['coupling', 'decouple', 'correlation']):
            return self.get_coupling_analysis()
        
        elif any(word in query_lower for word in ['core', 'shares', '100k', 'position']):
            return self.get_core_position_status()
        
        elif any(word in query_lower for word in ['covered', 'call income', 'cc']):
            return self.get_covered_call_opportunity()
        
        else:
            # Default to daily strategy
            return self.get_daily_strategy()
    
    def _extract_strike(self, query: str) -> Optional[float]:
        """Extract strike price from query"""
        patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*call',
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
    
    def get_available_expiries(self) -> List[Dict]:
        """
        Get available option expiries with filtering.
        
        RULES:
        1. At least 14 DTE
        2. Avoid 7 days before/after earnings (Feb 5)
        3. Prefer 21-35 DTE (Paul's sweet spot)
        """
        try:
            iren = yf.Ticker("IREN")
            expiries = list(iren.options)
        except:
            expiries = []
        
        today = datetime.now().date()
        earnings_date = IREN_EARNINGS_DATE.date()
        
        filtered_expiries = []
        for exp in expiries:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            days_out = (exp_date - today).days
            days_from_earnings = abs((exp_date - earnings_date).days)
            
            # Filter rules
            if days_out < self.min_dte:
                continue  # Too close
            
            # Determine quality
            if days_from_earnings <= 7:
                quality = "EARNINGS_RISK"
                paul_pick = False
            elif self.preferred_dte_range[0] <= days_out <= self.preferred_dte_range[1]:
                quality = "PAUL_PREFERRED"
                paul_pick = True
            elif days_out > 45:
                quality = "LONG_TERM"
                paul_pick = False
            else:
                quality = "GOOD"
                paul_pick = False
            
            filtered_expiries.append({
                'date': exp,
                'days_out': days_out,
                'days_from_earnings': days_from_earnings,
                'quality': quality,
                'paul_pick': paul_pick
            })
        
        return filtered_expiries
    
    def get_daily_strategy(self) -> Dict:
        """
        Morning brief with LONG-ONLY daily strategy.
        
        NO PUTS, NO SHORTS - Paul is bullish IREN.
        """
        # Get all market intelligence
        btc_data = self.btc.get_btc_signal()
        btc_price = self.btc.get_btc_price()
        iren_data = get_iren_analysis()
        iren_price = iren_data.get('price', 56.68)
        
        # Get coupling/decoupling status
        coupling_status = self.coupling.get_coupling_status()
        
        # Get filtered expiries
        expiries = self.get_available_expiries()
        paul_picks = [e for e in expiries if e['paul_pick']]
        
        # ALWAYS BULLISH for Paul - he's LONG the stock
        # Adjust confidence based on BTC signal
        if btc_data['signal'] == 'BUY' and btc_data['confidence'] >= 60:
            action = 'BUY CALLS'
            confidence = min(btc_data['confidence'] + 10, 95)  # Boost for alignment
            bias = 'STRONGLY BULLISH'
        elif btc_data['signal'] == 'BUY':
            action = 'BUY CALLS'
            confidence = btc_data['confidence']
            bias = 'BULLISH'
        elif coupling_status['status'] == 'DECOUPLED':
            # If decoupled, ignore BTC signal
            action = 'BUY CALLS'
            confidence = 70
            bias = 'BULLISH (AI THESIS)'
        else:
            action = 'WAIT FOR DIP'
            confidence = 50
            bias = 'NEUTRAL - WAIT'
        
        # Get recommended strike (slightly OTM call)
        recommended_strike = round(iren_price * 1.03 / 2.5) * 2.5
        
        # Find best expiry for Paul
        best_expiry = paul_picks[0]['date'] if paul_picks else (expiries[0]['date'] if expiries else '2026-02-20')
        
        # Find liquid strikes
        liquid_strikes = self.liquidity.find_liquid_strikes(
            symbol='IREN',
            expiry=best_expiry,
            option_type='call',
            min_volume=self.min_volume
        )
        
        # Check liquidity
        strike_liquidity = self.liquidity.check_liquidity(
            symbol='IREN',
            strike=recommended_strike,
            expiry=best_expiry,
            option_type='call',
            contracts_needed=100
        )
        
        # Greeks analysis
        greeks_analysis = self.greeks.analyze_position(
            contracts=100,
            strike=recommended_strike,
            expiry=best_expiry,
            current_price=iren_price,
            option_type='call'
        )
        
        # Get core position status
        core_status = self.core_position.get_core_position_status()
        
        return {
            'type': 'daily_strategy',
            'mode': 'LONG_ONLY',  # PAUL'S RULE
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            
            'signal': {
                'action': action,
                'bias': bias,
                'confidence': confidence,
                'note': 'LONG ONLY - No shorts for Paul'
            },
            
            'market_context': {
                'btc_price': btc_price['price'],
                'btc_signal': btc_data['signal'],
                'btc_confidence': btc_data['confidence'],
                'iren_price': iren_price,
                'iren_change': iren_data.get('change_pct', 0)
            },
            
            'coupling_analysis': {
                'status': coupling_status['status'],
                'correlation_7d': coupling_status['correlation_7d'],
                'correlation_30d': coupling_status['correlation_30d'],
                'trend': coupling_status['trend'],
                'beta': coupling_status['beta'],
                'driver': coupling_status['driver'],
                'recommendation': coupling_status['recommendation']
            },
            
            'recommendation': {
                'strike': recommended_strike,
                'expiry': best_expiry,
                'option_type': 'CALL',  # ALWAYS CALL for Paul
                'entry_range': f"${greeks_analysis['position']['premium'] * 0.95:.2f} - ${greeks_analysis['position']['premium'] * 1.05:.2f}",
                'max_contracts': strike_liquidity['max_clean_exit']['contracts']
            },
            
            'paul_preferred_expiries': paul_picks[:4],  # Top 4 preferred expiries
            
            'earnings_warning': {
                'date': IREN_EARNINGS_DATE.strftime('%Y-%m-%d'),
                'days_away': (IREN_EARNINGS_DATE.date() - datetime.now().date()).days,
                'warning': 'AVOID expiries within 7 days of earnings'
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
            
            'core_position': {
                'shares': core_status['core_position']['shares'],
                'value': core_status['core_position']['position_value'],
                'unrealized_pnl': core_status['core_position']['unrealized_pnl'],
                'target_price': core_status['target_analysis']['target_price'],
                'upside_remaining': core_status['target_analysis']['upside_remaining_pct']
            },
            
            'pauls_thesis': coupling_status['pauls_thesis'],
            
            'warnings': greeks_analysis['recommendations'][:3],
            
            'formatted': self._format_daily_strategy(
                action=action,
                bias=bias,
                confidence=confidence,
                btc_price=btc_price['price'],
                btc_signal=btc_data['signal'],
                iren_price=iren_price,
                coupling_status=coupling_status,
                strike=recommended_strike,
                expiry=best_expiry,
                max_contracts=strike_liquidity['max_clean_exit']['contracts'],
                liquid_strikes=liquid_strikes[:3],
                theta=greeks_analysis['theta']['daily_decay'],
                iv_rank=greeks_analysis['vega']['iv_percentile'],
                earnings_days=(IREN_EARNINGS_DATE.date() - datetime.now().date()).days
            )
        }
    
    def get_weekly_strategy(self) -> Dict:
        """Weekly outlook - LONG ONLY"""
        btc_signal = self.btc.get_btc_signal()
        coupling_status = self.coupling.get_coupling_status()
        iren_data = get_iren_analysis()
        
        # Always bullish for Paul
        weekly_bias = 'BULLISH'
        strategy = 'Accumulate calls on dips - Paul is LONG the stock'
        
        # Get preferred expiries
        expiries = self.get_available_expiries()
        paul_picks = [e for e in expiries if e['paul_pick']]
        
        return {
            'type': 'weekly_strategy',
            'mode': 'LONG_ONLY',
            'week_of': datetime.now().strftime('%Y-%m-%d'),
            
            'weekly_bias': weekly_bias,
            'strategy': strategy,
            
            'btc_analysis': {
                'signal': btc_signal['signal'],
                'confidence': btc_signal['confidence'],
                'use_for_timing': coupling_status['status'] == 'COUPLED'
            },
            
            'coupling_status': {
                'status': coupling_status['status'],
                'trend': coupling_status['trend'],
                'recommendation': coupling_status['recommendation']
            },
            
            'position_building': {
                'approach': 'Accumulate CALLS on any dips',
                'total_target': 200,
                'tranches': [
                    {'day': 'Monday', 'size': 50, 'condition': 'On any red day'},
                    {'day': 'Wed/Thurs', 'size': 50, 'condition': 'If still below $60'},
                    {'day': 'Friday', 'size': 50, 'condition': 'If weekend theta acceptable'},
                ]
            },
            
            'preferred_expiries': paul_picks[:4],
            
            'earnings_calendar': {
                'iren_earnings': IREN_EARNINGS_DATE.strftime('%Y-%m-%d'),
                'days_away': (IREN_EARNINGS_DATE.date() - datetime.now().date()).days,
                'strategy': 'Avoid expiries within 7 days of Feb 5 earnings'
            },
            
            'pauls_thesis': coupling_status['pauls_thesis']
        }
    
    def get_coupling_analysis(self) -> Dict:
        """
        Get BTC-IREN coupling/decoupling analysis.
        
        Paul's thesis: IREN is transitioning from BTC miner to AI datacenter.
        """
        status = self.coupling.get_coupling_status()
        should_use_btc = self.coupling.should_use_btc_signal()
        trading_mode = self.coupling.get_trading_mode()
        
        return {
            'type': 'coupling_analysis',
            'status': status['status'],
            'trend': status['trend'],
            
            'correlations': {
                '7d': status['correlation_7d'],
                '14d': status['correlation_14d'],
                '30d': status['correlation_30d'],
                '60d': status['correlation_60d']
            },
            
            'beta': status['beta'],
            
            'today_analysis': status['analysis'],
            
            'trading_mode': trading_mode,
            'use_btc_signal': should_use_btc['use_btc_signal'],
            
            'driver': status['driver'],
            'recommendation': status['recommendation'],
            
            'pauls_thesis': status['pauls_thesis'],
            
            'formatted': self._format_coupling_analysis(status, trading_mode)
        }
    
    def get_core_position_status(self) -> Dict:
        """Get Paul's 100K share core position status"""
        status = self.core_position.get_core_position_status()
        cc_opportunity = self.core_position.should_sell_covered_calls()
        protection = self.core_position.should_buy_protection()
        
        return {
            'type': 'core_position',
            'position': status['core_position'],
            'target': status['target_analysis'],
            'covered_call_capacity': status['covered_call_capacity'],
            'income_tracking': status['income_tracking'],
            'active_trades': status['active_trades'],
            'next_cc_opportunity': cc_opportunity,
            'protection_status': protection,
            'rules': [
                '🚫 NEVER SELL THE CORE SHARES',
                '📈 Target: $150/share',
                '💰 Generate income via covered calls',
                '🛡️ Buy protection only when VIX > 25'
            ]
        }
    
    def get_covered_call_opportunity(self) -> Dict:
        """Get covered call income opportunity"""
        return self.core_position.generate_income_opportunity()
    
    def check_volume(self, strike: float, contracts: int = 100) -> Dict:
        """Check volume/liquidity for a specific strike (CALLS ONLY for Paul)"""
        expiries = self.get_available_expiries()
        best_expiry = next((e['date'] for e in expiries if e['paul_pick']), expiries[0]['date'] if expiries else None)
        
        if not best_expiry:
            return {'error': 'No expiries available'}
        
        result = self.liquidity.check_liquidity(
            symbol='IREN',
            strike=strike,
            expiry=best_expiry,
            option_type='call',  # ALWAYS CALLS for Paul
            contracts_needed=contracts
        )
        
        result['formatted'] = self._format_volume_check(result, contracts)
        return result
    
    def analyze_greeks(self, contracts: int, strike: float) -> Dict:
        """Full Greeks analysis for a CALL position"""
        iren_data = get_iren_analysis()
        current_price = iren_data.get('price', 56.68)
        
        expiries = self.get_available_expiries()
        best_expiry = next((e['date'] for e in expiries if e['paul_pick']), expiries[0]['date'] if expiries else '2026-02-20')
        
        result = self.greeks.analyze_position(
            contracts=contracts,
            strike=strike,
            expiry=best_expiry,
            current_price=current_price,
            option_type='call'  # ALWAYS CALLS for Paul
        )
        
        result['formatted'] = self._format_greeks(result)
        return result
    
    def get_exit_strategy(self, contracts: int, strike: float) -> Dict:
        """How to exit a CALL position"""
        expiries = self.get_available_expiries()
        best_expiry = next((e['date'] for e in expiries if e['paul_pick']), expiries[0]['date'] if expiries else '2026-02-20')
        
        result = self.liquidity.optimal_exit_strategy(
            contracts=contracts,
            symbol='IREN',
            strike=strike,
            expiry=best_expiry,
            option_type='call',
            urgency='normal'
        )
        
        result['formatted'] = self._format_exit_strategy(result, contracts, strike)
        return result
    
    def recommend_size(self, strike: float) -> Dict:
        """Recommend position size based on liquidity (CALLS ONLY)"""
        expiries = self.get_available_expiries()
        best_expiry = next((e['date'] for e in expiries if e['paul_pick']), expiries[0]['date'] if expiries else '2026-02-20')
        
        liquidity = self.liquidity.check_liquidity(
            symbol='IREN',
            strike=strike,
            expiry=best_expiry,
            option_type='call',
            contracts_needed=200
        )
        
        max_clean = liquidity['max_clean_exit']['contracts']
        
        return {
            'strike': strike,
            'option_type': 'CALL',  # ALWAYS CALLS for Paul
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
        """BTC-specific outlook with coupling context"""
        price = self.btc.get_btc_price()
        signal = self.btc.get_btc_signal()
        technicals = self.btc.get_btc_technicals()
        coupling = self.coupling.get_coupling_status()
        
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
            'iren_coupling': {
                'status': coupling['status'],
                'correlation': coupling['correlation_7d'],
                'use_for_trading': coupling['status'] == 'COUPLED'
            },
            'reasoning': signal['reasoning'],
            'pauls_note': "BTC is secondary - IREN's AI datacenter thesis is primary driver"
        }
    
    def _format_daily_strategy(self, action: str, bias: str, confidence: int,
                                btc_price: float, btc_signal: str, iren_price: float,
                                coupling_status: Dict, strike: float, expiry: str,
                                max_contracts: int, liquid_strikes: List, theta: float,
                                iv_rank: float, earnings_days: int) -> str:
        """Format daily strategy as text"""
        lines = [
            "━" * 50,
            f"🎯 IREN DAILY SIGNAL - {datetime.now().strftime('%b %d, %Y')}",
            f"📌 MODE: LONG ONLY (Paul owns 100K shares)",
            "━" * 50,
            "",
            f"🎯 ACTION: {action}",
            f"   Bias: {bias} ({confidence}% confidence)",
            "",
            f"🔗 BTC-IREN COUPLING: {coupling_status['status']}",
            f"   Correlation (7d): {coupling_status['correlation_7d']:.2f}",
            f"   Trend: {coupling_status['trend']}",
            f"   → {coupling_status['recommendation']}",
            "",
            f"📈 RECOMMENDED TRADE (CALLS ONLY):",
            f"   Strike: ${strike}",
            f"   Expiry: {expiry}",
            f"   Size: Up to {max_contracts} contracts",
            "",
            f"📡 MARKET STATUS:",
            f"   BTC: ${btc_price:,.0f} ({btc_signal})",
            f"   IREN: ${iren_price:.2f}",
            "",
            f"⚠️ EARNINGS: {earnings_days} days away (Feb 5)",
            f"   → Avoid expiries within 7 days of earnings!",
            "",
            "💧 VOLUME HOT SPOTS:"
        ]
        
        for s in liquid_strikes[:3]:
            lines.append(f"   ${s['strike']}: {s['volume']:,} vol ({s['liquidity_grade']})")
        
        lines.extend([
            "",
            "⚠️ GREEK WARNING (per 100 contracts):",
            f"   Theta: ${abs(theta):,.0f}/day",
            f"   IV Rank: {iv_rank}%",
            "",
            "🎯 TARGETS:",
            "   TP1: +30% → Exit 50%",
            "   TP2: +50% → Exit 30%",
            "   Stop: -25% → Exit all",
            "",
            "💡 PAUL'S THESIS: AI datacenter demand > BTC mining",
            "   Power infrastructure = 3-5 year competitive moat",
            "━" * 50
        ])
        
        return "\n".join(lines)
    
    def _format_coupling_analysis(self, status: Dict, trading_mode: str) -> str:
        """Format coupling analysis as text"""
        lines = [
            "━" * 50,
            "🔗 BTC-IREN COUPLING ANALYSIS",
            "━" * 50,
            "",
            f"STATUS: {status['status']}",
            f"TREND: {status['trend']}",
            "",
            "📊 CORRELATIONS:",
            f"   7-day:  {status['correlation_7d']:.3f}",
            f"   14-day: {status['correlation_14d']:.3f}",
            f"   30-day: {status['correlation_30d']:.3f}",
            f"   60-day: {status['correlation_60d']:.3f}",
            "",
            f"📈 BETA: {status['beta']:.2f}x",
            "",
            f"🎯 TRADING MODE: {trading_mode}",
            "",
            f"💡 DRIVER: {status['driver']}",
            "",
            f"📌 RECOMMENDATION: {status['recommendation']}",
            "",
            "━" * 30,
            "PAUL'S THESIS:",
            "━" * 30,
        ]
        
        for key, value in status['pauls_thesis'].items():
            lines.append(f"  • {key}: {value}")
        
        lines.append("━" * 50)
        return "\n".join(lines)
    
    def _format_volume_check(self, result: Dict, contracts: int) -> str:
        """Format volume check as text"""
        can_trade = "✅ YES" if result['can_trade_size'] else "❌ NO"
        
        lines = [
            "━" * 50,
            f"📊 LIQUIDITY CHECK: IREN ${result['strike']} CALL",
            f"📌 FOR PAUL (LONG ONLY)",
            "━" * 50,
            "",
            f"{can_trade} - GOOD LIQUIDITY FOR {contracts} CONTRACTS",
            "",
            "📈 Current Stats:",
            f"   Volume: {result['market_data']['volume']:,}",
            f"   Open Interest: {result['market_data']['open_interest']:,}",
            f"   Bid: ${result['market_data']['bid']} | Ask: ${result['market_data']['ask']}",
            f"   Spread: ${result['market_data']['spread']} ({result['market_data']['spread_percent']:.1f}%)",
            "",
            f"🚪 Max Clean Exit: {result['max_clean_exit']['contracts']} contracts",
            "",
            "📋 RECOMMENDATIONS:"
        ]
        
        for rec in result['recommendations'][:3]:
            lines.append(f"   • {rec}")
        
        lines.append("━" * 50)
        return "\n".join(lines)
    
    def _format_greeks(self, result: Dict) -> str:
        """Format Greeks as text"""
        lines = [
            "━" * 50,
            f"📊 GREEKS: {result['position']['contracts']}x IREN ${result['position']['strike']} CALL",
            f"📌 FOR PAUL (LONG ONLY)",
            "━" * 50,
            "",
            f"   DELTA: +{result['delta']['total']:.1f} ({result['delta']['equivalent_shares']:,} shares)",
            f"   THETA: ${result['theta']['daily_decay']:,.0f}/day",
            f"   GAMMA: +{result['gamma']['total']:.2f}",
            f"   VEGA: ${result['vega']['total']:,.0f}",
            "",
            f"⚠️ Weekend Theta: ${abs(result['theta']['weekend_decay']):,.0f}",
            f"📊 IV Rank: {result['vega']['iv_current']}%",
            "",
            f"🎯 BREAKEVEN: ${result['breakeven']['price']}",
            "",
            "📋 WARNINGS:"
        ]
        
        for rec in result['recommendations'][:3]:
            lines.append(f"   • {rec}")
        
        lines.append("━" * 50)
        return "\n".join(lines)
    
    def _format_exit_strategy(self, result: Dict, contracts: int, strike: float) -> str:
        """Format exit strategy as text"""
        lines = [
            "━" * 50,
            f"🚪 EXIT STRATEGY: {contracts}x IREN ${strike} CALLS",
            "━" * 50,
            "",
            f"STRATEGY: {result['strategy']}",
            "",
            "📊 TRANCHES:"
        ]
        
        for t in result['tranches']:
            lines.append(f"   {t['size']} contracts @ {t['timing']}")
        
        lines.extend([
            "",
            "💰 COST SAVINGS:",
            f"   Market Order: ${result['vs_market_order']['market_order_cost']:,.0f}",
            f"   Tranche Exit: ${result['vs_market_order']['tranche_cost']:,.0f}",
            f"   Savings: ${result['vs_market_order']['savings']:,.0f}",
            "━" * 50
        ])
        
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
    """Get daily strategy (LONG ONLY)"""
    return get_paul_bot().get_daily_strategy()


def paul_weekly() -> Dict:
    """Get weekly strategy (LONG ONLY)"""
    return get_paul_bot().get_weekly_strategy()


def paul_coupling() -> Dict:
    """Get BTC-IREN coupling analysis"""
    return get_paul_bot().get_coupling_analysis()


def paul_core_position() -> Dict:
    """Get core position status"""
    return get_paul_bot().get_core_position_status()


# CLI Testing
if __name__ == '__main__':
    print("=" * 60)
    print("🎯 PAUL BOT - LONG ONLY MODE")
    print("=" * 60)
    
    bot = PaulBot()
    
    print("\n📊 Testing: Daily Strategy")
    result = bot.get_daily_strategy()
    print(result['formatted'])
    
    print("\n" + "=" * 60)
    print("📊 Testing: Coupling Analysis")
    result = bot.get_coupling_analysis()
    print(result['formatted'])
