"""
IREN Core Position Bot - 100K Share Strategy

Paul's Setup:
- 100,000 shares of IREN (NEVER SELL THE SHARES)
- Entry: ~$56.68 → Position value: ~$5.67M
- Target: $150/share → Target value: $15M (+165% upside)

Strategy: Generate income via covered calls while waiting for $150 target.
RULE: NEVER SELL THE CORE SHARES!

Thesis:
- AI datacenter demand >> BTC mining
- Legacy power infrastructure = 3-5 year competitive moat
- Power access already built (no govt waiting list)
- Will rent/lease datacenter space to AI companies
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IRENCorePosisionBot:
    """
    Trading bot for 100K share core position.
    NEVER SELLS SHARES - only trades options around position.
    """
    
    # Core position (Paul's setup)
    CORE_SHARES = 100_000
    ENTRY_PRICE = 56.68
    TARGET_PRICE = 150.00
    
    # Trading parameters
    DEFAULT_CC_ALLOCATION = 0.5  # Use 50% of shares for covered calls
    MAX_CONTRACTS = 500  # 500 covered calls at a time
    MIN_CALL_STRIKE_PCT = 1.10  # 10% OTM minimum for CCs
    PREFERRED_DTE = 30  # 30 days to expiration preferred
    
    # State file
    STATE_FILE = Path("/home/jbot/trading_ai/data/core_position_state.json")
    
    def __init__(self):
        self.state_file = self.STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load position state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'core_shares': self.CORE_SHARES,
            'entry_price': self.ENTRY_PRICE,
            'target_price': self.TARGET_PRICE,
            'open_covered_calls': [],
            'open_protective_puts': [],
            'cash_secured_puts': [],
            'premium_collected_mtd': 0,
            'premium_collected_ytd': 0,
            'trades_history': []
        }
    
    def _save_state(self):
        """Save position state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def get_current_price(self) -> float:
        """Get current IREN price"""
        try:
            iren = yf.Ticker("IREN")
            hist = iren.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        return self.ENTRY_PRICE
    
    def get_options_chain(self, expiry: str) -> Dict[str, Any]:
        """Get options chain for a specific expiry"""
        try:
            iren = yf.Ticker("IREN")
            chain = iren.option_chain(expiry)
            return {
                'calls': chain.calls.to_dict('records'),
                'puts': chain.puts.to_dict('records'),
                'expiry': expiry
            }
        except Exception as e:
            logger.error(f"Failed to get options chain: {e}")
            return {'calls': [], 'puts': [], 'expiry': expiry}
    
    def get_available_expiries(self) -> List[str]:
        """Get available option expiration dates"""
        try:
            iren = yf.Ticker("IREN")
            return list(iren.options)
        except:
            return []
    
    def get_core_position_status(self) -> Dict[str, Any]:
        """
        Get current status of the core position.
        
        Returns complete position analysis.
        """
        current_price = self.get_current_price()
        
        # Core position value
        position_value = self.CORE_SHARES * current_price
        entry_value = self.CORE_SHARES * self.ENTRY_PRICE
        target_value = self.CORE_SHARES * self.TARGET_PRICE
        
        unrealized_pnl = position_value - entry_value
        unrealized_pnl_pct = (current_price / self.ENTRY_PRICE - 1) * 100
        
        # Distance to target
        upside_to_target = (self.TARGET_PRICE / current_price - 1) * 100
        
        # Covered call capacity
        active_cc_contracts = len(self.state.get('open_covered_calls', []))
        shares_tied_in_cc = active_cc_contracts * 100
        available_for_cc = self.CORE_SHARES - shares_tied_in_cc
        
        return {
            'core_position': {
                'shares': self.CORE_SHARES,
                'entry_price': self.ENTRY_PRICE,
                'current_price': round(current_price, 2),
                'position_value': round(position_value, 2),
                'entry_value': round(entry_value, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
                'status': 'HOLD - NEVER SELL'
            },
            'target_analysis': {
                'target_price': self.TARGET_PRICE,
                'target_value': round(target_value, 2),
                'upside_remaining_pct': round(upside_to_target, 2),
                'upside_remaining_usd': round(target_value - position_value, 2),
                'thesis': 'AI datacenter demand + legacy BTC infrastructure = $150 target'
            },
            'covered_call_capacity': {
                'total_contracts_possible': self.CORE_SHARES // 100,
                'active_cc_contracts': active_cc_contracts,
                'shares_tied_in_cc': shares_tied_in_cc,
                'available_for_new_cc': available_for_cc // 100,
                'cc_allocation_used_pct': round(shares_tied_in_cc / self.CORE_SHARES * 100, 1)
            },
            'income_tracking': {
                'premium_mtd': self.state.get('premium_collected_mtd', 0),
                'premium_ytd': self.state.get('premium_collected_ytd', 0),
                'annualized_yield_pct': self._calculate_annualized_yield()
            },
            'active_trades': {
                'covered_calls': self.state.get('open_covered_calls', []),
                'protective_puts': self.state.get('open_protective_puts', []),
                'cash_secured_puts': self.state.get('cash_secured_puts', [])
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def _calculate_annualized_yield(self) -> float:
        """Calculate annualized yield from covered call income"""
        ytd_premium = self.state.get('premium_collected_ytd', 0)
        if ytd_premium <= 0:
            return 0.0
        
        # Calculate days into year
        today = datetime.now()
        year_start = datetime(today.year, 1, 1)
        days_elapsed = (today - year_start).days
        
        if days_elapsed <= 0:
            return 0.0
        
        # Annualize
        position_value = self.CORE_SHARES * self.ENTRY_PRICE
        annualized = (ytd_premium / days_elapsed) * 365
        yield_pct = (annualized / position_value) * 100
        
        return round(yield_pct, 2)
    
    def should_sell_covered_calls(self) -> Dict[str, Any]:
        """
        Determine if conditions are right to sell covered calls.
        
        Rules:
        1. Don't sell if price approaching target (preserve upside)
        2. Max 50% of shares in covered calls
        3. Strike minimum 10% OTM
        """
        current_price = self.get_current_price()
        
        # Rule 1: Preserve upside near target
        if current_price > self.TARGET_PRICE * 0.80:  # Within 20% of target
            return {
                'action': 'HOLD',
                'reason': f'Price ${current_price:.2f} is within 20% of ${self.TARGET_PRICE} target. Preserve upside!',
                'recommendation': 'Do not sell covered calls - let shares run to target'
            }
        
        # Rule 2: Check CC allocation
        active_cc_contracts = len(self.state.get('open_covered_calls', []))
        shares_in_cc = active_cc_contracts * 100
        max_cc_shares = int(self.CORE_SHARES * self.DEFAULT_CC_ALLOCATION)
        available_shares = max_cc_shares - shares_in_cc
        
        if available_shares < 10000:  # Less than 100 contracts available
            return {
                'action': 'HOLD',
                'reason': f'Already have {active_cc_contracts} CC contracts. Max allocation reached.',
                'recommendation': 'Wait for existing CCs to expire or close before adding more'
            }
        
        # Find optimal strike and expiry
        expiries = self.get_available_expiries()
        if not expiries:
            return {
                'action': 'UNAVAILABLE',
                'reason': 'No options expiries available'
            }
        
        # Find expiry ~30 days out
        target_expiry = None
        for exp in expiries:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            days_out = (exp_date - datetime.now().date()).days
            if 21 <= days_out <= 45:
                target_expiry = exp
                break
        
        if not target_expiry:
            target_expiry = expiries[0] if expiries else None
        
        if not target_expiry:
            return {
                'action': 'UNAVAILABLE',
                'reason': 'No suitable expiry found'
            }
        
        # Calculate strike (10% OTM minimum)
        strike = round(current_price * self.MIN_CALL_STRIKE_PCT / 2.5) * 2.5  # Round to 2.5
        
        # Get premium estimate
        chain = self.get_options_chain(target_expiry)
        premium = self._find_call_premium(chain['calls'], strike)
        
        contracts_to_sell = min(available_shares // 100, 100)  # Max 100 at a time
        total_premium = premium * contracts_to_sell * 100
        
        # Calculate annualized yield
        exp_date = datetime.strptime(target_expiry, '%Y-%m-%d').date()
        days_out = (exp_date - datetime.now().date()).days
        annual_yield = (premium / current_price) * (365 / days_out) * 100
        
        return {
            'action': 'SELL_COVERED_CALL',
            'strike': strike,
            'expiry': target_expiry,
            'days_out': days_out,
            'contracts': contracts_to_sell,
            'premium_per_contract': round(premium, 2),
            'total_premium': round(total_premium, 2),
            'annual_yield_pct': round(annual_yield, 2),
            'max_profit_if_called': round((strike - self.ENTRY_PRICE) * contracts_to_sell * 100, 2),
            'breakeven': round(self.ENTRY_PRICE - premium, 2),
            'recommendation': f'Sell {contracts_to_sell}x ${strike} Calls @ ${premium:.2f} = ${total_premium:,.0f} premium'
        }
    
    def _find_call_premium(self, calls: List[Dict], target_strike: float) -> float:
        """Find the premium for a target strike"""
        for call in calls:
            if abs(call.get('strike', 0) - target_strike) < 1:
                ask = call.get('ask', 0)
                bid = call.get('bid', 0)
                last = call.get('lastPrice', 0)
                
                if ask > 0:
                    return ask
                elif bid > 0 and last > 0:
                    return (bid + last) / 2
                elif last > 0:
                    return last
        
        # Estimate if not found
        return 2.50  # Default estimate
    
    def should_buy_protection(self) -> Dict[str, Any]:
        """
        Determine if we need protective puts.
        
        Buy protection when:
        1. VIX > 25 (high volatility)
        2. BTC signals bearish
        3. Major earnings/events coming
        """
        current_price = self.get_current_price()
        
        # Get VIX
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period='1d')
            vix_value = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 20
        except:
            vix_value = 20
        
        # Check BTC signal (simplified)
        try:
            from neo.btc_coupling_analyzer import get_coupling_analyzer
            analyzer = get_coupling_analyzer()
            coupling = analyzer.get_coupling_status()
            btc_change = coupling['analysis'].get('btc_change_24h', 0) or 0
            btc_bearish = btc_change < -3
        except:
            btc_bearish = False
        
        # Determine if protection needed
        needs_protection = vix_value > 25 or btc_bearish
        
        if not needs_protection:
            return {
                'action': 'NONE',
                'reason': f'VIX at {vix_value:.1f} (< 25) and BTC neutral. No protection needed.',
                'vix': round(vix_value, 1)
            }
        
        # Calculate protection parameters
        put_strike = round(current_price * 0.85 / 2.5) * 2.5  # 15% below, rounded
        
        # Get expiry
        expiries = self.get_available_expiries()
        target_expiry = expiries[1] if len(expiries) > 1 else expiries[0] if expiries else None
        
        if not target_expiry:
            return {
                'action': 'UNAVAILABLE',
                'reason': 'No options expiries available'
            }
        
        return {
            'action': 'BUY_PROTECTIVE_PUT',
            'strike': put_strike,
            'expiry': target_expiry,
            'contracts': self.MAX_CONTRACTS,
            'reason': f'VIX at {vix_value:.1f} / BTC bearish' if btc_bearish else f'VIX elevated at {vix_value:.1f}',
            'protection_level': f'Protected below ${put_strike}',
            'recommendation': f'Consider buying {self.MAX_CONTRACTS}x ${put_strike} Puts for protection'
        }
    
    def generate_income_opportunity(self) -> Dict[str, Any]:
        """
        Generate next covered call income opportunity.
        
        Returns the best current opportunity to generate income
        while preserving upside to $150 target.
        """
        cc_analysis = self.should_sell_covered_calls()
        protection_analysis = self.should_buy_protection()
        position_status = self.get_core_position_status()
        
        return {
            'position_status': position_status,
            'covered_call_opportunity': cc_analysis,
            'protection_recommendation': protection_analysis,
            'summary': {
                'core_position_value': position_status['core_position']['position_value'],
                'unrealized_pnl': position_status['core_position']['unrealized_pnl'],
                'income_ytd': position_status['income_tracking']['premium_ytd'],
                'next_action': cc_analysis.get('action', 'HOLD'),
                'potential_income': cc_analysis.get('total_premium', 0)
            },
            'rules_reminder': [
                '🚫 NEVER SELL THE CORE SHARES',
                '📈 Target: $150/share',
                '💰 Generate income via covered calls while waiting',
                '🛡️ Buy protection only when VIX > 25 or BTC crashes'
            ],
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def record_trade(self, trade_type: str, details: Dict):
        """Record a trade in the history"""
        trade = {
            'type': trade_type,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if 'trades_history' not in self.state:
            self.state['trades_history'] = []
        
        self.state['trades_history'].append(trade)
        
        # Update running totals
        if trade_type == 'COVERED_CALL_SELL':
            premium = details.get('total_premium', 0)
            self.state['premium_collected_mtd'] = self.state.get('premium_collected_mtd', 0) + premium
            self.state['premium_collected_ytd'] = self.state.get('premium_collected_ytd', 0) + premium
            
            # Track open position
            if 'open_covered_calls' not in self.state:
                self.state['open_covered_calls'] = []
            self.state['open_covered_calls'].append(details)
        
        self._save_state()
    
    def get_monthly_income_report(self) -> Dict[str, Any]:
        """
        Generate monthly income report.
        """
        position = self.get_core_position_status()
        
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'core_position': position['core_position'],
            'income': {
                'premium_mtd': self.state.get('premium_collected_mtd', 0),
                'premium_ytd': self.state.get('premium_collected_ytd', 0),
                'annualized_yield': position['income_tracking']['annualized_yield_pct']
            },
            'target_progress': {
                'current_price': position['core_position']['current_price'],
                'target_price': self.TARGET_PRICE,
                'progress_pct': round(
                    (position['core_position']['current_price'] - self.ENTRY_PRICE) /
                    (self.TARGET_PRICE - self.ENTRY_PRICE) * 100, 1
                ),
                'upside_remaining': position['target_analysis']['upside_remaining_pct']
            },
            'active_trades_count': {
                'covered_calls': len(self.state.get('open_covered_calls', [])),
                'protective_puts': len(self.state.get('open_protective_puts', [])),
                'cash_secured_puts': len(self.state.get('cash_secured_puts', []))
            },
            'total_trades_ytd': len([
                t for t in self.state.get('trades_history', [])
                if t.get('timestamp', '').startswith(str(datetime.now().year))
            ])
        }


# Singleton instance
_bot_instance = None

def get_core_position_bot() -> IRENCorePosisionBot:
    """Get singleton bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = IRENCorePosisionBot()
    return _bot_instance


if __name__ == "__main__":
    bot = IRENCorePosisionBot()
    
    print("=" * 70)
    print("IREN CORE POSITION BOT - 100K SHARES")
    print("TARGET: $150 | RULE: NEVER SELL THE SHARES")
    print("=" * 70)
    
    status = bot.get_core_position_status()
    
    print(f"\n📊 CORE POSITION:")
    print(f"   Shares: {status['core_position']['shares']:,}")
    print(f"   Entry: ${status['core_position']['entry_price']:.2f}")
    print(f"   Current: ${status['core_position']['current_price']:.2f}")
    print(f"   Value: ${status['core_position']['position_value']:,.0f}")
    print(f"   P&L: ${status['core_position']['unrealized_pnl']:,.0f} ({status['core_position']['unrealized_pnl_pct']:.1f}%)")
    
    print(f"\n🎯 TARGET ANALYSIS:")
    print(f"   Target Price: ${status['target_analysis']['target_price']:.2f}")
    print(f"   Target Value: ${status['target_analysis']['target_value']:,.0f}")
    print(f"   Upside: {status['target_analysis']['upside_remaining_pct']:.1f}%")
    
    print(f"\n💰 COVERED CALL OPPORTUNITY:")
    cc = bot.should_sell_covered_calls()
    if cc['action'] == 'SELL_COVERED_CALL':
        print(f"   Strike: ${cc['strike']}")
        print(f"   Expiry: {cc['expiry']} ({cc['days_out']}d)")
        print(f"   Contracts: {cc['contracts']}")
        print(f"   Premium: ${cc['total_premium']:,.0f}")
        print(f"   Annual Yield: {cc['annual_yield_pct']:.1f}%")
    else:
        print(f"   Action: {cc['action']}")
        print(f"   Reason: {cc['reason']}")
    
    print(f"\n🛡️ PROTECTION STATUS:")
    protection = bot.should_buy_protection()
    print(f"   Action: {protection['action']}")
    print(f"   Reason: {protection['reason']}")
