"""
NEO Portfolio Manager
Autonomously manages the paper trading portfolio

Responsibilities:
1. Monitor all open positions
2. Execute take profits and stop losses
3. DCA into positions on dips
4. Roll hedges before expiration
5. Rebalance based on market conditions
6. Log all decisions for learning

Runs every 15 minutes during market hours
"""

import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NEO_PortfolioManager")

# Configuration
STATE_FILE = Path("/home/jbot/trading_ai/data/paper_trading_state.json")
DECISIONS_LOG = Path("/home/jbot/trading_ai/neo/portfolio_decisions.json")
NEO_GOLD_API = "http://localhost:8020"
NEO_IREN_API = "http://localhost:8021"
PAPER_TRADING_API = "http://localhost:8787"


class NEOPortfolioManager:
    """
    NEO's autonomous portfolio management system
    """
    
    def __init__(self):
        self.positions = []
        self.decisions = []
        self.market_data = {}
        
        # Strategy rules
        self.rules = {
            # Take profit rules
            "shares_tp_percent": 50.0,      # Take profit on shares at 50% gain
            "calls_tp_percent": 100.0,      # Take profit on calls at 100% gain (2x)
            "scalp_tp_percent": 30.0,       # Scalp calls TP at 30%
            
            # Stop loss rules
            "shares_sl_percent": -20.0,     # Stop loss on shares at -20%
            "calls_sl_percent": -50.0,      # Stop loss on calls at -50%
            "hedge_sl_percent": -80.0,      # Let hedges expire (insurance)
            
            # DCA rules
            "dca_trigger_percent": -5.0,    # DCA when position drops 5%
            "dca_max_entries": 5,           # Max DCA entries per position
            "dca_contracts": 2,             # Contracts per DCA entry
            
            # Hedge rules
            "hedge_roll_days": 30,          # Roll hedges 30 days before expiry
            "hedge_target_ratio": 0.02,     # Keep 2% hedge ratio
            
            # Time rules
            "trading_hours_start": 9,       # 9 AM ET
            "trading_hours_end": 16,        # 4 PM ET
        }
        
        self._load_state()
    
    def _load_state(self):
        """Load current portfolio state"""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    state = json.load(f)
                    self.positions = state.get('positions', [])
                    logger.info(f"Loaded {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save portfolio state"""
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            state['positions'] = self.positions
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _log_decision(self, decision: Dict):
        """Log a trading decision"""
        decision['timestamp'] = datetime.now(timezone.utc).isoformat()
        self.decisions.append(decision)
        
        # Persist decisions
        try:
            existing = []
            if DECISIONS_LOG.exists():
                with open(DECISIONS_LOG) as f:
                    existing = json.load(f)
            existing.append(decision)
            # Keep last 1000 decisions
            existing = existing[-1000:]
            with open(DECISIONS_LOG, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
        
        logger.info(f"📝 DECISION: {decision['action']} {decision['symbol']} - {decision['reason']}")
    
    def _get_market_price(self, symbol: str, is_option: bool = False, 
                          strike: float = None, expiry: str = None,
                          option_type: str = None) -> Optional[float]:
        """Get current market price"""
        try:
            ticker = yf.Ticker(symbol)
            
            if is_option and strike and expiry and option_type:
                # Get option price
                chain = ticker.option_chain(expiry)
                opts = chain.calls if option_type == 'call' else chain.puts
                opt = opts[opts['strike'] == strike]
                if not opt.empty:
                    # Use mid price
                    bid = opt['bid'].iloc[0]
                    ask = opt['ask'].iloc[0]
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2
                    return opt['lastPrice'].iloc[0]
            else:
                # Get stock price
                hist = ticker.history(period='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
        
        return None
    
    def _get_neo_signal(self, symbol: str) -> Optional[Dict]:
        """Get NEO's current signal for a symbol"""
        try:
            if symbol == "XAUUSD":
                resp = requests.get(f"{NEO_GOLD_API}/neo/signal", timeout=5)
            elif symbol == "IREN":
                resp = requests.get(f"{NEO_IREN_API}/api/iren/prediction/summary", timeout=5)
            else:
                return None
            
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to get NEO signal for {symbol}: {e}")
        
        return None
    
    def _execute_trade(self, action: str, position: Dict, 
                       quantity: float = None) -> bool:
        """Execute a trade on the paper trading platform"""
        try:
            if action == "CLOSE":
                # Close position
                resp = requests.post(
                    f"{PAPER_TRADING_API}/paper-trading/positions/{position['id']}/close",
                    timeout=10
                )
                return resp.status_code == 200
            
            elif action == "OPEN":
                # Open new position
                resp = requests.post(
                    f"{PAPER_TRADING_API}/paper-trading/positions/open",
                    json=position,
                    timeout=10
                )
                return resp.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to execute {action}: {e}")
        
        return False
    
    def _close_position_direct(self, position_id: int, exit_price: float, reason: str) -> bool:
        """Close position by updating state directly"""
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            
            # Find and update position
            for i, p in enumerate(state['positions']):
                if p['id'] == position_id:
                    # Calculate final P&L
                    entry = p['entry_price']
                    size = p['size']
                    multiplier = 100 if p.get('is_option') else 1
                    
                    pnl = (exit_price - entry) * size * multiplier
                    if p.get('option_type') == 'put':
                        pnl = (entry - exit_price) * size * multiplier  # Puts gain when price drops
                    
                    # Move to closed positions
                    p['exit_price'] = exit_price
                    p['exit_time'] = datetime.now(timezone.utc).isoformat()
                    p['final_pnl'] = pnl
                    p['close_reason'] = reason
                    p['status'] = 'CLOSED'
                    
                    if 'closed_positions' not in state:
                        state['closed_positions'] = []
                    state['closed_positions'].append(p)
                    
                    # Remove from open positions
                    state['positions'].pop(i)
                    
                    with open(STATE_FILE, 'w') as f:
                        json.dump(state, f, indent=2)
                    
                    logger.info(f"✅ Closed position {position_id}: P&L ${pnl:.2f}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def analyze_position(self, position: Dict) -> Dict:
        """Analyze a single position and decide action"""
        symbol = position['symbol']
        entry_price = position['entry_price']
        size = position['size']
        is_option = position.get('is_option', False)
        option_type = position.get('option_type')
        strike = position.get('strike')
        expiry = position.get('expiry')
        source = position.get('signal_source', '')
        
        # Get current price
        current_price = self._get_market_price(
            symbol, is_option, strike, expiry, option_type
        )
        
        if current_price is None:
            return {"action": "HOLD", "reason": "Cannot get current price"}
        
        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        # Update position with current price
        position['current_price'] = current_price
        position['pnl_percent'] = pnl_percent
        
        # Decision logic based on position type
        if option_type == 'put':
            # HEDGE positions - let them ride unless huge gain
            if pnl_percent >= 200:  # 3x gain = market crashed
                return {
                    "action": "CLOSE",
                    "reason": f"Hedge profit target hit: +{pnl_percent:.1f}%",
                    "price": current_price
                }
            elif expiry:
                # Check if need to roll
                exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                days_to_expiry = (exp_date - datetime.now()).days
                if days_to_expiry <= self.rules['hedge_roll_days']:
                    return {
                        "action": "ROLL_HEDGE",
                        "reason": f"Hedge expiring in {days_to_expiry} days",
                        "price": current_price
                    }
            return {"action": "HOLD", "reason": "Hedge - keep for protection"}
        
        elif option_type == 'call':
            # CALL positions
            if 'SCALP' in source.upper():
                # Scalp calls - tighter targets
                if pnl_percent >= self.rules['scalp_tp_percent']:
                    return {
                        "action": "CLOSE",
                        "reason": f"Scalp TP hit: +{pnl_percent:.1f}%",
                        "price": current_price
                    }
                elif pnl_percent <= self.rules['calls_sl_percent']:
                    return {
                        "action": "CLOSE",
                        "reason": f"Scalp SL hit: {pnl_percent:.1f}%",
                        "price": current_price
                    }
            else:
                # Regular calls
                if pnl_percent >= self.rules['calls_tp_percent']:
                    return {
                        "action": "CLOSE",
                        "reason": f"Call TP hit: +{pnl_percent:.1f}% (2x)",
                        "price": current_price
                    }
                elif pnl_percent <= self.rules['calls_sl_percent']:
                    return {
                        "action": "CLOSE",
                        "reason": f"Call SL hit: {pnl_percent:.1f}%",
                        "price": current_price
                    }
            
            # Check for DCA opportunity
            if pnl_percent <= self.rules['dca_trigger_percent']:
                # Check if we can DCA
                neo_signal = self._get_neo_signal(symbol)
                if neo_signal and neo_signal.get('predicted_direction') == 'UP':
                    return {
                        "action": "DCA",
                        "reason": f"DCA opportunity: {pnl_percent:.1f}% + NEO bullish",
                        "price": current_price
                    }
            
            return {"action": "HOLD", "reason": f"In range: {pnl_percent:.1f}%"}
        
        else:
            # SHARE positions
            if pnl_percent >= self.rules['shares_tp_percent']:
                return {
                    "action": "CLOSE",
                    "reason": f"Shares TP hit: +{pnl_percent:.1f}%",
                    "price": current_price
                }
            elif pnl_percent <= self.rules['shares_sl_percent']:
                return {
                    "action": "CLOSE",
                    "reason": f"Shares SL hit: {pnl_percent:.1f}%",
                    "price": current_price
                }
            
            return {"action": "HOLD", "reason": f"In range: {pnl_percent:.1f}%"}
    
    def run_portfolio_check(self) -> Dict:
        """Run full portfolio analysis and execute decisions"""
        logger.info("=" * 60)
        logger.info("🤖 NEO PORTFOLIO MANAGER - RUNNING CHECK")
        logger.info("=" * 60)
        
        self._load_state()
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions_analyzed": 0,
            "actions_taken": [],
            "portfolio_value": 0,
            "total_pnl": 0
        }
        
        for position in self.positions:
            results['positions_analyzed'] += 1
            
            # Analyze position
            decision = self.analyze_position(position)
            
            symbol = position['symbol']
            pos_type = position.get('option_type', 'shares').upper() if position.get('is_option') else 'SHARES'
            strike = f" ${position.get('strike', '')}" if position.get('is_option') else ""
            
            logger.info(f"📊 {symbol}{strike} {pos_type}: {decision['action']} - {decision['reason']}")
            
            # Execute action if needed
            if decision['action'] == 'CLOSE':
                success = self._close_position_direct(
                    position['id'], 
                    decision['price'],
                    decision['reason']
                )
                if success:
                    results['actions_taken'].append({
                        "action": "CLOSED",
                        "symbol": symbol,
                        "reason": decision['reason']
                    })
                    self._log_decision({
                        "action": "CLOSE",
                        "symbol": symbol,
                        "position_id": position['id'],
                        "price": decision['price'],
                        "reason": decision['reason']
                    })
            
            elif decision['action'] == 'DCA':
                # Log DCA opportunity (manual review for now)
                self._log_decision({
                    "action": "DCA_OPPORTUNITY",
                    "symbol": symbol,
                    "position_id": position['id'],
                    "current_pnl": position.get('pnl_percent', 0),
                    "reason": decision['reason']
                })
                results['actions_taken'].append({
                    "action": "DCA_FLAGGED",
                    "symbol": symbol,
                    "reason": decision['reason']
                })
            
            elif decision['action'] == 'ROLL_HEDGE':
                self._log_decision({
                    "action": "ROLL_HEDGE_NEEDED",
                    "symbol": symbol,
                    "position_id": position['id'],
                    "expiry": position.get('expiry'),
                    "reason": decision['reason']
                })
                results['actions_taken'].append({
                    "action": "ROLL_FLAGGED",
                    "symbol": symbol,
                    "reason": decision['reason']
                })
            
            # Calculate portfolio value
            size = position['size']
            price = position.get('current_price', position['entry_price'])
            multiplier = 100 if position.get('is_option') else 1
            results['portfolio_value'] += size * price * multiplier
            results['total_pnl'] += position.get('pnl', 0)
        
        self._save_state()
        
        logger.info("=" * 60)
        logger.info(f"📈 Portfolio Value: ${results['portfolio_value']:,.2f}")
        logger.info(f"💰 Total P&L: ${results['total_pnl']:,.2f}")
        logger.info(f"🎯 Actions Taken: {len(results['actions_taken'])}")
        logger.info("=" * 60)
        
        return results
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        self._load_state()
        
        # Group positions
        shares = [p for p in self.positions if not p.get('is_option')]
        calls = [p for p in self.positions if p.get('option_type') == 'call']
        puts = [p for p in self.positions if p.get('option_type') == 'put']
        
        # Calculate values
        shares_value = sum(p['size'] * (p.get('current_price') or p['entry_price']) for p in shares)
        calls_value = sum(p['size'] * (p.get('current_price') or p['entry_price']) * 100 for p in calls)
        puts_value = sum(p['size'] * (p.get('current_price') or p['entry_price']) * 100 for p in puts)
        
        total_pnl = sum(p.get('pnl', 0) for p in self.positions)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": {
                "shares": len(shares),
                "calls": len(calls),
                "puts": len(puts),
                "total": len(self.positions)
            },
            "values": {
                "shares": shares_value,
                "calls": calls_value,
                "hedges": puts_value,
                "total": shares_value + calls_value + puts_value
            },
            "pnl": {
                "total": total_pnl,
                "percent": (total_pnl / (shares_value + calls_value + puts_value)) * 100 if (shares_value + calls_value + puts_value) > 0 else 0
            },
            "hedge_ratio": puts_value / (shares_value + calls_value) if (shares_value + calls_value) > 0 else 0
        }


# Create global instance
manager = NEOPortfolioManager()


def run_check():
    """Run portfolio check (called by cron)"""
    return manager.run_portfolio_check()


def get_status():
    """Get portfolio status"""
    return manager.get_portfolio_status()


if __name__ == "__main__":
    # Run portfolio check
    result = run_check()
    print(json.dumps(result, indent=2))
