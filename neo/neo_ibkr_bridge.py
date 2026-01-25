"""
NEO-IBKR Bridge
Connects NEO's IREN signal system to Interactive Brokers for automated execution

Flow:
1. NEO generates IREN signal
2. Bridge validates against Paul's rules
3. Bridge executes on IBKR (paper or live)
4. Bridge tracks positions and P&L

Usage:
    bridge = NeoIBKRBridge(paper_trading=True, auto_execute=False)
    bridge.connect()
    bridge.run_signal_loop()  # or bridge.process_signal_once()
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import json
import time
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo.ibkr_connector import IBKRConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "neo" / "ibkr_data"
DATA_DIR.mkdir(exist_ok=True)


class NeoIBKRBridge:
    """
    Bridge between NEO signals and IBKR execution
    """
    
    def __init__(self, paper_trading: bool = True, auto_execute: bool = False,
                 min_confidence: int = 70):
        """
        Initialize bridge
        
        Args:
            paper_trading: True for paper trading mode
            auto_execute: True to automatically execute signals
            min_confidence: Minimum signal confidence to act on
        """
        self.ibkr = IBKRConnector(paper_trading=paper_trading)
        self.auto_execute = auto_execute
        self.min_confidence = min_confidence
        self.paper_trading = paper_trading
        
        # Signal state
        self.pending_signals = []
        self.executed_signals = []
        self.rejected_signals = []
        
        # State file
        self.state_file = DATA_DIR / "bridge_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load saved state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.pending_signals = state.get('pending_signals', [])
                    self.executed_signals = state.get('executed_signals', [])[-50:]
                    self.rejected_signals = state.get('rejected_signals', [])[-50:]
            except:
                pass
    
    def _save_state(self):
        """Save state"""
        state = {
            'pending_signals': self.pending_signals,
            'executed_signals': self.executed_signals[-50:],
            'rejected_signals': self.rejected_signals[-50:],
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def connect(self) -> bool:
        """Connect to IBKR"""
        result = self.ibkr.connect()
        if result:
            time.sleep(2)
            logger.info("✅ NEO-IBKR Bridge connected")
            return True
        else:
            logger.error("❌ Failed to connect to IBKR")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        self.ibkr.disconnect()
    
    def get_neo_iren_signal(self) -> Dict:
        """
        Get latest IREN signal from NEO/paper trading system
        
        Returns signal in standardized format:
        {
            'action': 'BUY_CALL' | 'HOLD' | etc.,
            'symbol': 'IREN',
            'expiry': '20260220',
            'strike': 60.0,
            'quantity': 5,
            'limit_price': 2.50,
            'confidence': 85,
            'reason': 'String explanation',
            'source': 'neo_daily' | 'neo_realtime' | etc.
        }
        """
        try:
            # Try to load from NEO's daily signal
            signal_file = Path(__file__).parent.parent / "paper_trading" / "locked_signal.json"
            
            if signal_file.exists():
                with open(signal_file, 'r') as f:
                    data = json.load(f)
                
                iren_signal = data.get('iren', {})
                
                if not iren_signal:
                    return self._get_default_signal()
                
                # Parse expiry from format "2026-02-20" to "20260220"
                expiries = iren_signal.get('all_expiries', [])
                preferred_expiry = None
                
                for exp in expiries:
                    if exp.get('is_paul_pick'):
                        preferred_expiry = exp.get('expiry', '').replace('-', '')
                        break
                
                if not preferred_expiry and expiries:
                    preferred_expiry = expiries[0].get('expiry', '20260220').replace('-', '')
                
                # Determine action based on signal
                action = iren_signal.get('recommended_action', 'HOLD')
                if action == 'BUY':
                    action = 'BUY_CALL'
                elif action == 'HOLD':
                    action = 'HOLD'
                else:
                    action = 'HOLD'
                
                # Get strike
                strike = iren_signal.get('optimal_strike', 60)
                
                # Calculate suggested quantity based on confidence
                confidence = iren_signal.get('confidence', 50)
                if confidence >= 80:
                    suggested_qty = 10
                elif confidence >= 70:
                    suggested_qty = 5
                elif confidence >= 60:
                    suggested_qty = 3
                else:
                    suggested_qty = 1
                
                return {
                    'action': action,
                    'symbol': 'IREN',
                    'expiry': preferred_expiry or '20260220',
                    'strike': strike,
                    'quantity': suggested_qty,
                    'limit_price': None,  # Use market order or calculate from entry
                    'confidence': confidence,
                    'reason': iren_signal.get('thesis', 'NEO daily signal'),
                    'source': 'neo_daily',
                    'timestamp': datetime.now().isoformat()
                }
            
            return self._get_default_signal()
            
        except Exception as e:
            logger.error(f"Error getting NEO signal: {e}")
            return self._get_default_signal()
    
    def _get_default_signal(self) -> Dict:
        """Return default HOLD signal"""
        return {
            'action': 'HOLD',
            'symbol': 'IREN',
            'expiry': '20260220',
            'strike': 60,
            'quantity': 0,
            'limit_price': None,
            'confidence': 0,
            'reason': 'No active signal',
            'source': 'default',
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_signal(self, signal: Dict) -> Dict:
        """
        Validate signal against Paul's rules and IBKR requirements
        
        Returns:
            Dict with 'valid': bool, 'reason': str, 'warnings': list
        """
        warnings = []
        
        # Check action
        if signal['action'] not in ['BUY_CALL', 'HOLD']:
            return {
                'valid': False,
                'reason': f"Action '{signal['action']}' not allowed for Paul (LONG ONLY)",
                'warnings': warnings
            }
        
        # HOLD is always valid but no execution needed
        if signal['action'] == 'HOLD':
            return {
                'valid': True,
                'reason': 'HOLD signal - no action needed',
                'warnings': warnings,
                'execute': False
            }
        
        # Check confidence
        if signal['confidence'] < self.min_confidence:
            return {
                'valid': False,
                'reason': f"Confidence {signal['confidence']}% below threshold {self.min_confidence}%",
                'warnings': warnings
            }
        
        # Check expiry
        blocked = IBKRConnector.PAUL_RULES['blocked_expirations']
        if signal['expiry'] in blocked:
            return {
                'valid': False,
                'reason': f"Expiry {signal['expiry']} is blocked (near earnings)",
                'warnings': warnings
            }
        
        # Check DTE
        try:
            exp_date = datetime.strptime(signal['expiry'], "%Y%m%d").date()
            dte = (exp_date - datetime.now().date()).days
            
            if dte < IBKRConnector.PAUL_RULES['min_dte']:
                return {
                    'valid': False,
                    'reason': f"DTE {dte} is below minimum {IBKRConnector.PAUL_RULES['min_dte']}",
                    'warnings': warnings
                }
            
            # Warning for short DTE
            if dte < 21:
                warnings.append(f"DTE {dte} is relatively short - theta decay accelerating")
                
        except ValueError:
            return {
                'valid': False,
                'reason': f"Invalid expiry format: {signal['expiry']}",
                'warnings': warnings
            }
        
        # Check quantity
        max_per_trade = IBKRConnector.PAUL_RULES['max_contracts_per_trade']
        if signal['quantity'] > max_per_trade:
            warnings.append(f"Quantity {signal['quantity']} exceeds max {max_per_trade}, will be reduced")
            signal['quantity'] = max_per_trade
        
        # Check if Paul's preferred expiry
        preferred = IBKRConnector.PAUL_RULES['preferred_expirations']
        if signal['expiry'] not in preferred:
            warnings.append(f"Expiry {signal['expiry']} is not Paul's preferred (Feb 20 or Feb 27)")
        
        return {
            'valid': True,
            'reason': 'Signal passed all validations',
            'warnings': warnings,
            'execute': True
        }
    
    def execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a validated signal on IBKR
        
        Returns:
            Execution result dict
        """
        if not self.ibkr.is_connected():
            return {
                'success': False,
                'error': 'Not connected to IBKR'
            }
        
        # Validate first
        validation = self.validate_signal(signal)
        
        if not validation['valid']:
            self.rejected_signals.append({
                'signal': signal,
                'reason': validation['reason'],
                'timestamp': datetime.now().isoformat()
            })
            self._save_state()
            return {
                'success': False,
                'error': validation['reason']
            }
        
        if not validation.get('execute', True):
            return {
                'success': True,
                'message': validation['reason'],
                'action_taken': False
            }
        
        # Execute based on action
        if signal['action'] == 'BUY_CALL':
            result = self.ibkr.buy_iren_call(
                expiry=signal['expiry'],
                strike=signal['strike'],
                quantity=signal['quantity'],
                limit_price=signal.get('limit_price'),
                validate_rules=False  # Already validated above
            )
            
            if result.get('success'):
                self.executed_signals.append({
                    'signal': signal,
                    'result': result,
                    'warnings': validation.get('warnings', []),
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
            
            return result
        
        return {
            'success': False,
            'error': f"Unknown action: {signal['action']}"
        }
    
    def process_signal_once(self) -> Dict:
        """
        Get and process one signal
        
        Returns:
            Processing result
        """
        signal = self.get_neo_iren_signal()
        
        logger.info(f"📡 NEO Signal: {signal['action']} IREN {signal['strike']}C "
                   f"{signal['expiry']} ({signal['confidence']}%)")
        
        validation = self.validate_signal(signal)
        
        if not validation['valid']:
            logger.warning(f"❌ Signal rejected: {validation['reason']}")
            return {
                'signal': signal,
                'validation': validation,
                'executed': False
            }
        
        if signal['action'] == 'HOLD':
            logger.info("⏸️ HOLD signal - no action")
            return {
                'signal': signal,
                'validation': validation,
                'executed': False,
                'message': 'HOLD - no action needed'
            }
        
        if self.auto_execute:
            logger.info("🚀 Auto-executing signal...")
            result = self.execute_signal(signal)
            return {
                'signal': signal,
                'validation': validation,
                'executed': True,
                'result': result
            }
        else:
            logger.info("⏸️ Auto-execute OFF - signal added to pending")
            self.pending_signals.append({
                'signal': signal,
                'validation': validation,
                'added_at': datetime.now().isoformat()
            })
            self._save_state()
            return {
                'signal': signal,
                'validation': validation,
                'executed': False,
                'message': 'Signal added to pending (auto-execute OFF)'
            }
    
    def execute_pending(self) -> List[Dict]:
        """
        Execute all pending signals
        
        Returns:
            List of execution results
        """
        results = []
        
        for pending in self.pending_signals:
            signal = pending['signal']
            result = self.execute_signal(signal)
            results.append({
                'signal': signal,
                'result': result
            })
        
        self.pending_signals.clear()
        self._save_state()
        
        return results
    
    def run_signal_loop(self, check_interval: int = 60, max_iterations: int = None):
        """
        Continuously check for signals and execute
        
        Args:
            check_interval: Seconds between checks
            max_iterations: Maximum iterations (None = infinite)
        """
        logger.info("🚀 NEO-IBKR Bridge signal loop starting...")
        logger.info(f"   Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        logger.info(f"   Auto-execute: {self.auto_execute}")
        logger.info(f"   Min confidence: {self.min_confidence}%")
        logger.info(f"   Check interval: {check_interval}s")
        
        iteration = 0
        
        while True:
            try:
                result = self.process_signal_once()
                
                if result.get('executed'):
                    logger.info(f"✅ Signal executed: {result.get('result', {}).get('message', '')}")
                
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Signal loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in signal loop: {e}")
                time.sleep(check_interval)
    
    def get_bridge_status(self) -> Dict:
        """Get complete bridge status"""
        return {
            'connected': self.ibkr.is_connected(),
            'mode': 'PAPER' if self.paper_trading else 'LIVE',
            'auto_execute': self.auto_execute,
            'min_confidence': self.min_confidence,
            'pending_signals': len(self.pending_signals),
            'executed_today': len([e for e in self.executed_signals 
                                   if e.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))]),
            'ibkr_status': self.ibkr.get_full_status() if self.ibkr.is_connected() else None,
            'last_signal': self.get_neo_iren_signal(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_positions_summary(self) -> Dict:
        """Get IREN positions summary"""
        if not self.ibkr.is_connected():
            return {'error': 'Not connected'}
        
        positions = self.ibkr.get_iren_positions()
        account = self.ibkr.get_account_summary()
        
        total_value = sum(p.get('market_value', 0) for p in positions)
        total_pnl = sum(p.get('pnl', 0) for p in positions)
        
        return {
            'positions': positions,
            'total_contracts': sum(p.get('quantity', 0) for p in positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / total_value * 100) if total_value > 0 else 0,
            'account': {
                'net_liquidation': account.get('NetLiquidation', 0),
                'buying_power': account.get('BuyingPower', 0),
                'available_funds': account.get('AvailableFunds', 0)
            },
            'timestamp': datetime.now().isoformat()
        }


# Standalone test
if __name__ == "__main__":
    print("=" * 70)
    print("🏦 NEO-IBKR BRIDGE TEST")
    print("=" * 70)
    
    bridge = NeoIBKRBridge(paper_trading=True, auto_execute=False)
    
    # Test signal generation (without IBKR connection)
    print("\n📡 Testing signal generation...")
    signal = bridge.get_neo_iren_signal()
    print(f"Signal: {json.dumps(signal, indent=2)}")
    
    print("\n✅ Validating signal...")
    validation = bridge.validate_signal(signal)
    print(f"Validation: {json.dumps(validation, indent=2)}")
    
    print("\n🔌 To test with IBKR:")
    print("1. Start TWS/IB Gateway")
    print("2. Login to Paper Trading")
    print("3. Enable API on port 7497")
    print("4. Run: bridge.connect() then bridge.process_signal_once()")
