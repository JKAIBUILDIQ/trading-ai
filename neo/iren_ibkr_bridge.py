#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
IREN PREDICTOR → IBKR BRIDGE
═══════════════════════════════════════════════════════════════════════════════

Connects IREN 4-hour predictions to Interactive Brokers for options trading.

Flow:
1. Fetch IREN prediction from API (port 8021)
2. If signal = BUY_CALLS and confidence >= threshold
3. Format options order based on best_option
4. Send to IBKR via Tailscale connection to desktop TWS

Configuration:
- TWS running on desktop (Gringot) via Tailscale
- Paper trading mode by default
- Auto-execute OFF by default (requires manual enable)

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import asyncio
import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from neo.ibkr_connector import IBKRConnector
except ImportError:
    from ibkr_connector import IBKRConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IrenIBKRBridge")

# Configuration
IREN_API_URL = "http://localhost:8021"
DATA_DIR = Path(__file__).parent / "ibkr_data"
DATA_DIR.mkdir(exist_ok=True)
SIGNAL_HISTORY_FILE = DATA_DIR / "iren_ibkr_signals.json"


class IrenIBKRBridge:
    """
    Bridge between IREN Predictor and Interactive Brokers
    
    Fetches predictions, validates against rules, optionally executes.
    """
    
    # Paul's Trading Rules
    PAUL_RULES = {
        'min_confidence': 60,         # Minimum confidence to consider
        'min_dte': 14,                # Minimum days to expiration
        'max_dte': 45,                # Maximum days to expiration
        'preferred_dte_min': 21,      # Paul's sweet spot start
        'preferred_dte_max': 35,      # Paul's sweet spot end
        'allowed_strikes': [60, 70, 80],
        'default_quantity': 5,        # Default contracts
        'max_quantity': 50,           # Max contracts per order
        'earnings_buffer_days': 7,    # Avoid expiries near earnings
    }
    
    def __init__(self, 
                 paper_trading: bool = True,
                 auto_execute: bool = False,
                 min_confidence: int = 60,
                 default_quantity: int = 5):
        """
        Initialize the bridge
        
        Args:
            paper_trading: Use paper trading mode
            auto_execute: Automatically execute signals (dangerous!)
            min_confidence: Minimum confidence to consider signal
            default_quantity: Default number of contracts
        """
        self.paper_trading = paper_trading
        self.auto_execute = auto_execute
        self.min_confidence = min_confidence
        self.default_quantity = default_quantity
        
        self.ibkr: Optional[IBKRConnector] = None
        self.connected = False
        self.last_signal_id: Optional[str] = None
        self.signal_history: List[Dict] = []
        
        self._load_history()
        
        logger.info("=" * 60)
        logger.info("🌉 IREN-IBKR BRIDGE INITIALIZED")
        logger.info(f"   Paper Trading: {self.paper_trading}")
        logger.info(f"   Auto-Execute: {self.auto_execute}")
        logger.info(f"   Min Confidence: {self.min_confidence}%")
        logger.info("=" * 60)
    
    def _load_history(self):
        """Load signal history"""
        try:
            if SIGNAL_HISTORY_FILE.exists():
                with open(SIGNAL_HISTORY_FILE, 'r') as f:
                    self.signal_history = json.load(f)
                logger.info(f"Loaded {len(self.signal_history)} historical signals")
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            self.signal_history = []
    
    def _save_history(self):
        """Save signal history"""
        try:
            with open(SIGNAL_HISTORY_FILE, 'w') as f:
                json.dump(self.signal_history[-1000:], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save history: {e}")
    
    def connect_ibkr(self) -> bool:
        """Connect to Interactive Brokers"""
        try:
            self.ibkr = IBKRConnector(paper_trading=self.paper_trading)
            success = self.ibkr.connect()
            self.connected = success
            
            if success:
                logger.info("✅ Connected to IBKR")
            else:
                logger.error("❌ Failed to connect to IBKR")
            
            return success
        except Exception as e:
            logger.error(f"❌ IBKR connection error: {e}")
            self.connected = False
            return False
    
    def fetch_prediction(self) -> Optional[Dict]:
        """Fetch latest prediction from IREN API"""
        try:
            response = requests.get(f"{IREN_API_URL}/api/iren/prediction/summary", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching prediction: {e}")
            return None
    
    def fetch_options(self) -> Optional[Dict]:
        """Fetch options recommendations"""
        try:
            response = requests.get(f"{IREN_API_URL}/api/iren/prediction/options", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.error(f"Error fetching options: {e}")
            return None
    
    def validate_signal(self, prediction: Dict) -> tuple[bool, str]:
        """
        Validate signal against Paul's rules
        
        Returns:
            (valid, reason)
        """
        # Check signal type
        signal = prediction.get('signal', '')
        if signal != 'BUY_CALLS':
            return False, f"Signal is {signal}, not BUY_CALLS"
        
        # Check confidence
        confidence = prediction.get('confidence', 0)
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence}% below threshold {self.min_confidence}%"
        
        # Check best option
        best_opt = prediction.get('best_option')
        if not best_opt or not best_opt.get('strike'):
            return False, "No best option available"
        
        # Check DTE
        dte = best_opt.get('dte', 0)
        if dte < self.PAUL_RULES['min_dte']:
            return False, f"DTE {dte} below minimum {self.PAUL_RULES['min_dte']}"
        
        # Check strike
        strike = best_opt.get('strike')
        if strike not in self.PAUL_RULES['allowed_strikes']:
            return False, f"Strike ${strike} not in allowed list"
        
        # Check earnings proximity
        earnings_days = prediction.get('earnings_days_away', 100)
        if earnings_days and 0 < earnings_days < self.PAUL_RULES['earnings_buffer_days']:
            return False, f"Too close to earnings ({earnings_days} days)"
        
        # Check if already processed this signal
        signal_id = prediction.get('prediction_id')
        if signal_id == self.last_signal_id:
            return False, "Signal already processed"
        
        return True, "Signal validated"
    
    def format_order(self, prediction: Dict, quantity: Optional[int] = None) -> Dict:
        """Format the order for IBKR execution"""
        best_opt = prediction.get('best_option', {})
        
        return {
            'symbol': 'IREN',
            'action': 'BUY',
            'option_type': 'CALL',
            'strike': best_opt.get('strike', 60),
            'expiry': best_opt.get('expiration', '').replace('-', ''),  # YYYYMMDD format
            'quantity': quantity or self.default_quantity,
            'order_type': 'LIMIT',
            'limit_price': best_opt.get('price') or (best_opt.get('ask', 0) + best_opt.get('bid', 0)) / 2,
            'prediction_id': prediction.get('prediction_id'),
            'confidence': prediction.get('confidence'),
            'reasoning': prediction.get('reasoning')
        }
    
    def execute_order(self, order: Dict) -> Dict:
        """Execute order on IBKR"""
        if not self.connected:
            if not self.connect_ibkr():
                return {'success': False, 'error': 'Not connected to IBKR'}
        
        try:
            result = self.ibkr.buy_iren_call(
                expiry=order['expiry'],
                strike=order['strike'],
                quantity=order['quantity'],
                limit_price=order.get('limit_price')
            )
            
            return {
                'success': result.get('success', False),
                'order_id': result.get('order', {}).get('orderId'),
                'status': result.get('status'),
                'order': order,
                'ibkr_result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order': order
            }
    
    def process_signal(self, execute: bool = False, quantity: Optional[int] = None) -> Dict:
        """
        Process current IREN signal
        
        Args:
            execute: Whether to actually execute (overrides auto_execute)
            quantity: Override default quantity
        
        Returns:
            Signal processing result
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔍 PROCESSING IREN SIGNAL")
        logger.info("=" * 60)
        
        # Fetch prediction
        prediction = self.fetch_prediction()
        if not prediction:
            return {'success': False, 'error': 'Could not fetch prediction'}
        
        logger.info(f"\n📊 Current Prediction:")
        logger.info(f"   Direction: {prediction.get('predicted_direction')}")
        logger.info(f"   Confidence: {prediction.get('confidence')}%")
        logger.info(f"   Signal: {prediction.get('signal')}")
        
        # Validate
        valid, reason = self.validate_signal(prediction)
        
        if not valid:
            logger.info(f"\n❌ Signal Rejected: {reason}")
            return {
                'success': False,
                'valid': False,
                'reason': reason,
                'prediction': prediction
            }
        
        logger.info(f"\n✅ Signal Validated: {reason}")
        
        # Format order
        order = self.format_order(prediction, quantity)
        
        logger.info(f"\n📝 Order Prepared:")
        logger.info(f"   Symbol: IREN")
        logger.info(f"   Action: BUY CALL")
        logger.info(f"   Strike: ${order['strike']}")
        logger.info(f"   Expiry: {order['expiry']}")
        logger.info(f"   Quantity: {order['quantity']}")
        logger.info(f"   Limit: ${order.get('limit_price', 'MKT'):.2f}")
        
        # Record signal
        signal_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'prediction_id': prediction.get('prediction_id'),
            'signal': prediction.get('signal'),
            'confidence': prediction.get('confidence'),
            'order': order,
            'executed': False,
            'result': None
        }
        
        # Execute if enabled
        should_execute = execute or self.auto_execute
        
        if should_execute:
            logger.info(f"\n🚀 EXECUTING ORDER...")
            result = self.execute_order(order)
            signal_record['executed'] = True
            signal_record['result'] = result
            
            if result.get('success'):
                logger.info(f"✅ Order placed! ID: {result.get('order_id')}")
            else:
                logger.error(f"❌ Order failed: {result.get('error')}")
        else:
            logger.info(f"\n⏸️ Auto-execute OFF - Order NOT sent")
            logger.info(f"   To execute: bridge.process_signal(execute=True)")
        
        # Update state
        self.last_signal_id = prediction.get('prediction_id')
        self.signal_history.append(signal_record)
        self._save_history()
        
        return {
            'success': True,
            'valid': True,
            'order': order,
            'executed': should_execute,
            'result': signal_record.get('result'),
            'prediction': prediction
        }
    
    def get_pending_signals(self) -> List[Dict]:
        """Get signals that are valid but not yet executed"""
        prediction = self.fetch_prediction()
        if not prediction:
            return []
        
        valid, _ = self.validate_signal(prediction)
        if valid:
            return [{
                'prediction': prediction,
                'order': self.format_order(prediction)
            }]
        return []
    
    def get_status(self) -> Dict:
        """Get bridge status"""
        prediction = self.fetch_prediction()
        
        return {
            'connected_ibkr': self.connected,
            'auto_execute': self.auto_execute,
            'min_confidence': self.min_confidence,
            'default_quantity': self.default_quantity,
            'paper_trading': self.paper_trading,
            'last_signal_id': self.last_signal_id,
            'signals_processed': len(self.signal_history),
            'current_prediction': prediction,
            'pending_signals': len(self.get_pending_signals())
        }
    
    async def run_signal_loop(self, interval_minutes: int = 60):
        """
        Continuous signal monitoring loop
        
        Args:
            interval_minutes: How often to check for new signals
        """
        logger.info(f"\n🔄 Starting signal loop (interval: {interval_minutes}m)")
        
        while True:
            try:
                result = self.process_signal()
                
                if result.get('success') and result.get('valid'):
                    logger.info(f"✅ Valid signal processed")
                else:
                    logger.info(f"⏳ No actionable signal")
                
            except Exception as e:
                logger.error(f"Error in signal loop: {e}")
            
            await asyncio.sleep(interval_minutes * 60)
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ibkr:
            try:
                self.ibkr.disconnect()
            except:
                pass
        self.connected = False


def get_bridge_status():
    """Quick function to get bridge status"""
    bridge = IrenIBKRBridge(auto_execute=False)
    return bridge.get_status()


def process_iren_signal(execute: bool = False, quantity: int = 5):
    """Quick function to process current signal"""
    bridge = IrenIBKRBridge(auto_execute=False)
    return bridge.process_signal(execute=execute, quantity=quantity)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌉 IREN-IBKR BRIDGE TEST")
    print("=" * 70)
    
    bridge = IrenIBKRBridge(
        paper_trading=True,
        auto_execute=False,  # SAFETY: Don't auto-execute
        min_confidence=60
    )
    
    # Process signal (without executing)
    result = bridge.process_signal(execute=False)
    
    print("\n" + "=" * 70)
    print("📊 RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str)[:2000])
