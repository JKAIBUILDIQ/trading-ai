"""
BTC Miners Autonomous Trader
============================
Daily trading system for IREN, CLSK, CIFR

These are former BTC miners pivoting to AI/hyperscaling with:
- Built-in power infrastructure
- Land and facilities
- Grandfathered agreements
- Tremendous growth potential

Trading Strategy:
1. Long-term: Accumulate shares on dips (targeting $100-150)
2. Short-term: Scalp options on 5%+ moves
3. DCA: Add on 5%+ dips when Meta Bot bullish
4. Exit: Take profit at targets, trail on momentum

Runs every 4 hours during market hours
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
logger = logging.getLogger("BTC_Miners_Trader")

# Configuration
PAPER_TRADING_API = "http://localhost:8500"
META_BOT_API = "http://localhost:8035"
STATE_FILE = Path("/home/jbot/trading_ai/data/btc_miners_state.json")
DECISIONS_LOG = Path("/home/jbot/trading_ai/neo/btc_miners_decisions.json")

# Target stocks
BTC_MINERS = {
    "IREN": {
        "name": "Iris Energy",
        "thesis": "AI data centers with 510MW capacity, strong BTC correlation",
        "target_price": 150.0,
        "accumulation_target": 1000,  # shares to accumulate
        "dca_trigger": -5.0,          # % drop to trigger DCA
        "tp_shares": 50.0,            # % gain to take profit on shares
        "tp_options": 100.0,          # % gain to take profit on options
        "sl_options": -50.0,          # % loss to cut options
    },
    "CLSK": {
        "name": "CleanSpark",
        "thesis": "BTC mining + diversifying to AI compute, low-cost operations",
        "target_price": 50.0,
        "accumulation_target": 500,
        "dca_trigger": -5.0,
        "tp_shares": 50.0,
        "tp_options": 100.0,
        "sl_options": -50.0,
    },
    "CIFR": {
        "name": "Cipher Mining",
        "thesis": "Zero-carbon BTC mining, pivoting to HPC/AI hosting",
        "target_price": 30.0,
        "accumulation_target": 500,
        "dca_trigger": -5.0,
        "tp_shares": 50.0,
        "tp_options": 100.0,
        "sl_options": -50.0,
    }
}


class BTCMinersTrader:
    """
    Autonomous daily trader for BTC miner stocks
    Uses Meta Bot signals + technical analysis
    """
    
    def __init__(self):
        self.positions = {}
        self.signals = {}
        self.decisions = []
        self._load_state()
    
    def _load_state(self):
        """Load current state"""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            self.positions = {}
    
    def _save_state(self):
        """Save state"""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump({
                    'positions': self.positions,
                    'last_updated': datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _log_decision(self, decision: Dict):
        """Log trading decision"""
        decision['timestamp'] = datetime.utcnow().isoformat()
        self.decisions.append(decision)
        
        try:
            DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing decisions
            existing = []
            if DECISIONS_LOG.exists():
                with open(DECISIONS_LOG) as f:
                    existing = json.load(f)
            
            # Keep last 500 decisions
            existing.append(decision)
            existing = existing[-500:]
            
            with open(DECISIONS_LOG, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
        
        logger.info(f"📝 {decision['symbol']}: {decision['action']} - {decision['reason']}")
    
    def get_meta_signal(self, symbol: str) -> Optional[Dict]:
        """Get Meta Bot signal for a symbol"""
        try:
            resp = requests.get(
                f"{META_BOT_API}/api/meta/{symbol.lower()}/signal",
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to get Meta signal for {symbol}: {e}")
        return None
    
    def get_current_price(self, symbol: str) -> float:
        """Get real-time price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
        return 0.0
    
    def get_paper_positions(self, symbol: str) -> List[Dict]:
        """Get paper trading positions for a symbol"""
        try:
            resp = requests.get(f"{PAPER_TRADING_API}/paper-trading/positions", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # Handle both formats: direct list or {"positions": [...]}
                if isinstance(data, dict):
                    all_positions = data.get('positions', [])
                else:
                    all_positions = data
                return [p for p in all_positions if p.get('symbol') == symbol and p.get('status') == 'OPEN']
        except Exception as e:
            logger.warning(f"Failed to get positions for {symbol}: {e}")
        return []
    
    def execute_paper_trade(self, symbol: str, action: str, size: int, 
                           is_option: bool = False, strike: float = None,
                           expiry: str = None, option_type: str = 'call') -> bool:
        """Execute a paper trade"""
        try:
            payload = {
                "symbol": symbol,
                "type": action,  # LONG, SHORT, BUY_CALL, BUY_PUT
                "size": size,
                "is_option": is_option,
                "source": "BTC_MINERS_TRADER",
                "confidence": 75
            }
            
            if is_option:
                payload['strike'] = strike
                payload['expiry'] = expiry
                payload['option_type'] = option_type
            
            resp = requests.post(
                f"{PAPER_TRADING_API}/paper-trading/open",
                json=payload,
                timeout=10
            )
            
            if resp.status_code == 200:
                logger.info(f"✅ Executed {action} {size} {symbol}")
                return True
            else:
                logger.error(f"❌ Failed to execute trade: {resp.text}")
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
        
        return False
    
    def close_paper_position(self, position_id: int, reason: str) -> bool:
        """Close a paper trading position"""
        try:
            resp = requests.post(
                f"{PAPER_TRADING_API}/paper-trading/close/{position_id}",
                json={"reason": reason},
                timeout=10
            )
            if resp.status_code == 200:
                return True
        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
        return False
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a symbol and determine trading action"""
        config = BTC_MINERS[symbol]
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price == 0:
            return {"action": "HOLD", "reason": "No price data"}
        
        # Get Meta Bot signal
        meta_signal = self.get_meta_signal(symbol)
        
        # Get current positions
        positions = self.get_paper_positions(symbol)
        
        # Calculate position stats
        share_positions = [p for p in positions if not p.get('is_option')]
        option_positions = [p for p in positions if p.get('is_option')]
        
        total_shares = sum(p.get('size', 0) for p in share_positions)
        avg_share_cost = 0
        if share_positions:
            total_cost = sum(p.get('entry_price', 0) * p.get('size', 0) for p in share_positions)
            avg_share_cost = total_cost / total_shares if total_shares > 0 else 0
        
        # Calculate share P&L
        share_pnl_pct = ((current_price - avg_share_cost) / avg_share_cost * 100) if avg_share_cost > 0 else 0
        
        # Distance to target
        target_distance_pct = ((config['target_price'] - current_price) / current_price * 100)
        
        analysis = {
            "symbol": symbol,
            "current_price": current_price,
            "target_price": config['target_price'],
            "target_distance_pct": round(target_distance_pct, 2),
            "total_shares": total_shares,
            "avg_cost": round(avg_share_cost, 2),
            "share_pnl_pct": round(share_pnl_pct, 2),
            "option_positions": len(option_positions),
            "meta_signal": meta_signal.get('action', 'UNKNOWN') if meta_signal else 'NO_SIGNAL',
            "meta_confidence": meta_signal.get('confidence', 0) if meta_signal else 0,
            "action": "HOLD",
            "reason": "No action needed"
        }
        
        # Decision Logic
        
        # 1. Check for take profit on shares (50%+ gain)
        if share_pnl_pct >= config['tp_shares']:
            analysis['action'] = "TAKE_PROFIT_SHARES"
            analysis['reason'] = f"Shares up {share_pnl_pct:.1f}% - taking profit"
            return analysis
        
        # 2. Check for DCA opportunity (price dropped 5%+ from avg cost, Meta bullish)
        if total_shares > 0 and share_pnl_pct <= config['dca_trigger']:
            if meta_signal and meta_signal.get('action') in ['BUY', 'STRONG_BUY']:
                analysis['action'] = "DCA_SHARES"
                analysis['reason'] = f"Down {share_pnl_pct:.1f}%, Meta {meta_signal.get('action')} - DCA"
                return analysis
        
        # 3. Check if we should accumulate more shares (under target)
        if total_shares < config['accumulation_target']:
            if meta_signal and meta_signal.get('action') in ['BUY', 'STRONG_BUY']:
                if meta_signal.get('confidence', 0) >= 60:
                    analysis['action'] = "ACCUMULATE"
                    analysis['reason'] = f"Only {total_shares}/{config['accumulation_target']} shares, Meta bullish"
                    return analysis
        
        # 4. Check options for TP/SL
        for opt_pos in option_positions:
            entry = opt_pos.get('entry_price', 0)
            current_opt = opt_pos.get('current_price', entry)
            opt_pnl_pct = ((current_opt - entry) / entry * 100) if entry > 0 else 0
            
            if opt_pnl_pct >= config['tp_options']:
                analysis['action'] = "TAKE_PROFIT_OPTIONS"
                analysis['reason'] = f"Options up {opt_pnl_pct:.1f}% - closing"
                analysis['position_id'] = opt_pos.get('id')
                return analysis
            
            if opt_pnl_pct <= config['sl_options']:
                analysis['action'] = "STOP_LOSS_OPTIONS"
                analysis['reason'] = f"Options down {opt_pnl_pct:.1f}% - cutting"
                analysis['position_id'] = opt_pos.get('id')
                return analysis
        
        # 5. Check for new options opportunity
        if meta_signal and meta_signal.get('action') == 'STRONG_BUY':
            if meta_signal.get('confidence', 0) >= 70:
                analysis['action'] = "BUY_CALLS"
                analysis['reason'] = f"Meta STRONG_BUY {meta_signal.get('confidence')}% - buying calls"
                return analysis
        
        return analysis
    
    def run_daily_check(self) -> Dict:
        """Run daily trading check for all BTC miners"""
        logger.info("=" * 60)
        logger.info("🏭 BTC MINERS DAILY TRADING CHECK")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbols": {},
            "actions_taken": []
        }
        
        for symbol in BTC_MINERS.keys():
            logger.info(f"\n📊 Analyzing {symbol}...")
            
            analysis = self.analyze_symbol(symbol)
            results['symbols'][symbol] = analysis
            
            # Log the decision
            self._log_decision({
                "symbol": symbol,
                "action": analysis['action'],
                "reason": analysis['reason'],
                "price": analysis['current_price'],
                "meta_signal": analysis['meta_signal'],
                "meta_confidence": analysis['meta_confidence']
            })
            
            # Execute actions
            if analysis['action'] == "ACCUMULATE":
                # Buy 50 shares
                if self.execute_paper_trade(symbol, "LONG", 50):
                    results['actions_taken'].append(f"BOUGHT 50 {symbol} shares")
            
            elif analysis['action'] == "DCA_SHARES":
                # DCA with 25 shares
                if self.execute_paper_trade(symbol, "LONG", 25):
                    results['actions_taken'].append(f"DCA 25 {symbol} shares")
            
            elif analysis['action'] == "BUY_CALLS":
                # Buy 2 ATM calls, expiring in ~4 weeks
                expiry = (datetime.now() + timedelta(days=28)).strftime("%Y-%m-%d")
                strike = round(analysis['current_price'] / 5) * 5  # Round to nearest $5
                if self.execute_paper_trade(symbol, "BUY_CALL", 2, 
                                           is_option=True, strike=strike, 
                                           expiry=expiry, option_type='call'):
                    results['actions_taken'].append(f"BOUGHT 2 {symbol} ${strike} calls exp {expiry}")
            
            elif analysis['action'] in ["TAKE_PROFIT_OPTIONS", "STOP_LOSS_OPTIONS"]:
                pos_id = analysis.get('position_id')
                if pos_id and self.close_paper_position(pos_id, analysis['reason']):
                    results['actions_taken'].append(f"CLOSED {symbol} options: {analysis['reason']}")
        
        # Save state
        self._save_state()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 DAILY CHECK SUMMARY")
        logger.info("=" * 60)
        
        for symbol, data in results['symbols'].items():
            logger.info(f"{symbol}: ${data['current_price']:.2f} | {data['total_shares']} shares | "
                       f"Meta: {data['meta_signal']} ({data['meta_confidence']}%) | "
                       f"Action: {data['action']}")
        
        if results['actions_taken']:
            logger.info(f"\n🎯 Actions Taken: {len(results['actions_taken'])}")
            for action in results['actions_taken']:
                logger.info(f"  ✅ {action}")
        else:
            logger.info("\n😴 No actions needed")
        
        return results


def main():
    """Run the BTC Miners trader"""
    trader = BTCMinersTrader()
    results = trader.run_daily_check()
    
    # Print JSON summary
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
