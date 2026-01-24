"""
NEO Paper Trading Integration

Automatically opens paper positions based on NEO signals
All trades use REAL market data - no placeholders
"""
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'neo'))

from paper_trading.engine import get_engine, PaperTradingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEOPaperTrader:
    """
    Connects NEO signals to paper trading
    
    Supports:
    - IREN options signals (from Paul's IREN system)
    - BTC signals (correlation plays)
    - Gold signals (from NEO Gold)
    """
    
    def __init__(self):
        self.engine = get_engine()
        self.min_confidence = 70  # Only trade signals >= 70% confidence
        self.default_option_size = 50  # 50 contracts
        self.default_stock_size = 100  # 100 shares
        self.default_crypto_size = 0.1  # 0.1 BTC
        
    def process_iren_signal(self, signal: dict) -> dict:
        """
        Process IREN signal from Paul's system
        
        Expected signal format:
        {
            'signal': 'BUY' | 'SELL',
            'confidence': 75,
            'strike': 55,
            'expiry': '2026-02-21',
            'type': 'call' | 'put',
            'reason': 'BTC bullish + technical breakout'
        }
        """
        if signal.get('confidence', 0) < self.min_confidence:
            logger.info(f"IREN signal skipped: confidence {signal.get('confidence')}% < {self.min_confidence}%")
            return {'skipped': True, 'reason': 'Low confidence'}
        
        # Determine option type based on signal
        if signal.get('signal') == 'BUY':
            position_type = 'BUY_CALL'
            option_type = signal.get('type', 'call')
        else:
            position_type = 'BUY_PUT'
            option_type = signal.get('type', 'put')
        
        # Default expiry to 3 weeks out if not specified
        expiry = signal.get('expiry')
        if not expiry:
            from datetime import timedelta
            expiry = (datetime.now() + timedelta(weeks=3)).strftime('%Y-%m-%d')
        
        # Open position with REAL prices
        try:
            position = self.engine.open_position({
                'symbol': 'IREN',
                'type': position_type,
                'size': signal.get('size', self.default_option_size),
                'is_option': True,
                'strike': signal.get('strike', 55),
                'expiry': expiry,
                'option_type': option_type,
                'confidence': signal.get('confidence', 0),
                'source': 'NEO-IREN',
                'notes': signal.get('reason', '')
            })
            
            logger.info(f"Opened IREN position: {position_type} ${signal.get('strike')} @ ${position['entry_price']:.2f}")
            return {'success': True, 'position': position}
            
        except Exception as e:
            logger.error(f"Failed to open IREN position: {e}")
            return {'error': str(e)}
    
    def process_btc_signal(self, signal: dict) -> dict:
        """
        Process BTC signal
        
        Expected signal format:
        {
            'signal': 'BUY' | 'SELL',
            'confidence': 80,
            'entry': 95000,
            'stop_loss': 92000,
            'targets': [{'price': 100000}, {'price': 105000}]
        }
        """
        if signal.get('confidence', 0) < self.min_confidence:
            logger.info(f"BTC signal skipped: confidence {signal.get('confidence')}% < {self.min_confidence}%")
            return {'skipped': True, 'reason': 'Low confidence'}
        
        position_type = 'LONG' if signal.get('signal') == 'BUY' else 'SHORT'
        
        # Get first target as take profit
        targets = signal.get('targets', [])
        take_profit = targets[0].get('price') if targets else None
        
        try:
            position = self.engine.open_position({
                'symbol': 'BTC',
                'type': position_type,
                'size': signal.get('size', self.default_crypto_size),
                'is_option': False,
                'stop_loss': signal.get('stop_loss'),
                'take_profit': take_profit,
                'confidence': signal.get('confidence', 0),
                'source': 'NEO-BTC',
                'notes': signal.get('reason', '')
            })
            
            logger.info(f"Opened BTC position: {position_type} @ ${position['entry_price']:,.2f}")
            return {'success': True, 'position': position}
            
        except Exception as e:
            logger.error(f"Failed to open BTC position: {e}")
            return {'error': str(e)}
    
    def process_gold_signal(self, signal: dict) -> dict:
        """
        Process Gold signal from NEO-GOLD
        
        We use GLD ETF for paper trading (easier than futures)
        
        Expected signal format:
        {
            'signal': 'BUY' | 'SELL',
            'confidence': 75,
            'entry': 2800,
            'stop_loss': 2750,
            'take_profit': 2900
        }
        """
        if signal.get('confidence', 0) < self.min_confidence:
            logger.info(f"Gold signal skipped: confidence {signal.get('confidence')}% < {self.min_confidence}%")
            return {'skipped': True, 'reason': 'Low confidence'}
        
        position_type = 'LONG' if signal.get('signal') == 'BUY' else 'SHORT'
        
        try:
            position = self.engine.open_position({
                'symbol': 'GLD',  # Gold ETF
                'type': position_type,
                'size': signal.get('size', self.default_stock_size),
                'is_option': False,
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence', 0),
                'source': 'NEO-GOLD',
                'notes': signal.get('reason', '')
            })
            
            logger.info(f"Opened GLD position: {position_type} @ ${position['entry_price']:.2f}")
            return {'success': True, 'position': position}
            
        except Exception as e:
            logger.error(f"Failed to open GLD position: {e}")
            return {'error': str(e)}
    
    def process_any_signal(self, signal: dict) -> dict:
        """
        Generic signal processor - detects asset type and routes appropriately
        
        Expected signal format:
        {
            'asset': 'IREN' | 'BTC' | 'GLD' | etc.,
            'signal': 'BUY' | 'SELL',
            'confidence': 75,
            ...
        }
        """
        asset = signal.get('asset', signal.get('symbol', '')).upper()
        
        if asset == 'IREN':
            return self.process_iren_signal(signal)
        elif asset in ['BTC', 'BTCUSD', 'BITCOIN']:
            return self.process_btc_signal(signal)
        elif asset in ['GOLD', 'GLD', 'XAUUSD']:
            return self.process_gold_signal(signal)
        else:
            # Generic stock/ETF
            return self._process_generic_signal(signal, asset)
    
    def _process_generic_signal(self, signal: dict, symbol: str) -> dict:
        """Process generic stock/ETF signal"""
        if signal.get('confidence', 0) < self.min_confidence:
            return {'skipped': True, 'reason': 'Low confidence'}
        
        position_type = 'LONG' if signal.get('signal') == 'BUY' else 'SHORT'
        
        try:
            position = self.engine.open_position({
                'symbol': symbol,
                'type': position_type,
                'size': signal.get('size', self.default_stock_size),
                'is_option': False,
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence', 0),
                'source': signal.get('source', 'NEO'),
                'notes': signal.get('reason', '')
            })
            
            logger.info(f"Opened {symbol} position: {position_type} @ ${position['entry_price']:.2f}")
            return {'success': True, 'position': position}
            
        except Exception as e:
            logger.error(f"Failed to open {symbol} position: {e}")
            return {'error': str(e)}
    
    def get_open_positions(self) -> list:
        """Get all open paper positions"""
        return self.engine.positions
    
    def get_stats(self) -> dict:
        """Get paper trading statistics"""
        return self.engine.get_stats()
    
    def update_all(self):
        """Update all positions with current prices"""
        return self.engine.update_positions()


# Singleton instance
_trader_instance = None

def get_neo_paper_trader() -> NEOPaperTrader:
    """Get singleton NEO paper trader instance"""
    global _trader_instance
    if _trader_instance is None:
        _trader_instance = NEOPaperTrader()
    return _trader_instance


# API endpoints to add to FastAPI
def add_neo_routes(router):
    """Add NEO paper trading routes to router"""
    
    @router.post("/paper-trading/neo/signal")
    async def process_neo_signal(signal: dict):
        """Process incoming NEO signal"""
        trader = get_neo_paper_trader()
        return trader.process_any_signal(signal)
    
    @router.post("/paper-trading/neo/iren")
    async def process_iren_signal(signal: dict):
        """Process IREN-specific signal"""
        trader = get_neo_paper_trader()
        return trader.process_iren_signal(signal)
    
    @router.post("/paper-trading/neo/btc")
    async def process_btc_signal(signal: dict):
        """Process BTC-specific signal"""
        trader = get_neo_paper_trader()
        return trader.process_btc_signal(signal)
    
    @router.post("/paper-trading/neo/gold")
    async def process_gold_signal(signal: dict):
        """Process Gold-specific signal"""
        trader = get_neo_paper_trader()
        return trader.process_gold_signal(signal)


if __name__ == "__main__":
    # Test the integration
    trader = get_neo_paper_trader()
    
    # Test IREN signal
    result = trader.process_iren_signal({
        'signal': 'BUY',
        'confidence': 75,
        'strike': 55,
        'expiry': '2026-02-21',
        'type': 'call',
        'reason': 'Test signal'
    })
    print(f"IREN result: {result}")
    
    # Show stats
    print(f"\nStats: {trader.get_stats()}")
