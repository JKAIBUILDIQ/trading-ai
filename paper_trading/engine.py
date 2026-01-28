"""
Paper Trading Engine - Uses ONLY real market data

NO placeholders, NO fake data, NO simulated prices
All prices from Yahoo Finance (stocks/ETFs/options) and CoinGecko (crypto)
"""
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State file for persistence
STATE_FILE = Path("/home/jbot/trading_ai/data/paper_trading_state.json")


class PaperTradingEngine:
    """
    Paper trading with 100% real market data
    NO placeholders, NO fake data
    """
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Dict] = []
        self.closed_positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self._load_state()
        
    def _load_state(self):
        """Load persisted state if available"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                    self.balance = state.get('balance', self.initial_balance)
                    self.positions = state.get('positions', [])
                    self.closed_positions = state.get('closed_positions', [])
                    self.trade_history = state.get('trade_history', [])
                    logger.info(f"Loaded state: {len(self.positions)} open, {len(self.closed_positions)} closed")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Persist state to disk"""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump({
                    'balance': self.balance,
                    'positions': self.positions,
                    'closed_positions': self.closed_positions,
                    'trade_history': self.trade_history[-100:]  # Keep last 100
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        
    def get_real_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get REAL current price - NO PLACEHOLDERS
        
        Supported symbols:
        - BTC/BTCUSD: CoinGecko
        - ETH/ETHUSD: CoinGecko
        - IREN, GLD, SPY, etc: Yahoo Finance
        
        Raises ValueError if data unavailable
        """
        symbol = symbol.upper()
        
        # Crypto via CoinGecko
        if symbol in ['BTC', 'BTCUSD', 'BITCOIN']:
            return self._get_crypto_price('bitcoin', 'BTC')
        elif symbol in ['ETH', 'ETHUSD', 'ETHEREUM']:
            return self._get_crypto_price('ethereum', 'ETH')
        elif symbol in ['SOL', 'SOLUSD', 'SOLANA']:
            return self._get_crypto_price('solana', 'SOL')
        else:
            # Stocks/ETFs via Yahoo Finance
            return self._get_stock_price(symbol)
    
    def _get_crypto_price(self, coin_id: str, symbol: str) -> Dict[str, Any]:
        """Get REAL crypto price from CoinGecko"""
        try:
            url = 'https://api.coingecko.com/api/v3/simple/price'
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id not in data:
                raise ValueError(f"No data for {coin_id}")
            
            return {
                'symbol': symbol,
                'price': float(data[coin_id]['usd']),
                'change_24h': float(data[coin_id].get('usd_24h_change', 0)),
                'volume_24h': float(data[coin_id].get('usd_24h_vol', 0)),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'source': 'coingecko',
                'is_real': True
            }
        except Exception as e:
            logger.error(f"CoinGecko error for {symbol}: {e}")
            raise ValueError(f"Cannot get real price for {symbol}: {e}")
    
    def _get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get REAL stock/ETF price from Yahoo Finance"""
        # Symbol mapping for forex/commodities
        SYMBOL_MAP = {
            'XAUUSD': 'GC=F',      # Gold futures
            'GOLD': 'GC=F',
            'XAGUSD': 'SI=F',      # Silver futures
            'SILVER': 'SI=F',
            'EURUSD': 'EURUSD=X',  # Euro/USD
            'GBPUSD': 'GBPUSD=X',  # British Pound/USD
            'USDJPY': 'JPY=X',     # USD/Japanese Yen
            'OIL': 'CL=F',         # Crude Oil
            'WTICOUSD': 'CL=F',
        }
        
        # Map symbol if needed
        yahoo_symbol = SYMBOL_MAP.get(symbol.upper(), symbol)
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get current data
            hist = ticker.history(period='2d')
            
            if hist.empty:
                raise ValueError(f"No data available for {symbol}")
            
            current_price = float(hist['Close'].iloc[-1])
            
            # Calculate change
            if len(hist) >= 2:
                prev_close = float(hist['Close'].iloc[-2])
            else:
                prev_close = current_price
            
            change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0
            volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'change_24h': change_pct,
                'volume': volume,
                'prev_close': prev_close,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'source': 'yahoo_finance',
                'is_real': True
            }
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise ValueError(f"Cannot get real price for {symbol}: {e}")
    
    def get_real_option_price(self, symbol: str, strike: float, 
                               expiry: str, option_type: str = 'call') -> Dict[str, Any]:
        """
        Get REAL option price from Yahoo Finance
        NO PLACEHOLDERS
        
        Args:
            symbol: Underlying symbol (e.g., 'IREN')
            strike: Strike price
            expiry: Expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options available for {symbol}")
            
            # Find closest expiration to requested
            target_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            closest_exp = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - target_date))
            
            # Get options chain
            options = ticker.option_chain(closest_exp)
            chain = options.calls if option_type.lower() == 'call' else options.puts
            
            if chain.empty:
                raise ValueError(f"No {option_type}s available for {symbol}")
            
            # Find closest strike
            chain = chain.copy()
            chain['strike_diff'] = abs(chain['strike'] - strike)
            closest_row = chain.loc[chain['strike_diff'].idxmin()]
            
            bid = float(closest_row['bid']) if closest_row['bid'] else 0
            ask = float(closest_row['ask']) if closest_row['ask'] else 0
            last = float(closest_row['lastPrice']) if closest_row['lastPrice'] else 0
            
            # Use midpoint if both bid/ask available, else last price
            mid_price = (bid + ask) / 2 if bid and ask else last
            
            return {
                'symbol': f"{symbol}_{closest_row['strike']}_{option_type[0].upper()}_{closest_exp}",
                'underlying': symbol,
                'strike': float(closest_row['strike']),
                'type': option_type,
                'expiry': closest_exp,
                'bid': bid,
                'ask': ask,
                'last': last,
                'mid': mid_price,
                'volume': int(closest_row['volume']) if closest_row['volume'] and not pd_isna(closest_row['volume']) else 0,
                'open_interest': int(closest_row['openInterest']) if closest_row['openInterest'] and not pd_isna(closest_row['openInterest']) else 0,
                'iv': float(closest_row['impliedVolatility']) if closest_row['impliedVolatility'] and not pd_isna(closest_row['impliedVolatility']) else 0,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'source': 'yahoo_finance',
                'is_real': True
            }
        except Exception as e:
            logger.error(f"Options error for {symbol}: {e}")
            raise ValueError(f"Cannot get option price for {symbol} {strike} {option_type}: {e}")
    
    def open_position(self, signal: Dict) -> Dict:
        """
        Open a paper position based on signal
        Uses REAL current prices at time of entry
        
        Signal format:
        {
            'symbol': 'IREN',
            'type': 'LONG' | 'SHORT' | 'BUY_CALL' | 'BUY_PUT',
            'size': 100,  # Shares or contracts
            'is_option': False,
            'strike': 55,  # If option
            'expiry': '2026-02-14',  # If option
            'option_type': 'call',  # If option
            'stop_loss': 50.00,  # Optional
            'take_profit': 70.00,  # Optional
            'confidence': 75,
            'source': 'NEO'
        }
        """
        logger.info(f"Opening position: {signal}")
        
        # Get real price at time of entry
        is_option = signal.get('is_option', False)
        
        if is_option:
            price_data = self.get_real_option_price(
                signal['symbol'],
                signal['strike'],
                signal['expiry'],
                signal.get('option_type', 'call')
            )
            # Buy at ask (realistic execution)
            entry_price = price_data['ask'] if price_data['ask'] > 0 else price_data['last']
            actual_strike = price_data['strike']
            actual_expiry = price_data['expiry']
        else:
            price_data = self.get_real_price(signal['symbol'])
            entry_price = price_data['price']
            actual_strike = None
            actual_expiry = None
        
        if entry_price <= 0:
            raise ValueError(f"Invalid entry price: {entry_price}")
        
        position = {
            'id': len(self.positions) + len(self.closed_positions) + 1,
            'symbol': signal['symbol'],
            'type': signal['type'],
            'size': signal['size'],
            'entry_price': entry_price,
            'entry_time': datetime.utcnow().isoformat() + 'Z',
            'current_price': entry_price,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'status': 'OPEN',
            'signal_source': signal.get('source', 'MANUAL'),
            'signal_confidence': signal.get('confidence', 0),
            'is_option': is_option,
            'strike': actual_strike,
            'expiry': actual_expiry,
            'option_type': signal.get('option_type'),
            'price_source': price_data.get('source', 'unknown'),
            'notes': signal.get('notes', '')
        }
        
        # Calculate position cost
        multiplier = 100 if is_option else 1
        position_cost = entry_price * signal['size'] * multiplier
        
        if position_cost > self.balance:
            logger.warning(f"Insufficient balance: {self.balance} < {position_cost}")
            # Allow it for paper trading but log warning
        
        self.positions.append(position)
        self.trade_history.append({
            'action': 'OPEN',
            'position': position.copy(),
            'timestamp': position['entry_time'],
            'balance_before': self.balance
        })
        
        self._save_state()
        logger.info(f"Opened position {position['id']}: {signal['symbol']} @ ${entry_price}")
        
        return position
    
    def update_positions(self) -> List[Dict]:
        """
        Update all open positions with REAL current prices
        Checks stop loss / take profit
        """
        for position in self.positions:
            if position['status'] != 'OPEN':
                continue
                
            try:
                if position['is_option']:
                    price_data = self.get_real_option_price(
                        position['symbol'],
                        position['strike'],
                        position['expiry'],
                        position['option_type']
                    )
                    # Value at bid (exit price)
                    current_price = price_data['bid'] if price_data['bid'] > 0 else price_data['last']
                else:
                    price_data = self.get_real_price(position['symbol'])
                    current_price = price_data['price']
                
                position['current_price'] = current_price
                position['last_update'] = datetime.utcnow().isoformat() + 'Z'
                
                # Calculate P&L
                if position['type'] in ['LONG', 'BUY_CALL', 'BUY_PUT']:
                    pnl_per_unit = current_price - position['entry_price']
                else:  # SHORT
                    pnl_per_unit = position['entry_price'] - current_price
                
                # For options, multiply by 100 (contract size)
                multiplier = 100 if position['is_option'] else 1
                position['pnl'] = pnl_per_unit * position['size'] * multiplier
                position['pnl_percent'] = (pnl_per_unit / position['entry_price']) * 100 if position['entry_price'] else 0
                
                # Check stop loss / take profit for underlying price
                underlying_price = price_data['price'] if not position['is_option'] else current_price
                
                # For options, we check P&L percentage instead of absolute price
                if position['is_option']:
                    # Auto-close at -50% or +100%
                    if position['pnl_percent'] <= -50:
                        self.close_position(position['id'], 'STOP_LOSS_50PCT')
                    elif position['pnl_percent'] >= 100:
                        self.close_position(position['id'], 'TAKE_PROFIT_100PCT')
                else:
                    if position['stop_loss'] and current_price <= position['stop_loss']:
                        self.close_position(position['id'], 'STOP_LOSS')
                    elif position['take_profit'] and current_price >= position['take_profit']:
                        self.close_position(position['id'], 'TAKE_PROFIT')
                        
            except Exception as e:
                position['error'] = str(e)
                position['last_error_time'] = datetime.utcnow().isoformat() + 'Z'
                logger.error(f"Error updating position {position['id']}: {e}")
        
        self._save_state()
        return self.positions
    
    def close_position(self, position_id: int, reason: str = 'MANUAL') -> Dict:
        """
        Close a position and record the result
        Uses REAL exit price
        """
        position = next((p for p in self.positions if p['id'] == position_id), None)
        
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        # Get final price
        try:
            if position['is_option']:
                price_data = self.get_real_option_price(
                    position['symbol'],
                    position['strike'],
                    position['expiry'],
                    position['option_type']
                )
                exit_price = price_data['bid'] if price_data['bid'] > 0 else price_data['last']
            else:
                price_data = self.get_real_price(position['symbol'])
                exit_price = price_data['price']
        except Exception as e:
            # Use last known price if we can't get current
            logger.error(f"Failed to get exit price: {e}, using last known")
            exit_price = position['current_price']
        
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.utcnow().isoformat() + 'Z'
        position['status'] = 'CLOSED'
        position['close_reason'] = reason
        
        # Final P&L calculation
        if position['type'] in ['LONG', 'BUY_CALL', 'BUY_PUT']:
            pnl_per_unit = exit_price - position['entry_price']
        else:  # SHORT
            pnl_per_unit = position['entry_price'] - exit_price
            
        multiplier = 100 if position['is_option'] else 1
        position['pnl'] = pnl_per_unit * position['size'] * multiplier
        position['pnl_percent'] = (pnl_per_unit / position['entry_price']) * 100 if position['entry_price'] else 0
        
        # Move to closed
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        # Update balance
        self.balance += position['pnl']
        
        self.trade_history.append({
            'action': 'CLOSE',
            'position': position.copy(),
            'timestamp': position['exit_time'],
            'reason': reason,
            'balance_after': self.balance
        })
        
        self._save_state()
        logger.info(f"Closed position {position_id}: P&L ${position['pnl']:.2f} ({position['pnl_percent']:.1f}%)")
        
        return position
    
    def get_stats(self) -> Dict:
        """
        Get paper trading statistics
        """
        all_closed = self.closed_positions
        
        if not all_closed:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'balance': self.balance,
                'initial_balance': self.initial_balance,
                'return_pct': 0,
                'open_positions': len(self.positions),
                'open_pnl': sum(p.get('pnl', 0) for p in self.positions)
            }
        
        wins = [p for p in all_closed if p.get('pnl', 0) > 0]
        losses = [p for p in all_closed if p.get('pnl', 0) <= 0]
        total_pnl = sum(p.get('pnl', 0) for p in all_closed)
        
        return {
            'total_trades': len(all_closed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(all_closed)) * 100 if all_closed else 0,
            'total_pnl': total_pnl,
            'avg_win': sum(p['pnl'] for p in wins) / len(wins) if wins else 0,
            'avg_loss': sum(p['pnl'] for p in losses) / len(losses) if losses else 0,
            'largest_win': max((p['pnl'] for p in wins), default=0),
            'largest_loss': min((p['pnl'] for p in losses), default=0),
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'return_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'open_positions': len(self.positions),
            'open_pnl': sum(p.get('pnl', 0) for p in self.positions)
        }
    
    def reset(self, initial_balance: float = 100000):
        """Reset paper trading account"""
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = []
        self.closed_positions = []
        self.trade_history = []
        self._save_state()
        logger.info(f"Paper trading reset with balance ${initial_balance}")
    
    def to_json(self) -> str:
        """Export state to JSON"""
        return json.dumps({
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'positions': self.positions,
            'closed_positions': self.closed_positions,
            'stats': self.get_stats()
        }, indent=2)


def pd_isna(val) -> bool:
    """Check if value is NaN/None (without importing pandas)"""
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


# Singleton instance
_engine_instance = None

def get_engine() -> PaperTradingEngine:
    """Get singleton engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PaperTradingEngine()
    return _engine_instance
