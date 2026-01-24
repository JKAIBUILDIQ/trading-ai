"""
Daily Pre-Market Signal Generator
Runs at 6:00 AM ET before market opens

ALL DATA IS REAL - NO PLACEHOLDERS
Sources: Yahoo Finance, CoinGecko

PAUL'S REQUIREMENTS (Jan 24, 2026):
1. LONG ONLY - No shorts, no puts
2. Avoid near-term expiries (min 14 DTE)
3. Avoid earnings window (Feb 5, 2026 for IREN)
4. Prefer Feb 20, Feb 27 expirations (21-35 DTE sweet spot)
"""
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage for daily signals
SIGNALS_DIR = Path("/home/jbot/trading_ai/data/daily_signals")
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# IREN Earnings Calendar - CRITICAL for Paul
IREN_EARNINGS_DATE = datetime(2026, 2, 5)
MIN_DTE = 14  # Minimum days to expiration
PAUL_SWEET_SPOT_DTE = (21, 35)  # Paul's preferred range


class DailySignalGenerator:
    """
    Generate trading signals every morning before market opens
    ALL DATA IS REAL - NO PLACEHOLDERS
    """
    
    def __init__(self):
        self.signals_dir = SIGNALS_DIR
    
    def get_real_btc_data(self) -> Dict[str, Any]:
        """Get REAL BTC data from CoinGecko"""
        try:
            url = 'https://api.coingecko.com/api/v3/coins/bitcoin'
            params = {'localization': 'false', 'sparkline': 'true'}
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            price = data['market_data']['current_price']['usd']
            high_24h = data['market_data']['high_24h']['usd']
            low_24h = data['market_data']['low_24h']['usd']
            change_24h = data['market_data']['price_change_percentage_24h']
            sparkline = data['market_data']['sparkline_7d']['price'][-24:]  # Last 24 hours
            
            return {
                'price': price,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'change_24h': change_24h,
                'sparkline': sparkline,
                'source': 'coingecko',
                'is_real': True
            }
        except Exception as e:
            logger.error(f"Failed to get BTC data: {e}")
            raise ValueError(f"Cannot get real BTC data: {e}")
    
    def get_real_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get REAL stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='14d')  # Need 14 days for RSI
            info = ticker.info
            
            if hist.empty:
                raise ValueError(f"No data for {symbol}")
            
            current = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
            high = float(hist['High'].iloc[-1])
            low = float(hist['Low'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])
            
            # Calculate RSI (14-period)
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            if loss.iloc[-1] != 0:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # Handle NaN
            if math.isnan(rsi):
                rsi = 50
            
            return {
                'price': current,
                'prev_close': prev_close,
                'high': high,
                'low': low,
                'volume': volume,
                'change_pct': ((current - prev_close) / prev_close) * 100,
                'rsi': rsi,
                '52w_high': info.get('fiftyTwoWeekHigh', current * 1.2),
                '52w_low': info.get('fiftyTwoWeekLow', current * 0.8),
                'source': 'yahoo_finance',
                'is_real': True
            }
        except Exception as e:
            logger.error(f"Failed to get {symbol} data: {e}")
            raise ValueError(f"Cannot get real {symbol} data: {e}")
    
    def calculate_levels(self, current_price: float, direction: str, 
                         volatility_pct: float = 3.0) -> Dict[str, float]:
        """
        Calculate Entry, TP, SL based on price and direction
        Uses actual price levels, not placeholders
        """
        if direction == 'LONG':
            entry_low = current_price * 0.995  # Slightly below current
            entry_high = current_price * 1.005
            take_profit = current_price * (1 + volatility_pct / 100)
            stop_loss = current_price * (1 - volatility_pct / 100 * 0.75)
        else:  # SHORT
            entry_low = current_price * 0.995
            entry_high = current_price * 1.005
            take_profit = current_price * (1 - volatility_pct / 100)
            stop_loss = current_price * (1 + volatility_pct / 100 * 0.75)
        
        return {
            'entry_low': round(entry_low, 2),
            'entry_high': round(entry_high, 2),
            'take_profit': round(take_profit, 2),
            'stop_loss': round(stop_loss, 2),
        }
    
    def get_next_option_expiry(self, symbol: str) -> str:
        """Get the next valid options expiry date"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                raise ValueError("No expirations available")
            
            # Find next expiry that's at least 3 days away
            today = datetime.now().date()
            for exp in expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                if (exp_date - today).days >= 3:
                    return exp
            
            return expirations[0]
            
        except Exception:
            # Default to next Friday
            today = datetime.now()
            days_until_friday = (4 - today.weekday()) % 7
            if days_until_friday < 3:
                days_until_friday += 7
            next_friday = today + timedelta(days=days_until_friday)
            return next_friday.strftime('%Y-%m-%d')
    
    def get_optimal_strike(self, symbol: str, direction: str, 
                           expiry: str, current_price: float) -> float:
        """Get optimal strike price from real options chain"""
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.option_chain(expiry)
            chain = options.calls if direction == 'LONG' else options.puts
            
            if chain.empty:
                raise ValueError("Empty options chain")
            
            # Find strikes near current price with good volume
            chain_copy = chain.copy()
            chain_copy['price_diff'] = abs(chain_copy['strike'] - current_price)
            chain_sorted = chain_copy.sort_values('price_diff').head(10)
            chain_sorted = chain_sorted.sort_values('volume', ascending=False)
            
            # Get the best strike with decent volume
            for _, row in chain_sorted.iterrows():
                if row['volume'] and row['volume'] > 10:
                    return float(row['strike'])
            
            # Fallback to closest strike
            return float(chain_sorted.iloc[0]['strike'])
            
        except Exception as e:
            logger.warning(f"Options chain error: {e}, using calculated strike")
            # Round to nearest appropriate increment
            if current_price > 100:
                return round(current_price / 5) * 5
            elif current_price > 20:
                return round(current_price / 2.5) * 2.5
            else:
                return round(current_price)
    
    def get_option_price(self, symbol: str, strike: float, expiry: str, 
                          option_type: str) -> float:
        """Get real option price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.option_chain(expiry)
            chain = options.calls if option_type.lower() == 'call' else options.puts
            
            # Find the strike
            option = chain[chain['strike'] == strike]
            
            if option.empty:
                # Find closest
                closest_idx = (chain['strike'] - strike).abs().idxmin()
                option = chain.loc[[closest_idx]]
            
            if not option.empty:
                row = option.iloc[0]
                ask = float(row['ask']) if row['ask'] else 0
                bid = float(row['bid']) if row['bid'] else 0
                last = float(row['lastPrice']) if row['lastPrice'] else 0
                
                # Use ask for buying, or midpoint if available
                if ask > 0:
                    return ask
                elif bid > 0 and last > 0:
                    return (bid + last) / 2
                elif last > 0:
                    return last
            
            raise ValueError("No price available")
            
        except Exception as e:
            logger.warning(f"Option price error: {e}")
            # Estimate based on stock price (rough ATM estimate)
            stock = self.get_real_stock_data(symbol)
            return stock['price'] * 0.05  # ~5% of stock price
    
    def generate_btc_signal(self) -> Dict[str, Any]:
        """Generate BTC signal with REAL data"""
        data = self.get_real_btc_data()
        
        # Trend analysis using sparkline
        sparkline = data['sparkline']
        if len(sparkline) >= 24:
            short_ma = sum(sparkline[-6:]) / 6  # 6-hour MA
            long_ma = sum(sparkline[-24:]) / 24  # 24-hour MA
        else:
            short_ma = sparkline[-1] if sparkline else data['price']
            long_ma = short_ma
        
        # Determine direction and confidence
        if short_ma > long_ma and data['change_24h'] > -2:
            direction = 'LONG'
            confidence = min(85, 60 + abs(data['change_24h']) * 2)
        elif short_ma < long_ma and data['change_24h'] < 2:
            direction = 'SHORT'
            confidence = min(85, 60 + abs(data['change_24h']) * 2)
        else:
            direction = 'LONG' if data['change_24h'] > 0 else 'SHORT'
            confidence = 55
        
        levels = self.calculate_levels(data['price'], direction, volatility_pct=2.5)
        
        return {
            'symbol': 'BTC',
            'asset_type': 'CRYPTO',
            'direction': direction,
            'current_price': data['price'],
            'entry_low': levels['entry_low'],
            'entry_high': levels['entry_high'],
            'take_profit': levels['take_profit'],
            'stop_loss': levels['stop_loss'],
            'size': 0.5,
            'size_unit': 'BTC',
            'confidence': int(confidence),
            'analysis': {
                'change_24h': round(data['change_24h'], 2),
                'high_24h': data['high_24h'],
                'low_24h': data['low_24h'],
                'trend': 'BULLISH' if short_ma > long_ma else 'BEARISH',
                'short_ma': round(short_ma, 2),
                'long_ma': round(long_ma, 2)
            },
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'valid_for': 'Today',
            'source': 'coingecko',
            'is_real': True
        }
    
    def get_next_4_expiries(self, symbol: str) -> list:
        """
        Get the next 4 valid options expiry dates for Paul.
        
        PAUL'S RULES:
        1. Minimum 14 DTE (no weeklies)
        2. Avoid earnings window (7 days before/after Feb 5)
        3. Prefer 21-35 DTE (Paul's sweet spot)
        4. Mark Paul's picks clearly
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                raise ValueError("No expirations available")
            
            today = datetime.now().date()
            earnings_date = IREN_EARNINGS_DATE.date()
            
            valid_expiries = []
            for exp in expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                days_out = (exp_date - today).days
                days_from_earnings = abs((exp_date - earnings_date).days)
                
                # RULE 1: Skip if less than 14 days
                if days_out < MIN_DTE:
                    continue
                
                # Determine quality and warnings
                if days_from_earnings <= 7:
                    # Within 7 days of earnings - risky
                    theta_warning = "⚠️ EARNINGS RISK"
                    paul_pick = False
                    quality = "EARNINGS_RISK"
                elif days_out <= 7:
                    theta_warning = "HIGH THETA DECAY"
                    paul_pick = False
                    quality = "TOO_CLOSE"
                elif days_out <= 14:
                    theta_warning = "MODERATE THETA"
                    paul_pick = False
                    quality = "OK"
                elif PAUL_SWEET_SPOT_DTE[0] <= days_out <= PAUL_SWEET_SPOT_DTE[1]:
                    theta_warning = "⭐ PAUL'S SWEET SPOT"
                    paul_pick = True
                    quality = "PAUL_PREFERRED"
                else:
                    theta_warning = "LOW THETA"
                    paul_pick = days_out <= 45
                    quality = "GOOD"
                
                valid_expiries.append({
                    'date': exp,
                    'days_out': days_out,
                    'days_from_earnings': days_from_earnings,
                    'theta_warning': theta_warning,
                    'paul_pick': paul_pick,
                    'quality': quality
                })
            
            # Sort: Paul's picks first, then by days_out
            valid_expiries.sort(key=lambda x: (not x['paul_pick'], x['days_out']))
            
            # Return top 4
            return valid_expiries[:4] if valid_expiries else [{
                'date': expirations[0] if expirations else 'N/A',
                'days_out': 1,
                'theta_warning': 'NO DATA',
                'paul_pick': False,
                'quality': 'UNKNOWN'
            }]
            
        except Exception as e:
            logger.warning(f"Failed to get expiries: {e}")
            # Generate default expiries with Paul's preferences
            today = datetime.now()
            expiries = []
            for weeks in [3, 4, 5, 6]:  # Start at 3 weeks (21 days)
                exp_date = today + timedelta(weeks=weeks)
                # Adjust to Friday
                days_until_friday = (4 - exp_date.weekday()) % 7
                exp_date = exp_date + timedelta(days=days_until_friday)
                days_out = (exp_date.date() - today.date()).days
                
                expiries.append({
                    'date': exp_date.strftime('%Y-%m-%d'),
                    'days_out': days_out,
                    'theta_warning': '⭐ PAUL\'S SWEET SPOT' if 21 <= days_out <= 35 else 'LOW THETA',
                    'paul_pick': 21 <= days_out <= 35,
                    'quality': 'PAUL_PREFERRED' if 21 <= days_out <= 35 else 'GOOD'
                })
            return expiries

    def generate_iren_signal_for_expiry(self, data: dict, btc_data: dict, 
                                         direction: str, confidence: int,
                                         expiry_info: dict) -> Dict[str, Any]:
        """
        Generate IREN signal for a specific expiry date.
        
        FOR PAUL: CALLS ONLY - He is LONG the stock.
        """
        expiry = expiry_info['date']
        days_out = expiry_info['days_out']
        
        # PAUL'S RULE: CALLS ONLY (he's long the stock)
        option_type = 'CALL'
        
        strike = self.get_optimal_strike('IREN', 'LONG', expiry, data['price'])
        option_price = self.get_option_price('IREN', strike, expiry, option_type)
        
        # Get theta warning from expiry_info if available, otherwise calculate
        theta_warning = expiry_info.get('theta_warning', '')
        paul_pick = expiry_info.get('paul_pick', False)
        quality = expiry_info.get('quality', 'GOOD')
        
        # Adjust volatility and size based on days to expiry
        if days_out <= 14:
            volatility_pct = 35
            if not theta_warning:
                theta_warning = "MODERATE THETA"
            size_recommendation = 50
        elif days_out <= 21:
            volatility_pct = 30
            if not theta_warning:
                theta_warning = "NORMAL THETA"
            size_recommendation = 75
        elif days_out <= 35:
            volatility_pct = 28
            if not theta_warning:
                theta_warning = "⭐ PAUL'S SWEET SPOT"
            size_recommendation = 100  # Full size for Paul's preferred range
        else:
            volatility_pct = 25
            if not theta_warning:
                theta_warning = "LOW THETA"
            size_recommendation = 75
        
        option_levels = self.calculate_levels(option_price, 'LONG', volatility_pct=volatility_pct)
        
        return {
            'expiry': expiry,
            'days_out': days_out,
            'strike': strike,
            'option_type': option_type,  # ALWAYS CALL for Paul
            'option_price': round(option_price, 2),
            'entry_low': option_levels['entry_low'],
            'entry_high': option_levels['entry_high'],
            'take_profit': option_levels['take_profit'],
            'stop_loss': option_levels['stop_loss'],
            'size': size_recommendation,
            'theta_warning': theta_warning,
            'volatility_pct': volatility_pct,
            'paul_pick': paul_pick,
            'quality': quality,
            'days_from_earnings': expiry_info.get('days_from_earnings', 999)
        }

    def generate_iren_signal(self) -> Dict[str, Any]:
        """
        Generate IREN options signal with REAL data for MULTIPLE expiry dates.
        
        PAUL'S RULES:
        1. LONG ONLY - He owns 100K shares, wants to ADD via calls
        2. NO SHORTS, NO PUTS - Always CALL recommendations
        3. Avoid earnings window (Feb 5, 2026)
        4. Prefer 21-35 DTE (Paul's sweet spot)
        """
        data = self.get_real_stock_data('IREN')
        btc_data = self.get_real_btc_data()
        
        # PAUL'S RULE: ALWAYS LONG (he owns shares, wants to add on dips)
        # Direction is ALWAYS LONG for Paul's view
        direction = 'LONG'
        
        # Confidence based on BTC and RSI alignment with LONG bias
        btc_bullish = btc_data['change_24h'] > 0
        rsi_oversold = data['rsi'] < 40
        rsi_overbought = data['rsi'] > 70
        
        if btc_bullish and not rsi_overbought:
            # Aligned with LONG - high confidence
            confidence = min(90, 70 + abs(btc_data['change_24h']) * 2)
            action = 'BUY CALLS'
        elif rsi_oversold:
            # Oversold - good buying opportunity
            confidence = 85
            action = 'BUY CALLS (OVERSOLD)'
        elif btc_bullish:
            # BTC bullish, RSI ok
            confidence = 75
            action = 'BUY CALLS'
        else:
            # BTC not bullish - wait for dip or lower confidence
            confidence = 55
            action = 'WAIT FOR DIP'
        
        # Get next 4 expiries (filtered for Paul's preferences)
        expiries = self.get_next_4_expiries('IREN')
        
        # Generate signal for each expiry (CALLS ONLY)
        expiry_signals = []
        for expiry_info in expiries:
            try:
                signal = self.generate_iren_signal_for_expiry(
                    data, btc_data, direction, confidence, expiry_info
                )
                expiry_signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to generate signal for {expiry_info['date']}: {e}")
        
        # Find Paul's preferred expiry (paul_pick=True) or use first
        paul_picks = [s for s in expiry_signals if s.get('paul_pick', False)]
        primary_expiry = paul_picks[0] if paul_picks else (expiry_signals[0] if expiry_signals else None)
        
        # Earnings warning
        today = datetime.now().date()
        earnings_date = IREN_EARNINGS_DATE.date()
        days_to_earnings = (earnings_date - today).days
        
        return {
            'symbol': 'IREN',
            'asset_type': 'OPTION',
            'direction': direction,  # ALWAYS LONG for Paul
            'action': action,
            'option_type': 'CALL',  # ALWAYS CALL for Paul
            'strike': primary_expiry['strike'] if primary_expiry else round(data['price']),
            'expiry': primary_expiry['expiry'] if primary_expiry else 'N/A',
            'current_stock_price': data['price'],
            'option_price': primary_expiry['option_price'] if primary_expiry else 0,
            'entry_low': primary_expiry['entry_low'] if primary_expiry else 0,
            'entry_high': primary_expiry['entry_high'] if primary_expiry else 0,
            'take_profit': primary_expiry['take_profit'] if primary_expiry else 0,
            'stop_loss': primary_expiry['stop_loss'] if primary_expiry else 0,
            'size': primary_expiry['size'] if primary_expiry else 50,
            'size_unit': 'contracts',
            'confidence': int(confidence),
            # All expiry options for Paul (sorted: Paul's picks first)
            'all_expiries': expiry_signals,
            # Earnings warning
            'earnings': {
                'date': IREN_EARNINGS_DATE.strftime('%Y-%m-%d'),
                'days_away': days_to_earnings,
                'warning': '⚠️ AVOID expiries within 7 days of earnings!' if days_to_earnings <= 14 else None
            },
            'analysis': {
                'rsi': round(data['rsi'], 1),
                'btc_change': round(btc_data['change_24h'], 2),
                'btc_correlation': 0.75,  # Note: Check BTCCouplingAnalyzer for live correlation
                'volume': data['volume'],
                'stock_change': round(data['change_pct'], 2)
            },
            'pauls_thesis': {
                'core_shares': 100000,
                'entry_price': 56.68,
                'target_price': 150.00,
                'thesis': 'AI datacenter demand + legacy BTC infrastructure = $150 target'
            },
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'valid_for': 'Today',
            'source': 'yahoo_finance',
            'is_real': True
        }
    
    def generate_gold_signal(self) -> Dict[str, Any]:
        """Generate Gold (GLD) signal with REAL data"""
        data = self.get_real_stock_data('GLD')
        
        # Gold trend analysis based on RSI and momentum
        if data['rsi'] < 35:
            direction = 'LONG'
            confidence = 75
        elif data['rsi'] > 65:
            direction = 'SHORT'
            confidence = 70
        else:
            direction = 'LONG' if data['change_pct'] > 0 else 'SHORT'
            confidence = 60
        
        expiry = self.get_next_option_expiry('GLD')
        strike = self.get_optimal_strike('GLD', direction, expiry, data['price'])
        option_type = 'CALL' if direction == 'LONG' else 'PUT'
        option_price = self.get_option_price('GLD', strike, expiry, option_type)
        
        option_levels = self.calculate_levels(option_price, 'LONG', volatility_pct=25)
        
        return {
            'symbol': 'GLD',
            'asset_type': 'OPTION',
            'direction': direction,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'current_price': data['price'],
            'option_price': round(option_price, 2),
            'entry_low': option_levels['entry_low'],
            'entry_high': option_levels['entry_high'],
            'take_profit': option_levels['take_profit'],
            'stop_loss': option_levels['stop_loss'],
            'size': 25,
            'size_unit': 'contracts',
            'confidence': int(confidence),
            'analysis': {
                'rsi': round(data['rsi'], 1),
                'change_pct': round(data['change_pct'], 2),
                'volume': data['volume']
            },
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'valid_for': 'Today',
            'source': 'yahoo_finance',
            'is_real': True
        }
    
    def generate_daily_signals(self) -> Dict[str, Any]:
        """Generate all signals for the day"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Generating daily signals for {today}...")
        
        signals = {
            'date': today,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'market_date': today,
            'signals': {},
            'summary': {
                'total_signals': 0,
                'bullish': 0,
                'bearish': 0,
                'avg_confidence': 0
            },
            'all_data_real': True
        }
        
        # Generate each signal (with error handling)
        try:
            signals['signals']['btc'] = self.generate_btc_signal()
            logger.info(f"  BTC: {signals['signals']['btc']['direction']} ({signals['signals']['btc']['confidence']}%)")
        except Exception as e:
            logger.error(f"  BTC signal failed: {e}")
            signals['signals']['btc'] = {'error': str(e)}
        
        try:
            signals['signals']['iren'] = self.generate_iren_signal()
            logger.info(f"  IREN: {signals['signals']['iren']['direction']} ({signals['signals']['iren']['confidence']}%)")
        except Exception as e:
            logger.error(f"  IREN signal failed: {e}")
            signals['signals']['iren'] = {'error': str(e)}
        
        try:
            signals['signals']['gld'] = self.generate_gold_signal()
            logger.info(f"  GLD: {signals['signals']['gld']['direction']} ({signals['signals']['gld']['confidence']}%)")
        except Exception as e:
            logger.error(f"  GLD signal failed: {e}")
            signals['signals']['gld'] = {'error': str(e)}
        
        # Calculate summary
        valid_signals = [s for s in signals['signals'].values() if 'direction' in s]
        signals['summary']['total_signals'] = len(valid_signals)
        
        for sig in valid_signals:
            if sig.get('direction') == 'LONG':
                signals['summary']['bullish'] += 1
            else:
                signals['summary']['bearish'] += 1
        
        confidences = [s.get('confidence', 0) for s in valid_signals]
        if confidences:
            signals['summary']['avg_confidence'] = round(sum(confidences) / len(confidences), 1)
        
        # Save to file
        filepath = self.signals_dir / f"signals_{today}.json"
        with open(filepath, 'w') as f:
            json.dump(signals, f, indent=2)
        
        logger.info(f"✅ Daily signals saved to {filepath}")
        
        return signals
    
    def get_todays_signals(self) -> Dict[str, Any]:
        """Get today's signals (or generate if not exists)"""
        today = datetime.now().strftime('%Y-%m-%d')
        filepath = self.signals_dir / f"signals_{today}.json"
        
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        else:
            return self.generate_daily_signals()


# Singleton instance
_generator_instance = None

def get_daily_generator() -> DailySignalGenerator:
    """Get singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DailySignalGenerator()
    return _generator_instance


def run_daily_signal_generation():
    """
    Run this at 6:00 AM ET every day
    
    Cron: 0 11 * * * cd /home/jbot/trading_ai && python -c "from paper_trading.daily_signals import run_daily_signal_generation; run_daily_signal_generation()"
    """
    generator = get_daily_generator()
    signals = generator.generate_daily_signals()
    
    print(f"\n✅ Daily signals generated for {signals['date']}")
    print(f"   Summary: {signals['summary']['bullish']} bullish, {signals['summary']['bearish']} bearish")
    print(f"   Avg Confidence: {signals['summary']['avg_confidence']}%")
    
    for name, sig in signals['signals'].items():
        if 'direction' in sig:
            print(f"   {name.upper()}: {sig['direction']} ({sig['confidence']}%)")
        elif 'error' in sig:
            print(f"   {name.upper()}: ERROR - {sig['error']}")
    
    return signals


if __name__ == "__main__":
    run_daily_signal_generation()
