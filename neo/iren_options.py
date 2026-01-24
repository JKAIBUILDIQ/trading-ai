"""
IREN Options Chain Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Options analysis for IREN stock:
- Real-time options chain
- Greeks calculation
- Unusual activity detection
- Strategy recommendations

Created: 2026-01-24
For: Paul (Investment Partner)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
import math

log = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Single option contract"""
    contract_symbol: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    in_the_money: bool


@dataclass
class OptionsChain:
    """Complete options chain"""
    underlying_price: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    expirations: List[str]
    iv_rank: float  # 0-100
    iv_percentile: float  # 0-100
    put_call_ratio: float
    max_pain: float
    timestamp: datetime


@dataclass
class UnusualActivity:
    """Unusual options activity"""
    contract: OptionContract
    activity_type: str  # 'SWEEP', 'BLOCK', 'UNUSUAL_VOLUME'
    volume_oi_ratio: float
    premium_total: float
    sentiment: str  # 'BULLISH', 'BEARISH'


@dataclass
class OptionStrategy:
    """Options strategy recommendation"""
    name: str
    description: str
    legs: List[Dict]
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability_of_profit: float
    risk_reward_ratio: float
    sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    ideal_conditions: str


class BlackScholes:
    """Black-Scholes option pricing model"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2"""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate call option price"""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate put option price"""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='call'):
        """Calculate delta"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate gamma"""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='call'):
        """Calculate theta (per day)"""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == 'call':
            term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365  # Per day
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate vega (per 1% IV change)"""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) / 100


class IRENOptionsAnalyzer:
    """
    IREN options chain analyzer
    
    Features:
    - Real-time options chain from Yahoo Finance
    - Greeks calculation
    - Unusual activity detection
    - Strategy recommendations
    - Max pain calculation
    """
    
    def __init__(self):
        self.ticker = 'IREN'
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.last_chain = None
    
    def fetch_options_chain(self, expiration: str = None) -> OptionsChain:
        """
        Fetch options chain from Yahoo Finance
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(self.ticker)
            underlying_price = stock.history(period='1d')['Close'].iloc[-1]
            
            # Get all expirations
            expirations = list(stock.options)
            
            if not expirations:
                log.warning("No options data available for IREN")
                return self._create_empty_chain(underlying_price)
            
            # Use first expiration if none specified
            if expiration is None or expiration not in expirations:
                expiration = expirations[0]
            
            # Fetch options chain
            opt = stock.option_chain(expiration)
            
            # Process calls
            calls = self._process_options(opt.calls, underlying_price, expiration, 'call')
            
            # Process puts
            puts = self._process_options(opt.puts, underlying_price, expiration, 'put')
            
            # Calculate metrics
            total_call_volume = sum(c.volume for c in calls)
            total_put_volume = sum(p.volume for p in puts)
            put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 1.0
            
            # IV Rank (would need historical IV data for accurate calculation)
            current_iv = np.mean([c.implied_volatility for c in calls if c.implied_volatility > 0])
            iv_rank = self._estimate_iv_rank(current_iv)
            iv_percentile = iv_rank  # Simplified
            
            # Max Pain
            max_pain = self._calculate_max_pain(calls, puts, underlying_price)
            
            chain = OptionsChain(
                underlying_price=round(underlying_price, 2),
                calls=calls,
                puts=puts,
                expirations=expirations,
                iv_rank=round(iv_rank, 1),
                iv_percentile=round(iv_percentile, 1),
                put_call_ratio=round(put_call_ratio, 2),
                max_pain=round(max_pain, 2),
                timestamp=datetime.utcnow()
            )
            
            self.last_chain = chain
            return chain
            
        except Exception as e:
            log.error(f"Error fetching options chain: {e}")
            return self._create_empty_chain(0)
    
    def _process_options(self, df: pd.DataFrame, spot: float, 
                         expiration: str, option_type: str) -> List[OptionContract]:
        """Process options DataFrame into OptionContract list"""
        contracts = []
        
        # Calculate time to expiration
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        T = max(0.001, (exp_date - datetime.now()).days / 365)
        
        for _, row in df.iterrows():
            try:
                strike = row['strike']
                iv = row.get('impliedVolatility', 0.5)
                
                # Calculate Greeks
                delta = BlackScholes.delta(spot, strike, T, self.risk_free_rate, iv, option_type)
                gamma = BlackScholes.gamma(spot, strike, T, self.risk_free_rate, iv)
                theta = BlackScholes.theta(spot, strike, T, self.risk_free_rate, iv, option_type)
                vega = BlackScholes.vega(spot, strike, T, self.risk_free_rate, iv)
                
                contract = OptionContract(
                    contract_symbol=row.get('contractSymbol', ''),
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    last_price=row.get('lastPrice', 0),
                    bid=row.get('bid', 0),
                    ask=row.get('ask', 0),
                    volume=int(row.get('volume', 0) or 0),
                    open_interest=int(row.get('openInterest', 0) or 0),
                    implied_volatility=iv,
                    delta=round(delta, 3),
                    gamma=round(gamma, 4),
                    theta=round(theta, 4),
                    vega=round(vega, 4),
                    in_the_money=row.get('inTheMoney', False)
                )
                
                contracts.append(contract)
                
            except Exception as e:
                log.debug(f"Error processing option row: {e}")
                continue
        
        return contracts
    
    def _calculate_max_pain(self, calls: List[OptionContract], 
                            puts: List[OptionContract], spot: float) -> float:
        """
        Calculate max pain price
        Max pain = price at which total option holder losses are maximized
        """
        strikes = sorted(set([c.strike for c in calls] + [p.strike for p in puts]))
        
        if not strikes:
            return spot
        
        min_pain = float('inf')
        max_pain_strike = spot
        
        for test_price in strikes:
            total_pain = 0
            
            # Call holder losses
            for call in calls:
                if test_price > call.strike:
                    # Calls expire ITM - no loss
                    pass
                else:
                    # Calls expire OTM - full premium loss
                    total_pain += call.open_interest * call.last_price * 100
            
            # Put holder losses
            for put in puts:
                if test_price < put.strike:
                    # Puts expire ITM - no loss
                    pass
                else:
                    # Puts expire OTM - full premium loss
                    total_pain += put.open_interest * put.last_price * 100
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_price
        
        return max_pain_strike
    
    def _estimate_iv_rank(self, current_iv: float) -> float:
        """
        Estimate IV rank (0-100)
        Would need historical IV data for accurate calculation
        """
        # IREN typical IV range: 60% - 150%
        iv_low = 0.60
        iv_high = 1.50
        
        if current_iv <= iv_low:
            return 0
        elif current_iv >= iv_high:
            return 100
        else:
            return ((current_iv - iv_low) / (iv_high - iv_low)) * 100
    
    def _create_empty_chain(self, price: float) -> OptionsChain:
        """Create empty options chain"""
        return OptionsChain(
            underlying_price=price,
            calls=[],
            puts=[],
            expirations=[],
            iv_rank=50,
            iv_percentile=50,
            put_call_ratio=1.0,
            max_pain=price,
            timestamp=datetime.utcnow()
        )
    
    def detect_unusual_activity(self) -> List[UnusualActivity]:
        """
        Detect unusual options activity
        """
        chain = self.last_chain or self.fetch_options_chain()
        unusual = []
        
        for contract in chain.calls + chain.puts:
            # Volume > Open Interest (unusual)
            if contract.open_interest > 0:
                vol_oi_ratio = contract.volume / contract.open_interest
                
                if vol_oi_ratio > 2.0:  # Volume 2x OI
                    premium = contract.volume * contract.last_price * 100
                    
                    # Determine sentiment
                    if contract.option_type == 'call':
                        sentiment = 'BULLISH'
                    else:
                        sentiment = 'BEARISH'
                    
                    unusual.append(UnusualActivity(
                        contract=contract,
                        activity_type='UNUSUAL_VOLUME',
                        volume_oi_ratio=round(vol_oi_ratio, 2),
                        premium_total=round(premium, 2),
                        sentiment=sentiment
                    ))
            
            # Large volume absolute
            if contract.volume > 1000:
                premium = contract.volume * contract.last_price * 100
                
                sentiment = 'BULLISH' if contract.option_type == 'call' else 'BEARISH'
                
                # Check if it's a sweep (aggressive buying)
                if contract.last_price >= contract.ask * 0.98:
                    unusual.append(UnusualActivity(
                        contract=contract,
                        activity_type='SWEEP',
                        volume_oi_ratio=contract.volume / max(1, contract.open_interest),
                        premium_total=round(premium, 2),
                        sentiment=sentiment
                    ))
        
        return unusual
    
    def recommend_strategies(self, outlook: str = 'BULLISH') -> List[OptionStrategy]:
        """
        Recommend options strategies based on outlook
        """
        chain = self.last_chain or self.fetch_options_chain()
        strategies = []
        
        spot = chain.underlying_price
        
        if outlook == 'BULLISH':
            # 1. Long Call
            atm_call = self._find_atm_option(chain.calls, spot)
            if atm_call:
                strategies.append(OptionStrategy(
                    name='Long Call',
                    description=f'Buy {atm_call.strike} Call @ ${atm_call.ask:.2f}',
                    legs=[{'action': 'BUY', 'contract': atm_call}],
                    max_profit=float('inf'),
                    max_loss=atm_call.ask * 100,
                    breakeven=[atm_call.strike + atm_call.ask],
                    probability_of_profit=0.35,  # Rough estimate
                    risk_reward_ratio=0,  # Unlimited upside
                    sentiment='BULLISH',
                    ideal_conditions='Strong upward move expected'
                ))
            
            # 2. Bull Call Spread
            otm_call = self._find_otm_option(chain.calls, spot, 1)
            if atm_call and otm_call:
                debit = atm_call.ask - otm_call.bid
                max_profit = (otm_call.strike - atm_call.strike - debit) * 100
                
                strategies.append(OptionStrategy(
                    name='Bull Call Spread',
                    description=f'Buy {atm_call.strike} Call, Sell {otm_call.strike} Call',
                    legs=[
                        {'action': 'BUY', 'contract': atm_call},
                        {'action': 'SELL', 'contract': otm_call}
                    ],
                    max_profit=max_profit,
                    max_loss=debit * 100,
                    breakeven=[atm_call.strike + debit],
                    probability_of_profit=0.45,
                    risk_reward_ratio=max_profit / (debit * 100) if debit > 0 else 0,
                    sentiment='BULLISH',
                    ideal_conditions='Moderate upward move, lower cost than long call'
                ))
            
            # 3. Cash-Secured Put
            otm_put = self._find_otm_option(chain.puts, spot, -1)
            if otm_put:
                strategies.append(OptionStrategy(
                    name='Cash-Secured Put',
                    description=f'Sell {otm_put.strike} Put @ ${otm_put.bid:.2f}',
                    legs=[{'action': 'SELL', 'contract': otm_put}],
                    max_profit=otm_put.bid * 100,
                    max_loss=(otm_put.strike - otm_put.bid) * 100,
                    breakeven=[otm_put.strike - otm_put.bid],
                    probability_of_profit=0.65,
                    risk_reward_ratio=otm_put.bid / (otm_put.strike - otm_put.bid) if otm_put.strike > otm_put.bid else 0,
                    sentiment='BULLISH',
                    ideal_conditions='Willing to own shares at lower price, collect premium'
                ))
        
        elif outlook == 'BEARISH':
            # 1. Long Put
            atm_put = self._find_atm_option(chain.puts, spot)
            if atm_put:
                strategies.append(OptionStrategy(
                    name='Long Put',
                    description=f'Buy {atm_put.strike} Put @ ${atm_put.ask:.2f}',
                    legs=[{'action': 'BUY', 'contract': atm_put}],
                    max_profit=(atm_put.strike - atm_put.ask) * 100,
                    max_loss=atm_put.ask * 100,
                    breakeven=[atm_put.strike - atm_put.ask],
                    probability_of_profit=0.35,
                    risk_reward_ratio=(atm_put.strike - atm_put.ask) / atm_put.ask if atm_put.ask > 0 else 0,
                    sentiment='BEARISH',
                    ideal_conditions='Strong downward move expected'
                ))
        
        elif outlook == 'NEUTRAL':
            # Iron Condor
            strategies.append(OptionStrategy(
                name='Iron Condor',
                description='Sell OTM call spread + sell OTM put spread',
                legs=[],
                max_profit=0,  # Would calculate from actual positions
                max_loss=0,
                breakeven=[spot * 0.9, spot * 1.1],
                probability_of_profit=0.65,
                risk_reward_ratio=0.5,
                sentiment='NEUTRAL',
                ideal_conditions='Low volatility, range-bound price action'
            ))
        
        return strategies
    
    def _find_atm_option(self, options: List[OptionContract], spot: float) -> Optional[OptionContract]:
        """Find at-the-money option"""
        if not options:
            return None
        
        return min(options, key=lambda x: abs(x.strike - spot))
    
    def _find_otm_option(self, options: List[OptionContract], spot: float, 
                         direction: int = 1) -> Optional[OptionContract]:
        """
        Find out-of-the-money option
        direction: 1 for calls (above spot), -1 for puts (below spot)
        """
        if not options:
            return None
        
        if direction > 0:
            otm = [o for o in options if o.strike > spot]
        else:
            otm = [o for o in options if o.strike < spot]
        
        if not otm:
            return None
        
        # Return closest OTM
        return min(otm, key=lambda x: abs(x.strike - spot))
    
    def to_dict(self) -> Dict:
        """Convert options analysis to dictionary"""
        chain = self.last_chain or self.fetch_options_chain()
        unusual = self.detect_unusual_activity()
        
        return {
            'ticker': self.ticker,
            'underlying_price': chain.underlying_price,
            'iv_rank': chain.iv_rank,
            'iv_percentile': chain.iv_percentile,
            'put_call_ratio': chain.put_call_ratio,
            'max_pain': chain.max_pain,
            'expirations': chain.expirations,
            
            'calls_summary': {
                'count': len(chain.calls),
                'total_volume': sum(c.volume for c in chain.calls),
                'total_oi': sum(c.open_interest for c in chain.calls)
            },
            
            'puts_summary': {
                'count': len(chain.puts),
                'total_volume': sum(p.volume for p in chain.puts),
                'total_oi': sum(p.open_interest for p in chain.puts)
            },
            
            'unusual_activity': [
                {
                    'strike': ua.contract.strike,
                    'type': ua.contract.option_type,
                    'activity': ua.activity_type,
                    'vol_oi_ratio': ua.volume_oi_ratio,
                    'premium': ua.premium_total,
                    'sentiment': ua.sentiment
                }
                for ua in unusual[:5]  # Top 5
            ],
            
            'timestamp': chain.timestamp.isoformat()
        }


def get_iren_options() -> Dict:
    """Quick function for API integration"""
    analyzer = IRENOptionsAnalyzer()
    return analyzer.to_dict()


# Test
if __name__ == "__main__":
    print("="*70)
    print("📈 IREN OPTIONS CHAIN ANALYSIS")
    print("="*70)
    
    analyzer = IRENOptionsAnalyzer()
    chain = analyzer.fetch_options_chain()
    
    print(f"\n💰 UNDERLYING: ${chain.underlying_price:.2f}")
    print(f"   IV Rank: {chain.iv_rank:.1f}%")
    print(f"   Put/Call Ratio: {chain.put_call_ratio:.2f}")
    print(f"   Max Pain: ${chain.max_pain:.2f}")
    
    print(f"\n📊 OPTIONS SUMMARY:")
    print(f"   Calls: {len(chain.calls)} contracts")
    print(f"   Puts: {len(chain.puts)} contracts")
    print(f"   Expirations: {len(chain.expirations)}")
    
    if chain.expirations:
        print(f"   Next Expiry: {chain.expirations[0]}")
    
    # Unusual activity
    unusual = analyzer.detect_unusual_activity()
    if unusual:
        print(f"\n⚠️ UNUSUAL ACTIVITY:")
        for ua in unusual[:3]:
            print(f"   {ua.contract.option_type.upper()} ${ua.contract.strike}")
            print(f"      Activity: {ua.activity_type} | Vol/OI: {ua.volume_oi_ratio:.1f}x")
            print(f"      Sentiment: {ua.sentiment}")
    
    # Strategy recommendations
    print(f"\n📋 BULLISH STRATEGIES:")
    strategies = analyzer.recommend_strategies('BULLISH')
    for strat in strategies[:3]:
        print(f"   {strat.name}: {strat.description}")
        print(f"      Max Profit: ${strat.max_profit:,.0f} | Max Loss: ${strat.max_loss:,.0f}")
        print(f"      P(Profit): {strat.probability_of_profit:.0%}")
