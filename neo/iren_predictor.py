#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO IREN 4-HOUR PREDICTOR + OPTIONS RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

Predicts IREN's next 4-hour movement with:
1. Technical analysis (RSI, EMA, MACD, momentum)
2. BTC correlation analysis (decoupling detection)
3. Options recommendations (best contracts for the prediction)
4. Earnings awareness (position sizing near earnings)

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Pattern recognition
try:
    from pattern_detector import PatternDetector
    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IrenPredictor")

# IREN Configuration
IREN_EARNINGS_DATE = "2026-02-05"
PREFERRED_STRIKES = [60, 70, 80]
PAUL_PREFERRED_DTE = (21, 35)  # Paul's sweet spot


@dataclass
class OptionRecommendation:
    """A single options recommendation"""
    strike: float
    expiration: str
    dte: int
    option_type: str  # CALL only for IREN
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    is_pauls_pick: bool
    recommendation: str  # BUY, HOLD, AVOID
    reasoning: str


@dataclass 
class IrenPrediction:
    """IREN 4-hour prediction with options"""
    prediction_id: str
    timestamp: str
    target_time: str
    
    # Price data
    current_price: float
    predicted_direction: str  # UP, DOWN, FLAT
    predicted_change_pct: float
    predicted_price: float
    confidence: float
    
    # Technicals
    technicals: Dict = field(default_factory=dict)
    
    # BTC correlation
    btc_analysis: Dict = field(default_factory=dict)
    
    # Options recommendations
    options: List[Dict] = field(default_factory=list)
    best_option: Dict = field(default_factory=dict)
    
    # Earnings
    earnings: Dict = field(default_factory=dict)
    
    # Signal
    signal: str = ""  # BUY_CALLS, HOLD, WAIT
    reasoning: str = ""
    warnings: List[str] = field(default_factory=list)
    
    # Status
    status: str = "PENDING"
    time_remaining_minutes: int = 0
    time_remaining_display: str = ""


class IrenPredictor:
    """
    4-Hour IREN Price Predictor with Options Recommendations
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "prediction_data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.history_file = str(self.data_dir / "iren_prediction_history.json")
        
        logger.info("=" * 60)
        logger.info("🔮 IREN PREDICTOR INITIALIZED")
        logger.info("=" * 60)
    
    def _get_iren_data(self, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """Fetch IREN price data"""
        try:
            df = yf.download('IREN', period=period, interval=interval, progress=False)
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error fetching IREN data: {e}")
            return pd.DataFrame()
    
    def _get_btc_data(self, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """Fetch BTC price data for correlation"""
        try:
            df = yf.download('BTC-USD', period=period, interval=interval, progress=False)
            if hasattr(df.columns, 'levels'):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error fetching BTC data: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        if len(df) < period + 1:
            return 50.0
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        return float(rsi) if not np.isnan(rsi) else 50.0
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calculate EMA"""
        return float(df['close'].ewm(span=period, adjust=False).mean().iloc[-1])
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[float, float, str]:
        """Calculate MACD"""
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        
        if macd_val > signal_val:
            state = "BULLISH"
        elif macd_val < signal_val:
            state = "BEARISH"
        else:
            state = "NEUTRAL"
        
        return macd_val, signal_val, state
    
    def _get_options_chain(self) -> List[Dict]:
        """Fetch IREN options chain"""
        try:
            ticker = yf.Ticker('IREN')
            
            # Get all expiration dates
            expirations = ticker.options
            if not expirations:
                return []
            
            options_list = []
            now = datetime.now()
            earnings_date = datetime.strptime(IREN_EARNINGS_DATE, "%Y-%m-%d")
            
            for exp in expirations[:6]:  # Next 6 expirations
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d")
                    dte = (exp_date - now).days
                    
                    # Skip if too close or too far
                    if dte < 7 or dte > 90:
                        continue
                    
                    # Get calls
                    chain = ticker.option_chain(exp)
                    calls = chain.calls
                    
                    for strike in PREFERRED_STRIKES:
                        # Find closest strike
                        strike_options = calls[abs(calls['strike'] - strike) < 1]
                        
                        if not strike_options.empty:
                            opt = strike_options.iloc[0]
                            
                            # Check if Paul's pick (21-35 DTE, away from earnings)
                            is_pauls_pick = (
                                PAUL_PREFERRED_DTE[0] <= dte <= PAUL_PREFERRED_DTE[1] and
                                abs((exp_date - earnings_date).days) > 7
                            )
                            
                            # Determine recommendation
                            if dte < 14:
                                rec = "AVOID"
                                reason = "Too close to expiration"
                            elif abs((exp_date - earnings_date).days) < 3:
                                rec = "CAUTION"
                                reason = "Expires near earnings"
                            elif is_pauls_pick:
                                rec = "BUY"
                                reason = "Paul's sweet spot (21-35 DTE)"
                            else:
                                rec = "CONSIDER"
                                reason = f"{dte} DTE"
                            
                            options_list.append({
                                'strike': float(opt['strike']),
                                'expiration': exp,
                                'dte': dte,
                                'option_type': 'CALL',
                                'last_price': float(opt['lastPrice']) if pd.notna(opt['lastPrice']) else 0,
                                'bid': float(opt['bid']) if pd.notna(opt['bid']) else 0,
                                'ask': float(opt['ask']) if pd.notna(opt['ask']) else 0,
                                'volume': int(opt['volume']) if pd.notna(opt['volume']) else 0,
                                'open_interest': int(opt['openInterest']) if pd.notna(opt['openInterest']) else 0,
                                'implied_volatility': float(opt['impliedVolatility']) if pd.notna(opt['impliedVolatility']) else 0,
                                'delta': 0.5,  # Approximate
                                'is_pauls_pick': is_pauls_pick,
                                'recommendation': rec,
                                'reasoning': reason
                            })
                except Exception as e:
                    logger.warning(f"Error processing expiration {exp}: {e}")
                    continue
            
            return sorted(options_list, key=lambda x: (x['strike'], x['dte']))
            
        except Exception as e:
            logger.error(f"Error fetching options: {e}")
            return []
    
    def _calculate_btc_correlation(self, iren_df: pd.DataFrame, btc_df: pd.DataFrame) -> Dict:
        """Calculate BTC-IREN correlation"""
        try:
            # Align data
            common_idx = iren_df.index.intersection(btc_df.index)
            if len(common_idx) < 20:
                return {"correlation": 0, "status": "UNKNOWN"}
            
            iren_returns = iren_df.loc[common_idx, 'close'].pct_change().dropna()
            btc_returns = btc_df.loc[common_idx, 'close'].pct_change().dropna()
            
            # Align returns
            common_ret_idx = iren_returns.index.intersection(btc_returns.index)
            correlation = iren_returns.loc[common_ret_idx].corr(btc_returns.loc[common_ret_idx])
            
            # Determine status
            if correlation > 0.7:
                status = "COUPLED"
            elif correlation > 0.5:
                status = "MODERATE"
            elif correlation > 0.3:
                status = "WEAK"
            else:
                status = "DECOUPLING"
            
            return {
                "correlation": round(float(correlation), 3) if not np.isnan(correlation) else 0,
                "status": status,
                "btc_price": float(btc_df['close'].iloc[-1]),
                "btc_change_24h": float((btc_df['close'].iloc[-1] / btc_df['close'].iloc[-24] - 1) * 100) if len(btc_df) >= 24 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return {"correlation": 0, "status": "UNKNOWN"}
    
    def predict_4h(self) -> IrenPrediction:
        """
        Generate 4-hour IREN prediction with options recommendations.
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔮 GENERATING 4-HOUR IREN PREDICTION")
        logger.info("=" * 60)
        
        # Get data
        iren_df = self._get_iren_data(period="7d", interval="1h")
        btc_df = self._get_btc_data(period="7d", interval="1h")
        
        if iren_df.empty:
            return self._create_error_prediction("No IREN data available")
        
        current_price = float(iren_df['close'].iloc[-1])
        prev_close = float(iren_df['close'].iloc[-24]) if len(iren_df) >= 24 else current_price
        
        # Calculate technicals
        rsi = self._calculate_rsi(iren_df)
        ema20 = self._calculate_ema(iren_df, 20)
        ema50 = self._calculate_ema(iren_df, 50)
        macd_val, macd_signal, macd_state = self._calculate_macd(iren_df)
        
        # Momentum
        momentum_4h = (current_price - iren_df['close'].iloc[-5]) / iren_df['close'].iloc[-5] * 100 if len(iren_df) >= 5 else 0
        
        # Trend
        if ema20 > ema50 * 1.02:
            trend = "STRONG_UP"
        elif ema20 > ema50:
            trend = "UP"
        elif ema20 < ema50 * 0.98:
            trend = "STRONG_DOWN"
        elif ema20 < ema50:
            trend = "DOWN"
        else:
            trend = "RANGING"
        
        technicals = {
            "rsi": round(rsi, 1),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "macd": round(macd_val, 4),
            "macd_signal": round(macd_signal, 4),
            "macd_state": macd_state,
            "momentum_4h": round(momentum_4h, 2),
            "trend": trend,
            "price_vs_ema20": round((current_price / ema20 - 1) * 100, 2),
            "change_24h": round((current_price / prev_close - 1) * 100, 2)
        }
        
        # BTC correlation
        btc_analysis = self._calculate_btc_correlation(iren_df, btc_df)
        
        # Calculate prediction
        bullish_score = 0
        bearish_score = 0
        reasoning_parts = []
        warnings = []
        
        # RSI
        if rsi < 30:
            bullish_score += 0.2
            reasoning_parts.append(f"RSI {rsi:.0f} oversold")
        elif rsi > 70:
            if trend in ["STRONG_UP", "UP"]:
                reasoning_parts.append(f"RSI {rsi:.0f} overbought but UPTREND (reduced)")
                bearish_score += 0.05  # Reduced in uptrend
            else:
                bearish_score += 0.15
                reasoning_parts.append(f"RSI {rsi:.0f} overbought")
        
        # EMA trend
        if trend in ["STRONG_UP", "UP"]:
            bullish_score += 0.25
            reasoning_parts.append(f"EMA trend {trend}")
        elif trend in ["STRONG_DOWN", "DOWN"]:
            bearish_score += 0.25
            reasoning_parts.append(f"EMA trend {trend}")
        
        # MACD
        if macd_state == "BULLISH":
            bullish_score += 0.15
            reasoning_parts.append("MACD bullish")
        elif macd_state == "BEARISH":
            bearish_score += 0.15
            reasoning_parts.append("MACD bearish")
        
        # Momentum
        if momentum_4h > 1:
            bullish_score += 0.15
            reasoning_parts.append(f"Strong momentum +{momentum_4h:.1f}%")
        elif momentum_4h < -1:
            bearish_score += 0.15
            reasoning_parts.append(f"Weak momentum {momentum_4h:.1f}%")
        
        # BTC decoupling bonus
        if btc_analysis.get("status") == "DECOUPLING":
            reasoning_parts.append("BTC decoupling (independent movement)")
        
        # Pattern recognition
        if PATTERNS_AVAILABLE:
            try:
                detector = PatternDetector("IREN")
                patterns = detector.analyze(iren_df)
                pattern_signal = detector.get_combined_signal(patterns)
                
                if pattern_signal["direction"] != "NEUTRAL":
                    pattern_weight = 0.15 * (pattern_signal["confidence"] / 100)
                    
                    if pattern_signal["direction"] == "BUY":
                        bullish_score += pattern_weight
                    else:
                        bearish_score += pattern_weight
                    
                    reasoning_parts.append(
                        f"Patterns: {pattern_signal['direction']} "
                        f"({pattern_signal['pattern_count']} detected)"
                    )
                    logger.info(f"   🔍 Patterns: {pattern_signal['patterns']}")
            except Exception as e:
                logger.warning(f"Pattern detection failed: {e}")
        
        # Determine direction
        net_score = bullish_score - bearish_score
        if net_score > 0.1:
            direction = "UP"
            predicted_change = abs(net_score) * 3  # 3% max move
        elif net_score < -0.1:
            direction = "DOWN"
            predicted_change = -abs(net_score) * 3
        else:
            direction = "FLAT"
            predicted_change = 0
        
        predicted_price = current_price * (1 + predicted_change / 100)
        
        # Confidence
        confidence = 50 + abs(net_score) * 50
        if trend in ["STRONG_UP", "STRONG_DOWN"]:
            confidence += 10
        confidence = min(85, confidence)
        
        # Earnings check
        earnings_date = datetime.strptime(IREN_EARNINGS_DATE, "%Y-%m-%d")
        days_to_earnings = (earnings_date - datetime.now()).days
        
        earnings = {
            "date": IREN_EARNINGS_DATE,
            "days_away": days_to_earnings,
            "phase": "PRE_EARNINGS" if days_to_earnings > 0 else "POST_EARNINGS"
        }
        
        if 0 < days_to_earnings <= 7:
            warnings.append(f"⚠️ Earnings in {days_to_earnings} days - increased volatility expected!")
            confidence *= 0.8
        
        # Get options
        options = self._get_options_chain()
        
        # Find best option
        best_option = {}
        if options and direction == "UP":
            # Prefer Paul's picks, then by DTE
            pauls_picks = [o for o in options if o['is_pauls_pick'] and o['strike'] == 60]
            if pauls_picks:
                best_option = pauls_picks[0]
            else:
                # Find $60 strike with good DTE
                sixty_calls = [o for o in options if o['strike'] == 60 and 14 <= o['dte'] <= 45]
                if sixty_calls:
                    best_option = sixty_calls[0]
        
        # Signal
        if direction == "UP" and confidence >= 60:
            signal = "BUY_CALLS"
        elif direction == "DOWN":
            signal = "WAIT"  # Paul is LONG ONLY
        else:
            signal = "HOLD"
        
        # Create prediction
        now = datetime.now(timezone.utc)
        target_time = now + timedelta(hours=4)
        
        prediction = IrenPrediction(
            prediction_id=f"IREN_PRED_{now.strftime('%Y%m%d_%H%M%S')}",
            timestamp=now.isoformat(),
            target_time=target_time.isoformat(),
            current_price=current_price,
            predicted_direction=direction,
            predicted_change_pct=round(predicted_change, 2),
            predicted_price=round(predicted_price, 2),
            confidence=round(confidence, 1),
            technicals=technicals,
            btc_analysis=btc_analysis,
            options=options[:12],  # Top 12 options
            best_option=best_option,
            earnings=earnings,
            signal=signal,
            reasoning=" | ".join(reasoning_parts),
            warnings=warnings,
            status="PENDING",
            time_remaining_minutes=240,
            time_remaining_display="4h 0m"
        )
        
        logger.info(f"\n📊 PREDICTION: {direction} {predicted_change:+.2f}%")
        logger.info(f"   Confidence: {confidence:.0f}%")
        logger.info(f"   Signal: {signal}")
        logger.info(f"   Best Option: ${best_option.get('strike', 'N/A')} {best_option.get('expiration', 'N/A')}")
        
        return prediction
    
    def _create_error_prediction(self, error: str) -> IrenPrediction:
        """Create error prediction"""
        now = datetime.now(timezone.utc)
        return IrenPrediction(
            prediction_id=f"IREN_ERROR_{now.strftime('%Y%m%d_%H%M%S')}",
            timestamp=now.isoformat(),
            target_time=(now + timedelta(hours=4)).isoformat(),
            current_price=0,
            predicted_direction="UNKNOWN",
            predicted_change_pct=0,
            predicted_price=0,
            confidence=0,
            signal="ERROR",
            reasoning=error,
            warnings=[error],
            status="ERROR"
        )


# Singleton
_predictor = None

def get_iren_predictor() -> IrenPredictor:
    global _predictor
    if _predictor is None:
        _predictor = IrenPredictor()
    return _predictor


def predict_iren_4h() -> Dict:
    """Quick function to get IREN 4-hour prediction"""
    predictor = get_iren_predictor()
    prediction = predictor.predict_4h()
    return asdict(prediction)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING IREN PREDICTOR")
    print("=" * 70)
    
    result = predict_iren_4h()
    print(json.dumps(result, indent=2, default=str)[:2000])
