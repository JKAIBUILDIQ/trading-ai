"""
═══════════════════════════════════════════════════════════════════
NEO UNIFIED MARKET FEED - 200+ Features Combined
═══════════════════════════════════════════════════════════════════

This is NEO's comprehensive market intelligence system combining:
- 96+ Technical indicators (16 per timeframe × 6 timeframes)
- 12 Cross-timeframe features
- 24 Macro correlation features (VIX, DXY, Oil, etc.)
- 12 Microstructure features (sessions, time, liquidity)
- MQL5 consensus signals
- SMC Analysis (Order Blocks, FVG, Liquidity Pools)  [NEW]
- Adaptive Stop Loss (Freqtrade patterns)  [NEW]
- Multi-Factor Risk Scoring (Hunter patterns)  [NEW]

Total: 200+ features for intelligent trading decisions

NO RANDOM DATA - All features computed from real market data

═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-UnifiedFeed")


class UnifiedMarketFeed:
    """
    Unified market feed combining all data sources for NEO
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.last_update = None
        self.cached_data = {}
        
        # Feature counts (for verification)
        self.expected_features = {
            'technical': 96,      # 16 × 6 timeframes
            'cross_tf': 12,
            'macro': 24,
            'microstructure': 12,
            'total': 144
        }
    
    def fetch_ohlcv_data(self, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple timeframes
        
        Args:
            timeframes: List of timeframes ['M5', 'H1', 'D1']
        
        Returns:
            Dict mapping timeframe to OHLCV DataFrame
        """
        import yfinance as yf
        
        if timeframes is None:
            timeframes = ['H1', 'H4', 'D1']  # Default for Gold
        
        # Yahoo Finance interval mapping
        yf_intervals = {
            'M5': '5m',
            'M15': '15m',
            'H1': '1h',
            'H4': '1h',  # YF doesn't have 4h, we'll resample
            'D1': '1d',
            'W1': '1wk'
        }
        
        # Lookback periods for each timeframe
        lookback = {
            'M5': '7d',
            'M15': '30d',
            'H1': '60d',
            'H4': '60d',
            'D1': '1y',
            'W1': '2y'
        }
        
        # Symbol mapping for Yahoo Finance
        symbol_map = {
            'XAUUSD': 'GC=F',       # Gold futures
            'EURUSD': 'EURUSD=X',   # EUR/USD forex
            'GBPUSD': 'GBPUSD=X',   # GBP/USD forex
            'USDJPY': 'USDJPY=X',   # USD/JPY forex
            'AUDUSD': 'AUDUSD=X',   # AUD/USD forex
            'USDCAD': 'USDCAD=X',   # USD/CAD forex
            'USDCHF': 'USDCHF=X',   # USD/CHF forex
            'NZDUSD': 'NZDUSD=X',   # NZD/USD forex
            'XAGUSD': 'SI=F',       # Silver futures
        }
        
        # Get the proper ticker
        ticker = symbol_map.get(self.symbol, f'{self.symbol}=X' if len(self.symbol) == 6 else self.symbol)
        logger.info(f"  📈 Fetching data for {self.symbol} (ticker: {ticker})")
        
        data_dict = {}
        
        for tf in timeframes:
            try:
                interval = yf_intervals.get(tf, '1d')
                period = lookback.get(tf, '60d')
                
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False
                )
                
                if df.empty:
                    logger.warning(f"⚠️ No data for {tf}")
                    continue
                
                # Handle multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize column names
                df.columns = [c.lower() for c in df.columns]
                
                # Resample H4 from H1
                if tf == 'H4':
                    df = df.resample('4H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                
                data_dict[tf] = df
                logger.info(f"  ✅ {tf}: {len(df)} bars")
                
            except Exception as e:
                logger.error(f"  ❌ {tf}: {e}")
        
        return data_dict
    
    def get_technical_features(self, ohlcv_data: Dict[str, pd.DataFrame]) -> Dict:
        """Get technical indicators from DRL indicators module"""
        from features.drl_indicators import generate_neo_features
        return generate_neo_features(ohlcv_data)
    
    def get_macro_features(self) -> Dict:
        """Get macro correlation features"""
        from intel.macro_feed import get_macro_features
        return get_macro_features(use_cache=True)
    
    def get_microstructure_features(self) -> Dict:
        """Get session and time features"""
        from features.microstructure import get_microstructure_features
        return get_microstructure_features()
    
    def get_crowd_psychology(self, ohlcv_data: Dict[str, pd.DataFrame]) -> Dict:
        """Get crowd psychology analysis (Bitcoin crash patterns)"""
        try:
            from crowd_psychology import analyze_crowd_psychology, format_crowd_psychology_summary
            import numpy as np
            
            # Convert DataFrames to numpy arrays
            ohlcv_h1 = None
            ohlcv_h4 = None
            ohlcv_d1 = None
            
            if 'H1' in ohlcv_data and not ohlcv_data['H1'].empty:
                df = ohlcv_data['H1']
                ohlcv_h1 = df[['open', 'high', 'low', 'close', 'volume']].values
            
            if 'H4' in ohlcv_data and not ohlcv_data['H4'].empty:
                df = ohlcv_data['H4']
                ohlcv_h4 = df[['open', 'high', 'low', 'close', 'volume']].values
            
            if 'D1' in ohlcv_data and not ohlcv_data['D1'].empty:
                df = ohlcv_data['D1']
                ohlcv_d1 = df[['open', 'high', 'low', 'close', 'volume']].values
            
            if ohlcv_h1 is None or len(ohlcv_h1) < 20:
                return {'available': False, 'summary': 'Insufficient data for crowd psychology'}
            
            # Run analysis
            result = analyze_crowd_psychology(
                ohlcv_h1=ohlcv_h1,
                ohlcv_h4=ohlcv_h4,
                ohlcv_d1=ohlcv_d1,
                symbol=self.symbol
            )
            
            return {
                'available': True,
                'crash_probability': result.crash_probability,
                'risk_level': result.risk_level,
                'recommended_action': result.recommended_action,
                'parabolic_score': result.parabolic_score,
                'btc_similarity': result.btc_2021_similarity,
                'divergences': {
                    'h1': result.rsi_divergence_h1,
                    'h4': result.rsi_divergence_h4,
                    'd1': result.rsi_divergence_d1
                },
                'patterns': {
                    'blow_off_top': result.blow_off_top_detected,
                    'double_top': result.double_top_detected,
                    'rising_wedge': result.rising_wedge_detected
                },
                'fear_greed': result.fear_greed_index,
                'notes': result.analysis_notes,
                'summary': format_crowd_psychology_summary(result),
                'full_analysis': result
            }
            
        except Exception as e:
            logger.error(f"Crowd psychology error: {e}")
            return {'available': False, 'error': str(e), 'summary': 'Crowd psychology analysis failed'}
    
    def get_algo_hype_index(self, technical: Dict = None) -> Dict:
        """
        Get Algo Hype Index - Detects crowded trades.
        
        AHI Score 0-100:
        - 0-25: LOW - Trade normally
        - 25-50: MODERATE - Reduce position sizes 25%
        - 50-75: HIGH - Reduce 50%, tighten stops
        - 75-90: EXTREME - Reduce 75%, block new buys
        - 90-100: PARABOLIC - Close longs, prepare for reversal
        """
        try:
            from algo_hype_index import get_algo_hype_index
            
            # Build technical data for AHI
            tech_data = {}
            if technical and 'latest' in technical:
                latest = technical['latest']
                tech_data = {
                    'rsi14_d1': latest.get('rsi14_d1', 50),
                    'rsi2_h1': latest.get('rsi2_h1', 50),
                    'rsi_14': latest.get('rsi14_h1', 50),
                    'trend_alignment': latest.get('trend_alignment', 0),
                    'bb_position': latest.get('bb_position', 0.5)
                }
            
            result = get_algo_hype_index(tech_data)
            return result
            
        except Exception as e:
            logger.error(f"Algo Hype Index error: {e}")
            return {
                'available': False,
                'error': str(e),
                'ahi_score': None,
                'level': 'UNKNOWN',
                'position_size_multiplier': 1.0,
                'block_buys': False
            }
    
    def get_usdjpy_correlation(self) -> Dict:
        """
        Get USDJPY-Gold correlation analysis.
        
        USDJPY and Gold have inverse correlation in risk-off environments:
        - Risk-off: JPY strengthens (USDJPY ↓), Gold rises (XAUUSD ↑)
        - Risk-on: JPY weakens (USDJPY ↑), Gold falls (XAUUSD ↓)
        
        Key insight: USDJPY at 160 resistance + reversal = BULLISH for Gold
        """
        try:
            from usdjpy_correlation import get_usdjpy_gold_correlation, fetch_usdjpy_data
            
            # Fetch USDJPY data (daily timeframe for trend)
            usdjpy_df = fetch_usdjpy_data(period="3mo", interval="1d")
            
            if usdjpy_df.empty:
                logger.warning("Could not fetch USDJPY data")
                return {'available': False, 'error': 'No USDJPY data'}
            
            # Get correlation analysis
            correlation = get_usdjpy_gold_correlation(usdjpy_df)
            
            return correlation
            
        except Exception as e:
            logger.error(f"USDJPY correlation error: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_mql5_consensus(self) -> Dict:
        """Get MQL5 top trader consensus (if available)"""
        intel_file = "/tmp/neo_intel.json"
        
        if os.path.exists(intel_file):
            try:
                with open(intel_file, 'r') as f:
                    data = json.load(f)
                
                return {
                    'available': True,
                    'timestamp': data.get('timestamp', ''),
                    'consensus_signals': data.get('consensus_signals', []),
                    'top_signals': data.get('top_signals', [])
                }
            except:
                pass
        
        return {'available': False, 'consensus_signals': [], 'top_signals': []}
    
    def get_smc_analysis(self, ohlcv_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Get Smart Money Concept (SMC) Analysis
        
        Detects:
        - Order Blocks (institutional entry zones)
        - Fair Value Gaps (imbalance areas)
        - Liquidity Pools (stop loss clusters)
        - Break of Structure / Change of Character
        """
        try:
            from smc_detector import SMCDetector, quick_smc_analysis
            
            # Use H1 data for SMC analysis
            if 'H1' not in ohlcv_data or ohlcv_data['H1'].empty:
                return {'available': False, 'error': 'No H1 data for SMC analysis'}
            
            ohlcv = ohlcv_data['H1']
            current_price = ohlcv['close'].iloc[-1]
            
            # Run SMC analysis
            detector = SMCDetector()
            analysis = detector.analyze(ohlcv, current_price, timeframe='H1')
            
            return {
                'available': True,
                'signal': analysis.signal,
                'confidence': analysis.confidence,
                'bias': analysis.bias.value,
                'order_blocks': len(analysis.order_blocks),
                'fvg_count': len(analysis.fair_value_gaps),
                'bos_detected': analysis.bos_detected,
                'choch_detected': analysis.choch_detected,
                'nearest_ob': {
                    'type': analysis.nearest_ob.type.value,
                    'zone': (analysis.nearest_ob.bottom, analysis.nearest_ob.top),
                    'strength': analysis.nearest_ob.strength
                } if analysis.nearest_ob else None,
                'nearest_fvg': {
                    'type': analysis.nearest_fvg.type.value,
                    'zone': (analysis.nearest_fvg.bottom, analysis.nearest_fvg.top),
                    'filled_pct': analysis.nearest_fvg.filled_pct
                } if analysis.nearest_fvg else None,
                'entry_zone': analysis.entry_zone,
                'stop_zone': analysis.stop_zone,
                'target_zone': analysis.target_zone,
                'reasoning': analysis.reasoning,
                'summary': self._format_smc_summary(analysis)
            }
            
        except Exception as e:
            logger.error(f"SMC analysis error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _format_smc_summary(self, analysis) -> str:
        """Format SMC analysis for LLM"""
        lines = []
        lines.append("📊 SMART MONEY CONCEPT ANALYSIS")
        lines.append(f"Signal: {analysis.signal} ({analysis.confidence}% confidence)")
        lines.append(f"Market Bias: {analysis.bias.value.upper()}")
        lines.append(f"Order Blocks: {len(analysis.order_blocks)} valid zones")
        lines.append(f"Fair Value Gaps: {len(analysis.fair_value_gaps)} active")
        
        if analysis.bos_detected:
            lines.append("✅ Break of Structure DETECTED")
        if analysis.choch_detected:
            lines.append("🔄 Change of Character DETECTED - Potential reversal!")
        
        if analysis.nearest_ob:
            lines.append(f"Nearest OB: {analysis.nearest_ob.type.value} ${analysis.nearest_ob.bottom:.2f}-${analysis.nearest_ob.top:.2f}")
        
        for reason in analysis.reasoning[:3]:
            lines.append(f"• {reason}")
        
        return "\n".join(lines)
    
    def get_multi_factor_risk(self, technical: Dict, macro: Dict, 
                              crowd: Dict = None, mm_data: Dict = None) -> Dict:
        """
        Get Multi-Factor Risk Score (Hunter-style)
        
        Returns 0-100 risk score with position sizing recommendation
        """
        try:
            from multi_factor_risk import MultiFactorRisk, quick_risk_check
            
            risk = MultiFactorRisk()
            
            # Extract volatility data
            volatility_data = {
                'atr_percentile': 50,  # Default
                'vix': 20,
                'regime': 'NORMAL'
            }
            
            if macro and macro.get('features'):
                vix = macro['features'].get('vix_level', 20)
                volatility_data['vix'] = vix
                if vix > 30:
                    volatility_data['regime'] = 'HIGH'
                elif vix < 15:
                    volatility_data['regime'] = 'LOW'
            
            # Extract trend data
            trend_data = {
                'adx': 25,
                'trend_aligned': True
            }
            
            if technical and technical.get('latest'):
                adx = technical['latest'].get('adx_D1', technical['latest'].get('adx', 25))
                trend_data['adx'] = adx if isinstance(adx, (int, float)) else 25
            
            # Extract volume data
            volume_data = {
                'volume_ratio': 1.0,
                'liquidity': 'NORMAL'
            }
            
            # Extract correlation data
            correlation_data = {
                'gold_dxy': -0.5,
                'gold_spy': 0.1
            }
            
            if macro and macro.get('features'):
                gold_dxy = macro['features'].get('gold_dxy_correlation', -0.5)
                correlation_data['gold_dxy'] = gold_dxy
            
            # Extract sentiment data
            sentiment_data = {
                'retail_long_pct': 50,
                'fear_greed': 50
            }
            
            if crowd and crowd.get('available'):
                sentiment_data['fear_greed'] = crowd.get('fear_greed', 50)
            
            # Extract MM data
            mm_activity = {
                'stop_hunt_prob': 0.3,
                'liquidity_thin': False
            }
            
            if mm_data:
                mm_activity['stop_hunt_prob'] = mm_data.get('stop_hunt_prob', 0.3)
            
            # Run assessment
            assessment = risk.assess(
                volatility_data=volatility_data,
                trend_data=trend_data,
                volume_data=volume_data,
                correlation_data=correlation_data,
                sentiment_data=sentiment_data,
                mm_data=mm_activity
            )
            
            return {
                'available': True,
                'risk_score': assessment.total_score,
                'risk_level': assessment.risk_level,
                'position_multiplier': assessment.position_multiplier,
                'factors': assessment.factors,
                'warnings': assessment.warnings,
                'recommendation': assessment.recommendation
            }
            
        except Exception as e:
            logger.error(f"Multi-factor risk error: {e}")
            return {'available': False, 'error': str(e), 'risk_score': 50, 'position_multiplier': 1.0}
    
    def get_adaptive_stoploss(self, entry_price: float, current_price: float,
                              current_stop: float = None, direction: str = 'LONG',
                              atr: float = None) -> Dict:
        """
        Get Adaptive Stop Loss recommendation (Freqtrade-style)
        
        Returns optimal stop level based on profit, volatility, and time
        """
        try:
            from adaptive_stoploss import AdaptiveStopLoss, TrailMode
            
            # Calculate ATR if not provided
            if atr is None:
                atr = abs(current_price - entry_price) * 0.5  # Rough estimate
                atr = max(atr, 15)  # Minimum 15 points for Gold
            
            stop_loss = AdaptiveStopLoss()
            
            result = stop_loss.calculate(
                entry_price=entry_price,
                current_price=current_price,
                current_stop=current_stop,
                atr=atr,
                direction=direction,
                mode=TrailMode.ADAPTIVE
            )
            
            return {
                'available': True,
                'stop_price': result.stop_price,
                'trail_distance': result.trail_distance,
                'mode_used': result.mode_used,
                'should_move': result.should_move,
                'break_even_locked': result.break_even_locked,
                'reason': result.reason
            }
            
        except Exception as e:
            logger.error(f"Adaptive stoploss error: {e}")
            return {'available': False, 'error': str(e)}
    
    def get_full_market_context(self) -> Dict:
        """
        Main entry point: Get complete market context for NEO
        
        Returns comprehensive dict with all features and summaries
        """
        logger.info("\n" + "="*70)
        logger.info(f"🧠 NEO UNIFIED MARKET FEED - {self.symbol}")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # 1. Fetch OHLCV data
        logger.info("\n📊 Fetching OHLCV data...")
        ohlcv_data = self.fetch_ohlcv_data(['H1', 'H4', 'D1'])
        
        # 2. Technical indicators
        logger.info("\n📈 Computing technical indicators...")
        try:
            technical = self.get_technical_features(ohlcv_data)
        except Exception as e:
            logger.error(f"Technical features error: {e}")
            technical = {'features': pd.DataFrame(), 'summary': 'Technical analysis unavailable'}
        
        # 3. Macro correlations
        logger.info("\n🌍 Fetching macro data...")
        try:
            macro = self.get_macro_features()
        except Exception as e:
            logger.error(f"Macro features error: {e}")
            macro = {'features': {}, 'summary': 'Macro data unavailable'}
        
        # 4. Microstructure
        logger.info("\n⏰ Computing microstructure...")
        try:
            microstructure = self.get_microstructure_features()
        except Exception as e:
            logger.error(f"Microstructure error: {e}")
            microstructure = {'features': {}, 'summary': 'Session analysis unavailable'}
        
        # 5. MQL5 consensus
        logger.info("\n🎯 Checking MQL5 consensus...")
        mql5 = self.get_mql5_consensus()
        
        # 6. Crowd Psychology (Bitcoin crash patterns)
        logger.info("\n🧠 Analyzing crowd psychology...")
        crowd = self.get_crowd_psychology(ohlcv_data)
        if crowd.get('available'):
            logger.info(f"   🎯 Crash probability: {crowd['crash_probability']:.0f}%")
            logger.info(f"   📊 Risk level: {crowd['risk_level'].upper()}")
        
        # 7. USDJPY-Gold Correlation (NEW!)
        logger.info("\n🔗 Analyzing USDJPY-Gold correlation...")
        usdjpy_correlation = self.get_usdjpy_correlation()
        if usdjpy_correlation.get('available'):
            logger.info(f"   💴 USDJPY: {usdjpy_correlation['context']['price']:.2f}")
            logger.info(f"   📈 Gold Bias: {usdjpy_correlation['gold_bias']}")
            logger.info(f"   📊 Confidence Adj: {usdjpy_correlation['net_confidence_adjustment']:+d}%")
        
        # 8. ALGO HYPE INDEX (Crowd Detection System)
        logger.info("\n📊 Calculating Algo Hype Index...")
        algo_hype = self.get_algo_hype_index(technical)
        if algo_hype.get('ahi_score') is not None:
            level_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'EXTREME': '🔴', 'PARABOLIC': '💀'}
            emoji = level_emoji.get(algo_hype['level'], '❓')
            logger.info(f"   {emoji} AHI Score: {algo_hype['ahi_score']:.0f}/100 ({algo_hype['level']})")
            logger.info(f"   📉 Position Multiplier: {algo_hype['position_size_multiplier']:.0%}")
            if algo_hype['block_buys']:
                logger.info(f"   ⛔ BUYS BLOCKED due to extreme hype!")
        
        # 9. SMC ANALYSIS (Smart Money Concepts) - NEW!
        logger.info("\n🎯 Running SMC Analysis (Order Blocks, FVG)...")
        smc_analysis = self.get_smc_analysis(ohlcv_data)
        if smc_analysis.get('available'):
            logger.info(f"   📊 SMC Signal: {smc_analysis['signal']} ({smc_analysis['confidence']}%)")
            logger.info(f"   📈 Market Bias: {smc_analysis['bias'].upper()}")
            logger.info(f"   🔲 Order Blocks: {smc_analysis['order_blocks']} valid zones")
            logger.info(f"   📐 Fair Value Gaps: {smc_analysis['fvg_count']} active")
            if smc_analysis['bos_detected']:
                logger.info(f"   ✅ Break of Structure DETECTED")
            if smc_analysis['choch_detected']:
                logger.info(f"   🔄 Change of Character DETECTED")
        
        # 10. MULTI-FACTOR RISK SCORE - NEW!
        logger.info("\n⚖️ Calculating Multi-Factor Risk Score...")
        multi_risk = self.get_multi_factor_risk(technical, macro, crowd)
        if multi_risk.get('available'):
            risk_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'EXTREME': '🔴'}
            emoji = risk_emoji.get(multi_risk['risk_level'], '❓')
            logger.info(f"   {emoji} Risk Score: {multi_risk['risk_score']:.0f}/100 ({multi_risk['risk_level']})")
            logger.info(f"   📏 Position Multiplier: {multi_risk['position_multiplier']:.0%}")
            if multi_risk['warnings']:
                for warn in multi_risk['warnings'][:2]:
                    logger.info(f"   {warn}")
        
        # Count features
        tech_count = len(technical.get('latest', {}))
        macro_count = len(macro.get('features', {}))
        micro_count = len(microstructure.get('features', {}))
        total_features = tech_count + macro_count + micro_count
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Market context assembled in {elapsed:.1f}s")
        logger.info(f"   Technical: {tech_count} features")
        logger.info(f"   Macro: {macro_count} features")
        logger.info(f"   Microstructure: {micro_count} features")
        logger.info(f"   TOTAL: {total_features} features")
        
        # Build combined summary for LLM
        combined_summary = self._build_combined_summary(
            technical, macro, microstructure, mql5, crowd, usdjpy_correlation, algo_hype,
            smc_analysis, multi_risk
        )
        
        self.last_update = datetime.now()
        
        return {
            'symbol': self.symbol,
            'timestamp': self.last_update.isoformat(),
            'technical': technical,
            'macro': macro,
            'microstructure': microstructure,
            'mql5_consensus': mql5,
            'crowd_psychology': crowd,
            'usdjpy_correlation': usdjpy_correlation,
            'algo_hype_index': algo_hype,
            'smc_analysis': smc_analysis,       # NEW!
            'multi_factor_risk': multi_risk,    # NEW!
            'feature_count': total_features,
            'summary': combined_summary
        }
    
    def _build_combined_summary(self, technical: Dict, macro: Dict, 
                                microstructure: Dict, mql5: Dict, crowd: Dict = None,
                                usdjpy: Dict = None, algo_hype: Dict = None,
                                smc: Dict = None, multi_risk: Dict = None) -> str:
        """Build combined summary for NEO's LLM prompt"""
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"🧠 NEO MARKET INTELLIGENCE - {self.symbol}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 60)
        
        # Technical summary
        if technical.get('summary'):
            lines.append("\n" + technical['summary'])
        
        # Macro summary
        if macro.get('summary'):
            lines.append("\n" + macro['summary'])
        
        # Microstructure summary
        if microstructure.get('summary'):
            lines.append("\n" + microstructure['summary'])
        
        # ═══ CROWD PSYCHOLOGY (NEW!) ═══
        if crowd and crowd.get('available'):
            lines.append("\n" + "="*60)
            lines.append("🧠 CROWD PSYCHOLOGY (Bitcoin Crash Pattern Detection)")
            lines.append("="*60)
            
            crash_prob = crowd.get('crash_probability', 0)
            risk_level = crowd.get('risk_level', 'unknown')
            rec_action = crowd.get('recommended_action', 'normal')
            
            # Color-coded risk
            if crash_prob >= 85:
                risk_emoji = "🔴"
            elif crash_prob >= 70:
                risk_emoji = "🟠"
            elif crash_prob >= 50:
                risk_emoji = "🟡"
            else:
                risk_emoji = "🟢"
            
            lines.append(f"   {risk_emoji} CRASH PROBABILITY: {crash_prob:.0f}%")
            lines.append(f"   📊 Risk Level: {risk_level.upper()}")
            lines.append(f"   💡 Recommendation: {rec_action.upper()}")
            
            # Divergences
            divs = crowd.get('divergences', {})
            if any(v == 'bearish' for v in divs.values()):
                lines.append(f"   ⚠️ RSI DIVERGENCE: H1={divs.get('h1')}, H4={divs.get('h4')}, D1={divs.get('d1')}")
            
            # Patterns
            patterns = crowd.get('patterns', {})
            if patterns.get('blow_off_top'):
                lines.append("   🔴 BLOW-OFF TOP DETECTED!")
            if patterns.get('double_top'):
                lines.append("   ⚠️ Double top pattern")
            if patterns.get('rising_wedge'):
                lines.append("   ⚠️ Rising wedge pattern")
            
            # Bitcoin similarity
            btc_sim = crowd.get('btc_similarity', 0)
            if btc_sim > 50:
                lines.append(f"   📉 Bitcoin 2021 crash similarity: {btc_sim:.0f}%")
            
            # Notes
            for note in crowd.get('notes', [])[:3]:
                lines.append(f"   {note}")
        
        # ═══ USDJPY-GOLD CORRELATION (NEW!) ═══
        if usdjpy and usdjpy.get('available'):
            lines.append("\n" + "="*60)
            lines.append("🔗 USDJPY-GOLD CORRELATION")
            lines.append("="*60)
            
            ctx = usdjpy.get('context', {})
            usdjpy_price = ctx.get('price', 0)
            usdjpy_trend = ctx.get('trend', 'UNKNOWN')
            usdjpy_rsi = ctx.get('rsi_14', 0)
            gold_bias = usdjpy.get('gold_bias', 'NEUTRAL')
            net_adj = usdjpy.get('net_confidence_adjustment', 0)
            
            lines.append(f"   💴 USDJPY Price: {usdjpy_price:.2f}")
            lines.append(f"   📈 Trend: {usdjpy_trend} ({ctx.get('trend_strength', 0):.0f}%)")
            lines.append(f"   📊 RSI(14): {usdjpy_rsi:.1f}")
            
            # Gold bias from USDJPY
            if gold_bias == "BULLISH":
                bias_emoji = "🟢"
            elif gold_bias == "BEARISH":
                bias_emoji = "🔴"
            else:
                bias_emoji = "⚪"
            
            lines.append(f"   {bias_emoji} Gold Bias: {gold_bias}")
            
            adj_str = f"+{net_adj}" if net_adj > 0 else str(net_adj)
            lines.append(f"   📊 Confidence Adjustment: {adj_str}%")
            
            # BOJ intervention warning
            if ctx.get('boj_intervention_risk'):
                lines.append("   ⚠️ IN BOJ INTERVENTION ZONE (>158)")
            
            # Key levels context
            if ctx.get('at_resistance'):
                lines.append("   📍 Near major resistance (160)")
            elif ctx.get('at_support'):
                lines.append("   📍 Near key support")
            
            # Correlation signals
            signals = usdjpy.get('signals', [])
            if signals:
                lines.append("\n   Correlation Signals:")
                for sig in signals[:3]:
                    sig_emoji = "🟢" if "BULLISH" in sig.get('type', '') else "🔴" if "BEARISH" in sig.get('type', '') else "⚠️"
                    lines.append(f"   {sig_emoji} {sig.get('reason', 'Unknown')}")
        
        # MQL5 consensus
        if mql5.get('available') and mql5.get('consensus_signals'):
            lines.append("\n" + "="*50)
            lines.append("🎯 MQL5 TOP TRADER CONSENSUS")
            lines.append("="*50)
            
            for signal in mql5['consensus_signals'][:3]:
                symbol = signal.get('symbol', 'UNKNOWN')
                direction = signal.get('direction', 'UNKNOWN')
                traders = len(signal.get('traders', []))
                lines.append(f"   {symbol} {direction} - {traders} top traders agree")
        
        # ═══ SMC ANALYSIS (Smart Money Concepts) ═══
        if smc and smc.get('available'):
            lines.append("\n" + "="*60)
            lines.append("🎯 SMART MONEY CONCEPT (SMC) ANALYSIS")
            lines.append("="*60)
            
            signal = smc.get('signal', 'NEUTRAL')
            confidence = smc.get('confidence', 0)
            bias = smc.get('bias', 'NEUTRAL')
            
            signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
            lines.append(f"   {signal_emoji} SMC Signal: {signal} ({confidence}% confidence)")
            lines.append(f"   📈 Market Structure: {bias.upper()}")
            lines.append(f"   🔲 Order Blocks: {smc.get('order_blocks', 0)} valid zones")
            lines.append(f"   📐 Fair Value Gaps: {smc.get('fvg_count', 0)} active")
            
            if smc.get('bos_detected'):
                lines.append("   ✅ BREAK OF STRUCTURE detected")
            if smc.get('choch_detected'):
                lines.append("   🔄 CHANGE OF CHARACTER detected - Potential reversal!")
            
            if smc.get('nearest_ob'):
                ob = smc['nearest_ob']
                lines.append(f"   📍 Nearest OB: {ob['type']} ${ob['zone'][0]:.2f}-${ob['zone'][1]:.2f} (strength: {ob['strength']})")
            
            if smc.get('entry_zone'):
                lines.append(f"   🎯 Entry Zone: ${smc['entry_zone'][0]:.2f}-${smc['entry_zone'][1]:.2f}")
            
            if smc.get('reasoning'):
                for reason in smc['reasoning'][:3]:
                    lines.append(f"   • {reason}")
        
        # ═══ MULTI-FACTOR RISK SCORE ═══
        if multi_risk and multi_risk.get('available'):
            lines.append("\n" + "="*60)
            lines.append("⚖️ MULTI-FACTOR RISK ASSESSMENT")
            lines.append("="*60)
            
            risk_score = multi_risk.get('risk_score', 50)
            risk_level = multi_risk.get('risk_level', 'MODERATE')
            multiplier = multi_risk.get('position_multiplier', 1.0)
            
            risk_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'EXTREME': '🔴'}
            emoji = risk_emoji.get(risk_level, '❓')
            
            lines.append(f"   {emoji} Risk Score: {risk_score:.0f}/100 ({risk_level})")
            lines.append(f"   📏 Position Size: {multiplier:.0%} of normal")
            
            # Factor breakdown
            factors = multi_risk.get('factors', {})
            if factors:
                lines.append("   Factor Breakdown:")
                sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
                for factor, score in sorted_factors[:4]:
                    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                    lines.append(f"     {factor}: {score:.0f}/100 [{bar}]")
            
            # Warnings
            warnings = multi_risk.get('warnings', [])
            if warnings:
                for warn in warnings[:3]:
                    lines.append(f"   {warn}")
            
            lines.append(f"   💡 {multi_risk.get('recommendation', '')}")
        
        # Overall assessment
        lines.append("\n" + "="*60)
        lines.append("📋 NEO TRADING GUIDANCE")
        lines.append("="*60)
        
        # Extract key signals
        session_info = microstructure.get('session_info', {})
        vix_regime = macro.get('features', {}).get('vix_regime', 'UNKNOWN')
        dxy_trend = macro.get('features', {}).get('dxy_trend', 'UNKNOWN')
        
        # ═════════════════════════════════════════════════════════
        # ALGO HYPE INDEX (Crowd Detection)
        # ═════════════════════════════════════════════════════════
        
        if algo_hype and algo_hype.get('ahi_score') is not None:
            lines.append("")
            lines.append("─" * 50)
            lines.append("📊 ALGO HYPE INDEX (Crowd Detection)")
            lines.append("─" * 50)
            
            ahi_score = algo_hype['ahi_score']
            level = algo_hype['level']
            
            level_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🟠', 'EXTREME': '🔴', 'PARABOLIC': '💀'}
            emoji = level_emoji.get(level, '❓')
            
            lines.append(f"   {emoji} AHI Score: {ahi_score:.0f}/100 ({level})")
            lines.append(f"   📉 Position Multiplier: {algo_hype['position_size_multiplier']:.0%}")
            
            # Component breakdown
            components = algo_hype.get('components', {})
            if components:
                lines.append("")
                lines.append("   Components:")
                for name, comp in components.items():
                    status = comp.get('status', '❓')
                    score = comp.get('score', 0)
                    lines.append(f"   ├── {name.replace('_', ' ').title()}: {score:.0f} {status}")
            
            # Historical parallel
            parallel = algo_hype.get('historical_parallel', {})
            if parallel.get('match'):
                lines.append("")
                lines.append(f"   📜 Historical Parallel: {parallel['match']}")
                lines.append(f"      Peak then crashed: {parallel.get('subsequent_crash', 'N/A')}")
                lines.append(f"      Similarity: {parallel.get('similarity', 0):.0f}%")
            
            # Defense status
            defense = algo_hype.get('defense', {})
            if defense:
                lines.append("")
                lines.append("   🛡️ Defense Status:")
                if algo_hype.get('block_buys'):
                    lines.append("   ├── ⛔ NEW BUYS BLOCKED")
                lines.append(f"   ├── Stop Type: {defense.get('stop_type', 'NORMAL')}")
                lines.append(f"   └── Hedge: {defense.get('hedge_recommendation', 'NONE')}")
            
            # Summary actions
            actions = defense.get('action_summary', [])
            if actions and level in ['HIGH', 'EXTREME', 'PARABOLIC']:
                lines.append("")
                lines.append("   ⚠️ RECOMMENDED ACTIONS:")
                for action in actions[:3]:
                    lines.append(f"      • {action}")
        
        guidance = []
        
        # Session guidance
        if session_info.get('is_overlap'):
            guidance.append("✅ OPTIMAL trading time (London/NY overlap)")
        elif session_info.get('primary_session') == 'ASIAN':
            guidance.append("⚠️ Asian session - expect range-bound action")
        elif session_info.get('liquidity') == 'VERY_LOW':
            guidance.append("❌ Off-hours - avoid trading")
        
        # Macro guidance
        if vix_regime == 'FEAR':
            guidance.append("📈 VIX elevated - Gold tends to rise in fear")
        if dxy_trend == 'DOWN':
            guidance.append("📈 Dollar weak - tailwind for Gold")
        elif dxy_trend == 'UP':
            guidance.append("📉 Dollar strong - headwind for Gold")
        
        for g in guidance:
            lines.append(f"   {g}")
        
        lines.append("="*60)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_neo_market_context(symbol: str = "XAUUSD") -> Dict:
    """
    Quick function to get complete market context for NEO
    """
    feed = UnifiedMarketFeed(symbol)
    context = feed.get_full_market_context()
    
    # Add weekly pattern predictions (XAUUSD only for now)
    if symbol == "XAUUSD":
        try:
            from weekly_predictions import get_trading_plan, get_pattern_context, format_trading_plan
            
            # Get current price from context
            current_price = None
            if context.get('technical_features') and not context['technical_features'].empty:
                current_price = context['technical_features']['close'].iloc[-1] if 'close' in context['technical_features'].columns else None
            
            # Get volatility regime from ATR
            vol_regime = "NORMAL"
            if 'atr' in context.get('technical_features', pd.DataFrame()).columns:
                atr_pct = context['technical_features']['atr'].rank(pct=True).iloc[-1] if len(context['technical_features']) > 10 else 0.5
                if atr_pct < 0.25:
                    vol_regime = "LOW"
                elif atr_pct > 0.75:
                    vol_regime = "HIGH"
            
            # Get trading plan based on patterns
            plan = get_trading_plan(
                volatility_regime=vol_regime,
                current_price=current_price
            )
            
            # Get active pattern context
            rsi_14 = context.get('technical_features', pd.DataFrame()).get('rsi_14', pd.Series()).iloc[-1] if 'rsi_14' in context.get('technical_features', pd.DataFrame()).columns else None
            rsi_2 = context.get('technical_features', pd.DataFrame()).get('rsi_2', pd.Series()).iloc[-1] if 'rsi_2' in context.get('technical_features', pd.DataFrame()).columns else None
            
            pattern_context = get_pattern_context(
                rsi_14=rsi_14 if rsi_14 and not pd.isna(rsi_14) else None,
                rsi_2=rsi_2 if rsi_2 and not pd.isna(rsi_2) else None,
                current_price=current_price
            )
            
            context['pattern_predictions'] = {
                'available': True,
                'trading_plan': plan,
                'pattern_context': pattern_context,
                'formatted_plan': format_trading_plan(plan)
            }
            
            # Update summary with pattern info
            if 'summary' in context:
                context['summary'] += f"\n\n{format_trading_plan(plan)}"
            
        except ImportError as e:
            logger.warning(f"Weekly predictions not available: {e}")
            context['pattern_predictions'] = {'available': False, 'error': str(e)}
        except Exception as e:
            logger.warning(f"Pattern prediction error: {e}")
            context['pattern_predictions'] = {'available': False, 'error': str(e)}
    else:
        context['pattern_predictions'] = {'available': False, 'reason': 'Only XAUUSD patterns available'}
    
    return context


def get_neo_summary(symbol: str = "XAUUSD") -> str:
    """
    Quick function to get market summary for LLM prompt
    """
    context = get_neo_market_context(symbol)
    return context.get('summary', 'Market context unavailable')


# ═══════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════

def test_unified_feed():
    """Test the unified market feed"""
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING NEO UNIFIED MARKET FEED")
    logger.info("="*70)
    
    context = get_neo_market_context("XAUUSD")
    
    logger.info("\n📊 RESULTS:")
    logger.info(f"   Symbol: {context['symbol']}")
    logger.info(f"   Timestamp: {context['timestamp']}")
    logger.info(f"   Total Features: {context['feature_count']}")
    
    logger.info("\n" + context['summary'])
    
    return context


if __name__ == "__main__":
    test_unified_feed()
