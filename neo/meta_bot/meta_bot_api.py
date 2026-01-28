"""
Crellastein Meta Bot API
FastAPI endpoints for the weighted indicator ensemble system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
from typing import Dict, Optional

from .crellastein_meta import CrellaSteinMetaBot, get_xauusd_signal, get_iren_signal, get_clsk_signal, get_cifr_signal
from .aggressive_dca import AggressiveDCAManager, get_xauusd_dca_status, get_iren_dca_status
from .entry_calculator import EntryCalculator
from .volume_analyzer import VolumeAnalyzer
import sys
sys.path.insert(0, '/home/jbot/trading_ai/neo')
from pattern_detector import PatternDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaBotAPI")

app = FastAPI(
    title="Crellastein Meta Bot API",
    description="Weighted Indicator Ensemble Trading System for XAUUSD and IREN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bots
xauusd_bot = CrellaSteinMetaBot("XAUUSD")
iren_bot = CrellaSteinMetaBot("IREN")
clsk_bot = CrellaSteinMetaBot("CLSK")
cifr_bot = CrellaSteinMetaBot("CIFR")

xauusd_dca = AggressiveDCAManager("XAUUSD")
iren_dca = AggressiveDCAManager("IREN")
clsk_dca = AggressiveDCAManager("CLSK")
cifr_dca = AggressiveDCAManager("CIFR")


@app.get("/")
async def root():
    return {
        "service": "Crellastein Meta Bot",
        "version": "1.1.0",
        "status": "ACTIVE",
        "description": "Weighted Indicator Ensemble for XAUUSD, IREN, CLSK & CIFR",
        "supported_symbols": ["XAUUSD", "IREN", "CLSK", "CIFR"],
        "endpoints": {
            "signals": [
                "/api/meta/xauusd/signal",
                "/api/meta/iren/signal",
                "/api/meta/clsk/signal",
                "/api/meta/cifr/signal",
                "/api/meta/{symbol}/composite"
            ],
            "dca": [
                "/api/meta/{symbol}/dca/status",
                "/api/meta/{symbol}/dca/open",
                "/api/meta/{symbol}/dca/check",
                "/api/meta/{symbol}/dca/execute",
                "/api/meta/{symbol}/dca/close"
            ],
            "config": [
                "/api/meta/{symbol}/weights",
                "/api/meta/{symbol}/config"
            ]
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bots": ["XAUUSD", "IREN", "CLSK", "CIFR"]
    }


# ============== SIGNAL ENDPOINTS ==============

@app.get("/api/meta/xauusd/signal")
async def xauusd_signal():
    """Get XAUUSD composite signal from Meta Bot"""
    try:
        signal = xauusd_bot.get_signal_summary()
        return signal
    except Exception as e:
        logger.error(f"Error getting XAUUSD signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/iren/signal")
async def iren_signal():
    """Get IREN composite signal from Meta Bot"""
    try:
        signal = iren_bot.get_signal_summary()
        return signal
    except Exception as e:
        logger.error(f"Error getting IREN signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/clsk/signal")
async def clsk_signal():
    """Get CLSK composite signal from Meta Bot"""
    try:
        signal = clsk_bot.get_signal_summary()
        return signal
    except Exception as e:
        logger.error(f"Error getting CLSK signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/cifr/signal")
async def cifr_signal():
    """Get CIFR composite signal from Meta Bot"""
    try:
        signal = cifr_bot.get_signal_summary()
        return signal
    except Exception as e:
        logger.error(f"Error getting CIFR signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/{symbol}/composite")
async def get_composite_signal(symbol: str):
    """Get full composite signal with all indicator details"""
    symbol = symbol.upper()
    try:
        if symbol == "XAUUSD":
            signal = xauusd_bot.calculate_composite_signal()
        elif symbol == "IREN":
            signal = iren_bot.calculate_composite_signal()
        elif symbol == "CLSK":
            signal = clsk_bot.calculate_composite_signal()
        elif symbol == "CIFR":
            signal = cifr_bot.calculate_composite_signal()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}. Supported: XAUUSD, IREN, CLSK, CIFR")
        
        from dataclasses import asdict
        return asdict(signal)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting composite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/{symbol}/weights")
async def get_weights(symbol: str):
    """Get indicator weights for a symbol"""
    symbol = symbol.upper()
    try:
        if symbol == "XAUUSD":
            return {"symbol": symbol, "weights": xauusd_bot.weights}
        elif symbol == "IREN":
            return {"symbol": symbol, "weights": iren_bot.weights}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/meta/{symbol}/weights")
async def update_weights(symbol: str, weights: Dict[str, float]):
    """Update indicator weights"""
    symbol = symbol.upper()
    try:
        if symbol == "XAUUSD":
            xauusd_bot.weights.update(weights)
            xauusd_bot._save_weights(xauusd_bot.weights)
            return {"status": "updated", "weights": xauusd_bot.weights}
        elif symbol == "IREN":
            iren_bot.weights.update(weights)
            iren_bot._save_weights(iren_bot.weights)
            return {"status": "updated", "weights": iren_bot.weights}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/{symbol}/config")
async def get_config(symbol: str):
    """Get asset configuration"""
    symbol = symbol.upper()
    try:
        if symbol == "XAUUSD":
            return {"symbol": symbol, "config": xauusd_bot.config}
        elif symbol == "IREN":
            return {"symbol": symbol, "config": iren_bot.config}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== DCA ENDPOINTS ==============

@app.get("/api/meta/{symbol}/dca/status")
async def dca_status(symbol: str):
    """Get DCA position status"""
    symbol = symbol.upper()
    try:
        if symbol == "XAUUSD":
            return xauusd_dca.get_position_status()
        elif symbol == "IREN":
            return iren_dca.get_position_status()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meta/{symbol}/dca/open")
async def dca_open(symbol: str, direction: str = "LONG"):
    """Open initial DCA position"""
    symbol = symbol.upper()
    direction = direction.upper()
    
    try:
        # Get composite signal first
        if symbol == "XAUUSD":
            signal = xauusd_bot.get_signal_summary()
            result = xauusd_dca.open_initial_position(direction, signal['score'] / 100)
        elif symbol == "IREN":
            signal = iren_bot.get_signal_summary()
            result = iren_dca.open_initial_position(direction, signal['score'] / 100)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error opening DCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meta/{symbol}/dca/check")
async def dca_check(symbol: str):
    """Check if DCA should trigger"""
    symbol = symbol.upper()
    
    try:
        if symbol == "XAUUSD":
            signal = xauusd_bot.get_signal_summary()
            result = xauusd_dca.check_dca_trigger(signal['score'] / 100)
        elif symbol == "IREN":
            signal = iren_bot.get_signal_summary()
            result = iren_dca.check_dca_trigger(signal['score'] / 100)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
        
        result['composite_signal'] = signal
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking DCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meta/{symbol}/dca/execute")
async def dca_execute(symbol: str):
    """Execute DCA if triggered"""
    symbol = symbol.upper()
    
    try:
        if symbol == "XAUUSD":
            signal = xauusd_bot.get_signal_summary()
            result = xauusd_dca.execute_dca(signal['score'] / 100)
        elif symbol == "IREN":
            signal = iren_bot.get_signal_summary()
            result = iren_dca.execute_dca(signal['score'] / 100)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing DCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meta/{symbol}/dca/tp-check")
async def dca_tp_check(symbol: str):
    """Check and execute take profit levels"""
    symbol = symbol.upper()
    
    try:
        if symbol == "XAUUSD":
            result = xauusd_dca.check_take_profit()
        elif symbol == "IREN":
            result = iren_dca.check_take_profit()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking TP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meta/{symbol}/dca/close")
async def dca_close(symbol: str, reason: str = "Manual close"):
    """Close entire DCA position"""
    symbol = symbol.upper()
    
    try:
        if symbol == "XAUUSD":
            result = xauusd_dca.close_position(reason)
        elif symbol == "IREN":
            result = iren_dca.close_position(reason)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing DCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== COMBINED DASHBOARD ==============

@app.get("/api/meta/dashboard")
async def dashboard():
    """Get combined dashboard for both assets"""
    try:
        xauusd_sig = xauusd_bot.get_signal_summary()
        iren_sig = iren_bot.get_signal_summary()
        
        xauusd_dca_stat = xauusd_dca.get_position_status()
        iren_dca_stat = iren_dca.get_position_status()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": {
                "XAUUSD": xauusd_sig,
                "IREN": iren_sig
            },
            "dca_positions": {
                "XAUUSD": xauusd_dca_stat,
                "IREN": iren_dca_stat
            },
            "recommendations": {
                "XAUUSD": _get_recommendation("XAUUSD", xauusd_sig, xauusd_dca_stat),
                "IREN": _get_recommendation("IREN", iren_sig, iren_dca_stat)
            }
        }
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_recommendation(symbol: str, signal: Dict, dca_status: Dict) -> str:
    """Generate trading recommendation"""
    score = signal['score']
    has_position = dca_status.get('has_position', False)
    
    if score >= 75 and not has_position:
        return f"STRONG BUY: Open {symbol} position with full size"
    elif score >= 60 and not has_position:
        return f"BUY: Open {symbol} position with normal size"
    elif score >= 60 and has_position:
        if dca_status.get('dca_levels', 0) < 5:
            return f"HOLD: Consider DCA if price pulls back"
        return "HOLD: Max DCA levels reached"
    elif score <= 40:
        if has_position:
            return "CAUTION: Signal weakening, monitor closely"
        return "WAIT: Signal not strong enough"
    else:
        return "NEUTRAL: No action recommended"


# ============== GHOST COMMANDER INTEGRATION ==============

@app.get("/api/meta/ghost/xauusd")
async def ghost_xauusd_signal():
    """Signal endpoint for Ghost Commander (XAUUSD) with optimal entry, TP, patterns, and volume"""
    try:
        signal = xauusd_bot.get_signal_summary()
        
        # Map to Ghost format
        action = "HOLD"
        if signal['signal'] in ["STRONG_BUY", "BUY"]:
            action = "BUY"
        elif signal['signal'] in ["STRONG_SELL", "SELL"]:
            action = "SELL"
        
        # Calculate optimal entry and TP targets
        entry_calc = EntryCalculator("XAUUSD")
        entry_data = entry_calc.get_optimal_entry()
        tp_data = entry_calc.get_target_tp(entry_data["optimal_entry"])
        
        # Analyze volume
        vol_analyzer = VolumeAnalyzer("XAUUSD")
        volume_data = vol_analyzer.analyze_volume()
        
        # Detect chart patterns - get data first
        import yfinance as yf
        try:
            ticker = yf.Ticker("GC=F")
            pattern_df = ticker.history(period="30d", interval="1h")
            pattern_detector = PatternDetector("XAUUSD")
            patterns = pattern_detector.analyze(pattern_df) if not pattern_df.empty else []
        except Exception as pe:
            logger.warning(f"Pattern detection error: {pe}")
            patterns = []
        
        # Convert patterns to JSON-serializable format
        patterns_list = []
        pattern_bullish = 0
        pattern_bearish = 0
        for p in patterns:
            patterns_list.append({
                "type": p.pattern_type.value,
                "confidence": float(p.confidence),
                "direction": p.direction,
                "description": p.description,
            })
            if p.direction == "BUY":
                pattern_bullish += 1
            elif p.direction == "SELL":
                pattern_bearish += 1
        
        # Pattern summary
        if pattern_bullish > pattern_bearish:
            pattern_signal = "BULLISH"
        elif pattern_bearish > pattern_bullish:
            pattern_signal = "BEARISH"
        else:
            pattern_signal = "NEUTRAL"
        
        # Combined visual confirmation
        # Special case: Low volume during bull/bear flag CONSOLIDATION is normal and expected
        has_bull_flag_forming = any("FORMING" in p.get("description", "") for p in patterns_list if "bull_flag" in p.get("type", "").lower())
        has_bear_flag_forming = any("FORMING" in p.get("description", "") for p in patterns_list if "bear_flag" in p.get("type", "").lower())
        
        visual_confirms_signal = False
        
        # Bull flag forming = strong visual confirmation regardless of action
        if has_bull_flag_forming and pattern_signal == "BULLISH":
            visual_confirms_signal = True  # Bull flag + bullish patterns = confirmed buy setup
        elif has_bear_flag_forming and pattern_signal == "BEARISH":
            visual_confirms_signal = True  # Bear flag + bearish patterns = confirmed sell setup
        elif action == "BUY":
            visual_confirms_signal = (
                (pattern_signal == "BULLISH" or pattern_signal == "NEUTRAL") and
                (volume_data["volume_signal"] == "BULLISH" or volume_data["volume_signal"] == "NEUTRAL")
            )
        elif action == "SELL":
            visual_confirms_signal = (
                (pattern_signal == "BEARISH" or pattern_signal == "NEUTRAL") and
                (volume_data["volume_signal"] == "BEARISH" or volume_data["volume_signal"] == "NEUTRAL")
            )
        
        # Adjust confidence based on visual confirmation
        adjusted_confidence = signal['confidence']
        if visual_confirms_signal:
            adjusted_confidence = min(95, signal['confidence'] + 10)
        elif pattern_signal != "NEUTRAL" and pattern_signal != ("BULLISH" if action == "BUY" else "BEARISH"):
            adjusted_confidence = max(40, signal['confidence'] - 15)
        
        return {
            "symbol": "XAUUSD",
            "action": action,
            "signal": signal['signal'],
            "score": signal['score'],
            "confidence": adjusted_confidence,
            "original_confidence": signal['confidence'],
            "size_multiplier": signal['size_multiplier'],
            "dca_allowed": signal['dca_allowed'],
            "reasoning": signal['reasoning'],
            "timestamp": signal['timestamp'],
            "source": "CRELLASTEIN_META_V2",
            
            # Current price
            "price": entry_data["current_price"],
            
            # Optimal entry zone
            "optimal_entry": entry_data["optimal_entry"],
            "entry_zone_high": entry_data["entry_zone_high"],
            "entry_zone_low": entry_data["entry_zone_low"],
            "danger_zone": entry_data["danger_zone"],
            "entry_basis": entry_data["entry_basis"],
            "in_entry_zone": entry_data["in_entry_zone"],
            "distance_to_entry_pips": entry_data["distance_to_entry_pips"],
            
            # Take profit targets
            "target_tp": tp_data["target_tp"],
            "target_tp_2": tp_data["target_tp_2"],
            "target_tp_3": tp_data["target_tp_3"],
            "tp_basis": tp_data["tp_basis"],
            "risk_reward_ratio": tp_data["risk_reward_ratio"],
            
            # Support/Resistance levels
            "support_levels": entry_data["support_levels"],
            "resistance_levels": tp_data["resistance_levels"],
            
            # NEW: Volume Analysis
            "volume": {
                "signal": volume_data["volume_signal"],
                "confidence": volume_data["volume_confidence"],
                "reasoning": volume_data["volume_reasoning"],
                "ratio": volume_data["volume_ratio"],
                "spike": volume_data["volume_spike"],
                "surge": volume_data["volume_surge"],
                "accumulation": volume_data["accumulation_distribution"],
                "obv": volume_data["obv_signal"],
            },
            
            # NEW: Pattern Analysis  
            "patterns": {
                "signal": pattern_signal,
                "count": len(patterns_list),
                "bullish": pattern_bullish,
                "bearish": pattern_bearish,
                "detected": patterns_list[:5],  # Top 5 patterns
            },
            
            # NEW: Visual Confirmation
            "visual_confirmation": visual_confirms_signal,
            "visual_summary": f"Volume: {volume_data['volume_signal']}, Patterns: {pattern_signal} ({pattern_bullish}B/{pattern_bearish}S)",
        }
    except Exception as e:
        logger.error(f"Error getting Ghost signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta/ghost/iren")
async def ghost_iren_signal():
    """Signal endpoint for IREN with optimal entry, TP, patterns, and volume"""
    try:
        signal = iren_bot.get_signal_summary()
        
        # Map to Ghost format
        action = "HOLD"
        if signal['signal'] in ["STRONG_BUY", "BUY"]:
            action = "BUY"
        elif signal['signal'] in ["STRONG_SELL", "SELL"]:
            action = "SELL"
        
        # Calculate optimal entry and TP targets
        entry_calc = EntryCalculator("IREN")
        entry_data = entry_calc.get_optimal_entry()
        tp_data = entry_calc.get_target_tp(entry_data["optimal_entry"])
        
        # Analyze volume
        vol_analyzer = VolumeAnalyzer("IREN")
        volume_data = vol_analyzer.analyze_volume()
        
        # Detect chart patterns - get data first
        import yfinance as yf
        try:
            ticker = yf.Ticker("IREN")
            pattern_df = ticker.history(period="30d", interval="1h")
            pattern_detector = PatternDetector("IREN")
            patterns = pattern_detector.analyze(pattern_df) if not pattern_df.empty else []
        except Exception as pe:
            logger.warning(f"Pattern detection error: {pe}")
            patterns = []
        
        # Convert patterns to JSON-serializable format
        patterns_list = []
        pattern_bullish = 0
        pattern_bearish = 0
        for p in patterns:
            patterns_list.append({
                "type": p.pattern_type.value,
                "confidence": float(p.confidence),
                "direction": p.direction,
                "description": p.description,
            })
            if p.direction == "BUY":
                pattern_bullish += 1
            elif p.direction == "SELL":
                pattern_bearish += 1
        
        # Pattern summary
        if pattern_bullish > pattern_bearish:
            pattern_signal = "BULLISH"
        elif pattern_bearish > pattern_bullish:
            pattern_signal = "BEARISH"
        else:
            pattern_signal = "NEUTRAL"
        
        # Combined visual confirmation (same logic as XAUUSD)
        has_bull_flag_forming = any("FORMING" in p.get("description", "") for p in patterns_list if "bull_flag" in p.get("type", "").lower())
        has_bear_flag_forming = any("FORMING" in p.get("description", "") for p in patterns_list if "bear_flag" in p.get("type", "").lower())
        
        visual_confirms_signal = False
        
        # Flag patterns = strong visual confirmation regardless of action
        if has_bull_flag_forming and pattern_signal == "BULLISH":
            visual_confirms_signal = True
        elif has_bear_flag_forming and pattern_signal == "BEARISH":
            visual_confirms_signal = True
        elif action == "BUY":
            visual_confirms_signal = (
                (pattern_signal == "BULLISH" or pattern_signal == "NEUTRAL") and
                (volume_data["volume_signal"] == "BULLISH" or volume_data["volume_signal"] == "NEUTRAL")
            )
        elif action == "SELL":
            visual_confirms_signal = (
                (pattern_signal == "BEARISH" or pattern_signal == "NEUTRAL") and
                (volume_data["volume_signal"] == "BEARISH" or volume_data["volume_signal"] == "NEUTRAL")
            )
        
        # Adjust confidence based on visual confirmation
        adjusted_confidence = signal['confidence']
        if visual_confirms_signal:
            adjusted_confidence = min(95, signal['confidence'] + 10)
        elif pattern_signal != "NEUTRAL" and pattern_signal != ("BULLISH" if action == "BUY" else "BEARISH"):
            adjusted_confidence = max(40, signal['confidence'] - 15)
        
        return {
            "symbol": "IREN",
            "action": action,
            "signal": signal['signal'],
            "score": signal['score'],
            "confidence": adjusted_confidence,
            "original_confidence": signal['confidence'],
            "size_multiplier": signal['size_multiplier'],
            "dca_allowed": signal['dca_allowed'],
            "reasoning": signal['reasoning'],
            "timestamp": signal['timestamp'],
            "source": "CRELLASTEIN_META_V2",
            
            # Current price
            "price": entry_data["current_price"],
            
            # Optimal entry zone
            "optimal_entry": entry_data["optimal_entry"],
            "entry_zone_high": entry_data["entry_zone_high"],
            "entry_zone_low": entry_data["entry_zone_low"],
            "danger_zone": entry_data["danger_zone"],
            "entry_basis": entry_data["entry_basis"],
            "in_entry_zone": entry_data["in_entry_zone"],
            "distance_to_entry_pips": entry_data["distance_to_entry_pips"],
            
            # Take profit targets
            "target_tp": tp_data["target_tp"],
            "target_tp_2": tp_data["target_tp_2"],
            "target_tp_3": tp_data["target_tp_3"],
            "tp_basis": tp_data["tp_basis"],
            "risk_reward_ratio": tp_data["risk_reward_ratio"],
            
            # Support/Resistance levels
            "support_levels": entry_data["support_levels"],
            "resistance_levels": tp_data["resistance_levels"],
            
            # Volume Analysis
            "volume": {
                "signal": volume_data["volume_signal"],
                "confidence": volume_data["volume_confidence"],
                "reasoning": volume_data["volume_reasoning"],
                "ratio": volume_data["volume_ratio"],
                "spike": volume_data["volume_spike"],
                "surge": volume_data["volume_surge"],
                "accumulation": volume_data["accumulation_distribution"],
                "obv": volume_data["obv_signal"],
            },
            
            # Pattern Analysis  
            "patterns": {
                "signal": pattern_signal,
                "count": len(patterns_list),
                "bullish": pattern_bullish,
                "bearish": pattern_bearish,
                "detected": patterns_list[:5],  # Top 5 patterns
            },
            
            # Visual Confirmation
            "visual_confirmation": visual_confirms_signal,
            "visual_summary": f"Volume: {volume_data['volume_signal']}, Patterns: {pattern_signal} ({pattern_bullish}B/{pattern_bearish}S)",
        }
    except Exception as e:
        logger.error(f"Error getting IREN Ghost signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trade/update")
async def trade_update(data: Dict = None):
    """Receive trade updates from Ghost Commander/Casper"""
    if data is None:
        data = {}
    logger.info(f"Trade update received: {data}")
    return {
        "status": "received",
        "message": "Trade update logged",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/meta/entry/{symbol}")
async def get_entry_levels(symbol: str):
    """Get just the entry and TP levels for a symbol"""
    symbol = symbol.upper()
    if symbol not in ["XAUUSD", "IREN"]:
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}")
    
    try:
        entry_calc = EntryCalculator(symbol)
        data = entry_calc.get_complete_signal()
        return data
    except Exception as e:
        logger.error(f"Error getting entry levels for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8035)


# ═══════════════════════════════════════════════════════════════════════════════
# IREN PRE-MARKET REPORT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/iren/premarket")
async def get_iren_premarket():
    """Get IREN pre-market analysis report"""
    try:
        import sys
        sys.path.insert(0, '/home/jbot/trading_ai/neo')
        from iren_premarket_report import IrenPreMarketAnalyzer
        
        analyzer = IrenPreMarketAnalyzer()
        report = analyzer.generate_report()
        
        return report
    except Exception as e:
        logger.error(f"IREN premarket error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/iren/premarket/send")
async def send_iren_premarket():
    """Generate and send IREN pre-market report to Telegram"""
    try:
        import sys
        sys.path.insert(0, '/home/jbot/trading_ai/neo')
        from iren_premarket_report import send_iren_premarket_report
        
        result = send_iren_premarket_report()
        return result
    except Exception as e:
        logger.error(f"IREN premarket send error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
