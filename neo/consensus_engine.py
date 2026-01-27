#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO + META BOT CONSENSUS ENGINE
═══════════════════════════════════════════════════════════════════════════════

Combines NEO (prediction-based) and Meta Bot (indicator-based) into ONE unified
signal. This eliminates conflicting opinions and provides a stronger signal.

CONSENSUS RULES:
1. BOTH AGREE → Strong signal (HIGH confidence)
2. ONE says HOLD → Use the other's signal (MEDIUM confidence)
3. CONFLICT (BUY vs SELL) → HOLD until alignment

Special handling for SuperTrend:
- If SuperTrend conflicts with other indicators, flag it
- Give SuperTrend extra weight for trend direction

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConsensusEngine")

# API endpoints
NEO_API = "http://localhost:8036"  # NEO Ghost Integration API
META_API = "http://localhost:8035"  # Crellastein Meta Bot API


@dataclass
class ConsensusSignal:
    """Unified signal from NEO + Meta Bot"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: int  # 0-100
    consensus_level: str  # STRONG, MEDIUM, WEAK, CONFLICT
    
    # Individual signals
    neo_action: str
    neo_confidence: int
    meta_action: str
    meta_confidence: int
    
    # SuperTrend specific
    supertrend_direction: str
    supertrend_agrees: bool
    
    # Entry/Exit levels (unified)
    entry: float
    stop_loss: float
    take_profit: float
    
    # Reasoning
    reasoning: str
    conflict_warning: str = None


class ConsensusEngine:
    """
    Combines NEO and Meta Bot signals into unified consensus.
    """
    
    def __init__(self):
        self.neo_api = NEO_API
        self.meta_api = META_API
    
    def get_neo_signal(self, symbol: str) -> Dict:
        """Fetch NEO signal"""
        try:
            endpoint = f"{self.neo_api}/api/neo/{symbol.lower()}/signal"
            resp = requests.get(endpoint, timeout=10)
            if resp.ok:
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch NEO signal: {e}")
        return {"action": "HOLD", "confidence": 0, "valid": False}
    
    def get_meta_signal(self, symbol: str) -> Dict:
        """Fetch Meta Bot signal with indicator breakdown"""
        try:
            # Get signal
            signal_resp = requests.get(f"{self.meta_api}/api/meta/ghost/{symbol.lower()}", timeout=10)
            signal = signal_resp.json() if signal_resp.ok else {}
            
            # Get indicator breakdown
            composite_resp = requests.get(f"{self.meta_api}/api/meta/{symbol.lower()}/composite", timeout=10)
            composite = composite_resp.json() if composite_resp.ok else {}
            
            # Extract SuperTrend specifically
            indicators = composite.get("indicators", {})
            supertrend = indicators.get("supertrend", {})
            
            return {
                "action": signal.get("action", "HOLD"),
                "confidence": signal.get("confidence", 0),
                "supertrend": supertrend,
                "indicators": indicators,
                "patterns": signal.get("patterns", {}),
                "volume": signal.get("volume_analysis", {})
            }
        except Exception as e:
            logger.error(f"Failed to fetch Meta signal: {e}")
        return {"action": "HOLD", "confidence": 0}
    
    def calculate_consensus(self, symbol: str) -> ConsensusSignal:
        """
        Calculate unified consensus signal.
        
        Rules:
        1. Both BUY → STRONG BUY
        2. Both SELL → STRONG SELL
        3. Both HOLD → HOLD
        4. One BUY, one HOLD → MEDIUM BUY
        5. One SELL, one HOLD → MEDIUM SELL
        6. BUY vs SELL → CONFLICT (HOLD until aligned)
        """
        neo = self.get_neo_signal(symbol)
        meta = self.get_meta_signal(symbol)
        
        neo_action = neo.get("action", "HOLD").upper()
        neo_conf = neo.get("confidence", 0)
        
        meta_action = meta.get("action", "HOLD").upper()
        meta_conf = meta.get("confidence", 0)
        
        # Normalize actions
        if neo_action not in ["BUY", "SELL", "HOLD"]:
            neo_action = "HOLD"
        if meta_action not in ["BUY", "SELL", "HOLD"]:
            meta_action = "HOLD"
        
        # Get SuperTrend direction
        supertrend = meta.get("supertrend", {})
        st_signal = supertrend.get("signal", "NEUTRAL").upper()
        st_direction = "BUY" if st_signal == "BULLISH" else "SELL" if st_signal == "BEARISH" else "HOLD"
        
        # ═══════════════════════════════════════════════════════════════════
        # CONSENSUS LOGIC
        # ═══════════════════════════════════════════════════════════════════
        
        conflict_warning = None
        
        # Case 1: Both agree on direction
        if neo_action == meta_action and neo_action != "HOLD":
            action = neo_action
            confidence = int((neo_conf + meta_conf) / 2 * 1.2)  # Boost for agreement
            confidence = min(confidence, 95)
            consensus_level = "STRONG"
            reasoning = f"NEO and Meta Bot AGREE on {action}. Combined confidence boosted."
        
        # Case 2: Both HOLD
        elif neo_action == "HOLD" and meta_action == "HOLD":
            action = "HOLD"
            confidence = 50
            consensus_level = "NEUTRAL"
            reasoning = "Both NEO and Meta Bot say HOLD. No clear setup."
        
        # Case 3: Conflict (BUY vs SELL)
        elif (neo_action == "BUY" and meta_action == "SELL") or (neo_action == "SELL" and meta_action == "BUY"):
            action = "HOLD"
            confidence = 30
            consensus_level = "CONFLICT"
            conflict_warning = f"⚠️ CONFLICT: NEO says {neo_action}, Meta says {meta_action}. WAITING for alignment."
            
            # Use SuperTrend as tiebreaker if both have similar confidence
            if abs(neo_conf - meta_conf) < 20:
                reasoning = f"CONFLICT detected! SuperTrend points {st_direction}. Waiting for alignment before trading."
            else:
                # Higher confidence wins, but still mark as conflict
                winner = "NEO" if neo_conf > meta_conf else "Meta"
                reasoning = f"CONFLICT detected! {winner} has higher confidence but still waiting for alignment."
        
        # Case 4: One active, one HOLD
        elif neo_action in ["BUY", "SELL"] and meta_action == "HOLD":
            action = neo_action
            confidence = int(neo_conf * 0.7)  # Reduced confidence
            consensus_level = "MEDIUM"
            reasoning = f"NEO says {neo_action} ({neo_conf}%), Meta says HOLD. Using NEO with reduced confidence."
        
        elif meta_action in ["BUY", "SELL"] and neo_action == "HOLD":
            action = meta_action
            confidence = int(meta_conf * 0.7)  # Reduced confidence
            consensus_level = "MEDIUM"
            reasoning = f"Meta says {meta_action} ({meta_conf}%), NEO says HOLD. Using Meta with reduced confidence."
        
        else:
            action = "HOLD"
            confidence = 40
            consensus_level = "WEAK"
            reasoning = "Unclear signals. Defaulting to HOLD."
        
        # SuperTrend agreement check
        supertrend_agrees = (st_direction == action) or st_direction == "HOLD"
        if not supertrend_agrees and action != "HOLD":
            conflict_warning = (conflict_warning or "") + f" ⚠️ SuperTrend disagrees ({st_direction})!"
            confidence = int(confidence * 0.85)  # Reduce confidence if ST disagrees
        
        # Get unified levels (prefer Meta's calculated levels, fall back to NEO)
        entry = neo.get("entry", 0) or meta.get("optimal_entry", 0)
        stop_loss = neo.get("stop_loss", 0)
        take_profit = neo.get("take_profit", 0)
        
        return ConsensusSignal(
            symbol=symbol.upper(),
            action=action,
            confidence=confidence,
            consensus_level=consensus_level,
            neo_action=neo_action,
            neo_confidence=neo_conf,
            meta_action=meta_action,
            meta_confidence=meta_conf,
            supertrend_direction=st_direction,
            supertrend_agrees=supertrend_agrees,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            conflict_warning=conflict_warning
        )
    
    def get_xauusd_consensus(self) -> ConsensusSignal:
        """Get XAUUSD consensus"""
        return self.calculate_consensus("XAUUSD")
    
    def get_iren_consensus(self) -> ConsensusSignal:
        """Get IREN consensus"""
        return self.calculate_consensus("IREN")
    
    def format_telegram_report(self, signal: ConsensusSignal) -> str:
        """Format consensus signal for Telegram"""
        
        # Emoji based on consensus level
        level_emoji = {
            "STRONG": "🟢",
            "MEDIUM": "🟡",
            "WEAK": "🟠",
            "NEUTRAL": "⚪",
            "CONFLICT": "🔴"
        }
        
        action_emoji = {
            "BUY": "📈",
            "SELL": "📉",
            "HOLD": "⏸️"
        }
        
        lines = [
            f"🤝 <b>CONSENSUS SIGNAL - {signal.symbol}</b>",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"{action_emoji.get(signal.action, '❓')} <b>ACTION: {signal.action}</b>",
            f"{level_emoji.get(signal.consensus_level, '⚪')} Consensus: {signal.consensus_level} ({signal.confidence}%)",
            f"",
            f"📊 <b>INDIVIDUAL SIGNALS:</b>",
            f"├── NEO: {signal.neo_action} ({signal.neo_confidence}%)",
            f"├── Meta Bot: {signal.meta_action} ({signal.meta_confidence}%)",
            f"└── SuperTrend: {signal.supertrend_direction} {'✅' if signal.supertrend_agrees else '⚠️'}",
            f"",
        ]
        
        if signal.conflict_warning:
            lines.append(f"<b>{signal.conflict_warning}</b>")
            lines.append(f"")
        
        if signal.action != "HOLD":
            lines.extend([
                f"📍 <b>LEVELS:</b>",
                f"├── Entry: ${signal.entry:.2f}" if signal.symbol == "IREN" else f"├── Entry: ${signal.entry:.0f}",
                f"├── SL: ${signal.stop_loss:.2f}" if signal.symbol == "IREN" else f"├── SL: ${signal.stop_loss:.0f}",
                f"└── TP: ${signal.take_profit:.2f}" if signal.symbol == "IREN" else f"└── TP: ${signal.take_profit:.0f}",
                f"",
            ])
        
        lines.extend([
            f"💡 <b>REASONING:</b>",
            f"   {signal.reasoning}",
        ])
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NEO + Meta Bot Consensus Engine",
    description="Unified signals combining NEO predictions and Meta Bot indicators",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ConsensusEngine()


@app.get("/")
async def root():
    return {
        "service": "NEO + Meta Bot Consensus Engine",
        "version": "1.0.0",
        "endpoints": {
            "xauusd": "/api/consensus/xauusd",
            "iren": "/api/consensus/iren",
            "report": "/api/consensus/{symbol}/report"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/consensus/xauusd")
async def get_xauusd_consensus():
    """Get unified XAUUSD signal from NEO + Meta Bot"""
    signal = engine.get_xauusd_consensus()
    return {
        "symbol": signal.symbol,
        "action": signal.action,
        "confidence": signal.confidence,
        "consensus_level": signal.consensus_level,
        
        "neo": {
            "action": signal.neo_action,
            "confidence": signal.neo_confidence
        },
        "meta": {
            "action": signal.meta_action,
            "confidence": signal.meta_confidence
        },
        "supertrend": {
            "direction": signal.supertrend_direction,
            "agrees": signal.supertrend_agrees
        },
        
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        
        "reasoning": signal.reasoning,
        "conflict_warning": signal.conflict_warning,
        
        "timestamp": datetime.utcnow().isoformat(),
        "valid": signal.action != "HOLD" and signal.consensus_level in ["STRONG", "MEDIUM"]
    }


@app.get("/api/consensus/iren")
async def get_iren_consensus():
    """Get unified IREN signal from NEO + Meta Bot"""
    signal = engine.get_iren_consensus()
    return {
        "symbol": signal.symbol,
        "action": signal.action,
        "confidence": signal.confidence,
        "consensus_level": signal.consensus_level,
        
        "neo": {
            "action": signal.neo_action,
            "confidence": signal.neo_confidence
        },
        "meta": {
            "action": signal.meta_action,
            "confidence": signal.meta_confidence
        },
        "supertrend": {
            "direction": signal.supertrend_direction,
            "agrees": signal.supertrend_agrees
        },
        
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        
        "reasoning": signal.reasoning,
        "conflict_warning": signal.conflict_warning,
        
        "timestamp": datetime.utcnow().isoformat(),
        "valid": signal.action != "HOLD" and signal.consensus_level in ["STRONG", "MEDIUM"]
    }


@app.get("/api/consensus/{symbol}/report")
async def get_consensus_report(symbol: str):
    """Get formatted Telegram report"""
    signal = engine.calculate_consensus(symbol.upper())
    report = engine.format_telegram_report(signal)
    return {
        "symbol": symbol.upper(),
        "report": report,
        "signal": {
            "action": signal.action,
            "confidence": signal.confidence,
            "consensus_level": signal.consensus_level
        }
    }


# Ghost Commander compatible endpoint
@app.get("/api/ghost/{symbol}")
async def get_ghost_signal(symbol: str):
    """
    Ghost-compatible endpoint that returns ONLY consensus signal.
    Ghost should use THIS instead of individual NEO/Meta endpoints.
    """
    signal = engine.calculate_consensus(symbol.upper())
    
    return {
        "symbol": signal.symbol,
        "action": signal.action,
        "confidence": signal.confidence,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        
        "valid": signal.action != "HOLD" and signal.consensus_level in ["STRONG", "MEDIUM"],
        "consensus_level": signal.consensus_level,
        
        "invalidation": {
            "level": signal.stop_loss,
            "reason": signal.conflict_warning if signal.consensus_level == "CONFLICT" else None,
            "confidence_check": "PASS" if signal.confidence >= 50 else "FAIL"
        },
        
        "supertrend_agrees": signal.supertrend_agrees,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "consensus_engine:app",
        host="0.0.0.0",
        port=8037,
        reload=False,
        log_level="info"
    )
