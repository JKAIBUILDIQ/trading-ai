#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO-GHOST BRIDGE - Time-Based Stealth Trading Execution
═══════════════════════════════════════════════════════════════════════════════

Connects NEO's 4-hour predictions to Ghost Commander for stealth execution.

🛡️ WHY TIME-BASED?
- NO visible SL → MMs can't hunt stops
- NO visible TP → MMs can't see our target
- Pure direction bet → Just predict UP or DOWN
- Exit by time → No emotional decisions
- Learning focus → Clean data for improvement

Flow:
1. NEO predicts: "Gold UP +35 pips in 4 hours" (82% confidence)
2. Bridge validates: Confidence >= 60%? Symbol correct?
3. Ghost executes: BUY with NO TP, emergency SL only
4. 4 hours later: Ghost closes at market price
5. Result logged: NEO learns from outcome

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx

# Import NEO predictor
from gold_predictor import GoldPredictor, Prediction, get_predictor
from prediction_store import PredictionStore, get_store
from prediction_learner import PredictionLearner, get_learner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEOGhostBridge")

# Configuration
SIGNAL_DIR = Path("/home/jbot/trading_ai/signals")
SIGNAL_DIR.mkdir(exist_ok=True)

# Ghost Commander reads this file for signals
GHOST_SIGNAL_FILE = SIGNAL_DIR / "neo_timed_signal.json"
GHOST_RESULT_FILE = SIGNAL_DIR / "neo_timed_result.json"


@dataclass
class TimedTradeSignal:
    """Signal format for Ghost Commander time-based execution"""
    signal_id: str
    prediction_id: str
    symbol: str
    direction: str  # BUY or SELL
    confidence: float
    predicted_change: float
    hold_minutes: int = 240  # 4 hours
    emergency_sl_pips: float = 200  # Far away, flash crash protection only
    
    # Sizing
    risk_percent: float = 1.0
    max_lots: float = 5.0
    
    # Metadata
    reasoning: str = ""
    timestamp: str = ""
    target_time: str = ""
    
    # Status
    status: str = "PENDING"  # PENDING, EXECUTED, CLOSED, CANCELLED
    executed_at: str = ""
    closed_at: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit: float = 0.0


class NEOGhostBridge:
    """
    Bridge between NEO predictions and Ghost Commander execution.
    
    Features:
    - Converts predictions to time-based signals
    - Validates signals before execution
    - Tracks execution status
    - Logs results for learning
    """
    
    def __init__(
        self,
        min_confidence: float = 60.0,
        hold_minutes: int = 240,
        auto_execute: bool = False
    ):
        self.predictor = get_predictor()
        self.store = get_store()
        self.learner = get_learner()
        
        self.min_confidence = min_confidence
        self.hold_minutes = hold_minutes
        self.auto_execute = auto_execute
        
        self.current_signal: Optional[TimedTradeSignal] = None
        self.signal_history: List[TimedTradeSignal] = []
        
        logger.info("=" * 60)
        logger.info("🔗 NEO-GHOST BRIDGE INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   Min Confidence: {min_confidence}%")
        logger.info(f"   Hold Period: {hold_minutes} minutes")
        logger.info(f"   Auto Execute: {'YES ⚠️' if auto_execute else 'NO (manual)'}")
        logger.info(f"   Signal File: {GHOST_SIGNAL_FILE}")
        logger.info("=" * 60)
    
    def generate_signal(self) -> Optional[TimedTradeSignal]:
        """
        Generate a time-based trading signal from NEO prediction.
        
        Returns signal if valid, None if conditions not met.
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔮 GENERATING TIME-BASED SIGNAL")
        logger.info("=" * 60)
        
        # Get prediction
        prediction = self.predictor.predict_4h()
        
        # Store prediction
        self.store.save_prediction(prediction)
        
        # Validate
        if prediction.predicted_direction == "FLAT":
            logger.info("⏸️ Prediction is FLAT - no signal")
            return None
        
        if prediction.confidence < self.min_confidence:
            logger.warning(f"⚠️ Confidence {prediction.confidence:.0f}% < {self.min_confidence}% minimum")
            return None
        
        # Convert direction to BUY/SELL
        direction = "BUY" if prediction.predicted_direction == "UP" else "SELL"
        
        # Create signal
        now = datetime.now(timezone.utc)
        signal = TimedTradeSignal(
            signal_id=f"SIGNAL_{now.strftime('%Y%m%d_%H%M%S')}",
            prediction_id=prediction.prediction_id,
            symbol="XAUUSD",
            direction=direction,
            confidence=prediction.confidence,
            predicted_change=prediction.predicted_change_pips,
            hold_minutes=self.hold_minutes,
            reasoning=prediction.reasoning,
            timestamp=now.isoformat(),
            target_time=prediction.target_time
        )
        
        logger.info("\n📊 SIGNAL GENERATED:")
        logger.info(f"   ID: {signal.signal_id}")
        logger.info(f"   Direction: {signal.direction}")
        logger.info(f"   Predicted Change: {signal.predicted_change:+.1f} pips")
        logger.info(f"   Confidence: {signal.confidence:.0f}%")
        logger.info(f"   Hold Period: {signal.hold_minutes} minutes")
        logger.info(f"   Reasoning: {signal.reasoning[:100]}...")
        
        self.current_signal = signal
        
        return signal
    
    def write_signal_for_ghost(self, signal: TimedTradeSignal) -> bool:
        """
        Write signal to file for Ghost Commander to read.
        
        Ghost polls this file and executes when new signal appears.
        """
        try:
            signal_dict = asdict(signal)
            
            # Add execution instructions
            signal_dict["instructions"] = {
                "mode": "TIMED",
                "entry": "MARKET",
                "sl_type": "EMERGENCY_ONLY",
                "sl_pips": signal.emergency_sl_pips,
                "tp_type": "NONE",  # Stealth!
                "exit_method": "TIME",
                "exit_after_minutes": signal.hold_minutes
            }
            
            with open(GHOST_SIGNAL_FILE, 'w') as f:
                json.dump(signal_dict, f, indent=2)
            
            logger.info(f"📝 Signal written to: {GHOST_SIGNAL_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to write signal: {e}")
            return False
    
    def check_ghost_result(self) -> Optional[Dict]:
        """
        Check if Ghost has completed a trade and logged results.
        """
        try:
            if not GHOST_RESULT_FILE.exists():
                return None
            
            with open(GHOST_RESULT_FILE, 'r') as f:
                result = json.load(f)
            
            # Check if this is a new result
            if result.get('processed', False):
                return None
            
            logger.info(f"📥 Found Ghost result: {result.get('signal_id')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error reading Ghost result: {e}")
            return None
    
    def process_ghost_result(self, result: Dict):
        """
        Process trade result from Ghost and update learning.
        """
        logger.info("\n" + "=" * 60)
        logger.info("📊 PROCESSING GHOST RESULT")
        logger.info("=" * 60)
        
        prediction_id = result.get('prediction_id', '')
        won = result.get('won', False)
        change_pips = result.get('change_pips', 0)
        profit = result.get('profit', 0)
        
        logger.info(f"   Prediction: {prediction_id}")
        logger.info(f"   Won: {won}")
        logger.info(f"   Change: {change_pips:+.1f} pips")
        logger.info(f"   Profit: ${profit:.2f}")
        
        # Get original prediction from store
        prediction = self.store.get_prediction(prediction_id)
        
        if prediction:
            # Determine actual direction
            actual_direction = "UP" if change_pips > 0 else "DOWN" if change_pips < 0 else "FLAT"
            
            # Update store with evaluation
            self.store.update_prediction(prediction_id, {
                "status": "EVALUATED",
                "actual_change_pips": change_pips,
                "actual_direction": actual_direction,
                "direction_correct": won,
                "score": 1.0 if won else 0.0,
                "evaluated_at": datetime.utcnow().isoformat()
            })
            
            # Record for learner
            self.learner.record_outcome(
                prediction_id,
                prediction.get('feature_contributions', {}),
                prediction.get('predicted_direction', 'FLAT'),
                actual_direction,
                won,
                prediction.get('confidence', 50)
            )
            
            logger.info("✅ Result processed for learning")
            
            # Mark result as processed
            result['processed'] = True
            with open(GHOST_RESULT_FILE, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            logger.warning(f"⚠️ Prediction not found: {prediction_id}")
    
    async def run_bridge_loop(self):
        """
        Main bridge loop:
        1. Generate prediction every 4 hours
        2. Write signal for Ghost
        3. Check for results and process
        """
        logger.info("🔄 Starting NEO-Ghost bridge loop...")
        
        while True:
            try:
                # Check for Ghost results first
                result = self.check_ghost_result()
                if result:
                    self.process_ghost_result(result)
                
                # Check if we need a new signal
                now = datetime.now(timezone.utc)
                
                if self.current_signal:
                    target_time = datetime.fromisoformat(
                        self.current_signal.target_time.replace('Z', '+00:00')
                    )
                    if isinstance(target_time, datetime) and target_time.tzinfo is None:
                        target_time = target_time.replace(tzinfo=timezone.utc)
                    
                    if now < target_time:
                        # Signal still active
                        remaining = int((target_time - now).total_seconds() / 60)
                        logger.info(f"⏱️ Current signal active - {remaining} minutes remaining")
                    else:
                        # Signal expired - generate new one
                        logger.info("⏰ Signal expired - generating new one")
                        self.current_signal = None
                
                if not self.current_signal:
                    # Generate new signal
                    signal = self.generate_signal()
                    
                    if signal:
                        if self.auto_execute:
                            self.write_signal_for_ghost(signal)
                            logger.info("🚀 Signal sent to Ghost (auto-execute ON)")
                        else:
                            logger.info("📋 Signal ready (auto-execute OFF - manual confirmation needed)")
                            logger.info(f"   To execute, call: bridge.execute_signal()")
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in bridge loop: {e}")
                await asyncio.sleep(60)
    
    def execute_signal(self):
        """Manually execute the current pending signal"""
        if not self.current_signal:
            logger.warning("⚠️ No pending signal to execute")
            return False
        
        if self.current_signal.status != "PENDING":
            logger.warning(f"⚠️ Signal already {self.current_signal.status}")
            return False
        
        success = self.write_signal_for_ghost(self.current_signal)
        
        if success:
            self.current_signal.status = "SENT"
            logger.info("🚀 Signal sent to Ghost for execution")
        
        return success
    
    def cancel_signal(self):
        """Cancel the current pending signal"""
        if self.current_signal:
            self.current_signal.status = "CANCELLED"
            logger.info(f"❌ Signal cancelled: {self.current_signal.signal_id}")
            self.current_signal = None
    
    def get_status(self) -> Dict:
        """Get current bridge status"""
        accuracy = self.store.get_accuracy(100)
        stats = self.store.get_stats()
        
        return {
            "bridge_status": "ACTIVE",
            "auto_execute": self.auto_execute,
            "min_confidence": self.min_confidence,
            "hold_minutes": self.hold_minutes,
            "current_signal": asdict(self.current_signal) if self.current_signal else None,
            "accuracy": {
                "last_100": accuracy,
                "total_predictions": stats.get("total_predictions", 0),
                "correct_direction": stats.get("correct_direction", 0),
                "target": 60.0,
                "meets_target": accuracy >= 60.0
            },
            "learner_status": {
                "features_tracked": len(self.learner.feature_stats),
                "weights_file": self.learner.weights_file
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_bridge = None

def get_bridge(auto_execute: bool = False) -> NEOGhostBridge:
    """Get singleton bridge instance"""
    global _bridge
    if _bridge is None:
        _bridge = NEOGhostBridge(auto_execute=auto_execute)
    return _bridge


def generate_timed_signal() -> Optional[Dict]:
    """Quick function to generate a time-based signal"""
    bridge = get_bridge()
    signal = bridge.generate_signal()
    return asdict(signal) if signal else None


def execute_current_signal() -> bool:
    """Quick function to execute current signal"""
    return get_bridge().execute_signal()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_bridge():
    """Test the NEO-Ghost bridge"""
    print("\n" + "=" * 70)
    print("🧪 TESTING NEO-GHOST BRIDGE")
    print("=" * 70)
    
    bridge = NEOGhostBridge(min_confidence=50, auto_execute=False)
    
    # Generate signal
    signal = bridge.generate_signal()
    
    if signal:
        print("\n" + "=" * 70)
        print("📊 SIGNAL SUMMARY")
        print("=" * 70)
        print(f"ID: {signal.signal_id}")
        print(f"Direction: {signal.direction}")
        print(f"Predicted Change: {signal.predicted_change:+.1f} pips")
        print(f"Confidence: {signal.confidence:.0f}%")
        print(f"Hold Period: {signal.hold_minutes} minutes")
        print(f"Emergency SL: {signal.emergency_sl_pips} pips")
        print(f"\n🛡️ STEALTH MODE:")
        print(f"   - NO visible TP (MMs can't see target)")
        print(f"   - Emergency SL only (flash crash protection)")
        print(f"   - Exit by time, not price")
        print(f"\nReasoning: {signal.reasoning}")
        
        # Write signal file
        print("\n" + "=" * 70)
        print("Writing signal file for Ghost...")
        bridge.write_signal_for_ghost(signal)
        
        # Show what Ghost will see
        print("\nGhost Commander will read:")
        print(f"   File: {GHOST_SIGNAL_FILE}")
        with open(GHOST_SIGNAL_FILE, 'r') as f:
            content = json.load(f)
            print(json.dumps(content, indent=2)[:500] + "...")
    else:
        print("\n⚠️ No signal generated (confidence too low or direction FLAT)")
    
    # Show status
    print("\n" + "=" * 70)
    print("📊 BRIDGE STATUS")
    print("=" * 70)
    status = bridge.get_status()
    print(f"Auto Execute: {status['auto_execute']}")
    print(f"Min Confidence: {status['min_confidence']}%")
    print(f"Accuracy (last 100): {status['accuracy']['last_100']:.1f}%")
    print(f"Target: {status['accuracy']['target']}%")


if __name__ == "__main__":
    test_bridge()
