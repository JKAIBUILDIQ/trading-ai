"""
NEO-GOLD Market Maker Predictor
Predict what institutional players will likely do next
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .config import logger


class MMTactic(Enum):
    """Common Market Maker tactics on Gold."""
    STOP_HUNT_BELOW = "stop_hunt_below"
    STOP_HUNT_ABOVE = "stop_hunt_above"
    FAKE_BREAKOUT_UP = "fake_breakout_up"
    FAKE_BREAKOUT_DOWN = "fake_breakout_down"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    LIQUIDITY_GRAB = "liquidity_grab"
    ROUND_NUMBER_MAGNET = "round_number_magnet"
    NEWS_VOLATILITY_TRAP = "news_volatility_trap"
    SESSION_OPEN_SWEEP = "session_open_sweep"
    NONE = "none"


@dataclass
class MMPrediction:
    """A predicted Market Maker move."""
    tactic: MMTactic
    probability: int  # 0-100
    direction_after: str  # Expected direction AFTER the tactic plays out
    target_level: float
    description: str
    counter_strategy: str  # How retail should respond


class MarketMakerPredictor:
    """
    Predicts Market Maker behavior on Gold.
    
    Key insights:
    - MMs need liquidity to fill large orders
    - They hunt stops where retail traders cluster
    - They use news events to create volatility and trap traders
    - Round numbers attract price (and stops)
    - Session opens are prime time for manipulation
    
    "What Would Citadel Do?"
    """
    
    def __init__(self):
        self.current_price: float = 0
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.predictions: List[MMPrediction] = []
        
    def predict(self, features: Dict, patterns: List) -> List[MMPrediction]:
        """Predict MM behavior based on current market structure."""
        
        self.predictions = []
        self.current_price = features.get("price", 0)
        
        session = features.get("session", "")
        session_details = features.get("session_details", {})
        round_numbers = features.get("round_number", {})
        asian_range = features.get("asian_range", {})
        volatility = features.get("volatility", {})
        
        # Run each predictor
        self._predict_stop_hunt(features)
        self._predict_fake_breakout(asian_range)
        self._predict_round_number_games(round_numbers)
        self._predict_session_manipulation(session_details)
        self._predict_accumulation_distribution(features)
        
        # Sort by probability
        self.predictions.sort(key=lambda x: x.probability, reverse=True)
        
        logger.info(f"🎯 MM Predictions: {len(self.predictions)}")
        for p in self.predictions[:2]:
            logger.info(f"   • {p.tactic.value}: {p.probability}% → {p.direction_after}")
        
        return self.predictions
    
    # ═══════════════════════════════════════════════════════════════════
    # STOP HUNT PREDICTION
    # ═══════════════════════════════════════════════════════════════════
    
    def _predict_stop_hunt(self, features: Dict):
        """Predict where MMs will hunt stops."""
        
        momentum = features.get("momentum", {})
        rsi = momentum.get("rsi_14", 50)
        
        # If recent swing low is obvious, expect stop hunt below
        # If recent swing high is obvious, expect stop hunt above
        
        # Use support/resistance levels
        if features.get("session") in ["LONDON", "NEW_YORK", "OVERLAP_LONDON_NY"]:
            # High activity sessions = more manipulation
            
            if rsi < 40:
                # Market is bearish, MMs might push lower to grab stops
                # then reverse up
                prediction = MMPrediction(
                    tactic=MMTactic.STOP_HUNT_BELOW,
                    probability=65,
                    direction_after="BUY",
                    target_level=self.current_price - 10,  # Sweep $10 below
                    description="RSI bearish, expect stop hunt below recent lows. "
                               "MMs will push price down to grab stops, then reverse.",
                    counter_strategy="Wait for sweep candle (long wick down), "
                                    "then BUY on close above wick low."
                )
                self.predictions.append(prediction)
                
            elif rsi > 60:
                # Market is bullish, MMs might push higher to grab stops
                # then reverse down
                prediction = MMPrediction(
                    tactic=MMTactic.STOP_HUNT_ABOVE,
                    probability=65,
                    direction_after="SELL",
                    target_level=self.current_price + 10,
                    description="RSI bullish, expect stop hunt above recent highs. "
                               "MMs will push price up to grab stops, then reverse.",
                    counter_strategy="Wait for sweep candle (long wick up), "
                                    "then SELL on close below wick high."
                )
                self.predictions.append(prediction)
    
    # ═══════════════════════════════════════════════════════════════════
    # FAKE BREAKOUT PREDICTION
    # ═══════════════════════════════════════════════════════════════════
    
    def _predict_fake_breakout(self, asian_range: Dict):
        """Predict fake breakout scenarios."""
        
        high = asian_range.get("high", 0)
        low = asian_range.get("low", 0)
        range_type = asian_range.get("range_type", "")
        
        if range_type == "TIGHT":
            # Tight range = breakout expected
            # But MMs often fake the first breakout
            
            prediction = MMPrediction(
                tactic=MMTactic.FAKE_BREAKOUT_UP,
                probability=55,
                direction_after="SELL",
                target_level=high + 5,
                description=f"Tight Asian range. First breakout above ${high:.2f} "
                           f"may be fake. Wait for retest or failure.",
                counter_strategy="Don't chase initial breakout. Wait for price "
                                "to break, fail, and close back inside range."
            )
            self.predictions.append(prediction)
            
            prediction = MMPrediction(
                tactic=MMTactic.FAKE_BREAKOUT_DOWN,
                probability=55,
                direction_after="BUY",
                target_level=low - 5,
                description=f"Tight Asian range. First breakout below ${low:.2f} "
                           f"may be fake. Wait for retest or failure.",
                counter_strategy="Don't chase initial breakout. Wait for price "
                                "to break, fail, and close back inside range."
            )
            self.predictions.append(prediction)
    
    # ═══════════════════════════════════════════════════════════════════
    # ROUND NUMBER GAMES
    # ═══════════════════════════════════════════════════════════════════
    
    def _predict_round_number_games(self, round_numbers: Dict):
        """Predict MM behavior around round numbers."""
        
        nearest = round_numbers.get("nearest", 0)
        is_near = round_numbers.get("is_near", False)
        position = round_numbers.get("position", "")
        level_type = round_numbers.get("type", "")
        
        if not is_near:
            return
        
        # MMs use round numbers as magnets and trap zones
        if level_type == "MAJOR":
            probability = 75
        else:
            probability = 60
        
        if position == "BELOW":
            # Price below round number = expect push up to grab stops above
            prediction = MMPrediction(
                tactic=MMTactic.ROUND_NUMBER_MAGNET,
                probability=probability,
                direction_after="SELL",  # After touching, expect rejection
                target_level=nearest,
                description=f"Price below ${nearest:.0f}. MMs will likely push "
                           f"price up to touch/sweep round number, then reject.",
                counter_strategy=f"Wait for price to spike above ${nearest:.0f}, "
                                f"then look for rejection candle to SELL."
            )
            self.predictions.append(prediction)
            
        elif position == "ABOVE":
            prediction = MMPrediction(
                tactic=MMTactic.ROUND_NUMBER_MAGNET,
                probability=probability,
                direction_after="BUY",
                target_level=nearest,
                description=f"Price above ${nearest:.0f}. MMs will likely push "
                           f"price down to touch/sweep round number, then bounce.",
                counter_strategy=f"Wait for price to dip below ${nearest:.0f}, "
                                f"then look for rejection candle to BUY."
            )
            self.predictions.append(prediction)
    
    # ═══════════════════════════════════════════════════════════════════
    # SESSION MANIPULATION
    # ═══════════════════════════════════════════════════════════════════
    
    def _predict_session_manipulation(self, session_details: Dict):
        """Predict manipulation at session opens."""
        
        is_london_open = session_details.get("is_london_open", False)
        is_ny_open = session_details.get("is_ny_open", False)
        minutes_in = session_details.get("minutes_into_session", 0)
        
        if is_london_open and minutes_in < 30:
            # London open is famous for stop hunts
            prediction = MMPrediction(
                tactic=MMTactic.SESSION_OPEN_SWEEP,
                probability=70,
                direction_after="REVERSE",  # Opposite of initial move
                target_level=0,  # Unknown
                description="London open - expect initial fake move. "
                           "MMs sweep one direction, then reverse. "
                           "Wait 15-30 min for true direction.",
                counter_strategy="DO NOT TRADE first 15 minutes of London. "
                                "Watch the fake move, then trade the reversal."
            )
            self.predictions.append(prediction)
            
        elif is_ny_open and minutes_in < 30:
            prediction = MMPrediction(
                tactic=MMTactic.SESSION_OPEN_SWEEP,
                probability=65,
                direction_after="REVERSE",
                target_level=0,
                description="NY open - expect volatility and potential fake move. "
                           "MMs may reverse London direction or continue strongly.",
                counter_strategy="Wait for NY to establish direction. "
                                "If NY confirms London, add to winners. "
                                "If NY reverses, trade the new direction."
            )
            self.predictions.append(prediction)
    
    # ═══════════════════════════════════════════════════════════════════
    # ACCUMULATION / DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════
    
    def _predict_accumulation_distribution(self, features: Dict):
        """Predict if MMs are accumulating or distributing."""
        
        session = features.get("session", "")
        volatility = features.get("volatility", {})
        regime = volatility.get("regime", "NORMAL")
        
        if session == "ASIA" and regime == "QUIET":
            # Quiet Asia = accumulation
            prediction = MMPrediction(
                tactic=MMTactic.ACCUMULATION,
                probability=60,
                direction_after="BREAKOUT",  # Pending direction
                target_level=0,
                description="Quiet Asian session with low volatility. "
                           "MMs likely accumulating positions for London move.",
                counter_strategy="Don't trade during accumulation. "
                                "Wait for London breakout to reveal direction."
            )
            self.predictions.append(prediction)
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_highest_probability_prediction(self) -> Optional[MMPrediction]:
        """Get the most likely MM move."""
        if not self.predictions:
            return None
        return self.predictions[0]
    
    def get_actionable_prediction(self) -> Optional[MMPrediction]:
        """Get a prediction with clear trade direction."""
        for p in self.predictions:
            if p.direction_after in ["BUY", "SELL"]:
                return p
        return None
    
    def format_for_llm(self) -> str:
        """Format predictions for LLM context."""
        if not self.predictions:
            return "No MM predictions available."
        
        lines = ["🎯 MARKET MAKER PREDICTION (WWCD):", ""]
        
        for i, p in enumerate(self.predictions[:3]):
            lines.append(f"{i+1}. {p.tactic.value.upper()} ({p.probability}% probability)")
            lines.append(f"   → Expected direction after: {p.direction_after}")
            lines.append(f"   → {p.description}")
            lines.append(f"   → Counter: {p.counter_strategy}")
            lines.append("")
        
        return "\n".join(lines)
