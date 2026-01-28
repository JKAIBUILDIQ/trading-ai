"""
Crellastein Aggressive DCA System
Pyramid DCA with tiered take-profit and no stop-loss strategy

For assets with strong conviction (XAUUSD, IREN):
- Build position on pullbacks
- Pyramid lot sizes
- Multiple take-profit levels
- Trust the composite signal (no SL)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AggressiveDCA")


@dataclass
class DCALevel:
    level: int
    entry_price: float
    lot_size: float
    entry_time: str
    status: str  # ACTIVE, CLOSED


@dataclass
class DCAPosition:
    symbol: str
    direction: str  # LONG or SHORT
    levels: List[DCALevel]
    avg_entry: float
    total_lots: float
    unrealized_pnl: float
    tp_levels_hit: List[int]
    created_at: str
    updated_at: str


@dataclass
class TakeProfitLevel:
    level: int
    target_pips: float
    close_percent: float
    hit: bool
    hit_at: Optional[str]


class AggressiveDCAManager:
    """
    Manages aggressive DCA positions with pyramid sizing
    """
    
    # Asset-specific DCA configurations
    DCA_CONFIG = {
        "XAUUSD": {
            "ticker": "GC=F",
            "pip_value": 0.01,  # 1 pip = $0.01 for gold
            "base_lots": 0.1,
            "dca_trigger_pips": 30,  # DCA every 30 pips pullback
            "lot_multiplier": 1.5,  # Each DCA = 1.5x previous
            "max_total_lots": 10.0,
            "max_dca_levels": 5,
            "no_stop_loss": True,  # Trust the signal!
            "emergency_sl_pips": 200,  # Only for flash crash
            "take_profit_levels": [
                {"level": 1, "pips": 30, "close_pct": 20},
                {"level": 2, "pips": 60, "close_pct": 30},
                {"level": 3, "pips": 100, "close_pct": 30},
                {"level": 4, "pips": 150, "close_pct": 20},
            ],
            "trail_after_tp": 2,  # Start trailing after TP2
            "trailing_pips": 20,
        },
        "IREN": {
            "ticker": "IREN",
            "pip_value": 0.01,  # $0.01 per pip
            "base_lots": 100,  # 100 shares base
            "dca_trigger_pips": 500,  # 5% = ~$2.50 = 250 pips at $50
            "lot_multiplier": 1.5,
            "max_total_lots": 2000,  # 2000 shares max
            "max_dca_levels": 5,
            "no_stop_loss": True,
            "emergency_sl_pips": 2000,  # 20% emergency
            "take_profit_levels": [
                {"level": 1, "pips": 500, "close_pct": 20},   # +$5
                {"level": 2, "pips": 1000, "close_pct": 25},  # +$10
                {"level": 3, "pips": 2000, "close_pct": 25},  # +$20
                {"level": 4, "pips": 5000, "close_pct": 30},  # +$50 (target $100+)
            ],
            "trail_after_tp": 2,
            "trailing_pips": 300,  # $3 trailing
        },
        "CLSK": {
            "ticker": "CLSK",
            "pip_value": 0.01,  # $0.01 per pip
            "base_lots": 100,  # 100 shares base
            "dca_trigger_pips": 65,  # ~5% at $13 = $0.65
            "lot_multiplier": 1.5,
            "max_total_lots": 2000,  # 2000 shares max
            "max_dca_levels": 5,
            "no_stop_loss": True,
            "emergency_sl_pips": 260,  # 20% emergency at $13 = $2.60
            "take_profit_levels": [
                {"level": 1, "pips": 100, "close_pct": 20},   # +$1
                {"level": 2, "pips": 200, "close_pct": 25},   # +$2
                {"level": 3, "pips": 500, "close_pct": 25},   # +$5
                {"level": 4, "pips": 1000, "close_pct": 30},  # +$10
            ],
            "trail_after_tp": 2,
            "trailing_pips": 50,  # $0.50 trailing
        },
        "CIFR": {
            "ticker": "CIFR",
            "pip_value": 0.01,  # $0.01 per pip
            "base_lots": 100,  # 100 shares base
            "dca_trigger_pips": 90,  # ~5% at $18 = $0.90
            "lot_multiplier": 1.5,
            "max_total_lots": 2000,  # 2000 shares max
            "max_dca_levels": 5,
            "no_stop_loss": True,
            "emergency_sl_pips": 360,  # 20% emergency at $18 = $3.60
            "take_profit_levels": [
                {"level": 1, "pips": 100, "close_pct": 20},   # +$1
                {"level": 2, "pips": 200, "close_pct": 25},   # +$2
                {"level": 3, "pips": 500, "close_pct": 25},   # +$5
                {"level": 4, "pips": 1000, "close_pct": 30},  # +$10
            ],
            "trail_after_tp": 2,
            "trailing_pips": 60,  # $0.60 trailing
        }
    }
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.config = self.DCA_CONFIG.get(symbol, self.DCA_CONFIG["XAUUSD"])
        self.state_file = Path(__file__).parent / "reports" / f"{symbol}_dca_state.json"
        self.position: Optional[DCAPosition] = None
        self._load_state()
    
    def _load_state(self):
        """Load DCA state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    if data:
                        levels = [DCALevel(**l) for l in data.get('levels', [])]
                        self.position = DCAPosition(
                            symbol=data['symbol'],
                            direction=data['direction'],
                            levels=levels,
                            avg_entry=data['avg_entry'],
                            total_lots=data['total_lots'],
                            unrealized_pnl=data['unrealized_pnl'],
                            tp_levels_hit=data.get('tp_levels_hit', []),
                            created_at=data['created_at'],
                            updated_at=data['updated_at']
                        )
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save DCA state to file"""
        if self.position:
            data = {
                'symbol': self.position.symbol,
                'direction': self.position.direction,
                'levels': [asdict(l) for l in self.position.levels],
                'avg_entry': self.position.avg_entry,
                'total_lots': self.position.total_lots,
                'unrealized_pnl': self.position.unrealized_pnl,
                'tp_levels_hit': self.position.tp_levels_hit,
                'created_at': self.position.created_at,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            data = {}
        
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = yf.Ticker(self.config["ticker"])
            hist = ticker.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting price: {e}")
        return 0.0
    
    def calculate_avg_entry(self) -> float:
        """Calculate average entry price"""
        if not self.position or not self.position.levels:
            return 0.0
        
        total_value = sum(l.entry_price * l.lot_size for l in self.position.levels if l.status == 'ACTIVE')
        total_lots = sum(l.lot_size for l in self.position.levels if l.status == 'ACTIVE')
        
        return total_value / total_lots if total_lots > 0 else 0.0
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if not self.position:
            return 0.0
        
        avg_entry = self.calculate_avg_entry()
        total_lots = sum(l.lot_size for l in self.position.levels if l.status == 'ACTIVE')
        
        if self.position.direction == "LONG":
            pnl_pips = (current_price - avg_entry) / self.config["pip_value"]
        else:
            pnl_pips = (avg_entry - current_price) / self.config["pip_value"]
        
        return pnl_pips * total_lots
    
    def open_initial_position(self, direction: str, composite_score: float) -> Dict:
        """Open initial DCA position based on composite signal"""
        if self.position and self.position.levels:
            return {"status": "error", "message": "Position already exists"}
        
        current_price = self.get_current_price()
        if current_price == 0:
            return {"status": "error", "message": "Cannot get current price"}
        
        # Adjust lot size based on signal strength
        base_lots = self.config["base_lots"]
        if composite_score >= 0.75:
            base_lots *= 1.5  # Strong signal = larger initial
        elif composite_score < 0.60:
            base_lots *= 0.5  # Weak signal = smaller initial
        
        initial_level = DCALevel(
            level=0,
            entry_price=current_price,
            lot_size=base_lots,
            entry_time=datetime.now(timezone.utc).isoformat(),
            status="ACTIVE"
        )
        
        self.position = DCAPosition(
            symbol=self.symbol,
            direction=direction,
            levels=[initial_level],
            avg_entry=current_price,
            total_lots=base_lots,
            unrealized_pnl=0.0,
            tp_levels_hit=[],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        
        self._save_state()
        
        logger.info(f"🚀 INITIAL POSITION: {self.symbol} {direction}")
        logger.info(f"   Entry: ${current_price:.2f}")
        logger.info(f"   Lots: {base_lots}")
        logger.info(f"   Signal strength: {composite_score*100:.1f}%")
        
        return {
            "status": "success",
            "action": "OPEN_INITIAL",
            "symbol": self.symbol,
            "direction": direction,
            "entry_price": current_price,
            "lots": base_lots,
            "composite_score": composite_score
        }
    
    def check_dca_trigger(self, composite_score: float) -> Dict:
        """Check if DCA should be triggered"""
        if not self.position:
            return {"status": "no_position", "dca_triggered": False}
        
        current_price = self.get_current_price()
        avg_entry = self.calculate_avg_entry()
        total_lots = self.position.total_lots
        current_level = len([l for l in self.position.levels if l.status == 'ACTIVE'])
        
        # Calculate pullback in pips
        if self.position.direction == "LONG":
            pullback_pips = (avg_entry - current_price) / self.config["pip_value"]
        else:
            pullback_pips = (current_price - avg_entry) / self.config["pip_value"]
        
        # Check if DCA should trigger
        required_pullback = self.config["dca_trigger_pips"] * (current_level + 1)
        
        result = {
            "status": "checked",
            "current_price": current_price,
            "avg_entry": avg_entry,
            "pullback_pips": pullback_pips,
            "required_pullback": required_pullback,
            "current_level": current_level,
            "max_level": self.config["max_dca_levels"],
            "composite_score": composite_score,
            "dca_triggered": False
        }
        
        # DCA conditions
        if current_level >= self.config["max_dca_levels"]:
            result["message"] = "Max DCA levels reached"
            return result
        
        if total_lots >= self.config["max_total_lots"]:
            result["message"] = "Max lots reached"
            return result
        
        # Only DCA if composite still bullish (for LONG) or bearish (for SHORT)
        if self.position.direction == "LONG" and composite_score < 0.50:
            result["message"] = "Composite signal no longer bullish"
            return result
        elif self.position.direction == "SHORT" and composite_score > 0.50:
            result["message"] = "Composite signal no longer bearish"
            return result
        
        if pullback_pips >= required_pullback:
            result["dca_triggered"] = True
            result["message"] = "DCA triggered!"
        else:
            result["message"] = f"Need {required_pullback - pullback_pips:.1f} more pips pullback"
        
        return result
    
    def execute_dca(self, composite_score: float) -> Dict:
        """Execute DCA pyramid entry"""
        check = self.check_dca_trigger(composite_score)
        if not check["dca_triggered"]:
            return check
        
        current_price = self.get_current_price()
        current_level = len([l for l in self.position.levels if l.status == 'ACTIVE'])
        
        # Calculate pyramid lot size
        if self.position.levels:
            last_lot = self.position.levels[-1].lot_size
        else:
            last_lot = self.config["base_lots"]
        
        dca_lots = last_lot * self.config["lot_multiplier"]
        
        # Cap at max
        total_after = self.position.total_lots + dca_lots
        if total_after > self.config["max_total_lots"]:
            dca_lots = self.config["max_total_lots"] - self.position.total_lots
        
        if dca_lots <= 0:
            return {"status": "error", "message": "No lots available for DCA"}
        
        # Create new DCA level
        new_level = DCALevel(
            level=current_level + 1,
            entry_price=current_price,
            lot_size=dca_lots,
            entry_time=datetime.now(timezone.utc).isoformat(),
            status="ACTIVE"
        )
        
        self.position.levels.append(new_level)
        self.position.total_lots += dca_lots
        self.position.avg_entry = self.calculate_avg_entry()
        self._save_state()
        
        logger.info(f"🔥 AGGRESSIVE DCA LEVEL {current_level + 1}")
        logger.info(f"   Price: ${current_price:.2f}")
        logger.info(f"   Lots: {dca_lots} ({self.config['lot_multiplier']}x pyramid)")
        logger.info(f"   Total Position: {self.position.total_lots} lots")
        logger.info(f"   New Avg Entry: ${self.position.avg_entry:.2f}")
        
        return {
            "status": "success",
            "action": "DCA_EXECUTED",
            "level": current_level + 1,
            "entry_price": current_price,
            "lots": dca_lots,
            "total_lots": self.position.total_lots,
            "new_avg_entry": self.position.avg_entry,
            "composite_score": composite_score
        }
    
    def check_take_profit(self) -> Dict:
        """Check and execute tiered take profits"""
        if not self.position:
            return {"status": "no_position"}
        
        current_price = self.get_current_price()
        avg_entry = self.position.avg_entry
        
        # Calculate profit in pips
        if self.position.direction == "LONG":
            profit_pips = (current_price - avg_entry) / self.config["pip_value"]
        else:
            profit_pips = (avg_entry - current_price) / self.config["pip_value"]
        
        result = {
            "status": "checked",
            "current_price": current_price,
            "avg_entry": avg_entry,
            "profit_pips": profit_pips,
            "tp_hit": None,
            "close_percent": 0,
            "tp_levels_hit": self.position.tp_levels_hit.copy()
        }
        
        # Check each TP level
        for tp in self.config["take_profit_levels"]:
            if tp["level"] not in self.position.tp_levels_hit and profit_pips >= tp["pips"]:
                result["tp_hit"] = tp["level"]
                result["close_percent"] = tp["close_pct"]
                self.position.tp_levels_hit.append(tp["level"])
                self._save_state()
                
                logger.info(f"💰 TP{tp['level']} HIT! +{profit_pips:.1f} pips")
                logger.info(f"   Closing {tp['close_pct']}% of position")
                break
        
        return result
    
    def close_position(self, reason: str = "Manual") -> Dict:
        """Close entire position"""
        if not self.position:
            return {"status": "no_position"}
        
        current_price = self.get_current_price()
        final_pnl = self.calculate_unrealized_pnl(current_price)
        
        result = {
            "status": "closed",
            "symbol": self.symbol,
            "direction": self.position.direction,
            "avg_entry": self.position.avg_entry,
            "exit_price": current_price,
            "total_lots": self.position.total_lots,
            "dca_levels": len(self.position.levels),
            "final_pnl_pips": final_pnl,
            "tp_levels_hit": self.position.tp_levels_hit,
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"📊 POSITION CLOSED: {self.symbol}")
        logger.info(f"   Entry: ${self.position.avg_entry:.2f}")
        logger.info(f"   Exit: ${current_price:.2f}")
        logger.info(f"   P&L: {final_pnl:.1f} pips")
        logger.info(f"   Reason: {reason}")
        
        self.position = None
        self._save_state()
        
        return result
    
    def get_position_status(self) -> Dict:
        """Get current position status"""
        if not self.position:
            return {
                "status": "no_position",
                "symbol": self.symbol,
                "has_position": False
            }
        
        current_price = self.get_current_price()
        unrealized_pnl = self.calculate_unrealized_pnl(current_price)
        
        return {
            "status": "active",
            "symbol": self.symbol,
            "has_position": True,
            "direction": self.position.direction,
            "current_price": current_price,
            "avg_entry": self.position.avg_entry,
            "total_lots": self.position.total_lots,
            "dca_levels": len([l for l in self.position.levels if l.status == 'ACTIVE']),
            "max_dca_levels": self.config["max_dca_levels"],
            "unrealized_pnl_pips": unrealized_pnl,
            "tp_levels_hit": self.position.tp_levels_hit,
            "take_profit_config": self.config["take_profit_levels"],
            "created_at": self.position.created_at,
            "levels": [asdict(l) for l in self.position.levels]
        }


# Convenience functions
def get_xauusd_dca_status() -> Dict:
    """Get XAUUSD DCA position status"""
    manager = AggressiveDCAManager("XAUUSD")
    return manager.get_position_status()


def get_iren_dca_status() -> Dict:
    """Get IREN DCA position status"""
    manager = AggressiveDCAManager("IREN")
    return manager.get_position_status()


def get_clsk_dca_status() -> Dict:
    """Get CLSK DCA position status"""
    manager = AggressiveDCAManager("CLSK")
    return manager.get_position_status()


def get_cifr_dca_status() -> Dict:
    """Get CIFR DCA position status"""
    manager = AggressiveDCAManager("CIFR")
    return manager.get_position_status()


if __name__ == "__main__":
    print("=" * 70)
    print("🔥 AGGRESSIVE DCA MANAGER - Testing")
    print("=" * 70)
    
    for symbol in ["XAUUSD", "IREN", "CLSK", "CIFR"]:
        print(f"\n📊 {symbol} DCA Status:")
        manager = AggressiveDCAManager(symbol)
        status = manager.get_position_status()
        print(json.dumps(status, indent=2))
