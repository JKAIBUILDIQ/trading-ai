"""
Scout Swarm Coordinator
=======================
Coordinates all 5 sector scouts.
Runs during market hours on H100 using Ollama.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from .all_scouts import create_all_scouts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScoutSwarm")


class ScoutSwarmCoordinator:
    """
    Coordinates all 5 sector scouts.
    Runs during market hours on H100 using Ollama.
    """
    
    def __init__(self):
        self.scouts = create_all_scouts()
        self.running = False
        self.start_time: Optional[datetime] = None
        self.tasks: List[asyncio.Task] = []
    
    async def start_swarm(self):
        """Start all scouts in parallel."""
        
        self.running = True
        self.start_time = datetime.now()
        
        print("╔═══════════════════════════════════════════════════════════════════════╗")
        print("║              🦅  SECTOR SCOUT SWARM ACTIVATED  🦅                     ║")
        print("╠═══════════════════════════════════════════════════════════════════════╣")
        print("║                                                                        ║")
        print("║   SCOUT 1: TECH_TITAN       │  AAPL, NVDA, MSFT, META, GOOGL...      ║")
        print("║   SCOUT 2: ENERGY_EAGLE     │  XOM, CVX, COP, SLB, HAL...            ║")
        print("║   SCOUT 3: MINER_HAWK       │  IREN, CLSK, CIFR, MARA... (LONG ONLY)║")
        print("║   SCOUT 4: GROWTH_HUNTER    │  COIN, SHOP, PLTR, SNOW, NET...        ║")
        print("║   SCOUT 5: DEFENSE_FORTRESS │  LMT, RTX, GD, NOC, BA...              ║")
        print("║                                                                        ║")
        print("║   Engine: OLLAMA (llama3.1:8b) - FREE, LOCAL                          ║")
        print("║   Confidence Threshold: 75%                                           ║")
        print("║   Scan Interval: 5 minutes                                            ║")
        print("║                                                                        ║")
        print("╚═══════════════════════════════════════════════════════════════════════╝")
        
        # Create tasks for all scouts
        self.tasks = [
            asyncio.create_task(scout.start())
            for scout in self.scouts
        ]
        
        try:
            # Wait for all scouts (they run forever until stopped)
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("[SWARM] Received cancellation signal")
        finally:
            self.stop_swarm()
    
    def stop_swarm(self):
        """Stop all scouts."""
        
        logger.info("[SWARM] Stopping all scouts...")
        
        for scout in self.scouts:
            scout.stop()
        
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        self.running = False
        logger.info("[SWARM] All scouts deactivated")
    
    def get_stats(self) -> Dict:
        """Get swarm statistics."""
        
        total_scans = sum(s.stats['scans'] for s in self.scouts)
        total_symbols = sum(s.stats['symbols_checked'] for s in self.scouts)
        total_patterns = sum(s.stats['patterns_found'] for s in self.scouts)
        total_alerts = sum(s.stats['alerts_sent'] for s in self.scouts)
        total_errors = sum(s.stats['errors'] for s in self.scouts)
        
        uptime = None
        if self.start_time:
            uptime = str(datetime.now() - self.start_time).split('.')[0]
        
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': uptime,
            'scouts_active': len(self.scouts),
            'total_symbols_watched': sum(len(s.watchlist) for s in self.scouts),
            'totals': {
                'scans': total_scans,
                'symbols_checked': total_symbols,
                'patterns_found': total_patterns,
                'alerts_sent': total_alerts,
                'errors': total_errors,
            },
            'scouts': [s.get_stats() for s in self.scouts],
        }
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts from all scouts."""
        
        all_alerts = []
        for scout in self.scouts:
            all_alerts.extend(list(scout.alerts))
        
        # Sort by timestamp (newest first)
        all_alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return all_alerts[:limit]
    
    def get_scout_by_name(self, name: str):
        """Get a specific scout by name."""
        for scout in self.scouts:
            if scout.name == name:
                return scout
        return None


# Global instance for API access
_coordinator: Optional[ScoutSwarmCoordinator] = None


def get_coordinator() -> Optional[ScoutSwarmCoordinator]:
    """Get the global coordinator instance."""
    return _coordinator


def set_coordinator(coordinator: ScoutSwarmCoordinator):
    """Set the global coordinator instance."""
    global _coordinator
    _coordinator = coordinator


# === MAIN ENTRY POINT ===

async def run_scout_swarm():
    """Run the Scout Swarm."""
    
    coordinator = ScoutSwarmCoordinator()
    set_coordinator(coordinator)
    
    try:
        await coordinator.start_swarm()
    except KeyboardInterrupt:
        coordinator.stop_swarm()


if __name__ == "__main__":
    asyncio.run(run_scout_swarm())
