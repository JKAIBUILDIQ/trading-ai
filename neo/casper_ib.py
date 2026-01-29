"""
Casper IB - Ghost Commander's Conservative Companion
For IB Gold Futures (MGC)

Casper is more conservative than Ghost:
- Tighter DCA levels
- Smaller position sizes
- Lower take profit targets
- No free roll runner
- Best for ranging/quiet markets

Author: QUINN001
Created: 2026-01-28
"""

from ghost_commander_ib import GhostCommanderIB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Casper")


class CasperIB(GhostCommanderIB):
    """
    Casper - Conservative Gold Futures Strategy
    
    Differences from Ghost:
    - Tighter DCA spacing (smaller drops)
    - Fewer contracts per level
    - Smaller TP targets
    - No runner (takes profits quickly)
    - Lower max contracts
    """
    
    def __init__(self, paper_trading: bool = True, client_id: int = 52):
        super().__init__(paper_trading=paper_trading, client_id=client_id)
        self.name = "CASPER"
        
        # ═══════════════════════════════════════════════════════════════
        # CASPER SETTINGS (More Conservative)
        # ═══════════════════════════════════════════════════════════════
        
        self.settings.update({
            # Tighter DCA (smaller drops trigger)
            'dca_enabled': True,
            'dca_levels': 4,                              # Fewer levels
            'dca_drop_pips': [0, 30, 60, 100],            # Tighter spacing
            'dca_contracts': [1, 1, 1, 2],                # Smaller sizes
            
            # Lower position limits
            'max_contracts': 5,
            'max_daily_trades': 5,
            
            # Smaller take profits (scalping style)
            'tp_pips': 30,                   # $3 take profit
            'partial_tp_enabled': False,     # Take full profit
            
            # NO RUNNER (Casper is conservative)
            'runner_enabled': False,
            
            # Tighter risk management
            'stop_loss_pips': 150,           # $15 stop
            'trailing_stop_enabled': True,
            'trailing_stop_pips': 20,        # $2 trail
            
            # Regime - Casper prefers quiet markets
            'regime_atr_threshold': 15,      # Lower threshold
        })
        
        # Update state file for Casper
        from pathlib import Path
        STATE_DIR = Path(__file__).parent / "ghost_data"
        self.state_file = STATE_DIR / "casper_state.json"
        self._load_state()
    
    def _on_regime_change(self, old: str, new: str):
        """Casper handles regime changes conservatively"""
        logger.info(f"👻 CASPER: Regime changed {old} → {new}")
        
        if new == "TRENDING":
            # In trending - Casper steps back
            self.settings['tp_pips'] = 40
            self.settings['max_contracts'] = 3
            logger.info("   Reducing exposure in trending market")
        else:
            # In quiet - Casper's comfort zone
            self.settings['tp_pips'] = 30
            self.settings['max_contracts'] = 5
            logger.info("   Quiet market - normal operations")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("👻 CASPER IB - Conservative Gold Futures Strategy")
    print("=" * 70)
    
    casper = CasperIB(paper_trading=True, client_id=52)
    
    if casper.connect():
        try:
            casper.run(check_interval=30)
        finally:
            casper.disconnect()
    else:
        print("❌ Failed to connect")
