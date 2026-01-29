"""
Crellastein IB - MT5 Strategies Ported to Interactive Brokers

Strategies:
- Ghost Commander: NEO signals + SuperTrend DCA ladder
- Casper: Meta Bot + Drop-Buy Martingale DCA
- Free Roller: Profit-funded runner system
- Counterstrike: Bearish deployment

Author: QUINN001
Ported: January 29, 2026
"""

from .indicators import Indicators
from .ghost_commander_ib import GhostCommanderIB
from .casper_ib import CasperIB
from .free_roller_ib import FreeRollerIB

__all__ = ['Indicators', 'GhostCommanderIB', 'CasperIB', 'FreeRollerIB']
__version__ = '1.0.0'
