"""
IB Connection Manager
=====================
Manages connection to Interactive Brokers TWS/Gateway.
Supports both Paper and Live trading modes.
"""

import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime

from ib_insync import IB, Stock, Forex, Future, Contract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IBConnection")


class IBConnection:
    """
    Manages connection to Interactive Brokers TWS/Gateway.
    Supports both Paper and Live trading modes.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        paper_port: int = 7497,
        live_port: int = 7496,
        client_id: int = 10,
    ):
        self.host = host
        self.paper_port = paper_port
        self.live_port = live_port
        self.client_id = client_id
        
        self.ib = IB()
        self.connected = False
        self.is_paper = True
        self.account_info = {}
    
    async def connect(self, paper: bool = True) -> bool:
        """Connect to IB TWS/Gateway."""
        
        self.is_paper = paper
        port = self.paper_port if paper else self.live_port
        
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=port,
                clientId=self.client_id,
                timeout=20,
            )
            
            self.connected = True
            
            # Get account info
            accounts = self.ib.managedAccounts()
            self.account_info = {
                'accounts': accounts,
                'mode': 'PAPER' if paper else 'LIVE',
                'port': port,
                'connected_at': datetime.now().isoformat(),
            }
            
            logger.info(f"[IB] Connected to {'PAPER' if paper else 'LIVE'} on port {port}")
            logger.info(f"[IB] Accounts: {accounts}")
            
            return True
            
        except Exception as e:
            logger.error(f"[IB] Connection failed: {e}")
            self.connected = False
            return False
    
    def connect_sync(self, paper: bool = True) -> bool:
        """Synchronous connect for non-async contexts."""
        
        self.is_paper = paper
        port = self.paper_port if paper else self.live_port
        
        try:
            self.ib.connect(
                host=self.host,
                port=port,
                clientId=self.client_id,
                timeout=20,
            )
            
            self.connected = True
            accounts = self.ib.managedAccounts()
            self.account_info = {
                'accounts': accounts,
                'mode': 'PAPER' if paper else 'LIVE',
                'port': port,
                'connected_at': datetime.now().isoformat(),
            }
            
            logger.info(f"[IB] Connected to {'PAPER' if paper else 'LIVE'} on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"[IB] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IB."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("[IB] Disconnected")
    
    def get_contract(self, symbol: str, sec_type: str = "STK") -> Optional[Contract]:
        """
        Get IB contract for a symbol.
        
        sec_type: STK (stock), FX (forex), FUT (futures)
        """
        
        # Normalize symbol
        symbol = symbol.replace("=F", "").replace("-USD", "").upper()
        
        if sec_type == "STK" or sec_type == "AUTO":
            # Check if it's a known forex/futures symbol
            if symbol in ["XAUUSD", "XAU", "GC", "GOLD"]:
                return Forex("XAUUSD")
            elif symbol in ["XAGUSD", "XAG", "SI", "SILVER"]:
                return Forex("XAGUSD")
            elif symbol in ["EURUSD", "EUR"]:
                return Forex("EURUSD")
            elif symbol in ["BTC", "BTCUSD"]:
                # Crypto - use futures or skip
                return None
            else:
                # Assume stock
                return Stock(symbol, "SMART", "USD")
        
        elif sec_type == "FX":
            if "XAU" in symbol or symbol == "GOLD":
                return Forex("XAUUSD")
            elif "XAG" in symbol or symbol == "SILVER":
                return Forex("XAGUSD")
            elif "EUR" in symbol:
                return Forex("EURUSD")
            return Forex(symbol)
        
        elif sec_type == "FUT":
            if symbol in ["GC", "GOLD", "XAU"]:
                return Future("GC", exchange="COMEX")
            elif symbol in ["SI", "SILVER", "XAG"]:
                return Future("SI", exchange="COMEX")
            return Future(symbol)
        
        return Stock(symbol, "SMART", "USD")
    
    def get_position(self, symbol: str) -> Dict:
        """Get current position for a symbol."""
        
        symbol = symbol.replace("=F", "").upper()
        
        try:
            positions = self.ib.positions()
            
            for pos in positions:
                if pos.contract.symbol.upper() == symbol:
                    return {
                        'symbol': symbol,
                        'quantity': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_value': getattr(pos, 'marketValue', 0),
                    }
        except Exception as e:
            logger.error(f"[IB] Error getting position: {e}")
        
        return {'symbol': symbol, 'quantity': 0, 'avg_cost': 0}
    
    def get_all_positions(self) -> List[Dict]:
        """Get all positions."""
        
        try:
            positions = self.ib.positions()
            return [
                {
                    'symbol': pos.contract.symbol,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'sec_type': pos.contract.secType,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"[IB] Error getting positions: {e}")
            return []
    
    def get_account_summary(self) -> Dict:
        """Get account summary."""
        
        try:
            summary = self.ib.accountSummary()
            
            result = {}
            for item in summary:
                result[item.tag] = item.value
            
            return result
        except Exception as e:
            logger.error(f"[IB] Error getting account: {e}")
            return {}
    
    def get_account_value(self) -> float:
        """Get total account value (NetLiquidation)."""
        
        summary = self.get_account_summary()
        return float(summary.get('NetLiquidation', 100000))
