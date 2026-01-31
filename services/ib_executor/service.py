"""
IB Execution Service
====================
Main service that connects scouts/sentinel to IB execution.
"""

import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime

from .connection import IBConnection
from .executor import IBExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IBService")


class IBExecutionService:
    """
    Main service that connects scouts/sentinel to IB execution.
    """
    
    def __init__(self):
        self.connection = IBConnection()
        self.executor: Optional[IBExecutor] = None
        self.running = False
        self.start_time: Optional[datetime] = None
    
    async def start(self, paper: bool = True) -> bool:
        """Start the execution service (async)."""
        
        connected = await self.connection.connect(paper=paper)
        
        if not connected:
            logger.error("[IB SERVICE] Failed to connect to IB")
            return False
        
        self.executor = IBExecutor(self.connection)
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"[IB SERVICE] Started in {'PAPER' if paper else 'LIVE'} mode")
        logger.info(f"[IB SERVICE] Auto-execute: {'ON' if paper else 'OFF (confirmation required)'}")
        
        return True
    
    def start_sync(self, paper: bool = True) -> bool:
        """Start the execution service (synchronous)."""
        
        connected = self.connection.connect_sync(paper=paper)
        
        if not connected:
            logger.error("[IB SERVICE] Failed to connect to IB")
            return False
        
        self.executor = IBExecutor(self.connection)
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"[IB SERVICE] Started in {'PAPER' if paper else 'LIVE'} mode")
        
        return True
    
    def process_scout_alert(self, alert: dict) -> dict:
        """Process incoming scout alert."""
        
        if not self.running or self.executor is None:
            return {'status': 'not_running', 'message': 'IB service not connected'}
        
        logger.info(f"[IB SERVICE] Processing scout alert: {alert.get('symbol')} ({alert.get('confidence')}%)")
        
        return self.executor.execute_scout_alert(alert)
    
    def process_sentinel_alert(self, alert: dict) -> dict:
        """Process incoming sentinel alert."""
        
        if not self.running or self.executor is None:
            return {'status': 'not_running', 'message': 'IB service not connected'}
        
        logger.info(f"[IB SERVICE] Processing sentinel alert: {alert.get('patterns', [])}")
        
        return self.executor.execute_sentinel_alert(alert)
    
    def stop(self):
        """Stop the service."""
        
        self.running = False
        self.connection.disconnect()
        logger.info("[IB SERVICE] Stopped")
    
    def get_status(self) -> dict:
        """Get service status."""
        
        uptime = None
        if self.start_time:
            uptime = str(datetime.now() - self.start_time).split('.')[0]
        
        return {
            'running': self.running,
            'connected': self.connection.connected,
            'mode': 'PAPER' if self.connection.is_paper else 'LIVE',
            'account_info': self.connection.account_info,
            'uptime': uptime,
            'pending_orders': len(self.executor.pending_orders) if self.executor else 0,
            'executed_orders': len(self.executor.executed_orders) if self.executor else 0,
            'failed_orders': len(self.executor.failed_orders) if self.executor else 0,
        }
    
    def get_positions(self) -> list:
        """Get all positions."""
        if self.connection.connected:
            return self.connection.get_all_positions()
        return []
    
    def get_account(self) -> dict:
        """Get account summary."""
        if self.connection.connected:
            return self.connection.get_account_summary()
        return {}


# Global service instance
ib_service = IBExecutionService()
