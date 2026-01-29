#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    TWS CONNECTOR - Reliable IBKR Connection
═══════════════════════════════════════════════════════════════════════════════

Centralized connection management:
- Auto-loads config from tws_config.json
- Tries fallback hosts if primary fails
- Auto-reconnect on disconnect
- Health checks
- All bots use this for consistent connection

Usage:
    from tws_connector import TWSConnector
    
    connector = TWSConnector()
    ib = connector.connect()  # Returns connected IB instance
    
    # Or use context manager
    with TWSConnector() as ib:
        positions = ib.positions()

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional
from ib_insync import IB, Future
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TWSConnector')

CONFIG_FILE = Path(__file__).parent / 'tws_config.json'


class TWSConnector:
    """
    Reliable TWS connection manager with auto-retry and fallback
    """
    
    def __init__(self, client_id: int = None):
        self.config = self._load_config()
        self.client_id = client_id or self.config['connection']['client_id_base']
        self.ib = IB()
        self._connected = False
    
    def _load_config(self) -> dict:
        """Load connection config"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            # Default config
            return {
                'connection': {
                    'host': '100.119.161.65',
                    'port': 7497,
                    'client_id_base': 100,
                    'timeout': 15,
                },
                'fallback_hosts': ['100.119.161.65', '127.0.0.1'],
            }
    
    def connect(self, max_retries: int = 3) -> Optional[IB]:
        """
        Connect to TWS with auto-retry and fallback hosts
        """
        hosts = self.config.get('fallback_hosts', [self.config['connection']['host']])
        port = self.config['connection']['port']
        timeout = self.config['connection']['timeout']
        
        for attempt in range(max_retries):
            for host in hosts:
                try:
                    logger.info(f"Connecting to TWS at {host}:{port} (attempt {attempt + 1})...")
                    
                    if self.ib.isConnected():
                        self.ib.disconnect()
                    
                    self.ib.connect(host, port, clientId=self.client_id, timeout=timeout)
                    
                    if self.ib.isConnected():
                        logger.info(f"✅ Connected to TWS at {host}:{port}")
                        self._connected = True
                        return self.ib
                        
                except Exception as e:
                    logger.warning(f"Failed to connect to {host}:{port}: {e}")
                    continue
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
        
        logger.error("❌ Failed to connect to TWS after all retries")
        return None
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from TWS")
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.ib.isConnected()
    
    def ensure_connected(self) -> bool:
        """Ensure connection is alive, reconnect if needed"""
        if not self.ib.isConnected():
            logger.warning("Connection lost, attempting reconnect...")
            return self.connect() is not None
        return True
    
    def get_mgc_contract(self) -> Future:
        """Get properly configured MGC contract"""
        cfg = self.config.get('contract', {})
        return Future(
            conId=cfg.get('con_id', 706903676),
            symbol=cfg.get('symbol', 'MGC'),
            lastTradeDateOrContractMonth=cfg.get('last_trade_date', '20260428'),
            exchange=cfg.get('exchange', 'COMEX'),
            multiplier=cfg.get('multiplier', '10'),
            currency='USD',
            localSymbol=cfg.get('local_symbol', 'MGCJ6'),
            tradingClass='MGC',
        )
    
    def get_position(self, symbol: str = 'MGC') -> dict:
        """Get current position for symbol"""
        if not self.ensure_connected():
            return {'error': 'Not connected'}
        
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == symbol:
                return {
                    'symbol': symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost / 10,  # MGC multiplier
                    'contract': pos.contract,
                }
        return {'symbol': symbol, 'position': 0, 'avg_cost': 0}
    
    def get_price(self, symbol: str = 'MGC') -> Optional[float]:
        """Get current price"""
        if not self.ensure_connected():
            return None
        
        contract = self.get_mgc_contract()
        self.ib.reqMarketDataType(3)  # Delayed data OK
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(2)
        
        price = ticker.last or ticker.close
        self.ib.cancelMktData(contract)
        return price
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self.ib
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def queue_backup_command(action: str, quantity: int, note: str = ''):
    """
    Queue a command for desktop backup execution when H100 can't connect
    """
    backup_file = Path(__file__).parent / 'desktop_backup' / 'pending_commands.json'
    
    if backup_file.exists():
        with open(backup_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'commands': []}
    
    data['commands'].append({
        'action': action.upper(),
        'quantity': quantity,
        'note': note,
        'status': 'PENDING',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })
    data['last_update'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"📋 Queued backup command: {action} {quantity} MGC")
    
    # Generate prompt for Crella
    prompt = f"""
═══════════════════════════════════════════════════════════════════════════════
⚠️  H100 CAN'T CONNECT TO TWS - SEND THIS TO DESKTOP CRELLA:
═══════════════════════════════════════════════════════════════════════════════

Execute this trade on desktop TWS:

  ACTION: {action.upper()} {quantity} MGC contracts
  NOTE: {note}

Quick command for Crella:
  python execute_local.py {action.lower()} {quantity}

Or tell Crella:
  "Sell {quantity} MGC contracts at market"

═══════════════════════════════════════════════════════════════════════════════
"""
    print(prompt)
    return prompt


def health_check():
    """Run a health check on TWS connection"""
    connector = TWSConnector(client_id=998)
    
    print("═" * 60)
    print("          TWS CONNECTION HEALTH CHECK")
    print("═" * 60)
    
    # Load config
    print(f"\n📋 Config: {CONFIG_FILE}")
    print(f"   Host: {connector.config['connection']['host']}")
    print(f"   Port: {connector.config['connection']['port']}")
    
    # Try to connect
    print(f"\n🔌 Attempting connection...")
    ib = connector.connect(max_retries=2)
    
    if ib and ib.isConnected():
        print(f"✅ Connected!")
        
        # Get position
        pos = connector.get_position('MGC')
        print(f"\n📊 MGC Position: {pos['position']} @ ${pos['avg_cost']:.2f}")
        
        # Get price
        price = connector.get_price()
        print(f"💰 Current Price: ${price}")
        
        connector.disconnect()
        print(f"\n✅ Health check PASSED")
    else:
        print(f"\n❌ Health check FAILED - Could not connect")
        print("\nTroubleshooting:")
        print("  1. Is TWS running on desktop-gringot?")
        print("  2. Is API enabled in TWS? (File → Global Config → API → Settings)")
        print("  3. Is 100.91.17.86 in Trusted IPs?")
        print("  4. Is Windows Firewall allowing port 7497?")
    
    print("═" * 60)


if __name__ == "__main__":
    health_check()
