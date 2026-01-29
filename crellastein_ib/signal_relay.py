#!/usr/bin/env python3
"""
MT5 → IB Signal Relay

Receives signals from MT5 Ghost/Casper EAs via HTTP webhook
and executes equivalent trades on Interactive Brokers.

MT5 (XAUUSD) → Relay → IB (MGC)

Author: Quinn (NEO Training)
"""

from flask import Flask, request, jsonify
from ib_insync import IB, MarketOrder, Future
from datetime import datetime
from pathlib import Path
import json
import logging
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

AUTH_TOKEN = os.environ.get('RELAY_AUTH_TOKEN', 'ghost-casper-relay-2026')
IB_HOST = os.environ.get('IB_HOST', '100.119.161.65')  # Desktop TWS via Tailscale
IB_PORT = int(os.environ.get('IB_PORT', '7497'))
CLIENT_ID = 600  # Unique client ID for relay

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/jbot/trading_ai/logs/signal_relay.log')
    ]
)
logger = logging.getLogger(__name__)

# Signal log for history
SIGNAL_LOG = Path('/home/jbot/trading_ai/crellastein_ib/signal_history.json')

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Global IB connection
ib = None


def connect_ib() -> bool:
    """Connect to IB TWS"""
    global ib
    try:
        if ib is not None and ib.isConnected():
            return True
        
        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=15)
        logger.info(f"✅ Connected to IB TWS at {IB_HOST}:{IB_PORT}")
        return True
    except Exception as e:
        logger.error(f"❌ IB connection failed: {e}")
        return False


def get_contract(symbol: str) -> Future:
    """Get IB contract for symbol"""
    if symbol in ['MGC', 'XAUUSD', 'GOLD']:
        return Future(conId=706903676, symbol='MGC', exchange='COMEX')
    elif symbol == 'GC':
        return Future(symbol='GC', exchange='COMEX')
    else:
        raise ValueError(f"Unknown symbol: {symbol}")


def lots_to_contracts(symbol: str, lots: float) -> int:
    """
    Convert MT5 lots to IB contracts.
    
    XAUUSD: 0.10 lot ≈ 1 MGC contract (at ~$2750/oz)
    - 0.10 lot XAUUSD = 10 oz exposure = ~$27,500
    - 1 MGC contract = 10 oz exposure = ~$27,500
    """
    if symbol in ['MGC', 'XAUUSD', 'GOLD']:
        contracts = max(1, int(round(lots * 10)))  # 0.10 lot = 1 contract
        return contracts
    return max(1, int(round(lots)))


def log_signal(signal: dict, result: dict):
    """Log signal to history file"""
    try:
        history = []
        if SIGNAL_LOG.exists():
            history = json.loads(SIGNAL_LOG.read_text())
        
        history.append({
            **signal,
            'result': result,
            'processed_at': datetime.now().isoformat()
        })
        
        # Keep last 1000 signals
        history = history[-1000:]
        SIGNAL_LOG.write_text(json.dumps(history, indent=2))
    except Exception as e:
        logger.error(f"Failed to log signal: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/signal', methods=['POST'])
def receive_signal():
    """
    Receive and execute signal from MT5.
    
    Expected JSON:
    {
        "action": "BUY" | "SELL" | "CLOSE",
        "symbol": "MGC" | "XAUUSD",
        "lots": 0.10,
        "price": 2750.50,
        "tp": 2780.00,
        "sl": 2730.00,
        "comment": "DROPBUY|L3|1",
        "source": "MT5_Ghost",
        "token": "auth-token"
    }
    """
    global ib
    
    try:
        data = request.json
        
        # Auth check
        if data.get('token') != AUTH_TOKEN:
            logger.warning(f"⚠️ Unauthorized signal attempt from {request.remote_addr}")
            return jsonify({"status": "error", "message": "unauthorized"}), 401
        
        # Parse signal
        action = data.get('action', '').upper()
        symbol = data.get('symbol', 'MGC')
        lots = float(data.get('lots', 0.1))
        price = float(data.get('price', 0))
        tp = float(data.get('tp', 0))
        sl = float(data.get('sl', 0))
        comment = data.get('comment', '')
        source = data.get('source', 'unknown')
        
        logger.info(f"📡 Signal received: {action} {symbol} {lots} lots @ {price} [{comment}] from {source}")
        
        # Connect to IB
        if not connect_ib():
            result = {"status": "error", "message": "IB connection failed"}
            log_signal(data, result)
            return jsonify(result), 500
        
        # Get contract and convert lots
        contract = get_contract(symbol)
        contracts = lots_to_contracts(symbol, lots)
        
        # Execute based on action
        trade = None
        
        if action == "BUY":
            order = MarketOrder('BUY', contracts)
            trade = ib.placeOrder(contract, order)
            logger.info(f"🟢 BUY {contracts} MGC submitted | Comment: {comment}")
            
        elif action == "SELL":
            order = MarketOrder('SELL', contracts)
            trade = ib.placeOrder(contract, order)
            logger.info(f"🔴 SELL {contracts} MGC submitted | Comment: {comment}")
            
        elif action == "CLOSE":
            # Close = reduce position
            positions = ib.positions()
            for pos in positions:
                if pos.contract.symbol == 'MGC':
                    if pos.position > 0:
                        # Long position - sell to close
                        close_qty = min(contracts, int(pos.position))
                        order = MarketOrder('SELL', close_qty)
                        trade = ib.placeOrder(contract, order)
                        logger.info(f"🔻 CLOSE (SELL) {close_qty} MGC | Comment: {comment}")
                    elif pos.position < 0:
                        # Short position - buy to close
                        close_qty = min(contracts, abs(int(pos.position)))
                        order = MarketOrder('BUY', close_qty)
                        trade = ib.placeOrder(contract, order)
                        logger.info(f"🔺 CLOSE (BUY) {close_qty} MGC | Comment: {comment}")
                    break
            else:
                logger.warning(f"⚠️ No MGC position to close")
                result = {"status": "warning", "message": "no position to close"}
                log_signal(data, result)
                return jsonify(result), 200
                
        elif action == "CLOSE_ALL_LONG":
            # Flatten all long positions
            positions = ib.positions()
            for pos in positions:
                if pos.contract.symbol == 'MGC' and pos.position > 0:
                    order = MarketOrder('SELL', int(pos.position))
                    trade = ib.placeOrder(contract, order)
                    logger.info(f"🔻 CLOSE ALL LONG: SELL {int(pos.position)} MGC")
                    
        elif action == "CLOSE_ALL_SHORT":
            # Flatten all short positions
            positions = ib.positions()
            for pos in positions:
                if pos.contract.symbol == 'MGC' and pos.position < 0:
                    order = MarketOrder('BUY', abs(int(pos.position)))
                    trade = ib.placeOrder(contract, order)
                    logger.info(f"🔺 CLOSE ALL SHORT: BUY {abs(int(pos.position))} MGC")
                    
        else:
            logger.error(f"❌ Unknown action: {action}")
            result = {"status": "error", "message": f"unknown action: {action}"}
            log_signal(data, result)
            return jsonify(result), 400
        
        # Wait briefly for order status
        if trade:
            ib.sleep(1)
            status = trade.orderStatus.status
        else:
            status = "no_trade"
        
        result = {
            "status": "success",
            "action": action,
            "symbol": "MGC",
            "contracts": contracts,
            "order_status": status,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        log_signal(data, result)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error processing signal: {e}")
        result = {"status": "error", "message": str(e)}
        log_signal(data if 'data' in dir() else {}, result)
        return jsonify(result), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    connected = ib is not None and ib.isConnected()
    return jsonify({
        "status": "ok",
        "service": "MT5-IB Signal Relay",
        "ib_connected": connected,
        "ib_host": IB_HOST,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/positions', methods=['GET'])
def get_positions():
    """Get current IB positions"""
    if not connect_ib():
        return jsonify({"status": "error", "message": "not connected"}), 500
    
    positions = ib.positions()
    return jsonify({
        "positions": [
            {
                "symbol": p.contract.symbol,
                "quantity": int(p.position),
                "avg_cost": p.avgCost / 10 if p.contract.symbol == 'MGC' else p.avgCost
            }
            for p in positions
            if p.contract.symbol in ['MGC', 'GC']
        ],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/history', methods=['GET'])
def get_history():
    """Get signal history"""
    try:
        if SIGNAL_LOG.exists():
            history = json.loads(SIGNAL_LOG.read_text())
            # Return last 50
            return jsonify({"signals": history[-50:]})
        return jsonify({"signals": []})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("  MT5 → IB Signal Relay Starting")
    logger.info(f"  IB Host: {IB_HOST}:{IB_PORT}")
    logger.info(f"  Listening on: 0.0.0.0:5000")
    logger.info("="*60)
    
    # Pre-connect to IB
    connect_ib()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
