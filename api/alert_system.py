"""
Alert System - Sends pattern alerts through multiple channels.

Channels:
- MongoDB (logging)
- Telegram notifications
- War Room API (for WebSocket push)
- Signal file (for Ghost/bots)

Includes cooldown to prevent duplicate alerts.
"""

import os
import json
import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlertSystem")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.trading_ai

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Signal file path
ALERT_SIGNAL_FILE = "/home/jbot/trading_ai/neo/signals/pattern_alerts.json"


class AlertSystem:
    """Send alerts through multiple channels."""
    
    def __init__(self):
        self.recent_alerts: Dict[str, datetime] = {}
        self.alert_cooldown = 3600  # 1 hour between same pattern alerts
        self.critical_cooldown = 1800  # 30 min for critical alerts
    
    def _get_cooldown(self, severity: str) -> int:
        """Get cooldown based on severity."""
        if severity == "CRITICAL":
            return self.critical_cooldown
        return self.alert_cooldown
    
    def _should_send(self, alert: Dict, symbol: str) -> bool:
        """Check if alert should be sent (not in cooldown)."""
        alert_key = f"{symbol}_{alert['pattern']}"
        
        if alert_key in self.recent_alerts:
            elapsed = (datetime.now() - self.recent_alerts[alert_key]).total_seconds()
            cooldown = self._get_cooldown(alert.get("severity", "MEDIUM"))
            if elapsed < cooldown:
                logger.debug(f"Alert {alert_key} in cooldown ({cooldown - elapsed:.0f}s remaining)")
                return False
        
        return True
    
    async def send_alert(self, alert: Dict, symbol: str) -> bool:
        """Send alert through all channels."""
        
        # Check cooldown
        if not self._should_send(alert, symbol):
            return False
        
        # Mark as sent
        alert_key = f"{symbol}_{alert['pattern']}"
        self.recent_alerts[alert_key] = datetime.now()
        
        # Add metadata
        alert["symbol"] = symbol
        alert["timestamp"] = datetime.now().isoformat()
        alert["alert_id"] = f"{symbol}_{alert['pattern']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Format message
        message = self.format_alert(alert)
        
        # Send through all channels (parallel)
        results = await asyncio.gather(
            self.save_to_db(alert),
            self.send_telegram(message, alert.get("severity", "MEDIUM")),
            self.save_signal_file(alert),
            self.push_to_war_room(alert),
            self.update_defcon_recommendation(alert),
            return_exceptions=True
        )
        
        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Alert channel {i} failed: {result}")
        
        logger.info(f"Alert sent: {alert['pattern']} on {symbol}")
        return True
    
    def format_alert(self, alert: Dict) -> str:
        """Format alert message for display."""
        severity_emoji = {
            "CRITICAL": "🚨🚨🚨",
            "HIGH": "⚠️",
            "MEDIUM": "📊",
            "LOW": "ℹ️",
        }
        
        direction_emoji = "🔻" if alert.get("direction") == "BEARISH" else "🔺"
        defcon_impact = alert.get("defcon_impact", 0)
        defcon_str = f"+{defcon_impact}" if defcon_impact > 0 else str(defcon_impact)
        
        emoji = severity_emoji.get(alert.get("severity", "MEDIUM"), "📊")
        
        message = f"""
{emoji} PATTERN ALERT {emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} {alert.get('pattern', 'UNKNOWN')} on {alert.get('symbol', 'N/A')}

{alert.get('message', 'Pattern detected')}

SEVERITY: {alert.get('severity', 'MEDIUM')}
DEFCON IMPACT: {defcon_str}
PRICE: ${alert.get('price', 0):.2f}

ACTION: {alert.get('action', 'Review chart')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return message.strip()
    
    async def save_to_db(self, alert: Dict) -> str:
        """Save alert to MongoDB."""
        try:
            # Remove non-serializable data
            alert_doc = {k: v for k, v in alert.items() if k != "candle"}
            alert_doc["created_at"] = datetime.now()
            alert_doc["acknowledged"] = False
            
            result = await db.pattern_alerts.insert_one(alert_doc)
            logger.debug(f"Alert saved to MongoDB: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}")
            raise
    
    async def send_telegram(self, message: str, severity: str = "MEDIUM") -> bool:
        """Send Telegram notification."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.debug("Telegram not configured")
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": message,
                        "parse_mode": "Markdown" if "**" not in message else None,
                    }
                )
                
                if response.status_code == 200:
                    logger.debug("Telegram message sent")
                    return True
                else:
                    logger.warning(f"Telegram API error: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            raise
    
    async def save_signal_file(self, alert: Dict) -> bool:
        """Save alert to signal file for Ghost/bots."""
        try:
            # Load existing alerts
            try:
                with open(ALERT_SIGNAL_FILE, "r") as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                signals = {"alerts": [], "last_updated": None}
            
            # Add new alert (keep last 20)
            alert_data = {
                "pattern": alert.get("pattern"),
                "symbol": alert.get("symbol"),
                "severity": alert.get("severity"),
                "direction": alert.get("direction"),
                "defcon_impact": alert.get("defcon_impact"),
                "message": alert.get("message"),
                "action": alert.get("action"),
                "price": alert.get("price"),
                "timestamp": alert.get("timestamp"),
            }
            
            signals["alerts"] = [alert_data] + signals.get("alerts", [])[:19]
            signals["last_updated"] = datetime.now().isoformat()
            signals["latest_alert"] = alert_data
            
            # Write file
            os.makedirs(os.path.dirname(ALERT_SIGNAL_FILE), exist_ok=True)
            with open(ALERT_SIGNAL_FILE, "w") as f:
                json.dump(signals, f, indent=2)
            
            logger.debug(f"Signal file updated: {ALERT_SIGNAL_FILE}")
            return True
        except Exception as e:
            logger.error(f"Signal file save failed: {e}")
            raise
    
    async def push_to_war_room(self, alert: Dict) -> bool:
        """Push alert to War Room API for WebSocket broadcast."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    "http://localhost:8889/war-room/alert",
                    json={
                        "pattern": alert.get("pattern"),
                        "symbol": alert.get("symbol"),
                        "severity": alert.get("severity"),
                        "direction": alert.get("direction"),
                        "defcon_impact": alert.get("defcon_impact"),
                        "message": alert.get("message"),
                        "action": alert.get("action"),
                        "price": alert.get("price"),
                        "timestamp": alert.get("timestamp"),
                    }
                )
                
                if response.status_code in [200, 201]:
                    logger.debug("War Room notified")
                    return True
                else:
                    logger.debug(f"War Room API returned {response.status_code}")
                    return False
        except httpx.ConnectError:
            logger.debug("War Room API not available")
            return False
        except Exception as e:
            logger.error(f"War Room push failed: {e}")
            return False
    
    async def update_defcon_recommendation(self, alert: Dict) -> bool:
        """Update DEFCON recommendation based on alert."""
        try:
            # Get current DEFCON
            current_doc = await db.active_defcon.find_one({"_id": "current"})
            current_defcon = current_doc.get("defcon", 3) if current_doc else 3
            
            # Calculate recommended DEFCON
            impact = alert.get("defcon_impact", 0)
            recommended = max(1, min(5, current_defcon + impact))
            
            if recommended != current_defcon:
                # Save recommendation (not auto-apply)
                await db.defcon_recommendations.insert_one({
                    "current_defcon": current_defcon,
                    "recommended_defcon": recommended,
                    "trigger_pattern": alert.get("pattern"),
                    "trigger_symbol": alert.get("symbol"),
                    "message": alert.get("message"),
                    "severity": alert.get("severity"),
                    "created_at": datetime.now(),
                    "applied": False,
                })
                
                logger.info(f"DEFCON recommendation: {current_defcon} → {recommended} (triggered by {alert.get('pattern')})")
            
            return True
        except Exception as e:
            logger.error(f"DEFCON recommendation update failed: {e}")
            raise
    
    async def get_recent_alerts(self, symbol: Optional[str] = None, limit: int = 20) -> list:
        """Get recent alerts from MongoDB."""
        try:
            query = {"symbol": symbol} if symbol else {}
            cursor = db.pattern_alerts.find(query).sort("created_at", -1).limit(limit)
            alerts = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for alert in alerts:
                alert["_id"] = str(alert["_id"])
            
            return alerts
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark alert as acknowledged."""
        try:
            from bson import ObjectId
            result = await db.pattern_alerts.update_one(
                {"_id": ObjectId(alert_id)},
                {"$set": {"acknowledged": True, "acknowledged_at": datetime.now()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
