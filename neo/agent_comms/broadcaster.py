"""
═══════════════════════════════════════════════════════════════════════════════
AGENT BROADCASTER
═══════════════════════════════════════════════════════════════════════════════

Real-time message broadcasting using Redis Pub/Sub.
Agents receive messages instantly (< 1 second latency).

Channels:
- agent:signals     - Trading signals (BUY/SELL)
- agent:analysis    - Market analysis updates
- agent:alerts      - Warnings (crash probability, etc.)
- agent:heartbeat   - Agent status pulses
- agent:chat        - Direct agent-to-agent messages
- agent:inbox:{name} - Private inbox for specific agent
- human:alerts      - Forward to Telegram for human

═══════════════════════════════════════════════════════════════════════════════
"""

import redis
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from .memory import AgentMemory, get_memory

logger = logging.getLogger("AgentBroadcaster")

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Channels
CHANNEL_SIGNALS = "agent:signals"
CHANNEL_ANALYSIS = "agent:analysis"
CHANNEL_ALERTS = "agent:alerts"
CHANNEL_HEARTBEAT = "agent:heartbeat"
CHANNEL_CHAT = "agent:chat"
CHANNEL_HUMAN = "human:alerts"


class AgentBroadcaster:
    """
    Broadcast messages to all agents and humans.
    
    Usage:
        broadcaster = AgentBroadcaster("NEO")
        broadcaster.signal("XAUUSD", "SELL", 75, entry_price=4950.00)
        broadcaster.heartbeat({"status": "analyzing"})
        broadcaster.chat("CRELLA001", "High crash probability detected!")
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize broadcaster.
        
        Args:
            agent_name: Unique identifier for this agent
        """
        self.agent = agent_name
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.memory = get_memory(agent_name)
        
        logger.info(f"📡 {agent_name} broadcaster initialized")
    
    def _publish(self, channel: str, data: Dict[str, Any]):
        """Internal method to publish to a Redis channel."""
        message = json.dumps(data, default=str)
        self.redis.publish(channel, message)
        logger.debug(f"📤 Published to {channel}")
    
    def _build_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a standard message envelope."""
        return {
            "agent": self.agent,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # TRADING SIGNALS
    # ═══════════════════════════════════════════════════════════════════
    
    def signal(
        self,
        symbol: str,
        action: str,
        confidence: int,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        reasoning: str = None,
        crowd_psychology: Dict = None,
        **kwargs
    ):
        """
        Broadcast a trading signal to all agents.
        
        Args:
            symbol: Trading symbol (XAUUSD, EURUSD, etc.)
            action: BUY, SELL, or WAIT
            confidence: 0-100 confidence level
            entry_price: Suggested entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            reasoning: Explanation for the signal
            crowd_psychology: Crash probability data
            **kwargs: Additional signal data
        """
        signal_data = {
            "action": action,
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": reasoning,
            "crowd_psychology": crowd_psychology,
            **kwargs
        }
        
        # Clean None values
        signal_data = {k: v for k, v in signal_data.items() if v is not None}
        
        # Save to persistent memory
        doc_id = self.memory.broadcast("signal", symbol, signal_data)
        
        # Build message for pub/sub
        msg = self._build_message(signal_data)
        msg["symbol"] = symbol
        msg["doc_id"] = doc_id
        
        # Publish to agent channel
        self._publish(CHANNEL_SIGNALS, msg)
        
        # If high confidence, also alert human
        if confidence >= 70:
            self._publish(CHANNEL_HUMAN, msg)
            logger.info(f"🔔 High confidence signal forwarded to human: {symbol} {action}")
        
        logger.info(f"📊 Signal broadcast: {symbol} {action} @ {confidence}%")
    
    # ═══════════════════════════════════════════════════════════════════
    # MARKET ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    def analysis(
        self,
        symbol: str,
        analysis_data: Dict[str, Any],
        summary: str = None
    ):
        """
        Broadcast market analysis update.
        
        Args:
            symbol: Trading symbol
            analysis_data: Full analysis data
            summary: Human-readable summary
        """
        data = {
            "analysis": analysis_data,
            "summary": summary
        }
        
        # Save to persistent memory
        doc_id = self.memory.broadcast("analysis", symbol, data)
        
        # Build and publish message
        msg = self._build_message(data)
        msg["symbol"] = symbol
        msg["doc_id"] = doc_id
        
        self._publish(CHANNEL_ANALYSIS, msg)
        logger.info(f"📈 Analysis broadcast: {symbol}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ALERTS
    # ═══════════════════════════════════════════════════════════════════
    
    def alert(
        self,
        alert_type: str,
        message: str,
        symbol: str = None,
        severity: str = "warning",
        data: Dict = None
    ):
        """
        Broadcast an alert to all agents.
        
        Args:
            alert_type: Type of alert (crash_warning, divergence, etc.)
            message: Alert message
            symbol: Related symbol (optional)
            severity: low, warning, high, critical
            data: Additional data
        """
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "data": data or {}
        }
        
        # Save to persistent memory
        doc_id = self.memory.broadcast("alert", symbol or "SYSTEM", alert_data)
        
        # Build and publish message
        msg = self._build_message(alert_data)
        msg["symbol"] = symbol
        msg["doc_id"] = doc_id
        
        self._publish(CHANNEL_ALERTS, msg)
        
        # High severity alerts go to human
        if severity in ["high", "critical"]:
            self._publish(CHANNEL_HUMAN, msg)
            logger.warning(f"🚨 Critical alert forwarded to human: {message}")
        
        logger.info(f"⚠️ Alert broadcast: [{severity}] {alert_type}")
    
    def crash_warning(self, symbol: str, crash_probability: float, details: Dict = None):
        """
        Broadcast a crash probability warning.
        
        Args:
            symbol: Trading symbol
            crash_probability: 0-100 crash probability
            details: Crowd psychology details
        """
        severity = "critical" if crash_probability >= 85 else \
                   "high" if crash_probability >= 70 else \
                   "warning" if crash_probability >= 50 else "low"
        
        self.alert(
            alert_type="crash_warning",
            message=f"Crash probability {crash_probability:.0f}% on {symbol}",
            symbol=symbol,
            severity=severity,
            data={
                "crash_probability": crash_probability,
                "details": details or {}
            }
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # HEARTBEAT
    # ═══════════════════════════════════════════════════════════════════
    
    def heartbeat(self, status_data: Dict[str, Any] = None):
        """
        Send a heartbeat pulse to indicate agent is alive.
        
        Args:
            status_data: Current status information
        """
        data = status_data or {}
        
        # Update persistent status
        self.memory.update_status(data)
        
        # Publish heartbeat
        msg = self._build_message(data)
        msg["status"] = "ONLINE"
        
        self._publish(CHANNEL_HEARTBEAT, msg)
        
        # Set Redis key with TTL for quick alive check
        self.redis.setex(f"agent:{self.agent}:alive", 60, "1")
        
        # Store last heartbeat data
        self.redis.hset(f"agent:{self.agent}:heartbeat", mapping={
            "timestamp": datetime.utcnow().isoformat(),
            "data": json.dumps(data, default=str)
        })
        
        logger.debug(f"💓 Heartbeat sent")
    
    # ═══════════════════════════════════════════════════════════════════
    # DIRECT CHAT
    # ═══════════════════════════════════════════════════════════════════
    
    def chat(self, to_agent: str, message: str, data: Dict = None):
        """
        Send a direct message to another agent.
        
        Args:
            to_agent: Target agent name
            message: Message text
            data: Optional additional data
        """
        # Save to persistent memory
        self.memory.send_message(to_agent, message, data)
        
        # Build message for real-time delivery
        msg = {
            "from": self.agent,
            "to": to_agent,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Publish to agent's private inbox
        self._publish(f"agent:inbox:{to_agent}", msg)
        
        # Also publish to general chat channel
        self._publish(CHANNEL_CHAT, msg)
        
        logger.info(f"💬 Chat sent to {to_agent}: {message[:50]}...")
    
    def chat_all(self, message: str, data: Dict = None):
        """
        Broadcast a message to all agents.
        
        Args:
            message: Message text
            data: Optional additional data
        """
        msg = {
            "from": self.agent,
            "to": "ALL",
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._publish(CHANNEL_CHAT, msg)
        logger.info(f"📢 Broadcast to all: {message[:50]}...")
    
    # ═══════════════════════════════════════════════════════════════════
    # HUMAN COMMUNICATION
    # ═══════════════════════════════════════════════════════════════════
    
    def notify_human(self, message: str, data: Dict = None):
        """
        Send a notification to the human (via Telegram).
        
        Args:
            message: Message text
            data: Optional additional data
        """
        msg = {
            "agent": self.agent,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._publish(CHANNEL_HUMAN, msg)
        logger.info(f"📱 Human notification sent: {message[:50]}...")
    
    # ═══════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════
    
    def is_agent_online(self, agent_name: str) -> bool:
        """Check if another agent is online."""
        alive = self.redis.get(f"agent:{agent_name}:alive")
        return bool(alive)
    
    def get_online_agents(self) -> list:
        """Get list of all online agents."""
        agents = []
        for key in self.redis.scan_iter("agent:*:alive"):
            agent_name = key.decode().split(":")[1] if isinstance(key, bytes) else key.split(":")[1]
            agents.append(agent_name)
        return agents


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_broadcaster_instances: Dict[str, AgentBroadcaster] = {}

def get_broadcaster(agent_name: str) -> AgentBroadcaster:
    """Get or create a singleton AgentBroadcaster instance."""
    if agent_name not in _broadcaster_instances:
        _broadcaster_instances[agent_name] = AgentBroadcaster(agent_name)
    return _broadcaster_instances[agent_name]


if __name__ == "__main__":
    # Test
    broadcaster = AgentBroadcaster("TEST_AGENT")
    
    # Test heartbeat
    broadcaster.heartbeat({"status": "testing", "uptime": 100})
    
    # Test signal
    broadcaster.signal(
        symbol="XAUUSD",
        action="SELL",
        confidence=75,
        entry_price=4950.00,
        stop_loss=4980.00,
        take_profit=4920.00,
        reasoning="RSI divergence detected",
        crowd_psychology={"crash_probability": 45}
    )
    
    # Test chat
    broadcaster.chat("NEO", "Hello NEO, this is a test message!")
    
    # Test alert
    broadcaster.crash_warning("XAUUSD", 72, {"divergence": "bearish"})
    
    print("✅ All broadcasts sent!")
