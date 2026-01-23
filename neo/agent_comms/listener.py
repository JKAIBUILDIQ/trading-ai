"""
═══════════════════════════════════════════════════════════════════════════════
AGENT LISTENER
═══════════════════════════════════════════════════════════════════════════════

Subscribe to agent messages via Redis Pub/Sub.
Receive messages in real-time (< 1 second latency).

Usage:
    listener = AgentListener("CRELLA001")
    
    @listener.on("agent:signals")
    def handle_signal(msg):
        print(f"NEO says: {msg['data']['action']} {msg['symbol']}")
    
    listener.start()  # Runs in background thread

═══════════════════════════════════════════════════════════════════════════════
"""

import redis
import json
import os
import threading
from typing import Callable, Dict, Any, List
from datetime import datetime
import logging

from .memory import AgentMemory, get_memory

logger = logging.getLogger("AgentListener")

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Standard channels
CHANNELS = {
    "signals": "agent:signals",
    "analysis": "agent:analysis",
    "alerts": "agent:alerts",
    "heartbeat": "agent:heartbeat",
    "chat": "agent:chat",
    "human_requests": "human:request"
}


class AgentListener:
    """
    Listen to agent messages via Redis Pub/Sub.
    
    Usage:
        listener = AgentListener("CRELLA001")
        
        @listener.on("agent:signals")
        def handle_signal(msg):
            print(f"Signal received: {msg}")
        
        listener.start()  # Background thread
    """
    
    def __init__(self, agent_name: str, auto_subscribe_inbox: bool = True):
        """
        Initialize listener.
        
        Args:
            agent_name: This agent's name
            auto_subscribe_inbox: Auto-subscribe to private inbox
        """
        self.agent = agent_name
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.pubsub = self.redis.pubsub()
        self.memory = get_memory(agent_name)
        
        self.handlers: Dict[str, List[Callable]] = {}
        self.running = False
        self.thread: threading.Thread = None
        
        # Auto-subscribe to private inbox
        if auto_subscribe_inbox:
            self.subscribe(f"agent:inbox:{agent_name}")
        
        logger.info(f"👂 {agent_name} listener initialized")
    
    def subscribe(self, channel: str):
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel name to subscribe to
        """
        self.pubsub.subscribe(channel)
        if channel not in self.handlers:
            self.handlers[channel] = []
        logger.info(f"📻 Subscribed to {channel}")
    
    def unsubscribe(self, channel: str):
        """Unsubscribe from a channel."""
        self.pubsub.unsubscribe(channel)
        if channel in self.handlers:
            del self.handlers[channel]
        logger.info(f"📴 Unsubscribed from {channel}")
    
    def on(self, channel: str):
        """
        Decorator to register a handler for a channel.
        
        Usage:
            @listener.on("agent:signals")
            def handle_signal(msg):
                print(msg)
        """
        def decorator(func: Callable):
            self.add_handler(channel, func)
            return func
        return decorator
    
    def add_handler(self, channel: str, handler: Callable):
        """
        Add a handler for a channel.
        
        Args:
            channel: Channel to handle
            handler: Function to call when message received
        """
        if channel not in self.handlers:
            self.subscribe(channel)
            self.handlers[channel] = []
        
        self.handlers[channel].append(handler)
        logger.info(f"📝 Handler registered for {channel}")
    
    def remove_handler(self, channel: str, handler: Callable):
        """Remove a handler from a channel."""
        if channel in self.handlers and handler in self.handlers[channel]:
            self.handlers[channel].remove(handler)
    
    def _handle_message(self, message: Dict):
        """Internal message handler."""
        if message['type'] != 'message':
            return
        
        channel = message['channel']
        try:
            data = json.loads(message['data'])
        except json.JSONDecodeError:
            data = {"raw": message['data']}
        
        # Don't process own messages
        if data.get("agent") == self.agent:
            return
        
        # Call all handlers for this channel
        if channel in self.handlers:
            for handler in self.handlers[channel]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Handler error on {channel}: {e}")
    
    def _listen_loop(self):
        """Main listening loop (runs in thread)."""
        logger.info(f"🎧 {self.agent} listener started")
        
        for message in self.pubsub.listen():
            if not self.running:
                break
            self._handle_message(message)
        
        logger.info(f"🛑 {self.agent} listener stopped")
    
    def start(self):
        """Start listening in a background thread."""
        if self.running:
            logger.warning("Listener already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info(f"🚀 {self.agent} listener started in background")
    
    def stop(self):
        """Stop the listener."""
        self.running = False
        self.pubsub.close()
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"⏹️ {self.agent} listener stopped")
    
    def listen_blocking(self):
        """Start listening (blocks current thread)."""
        self.running = True
        self._listen_loop()
    
    # ═══════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def on_signals(self, handler: Callable):
        """Subscribe to trading signals."""
        self.add_handler(CHANNELS["signals"], handler)
    
    def on_analysis(self, handler: Callable):
        """Subscribe to market analysis."""
        self.add_handler(CHANNELS["analysis"], handler)
    
    def on_alerts(self, handler: Callable):
        """Subscribe to alerts."""
        self.add_handler(CHANNELS["alerts"], handler)
    
    def on_heartbeat(self, handler: Callable):
        """Subscribe to agent heartbeats."""
        self.add_handler(CHANNELS["heartbeat"], handler)
    
    def on_chat(self, handler: Callable):
        """Subscribe to chat messages."""
        self.add_handler(CHANNELS["chat"], handler)
    
    def on_human_request(self, handler: Callable):
        """Subscribe to human requests (from Telegram)."""
        self.add_handler(CHANNELS["human_requests"], handler)
    
    def on_inbox(self, handler: Callable):
        """Subscribe to private inbox messages."""
        self.add_handler(f"agent:inbox:{self.agent}", handler)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_listener_instances: Dict[str, AgentListener] = {}

def get_listener(agent_name: str) -> AgentListener:
    """Get or create a singleton AgentListener instance."""
    if agent_name not in _listener_instances:
        _listener_instances[agent_name] = AgentListener(agent_name)
    return _listener_instances[agent_name]


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

def example_crella_listener():
    """Example: CRELLA listening to NEO's signals."""
    
    listener = AgentListener("CRELLA001")
    
    @listener.on("agent:signals")
    def handle_neo_signal(msg):
        print(f"\n🧠 NEO SIGNAL RECEIVED!")
        print(f"   Symbol: {msg.get('symbol')}")
        print(f"   Action: {msg['data'].get('action')}")
        print(f"   Confidence: {msg['data'].get('confidence')}%")
        
        # React to high-confidence signals
        if msg['data'].get('confidence', 0) >= 70:
            print(f"   ⚡ Adjusting Jackal strategy...")
            # update_jackal_for_signal(msg)
    
    @listener.on("agent:alerts")
    def handle_alert(msg):
        print(f"\n⚠️ ALERT: {msg['data'].get('message')}")
        severity = msg['data'].get('severity', 'info')
        
        if severity in ['high', 'critical']:
            print(f"   🚨 CRITICAL: Taking protective action!")
            # hedge_positions()
    
    @listener.on(f"agent:inbox:CRELLA001")
    def handle_direct_message(msg):
        print(f"\n💬 {msg.get('from')}: {msg.get('message')}")
    
    print("CRELLA listener starting...")
    listener.start()
    
    # Keep running
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
        print("Stopped.")


if __name__ == "__main__":
    # Run example
    example_crella_listener()
