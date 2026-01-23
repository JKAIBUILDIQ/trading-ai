"""
═══════════════════════════════════════════════════════════════════════════════
AGENT SHARED MEMORY
═══════════════════════════════════════════════════════════════════════════════

Persistent shared memory using MongoDB.
All agents can read/write to this shared brain.

Collections:
- agent_context: Trading signals, analysis, alerts
- agent_chat: Direct messages between agents
- agent_status: Heartbeats and status updates

═══════════════════════════════════════════════════════════════════════════════
"""

from pymongo import MongoClient, DESCENDING
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger("AgentMemory")

# MongoDB connection string
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "agentdb"


class AgentMemory:
    """
    Shared memory for agent-to-agent communication.
    
    Usage:
        memory = AgentMemory("NEO")
        memory.broadcast("signal", "XAUUSD", {"action": "SELL", "confidence": 75})
        latest = memory.get_latest(agent="NEO", msg_type="signal")
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize agent memory.
        
        Args:
            agent_name: Unique identifier for this agent (NEO, CRELLA001, QUINN001)
        """
        self.agent = agent_name
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        
        # Collections
        self.context = self.db.agent_context
        self.chat = self.db.agent_chat
        self.status = self.db.agent_status
        
        # Ensure indexes
        self._ensure_indexes()
        
        logger.info(f"📂 {agent_name} connected to AgentDB")
    
    def _ensure_indexes(self):
        """Create necessary indexes for performance."""
        # Context collection indexes
        self.context.create_index([("timestamp", DESCENDING)])
        self.context.create_index([("agent", 1), ("timestamp", DESCENDING)])
        self.context.create_index([("type", 1), ("timestamp", DESCENDING)])
        self.context.create_index([("symbol", 1), ("timestamp", DESCENDING)])
        self.context.create_index("expires_at", expireAfterSeconds=0)  # TTL index
        
        # Chat collection indexes
        self.chat.create_index([("timestamp", DESCENDING)])
        self.chat.create_index([("to", 1), ("read", 1)])
        
        # Status collection indexes
        self.status.create_index("agent", unique=True)
        self.status.create_index("last_heartbeat")
    
    # ═══════════════════════════════════════════════════════════════════
    # CONTEXT (Signals, Analysis, Alerts)
    # ═══════════════════════════════════════════════════════════════════
    
    def broadcast(
        self, 
        msg_type: str, 
        symbol: str, 
        data: Dict[str, Any],
        ttl_hours: int = 1
    ) -> str:
        """
        Broadcast a message to all agents via shared memory.
        
        Args:
            msg_type: Type of message (signal, analysis, alert, status)
            symbol: Trading symbol (XAUUSD, EURUSD, etc.)
            data: Message payload
            ttl_hours: Time-to-live in hours
        
        Returns:
            Inserted document ID
        """
        doc = {
            "agent": self.agent,
            "type": msg_type,
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=ttl_hours),
            "read_by": []
        }
        
        result = self.context.insert_one(doc)
        doc_id = str(result.inserted_id)
        
        logger.info(f"📤 {self.agent} broadcast {msg_type} for {symbol}")
        
        return doc_id
    
    def get_latest(
        self, 
        agent: str = None, 
        msg_type: str = None, 
        symbol: str = None
    ) -> Optional[Dict]:
        """
        Get the latest context message matching criteria.
        
        Args:
            agent: Filter by agent name
            msg_type: Filter by message type
            symbol: Filter by symbol
        
        Returns:
            Latest matching document or None
        """
        query = {}
        if agent:
            query["agent"] = agent
        if msg_type:
            query["type"] = msg_type
        if symbol:
            query["symbol"] = symbol
        
        doc = self.context.find_one(query, sort=[("timestamp", DESCENDING)])
        
        if doc:
            doc["_id"] = str(doc["_id"])
        
        return doc
    
    def get_history(
        self, 
        agent: str = None, 
        msg_type: str = None, 
        symbol: str = None,
        limit: int = 50,
        hours: int = 24
    ) -> List[Dict]:
        """
        Get historical messages.
        
        Args:
            agent: Filter by agent name
            msg_type: Filter by message type
            symbol: Filter by symbol
            limit: Maximum number of results
            hours: How far back to look
        
        Returns:
            List of matching documents
        """
        query = {
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours)}
        }
        if agent:
            query["agent"] = agent
        if msg_type:
            query["type"] = msg_type
        if symbol:
            query["symbol"] = symbol
        
        cursor = self.context.find(query).sort("timestamp", DESCENDING).limit(limit)
        
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return results
    
    def mark_read(self, doc_id: str):
        """
        Mark a message as read by this agent.
        
        Args:
            doc_id: Document ID to mark as read
        """
        from bson import ObjectId
        
        self.context.update_one(
            {"_id": ObjectId(doc_id)},
            {"$addToSet": {"read_by": self.agent}}
        )
    
    def get_unread(self, limit: int = 100) -> List[Dict]:
        """
        Get messages not yet read by this agent.
        
        Returns:
            List of unread documents
        """
        cursor = self.context.find({
            "read_by": {"$ne": self.agent},
            "agent": {"$ne": self.agent}  # Not my own messages
        }).sort("timestamp", DESCENDING).limit(limit)
        
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════
    # DIRECT CHAT (Agent-to-Agent Messages)
    # ═══════════════════════════════════════════════════════════════════
    
    def send_message(self, to_agent: str, message: str, data: Dict = None) -> str:
        """
        Send a direct message to another agent.
        
        Args:
            to_agent: Target agent name
            message: Message text
            data: Optional additional data
        
        Returns:
            Message document ID
        """
        doc = {
            "from": self.agent,
            "to": to_agent,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow(),
            "read": False
        }
        
        result = self.chat.insert_one(doc)
        logger.info(f"💬 {self.agent} → {to_agent}: {message[:50]}...")
        
        return str(result.inserted_id)
    
    def get_messages(self, unread_only: bool = False, limit: int = 50) -> List[Dict]:
        """
        Get messages sent to this agent.
        
        Args:
            unread_only: Only return unread messages
            limit: Maximum number of results
        
        Returns:
            List of messages
        """
        query = {"to": self.agent}
        if unread_only:
            query["read"] = False
        
        cursor = self.chat.find(query).sort("timestamp", DESCENDING).limit(limit)
        
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return results
    
    def mark_message_read(self, message_id: str):
        """Mark a direct message as read."""
        from bson import ObjectId
        
        self.chat.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # STATUS (Heartbeats)
    # ═══════════════════════════════════════════════════════════════════
    
    def update_status(self, status_data: Dict[str, Any]):
        """
        Update this agent's status (heartbeat).
        
        Args:
            status_data: Status information to save
        """
        doc = {
            "agent": self.agent,
            "status": "ONLINE",
            "last_heartbeat": datetime.utcnow(),
            "data": status_data
        }
        
        self.status.update_one(
            {"agent": self.agent},
            {"$set": doc},
            upsert=True
        )
    
    def get_agent_status(self, agent_name: str = None) -> Optional[Dict]:
        """
        Get status of an agent.
        
        Args:
            agent_name: Agent to check (defaults to self)
        
        Returns:
            Status document or None
        """
        agent = agent_name or self.agent
        doc = self.status.find_one({"agent": agent})
        
        if doc:
            doc["_id"] = str(doc["_id"])
            
            # Calculate if agent is online (heartbeat within 60 seconds)
            last_hb = doc.get("last_heartbeat")
            if last_hb:
                age = (datetime.utcnow() - last_hb).total_seconds()
                doc["is_online"] = age < 60
                doc["seconds_since_heartbeat"] = int(age)
            else:
                doc["is_online"] = False
        
        return doc
    
    def get_all_agent_statuses(self) -> List[Dict]:
        """Get status of all agents."""
        cursor = self.status.find({})
        
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            
            last_hb = doc.get("last_heartbeat")
            if last_hb:
                age = (datetime.utcnow() - last_hb).total_seconds()
                doc["is_online"] = age < 60
                doc["seconds_since_heartbeat"] = int(age)
            else:
                doc["is_online"] = False
            
            results.append(doc)
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════
    # SEMANTIC SEARCH (Future: Vector embeddings)
    # ═══════════════════════════════════════════════════════════════════
    
    def search_context(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search context by text query.
        
        In the future, this will use vector embeddings for semantic search.
        For now, uses text search.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching documents
        """
        # Create text index if not exists
        try:
            self.context.create_index([("data.reasoning", "text")])
        except:
            pass
        
        # Text search
        cursor = self.context.find(
            {"$text": {"$search": query}}
        ).limit(limit)
        
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_memory_instances: Dict[str, AgentMemory] = {}

def get_memory(agent_name: str) -> AgentMemory:
    """Get or create a singleton AgentMemory instance."""
    if agent_name not in _memory_instances:
        _memory_instances[agent_name] = AgentMemory(agent_name)
    return _memory_instances[agent_name]


if __name__ == "__main__":
    # Test
    memory = AgentMemory("TEST_AGENT")
    
    # Test broadcast
    doc_id = memory.broadcast("signal", "XAUUSD", {
        "action": "SELL",
        "confidence": 75,
        "entry": 4950.00
    })
    print(f"Broadcast: {doc_id}")
    
    # Test get latest
    latest = memory.get_latest(msg_type="signal")
    print(f"Latest: {latest}")
    
    # Test status
    memory.update_status({"current_price": 4950.00, "signals_today": 3})
    status = memory.get_agent_status()
    print(f"Status: {status}")
