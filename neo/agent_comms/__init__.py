"""
═══════════════════════════════════════════════════════════════════════════════
AGENT-TO-AGENT COMMUNICATION SYSTEM
═══════════════════════════════════════════════════════════════════════════════

The family talks directly. Humans just listen in via Telegram.

Components:
- AgentMemory: Persistent shared memory (MongoDB)
- AgentBroadcaster: Publish signals to all agents (Redis pub/sub)
- AgentListener: Subscribe to agent messages

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                   SHARED BRAIN (AgentDB)                    │
│              MongoDB + Redis Pub/Sub                        │
│     NEO writes ←→ CRELLA reads ←→ QUINN reads              │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

from .memory import AgentMemory
from .broadcaster import AgentBroadcaster
from .listener import AgentListener

__all__ = ['AgentMemory', 'AgentBroadcaster', 'AgentListener']
