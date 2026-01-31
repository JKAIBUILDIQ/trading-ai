"""
═══════════════════════════════════════════════════════════════════════════════
TRADING AGENTS KNOWLEDGE BASE v2.0
MongoDB-powered persistent memory for AI trading agents
═══════════════════════════════════════════════════════════════════════════════

Knowledge Flow:
- NEO (Macro) → feeds correlations, events, DEFCON
- Claudia (Lessons) → feeds rules, lessons, warnings
- Meta (Signals) → feeds patterns, signals, alerts

Consumers:
- Ghost → gets entry/DCA focused knowledge
- Casper → gets hedge/risk focused knowledge
- FOMO → gets exhaustion/reversal focused knowledge
- Chart → gets pattern/technical focused knowledge
- Commander → gets everything
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

# MongoDB async driver
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    print("WARNING: motor not installed. Run: pip install motor")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentKnowledge")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "trading_agents"

# Global client
_client = None
_db = None

# Define what tags each agent cares about
AGENT_TAGS = {
    "ghost": ["entry", "dca", "accumulation", "support", "bullish", "buy", "levels", "timing"],
    "casper": ["hedge", "risk", "protection", "correlation", "warning", "defcon", "sizing"],
    "fomo": ["exhaustion", "reversal", "warning", "overbought", "oversold", "climax", "fomo"],
    "chart": ["pattern", "technical", "level", "structure", "bullish", "bearish"],
    "neo": ["macro", "correlation", "event", "fundamental", "fed", "nfp"],
    "sequence": ["candle", "pattern", "momentum", "reversal"],
    "spy": ["short", "bearish", "fade", "breakdown", "weakness", "collateral", "gdx", "nugt", "miners", "decay"],
    "commander": [],  # Gets everything
}

# Pattern types relevant to each agent
AGENT_PATTERN_TYPES = {
    "ghost": ["bullish", "continuation"],
    "casper": ["reversal", "warning", "bearish"],
    "fomo": ["reversal", "exhaustion"],
    "chart": ["bullish", "bearish", "reversal", "continuation", "neutral"],
    "sequence": ["bullish", "bearish", "reversal"],
    "spy": ["bearish", "reversal", "breakdown", "exhaustion", "failed_breakout"],
}


async def get_database():
    """Get MongoDB database connection"""
    global _client, _db
    
    if not MOTOR_AVAILABLE:
        raise HTTPException(status_code=500, detail="MongoDB driver not installed")
    
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
        _db = _client[DB_NAME]
        
        # Create indexes
        await _db.analyses.create_index([("agent", 1), ("timestamp", -1)])
        await _db.analyses.create_index([("symbol", 1), ("timestamp", -1)])
        await _db.outcomes.create_index([("analysis_id", 1)])
        await _db.outcomes.create_index([("agent", 1), ("recorded_at", -1)])
        await _db.patterns.create_index([("pattern_type", 1)])
        await _db.patterns.create_index([("pattern_name", 1)])
        await _db.memories.create_index([("agent", 1), ("importance", -1)])
        await _db.memories.create_index([("tags", 1)])
        await _db.memories.create_index([("source", 1)])
        await _db.memories.create_index([("expires_at", 1)])
        await _db.market_context.create_index([("date", -1)])
        await _db.journal.create_index([("timestamp", -1)])
        await _db.correlations.create_index([("asset1", 1), ("asset2", 1)])
        
        logger.info(f"Connected to MongoDB: {MONGO_URI}")
    
    return _db


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisRecord(BaseModel):
    agent: str
    symbol: str = "XAUUSD"
    analysis: str
    recommendation: str = ""
    confidence: int = 50
    key_levels: Dict[str, Any] = {}
    context: str = ""
    price_at_analysis: float = 0
    defcon: int = 3
    image_hash: str = ""


class OutcomeRecord(BaseModel):
    analysis_id: str
    actual_outcome: str  # "correct", "partially_correct", "incorrect"
    price_after: float
    pnl_result: float = 0
    notes: str = ""


class PatternRecord(BaseModel):
    pattern_name: str
    pattern_type: str  # "bullish", "bearish", "continuation", "reversal"
    description: str
    entry_rules: List[str] = []
    exit_rules: List[str] = []
    win_rate: int = 50
    example_images: List[str] = []


class MemoryRecord(BaseModel):
    agent: str  # "ghost", "fomo", "all", etc.
    memory_type: str  # "lesson", "rule", "observation", "warning", "context", "signal"
    content: str
    importance: int = 5  # 1-10, 10 being most important
    expires_days: Optional[int] = None
    source: Optional[str] = None  # "neo", "claudia", "meta", "user"
    tags: List[str] = []
    symbol: Optional[str] = None


class MarketContextRecord(BaseModel):
    date: str
    xauusd: float = 0
    dxy: float = 0
    usdjpy: float = 0
    vix: float = 0
    sp500: float = 0
    btc: float = 0
    us10y: float = 0
    defcon: int = 3
    bias: str = "neutral"
    key_events: List[str] = []
    notes: str = ""


class JournalEntry(BaseModel):
    action: str  # "BUY", "SELL", "SCALE_IN", "CLOSE", etc.
    symbol: str = "XAUUSD"
    size: float = 0
    price: float = 0
    reason: str = ""
    agent_recommendation: str = ""
    defcon: int = 3
    tags: List[str] = []


class CorrelationRecord(BaseModel):
    asset1: str
    asset2: str
    correlation_type: str  # "positive", "negative", "leading", "lagging"
    strength: int = 50  # 0-100
    observation: str
    timeframe: str = "daily"


# ═══════════════════════════════════════════════════════════════════════════════
# FEED MODELS (NEO, Claudia, Meta)
# ═══════════════════════════════════════════════════════════════════════════════

class NeoFeed(BaseModel):
    analysis_type: str  # "correlation", "macro", "event", "defcon"
    content: str
    symbol: str = "XAUUSD"
    related_assets: List[str] = []
    defcon_recommendation: Optional[int] = None
    bias: Optional[str] = None
    importance: int = 7


class ClaudiaFeed(BaseModel):
    lesson_type: str  # "lesson", "rule", "observation", "warning"
    content: str
    applies_to: List[str] = ["all"]  # ["ghost", "casper"] or ["all"]
    tags: List[str] = []
    importance: int = 7
    permanent: bool = False


class MetaFeed(BaseModel):
    signal_type: str  # "pattern", "signal", "alert"
    pattern_name: Optional[str] = None
    description: str = ""
    symbol: str = "XAUUSD"
    direction: Optional[str] = None  # "bullish", "bearish", "reversal", "neutral"
    confidence: Optional[int] = None
    entry_rules: List[str] = []
    exit_rules: List[str] = []


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


# ─────────────────────────────────────────────────────────────────────────────
# FEED ENDPOINTS (Knowledge Producers)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/feed/neo")
async def neo_feed(feed: NeoFeed):
    """NEO feeds macro intelligence into the knowledge base."""
    db = await get_database()
    
    # Determine tags based on analysis type
    tags = ["macro", "neo", feed.analysis_type]
    if feed.related_assets:
        tags.extend([a.lower() for a in feed.related_assets])
    
    # Add to memories
    await db.memories.insert_one({
        "agent": "all",
        "memory_type": "context",
        "content": feed.content,
        "importance": feed.importance,
        "source": "neo",
        "tags": tags,
        "symbol": feed.symbol,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7),  # NEO context expires in 7 days
        "times_used": 0,
    })
    
    # If correlation, also add to correlations collection
    if feed.analysis_type == "correlation" and feed.related_assets:
        for asset in feed.related_assets:
            await db.correlations.update_one(
                {"asset1": feed.symbol, "asset2": asset},
                {"$set": {
                    "asset1": feed.symbol,
                    "asset2": asset,
                    "correlation_type": "observed",
                    "observation": feed.content,
                    "timeframe": "current",
                    "timestamp": datetime.utcnow(),
                    "source": "neo",
                }},
                upsert=True
            )
    
    # If DEFCON recommendation, update latest context
    if feed.defcon_recommendation:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        await db.market_context.update_one(
            {"date": today},
            {"$set": {
                "defcon": feed.defcon_recommendation,
                "bias": feed.bias or "neutral",
                "neo_update": datetime.utcnow(),
            }},
            upsert=True
        )
    
    logger.info(f"NEO feed accepted: {feed.analysis_type}")
    return {"status": "neo feed accepted", "type": feed.analysis_type}


@router.post("/feed/claudia")
async def claudia_feed(feed: ClaudiaFeed):
    """Claudia feeds lessons and observations into the knowledge base."""
    db = await get_database()
    
    agents = feed.applies_to or ["all"]
    tags = ["claudia", feed.lesson_type] + feed.tags
    
    for agent in agents:
        await db.memories.insert_one({
            "agent": agent,
            "memory_type": feed.lesson_type,
            "content": feed.content,
            "importance": feed.importance,
            "source": "claudia",
            "tags": tags,
            "created_at": datetime.utcnow(),
            "expires_at": None if feed.permanent else datetime.utcnow() + timedelta(days=30),
            "times_used": 0,
        })
    
    logger.info(f"Claudia feed accepted: {feed.lesson_type} for {agents}")
    return {"status": "claudia feed accepted", "agents": agents}


@router.post("/feed/meta")
async def meta_feed(feed: MetaFeed):
    """Meta bot feeds patterns and signals into the knowledge base."""
    db = await get_database()
    
    tags = ["meta", feed.signal_type]
    if feed.direction:
        tags.append(feed.direction)
    tags.append(feed.symbol.lower())
    
    if feed.signal_type == "pattern" and feed.pattern_name:
        # Update or insert pattern
        await db.patterns.update_one(
            {"pattern_name": feed.pattern_name},
            {"$set": {
                "pattern_name": feed.pattern_name,
                "pattern_type": feed.direction or "neutral",
                "description": feed.description,
                "entry_rules": feed.entry_rules,
                "exit_rules": feed.exit_rules,
                "source": "meta",
                "last_detected": datetime.utcnow(),
            },
            "$inc": {"times_detected": 1}},
            upsert=True
        )
    
    # Also add as a memory/alert
    content = f"{feed.pattern_name or feed.signal_type}: {feed.description}" if feed.description else feed.pattern_name or feed.signal_type
    
    await db.memories.insert_one({
        "agent": "all",
        "memory_type": "signal" if feed.signal_type == "signal" else "observation",
        "content": content,
        "importance": feed.confidence or 5,
        "source": "meta",
        "tags": tags,
        "symbol": feed.symbol,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24),  # Signals expire in 24h
        "times_used": 0,
    })
    
    logger.info(f"Meta feed accepted: {feed.signal_type}")
    return {"status": "meta feed accepted", "type": feed.signal_type}


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS STORAGE
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/store_analysis")
async def store_analysis(record: AnalysisRecord):
    """Store an agent's analysis in the knowledge base"""
    db = await get_database()
    
    doc = {
        **record.dict(),
        "timestamp": datetime.utcnow(),
        "outcome": None,
    }
    
    result = await db.analyses.insert_one(doc)
    
    return {
        "status": "stored",
        "analysis_id": str(result.inserted_id),
        "agent": record.agent,
        "timestamp": doc["timestamp"].isoformat()
    }


@router.get("/recent_analyses")
async def get_recent_analyses(
    agent: Optional[str] = None,
    symbol: str = "XAUUSD",
    hours: int = 24,
    limit: int = 20
):
    """Get recent analyses from agents"""
    db = await get_database()
    
    query = {
        "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours)}
    }
    if agent:
        query["agent"] = agent
    if symbol:
        query["symbol"] = symbol
    
    cursor = db.analyses.find(query).sort("timestamp", -1).limit(limit)
    analyses = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"].isoformat()
        analyses.append(doc)
    
    return {"analyses": analyses, "count": len(analyses)}


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME TRACKING
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/record_outcome")
async def record_outcome(record: OutcomeRecord):
    """Record the outcome of an analysis"""
    db = await get_database()
    from bson import ObjectId
    
    # Get the original analysis to extract agent
    analysis = await db.analyses.find_one({"_id": ObjectId(record.analysis_id)})
    agent = analysis.get("agent", "unknown") if analysis else "unknown"
    
    # Update the original analysis
    await db.analyses.update_one(
        {"_id": ObjectId(record.analysis_id)},
        {"$set": {"outcome": record.actual_outcome}}
    )
    
    # Store detailed outcome
    doc = {
        **record.dict(),
        "agent": agent,
        "recorded_at": datetime.utcnow()
    }
    await db.outcomes.insert_one(doc)
    
    return {"status": "recorded", "outcome": record.actual_outcome, "agent": agent}


@router.get("/agent_accuracy")
async def get_agent_accuracy(agent: str, days: int = 30):
    """Get accuracy stats for an agent"""
    db = await get_database()
    
    since = datetime.utcnow() - timedelta(days=days)
    
    pipeline = [
        {"$match": {
            "agent": agent,
            "recorded_at": {"$gte": since}
        }},
        {"$group": {
            "_id": "$actual_outcome",
            "count": {"$sum": 1}
        }}
    ]
    
    results = {}
    async for doc in db.outcomes.aggregate(pipeline):
        results[doc["_id"]] = doc["count"]
    
    correct = results.get("correct", 0)
    partial = results.get("partially_correct", 0)
    incorrect = results.get("incorrect", 0)
    total = correct + partial + incorrect
    
    accuracy_pct = ((correct + partial * 0.5) / total * 100) if total > 0 else 0
    
    return {
        "agent": agent,
        "days": days,
        "correct": correct,
        "partially_correct": partial,
        "incorrect": incorrect,
        "total": total,
        "accuracy_pct": round(accuracy_pct, 1)
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/save_pattern")
async def save_pattern(record: PatternRecord):
    """Save a trading pattern to the library"""
    db = await get_database()
    
    result = await db.patterns.update_one(
        {"pattern_name": record.pattern_name},
        {"$set": {
            **record.dict(),
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )
    
    return {"status": "saved", "pattern": record.pattern_name}


@router.get("/patterns")
async def get_patterns(pattern_type: Optional[str] = None, limit: int = 50):
    """Get patterns from the library"""
    db = await get_database()
    
    query = {}
    if pattern_type:
        query["pattern_type"] = pattern_type
    
    cursor = db.patterns.find(query).limit(limit)
    patterns = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if "updated_at" in doc:
            doc["updated_at"] = doc["updated_at"].isoformat()
        if "last_detected" in doc:
            doc["last_detected"] = doc["last_detected"].isoformat()
        patterns.append(doc)
    
    return {"patterns": patterns, "count": len(patterns)}


@router.get("/pattern/{name}")
async def get_pattern(name: str):
    """Get a specific pattern by name"""
    db = await get_database()
    
    doc = await db.patterns.find_one({"pattern_name": name})
    if not doc:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    doc["_id"] = str(doc["_id"])
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# AGENT MEMORIES
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/add_memory")
async def add_memory(record: MemoryRecord):
    """Add a memory/lesson/rule for an agent"""
    db = await get_database()
    
    expires_at = None
    if record.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=record.expires_days)
    
    doc = {
        "agent": record.agent,
        "memory_type": record.memory_type,
        "content": record.content,
        "importance": record.importance,
        "source": record.source or "user",
        "tags": record.tags,
        "symbol": record.symbol,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "active": True,
        "times_used": 0,
    }
    
    result = await db.memories.insert_one(doc)
    
    return {
        "status": "added",
        "memory_id": str(result.inserted_id),
        "agent": record.agent
    }


@router.get("/memories")
async def get_memories(
    agent: str,
    memory_type: Optional[str] = None,
    source: Optional[str] = None,
    include_global: bool = True,
    limit: int = 20
):
    """Get memories for an agent"""
    db = await get_database()
    
    now = datetime.utcnow()
    
    # Query for agent-specific and global memories
    agents = [agent]
    if include_global:
        agents.append("all")
    
    query = {
        "agent": {"$in": agents},
        "active": {"$ne": False},
        "$or": [
            {"expires_at": None},
            {"expires_at": {"$gt": now}}
        ]
    }
    if memory_type:
        query["memory_type"] = memory_type
    if source:
        query["source"] = source
    
    cursor = db.memories.find(query).sort("importance", -1).limit(limit)
    memories = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["created_at"] = doc["created_at"].isoformat()
        if doc.get("expires_at"):
            doc["expires_at"] = doc["expires_at"].isoformat()
        memories.append(doc)
    
    return {"memories": memories, "count": len(memories)}


@router.delete("/memory/{memory_id}")
async def deactivate_memory(memory_id: str):
    """Deactivate a memory (soft delete)"""
    db = await get_database()
    from bson import ObjectId
    
    await db.memories.update_one(
        {"_id": ObjectId(memory_id)},
        {"$set": {"active": False}}
    )
    
    return {"status": "deactivated"}


# ─────────────────────────────────────────────────────────────────────────────
# MARKET CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/save_market_context")
async def save_market_context(record: MarketContextRecord):
    """Save daily market context snapshot"""
    db = await get_database()
    
    result = await db.market_context.update_one(
        {"date": record.date},
        {"$set": {
            **record.dict(),
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )
    
    return {"status": "saved", "date": record.date}


@router.get("/market_context")
async def get_market_context(days: int = 7):
    """Get recent market context"""
    db = await get_database()
    
    cursor = db.market_context.find().sort("date", -1).limit(days)
    contexts = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        contexts.append(doc)
    
    return {"contexts": contexts}


@router.get("/market_context/{date}")
async def get_market_context_by_date(date: str):
    """Get market context for a specific date"""
    db = await get_database()
    
    doc = await db.market_context.find_one({"date": date})
    if not doc:
        raise HTTPException(status_code=404, detail="No context for this date")
    
    doc["_id"] = str(doc["_id"])
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/save_correlation")
async def save_correlation(record: CorrelationRecord):
    """Save an observed correlation between assets"""
    db = await get_database()
    
    doc = {
        **record.dict(),
        "created_at": datetime.utcnow()
    }
    
    await db.correlations.insert_one(doc)
    
    return {"status": "saved", "assets": f"{record.asset1}/{record.asset2}"}


@router.get("/correlations")
async def get_correlations(asset: Optional[str] = None, limit: int = 20):
    """Get correlation observations"""
    db = await get_database()
    
    query = {}
    if asset:
        query["$or"] = [{"asset1": asset}, {"asset2": asset}]
    
    cursor = db.correlations.find(query).sort("created_at", -1).limit(limit)
    correlations = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc:
            doc["created_at"] = doc["created_at"].isoformat()
        if "timestamp" in doc:
            doc["timestamp"] = doc["timestamp"].isoformat()
        correlations.append(doc)
    
    return {"correlations": correlations}


# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/journal")
async def add_journal_entry(entry: JournalEntry):
    """Add a trading journal entry"""
    db = await get_database()
    
    doc = {
        **entry.dict(),
        "timestamp": datetime.utcnow()
    }
    
    result = await db.journal.insert_one(doc)
    
    return {"status": "logged", "entry_id": str(result.inserted_id)}


@router.get("/journal")
async def get_journal(
    symbol: Optional[str] = None,
    action: Optional[str] = None,
    days: int = 7,
    limit: int = 50
):
    """Get journal entries"""
    db = await get_database()
    
    query = {
        "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}
    }
    if symbol:
        query["symbol"] = symbol
    if action:
        query["action"] = action
    
    cursor = db.journal.find(query).sort("timestamp", -1).limit(limit)
    entries = []
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"].isoformat()
        entries.append(doc)
    
    return {"entries": entries, "count": len(entries)}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT BRIEFING (THE KEY ENDPOINT!)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/agent_briefing/{agent}")
async def get_agent_briefing(agent: str, symbol: str = "XAUUSD"):
    """
    Get filtered, relevant knowledge for an agent.
    Each agent only gets memories/patterns relevant to their role.
    """
    db = await get_database()
    now = datetime.utcnow()
    
    # Get relevant tags for this agent
    relevant_tags = AGENT_TAGS.get(agent, [])
    relevant_pattern_types = AGENT_PATTERN_TYPES.get(agent, [])
    
    # 1. Get accuracy stats (7 days)
    accuracy_query = {"agent": agent, "recorded_at": {"$gte": now - timedelta(days=7)}}
    outcomes = await db.outcomes.find(accuracy_query).to_list(length=100)
    
    correct = sum(1 for o in outcomes if o.get("actual_outcome") == "correct")
    partial = sum(1 for o in outcomes if o.get("actual_outcome") == "partially_correct")
    total = len(outcomes)
    accuracy_pct = round((correct + partial * 0.5) / total * 100, 1) if total > 0 else None
    
    accuracy_data = {
        "correct": correct,
        "partial": partial,
        "incorrect": total - correct - partial,
        "total": total,
        "accuracy_pct": accuracy_pct
    } if total > 0 else None
    
    # 2. Get memories filtered by agent OR tags
    memory_query = {
        "$and": [
            {"active": {"$ne": False}},
            {"$or": [
                {"expires_at": None},
                {"expires_at": {"$gt": now}}
            ]},
            {"$or": [
                {"agent": agent},
                {"agent": "all"},
            ]}
        ]
    }
    
    # If agent has relevant tags, also include memories with those tags
    if relevant_tags:
        memory_query["$and"][-1]["$or"].append({"tags": {"$in": relevant_tags}})
    
    cursor = db.memories.find(memory_query).sort("importance", -1).limit(15)
    memories = []
    async for m in cursor:
        memories.append({
            "content": m["content"],
            "source": m.get("source", "unknown"),
            "importance": m["importance"],
            "type": m.get("memory_type", "unknown")
        })
        # Increment usage counter
        await db.memories.update_one({"_id": m["_id"]}, {"$inc": {"times_used": 1}})
    
    # 3. Get patterns relevant to agent
    pattern_query = {}
    if relevant_pattern_types:
        pattern_query["pattern_type"] = {"$in": relevant_pattern_types}
    
    patterns = await db.patterns.find(pattern_query).limit(10).to_list(length=10)
    patterns_data = [
        {
            "name": p["pattern_name"],
            "type": p["pattern_type"],
            "win_rate": p.get("win_rate"),
            "description": p.get("description", "")[:100]
        }
        for p in patterns
    ]
    
    # 4. Get correlations for the symbol
    correlations = await db.correlations.find({
        "$or": [{"asset1": symbol}, {"asset2": symbol}]
    }).sort("timestamp", -1).limit(5).to_list(length=5)
    
    correlations_data = [
        {
            "pair": f"{c['asset1']}/{c['asset2']}",
            "type": c.get("correlation_type", "observed"),
            "note": c.get("observation", "")
        }
        for c in correlations
    ]
    
    # 5. Get recent market context
    contexts = await db.market_context.find().sort("date", -1).limit(3).to_list(length=3)
    context_data = [
        {
            "date": c["date"],
            "defcon": c.get("defcon", 5),
            "bias": c.get("bias", "neutral"),
            "events": c.get("key_events", [])
        }
        for c in contexts
    ]
    
    # 6. Get agent's recent analyses count
    recent_analyses = await db.analyses.count_documents({
        "agent": agent,
        "timestamp": {"$gte": now - timedelta(hours=48)}
    })
    
    # 7. Get last analysis
    last_analysis = await db.analyses.find_one(
        {"agent": agent},
        sort=[("timestamp", -1)]
    )
    last_analysis_data = None
    if last_analysis:
        last_analysis_data = {
            "timestamp": last_analysis["timestamp"].isoformat(),
            "recommendation": last_analysis.get("recommendation", ""),
            "confidence": last_analysis.get("confidence", 0)
        }
    
    return {
        "agent": agent,
        "symbol": symbol,
        "briefing_time": now.isoformat(),
        "accuracy_7d": accuracy_data,
        "memories": memories,
        "patterns_to_watch": patterns_data,
        "correlations": correlations_data,
        "market_context": context_data,
        "recent_analyses_count": recent_analyses,
        "last_analysis": last_analysis_data
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATS & HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_knowledge_stats():
    """Get overall knowledge base statistics"""
    db = await get_database()
    
    stats = {
        "analyses_count": await db.analyses.count_documents({}),
        "outcomes_recorded": await db.outcomes.count_documents({}),
        "patterns_count": await db.patterns.count_documents({}),
        "memories_count": await db.memories.count_documents({"active": {"$ne": False}}),
        "journal_entries": await db.journal.count_documents({}),
        "market_contexts": await db.market_context.count_documents({}),
        "correlations": await db.correlations.count_documents({})
    }
    
    # Agent-specific stats
    agents = ["ghost", "casper", "neo", "fomo", "chart", "sequence"]
    agent_stats = {}
    
    for agent in agents:
        count = await db.analyses.count_documents({"agent": agent})
        agent_stats[agent] = {"analyses": count}
    
    # Source stats
    source_stats = {}
    for source in ["neo", "claudia", "meta", "user"]:
        count = await db.memories.count_documents({"source": source, "active": {"$ne": False}})
        source_stats[source] = count
    
    return {
        "database": DB_NAME,
        "totals": stats,
        "by_agent": agent_stats,
        "by_source": source_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health")
async def health_check():
    """Check if knowledge base is healthy"""
    try:
        db = await get_database()
        await db.command("ping")
        return {
            "status": "healthy",
            "database": DB_NAME,
            "motor_available": MOTOR_AVAILABLE,
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "motor_available": MOTOR_AVAILABLE
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR EXTERNAL USE
# ═══════════════════════════════════════════════════════════════════════════════

async def inject_knowledge_into_prompt(agent: str, base_prompt: str, symbol: str = "XAUUSD") -> str:
    """
    Helper function to inject knowledge into an agent's prompt.
    Call this before sending to LLM.
    """
    try:
        briefing = await get_agent_briefing(agent, symbol)
        
        knowledge_section = "\n\n## KNOWLEDGE BRIEFING:\n"
        
        # Accuracy
        if briefing.get("accuracy_7d"):
            acc = briefing["accuracy_7d"]
            knowledge_section += f"\n### Your Recent Performance (7 days):\n"
            knowledge_section += f"- Accuracy: {acc['accuracy_pct']}%\n"
            knowledge_section += f"- Record: {acc['correct']} correct, {acc['partial']} partial, {acc['incorrect']} incorrect\n"
        
        # Memories
        if briefing.get("memories"):
            knowledge_section += f"\n### Things to Remember:\n"
            for m in briefing["memories"][:8]:
                source = f" ({m['source']})" if m.get('source') not in ['unknown', 'user'] else ""
                knowledge_section += f"- {m['content']}{source}\n"
        
        # Patterns
        if briefing.get("patterns_to_watch"):
            knowledge_section += f"\n### Patterns to Watch:\n"
            for p in briefing["patterns_to_watch"][:5]:
                wr = f" (Win rate: {p['win_rate']}%)" if p.get('win_rate') else ""
                knowledge_section += f"- {p['name']} [{p['type']}]{wr}\n"
        
        # Correlations
        if briefing.get("correlations"):
            knowledge_section += f"\n### Correlation Notes:\n"
            for c in briefing["correlations"][:3]:
                knowledge_section += f"- {c['pair']}: {c['note']}\n"
        
        # Market Context
        if briefing.get("market_context"):
            ctx = briefing["market_context"][0]
            knowledge_section += f"\n### Current Market Context:\n"
            knowledge_section += f"- Date: {ctx['date']}, DEFCON: {ctx['defcon']}, Bias: {ctx['bias']}\n"
            if ctx.get("events"):
                knowledge_section += f"- Events: {', '.join(ctx['events'])}\n"
        
        return base_prompt + knowledge_section
        
    except Exception as e:
        logger.warning(f"Could not inject knowledge: {e}")
        return base_prompt


async def store_analysis_result(
    agent: str,
    analysis: str,
    recommendation: str,
    confidence: int,
    symbol: str = "XAUUSD",
    price: float = 0,
    defcon: int = 3,
    context: str = ""
) -> str:
    """
    Helper function to store an analysis result.
    Returns the analysis_id for later outcome tracking.
    """
    try:
        db = await get_database()
        
        doc = {
            "agent": agent,
            "symbol": symbol,
            "analysis": analysis,
            "recommendation": recommendation,
            "confidence": confidence,
            "price_at_analysis": price,
            "defcon": defcon,
            "context": context,
            "timestamp": datetime.utcnow(),
            "outcome": None
        }
        
        result = await db.analyses.insert_one(doc)
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Failed to store analysis: {e}")
        return ""
