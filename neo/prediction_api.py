#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NEO GOLD PREDICTION API
═══════════════════════════════════════════════════════════════════════════════

FastAPI endpoints for the Gold Prediction Learning System.

Endpoints:
- GET  /api/neo/gold/prediction           - Get current prediction
- GET  /api/neo/gold/prediction/history   - Get prediction history
- GET  /api/neo/gold/prediction/accuracy  - Get accuracy stats
- GET  /api/neo/gold/prediction/features  - Get feature performance
- POST /api/neo/gold/prediction/evaluate  - Manually trigger evaluation
- POST /api/neo/gold/prediction/learn     - Manually trigger learning

Created: 2026-01-26
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our modules
from gold_predictor import GoldPredictor, Prediction
from prediction_store import PredictionStore
from prediction_learner import PredictionLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionAPI")

# Initialize components
predictor = GoldPredictor()
store = PredictionStore()
learner = PredictionLearner()

# FastAPI app
app = FastAPI(
    title="NEO Gold Prediction API",
    description="4-Hour Gold Price Prediction with Learning",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
current_prediction: Optional[Prediction] = None
learning_loop_running = False


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    target_time: str
    current_price: float
    predicted_direction: str
    predicted_change_pips: float
    predicted_price: float
    confidence: float
    reasoning: str
    status: str
    time_remaining_minutes: Optional[int] = None


class AccuracyResponse(BaseModel):
    total_predictions: int
    evaluated_predictions: int
    correct_direction: int
    accuracy: float
    target_accuracy: float = 60.0
    meets_target: bool
    current_streak: int
    best_streak: int
    avg_confidence_when_correct: float
    avg_confidence_when_wrong: float


class FeatureResponse(BaseModel):
    feature: str
    accuracy: float
    total_uses: int
    weight: float
    inverted: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """API root"""
    return {
        "service": "NEO Gold Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/api/neo/gold/prediction",
            "history": "/api/neo/gold/prediction/history",
            "accuracy": "/api/neo/gold/prediction/accuracy",
            "features": "/api/neo/gold/prediction/features",
            "evaluate": "/api/neo/gold/prediction/evaluate (POST)",
            "learn": "/api/neo/gold/prediction/learn (POST)"
        }
    }


@app.get("/api/neo/gold/prediction")
async def get_prediction():
    """
    Get current active prediction.
    
    If no active prediction or prediction is expired, generates a new one.
    """
    global current_prediction
    
    now = datetime.now(timezone.utc)
    
    # Check if we need a new prediction
    need_new = False
    if current_prediction is None:
        need_new = True
        logger.info("No current prediction - generating new one")
    else:
        target_time = datetime.fromisoformat(current_prediction.target_time.replace('Z', '+00:00'))
        if isinstance(target_time, datetime) and target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        
        if now > target_time:
            need_new = True
            logger.info("Current prediction expired - generating new one")
    
    if need_new:
        # Evaluate previous prediction if exists
        if current_prediction and current_prediction.status == "PENDING":
            await evaluate_current_prediction()
        
        # Generate new prediction
        current_prediction = predictor.predict_4h()
        store.save_prediction(current_prediction)
    
    # Calculate time remaining
    target_time = datetime.fromisoformat(current_prediction.target_time.replace('Z', '+00:00'))
    if isinstance(target_time, datetime) and target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    
    time_remaining = int((target_time - now).total_seconds() / 60)
    
    return {
        "prediction_id": current_prediction.prediction_id,
        "timestamp": current_prediction.timestamp,
        "target_time": current_prediction.target_time,
        "current_price": current_prediction.current_price,
        "predicted_direction": current_prediction.predicted_direction,
        "predicted_change_pips": current_prediction.predicted_change_pips,
        "predicted_price": current_prediction.predicted_price,
        "confidence": current_prediction.confidence,
        "reasoning": current_prediction.reasoning,
        "status": current_prediction.status,
        "time_remaining_minutes": max(0, time_remaining),
        "time_remaining_display": f"{time_remaining // 60}h {time_remaining % 60}m" if time_remaining > 0 else "Expired"
    }


@app.get("/api/neo/gold/prediction/history")
async def get_prediction_history(limit: int = 20):
    """Get recent prediction history"""
    predictions = store.get_recent_predictions(limit)
    
    return {
        "count": len(predictions),
        "predictions": predictions
    }


@app.get("/api/neo/gold/prediction/accuracy")
async def get_accuracy():
    """Get accuracy statistics"""
    stats = store.get_stats()
    
    accuracy = stats.get("accuracy", 0)
    
    return {
        "total_predictions": stats.get("total_predictions", 0),
        "evaluated_predictions": stats.get("evaluated_predictions", 0),
        "correct_direction": stats.get("correct_direction", 0),
        "accuracy": accuracy,
        "target_accuracy": 60.0,
        "meets_target": accuracy >= 60.0,
        "accuracy_trend": "↑ Improving" if accuracy >= 55 else "↓ Needs work" if accuracy < 45 else "→ Stable",
        "current_streak": stats.get("current_streak", 0),
        "best_streak": stats.get("best_streak", 0),
        "avg_confidence_when_correct": stats.get("avg_confidence_when_correct", 0),
        "avg_confidence_when_wrong": stats.get("avg_confidence_when_wrong", 0)
    }


@app.get("/api/neo/gold/prediction/features")
async def get_features():
    """Get feature performance leaderboard"""
    leaderboard = store.get_feature_leaderboard()
    
    # Add weight info from learner
    for feat in leaderboard:
        feat["weight"] = learner.weights.get(feat["feature"], 0)
        feat_stats = learner.feature_stats.get(feat["feature"], {})
        feat["inverted"] = feat_stats.get("inverted", False)
    
    return {
        "features": leaderboard,
        "total_features": len(leaderboard),
        "best_feature": leaderboard[0] if leaderboard else None,
        "worst_feature": leaderboard[-1] if leaderboard else None,
        "recommendations": learner.get_recommended_actions()
    }


@app.get("/api/neo/gold/prediction/summary")
async def get_summary():
    """Get full summary including prediction, accuracy, and features"""
    prediction = await get_prediction()
    accuracy = await get_accuracy()
    features = await get_features()
    
    return {
        "current_prediction": prediction,
        "accuracy": accuracy,
        "top_features": features["features"][:5] if features["features"] else [],
        "recommendations": features["recommendations"]
    }


@app.post("/api/neo/gold/prediction/evaluate")
async def evaluate_prediction():
    """Manually trigger evaluation of current prediction"""
    global current_prediction
    
    if current_prediction is None:
        raise HTTPException(status_code=404, detail="No active prediction to evaluate")
    
    if current_prediction.status == "EVALUATED":
        return {
            "message": "Prediction already evaluated",
            "result": asdict(current_prediction)
        }
    
    result = await evaluate_current_prediction()
    
    return {
        "message": "Prediction evaluated",
        "result": result
    }


@app.post("/api/neo/gold/prediction/learn")
async def trigger_learning():
    """Manually trigger learning algorithm"""
    adjustments = learner.learn()
    
    return {
        "message": "Learning complete",
        "adjustments": adjustments,
        "feature_report": learner.get_feature_report()
    }


@app.post("/api/neo/gold/prediction/reset")
async def reset_prediction():
    """Force a new prediction (for testing)"""
    global current_prediction
    
    # Evaluate current if pending
    if current_prediction and current_prediction.status == "PENDING":
        await evaluate_current_prediction()
    
    # Generate new
    current_prediction = predictor.predict_4h()
    store.save_prediction(current_prediction)
    
    return {
        "message": "New prediction generated",
        "prediction": asdict(current_prediction)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def evaluate_current_prediction():
    """Evaluate the current prediction against actual price"""
    global current_prediction
    
    if current_prediction is None or current_prediction.status == "EVALUATED":
        return None
    
    logger.info(f"Evaluating prediction: {current_prediction.prediction_id}")
    
    # Evaluate
    evaluated = predictor.evaluate_prediction(current_prediction)
    
    # Update store
    store.update_prediction(evaluated.prediction_id, {
        "status": "EVALUATED",
        "actual_price": evaluated.actual_price,
        "actual_change_pips": evaluated.actual_change_pips,
        "actual_direction": evaluated.actual_direction,
        "direction_correct": evaluated.direction_correct,
        "magnitude_accuracy": evaluated.magnitude_accuracy,
        "score": evaluated.score,
        "evaluated_at": evaluated.evaluated_at
    })
    
    # Record for learner
    learner.record_outcome(
        evaluated.prediction_id,
        evaluated.feature_contributions,
        evaluated.predicted_direction,
        evaluated.actual_direction,
        evaluated.direction_correct,
        evaluated.confidence
    )
    
    current_prediction = evaluated
    
    return asdict(evaluated)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

async def prediction_learning_loop():
    """
    Background loop that:
    1. Evaluates previous predictions when they expire
    2. Generates new predictions every 4 hours
    3. Runs learning periodically
    """
    global current_prediction, learning_loop_running
    
    learning_loop_running = True
    predictions_since_learn = 0
    
    logger.info("🔄 Starting prediction learning loop...")
    
    while learning_loop_running:
        try:
            now = datetime.now(timezone.utc)
            
            # Check if current prediction needs evaluation
            if current_prediction and current_prediction.status == "PENDING":
                target_time = datetime.fromisoformat(
                    current_prediction.target_time.replace('Z', '+00:00')
                )
                if isinstance(target_time, datetime) and target_time.tzinfo is None:
                    target_time = target_time.replace(tzinfo=timezone.utc)
                
                if now > target_time:
                    logger.info("⏰ Prediction expired - evaluating...")
                    await evaluate_current_prediction()
                    predictions_since_learn += 1
                    
                    # Generate new prediction
                    logger.info("🔮 Generating new prediction...")
                    current_prediction = predictor.predict_4h()
                    store.save_prediction(current_prediction)
            
            # Run learning periodically (every 5 predictions)
            if predictions_since_learn >= 5:
                logger.info("🧠 Running learning algorithm...")
                learner.learn()
                predictions_since_learn = 0
            
            # Check every 5 minutes
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in learning loop: {e}")
            await asyncio.sleep(60)


@app.on_event("startup")
async def startup_event():
    """Start background tasks on API startup"""
    global current_prediction
    
    # Load or create initial prediction
    last_pred = store.get_last_prediction()
    if last_pred and last_pred.get("status") == "PENDING":
        current_prediction = Prediction(**{k: v for k, v in last_pred.items() if k in Prediction.__dataclass_fields__})
        logger.info(f"Loaded pending prediction: {current_prediction.prediction_id}")
    else:
        current_prediction = predictor.predict_4h()
        store.save_prediction(current_prediction)
        logger.info(f"Generated new prediction: {current_prediction.prediction_id}")
    
    # Start background loop
    asyncio.create_task(prediction_learning_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global learning_loop_running
    learning_loop_running = False
    logger.info("Prediction API shutting down...")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8020,
        log_level="info"
    )
