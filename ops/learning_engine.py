"""
LEARNING ENGINE - AI Agent Training from Experience
Analyzes historical performance to improve agent decision-making
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class LearningEngine:
    """Learns from daily operations to improve agent performance"""
    
    def __init__(self):
        self.weights_file = DATA_DIR / "agent_weights.json"
        self.learning_history_file = DATA_DIR / "learning_history.json"
        self.pattern_success_file = DATA_DIR / "pattern_success_rates.json"
        self.strategy_performance_file = DATA_DIR / "strategy_performance.json"
    
    def load_weights(self) -> Dict:
        """Load current agent weights"""
        if self.weights_file.exists():
            with open(self.weights_file) as f:
                return json.load(f)
        return self._default_weights()
    
    def _default_weights(self) -> Dict:
        return {
            "quant": {
                "weight": 1.0,
                "accuracy": 0.50,
                "bullish_accuracy": 0.50,
                "bearish_accuracy": 0.50,
                "total_predictions": 0,
                "confidence_calibration": 1.0,  # How well does confidence match reality
                "best_timeframe": "1d",
                "worst_conditions": []
            },
            "neo": {
                "weight": 1.0,
                "accuracy": 0.50,
                "bullish_accuracy": 0.50,
                "bearish_accuracy": 0.50,
                "total_predictions": 0,
                "confidence_calibration": 1.0,
                "best_patterns": [],
                "worst_patterns": []
            },
            "claudia": {
                "weight": 1.0,
                "accuracy": 0.50,
                "bullish_accuracy": 0.50,
                "bearish_accuracy": 0.50,
                "total_predictions": 0,
                "confidence_calibration": 1.0,
                "best_sectors": [],
                "catalyst_accuracy": 0.50
            },
            "sentinel": {
                "weight": 1.0,
                "pattern_accuracy": {},  # By pattern type
                "alert_accuracy": 0.50,
                "false_positive_rate": 0.20,
                "total_alerts": 0
            },
            "scouts": {
                "tech_titan": {"weight": 1.0, "accuracy": 0.50},
                "energy_eagle": {"weight": 1.0, "accuracy": 0.50},
                "miner_hawk": {"weight": 1.0, "accuracy": 0.50},
                "growth_hunter": {"weight": 1.0, "accuracy": 0.50},
                "defense_fortress": {"weight": 1.0, "accuracy": 0.50}
            }
        }
    
    def save_weights(self, weights: Dict):
        """Save updated weights"""
        with open(self.weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
    
    def analyze_predictions(self, days: int = 30) -> Dict:
        """Analyze prediction accuracy over specified days"""
        results = defaultdict(lambda: {
            "total": 0, "correct": 0, "bullish_correct": 0, "bullish_total": 0,
            "bearish_correct": 0, "bearish_total": 0, "by_confidence": defaultdict(lambda: {"total": 0, "correct": 0})
        })
        
        # Load historical ops files
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            ops_file = DATA_DIR / f"ops_{date}.json"
            
            if not ops_file.exists():
                continue
            
            with open(ops_file) as f:
                ops_data = json.load(f)
            
            for pred in ops_data.get("agent_predictions", []):
                if not pred.get("verified"):
                    continue
                
                agent = pred["agent"]
                prediction = pred["prediction"]
                outcome = pred["outcome"]
                
                results[agent]["total"] += 1
                
                # Check if prediction was correct
                predicted_direction = prediction.get("direction", prediction.get("bias", ""))
                actual_direction = outcome.get("direction", "")
                
                is_correct = (
                    (predicted_direction.upper() in ["BULLISH", "LONG", "UP"] and actual_direction.upper() in ["UP", "BULLISH"]) or
                    (predicted_direction.upper() in ["BEARISH", "SHORT", "DOWN"] and actual_direction.upper() in ["DOWN", "BEARISH"])
                )
                
                if is_correct:
                    results[agent]["correct"] += 1
                
                # Track bullish vs bearish accuracy
                if predicted_direction.upper() in ["BULLISH", "LONG", "UP"]:
                    results[agent]["bullish_total"] += 1
                    if is_correct:
                        results[agent]["bullish_correct"] += 1
                elif predicted_direction.upper() in ["BEARISH", "SHORT", "DOWN"]:
                    results[agent]["bearish_total"] += 1
                    if is_correct:
                        results[agent]["bearish_correct"] += 1
                
                # Track by confidence level
                confidence = prediction.get("conviction", prediction.get("confidence", 50))
                if isinstance(confidence, str):
                    confidence_map = {"LOW": 30, "MEDIUM": 50, "HIGH": 75, "VERY HIGH": 90}
                    confidence = confidence_map.get(confidence.upper(), 50)
                
                conf_bucket = f"{(confidence // 20) * 20}-{(confidence // 20) * 20 + 19}%"
                results[agent]["by_confidence"][conf_bucket]["total"] += 1
                if is_correct:
                    results[agent]["by_confidence"][conf_bucket]["correct"] += 1
        
        return dict(results)
    
    def calculate_accuracy(self, correct: int, total: int) -> float:
        """Calculate accuracy with Bayesian smoothing"""
        if total == 0:
            return 0.50  # Prior
        # Use Laplace smoothing
        return (correct + 1) / (total + 2)
    
    def update_agent_weights(self) -> Dict:
        """Update agent weights based on recent performance"""
        weights = self.load_weights()
        analysis = self.analyze_predictions(days=30)
        
        for agent, data in analysis.items():
            if agent not in weights:
                continue
            
            if data["total"] < 5:
                continue  # Not enough data
            
            # Calculate overall accuracy
            accuracy = self.calculate_accuracy(data["correct"], data["total"])
            
            # Update weights
            if agent in ["quant", "neo", "claudia"]:
                old_weight = weights[agent]["weight"]
                
                # Adjust weight based on accuracy
                if accuracy > 0.60:
                    new_weight = min(1.5, old_weight * 1.02)  # Gradual increase
                elif accuracy < 0.45:
                    new_weight = max(0.5, old_weight * 0.98)  # Gradual decrease
                else:
                    new_weight = old_weight
                
                weights[agent]["weight"] = round(new_weight, 3)
                weights[agent]["accuracy"] = round(accuracy, 3)
                weights[agent]["total_predictions"] = data["total"]
                
                # Bullish/bearish breakdown
                if data["bullish_total"] > 0:
                    weights[agent]["bullish_accuracy"] = round(
                        self.calculate_accuracy(data["bullish_correct"], data["bullish_total"]), 3
                    )
                if data["bearish_total"] > 0:
                    weights[agent]["bearish_accuracy"] = round(
                        self.calculate_accuracy(data["bearish_correct"], data["bearish_total"]), 3
                    )
                
                # Confidence calibration
                calibration_score = self._calculate_calibration(data["by_confidence"])
                weights[agent]["confidence_calibration"] = round(calibration_score, 3)
        
        self.save_weights(weights)
        self._log_learning_event(analysis, weights)
        
        return weights
    
    def _calculate_calibration(self, by_confidence: Dict) -> float:
        """Calculate how well confidence matches actual accuracy"""
        if not by_confidence:
            return 1.0
        
        errors = []
        for bucket, data in by_confidence.items():
            if data["total"] < 3:
                continue
            
            # Extract expected confidence from bucket (e.g., "60-79%" -> 70%)
            try:
                expected = int(bucket.split("-")[0]) + 10  # Midpoint
            except:
                continue
            
            actual = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 50
            error = abs(expected - actual) / 100
            errors.append(error)
        
        if not errors:
            return 1.0
        
        # Return calibration score (1 = perfect, 0 = terrible)
        avg_error = sum(errors) / len(errors)
        return max(0, 1 - avg_error)
    
    def _log_learning_event(self, analysis: Dict, weights: Dict):
        """Log learning event for history"""
        history = []
        if self.learning_history_file.exists():
            with open(self.learning_history_file) as f:
                history = json.load(f)
        
        history.append({
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                agent: {
                    "accuracy": self.calculate_accuracy(data["correct"], data["total"]),
                    "total": data["total"]
                }
                for agent, data in analysis.items()
            },
            "updated_weights": {
                agent: weights[agent].get("weight", 1.0)
                for agent in ["quant", "neo", "claudia"]
                if agent in weights
            }
        })
        
        # Keep last 100 entries
        history = history[-100:]
        
        with open(self.learning_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def analyze_pattern_success(self) -> Dict:
        """Analyze which patterns lead to successful trades"""
        pattern_stats = defaultdict(lambda: {"total": 0, "wins": 0, "avg_pnl": 0, "pnl_sum": 0})
        
        # Load all ops files
        for ops_file in DATA_DIR.glob("ops_*.json"):
            with open(ops_file) as f:
                ops_data = json.load(f)
            
            for trade in ops_data.get("trades", []):
                patterns = trade.get("patterns", [])
                pnl = trade.get("pnl", 0)
                is_win = pnl > 0
                
                for pattern in patterns:
                    pattern_stats[pattern]["total"] += 1
                    pattern_stats[pattern]["pnl_sum"] += pnl
                    if is_win:
                        pattern_stats[pattern]["wins"] += 1
        
        # Calculate averages
        results = {}
        for pattern, stats in pattern_stats.items():
            if stats["total"] > 0:
                results[pattern] = {
                    "total_trades": stats["total"],
                    "win_rate": round(stats["wins"] / stats["total"], 3),
                    "avg_pnl": round(stats["pnl_sum"] / stats["total"], 2),
                    "edge_score": round((stats["wins"] / stats["total"]) * (stats["pnl_sum"] / stats["total"]) if stats["total"] > 0 else 0, 2)
                }
        
        # Save results
        with open(self.pattern_success_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def get_agent_recommendations(self) -> Dict:
        """Get recommendations for each agent based on learning"""
        weights = self.load_weights()
        recommendations = {}
        
        for agent in ["quant", "neo", "claudia"]:
            if agent not in weights:
                continue
            
            data = weights[agent]
            recs = []
            
            # Check overall accuracy
            if data.get("accuracy", 0.5) < 0.45:
                recs.append("⚠️ Below average accuracy - consider reducing position sizes")
            elif data.get("accuracy", 0.5) > 0.60:
                recs.append("✅ Strong accuracy - can trust higher conviction calls")
            
            # Check directional bias
            bullish_acc = data.get("bullish_accuracy", 0.5)
            bearish_acc = data.get("bearish_accuracy", 0.5)
            
            if bullish_acc > bearish_acc + 0.15:
                recs.append(f"📈 Better at BULLISH calls ({bullish_acc:.0%} vs {bearish_acc:.0%})")
            elif bearish_acc > bullish_acc + 0.15:
                recs.append(f"📉 Better at BEARISH calls ({bearish_acc:.0%} vs {bullish_acc:.0%})")
            
            # Check confidence calibration
            calibration = data.get("confidence_calibration", 1.0)
            if calibration < 0.7:
                recs.append("🎯 Overconfident - discount high conviction calls")
            elif calibration > 0.9:
                recs.append("🎯 Well calibrated - confidence scores reliable")
            
            recommendations[agent] = {
                "weight": data.get("weight", 1.0),
                "accuracy": data.get("accuracy", 0.5),
                "recommendations": recs
            }
        
        return recommendations
    
    def generate_training_report(self) -> str:
        """Generate human-readable training report"""
        weights = self.load_weights()
        recs = self.get_agent_recommendations()
        
        report = "=" * 50 + "\n"
        report += "🧠 AI AGENT LEARNING REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report += "=" * 50 + "\n\n"
        
        report += "📊 AGENT PERFORMANCE WEIGHTS\n"
        report += "-" * 30 + "\n"
        
        for agent in ["quant", "neo", "claudia"]:
            if agent not in weights:
                continue
            
            data = weights[agent]
            report += f"\n{agent.upper()}:\n"
            report += f"  Weight: {data.get('weight', 1.0):.2f}x\n"
            report += f"  Accuracy: {data.get('accuracy', 0.5):.1%}\n"
            report += f"  Predictions: {data.get('total_predictions', 0)}\n"
            report += f"  Calibration: {data.get('confidence_calibration', 1.0):.2f}\n"
            
            if agent in recs:
                for rec in recs[agent].get("recommendations", []):
                    report += f"  {rec}\n"
        
        # Scouts summary
        if "scouts" in weights:
            report += "\nSCOUT SWARM:\n"
            for scout, data in weights["scouts"].items():
                report += f"  {scout}: {data.get('accuracy', 0.5):.1%} accuracy\n"
        
        # Pattern analysis
        if self.pattern_success_file.exists():
            with open(self.pattern_success_file) as f:
                patterns = json.load(f)
            
            if patterns:
                report += "\n📈 TOP PERFORMING PATTERNS:\n"
                report += "-" * 30 + "\n"
                
                sorted_patterns = sorted(
                    patterns.items(),
                    key=lambda x: x[1].get("edge_score", 0),
                    reverse=True
                )[:5]
                
                for pattern, stats in sorted_patterns:
                    report += f"  {pattern}: {stats['win_rate']:.0%} win rate, ${stats['avg_pnl']:.2f} avg\n"
        
        report += "\n" + "=" * 50 + "\n"
        report += "END REPORT\n"
        
        return report


def run_daily_learning():
    """Run daily learning update"""
    engine = LearningEngine()
    
    print("Analyzing predictions...")
    engine.update_agent_weights()
    
    print("Analyzing pattern success...")
    engine.analyze_pattern_success()
    
    print("\nGenerating report...")
    report = engine.generate_training_report()
    print(report)
    
    return report


if __name__ == "__main__":
    run_daily_learning()
