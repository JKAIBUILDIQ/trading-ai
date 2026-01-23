#!/usr/bin/env python3
"""
Aggregate Insights from Multiple LLM Analyses
Find consensus recommendations across models

NO RANDOM DATA - All analysis based on LLM outputs and proven parameters
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter


class InsightAggregator:
    """Aggregates insights from multiple LLM analysis files."""
    
    def __init__(self, analysis_dir: str = None):
        if analysis_dir is None:
            analysis_dir = os.path.dirname(os.path.abspath(__file__))
        self.analysis_dir = analysis_dir
        self.analyses: Dict[str, Dict] = {}
        self._load_analyses()
    
    def _load_analyses(self):
        """Load all analysis JSON files."""
        for f in os.listdir(self.analysis_dir):
            if f.startswith('analysis_') and f.endswith('.json'):
                filepath = os.path.join(self.analysis_dir, f)
                try:
                    with open(filepath, 'r') as file:
                        content = file.read()
                        # Try to extract JSON from the response
                        data = self._extract_json(content)
                        if data:
                            self.analyses[f] = data
                            print(f"✅ Loaded: {f}")
                        else:
                            print(f"⚠️  Could not parse JSON from: {f}")
                except Exception as e:
                    print(f"❌ Error loading {f}: {e}")
    
    def _extract_json(self, content: str) -> Optional[Dict]:
        """Extract JSON from LLM response (may have extra text)."""
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in response
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, content)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def find_consensus_recommendations(self) -> List[Dict[str, Any]]:
        """Find recommendations that multiple LLMs agree on."""
        all_recs = []
        rec_counter = Counter()
        
        for model_name, analysis in self.analyses.items():
            recs = analysis.get('recommendations', [])
            for rec in recs:
                if isinstance(rec, dict):
                    # Normalize recommendation for counting
                    key = f"{rec.get('bot', 'unknown')}:{rec.get('change', 'unknown')[:50]}"
                    rec_counter[key] += 1
                    all_recs.append({
                        **rec,
                        'source_model': model_name
                    })
                elif isinstance(rec, str):
                    rec_counter[rec[:50]] += 1
                    all_recs.append({
                        'change': rec,
                        'source_model': model_name
                    })
        
        # Find recommendations mentioned by multiple models
        consensus = []
        seen_keys = set()
        for rec in all_recs:
            if isinstance(rec, dict):
                key = f"{rec.get('bot', 'unknown')}:{rec.get('change', 'unknown')[:50]}"
            else:
                key = str(rec)[:50]
            
            count = rec_counter.get(key, 1)
            if count >= 2 and key not in seen_keys:  # At least 2 models agree
                seen_keys.add(key)
                if isinstance(rec, dict):
                    consensus_item = dict(rec)
                else:
                    consensus_item = {'change': rec}
                consensus_item['agreement_count'] = count
                consensus.append(consensus_item)
        
        # Sort by agreement count
        consensus.sort(key=lambda x: x.get('agreement_count', 0), reverse=True)
        return consensus
    
    def calculate_average_grades(self) -> Dict[str, Dict[str, float]]:
        """Calculate average grades across all models for each bot."""
        grades = {
            'v007': {'coherence': [], 'risk': [], 'edge': []},
            'v008': {'coherence': [], 'risk': [], 'edge': []},
            'v010': {'coherence': [], 'risk': [], 'edge': []},
            'v015': {'coherence': [], 'risk': [], 'edge': []},
            'v020': {'coherence': [], 'risk': [], 'edge': []},
            'fleet': {'coordination': []}
        }
        
        for model_name, analysis in self.analyses.items():
            for bot in ['v007', 'v008', 'v010', 'v015', 'v020']:
                bot_analysis = analysis.get(f'{bot}_analysis', {})
                if 'coherence_grade' in bot_analysis:
                    grades[bot]['coherence'].append(bot_analysis['coherence_grade'])
                if 'risk_grade' in bot_analysis:
                    grades[bot]['risk'].append(bot_analysis['risk_grade'])
                if 'edge_grade' in bot_analysis:
                    grades[bot]['edge'].append(bot_analysis['edge_grade'])
            
            fleet = analysis.get('fleet_analysis', {})
            if 'coordination_grade' in fleet:
                grades['fleet']['coordination'].append(fleet['coordination_grade'])
        
        # Calculate averages
        averages = {}
        for bot, metrics in grades.items():
            averages[bot] = {}
            for metric, values in metrics.items():
                if values:
                    averages[bot][metric] = round(sum(values) / len(values), 1)
                else:
                    averages[bot][metric] = None
        
        return averages
    
    def collect_risk_warnings(self) -> List[str]:
        """Collect all unique risk warnings across models."""
        warnings = set()
        for model_name, analysis in self.analyses.items():
            for warning in analysis.get('risk_warnings', []):
                warnings.add(warning if isinstance(warning, str) else str(warning))
            for risk in analysis.get('key_risks', []):
                warnings.add(risk if isinstance(risk, str) else str(risk))
        return list(warnings)
    
    def get_overall_grades(self) -> Dict[str, str]:
        """Collect overall grades from each model."""
        grades = {}
        for model_name, analysis in self.analyses.items():
            grades[model_name] = {
                'grade': analysis.get('overall_grade', 'N/A'),
                'profit_probability': analysis.get('profit_probability_1yr', 'N/A')
            }
        return grades
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive consensus report."""
        report = {
            "report_date": datetime.now().isoformat(),
            "models_consulted": list(self.analyses.keys()),
            "model_count": len(self.analyses),
            "consensus_recommendations": self.find_consensus_recommendations(),
            "average_grades": self.calculate_average_grades(),
            "risk_warnings": self.collect_risk_warnings(),
            "overall_grades_by_model": self.get_overall_grades(),
            "status": "AGGREGATED" if self.analyses else "NO_DATA"
        }
        
        # Add summary statistics
        if self.analyses:
            grades = [a.get('overall_grade', 'C') for a in self.analyses.values() if a.get('overall_grade')]
            if grades:
                # Convert letter grades to numbers for averaging
                grade_values = {'A+': 4.3, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                               'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.7, 'F': 0}
                numeric_grades = [grade_values.get(g, 2.0) for g in grades]
                avg_numeric = sum(numeric_grades) / len(numeric_grades)
                
                # Convert back to letter
                for letter, value in sorted(grade_values.items(), key=lambda x: -x[1]):
                    if avg_numeric >= value:
                        report['consensus_overall_grade'] = letter
                        break
        
        return report
    
    def save_report(self, filename: str = 'consensus_report.json'):
        """Save the consensus report to file."""
        report = self.generate_report()
        filepath = os.path.join(self.analysis_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Consensus report saved to: {filepath}")
        return report


def create_sample_analysis():
    """Create a sample analysis file for testing."""
    sample = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "model_name": "sample_model",
        "v007_analysis": {
            "alignment_rating": "Fair",
            "coherence_grade": 7,
            "risk_grade": 5,
            "edge_grade": 6,
            "top_3_changes": [
                "Reduce position size to 2% max",
                "Add volatility filter",
                "Tighten stop to 2 ATR"
            ],
            "notes": "Good price action but aggressive sizing"
        },
        "v008_analysis": {
            "alignment_rating": "Poor",
            "coherence_grade": 6,
            "risk_grade": 5,
            "edge_grade": 4,
            "top_3_changes": [
                "CRITICAL: Change RSI from 14 to 2",
                "Lower oversold threshold to 10",
                "Add 200 SMA trend filter"
            ],
            "critical_finding": "RSI(14) has ~50% win rate vs RSI(2) with 88%",
            "notes": "Major deviation from proven parameters"
        },
        "v010_analysis": {
            "alignment_rating": "Good",
            "coherence_grade": 8,
            "risk_grade": 5,
            "edge_grade": 7,
            "top_3_changes": [
                "Reduce position size",
                "Add session filter (London/NY only)",
                "Increase confirmation bars to 3"
            ],
            "notes": "Novel strategy with theoretical edge"
        },
        "v015_analysis": {
            "alignment_rating": "Fair",
            "coherence_grade": 7,
            "risk_grade": 6,
            "edge_grade": 6,
            "top_3_changes": [
                "Reduce GTO randomization",
                "Add time-of-day filter",
                "Implement trailing stop"
            ],
            "notes": "GTO approach is innovative but unproven"
        },
        "v020_analysis": {
            "alignment_rating": "Good",
            "coherence_grade": 8,
            "risk_grade": 7,
            "edge_grade": 7,
            "top_3_changes": [
                "Reduce max daily loss to 5%",
                "Add inter-bot correlation check",
                "Implement gradual position scaling"
            ],
            "notes": "Good risk management framework"
        },
        "fleet_analysis": {
            "coordination_grade": 7,
            "herding_risk": "Medium",
            "coverage_gaps": ["No volatility regime detection", "No news filter"],
            "redundancies": ["v007 and v010 may conflict"]
        },
        "recommendations": [
            {
                "priority": 1,
                "bot": "v008",
                "change": "Change RSI period from 14 to 2",
                "source": "Connors RSI(2) research - proven 88% win rate",
                "impact": "High"
            },
            {
                "priority": 2,
                "bot": "ALL",
                "change": "Reduce position sizing from 5-10% to 1-2%",
                "source": "Turtle Trading rules - proven over 40 years",
                "impact": "High"
            },
            {
                "priority": 3,
                "bot": "v020",
                "change": "Add volatility filter (pause during VIX > 25)",
                "source": "Academic research on vol regimes",
                "impact": "Medium"
            }
        ],
        "risk_warnings": [
            "Position sizing 5-10% is aggressive vs 1-2% proven optimal",
            "No volatility regime filter - vulnerable in high VIX",
            "v008 RSI(14) has no proven edge"
        ],
        "overall_grade": "B-",
        "profit_probability_1yr": "60%",
        "key_risks": [
            "Aggressive position sizing could lead to large drawdowns",
            "RSI(14) in v008 may be net negative",
            "Correlation between bots not fully managed"
        ]
    }
    return sample


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate LLM analysis insights")
    parser.add_argument("--create-sample", action="store_true", 
                        help="Create a sample analysis file for testing")
    parser.add_argument("--report", action="store_true",
                        help="Generate consensus report")
    args = parser.parse_args()
    
    if args.create_sample:
        sample = create_sample_analysis()
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'analysis_sample.json')
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
        print(f"✅ Sample analysis created: {filepath}")
        print("   Run --report to generate consensus from this sample")
        return
    
    print("=" * 60)
    print("CRELLASTEIN BOT ANALYSIS - Insight Aggregation")
    print("=" * 60)
    print("")
    
    aggregator = InsightAggregator()
    
    if not aggregator.analyses:
        print("⚠️  No analysis files found!")
        print("")
        print("Options:")
        print("  1. Run LLM analysis: bash run_analysis.sh")
        print("  2. Create sample: python3 aggregate_insights.py --create-sample")
        print("")
        return
    
    report = aggregator.save_report()
    
    # Print summary
    print("")
    print("=" * 60)
    print("CONSENSUS SUMMARY")
    print("=" * 60)
    print(f"Models consulted: {report['model_count']}")
    print(f"Consensus recommendations: {len(report['consensus_recommendations'])}")
    print(f"Risk warnings: {len(report['risk_warnings'])}")
    
    if report.get('consensus_overall_grade'):
        print(f"Consensus grade: {report['consensus_overall_grade']}")
    
    print("")
    print("Top Recommendations:")
    for i, rec in enumerate(report['consensus_recommendations'][:5], 1):
        bot = rec.get('bot', 'ALL')
        change = rec.get('change', 'Unknown')[:60]
        agreement = rec.get('agreement_count', 1)
        print(f"  {i}. [{bot}] {change}... ({agreement} models agree)")
    
    print("")
    print("Risk Warnings:")
    for warning in report['risk_warnings'][:5]:
        print(f"  ⚠️  {warning[:70]}...")
    
    print("")
    print("=" * 60)


if __name__ == "__main__":
    main()
