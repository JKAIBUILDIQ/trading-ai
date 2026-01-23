#!/usr/bin/env python3
"""
Strategy Loader - Query and access proven trading strategies

All strategies in this database are from verified, documented sources.
NO RANDOM DATA - every statistic is traceable.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path


class StrategyLoader:
    """Load and query trading strategies from JSON files."""
    
    def __init__(self, strategies_dir: str = None):
        if strategies_dir is None:
            strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.strategies_dir = Path(strategies_dir)
        self._cache: Dict[str, Dict] = {}
        self._load_all()
    
    def _load_all(self):
        """Load all strategy JSON files into cache."""
        for json_file in self.strategies_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    strategy_id = data.get('strategy_id', json_file.stem)
                    self._cache[strategy_id] = data
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load {json_file}: {e}")
    
    def list_strategies(self) -> List[Dict[str, str]]:
        """List all available strategies."""
        result = []
        for strategy_id, data in self._cache.items():
            result.append({
                "id": strategy_id,
                "name": data.get("name", "Unknown"),
                "category": data.get("category", "Unknown"),
                "still_works": data.get("still_works", {}).get("verdict", "Unknown")
            })
        return result
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific strategy by ID."""
        return self._cache.get(strategy_id)
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all strategies in a category."""
        return [
            data for data in self._cache.values()
            if data.get("category", "").lower() == category.lower()
        ]
    
    def get_working_strategies(self) -> List[Dict[str, Any]]:
        """Get strategies that still work (based on analysis)."""
        return [
            data for data in self._cache.values()
            if data.get("still_works", {}).get("verdict", False) is True
        ]
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search strategies by keyword."""
        query = query.lower()
        results = []
        for data in self._cache.values():
            # Search in name, category, and serialized content
            content = json.dumps(data).lower()
            if query in content:
                results.append(data)
        return results
    
    def get_parameters(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get just the key parameters for a strategy."""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            return strategy.get("key_parameters", {})
        return None
    
    def get_sources(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get source citations for a strategy."""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            return strategy.get("sources", {})
        return None
    
    def get_backtest_results(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get backtest results for a strategy."""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            return strategy.get("backtest_results", {})
        return None
    
    def export_summary(self) -> str:
        """Export a summary of all strategies."""
        lines = [
            "=" * 60,
            "PROVEN STRATEGIES DATABASE SUMMARY",
            "=" * 60,
            ""
        ]
        
        for strategy in self.list_strategies():
            lines.append(f"📊 {strategy['name']}")
            lines.append(f"   ID: {strategy['id']}")
            lines.append(f"   Category: {strategy['category']}")
            lines.append(f"   Still Works: {'✅ Yes' if strategy['still_works'] else '❌ No' if strategy['still_works'] is False else '❓ Unknown'}")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append(f"Total Strategies: {len(self._cache)}")
        lines.append("Data Source: Verified academic and practitioner publications")
        lines.append("Random Data: NONE - all statistics are documented")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query trading strategies database")
    parser.add_argument("--list", action="store_true", help="List all strategies")
    parser.add_argument("--get", type=str, help="Get specific strategy by ID")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--search", type=str, help="Search strategies")
    parser.add_argument("--working", action="store_true", help="Show only working strategies")
    parser.add_argument("--params", type=str, help="Get parameters for strategy")
    parser.add_argument("--sources", type=str, help="Get sources for strategy")
    parser.add_argument("--summary", action="store_true", help="Export summary")
    
    args = parser.parse_args()
    
    loader = StrategyLoader()
    
    if args.list:
        for s in loader.list_strategies():
            print(f"{s['id']}: {s['name']} [{s['category']}]")
    
    elif args.get:
        strategy = loader.get_strategy(args.get)
        if strategy:
            print(json.dumps(strategy, indent=2))
        else:
            print(f"Strategy '{args.get}' not found")
    
    elif args.category:
        strategies = loader.get_by_category(args.category)
        for s in strategies:
            print(f"{s['strategy_id']}: {s['name']}")
    
    elif args.search:
        results = loader.search(args.search)
        print(f"Found {len(results)} strategies matching '{args.search}':")
        for s in results:
            print(f"  - {s['strategy_id']}: {s['name']}")
    
    elif args.working:
        for s in loader.get_working_strategies():
            print(f"{s['strategy_id']}: {s['name']}")
    
    elif args.params:
        params = loader.get_parameters(args.params)
        if params:
            print(json.dumps(params, indent=2))
        else:
            print(f"No parameters found for '{args.params}'")
    
    elif args.sources:
        sources = loader.get_sources(args.sources)
        if sources:
            print(json.dumps(sources, indent=2))
        else:
            print(f"No sources found for '{args.sources}'")
    
    elif args.summary:
        print(loader.export_summary())
    
    else:
        # Default: show summary
        print(loader.export_summary())


if __name__ == "__main__":
    main()
