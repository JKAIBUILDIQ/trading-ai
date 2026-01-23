#!/usr/bin/env python3
"""
NEO Knowledge Loader
Loads all knowledge files and formats them for LLM context.

Like downloading martial arts into Neo's brain in The Matrix.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class KnowledgeLoader:
    """
    Loads and manages NEO's knowledge base.
    All knowledge is from verified sources - NO random data.
    """
    
    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent / "knowledge"
        self.knowledge_dir = Path(knowledge_dir)
        
        self.strategies: Dict = {}
        self.wwcd: Dict = {}
        self.fleet: Dict = {}
        self.master_prompt: str = ""
        self.reflection_prompt: str = ""
        
        self._load_all()
    
    def _load_all(self):
        """Load all knowledge files."""
        # Load strategies
        strategies_file = self.knowledge_dir / "proven_strategies.json"
        if strategies_file.exists():
            with open(strategies_file) as f:
                self.strategies = json.load(f)
        
        # Load WWCD
        wwcd_file = self.knowledge_dir / "wwcd_playbook.json"
        if wwcd_file.exists():
            with open(wwcd_file) as f:
                self.wwcd = json.load(f)
        
        # Load fleet info
        fleet_file = self.knowledge_dir / "crellastein_fleet.json"
        if fleet_file.exists():
            with open(fleet_file) as f:
                self.fleet = json.load(f)
        
        # Load master prompt
        prompt_file = self.knowledge_dir.parent / "prompts" / "master_prompt.txt"
        if prompt_file.exists():
            with open(prompt_file) as f:
                self.master_prompt = f.read()
        
        # Load reflection prompt
        reflection_file = self.knowledge_dir.parent / "learning" / "reflection_prompt.txt"
        if reflection_file.exists():
            with open(reflection_file) as f:
                self.reflection_prompt = f.read()
    
    def get_strategy(self, name: str) -> Dict:
        """Get a specific strategy by name."""
        return self.strategies.get(name, {})
    
    def get_strategy_for_regime(self, regime: str) -> list:
        """Get strategies suitable for a market regime."""
        if regime == "trending":
            return [
                self.strategies.get("turtle_traders", {}),
                self.strategies.get("stop_hunt_fade", {})
            ]
        elif regime == "ranging":
            return [
                self.strategies.get("rsi2_connors", {}),
                self.strategies.get("stop_hunt_fade", {})
            ]
        elif regime == "news_event":
            return [
                self.strategies.get("news_fade", {})
            ]
        else:
            return [self.strategies.get("stop_hunt_fade", {})]
    
    def get_wwcd_context(self) -> str:
        """Get WWCD playbook as formatted text for LLM."""
        if not self.wwcd:
            return ""
        
        lines = [
            "=== WWCD PLAYBOOK (What Would Citadel Do) ===",
            f"Philosophy: {self.wwcd.get('core_principle', {}).get('statement', '')}",
            "",
            "LEVELS TO WATCH FOR STOP HUNTS:"
        ]
        
        levels = self.wwcd.get('tactics', {}).get('stop_hunt_awareness', {}).get('levels_to_watch', [])
        for level in levels:
            lines.append(f"  • {level}")
        
        lines.append("")
        lines.append("BEST TIMES TO TRADE:")
        for timing in self.wwcd.get('timing', {}).get('best_times_to_trade', []):
            lines.append(f"  • {timing.get('time', '')}: {timing.get('reason', '')}")
        
        lines.append("")
        lines.append("WORST TIMES:")
        for timing in self.wwcd.get('timing', {}).get('worst_times_to_trade', []):
            lines.append(f"  • {timing.get('time', '')}: {timing.get('reason', '')}")
        
        lines.append("")
        lines.append("RULES:")
        for rule in self.wwcd.get('rules_for_neo', []):
            lines.append(f"  • {rule}")
        
        return "\n".join(lines)
    
    def get_fleet_context(self) -> str:
        """Get fleet info as formatted text for LLM."""
        if not self.fleet:
            return ""
        
        lines = [
            "=== CRELLASTEIN FLEET ===",
            ""
        ]
        
        for bot_id, bot_info in self.fleet.get('bots', {}).items():
            lines.append(f"{bot_id.upper()} - {bot_info.get('nickname', '')}")
            lines.append(f"  Strategy: {bot_info.get('strategy', '')}")
            lines.append(f"  Magic: {bot_info.get('magic', '')}")
            lines.append(f"  Best when: {bot_info.get('best_used_when', '')}")
            lines.append("")
        
        lines.append("BOT SELECTION:")
        rules = self.fleet.get('coordination', {}).get('bot_selection_rules', {})
        for regime, info in rules.items():
            lines.append(f"  {regime}: Use {info.get('use', [])}")
        
        return "\n".join(lines)
    
    def get_strategies_summary(self) -> str:
        """Get strategies as formatted text for LLM."""
        if not self.strategies:
            return ""
        
        lines = [
            "=== PROVEN STRATEGIES ===",
            ""
        ]
        
        for name, strategy in self.strategies.items():
            if name == "metadata":
                continue
            
            lines.append(f"📊 {name.upper().replace('_', ' ')}")
            lines.append(f"   Source: {strategy.get('source', 'Unknown')}")
            lines.append(f"   Win Rate: {strategy.get('expected_win_rate', 'N/A')}")
            
            when_to_use = strategy.get('when_to_use', [])
            if when_to_use:
                lines.append(f"   Use when: {', '.join(when_to_use[:2])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def build_full_context(self) -> str:
        """Build complete knowledge context for NEO."""
        sections = [
            self.get_strategies_summary(),
            "",
            self.get_wwcd_context(),
            "",
            self.get_fleet_context()
        ]
        return "\n".join(sections)
    
    def get_master_prompt_with_context(self, market_context: str, account_context: str, memory_context: str) -> str:
        """Build complete prompt with all knowledge."""
        if self.master_prompt:
            # Use the master prompt as base, add current context
            return f"""{self.master_prompt}

═══════════════════════════════════════════════════════════════
CURRENT MARKET STATE
═══════════════════════════════════════════════════════════════

{market_context}

═══════════════════════════════════════════════════════════════
ACCOUNT STATE
═══════════════════════════════════════════════════════════════

{account_context}

═══════════════════════════════════════════════════════════════
YOUR RECENT MEMORY
═══════════════════════════════════════════════════════════════

{memory_context}

═══════════════════════════════════════════════════════════════
NOW ANALYZE AND MAKE YOUR DECISION
═══════════════════════════════════════════════════════════════
"""
        else:
            # Fallback to building context
            return f"""You are NEO, an autonomous trader.

{self.build_full_context()}

{market_context}

{account_context}

{memory_context}

Make your trading decision in JSON format.
"""
    
    def format_reflection_prompt(self, trade_data: Dict) -> str:
        """Format reflection prompt with trade data."""
        if not self.reflection_prompt:
            return ""
        
        return self.reflection_prompt.format(**trade_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "strategies_loaded": len(self.strategies) - 1,  # Minus metadata
            "wwcd_tactics": len(self.wwcd.get('tactics', {})),
            "fleet_bots": len(self.fleet.get('bots', {})),
            "master_prompt_loaded": bool(self.master_prompt),
            "reflection_prompt_loaded": bool(self.reflection_prompt)
        }


def test_knowledge_loader():
    """Test the knowledge loader."""
    print("=" * 60)
    print("NEO KNOWLEDGE LOADER TEST")
    print("=" * 60)
    
    loader = KnowledgeLoader()
    stats = loader.get_stats()
    
    print(f"\n📚 Knowledge Base Statistics:")
    print(f"   Strategies loaded: {stats['strategies_loaded']}")
    print(f"   WWCD tactics: {stats['wwcd_tactics']}")
    print(f"   Fleet bots: {stats['fleet_bots']}")
    print(f"   Master prompt: {'✅' if stats['master_prompt_loaded'] else '❌'}")
    print(f"   Reflection prompt: {'✅' if stats['reflection_prompt_loaded'] else '❌'}")
    
    print(f"\n📊 Available Strategies:")
    for name in loader.strategies.keys():
        if name != "metadata":
            strategy = loader.strategies[name]
            print(f"   • {name}: {strategy.get('expected_win_rate', 'N/A')} win rate")
    
    print(f"\n🤖 Fleet Bots:")
    for bot_id, bot in loader.fleet.get('bots', {}).items():
        print(f"   • {bot_id}: {bot.get('nickname', '')} - {bot.get('strategy', '')}")
    
    print("\n" + "=" * 60)
    
    return loader


if __name__ == "__main__":
    test_knowledge_loader()
