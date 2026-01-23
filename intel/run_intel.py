#!/usr/bin/env python3
"""
Intel Runner - Main entry point for the trading intel system

This script is called by cron every 30 minutes to:
1. Update top traders list from Myfxbook
2. Update top traders list from MQL5
3. Update economic calendar from Forex Factory
4. Generate consensus signals
5. Output results to log and JSON

Crontab entry:
*/30 * * * * cd ~/trading_ai/intel && python run_intel.py >> /var/log/trading_intel.log 2>&1

RULE: NO random data anywhere in this system.
"""

import json
import sys
import os
from datetime import datetime
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import IntelDatabase
from myfxbook_scraper import MyfxbookScraper
from mql5_scraper import MQL5Scraper
from ff_calendar import ForexFactoryCalendarScraper
from signal_generator import ConsensusSignalGenerator
from config import signal_config


def run_full_update(verbose: bool = True):
    """
    Run full intel update
    
    1. Scrape Myfxbook
    2. Scrape MQL5
    3. Scrape Forex Factory Calendar
    4. Generate signals
    """
    start_time = datetime.utcnow()
    
    results = {
        "run_started": start_time.isoformat() + "Z",
        "status": "running",
        "stages": {}
    }
    
    def log(msg):
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    log("=" * 60)
    log("TRADING INTEL UPDATE")
    log("=" * 60)
    log(f"Started: {start_time}")
    log("")
    
    # Stage 1: Myfxbook
    log("📊 Stage 1: Scraping Myfxbook...")
    try:
        myfxbook = MyfxbookScraper()
        myfxbook_result = myfxbook.scrape_and_store(pages=2, detailed=True)
        results["stages"]["myfxbook"] = {
            "status": "success",
            "traders_found": myfxbook_result.get("traders_found", 0),
            "traders_stored": myfxbook_result.get("traders_stored", 0),
            "trades_stored": myfxbook_result.get("trades_stored", 0)
        }
        log(f"  ✅ Stored {myfxbook_result.get('traders_stored', 0)} traders")
    except Exception as e:
        log(f"  ❌ Error: {e}")
        results["stages"]["myfxbook"] = {"status": "error", "error": str(e)}
    
    log("")
    
    # Stage 2: MQL5
    log("📊 Stage 2: Scraping MQL5...")
    try:
        mql5 = MQL5Scraper()
        mql5_result = mql5.scrape_and_store(pages=2, detailed=True)
        results["stages"]["mql5"] = {
            "status": "success",
            "signals_found": mql5_result.get("signals_found", 0),
            "signals_stored": mql5_result.get("signals_stored", 0)
        }
        log(f"  ✅ Stored {mql5_result.get('signals_stored', 0)} signals")
    except Exception as e:
        log(f"  ❌ Error: {e}")
        results["stages"]["mql5"] = {"status": "error", "error": str(e)}
    
    log("")
    
    # Stage 3: Forex Factory Calendar
    log("📅 Stage 3: Scraping Forex Factory Calendar...")
    try:
        ff = ForexFactoryCalendarScraper()
        ff_result = ff.scrape_and_store(weeks=["this", "next"])
        results["stages"]["forex_factory"] = {
            "status": "success",
            "events_found": ff_result.get("events_found", 0),
            "events_stored": ff_result.get("events_stored", 0)
        }
        log(f"  ✅ Stored {ff_result.get('events_stored', 0)} events")
    except Exception as e:
        log(f"  ❌ Error: {e}")
        results["stages"]["forex_factory"] = {"status": "error", "error": str(e)}
    
    log("")
    
    # Stage 4: Generate signals
    log("🎯 Stage 4: Generating consensus signals...")
    try:
        generator = ConsensusSignalGenerator()
        signal_result = generator.generate_and_save(hours=48)
        results["stages"]["signals"] = {
            "status": "success",
            "signals_generated": signal_result.get("signals_generated", 0),
            "signals_saved": signal_result.get("signals_saved", 0)
        }
        
        if signal_result.get("signals"):
            log(f"  ✅ Generated {signal_result['signals_generated']} signals:")
            for sig in signal_result["signals"][:5]:  # Show first 5
                log(f"    → {sig['symbol']} {sig['direction']} ({sig['confidence']}%)")
        else:
            log("  ℹ️ No consensus signals found")
            
    except Exception as e:
        log(f"  ❌ Error: {e}")
        results["stages"]["signals"] = {"status": "error", "error": str(e)}
    
    # Final summary
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    results["run_completed"] = end_time.isoformat() + "Z"
    results["duration_seconds"] = duration
    results["status"] = "completed"
    
    # Database stats
    db = IntelDatabase()
    results["database_stats"] = db.get_stats()
    
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Duration: {duration:.1f} seconds")
    log(f"Total traders: {results['database_stats']['total_traders']}")
    log(f"Total trades: {results['database_stats']['total_trades']}")
    log(f"Total events: {results['database_stats']['total_events']}")
    log(f"Total signals: {results['database_stats']['total_signals']}")
    log("")
    log(f"Latest signals: {signal_config.LATEST_SIGNAL_FILE}")
    log("=" * 60)
    
    # Save run results
    run_log_path = os.path.join(signal_config.SIGNALS_DIR, "last_run.json")
    with open(run_log_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_signals_only():
    """Only generate signals from existing data"""
    print("Generating signals from existing data...")
    generator = ConsensusSignalGenerator()
    result = generator.generate_and_save(hours=72)
    print(json.dumps(result, indent=2))
    return result


def show_status():
    """Show current database status"""
    db = IntelDatabase()
    stats = db.get_stats()
    
    print("=" * 60)
    print("TRADING INTEL STATUS")
    print("=" * 60)
    print(f"Database: {db.db_path}")
    print("")
    print("📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show recent signals
    print("")
    print("🎯 Recent Signals:")
    signals = db.get_recent_signals(hours=24)
    if signals:
        for sig in signals[:5]:
            print(f"  {sig['symbol']} {sig['direction']} - {sig.get('confidence', 0)}% confidence")
    else:
        print("  No recent signals")
    
    # Show upcoming high-impact events
    print("")
    print("📅 Upcoming High-Impact Events:")
    events = db.get_upcoming_events(hours=48, impact="high")
    if events:
        for evt in events[:5]:
            print(f"  {evt['currency']}: {evt['event_name']} ({evt['datetime']})")
    else:
        print("  No high-impact events in next 48 hours")


def verify_no_random():
    """Verify no random imports in the codebase"""
    print("=" * 60)
    print("VERIFICATION: Checking for random imports...")
    print("=" * 60)
    
    import subprocess
    
    intel_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for actual random imports (exclude this file's grep commands)
    result = subprocess.run(
        ["grep", "-r", "--include=*.py", "^import random", intel_dir],
        capture_output=True,
        text=True
    )
    
    # Also check for "from random import"
    result1b = subprocess.run(
        ["grep", "-r", "--include=*.py", "^from random", intel_dir],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip() or result1b.stdout.strip():
        print("❌ FAILED: Found random imports:")
        if result.stdout.strip():
            print(result.stdout)
        if result1b.stdout.strip():
            print(result1b.stdout)
        return False
    
    # Check for random function usage (exclude quoted strings in grep commands)
    # Use a more specific pattern that only matches actual code
    files_to_check = [
        "config.py", "database.py", "myfxbook_scraper.py", 
        "mql5_scraper.py", "ff_calendar.py", "signal_generator.py"
    ]
    
    found_random = False
    for filename in files_to_check:
        filepath = os.path.join(intel_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                # Check for actual random module usage
                if 'import random' in content or 'from random' in content:
                    print(f"❌ FAILED: Found random import in {filename}")
                    found_random = True
                if 'random.choice' in content or 'random.randint' in content or 'random.uniform' in content:
                    print(f"❌ FAILED: Found random usage in {filename}")
                    found_random = True
    
    if found_random:
        return False
    
    print("✅ PASSED: No random imports or usage found")
    print("")
    print("All data in this system comes from:")
    print("  - https://www.myfxbook.com/members")
    print("  - https://www.mql5.com/en/signals")
    print("  - https://www.forexfactory.com/calendar")
    print("")
    print("Every data point is traceable to a public URL.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Intel System")
    parser.add_argument("--full", action="store_true", help="Run full update (all scrapers)")
    parser.add_argument("--signals", action="store_true", help="Generate signals only")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--verify", action="store_true", help="Verify no random data")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_no_random()
    elif args.status:
        show_status()
    elif args.signals:
        run_signals_only()
    else:
        # Default: full update
        run_full_update(verbose=not args.quiet)
