#!/usr/bin/env python3
"""
MQL5 Intel Runner for NEO
Runs the browser scraper and outputs intel for NEO's decision making.

Schedule: Every 15 minutes via cron
NO RANDOM DATA - All data from MQL5.com

Usage:
  python3 run_mql5_intel.py          # Normal run
  python3 run_mql5_intel.py --test   # Quick test
"""

import sys
import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))

from config import NEO_INTEL_FILE, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MQL5-Runner] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / f"runner_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("MQL5-Runner")


async def run_scraper(max_signals: int = 30) -> dict:
    """Run the MQL5 browser scraper."""
    from mql5_browser_scraper import MQL5BrowserScraper
    
    logger.info("=" * 60)
    logger.info("MQL5 INTEL RUN - Starting")
    logger.info(f"Target: {max_signals} signals")
    logger.info("=" * 60)
    
    try:
        scraper = MQL5BrowserScraper(headless=True)
        result = await scraper.run()
        
        logger.info(f"Result: {result['status']}")
        logger.info(f"Signals: {result.get('signals_found', 0)}")
        logger.info(f"Positions: {result.get('positions_found', 0)}")
        logger.info(f"Consensus: {result.get('consensus_signals', 0)}")
        
        # Show top signals
        if scraper.signals:
            logger.info("Top 5 Signals:")
            for s in scraper.signals[:5]:
                logger.info(f"  📈 {s.name}: {s.growth_pct}% growth")
        
        # Show consensus if found
        if result.get('consensus_signals', 0) > 0:
            logger.info("=" * 60)
            logger.info("🎯 CONSENSUS SIGNALS FOUND!")
            with open(NEO_INTEL_FILE) as f:
                intel = json.load(f)
            for sig in intel.get('consensus_signals', []):
                logger.info(f"  {sig['symbol']} {sig['direction']} - {sig['trader_count']} traders, {sig['confidence']}% confidence")
            logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


async def quick_test() -> dict:
    """Quick test - scrape fewer signals."""
    from mql5_browser_scraper import MQL5BrowserScraper
    from config import MIN_GROWTH_PCT
    
    logger.info("=" * 60)
    logger.info("MQL5 QUICK TEST")
    logger.info("=" * 60)
    
    scraper = MQL5BrowserScraper(headless=True)
    
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()
        
        # Get signals
        signals = await scraper.scrape_signals_page(page)
        logger.info(f"Found {len(signals)} signal IDs")
        
        # Scrape 10 signals
        for summary in signals[:10]:
            signal = await scraper.scrape_signal_details(page, summary['signal_id'], summary['url'])
            if signal:
                scraper.signals.append(signal)
            await page.wait_for_timeout(800)
        
        # Filter
        filtered = [s for s in scraper.signals if s.growth_pct >= MIN_GROWTH_PCT and s.drawdown <= 50]
        filtered.sort(key=lambda x: x.growth_pct, reverse=True)
        scraper.signals = filtered[:10]
        
        await browser.close()
    
    # Save results
    consensus = scraper.detect_consensus()
    scraper.save_results(consensus)
    
    logger.info(f"✅ Test complete: {len(scraper.signals)} qualifying signals")
    
    return {
        "status": "success",
        "signals_found": len(scraper.signals),
        "consensus_signals": len(consensus)
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MQL5 Intel Runner for NEO")
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument("--max-signals", type=int, default=30, help="Max signals to scrape")
    args = parser.parse_args()
    
    if args.test:
        result = asyncio.run(quick_test())
    else:
        result = asyncio.run(run_scraper(args.max_signals))
    
    print(json.dumps(result, indent=2))
    
    # Return exit code based on success
    sys.exit(0 if result.get('status') in ['success', 'no_signals'] else 1)


if __name__ == "__main__":
    main()
