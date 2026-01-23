#!/usr/bin/env python3
"""
MQL5 Browser Scraper using Playwright
Handles JavaScript rendering and basic bot detection bypass.

NO RANDOM DATA - All data from actual MQL5 pages.
All sources logged for audit.
"""

import json
import re
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from config import (
    MQL5_BASE_URL,
    MIN_GROWTH_PCT, MAX_DRAWDOWN_PCT, TOP_SIGNALS_COUNT,
    CONSENSUS_THRESHOLD, CONFIDENCE_BOOST,
    MQL5_SIGNALS_FILE, CONSENSUS_FILE, NEO_INTEL_FILE, LOGS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MQL5-Browser] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / f"browser_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("MQL5-Browser")


@dataclass
class Signal:
    """MQL5 Signal data"""
    signal_id: str
    name: str
    url: str
    growth_pct: float
    profit: float
    drawdown: float
    subscribers: int
    weeks: int
    win_rate: float
    trades: int
    balance: float
    equity: float
    scraped_at: str
    source_url: str


@dataclass
class Position:
    """Open position from a signal"""
    provider_id: str
    provider_name: str
    symbol: str
    direction: str
    volume: float
    profit: float
    scraped_at: str


class MQL5BrowserScraper:
    """
    Scrapes MQL5 using Playwright browser automation.
    Can handle JavaScript-rendered content.
    """
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.signals: List[Signal] = []
        self.positions: List[Position] = []
    
    async def scrape_signals_page(self, page) -> List[Dict]:
        """Scrape the main signals listing page."""
        signals = []
        
        # Navigate to signals page
        url = f"{MQL5_BASE_URL}/en/signals/mt5"
        logger.info(f"Navigating to {url}")
        
        await page.goto(url, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(5000)  # Wait for dynamic content
        
        # Take screenshot
        await page.screenshot(path=str(LOGS_DIR / "signals_page.png"))
        
        # Get page content
        html = await page.content()
        
        # Extract all signal IDs using regex
        signal_ids = set(re.findall(r'/en/signals/(\d{6,})', html))
        logger.info(f"Found {len(signal_ids)} unique signal IDs")
        
        for sig_id in signal_ids:
            signals.append({
                'signal_id': sig_id,
                'url': f"{MQL5_BASE_URL}/en/signals/{sig_id}",
                'source': url
            })
        
        return signals[:TOP_SIGNALS_COUNT * 2]
    
    async def scrape_signal_details(self, page, signal_id: str, url: str) -> Optional[Signal]:
        """Scrape detailed statistics for a signal."""
        logger.info(f"Scraping signal {signal_id}")
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)
            
            # Get page content
            html = await page.content()
            page_text = await page.inner_text('body')
            
            # Extract signal name from h1 or title
            name = f"Signal_{signal_id}"
            name_elem = await page.query_selector('h1')
            if name_elem:
                name = (await name_elem.inner_text()).strip()
                # Clean up name
                name = name.split('\n')[0].strip()
            
            # Extract statistics
            stats = {
                'growth': 0,
                'profit': 0,
                'drawdown': 0,
                'subscribers': 0,
                'weeks': 0,
                'win_rate': 0,
                'trades': 0,
                'balance': 0,
                'equity': 0
            }
            
            # Try to find stats table/section
            # MQL5 uses specific classes for stats
            
            # Growth - look for percentage with color coding
            growth_match = re.search(r'Growth[:\s]*(?:<[^>]+>)*([\d,]+(?:\.\d+)?)\s*%', html, re.I)
            if growth_match:
                stats['growth'] = float(growth_match.group(1).replace(',', ''))
            else:
                # Alternative: look in page text
                growth_match = re.search(r'Growth[:\s]*([\d,]+(?:\.\d+)?)\s*%', page_text, re.I)
                if growth_match:
                    stats['growth'] = float(growth_match.group(1).replace(',', ''))
            
            # Profit
            profit_match = re.search(r'Profit[:\s]*(?:<[^>]+>)*\$?([\d,]+(?:\.\d+)?)', html, re.I)
            if profit_match:
                stats['profit'] = float(profit_match.group(1).replace(',', ''))
            
            # Drawdown
            dd_match = re.search(r'(?:Max\s*)?(?:Draw\s*down|DD)[:\s]*(?:<[^>]+>)*([\d.]+)\s*%', html, re.I)
            if dd_match:
                stats['drawdown'] = float(dd_match.group(1))
            else:
                dd_match = re.search(r'(?:Max\s*)?Drawdown[:\s]*([\d.]+)\s*%', page_text, re.I)
                if dd_match:
                    stats['drawdown'] = float(dd_match.group(1))
            
            # Subscribers
            sub_match = re.search(r'(\d+)\s*(?:subscribers?|followers?)', page_text, re.I)
            if sub_match:
                stats['subscribers'] = int(sub_match.group(1))
            
            # Weeks/Age
            weeks_match = re.search(r'(\d+)\s*weeks?', page_text, re.I)
            if weeks_match:
                stats['weeks'] = int(weeks_match.group(1))
            
            # Win rate / Profitable trades
            win_match = re.search(r'(?:Profitable|Winning)[^:]*[:\s]*([\d.]+)\s*%', page_text, re.I)
            if win_match:
                stats['win_rate'] = float(win_match.group(1))
            
            # Total trades
            trades_match = re.search(r'(?:Trades|Total\s*trades)[:\s]*(\d+)', page_text, re.I)
            if trades_match:
                stats['trades'] = int(trades_match.group(1))
            
            # Balance
            balance_match = re.search(r'Balance[:\s]*\$?([\d,]+(?:\.\d+)?)', page_text, re.I)
            if balance_match:
                stats['balance'] = float(balance_match.group(1).replace(',', ''))
            
            # Equity
            equity_match = re.search(r'Equity[:\s]*\$?([\d,]+(?:\.\d+)?)', page_text, re.I)
            if equity_match:
                stats['equity'] = float(equity_match.group(1).replace(',', ''))
            
            logger.info(f"  {name}: Growth={stats['growth']}%, DD={stats['drawdown']}%")
            
            return Signal(
                signal_id=signal_id,
                name=name,
                url=url,
                growth_pct=stats['growth'],
                profit=stats['profit'],
                drawdown=stats['drawdown'],
                subscribers=stats['subscribers'],
                weeks=stats['weeks'],
                win_rate=stats['win_rate'],
                trades=stats['trades'],
                balance=stats['balance'],
                equity=stats['equity'],
                scraped_at=datetime.utcnow().isoformat(),
                source_url=url
            )
            
        except Exception as e:
            logger.error(f"Error scraping signal {signal_id}: {e}")
            return None
    
    async def scrape_positions(self, page, signal_id: str, url: str, provider_name: str) -> List[Position]:
        """Scrape open positions for a signal."""
        positions = []
        
        try:
            # Navigate to trading tab
            trading_url = f"{url}#!tab=trading"
            await page.goto(trading_url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)
            
            # Get page content
            html = await page.content()
            page_text = await page.inner_text('body')
            
            # Look for position patterns
            # Common forex pairs and commodities
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
                      'EURGBP', 'EURJPY', 'GBPJPY', 'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD',
                      'US30', 'US100', 'NAS100', 'SPX500', 'GER40']
            
            for symbol in symbols:
                # Check if symbol appears with buy/sell context
                pattern = rf'{symbol}\s*(?:[\s\-:]*)?(Buy|Sell)'
                matches = re.findall(pattern, page_text, re.I)
                for direction in matches:
                    positions.append(Position(
                        provider_id=signal_id,
                        provider_name=provider_name,
                        symbol=symbol,
                        direction=direction.upper(),
                        volume=0.01,  # Default
                        profit=0,
                        scraped_at=datetime.utcnow().isoformat()
                    ))
            
            # Alternative: look for table rows with positions
            # Pattern: Symbol Direction Volume Profit
            rows = re.findall(r'([A-Z]{6}|XAU|XAG)\w*\s+(Buy|Sell)\s+([\d.]+)\s+lot', page_text, re.I)
            for symbol, direction, volume in rows:
                # Avoid duplicates
                exists = any(p.symbol == symbol.upper() and p.direction == direction.upper() 
                           for p in positions)
                if not exists:
                    positions.append(Position(
                        provider_id=signal_id,
                        provider_name=provider_name,
                        symbol=symbol.upper(),
                        direction=direction.upper(),
                        volume=float(volume),
                        profit=0,
                        scraped_at=datetime.utcnow().isoformat()
                    ))
            
        except Exception as e:
            logger.debug(f"Error scraping positions for {signal_id}: {e}")
        
        if positions:
            logger.info(f"  Found {len(positions)} positions for {provider_name}")
        
        return positions
    
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on criteria."""
        filtered = []
        
        for signal in signals:
            # Skip signals without growth data
            if signal.growth_pct <= 0:
                continue
            
            if signal.growth_pct < MIN_GROWTH_PCT:
                logger.debug(f"Filtered {signal.name}: growth {signal.growth_pct}% < {MIN_GROWTH_PCT}%")
                continue
            
            if signal.drawdown > MAX_DRAWDOWN_PCT and signal.drawdown > 0:
                logger.debug(f"Filtered {signal.name}: DD {signal.drawdown}% > {MAX_DRAWDOWN_PCT}%")
                continue
            
            filtered.append(signal)
        
        # Sort by growth
        filtered.sort(key=lambda s: s.growth_pct, reverse=True)
        return filtered[:TOP_SIGNALS_COUNT]
    
    def detect_consensus(self) -> List[Dict]:
        """Detect consensus when multiple traders agree."""
        consensus = []
        
        # Group positions by symbol and direction
        groups = {}
        for pos in self.positions:
            key = f"{pos.symbol}_{pos.direction}"
            if key not in groups:
                groups[key] = []
            groups[key].append(pos)
        
        for key, group_positions in groups.items():
            if len(group_positions) >= CONSENSUS_THRESHOLD:
                symbol, direction = key.rsplit('_', 1)
                traders = list(set(p.provider_name for p in group_positions))
                
                # Calculate average growth of traders
                avg_growth = 0
                avg_win_rate = 0
                matching_signals = [s for s in self.signals if s.name in traders]
                if matching_signals:
                    avg_growth = sum(s.growth_pct for s in matching_signals) / len(matching_signals)
                    avg_win_rate = sum(s.win_rate for s in matching_signals) / len(matching_signals)
                
                confidence = min(95, 60 + (len(traders) - CONSENSUS_THRESHOLD) * 5 + CONFIDENCE_BOOST)
                
                consensus.append({
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": int(confidence),
                    "traders": traders,
                    "trader_count": len(traders),
                    "avg_growth_pct": round(avg_growth, 1),
                    "avg_win_rate": round(avg_win_rate, 1),
                    "generated_at": datetime.utcnow().isoformat(),
                    "source": "MQL5 Top Signals Consensus"
                })
                
                logger.info(f"🎯 CONSENSUS: {symbol} {direction} - {len(traders)} traders agree!")
        
        return consensus
    
    def save_results(self, consensus: List[Dict]):
        """Save results to files."""
        # Full signals data
        signals_data = {
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "MQL5.com",
            "source_url": f"{MQL5_BASE_URL}/en/signals/mt5",
            "signals": [asdict(s) for s in self.signals],
            "positions": [asdict(p) for p in self.positions]
        }
        
        with open(MQL5_SIGNALS_FILE, 'w') as f:
            json.dump(signals_data, f, indent=2)
        logger.info(f"Saved {len(self.signals)} signals to {MQL5_SIGNALS_FILE}")
        
        # NEO intel file
        intel_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "MQL5 Top Signals",
            "consensus_signals": consensus,
            "top_traders_positions": [
                {
                    "trader": p.provider_name,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "provider_id": p.provider_id
                }
                for p in self.positions[:20]
            ],
            "top_signals": [
                {
                    "name": s.name,
                    "growth_pct": s.growth_pct,
                    "drawdown": s.drawdown,
                    "subscribers": s.subscribers,
                    "url": s.url
                }
                for s in self.signals[:10]
            ],
            "summary": {
                "total_signals_tracked": len(self.signals),
                "total_positions_found": len(self.positions),
                "consensus_count": len(consensus),
                "confidence_boost": CONFIDENCE_BOOST if consensus else 0,
                "top_growth": self.signals[0].growth_pct if self.signals else 0
            }
        }
        
        with open(CONSENSUS_FILE, 'w') as f:
            json.dump(intel_data, f, indent=2)
        
        with open(NEO_INTEL_FILE, 'w') as f:
            json.dump(intel_data, f, indent=2)
        logger.info(f"Saved intel to {NEO_INTEL_FILE}")
    
    async def run(self) -> Dict:
        """Run the full scraping process."""
        logger.info("=" * 60)
        logger.info("MQL5 BROWSER SCRAPER - Starting")
        logger.info("=" * 60)
        
        from playwright.async_api import async_playwright
        
        start_time = datetime.now()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            
            # Create context with realistic browser settings
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()
            
            # Step 1: Get signals list
            signal_summaries = await self.scrape_signals_page(page)
            
            if not signal_summaries:
                logger.warning("No signals found")
                await browser.close()
                return {"status": "no_signals", "signals_found": 0}
            
            # Step 2: Scrape details for each signal
            self.signals = []
            for i, summary in enumerate(signal_summaries):
                signal = await self.scrape_signal_details(page, summary['signal_id'], summary['url'])
                if signal:
                    self.signals.append(signal)
                
                # Rate limiting
                await page.wait_for_timeout(1500)
                
                # Progress update
                if (i + 1) % 5 == 0:
                    logger.info(f"Progress: {i + 1}/{len(signal_summaries)} signals scraped")
                
                # Stop if we have enough qualifying signals
                qualifying = [s for s in self.signals if s.growth_pct >= MIN_GROWTH_PCT]
                if len(qualifying) >= TOP_SIGNALS_COUNT:
                    logger.info(f"Reached {TOP_SIGNALS_COUNT} qualifying signals, stopping")
                    break
            
            # Step 3: Filter signals
            self.signals = self.filter_signals(self.signals)
            logger.info(f"Filtered to {len(self.signals)} qualifying signals")
            
            # Step 4: Scrape positions from top signals
            self.positions = []
            for signal in self.signals[:10]:
                positions = await self.scrape_positions(page, signal.signal_id, signal.url, signal.name)
                self.positions.extend(positions)
                await page.wait_for_timeout(1000)
            
            logger.info(f"Found {len(self.positions)} total positions")
            
            await browser.close()
        
        # Step 5: Detect consensus
        consensus = self.detect_consensus()
        
        # Step 6: Save results
        self.save_results(consensus)
        
        elapsed = (datetime.now() - start_time).seconds
        
        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": elapsed,
            "signals_found": len(self.signals),
            "positions_found": len(self.positions),
            "consensus_signals": len(consensus)
        }
        
        logger.info("=" * 60)
        logger.info(f"COMPLETE: {len(self.signals)} signals, {len(self.positions)} positions, {len(consensus)} consensus")
        logger.info("=" * 60)
        
        return result


async def test_browser_scraper():
    """Test the browser scraper."""
    print("=" * 60)
    print("MQL5 BROWSER SCRAPER TEST")
    print("=" * 60)
    
    scraper = MQL5BrowserScraper(headless=True)
    result = await scraper.run()
    
    print("\nResults:")
    print(json.dumps(result, indent=2))
    
    if result.get('signals_found', 0) > 0:
        print(f"\n✅ SUCCESS - Found {result['signals_found']} qualifying signals")
        
        # Show top signals
        if scraper.signals:
            print("\nTop 5 Signals (by growth):")
            for s in scraper.signals[:5]:
                print(f"  📈 {s.name}")
                print(f"     Growth: {s.growth_pct}% | DD: {s.drawdown}% | Subs: {s.subscribers}")
                print(f"     URL: {s.url}")
        
        # Show positions
        if scraper.positions:
            print(f"\nPositions found: {len(scraper.positions)}")
            for p in scraper.positions[:10]:
                print(f"  {p.provider_name}: {p.symbol} {p.direction}")
    else:
        print("\n⚠️ No qualifying signals found")
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MQL5 Browser Scraper")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_browser_scraper())
    else:
        scraper = MQL5BrowserScraper(headless=not args.visible)
        result = asyncio.run(scraper.run())
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
