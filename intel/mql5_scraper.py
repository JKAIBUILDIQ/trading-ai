#!/usr/bin/env python3
"""
MQL5 Signal Intel Scraper
Scrapes top trading signals from MQL5.com and detects consensus.

NO RANDOM DATA - All data from real MQL5 pages.
All sources logged for audit.

Feeds intel to NEO for enhanced decision making.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

from config import (
    MQL5_BASE_URL, MQL5_SIGNALS_URL,
    USER_AGENT, REQUEST_TIMEOUT, REQUEST_DELAY, MAX_RETRIES,
    MIN_GROWTH_PCT, MIN_HISTORY_WEEKS, MIN_ALGO_TRADING_PCT, MAX_DRAWDOWN_PCT,
    TOP_SIGNALS_COUNT, CONSENSUS_THRESHOLD, CONSENSUS_WINDOW_HOURS,
    MQL5_SIGNALS_FILE, CONSENSUS_FILE, NEO_INTEL_FILE, LOGS_DIR,
    LOG_LEVEL, LOG_FORMAT, CONFIDENCE_BOOST
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / f"mql5_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("MQL5")


@dataclass
class SignalProvider:
    """MQL5 Signal Provider data"""
    signal_id: str
    name: str
    url: str
    growth_pct: float
    win_rate: float
    max_drawdown: float
    subscribers: int
    weeks_history: int
    algo_trading_pct: float
    profit: float
    balance: float
    equity: float
    scraped_at: str
    source_url: str


@dataclass
class TradePosition:
    """Open position from a signal provider"""
    provider_id: str
    provider_name: str
    symbol: str
    direction: str  # BUY or SELL
    volume: float
    open_price: float
    open_time: str
    profit: float
    scraped_at: str


@dataclass
class ConsensusSignal:
    """Consensus signal when multiple traders agree"""
    symbol: str
    direction: str
    confidence: int
    traders: List[str]
    trader_count: int
    avg_growth_pct: float
    avg_win_rate: float
    generated_at: str
    source: str = "MQL5 Top Signals Consensus"


class MQL5Scraper:
    """
    Scrapes MQL5.com for top signal providers and their positions.
    Detects consensus when multiple top traders enter same direction.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        
        self.signals: List[SignalProvider] = []
        self.positions: List[TradePosition] = []
        self.consensus: List[ConsensusSignal] = []
        self.last_scrape: Optional[datetime] = None
    
    def _request(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """Make HTTP request with retry logic."""
        for attempt in range(retries):
            try:
                time.sleep(REQUEST_DELAY)  # Rate limiting
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 403:
                    logger.warning(f"Access forbidden (403) - MQL5 may be blocking scrapers")
                    return None
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429) - waiting 60s")
                    time.sleep(60)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
        
        return None
    
    def _parse_number(self, text: str) -> float:
        """Parse number from text, handling various formats."""
        if not text:
            return 0.0
        # Remove currency symbols, spaces, commas
        cleaned = re.sub(r'[^\d.\-]', '', text.replace(',', ''))
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    def _parse_percentage(self, text: str) -> float:
        """Parse percentage from text."""
        if not text:
            return 0.0
        # Extract number before %
        match = re.search(r'([\d.]+)\s*%', text)
        if match:
            return float(match.group(1))
        return self._parse_number(text)
    
    def scrape_signals_list(self) -> List[Dict]:
        """
        Scrape the main signals page to get top performers.
        Returns list of signal summaries.
        """
        logger.info(f"Scraping signals list from {MQL5_SIGNALS_URL}")
        
        # MQL5 uses different sorting options
        # Sort by growth: ?s=growth&sd=1
        url = f"{MQL5_SIGNALS_URL}/mt5/list?s=profit&sd=1"
        
        html = self._request(url)
        if not html:
            logger.error("Failed to fetch signals list")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        signals = []
        
        # Find signal cards/rows
        # MQL5 structure varies, try multiple selectors
        signal_elements = soup.select('.signal-card, .signals-table tr, .signal-row, [data-signal-id]')
        
        if not signal_elements:
            # Try alternative structure
            signal_elements = soup.select('a[href*="/en/signals/"]')
            logger.info(f"Found {len(signal_elements)} signal links")
        
        for elem in signal_elements[:50]:  # Process more to filter later
            try:
                # Extract signal URL
                link = elem.get('href') if elem.name == 'a' else elem.select_one('a[href*="/signals/"]')
                if link:
                    href = link if isinstance(link, str) else link.get('href', '')
                    if '/signals/' in href and not href.endswith('/signals/'):
                        # Extract signal ID from URL
                        match = re.search(r'/signals/(\d+)', href)
                        if match:
                            signal_id = match.group(1)
                            full_url = f"{MQL5_BASE_URL}{href}" if href.startswith('/') else href
                            
                            signals.append({
                                'signal_id': signal_id,
                                'url': full_url,
                                'source': url
                            })
            except Exception as e:
                logger.debug(f"Error parsing signal element: {e}")
                continue
        
        # Remove duplicates
        seen = set()
        unique_signals = []
        for s in signals:
            if s['signal_id'] not in seen:
                seen.add(s['signal_id'])
                unique_signals.append(s)
        
        logger.info(f"Found {len(unique_signals)} unique signals")
        return unique_signals[:TOP_SIGNALS_COUNT * 2]  # Get extra for filtering
    
    def scrape_signal_details(self, signal_id: str, url: str) -> Optional[SignalProvider]:
        """
        Scrape detailed info for a specific signal.
        """
        logger.info(f"Scraping signal {signal_id}: {url}")
        
        html = self._request(url)
        if not html:
            logger.warning(f"Failed to fetch signal {signal_id}")
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # Extract signal name
            name_elem = soup.select_one('h1, .signal-name, .s-name')
            name = name_elem.get_text(strip=True) if name_elem else f"Signal_{signal_id}"
            
            # Extract statistics - MQL5 uses various class names
            stats = {}
            
            # Look for stat rows/cells
            stat_elements = soup.select('.s-stat-value, .stat-value, .signal-stat, [class*="stat"]')
            stat_labels = soup.select('.s-stat-name, .stat-name, .signal-stat-label, [class*="label"]')
            
            # Try to match labels to values
            for label_elem in stat_labels:
                label = label_elem.get_text(strip=True).lower()
                value_elem = label_elem.find_next_sibling() or label_elem.find_parent().select_one('[class*="value"]')
                if value_elem:
                    value = value_elem.get_text(strip=True)
                    
                    if 'growth' in label:
                        stats['growth'] = self._parse_percentage(value)
                    elif 'profit' in label and 'factor' not in label:
                        stats['profit'] = self._parse_number(value)
                    elif 'drawdown' in label:
                        stats['drawdown'] = self._parse_percentage(value)
                    elif 'win' in label or 'profitable' in label:
                        stats['win_rate'] = self._parse_percentage(value)
                    elif 'subscriber' in label:
                        stats['subscribers'] = int(self._parse_number(value))
                    elif 'week' in label:
                        stats['weeks'] = int(self._parse_number(value))
                    elif 'balance' in label:
                        stats['balance'] = self._parse_number(value)
                    elif 'equity' in label:
                        stats['equity'] = self._parse_number(value)
                    elif 'algo' in label or 'automated' in label:
                        stats['algo_pct'] = self._parse_percentage(value)
            
            # Alternative: look for specific patterns in page text
            page_text = soup.get_text()
            
            if 'growth' not in stats:
                match = re.search(r'Growth[:\s]*([\d,]+(?:\.\d+)?)\s*%', page_text, re.I)
                if match:
                    stats['growth'] = float(match.group(1).replace(',', ''))
            
            if 'drawdown' not in stats:
                match = re.search(r'(?:Max\s*)?Drawdown[:\s]*([\d.]+)\s*%', page_text, re.I)
                if match:
                    stats['drawdown'] = float(match.group(1))
            
            if 'win_rate' not in stats:
                match = re.search(r'(?:Win|Profitable)[^:]*[:\s]*([\d.]+)\s*%', page_text, re.I)
                if match:
                    stats['win_rate'] = float(match.group(1))
            
            if 'subscribers' not in stats:
                match = re.search(r'(\d+)\s*(?:subscriber|follower)', page_text, re.I)
                if match:
                    stats['subscribers'] = int(match.group(1))
            
            return SignalProvider(
                signal_id=signal_id,
                name=name,
                url=url,
                growth_pct=stats.get('growth', 0),
                win_rate=stats.get('win_rate', 0),
                max_drawdown=stats.get('drawdown', 0),
                subscribers=stats.get('subscribers', 0),
                weeks_history=stats.get('weeks', 0),
                algo_trading_pct=stats.get('algo_pct', 0),
                profit=stats.get('profit', 0),
                balance=stats.get('balance', 0),
                equity=stats.get('equity', 0),
                scraped_at=datetime.utcnow().isoformat(),
                source_url=url
            )
            
        except Exception as e:
            logger.error(f"Error parsing signal {signal_id}: {e}")
            return None
    
    def scrape_signal_positions(self, signal_id: str, url: str, provider_name: str) -> List[TradePosition]:
        """
        Scrape current open positions for a signal.
        """
        # MQL5 shows positions on the signal page or a /trading tab
        positions_url = f"{url}#!tab=trading"
        
        html = self._request(positions_url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        positions = []
        
        # Look for position rows
        position_rows = soup.select('.position-row, .trade-row, tr[data-symbol], .trading-table tr')
        
        for row in position_rows:
            try:
                cells = row.select('td') or row.select('.cell')
                if len(cells) < 4:
                    continue
                
                # Try to extract position data
                symbol = None
                direction = None
                volume = 0
                price = 0
                profit = 0
                
                # Look for symbol (EURUSD, XAUUSD, etc.)
                row_text = row.get_text()
                symbol_match = re.search(r'([A-Z]{6}|XAU|XAG|US30|NAS100)', row_text)
                if symbol_match:
                    symbol = symbol_match.group(1)
                
                # Look for direction
                if 'buy' in row_text.lower():
                    direction = 'BUY'
                elif 'sell' in row_text.lower():
                    direction = 'SELL'
                
                # Look for volume
                vol_match = re.search(r'(\d+\.?\d*)\s*(?:lot|lots)', row_text, re.I)
                if vol_match:
                    volume = float(vol_match.group(1))
                
                if symbol and direction:
                    positions.append(TradePosition(
                        provider_id=signal_id,
                        provider_name=provider_name,
                        symbol=symbol,
                        direction=direction,
                        volume=volume,
                        open_price=price,
                        open_time=datetime.utcnow().isoformat(),
                        profit=profit,
                        scraped_at=datetime.utcnow().isoformat()
                    ))
                    
            except Exception as e:
                logger.debug(f"Error parsing position row: {e}")
                continue
        
        return positions
    
    def filter_signals(self, signals: List[SignalProvider]) -> List[SignalProvider]:
        """
        Filter signals based on criteria.
        """
        filtered = []
        
        for signal in signals:
            # Apply filters
            if signal.growth_pct < MIN_GROWTH_PCT:
                logger.debug(f"Filtered {signal.name}: growth {signal.growth_pct}% < {MIN_GROWTH_PCT}%")
                continue
            
            if signal.max_drawdown > MAX_DRAWDOWN_PCT:
                logger.debug(f"Filtered {signal.name}: drawdown {signal.max_drawdown}% > {MAX_DRAWDOWN_PCT}%")
                continue
            
            # Note: weeks_history and algo_pct may not always be available
            # so we're lenient on those
            
            filtered.append(signal)
        
        # Sort by growth and take top N
        filtered.sort(key=lambda s: s.growth_pct, reverse=True)
        return filtered[:TOP_SIGNALS_COUNT]
    
    def detect_consensus(self, positions: List[TradePosition]) -> List[ConsensusSignal]:
        """
        Detect consensus when multiple traders have positions in same direction.
        """
        consensus_signals = []
        
        # Group positions by symbol and direction
        position_groups: Dict[str, List[TradePosition]] = {}
        
        for pos in positions:
            key = f"{pos.symbol}_{pos.direction}"
            if key not in position_groups:
                position_groups[key] = []
            position_groups[key].append(pos)
        
        # Check for consensus
        for key, group_positions in position_groups.items():
            if len(group_positions) >= CONSENSUS_THRESHOLD:
                symbol, direction = key.rsplit('_', 1)
                
                # Get unique traders
                traders = list(set(p.provider_name for p in group_positions))
                
                # Calculate averages from signal data
                avg_growth = 0
                avg_win_rate = 0
                matching_signals = [s for s in self.signals if s.name in traders]
                if matching_signals:
                    avg_growth = sum(s.growth_pct for s in matching_signals) / len(matching_signals)
                    avg_win_rate = sum(s.win_rate for s in matching_signals) / len(matching_signals)
                
                # Calculate confidence
                base_confidence = 60
                trader_bonus = (len(traders) - CONSENSUS_THRESHOLD) * 5
                growth_bonus = min(20, avg_growth / 500)  # Up to +20 for high growth
                confidence = min(95, base_confidence + trader_bonus + growth_bonus + CONFIDENCE_BOOST)
                
                consensus_signals.append(ConsensusSignal(
                    symbol=symbol,
                    direction=direction,
                    confidence=int(confidence),
                    traders=traders,
                    trader_count=len(traders),
                    avg_growth_pct=round(avg_growth, 1),
                    avg_win_rate=round(avg_win_rate, 1),
                    generated_at=datetime.utcnow().isoformat()
                ))
                
                logger.info(f"🎯 CONSENSUS DETECTED: {symbol} {direction} - {len(traders)} traders agree")
        
        return consensus_signals
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full scraping process.
        Returns summary of results.
        """
        logger.info("=" * 60)
        logger.info("MQL5 SIGNAL INTEL SCRAPER - Starting")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Get signals list
        signal_summaries = self.scrape_signals_list()
        
        if not signal_summaries:
            logger.warning("No signals found - MQL5 may be blocking or page structure changed")
            # Return cached data if available
            return self._load_cached_data()
        
        # Step 2: Scrape details for each signal
        self.signals = []
        for summary in signal_summaries:
            signal = self.scrape_signal_details(summary['signal_id'], summary['url'])
            if signal:
                self.signals.append(signal)
            
            # Stop if we have enough good signals
            if len(self.signals) >= TOP_SIGNALS_COUNT:
                break
        
        # Step 3: Filter signals
        self.signals = self.filter_signals(self.signals)
        logger.info(f"Filtered to {len(self.signals)} qualifying signals")
        
        # Step 4: Scrape positions for top signals
        self.positions = []
        for signal in self.signals[:10]:  # Only top 10 for positions
            positions = self.scrape_signal_positions(signal.signal_id, signal.url, signal.name)
            self.positions.extend(positions)
        
        logger.info(f"Found {len(self.positions)} open positions")
        
        # Step 5: Detect consensus
        self.consensus = self.detect_consensus(self.positions)
        
        # Step 6: Save results
        self._save_results()
        
        # Summary
        elapsed = (datetime.now() - start_time).seconds
        self.last_scrape = datetime.now()
        
        summary = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": elapsed,
            "signals_found": len(self.signals),
            "positions_found": len(self.positions),
            "consensus_signals": len(self.consensus),
            "top_growth": self.signals[0].growth_pct if self.signals else 0
        }
        
        logger.info(f"Completed in {elapsed}s: {len(self.signals)} signals, {len(self.consensus)} consensus")
        
        return summary
    
    def _save_results(self):
        """Save scraped data to files."""
        # Save full signal data
        signals_data = {
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "MQL5.com",
            "signals": [asdict(s) for s in self.signals],
            "positions": [asdict(p) for p in self.positions]
        }
        
        with open(MQL5_SIGNALS_FILE, 'w') as f:
            json.dump(signals_data, f, indent=2)
        logger.info(f"Saved signals to {MQL5_SIGNALS_FILE}")
        
        # Save consensus data for NEO
        consensus_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "MQL5 Top Signals",
            "consensus_signals": [asdict(c) for c in self.consensus],
            "top_traders_positions": [
                {
                    "trader": p.provider_name,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "provider_id": p.provider_id
                }
                for p in self.positions[:20]
            ],
            "summary": {
                "total_signals_tracked": len(self.signals),
                "consensus_count": len(self.consensus),
                "confidence_boost": CONFIDENCE_BOOST if self.consensus else 0
            }
        }
        
        with open(CONSENSUS_FILE, 'w') as f:
            json.dump(consensus_data, f, indent=2)
        
        with open(NEO_INTEL_FILE, 'w') as f:
            json.dump(consensus_data, f, indent=2)
        logger.info(f"Saved consensus to {NEO_INTEL_FILE}")
    
    def _load_cached_data(self) -> Dict[str, Any]:
        """Load previously cached data if scraping fails."""
        try:
            if MQL5_SIGNALS_FILE.exists():
                with open(MQL5_SIGNALS_FILE) as f:
                    data = json.load(f)
                logger.info("Loaded cached signal data")
                return {
                    "status": "cached",
                    "timestamp": data.get("scraped_at"),
                    "signals_found": len(data.get("signals", [])),
                    "note": "Using cached data - live scrape failed"
                }
        except:
            pass
        
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "signals_found": 0,
            "error": "No cached data available"
        }


def test_scraper():
    """Test the scraper with a quick run."""
    print("=" * 60)
    print("MQL5 SCRAPER TEST")
    print("=" * 60)
    
    scraper = MQL5Scraper()
    
    # Just test fetching the signals list
    print("\nFetching signals list...")
    signals = scraper.scrape_signals_list()
    
    print(f"Found {len(signals)} signals")
    
    if signals:
        print("\nFirst 5 signals:")
        for s in signals[:5]:
            print(f"  - ID: {s['signal_id']}, URL: {s['url'][:50]}...")
        
        # Test scraping one signal
        if signals:
            print(f"\nScraping details for first signal...")
            details = scraper.scrape_signal_details(signals[0]['signal_id'], signals[0]['url'])
            if details:
                print(f"  Name: {details.name}")
                print(f"  Growth: {details.growth_pct}%")
                print(f"  Win Rate: {details.win_rate}%")
                print(f"  Drawdown: {details.max_drawdown}%")
    else:
        print("\n⚠️ No signals found - MQL5 may be blocking the scraper")
        print("This is common with sites that detect automated access")
        print("\nAlternative approaches:")
        print("  1. Use MQL5 API (requires paid account)")
        print("  2. Use Selenium/Playwright for browser automation")
        print("  3. Use the MQL5 trade copier to get live positions")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MQL5 Signal Intel Scraper")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    args = parser.parse_args()
    
    if args.test:
        test_scraper()
    elif args.daemon:
        from config import UPDATE_INTERVAL_MINUTES
        scraper = MQL5Scraper()
        
        print(f"Running in daemon mode (every {UPDATE_INTERVAL_MINUTES} minutes)")
        print("Press Ctrl+C to stop\n")
        
        while True:
            try:
                scraper.run()
                print(f"\nSleeping {UPDATE_INTERVAL_MINUTES} minutes...\n")
                time.sleep(UPDATE_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                print("\nStopping daemon...")
                break
    else:
        scraper = MQL5Scraper()
        result = scraper.run()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
