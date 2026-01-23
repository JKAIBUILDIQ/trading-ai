#!/usr/bin/env python3
"""
Myfxbook Scraper - Verified Trader Data

SOURCE: https://www.myfxbook.com/members
All data is PUBLIC and can be manually verified at the URLs

RULE: NO random data. Every field comes from the actual webpage.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse

from config import myfxbook_config, scraper_config, get_headers
from database import IntelDatabase


class MyfxbookScraper:
    """
    Scrapes verified trader data from Myfxbook.com
    
    Source URLs:
    - Members list: https://www.myfxbook.com/members
    - Individual: https://www.myfxbook.com/members/{username}
    - Trades: https://www.myfxbook.com/members/{username}/trades
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(get_headers())
        self.db = IntelDatabase()
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Respect rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < scraper_config.MIN_REQUEST_INTERVAL:
            time.sleep(scraper_config.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, retries: int = None) -> Optional[requests.Response]:
        """Make a rate-limited request with retries"""
        retries = retries or scraper_config.RETRY_COUNT
        
        for attempt in range(retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=scraper_config.REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    print(f"Rate limited, waiting {scraper_config.RETRY_DELAY * 2}s...")
                    time.sleep(scraper_config.RETRY_DELAY * 2)
                else:
                    print(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(scraper_config.RETRY_DELAY)
        
        return None
    
    def scrape_members_list(self, pages: int = 5) -> List[Dict]:
        """
        Scrape the members list to find top traders
        
        Source: https://www.myfxbook.com/members
        
        Returns list of trader basic info
        """
        print(f"Scraping Myfxbook members list ({pages} pages)...")
        traders = []
        
        for page in range(1, pages + 1):
            url = f"{myfxbook_config.MEMBERS_URL}?page={page}"
            print(f"  Page {page}: {url}")
            
            response = self._make_request(url)
            if not response:
                continue
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find member rows (adjust selectors based on actual page structure)
            member_rows = soup.select('table.table-hover tbody tr')
            
            if not member_rows:
                # Try alternative selectors
                member_rows = soup.select('.member-row, .portfolio-row')
            
            for row in member_rows:
                try:
                    trader = self._parse_member_row(row)
                    if trader:
                        traders.append(trader)
                except Exception as e:
                    print(f"    Error parsing row: {e}")
                    continue
            
            print(f"    Found {len(member_rows)} members")
        
        print(f"Total members found: {len(traders)}")
        return traders
    
    def _parse_member_row(self, row) -> Optional[Dict]:
        """Parse a single member row from the list"""
        try:
            # Extract name and URL
            name_link = row.select_one('a[href*="/members/"]')
            if not name_link:
                return None
            
            username = name_link.get('href', '').split('/members/')[-1].split('/')[0]
            name = name_link.get_text(strip=True)
            
            # Extract stats from cells
            cells = row.select('td')
            
            trader = {
                "trader_id": f"myfxbook_{username}",
                "name": name,
                "username": username,
                "source": "myfxbook",
                "source_url": f"{myfxbook_config.BASE_URL}/members/{username}",
                "scraped_at": datetime.utcnow().isoformat() + "Z"
            }
            
            # Parse numeric values from cells (structure varies)
            for cell in cells:
                text = cell.get_text(strip=True)
                
                # Look for percentage values
                if '%' in text:
                    value = self._parse_percentage(text)
                    if 'gain' in cell.get('class', []) or 'Gain' in text:
                        trader['gain_pct'] = value
                    elif 'dd' in str(cell.get('class', [])).lower() or 'Drawdown' in text:
                        trader['drawdown_pct'] = abs(value) if value else None
                
                # Look for verified badge
                if cell.select_one('.verified, .badge-verified, [title*="Verified"]'):
                    trader['verified'] = True
            
            return trader
            
        except Exception as e:
            return None
    
    def scrape_trader_profile(self, username: str) -> Optional[Dict]:
        """
        Scrape detailed data for a single trader
        
        Source: https://www.myfxbook.com/members/{username}
        """
        url = f"{myfxbook_config.BASE_URL}/members/{username}"
        print(f"Scraping trader profile: {url}")
        
        response = self._make_request(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        trader = {
            "trader_id": f"myfxbook_{username}",
            "name": username,
            "username": username,
            "source": "myfxbook",
            "source_url": url,
            "scraped_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Parse statistics from the page
        # Look for stat boxes/cards
        stat_boxes = soup.select('.stat-box, .statistics-item, [class*="stat"]')
        
        for box in stat_boxes:
            label = box.select_one('.label, .stat-label, small')
            value = box.select_one('.value, .stat-value, strong')
            
            if label and value:
                label_text = label.get_text(strip=True).lower()
                value_text = value.get_text(strip=True)
                
                if 'gain' in label_text:
                    trader['gain_pct'] = self._parse_percentage(value_text)
                elif 'drawdown' in label_text:
                    trader['drawdown_pct'] = abs(self._parse_percentage(value_text) or 0)
                elif 'win' in label_text and 'rate' in label_text:
                    trader['win_rate'] = self._parse_percentage(value_text)
                elif 'trades' in label_text:
                    trader['total_trades'] = self._parse_number(value_text)
        
        # Check for verified badge
        if soup.select_one('.verified-badge, .account-verified, [title*="Verified"]'):
            trader['verified'] = True
        else:
            trader['verified'] = False
        
        # Extract profile data from any visible tables
        tables = soup.select('table')
        for table in tables:
            for row in table.select('tr'):
                cells = row.select('td, th')
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower()
                    val = cells[1].get_text(strip=True)
                    
                    if 'gain' in key:
                        trader['gain_pct'] = self._parse_percentage(val)
                    elif 'drawdown' in key:
                        trader['drawdown_pct'] = abs(self._parse_percentage(val) or 0)
                    elif 'win' in key:
                        trader['win_rate'] = self._parse_percentage(val)
                    elif 'trades' in key and 'total' not in trader:
                        trader['total_trades'] = self._parse_number(val)
        
        return trader
    
    def scrape_trader_trades(self, username: str, limit: int = 20) -> List[Dict]:
        """
        Scrape recent trades for a trader
        
        Source: https://www.myfxbook.com/members/{username}/trades
        """
        url = f"{myfxbook_config.BASE_URL}/members/{username}/trades"
        print(f"Scraping trades: {url}")
        
        response = self._make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'lxml')
        trades = []
        
        # Find trade tables
        trade_tables = soup.select('table.trade-table, table.trades, table')
        
        for table in trade_tables:
            rows = table.select('tbody tr')
            
            for row in rows[:limit]:
                try:
                    trade = self._parse_trade_row(row, username)
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    continue
        
        print(f"  Found {len(trades)} trades")
        return trades
    
    def _parse_trade_row(self, row, username: str) -> Optional[Dict]:
        """Parse a single trade row"""
        cells = row.select('td')
        if len(cells) < 4:
            return None
        
        trade = {
            "trader_id": f"myfxbook_{username}",
            "source_url": f"{myfxbook_config.BASE_URL}/members/{username}/trades",
            "scraped_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Parse cells - structure varies by page layout
        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)
            
            # Look for symbol (usually 6 chars like EURUSD)
            if re.match(r'^[A-Z]{6}$', text):
                trade['symbol'] = text
            
            # Look for direction
            if text.upper() in ['BUY', 'SELL', 'LONG', 'SHORT']:
                trade['direction'] = 'BUY' if text.upper() in ['BUY', 'LONG'] else 'SELL'
            
            # Look for lots
            if re.match(r'^\d+\.?\d*$', text) and 'lots' not in trade:
                val = float(text)
                if 0 < val < 100:  # Reasonable lot size
                    trade['lots'] = val
            
            # Look for P&L
            if '$' in text or text.startswith('-') or text.startswith('+'):
                trade['pnl'] = self._parse_money(text)
            
            # Look for dates
            if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', text):
                if 'open_time' not in trade:
                    trade['open_time'] = text
                else:
                    trade['close_time'] = text
        
        # Validate required fields
        if 'symbol' in trade and 'direction' in trade:
            return trade
        
        return None
    
    def _parse_percentage(self, text: str) -> Optional[float]:
        """Parse percentage value from text"""
        if not text:
            return None
        # Remove % and parse
        match = re.search(r'[-+]?\d+\.?\d*', text.replace(',', ''))
        if match:
            return float(match.group())
        return None
    
    def _parse_number(self, text: str) -> Optional[int]:
        """Parse integer from text"""
        if not text:
            return None
        match = re.search(r'\d+', text.replace(',', ''))
        if match:
            return int(match.group())
        return None
    
    def _parse_money(self, text: str) -> Optional[float]:
        """Parse money value from text"""
        if not text:
            return None
        # Remove $ and commas
        cleaned = re.sub(r'[$,]', '', text)
        match = re.search(r'[-+]?\d+\.?\d*', cleaned)
        if match:
            return float(match.group())
        return None
    
    def scrape_and_store(self, pages: int = 3, detailed: bool = True) -> Dict:
        """
        Main scraping function - scrapes and stores to database
        
        Args:
            pages: Number of member list pages to scrape
            detailed: Whether to scrape individual profiles
        
        Returns:
            Summary of results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "myfxbook",
            "traders_found": 0,
            "traders_stored": 0,
            "trades_stored": 0,
            "errors": []
        }
        
        # Scrape members list
        traders = self.scrape_members_list(pages=pages)
        results["traders_found"] = len(traders)
        
        # Filter by criteria
        filtered = []
        for t in traders:
            # Apply filters from config
            if t.get('gain_pct') and t['gain_pct'] < myfxbook_config.MIN_GAIN_PCT:
                continue
            if t.get('drawdown_pct') and t['drawdown_pct'] > myfxbook_config.MAX_DRAWDOWN_PCT:
                continue
            if myfxbook_config.REQUIRE_VERIFIED and not t.get('verified'):
                continue
            filtered.append(t)
        
        print(f"\nFiltered to {len(filtered)} traders meeting criteria")
        
        # Scrape detailed profiles and trades
        for trader in filtered[:20]:  # Limit to avoid overloading
            try:
                if detailed:
                    # Get detailed profile
                    profile = self.scrape_trader_profile(trader['username'])
                    if profile:
                        trader.update(profile)
                    
                    # Get recent trades
                    trades = self.scrape_trader_trades(trader['username'])
                    for trade in trades:
                        try:
                            self.db.insert_trade(trade)
                            results["trades_stored"] += 1
                        except Exception as e:
                            results["errors"].append(f"Trade insert error: {e}")
                
                # Store trader
                self.db.upsert_trader(trader)
                results["traders_stored"] += 1
                
            except Exception as e:
                results["errors"].append(f"Error with {trader.get('username')}: {e}")
        
        print(f"\n✅ Scraping complete: {results['traders_stored']} traders, {results['trades_stored']} trades")
        return results


def test_scraper():
    """Test the scraper with a simple request"""
    print("=" * 60)
    print("MYFXBOOK SCRAPER TEST")
    print("=" * 60)
    print(f"Source: {myfxbook_config.MEMBERS_URL}")
    print()
    
    scraper = MyfxbookScraper()
    
    # Test single page
    print("Testing members list scrape...")
    response = scraper._make_request(myfxbook_config.MEMBERS_URL)
    
    if response:
        print(f"✅ Successfully connected to Myfxbook")
        print(f"  Status: {response.status_code}")
        print(f"  Content length: {len(response.text)} bytes")
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find any links to member profiles
        member_links = soup.select('a[href*="/members/"]')
        print(f"  Found {len(member_links)} member links")
        
        # Show sample
        if member_links:
            print("\n  Sample member links:")
            for link in member_links[:5]:
                href = link.get('href', '')
                name = link.get_text(strip=True)
                if name and '/members/' in href:
                    print(f"    - {name}: {myfxbook_config.BASE_URL}{href}")
        
        print("\n✅ VERIFICATION: This data can be manually verified at:")
        print(f"   {myfxbook_config.MEMBERS_URL}")
    else:
        print("❌ Failed to connect to Myfxbook")
        print("  This could be due to rate limiting or network issues")
        print("  The scraper will retry automatically in production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Myfxbook Scraper")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--pages", type=int, default=3, help="Pages to scrape")
    parser.add_argument("--no-detailed", action="store_true", help="Skip detailed profiles")
    
    args = parser.parse_args()
    
    if args.test:
        test_scraper()
    else:
        scraper = MyfxbookScraper()
        results = scraper.scrape_and_store(
            pages=args.pages,
            detailed=not args.no_detailed
        )
        print(json.dumps(results, indent=2))
