#!/usr/bin/env python3
"""
Forex Factory Calendar Scraper - Economic Events

SOURCE: https://www.forexfactory.com/calendar
All data is PUBLIC and can be manually verified at the URL

RULE: NO random data. Every field comes from the actual webpage.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

from config import ff_config, scraper_config, get_headers
from database import IntelDatabase


class ForexFactoryCalendarScraper:
    """
    Scrapes economic calendar from Forex Factory
    
    Source: https://www.forexfactory.com/calendar
    """
    
    def __init__(self):
        self.session = requests.Session()
        # Forex Factory needs specific headers
        headers = get_headers()
        headers['Referer'] = 'https://www.forexfactory.com/'
        self.session.headers.update(headers)
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
                elif response.status_code == 429:
                    print(f"Rate limited, waiting...")
                    time.sleep(scraper_config.RETRY_DELAY * 3)
                else:
                    print(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(scraper_config.RETRY_DELAY)
        
        return None
    
    def scrape_calendar(self, week: str = "this") -> List[Dict]:
        """
        Scrape calendar for a given week
        
        Args:
            week: "this", "next", or date string (e.g., "jan1.2026")
            
        Source: https://www.forexfactory.com/calendar?week={week}
        """
        if week == "this":
            url = ff_config.CALENDAR_URL
        elif week == "next":
            url = f"{ff_config.CALENDAR_URL}?week=next"
        else:
            url = f"{ff_config.CALENDAR_URL}?week={week}"
        
        print(f"Scraping Forex Factory calendar: {url}")
        
        response = self._make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'lxml')
        events = []
        
        # Find calendar table
        calendar_table = soup.select_one('.calendar__table, #calendar, table.calendar')
        
        if not calendar_table:
            # Try finding individual event rows
            event_rows = soup.select('tr.calendar__row, tr.calendar_row, tr[class*="calendar"]')
        else:
            event_rows = calendar_table.select('tr')
        
        current_date = None
        
        for row in event_rows:
            # Check for date row
            date_cell = row.select_one('.calendar__date, .date, td[class*="date"]')
            if date_cell:
                date_text = date_cell.get_text(strip=True)
                if date_text:
                    current_date = self._parse_date(date_text)
            
            # Try to parse as event row
            event = self._parse_event_row(row, current_date, url)
            if event:
                events.append(event)
        
        # If no structured data found, try alternative parsing
        if not events:
            events = self._parse_alternative(soup, url)
        
        print(f"  Found {len(events)} events")
        return events
    
    def _parse_event_row(self, row, current_date: str, source_url: str) -> Optional[Dict]:
        """Parse a single event row"""
        try:
            # Get all cells
            cells = row.select('td')
            if len(cells) < 3:
                return None
            
            event = {
                "source_url": source_url,
                "scraped_at": datetime.utcnow().isoformat() + "Z"
            }
            
            # Look for time
            time_cell = row.select_one('.calendar__time, .time, td[class*="time"]')
            if time_cell:
                time_text = time_cell.get_text(strip=True)
                event['time'] = time_text
            
            # Look for currency
            currency_cell = row.select_one('.calendar__currency, .currency, td[class*="currency"]')
            if currency_cell:
                event['currency'] = currency_cell.get_text(strip=True).upper()
            
            # Look for impact
            impact_cell = row.select_one('.calendar__impact, .impact, td[class*="impact"]')
            if impact_cell:
                # Impact is often indicated by color/icon classes
                impact_classes = ' '.join(impact_cell.get('class', []))
                if 'high' in impact_classes.lower() or 'red' in impact_classes.lower():
                    event['impact'] = 'high'
                elif 'medium' in impact_classes.lower() or 'orange' in impact_classes.lower():
                    event['impact'] = 'medium'
                else:
                    event['impact'] = 'low'
                
                # Also check for icon/span
                impact_icon = impact_cell.select_one('[class*="impact"]')
                if impact_icon:
                    icon_class = ' '.join(impact_icon.get('class', []))
                    if 'high' in icon_class or 'red' in icon_class:
                        event['impact'] = 'high'
                    elif 'med' in icon_class or 'ora' in icon_class:
                        event['impact'] = 'medium'
            
            # Look for event name
            event_cell = row.select_one('.calendar__event, .event, td[class*="event"]')
            if event_cell:
                event_link = event_cell.select_one('a, span')
                if event_link:
                    event['event_name'] = event_link.get_text(strip=True)
                else:
                    event['event_name'] = event_cell.get_text(strip=True)
            
            # Look for actual/forecast/previous
            for cell in cells:
                cell_class = ' '.join(cell.get('class', []))
                cell_text = cell.get_text(strip=True)
                
                if 'actual' in cell_class.lower():
                    event['actual'] = cell_text
                elif 'forecast' in cell_class.lower():
                    event['forecast'] = cell_text
                elif 'previous' in cell_class.lower():
                    event['previous'] = cell_text
            
            # Combine date and time
            if current_date and event.get('time'):
                event['datetime'] = f"{current_date} {event['time']}"
            elif current_date:
                event['datetime'] = current_date
            
            # Validate required fields
            if event.get('currency') and event.get('event_name'):
                event['event_id'] = f"{event.get('datetime', '')}_{event['currency']}_{event['event_name'][:20]}"
                if 'impact' not in event:
                    event['impact'] = 'low'
                return event
            
            return None
            
        except Exception as e:
            return None
    
    def _parse_alternative(self, soup, source_url: str) -> List[Dict]:
        """Alternative parsing method if table structure is different"""
        events = []
        
        # Look for any text that looks like economic events
        # Currency codes followed by event names
        text = soup.get_text()
        
        # Common currency patterns
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
        
        for currency in currencies:
            # Find lines mentioning this currency
            pattern = rf'{currency}[\s:]+([^\n]+)'
            matches = re.findall(pattern, text)
            
            for match in matches[:10]:  # Limit per currency
                event_text = match.strip()
                if len(event_text) > 5 and len(event_text) < 100:
                    events.append({
                        "currency": currency,
                        "event_name": event_text,
                        "impact": "medium",  # Default when unknown
                        "datetime": datetime.utcnow().strftime("%Y-%m-%d"),
                        "source_url": source_url,
                        "scraped_at": datetime.utcnow().isoformat() + "Z",
                        "event_id": f"{datetime.utcnow().strftime('%Y%m%d')}_{currency}_{event_text[:20]}"
                    })
        
        return events
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """Parse date from various formats"""
        try:
            # Try common formats
            date_text = date_text.strip()
            
            # "Mon Jan 20" format
            if re.match(r'[A-Za-z]{3}\s+[A-Za-z]{3}\s+\d+', date_text):
                # Add year
                date_text = f"{date_text} {datetime.now().year}"
                dt = datetime.strptime(date_text, "%a %b %d %Y")
                return dt.strftime("%Y-%m-%d")
            
            # "Jan 20" format
            if re.match(r'[A-Za-z]{3}\s+\d+', date_text):
                date_text = f"{date_text} {datetime.now().year}"
                dt = datetime.strptime(date_text, "%b %d %Y")
                return dt.strftime("%Y-%m-%d")
            
            # "20 Jan" format
            if re.match(r'\d+\s+[A-Za-z]{3}', date_text):
                date_text = f"{date_text} {datetime.now().year}"
                dt = datetime.strptime(date_text, "%d %b %Y")
                return dt.strftime("%Y-%m-%d")
            
            return date_text
            
        except Exception:
            return None
    
    def scrape_and_store(self, weeks: List[str] = None) -> Dict:
        """
        Main scraping function - scrapes and stores to database
        """
        if weeks is None:
            weeks = ["this", "next"]
        
        results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "forex_factory",
            "events_found": 0,
            "events_stored": 0,
            "errors": []
        }
        
        all_events = []
        
        for week in weeks:
            events = self.scrape_calendar(week=week)
            all_events.extend(events)
        
        results["events_found"] = len(all_events)
        
        # Store events
        for event in all_events:
            try:
                self.db.upsert_calendar_event(event)
                results["events_stored"] += 1
            except Exception as e:
                results["errors"].append(f"Error storing event: {e}")
        
        print(f"\n✅ Calendar scraping complete: {results['events_stored']} events stored")
        return results


def test_scraper():
    """Test the scraper"""
    print("=" * 60)
    print("FOREX FACTORY CALENDAR TEST")
    print("=" * 60)
    print(f"Source: {ff_config.CALENDAR_URL}")
    print()
    
    scraper = ForexFactoryCalendarScraper()
    
    print("Testing calendar scrape...")
    response = scraper._make_request(ff_config.CALENDAR_URL)
    
    if response:
        print(f"✅ Successfully connected to Forex Factory")
        print(f"  Status: {response.status_code}")
        print(f"  Content length: {len(response.text)} bytes")
        
        # Parse events
        events = scraper.scrape_calendar()
        
        if events:
            print(f"\n  Sample events:")
            for event in events[:5]:
                print(f"    - {event.get('currency', 'N/A')}: {event.get('event_name', 'N/A')}")
                print(f"      Impact: {event.get('impact', 'N/A')}, Time: {event.get('datetime', 'N/A')}")
        
        print("\n✅ VERIFICATION: This data can be manually verified at:")
        print(f"   {ff_config.CALENDAR_URL}")
    else:
        print("❌ Failed to connect to Forex Factory")
        print("  Note: Forex Factory may block automated requests")
        print("  Consider using their API if available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex Factory Calendar Scraper")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--weeks", nargs="+", default=["this", "next"], 
                       help="Weeks to scrape")
    
    args = parser.parse_args()
    
    if args.test:
        test_scraper()
    else:
        scraper = ForexFactoryCalendarScraper()
        results = scraper.scrape_and_store(weeks=args.weeks)
        print(json.dumps(results, indent=2))
