"""
IREN News Feed - Claudia's Intelligence Gathering

Monitors all IREN-related news from multiple sources.
Part of the Claudia IREN Intelligence System.

Author: Claudia (Research Lead)
For: Paul (Investment Partner)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Serper API for web search
SERPER_API_KEY = os.getenv('SERPER_API_KEY', '2ae3bf9c33d9a3bb98176cae8d58795ea79eebf1')


class IRENNewsFeed:
    """
    Monitor all IREN-related news from multiple sources.
    
    Sources:
    - Yahoo Finance (company news)
    - Serper/Google (web search)
    - SEC EDGAR (filings)
    - Social sentiment (future)
    """
    
    def __init__(self):
        self.ticker = 'IREN'
        self.company_name = 'IREN Limited'
        self.alt_names = ['Iris Energy', 'IREN']
        self.cache_dir = Path('/home/jbot/trading_ai/claudia/iren/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_yahoo_news(self, limit: int = 10) -> List[Dict]:
        """Get news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.ticker)
            news = ticker.news or []
            
            results = []
            for item in news[:limit]:
                results.append({
                    'source': 'yahoo_finance',
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'publisher': item.get('publisher', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'type': item.get('type', 'news'),
                    'thumbnail': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '')
                })
            
            return results
        except Exception as e:
            logger.error(f"Yahoo news fetch failed: {e}")
            return []
    
    def search_web_news(self, query: str = None, days_back: int = 7, limit: int = 10) -> List[Dict]:
        """Search for IREN news using Serper API"""
        if not query:
            query = f"IREN Limited stock news OR Iris Energy AI datacenter"
        
        try:
            url = "https://google.serper.dev/news"
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'num': limit,
                'tbs': f'qdr:d{days_back}'  # Last N days
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('news', [])[:limit]:
                results.append({
                    'source': 'web_search',
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'publisher': item.get('source', ''),
                    'published': item.get('date', ''),
                    'sentiment': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('snippet', ''))
                })
            
            return results
        except Exception as e:
            logger.error(f"Web news search failed: {e}")
            return []
    
    def get_sec_filings(self, filing_types: List[str] = None) -> List[Dict]:
        """Get recent SEC filings for IREN"""
        if not filing_types:
            filing_types = ['10-Q', '10-K', '8-K', '4']  # 4 = insider trades
        
        try:
            # SEC EDGAR API
            cik = '0001878848'  # IREN's CIK
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = {'User-Agent': 'Claudia Research claudia@crella.ai'}
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            filings = data.get('filings', {}).get('recent', {})
            results = []
            
            for i in range(min(20, len(filings.get('form', [])))):
                form_type = filings['form'][i]
                if filing_types and form_type not in filing_types:
                    continue
                    
                results.append({
                    'source': 'sec_edgar',
                    'form_type': form_type,
                    'filing_date': filings['filingDate'][i],
                    'description': filings.get('primaryDocDescription', [''])[i] if i < len(filings.get('primaryDocDescription', [])) else '',
                    'accession_number': filings['accessionNumber'][i],
                    'link': f"https://www.sec.gov/Archives/edgar/data/{cik}/{filings['accessionNumber'][i].replace('-', '')}/{filings['primaryDocument'][i]}"
                })
            
            return results[:10]
        except Exception as e:
            logger.error(f"SEC filings fetch failed: {e}")
            return []
    
    def get_insider_trades(self) -> List[Dict]:
        """Get recent insider trading activity"""
        try:
            # Get Form 4 filings (insider trades)
            filings = self.get_sec_filings(filing_types=['4'])
            
            insider_trades = []
            for filing in filings:
                insider_trades.append({
                    'source': 'sec_form4',
                    'date': filing['filing_date'],
                    'link': filing['link'],
                    'type': 'insider_trade',
                    'details': 'See filing for details'  # Would need to parse XML for full details
                })
            
            return insider_trades
        except Exception as e:
            logger.error(f"Insider trades fetch failed: {e}")
            return []
    
    def get_analyst_actions(self) -> List[Dict]:
        """Get recent analyst rating changes and price targets"""
        try:
            # Search for analyst actions
            query = f"IREN stock analyst rating price target upgrade downgrade"
            news = self.search_web_news(query=query, days_back=30, limit=5)
            
            # Filter for analyst-related news
            analyst_news = []
            keywords = ['analyst', 'rating', 'price target', 'upgrade', 'downgrade', 'buy', 'sell', 'hold']
            
            for item in news:
                title_lower = item['title'].lower()
                if any(kw in title_lower for kw in keywords):
                    analyst_news.append({
                        **item,
                        'type': 'analyst_action'
                    })
            
            return analyst_news
        except Exception as e:
            logger.error(f"Analyst actions fetch failed: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['surge', 'jump', 'rise', 'gain', 'bullish', 'upgrade', 'buy', 'growth', 
                         'strong', 'beat', 'exceed', 'record', 'high', 'expansion', 'deal', 'contract']
        negative_words = ['fall', 'drop', 'decline', 'bearish', 'downgrade', 'sell', 'weak', 
                         'miss', 'loss', 'concern', 'risk', 'warning', 'lawsuit', 'investigation']
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def get_all_news(self, limit: int = 20) -> Dict[str, Any]:
        """
        Aggregate news from all sources.
        
        Returns:
        {
            'timestamp': '2026-01-24T12:00:00Z',
            'ticker': 'IREN',
            'total_items': 15,
            'sentiment_summary': {'positive': 8, 'neutral': 5, 'negative': 2},
            'news': [...],
            'filings': [...],
            'insider_trades': [...],
            'analyst_actions': [...]
        }
        """
        yahoo_news = self.get_yahoo_news(limit=limit // 2)
        web_news = self.search_web_news(limit=limit // 2)
        filings = self.get_sec_filings()
        insider_trades = self.get_insider_trades()
        analyst_actions = self.get_analyst_actions()
        
        # Combine and dedupe news
        all_news = yahoo_news + web_news
        
        # Sentiment summary
        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
        for item in all_news:
            sentiment = item.get('sentiment', self._analyze_sentiment(item.get('title', '')))
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        result = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'ticker': self.ticker,
            'total_items': len(all_news) + len(filings),
            'sentiment_summary': sentiments,
            'news': all_news,
            'filings': filings,
            'insider_trades': insider_trades,
            'analyst_actions': analyst_actions
        }
        
        # Cache the result
        cache_file = self.cache_dir / f"news_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result


# Convenience function
def get_iren_news(limit: int = 20) -> Dict[str, Any]:
    """Get all IREN news"""
    feed = IRENNewsFeed()
    return feed.get_all_news(limit=limit)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    feed = IRENNewsFeed()
    
    print("=== Yahoo Finance News ===")
    yahoo = feed.get_yahoo_news(limit=5)
    for item in yahoo:
        print(f"  - {item['title'][:60]}...")
    
    print("\n=== Web Search News ===")
    web = feed.search_web_news(limit=5)
    for item in web:
        print(f"  - [{item['sentiment']}] {item['title'][:50]}...")
    
    print("\n=== SEC Filings ===")
    filings = feed.get_sec_filings()
    for item in filings[:5]:
        print(f"  - {item['form_type']} ({item['filing_date']})")
    
    print("\n=== All News Summary ===")
    all_news = feed.get_all_news()
    print(f"Total items: {all_news['total_items']}")
    print(f"Sentiment: {all_news['sentiment_summary']}")
