"""
Claudia IREN Intelligence API

Endpoints for IREN research and intelligence.

Author: Claudia (Research Lead)
For: Paul (Investment Partner)
"""

import sys
sys.path.insert(0, '/home/jbot/trading_ai/claudia/iren')

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claudia/iren", tags=["Claudia IREN Intelligence"])


@router.get("/")
async def get_iren_overview():
    """
    Get IREN intelligence overview.
    
    Returns company info, current signal, and key metrics.
    """
    try:
        from daily_brief import IRENDailyBrief
        
        briefer = IRENDailyBrief()
        
        price = briefer.get_price_summary()
        technicals = briefer.get_technical_analysis()
        btc = briefer.get_btc_correlation()
        
        return {
            'ticker': 'IREN',
            'company': 'IREN Limited',
            'sector': 'Bitcoin Mining / AI Data Centers',
            'price': price,
            'technicals': technicals,
            'btc_correlation': btc,
            'thesis': {
                'status': 'ACTIVE',
                'target_price': 150.00,
                'power_moat': '3-5 year competitive advantage',
                'decoupling': btc.get('coupling_status', 'MONITORING')
            }
        }
    except Exception as e:
        logger.error(f"IREN overview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brief")
async def get_daily_brief():
    """
    Get today's IREN daily intelligence brief.
    
    Comprehensive analysis including:
    - Price summary
    - Technical analysis
    - BTC correlation
    - News summary
    - Catalysts
    - Risk alerts
    - Claudia's take
    """
    try:
        from daily_brief import generate_iren_daily_brief
        return generate_iren_daily_brief()
    except Exception as e:
        logger.error(f"Daily brief generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news")
async def get_iren_news(limit: int = 20):
    """
    Get latest IREN news from all sources.
    
    Sources:
    - Yahoo Finance
    - Web search (Serper)
    - SEC filings
    - Analyst actions
    """
    try:
        from news_feed import get_iren_news
        return get_iren_news(limit=limit)
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technicals")
async def get_technicals():
    """
    Get IREN technical analysis.
    
    Includes RSI, MACD, moving averages, trend analysis.
    """
    try:
        from daily_brief import IRENDailyBrief
        briefer = IRENDailyBrief()
        return briefer.get_technical_analysis()
    except Exception as e:
        logger.error(f"Technicals failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/btc-correlation")
async def get_btc_correlation():
    """
    Get BTC-IREN correlation analysis.
    
    Paul's thesis: IREN is decoupling from BTC.
    """
    try:
        from daily_brief import IRENDailyBrief
        briefer = IRENDailyBrief()
        return briefer.get_btc_correlation()
    except Exception as e:
        logger.error(f"BTC correlation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/catalysts")
async def get_catalysts():
    """
    Get upcoming IREN catalysts and events.
    """
    try:
        from daily_brief import IRENDailyBrief
        briefer = IRENDailyBrief()
        return {
            'catalysts': briefer.get_upcoming_catalysts(),
            'note': 'Feb 5 earnings is KEY - watch AI datacenter revenue!'
        }
    except Exception as e:
        logger.error(f"Catalysts failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/earnings-scenarios")
async def get_earnings_scenarios():
    """
    Get earnings scenario analysis for Q4 2025 (Feb 5, 2026).
    
    Returns possible outcomes and trading plans.
    """
    return {
        'earnings_date': '2026-02-05',
        'days_away': 12,
        'scenarios': [
            {
                'name': 'Beat + AI Guidance Raised',
                'probability': 'MODERATE-HIGH',
                'expected_move': '+15% to +25%',
                'trading_plan': 'Add on breakout confirmation'
            },
            {
                'name': 'In-Line + Positive Guidance',
                'probability': 'MODERATE',
                'expected_move': '-5% to +5%',
                'trading_plan': 'Buy IV crush dip'
            },
            {
                'name': 'Mining Miss + AI Accelerating',
                'probability': 'LOW-MODERATE',
                'expected_move': '-10% to -15% then recovery',
                'trading_plan': 'BUY THE DIP - decoupling thesis!'
            },
            {
                'name': 'Disaster (Miss + Cut)',
                'probability': 'LOW',
                'expected_move': '-25% to -40%',
                'trading_plan': 'Wait for dust to settle'
            }
        ],
        'paul_note': 'Avoid options expiring within 7 days of earnings!'
    }


@router.get("/filings")
async def get_sec_filings():
    """
    Get recent SEC filings for IREN.
    """
    try:
        from news_feed import IRENNewsFeed
        feed = IRENNewsFeed()
        return {
            'filings': feed.get_sec_filings(),
            'insider_trades': feed.get_insider_trades()
        }
    except Exception as e:
        logger.error(f"SEC filings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/thesis")
async def get_pauls_thesis():
    """
    Get Paul's investment thesis for IREN.
    """
    return {
        'thesis_name': "IREN AI Datacenter Transformation",
        'target_price': 150.00,
        'current_price': 56.68,
        'upside_potential': '165%',
        
        'core_position': {
            'shares': 100000,
            'entry_price': 56.68,
            'position_value': 5668000,
            'target_value': 15000000
        },
        
        'key_points': [
            'AI datacenter demand >> BTC mining revenue (future)',
            'Legacy power infrastructure = 3-5 year competitive moat',
            'Power access already built - others cannot replicate quickly',
            'Texas land deal expected to complete Late 2026',
            'Decoupling from BTC - trade on fundamentals'
        ],
        
        'competitive_moat': {
            'type': 'Power Access',
            'advantage': '3-5 years',
            'reason': 'Existing grid connections, permits, and PPAs'
        },
        
        'strategy': {
            'core_shares': 'NEVER SELL',
            'options': 'CALLS ONLY (long-only)',
            'preferred_expiry': '21-35 DTE (Feb 20, Feb 27)',
            'avoid': 'Options near Feb 5 earnings'
        },
        
        'insider_connection': {
            'note': 'Paul co-owns land potentially used by IREN',
            'location': 'Tyler, TX area',
            'timeline': 'Late 2026 completion',
            'includes': 'Utility rights and grants'
        }
    }
