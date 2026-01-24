"""
Paper Trading API - FastAPI endpoints

All data is REAL market data from Yahoo Finance and CoinGecko
NO placeholders, NO fake data
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_trading.engine import get_engine, PaperTradingEngine
from paper_trading.daily_signals import get_daily_generator
from paper_trading.stable_signal import StableSignalManager, get_stable_signal

router = APIRouter(prefix="/paper-trading", tags=["Paper Trading"])

# Stable signal manager (locks signals for the day)
stable_signal_manager = StableSignalManager()

# Daily signal generator
daily_generator = get_daily_generator()


class OpenPositionRequest(BaseModel):
    """Request model for opening a position"""
    symbol: str
    type: str  # LONG, SHORT, BUY_CALL, BUY_PUT
    size: float
    is_option: bool = False
    strike: Optional[float] = None
    expiry: Optional[str] = None  # YYYY-MM-DD
    option_type: Optional[str] = None  # call, put
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: Optional[int] = 0
    source: Optional[str] = "MANUAL"
    notes: Optional[str] = ""


class ClosePositionRequest(BaseModel):
    """Request model for closing a position"""
    reason: str = "MANUAL"


@router.get("/positions")
async def get_positions():
    """
    Get all positions with REAL current prices
    Updates prices before returning
    """
    engine = get_engine()
    
    try:
        # Update all positions with real prices
        engine.update_positions()
    except Exception as e:
        # Log but don't fail - return stale prices if update fails
        print(f"Warning: Failed to update some positions: {e}")
    
    return {
        'positions': engine.positions,
        'closed_positions': engine.closed_positions[-20:],  # Last 20
        'stats': engine.get_stats(),
        'data_sources': ['yahoo_finance', 'coingecko'],
        'all_data_real': True
    }


@router.post("/open")
async def open_position(request: OpenPositionRequest):
    """
    Open new paper position with REAL entry price
    Price is fetched live from Yahoo Finance or CoinGecko
    """
    engine = get_engine()
    
    try:
        signal = {
            'symbol': request.symbol.upper(),
            'type': request.type.upper(),
            'size': request.size,
            'is_option': request.is_option,
            'strike': request.strike,
            'expiry': request.expiry,
            'option_type': request.option_type,
            'stop_loss': request.stop_loss,
            'take_profit': request.take_profit,
            'confidence': request.confidence,
            'source': request.source,
            'notes': request.notes
        }
        
        position = engine.open_position(signal)
        
        return {
            'success': True,
            'position': position,
            'message': f"Opened {request.type} position on {request.symbol} @ ${position['entry_price']:.2f}",
            'data_real': True
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open position: {e}")


@router.post("/close/{position_id}")
async def close_position(position_id: int, request: ClosePositionRequest = None):
    """
    Close position with REAL exit price
    """
    engine = get_engine()
    reason = request.reason if request else "MANUAL"
    
    try:
        position = engine.close_position(position_id, reason)
        
        return {
            'success': True,
            'position': position,
            'message': f"Closed position {position_id}: P&L ${position['pnl']:.2f}",
            'data_real': True
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close position: {e}")


@router.get("/price/{symbol}")
async def get_price(symbol: str):
    """
    Get REAL current price for any supported symbol
    
    Supported: IREN, GLD, SPY, BTC, ETH, SOL, and most US stocks/ETFs
    """
    engine = get_engine()
    
    try:
        price_data = engine.get_real_price(symbol.upper())
        return price_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get price: {e}")


@router.get("/option-price/{symbol}")
async def get_option_price(
    symbol: str,
    strike: float,
    expiry: str,
    option_type: str = "call"
):
    """
    Get REAL option price from Yahoo Finance
    
    Args:
        symbol: Underlying (e.g., IREN)
        strike: Strike price
        expiry: Expiration date (YYYY-MM-DD)
        option_type: 'call' or 'put'
    """
    engine = get_engine()
    
    try:
        option_data = engine.get_real_option_price(
            symbol.upper(),
            strike,
            expiry,
            option_type.lower()
        )
        return option_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get option price: {e}")


@router.get("/stats")
async def get_stats():
    """Get paper trading statistics"""
    engine = get_engine()
    
    # Update positions first
    try:
        engine.update_positions()
    except Exception:
        pass
    
    return engine.get_stats()


@router.post("/reset")
async def reset_account(initial_balance: float = 100000):
    """Reset paper trading account"""
    engine = get_engine()
    engine.reset(initial_balance)
    
    return {
        'success': True,
        'message': f'Paper trading reset with ${initial_balance:,.2f}',
        'stats': engine.get_stats()
    }


@router.get("/health")
async def health_check():
    """Check if paper trading API is working"""
    engine = get_engine()
    
    # Test price fetch
    try:
        btc_price = engine.get_real_price('BTC')
        price_working = True
    except:
        btc_price = None
        price_working = False
    
    return {
        'status': 'healthy' if price_working else 'degraded',
        'engine_loaded': True,
        'price_api_working': price_working,
        'sample_price': btc_price,
        'open_positions': len(engine.positions),
        'balance': engine.balance
    }


# ==================== DAILY SIGNALS ====================

@router.get("/daily-signals")
async def get_daily_signals():
    """
    Get today's trading signals
    Generated at 6 AM ET with REAL market data
    """
    try:
        signals = daily_generator.get_todays_signals()
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get daily signals: {e}")


@router.post("/daily-signals/generate")
async def generate_daily_signals():
    """
    Force regenerate today's signals with fresh data
    Use sparingly - signals should typically be generated once at 6 AM
    """
    try:
        signals = daily_generator.generate_daily_signals()
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate signals: {e}")


@router.get("/daily-signals/{symbol}")
async def get_daily_signal_for_symbol(symbol: str):
    """
    Get today's signal for a specific symbol (btc, iren, gld)
    """
    symbol = symbol.lower()
    if symbol not in ['btc', 'iren', 'gld']:
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol}. Use btc, iren, or gld")
    
    try:
        signals = daily_generator.get_todays_signals()
        if symbol in signals.get('signals', {}):
            return signals['signals'][symbol]
        else:
            raise HTTPException(status_code=404, detail=f"No signal found for {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signal: {e}")


# ==================== LOCKED SIGNALS (STABLE - No flip-flopping!) ====================

@router.get("/locked-signal")
async def get_locked_signal():
    """
    Get today's LOCKED signal - STABLE, won't change on refresh!
    
    Signal is generated ONCE per day at 6 AM ET and locked until next day.
    Weekend: Friday's signal locked until Monday.
    
    This prevents the annoying flip-flopping of signals.
    """
    try:
        return get_stable_signal()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get locked signal: {e}")


@router.post("/locked-signal/regenerate")
async def regenerate_locked_signal():
    """
    Force regenerate the locked signal (ADMIN USE ONLY)
    
    Use sparingly - signals should be stable for the day.
    Only use if there's a critical market event.
    """
    try:
        signal = stable_signal_manager.force_regenerate()
        return {
            'success': True,
            'message': 'Signal regenerated and locked',
            'signal': signal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate signal: {e}")


@router.get("/market-status")
async def get_market_status():
    """
    Get current market status
    
    Returns: OPEN, CLOSED, PRE-MARKET, AFTER-HOURS, or WEEKEND
    """
    return {
        'status': stable_signal_manager.get_market_status(),
        'is_weekend': stable_signal_manager.is_weekend(),
        'is_market_open': stable_signal_manager.is_market_open(),
        'is_premarket': stable_signal_manager.is_premarket(),
        'is_afterhours': stable_signal_manager.is_afterhours(),
        'signal_date': stable_signal_manager.get_signal_date(),
        'next_signal_update': stable_signal_manager.get_next_update_time()
    }
