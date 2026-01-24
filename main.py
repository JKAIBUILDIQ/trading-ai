"""
Trading AI API Server

Paper Trading with 100% REAL market data
NO placeholders, NO fake data
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.paper_trading_api import router as paper_trading_router
from api.iren_api import router as iren_router
from paper_trading.neo_integration import add_neo_routes

app = FastAPI(
    title="Trading AI API",
    description="Paper Trading System with Real Market Data - Yahoo Finance & CoinGecko",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(paper_trading_router)
app.include_router(iren_router)

# Add NEO integration routes
add_neo_routes(paper_trading_router)


@app.get("/")
async def root():
    return {
        "service": "Trading AI API",
        "version": "1.0.0",
        "endpoints": [
            "/paper-trading/positions",
            "/paper-trading/open",
            "/paper-trading/close/{id}",
            "/paper-trading/price/{symbol}",
            "/paper-trading/stats",
            "/paper-trading/health",
            "/iren/rsi-history",
            "/iren/correlation-analysis"
        ],
        "data_sources": ["yahoo_finance", "coingecko"],
        "all_data_real": True
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "trading-ai"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8500,
        reload=True
    )
