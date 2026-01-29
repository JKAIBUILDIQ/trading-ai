"""
Paper Trading API - Main entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from paper_trading_api import router

app = FastAPI(
    title="Paper Trading API",
    description="Paper trading with 100% real market data",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Paper Trading API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
