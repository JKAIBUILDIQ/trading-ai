"""
Individual Sector Scouts
========================
5 specialized scouts for different market sectors.
"""

from .scout_base import SectorScout


class TechScout(SectorScout):
    """Scout 1: TECH TITAN - Technology sector."""
    
    def __init__(self):
        super().__init__(
            name="TECH_TITAN",
            sector="Technology",
            watchlist=[
                # Mega caps
                "AAPL", "NVDA", "MSFT", "META", "GOOGL", "AMZN",
                # Semiconductors
                "AMD", "INTC", "AVGO", "TXN", "QCOM", "MU",
                # Software
                "CRM", "ORCL", "ADBE", "NOW",
            ],
            ollama_model="llama3.1:8b",
            confidence_threshold=75.0,
            scan_interval=300,
        )


class EnergyScout(SectorScout):
    """Scout 2: ENERGY EAGLE - Energy sector."""
    
    def __init__(self):
        super().__init__(
            name="ENERGY_EAGLE",
            sector="Energy",
            watchlist=[
                # Oil majors
                "XOM", "CVX", "COP", "OXY", "EOG", "PXD",
                # Services
                "SLB", "HAL", "BKR",
                # Refiners
                "MPC", "VLO", "PSX",
                # Midstream
                "KMI", "WMB", "ET",
            ],
            ollama_model="llama3.1:8b",
            confidence_threshold=75.0,
            scan_interval=300,
        )


class MinerScout(SectorScout):
    """Scout 3: MINER HAWK - Crypto miners (LONG ONLY - your protected longs)."""
    
    def __init__(self):
        super().__init__(
            name="MINER_HAWK",
            sector="Crypto Miners",
            watchlist=[
                # Your core holdings
                "IREN", "CLSK", "CIFR",
                # Major miners
                "MARA", "RIOT", "HUT", "BITF",
                # Smaller miners
                "BTBT", "CORZ", "WULF", "HIVE",
            ],
            ollama_model="llama3.1:8b",
            confidence_threshold=75.0,
            scan_interval=300,
            long_only=True,  # Only look for LONG setups!
        )


class GrowthScout(SectorScout):
    """Scout 4: GROWTH HUNTER - High-growth stocks."""
    
    def __init__(self):
        super().__init__(
            name="GROWTH_HUNTER",
            sector="Growth Stocks",
            watchlist=[
                # Fintech/Crypto
                "COIN", "SQ", "PYPL", "HOOD",
                # Cloud/SaaS
                "SNOW", "DDOG", "NET", "ZS", "CRWD",
                # Data/AI
                "PLTR", "MDB", "TEAM",
                # E-commerce
                "SHOP", "ETSY",
                # ETFs
                "ARKK",
            ],
            ollama_model="llama3.1:8b",
            confidence_threshold=75.0,
            scan_interval=300,
        )


class DefenseScout(SectorScout):
    """Scout 5: DEFENSE FORTRESS - Defense & Aerospace."""
    
    def __init__(self):
        super().__init__(
            name="DEFENSE_FORTRESS",
            sector="Defense & Aerospace",
            watchlist=[
                # Prime contractors
                "LMT", "RTX", "GD", "NOC", "BA",
                # Shipbuilders
                "HII",
                # Tech/Systems
                "LHX", "TDG", "HWM",
                # Aviation
                "TXT", "SPR",
            ],
            ollama_model="llama3.1:8b",
            confidence_threshold=75.0,
            scan_interval=300,
        )


# Factory function to create all scouts
def create_all_scouts():
    """Create instances of all 5 sector scouts."""
    return [
        TechScout(),
        EnergyScout(),
        MinerScout(),
        GrowthScout(),
        DefenseScout(),
    ]
