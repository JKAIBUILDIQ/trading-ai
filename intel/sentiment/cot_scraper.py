"""
CFTC Commitments of Traders (COT) Scraper
=========================================
OFFICIAL government data from the CFTC - the most reliable positioning data.

Source: https://www.cftc.gov/dea/newcot/deafut.txt
Updated: Every Friday (covers Tuesday positions)

NO RANDOM DATA. NO SIMULATED DATA. REAL OR NOTHING.
"""

import requests
import json
import os
import re
from datetime import datetime
import logging

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "cot")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [COT-Scraper] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "cot_scraper.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("COT-Scraper")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CFTC_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "cot")

# Commodities we care about
TRACKED_COMMODITIES = {
    "GOLD - COMMODITY EXCHANGE INC.": {
        "symbol": "XAUUSD",
        "cftc_code": "088691",
        "contract_unit": "100 troy ounces"
    },
    "SILVER - COMMODITY EXCHANGE INC.": {
        "symbol": "XAGUSD",
        "cftc_code": "084691",
        "contract_unit": "5000 troy ounces"
    },
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": {
        "symbol": "EURUSD",
        "cftc_code": "099741",
        "contract_unit": "125000 EUR"
    },
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": {
        "symbol": "USDJPY",
        "cftc_code": "097741",
        "contract_unit": "12500000 JPY"
    },
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": {
        "symbol": "GBPUSD",
        "cftc_code": "096742",
        "contract_unit": "62500 GBP"
    }
}


class COTScraper:
    """
    Scrapes CFTC Commitments of Traders data.
    All data is logged raw for verification.
    """
    
    def __init__(self):
        self.raw_data = None
        self.parsed_data = {}
        self.last_fetch_time = None
        self.raw_file_path = None
    
    def fetch(self) -> bool:
        """
        Fetch raw COT data from CFTC.
        Saves raw response to file for audit.
        """
        logger.info("═" * 60)
        logger.info("FETCHING CFTC COT DATA")
        logger.info(f"Source: {CFTC_URL}")
        logger.info("═" * 60)
        
        try:
            response = requests.get(CFTC_URL, timeout=30)
            response.raise_for_status()
            
            self.raw_data = response.text
            self.last_fetch_time = datetime.utcnow()
            
            # Save raw data for verification
            timestamp = self.last_fetch_time.strftime("%Y-%m-%d_%H%M")
            self.raw_file_path = os.path.join(OUTPUT_DIR, f"cot_raw_{timestamp}.txt")
            
            with open(self.raw_file_path, 'w') as f:
                f.write(f"# CFTC COT Data - Fetched at {self.last_fetch_time.isoformat()}\n")
                f.write(f"# Source: {CFTC_URL}\n")
                f.write(f"# Lines: {len(self.raw_data.splitlines())}\n")
                f.write("#" + "=" * 70 + "\n")
                f.write(self.raw_data)
            
            logger.info(f"✅ Raw data saved: {self.raw_file_path}")
            logger.info(f"   Lines: {len(self.raw_data.splitlines())}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to fetch COT data: {e}")
            return False
    
    def parse(self) -> dict:
        """
        Parse raw COT data for tracked commodities.
        
        COT Data Format (key columns):
        - Col 0: Market Name
        - Col 1: Report Date (YYMMDD)
        - Col 2: As Of Date (YYYY-MM-DD)
        - Col 7: Open Interest
        - Col 8: Non-Commercial Long
        - Col 9: Non-Commercial Short
        - Col 10: Commercial Long
        - Col 11: Commercial Short
        """
        if not self.raw_data:
            logger.error("No raw data to parse")
            return {}
        
        logger.info("Parsing COT data...")
        
        for line in self.raw_data.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            
            # Check if this line contains a tracked commodity
            for commodity_name, config in TRACKED_COMMODITIES.items():
                if commodity_name in line:
                    parsed = self._parse_line(line, commodity_name, config)
                    if parsed:
                        self.parsed_data[config["symbol"]] = parsed
                        logger.info(f"   ✅ {config['symbol']}: {parsed['net_positioning']}")
        
        return self.parsed_data
    
    def _parse_line(self, line: str, commodity_name: str, config: dict) -> dict:
        """Parse a single COT data line."""
        try:
            # Remove quotes and split by comma
            # The format is CSV with some fields quoted
            parts = []
            current = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current.strip().strip('"'))
                    current = ""
                else:
                    current += char
            parts.append(current.strip().strip('"'))
            
            # Extract key data
            report_date_raw = parts[1]  # Format: YYMMDD or YYYY-MM-DD
            as_of_date = parts[2] if len(parts) > 2 else ""
            
            # Parse date
            if "-" in as_of_date:
                report_date = datetime.strptime(as_of_date, "%Y-%m-%d")
            else:
                # YYMMDD format
                year = 2000 + int(report_date_raw[:2])
                month = int(report_date_raw[2:4])
                day = int(report_date_raw[4:6])
                report_date = datetime(year, month, day)
            
            # Key positioning data
            open_interest = int(parts[7].replace(",", "").strip()) if parts[7].strip() else 0
            
            # Non-Commercial (Speculative)
            nc_long = int(parts[8].replace(",", "").strip()) if parts[8].strip() else 0
            nc_short = int(parts[9].replace(",", "").strip()) if parts[9].strip() else 0
            
            # Commercial (Hedgers)
            comm_long = int(parts[10].replace(",", "").strip()) if parts[10].strip() else 0
            comm_short = int(parts[11].replace(",", "").strip()) if parts[11].strip() else 0
            
            # Calculate net positioning
            nc_net = nc_long - nc_short  # Speculator net position
            comm_net = comm_long - comm_short  # Hedger net position
            
            # Calculate percentages
            total_nc = nc_long + nc_short
            if total_nc > 0:
                nc_long_pct = round((nc_long / total_nc) * 100, 1)
                nc_short_pct = round((nc_short / total_nc) * 100, 1)
            else:
                nc_long_pct = nc_short_pct = 50.0
            
            return {
                "symbol": config["symbol"],
                "cftc_code": config["cftc_code"],
                "commodity_name": commodity_name,
                "report_date": report_date.strftime("%Y-%m-%d"),
                "fetched_at": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
                
                # Raw numbers
                "open_interest": open_interest,
                "non_commercial_long": nc_long,
                "non_commercial_short": nc_short,
                "commercial_long": comm_long,
                "commercial_short": comm_short,
                
                # Calculated
                "net_positioning": nc_net,
                "net_positioning_pct": nc_long_pct,  # % of speculators that are long
                "speculators_long_pct": nc_long_pct,
                "speculators_short_pct": nc_short_pct,
                "hedgers_net": comm_net,
                
                # Sentiment interpretation
                "sentiment": "BULLISH" if nc_net > 0 else "BEARISH" if nc_net < 0 else "NEUTRAL",
                "extreme_reading": nc_long_pct > 80 or nc_short_pct > 80,
                
                # Audit trail
                "source": "CFTC",
                "source_url": CFTC_URL,
                "raw_file": self.raw_file_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing {commodity_name}: {e}")
            return None
    
    def save_parsed(self) -> str:
        """Save parsed data to JSON file."""
        if not self.parsed_data:
            logger.warning("No parsed data to save")
            return None
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M")
        output_file = os.path.join(OUTPUT_DIR, f"cot_parsed_{timestamp}.json")
        
        output = {
            "fetched_at": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            "source": "CFTC Commitments of Traders",
            "source_url": CFTC_URL,
            "raw_file": self.raw_file_path,
            "data": self.parsed_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"✅ Parsed data saved: {output_file}")
        
        # Also save as latest.json for easy access
        latest_file = os.path.join(OUTPUT_DIR, "cot_latest.json")
        with open(latest_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output_file
    
    def get_gold_sentiment(self) -> dict:
        """Get Gold (XAUUSD) sentiment specifically."""
        return self.parsed_data.get("XAUUSD", {})
    
    def format_for_neo(self) -> str:
        """Format COT data for NEO-GOLD prompt."""
        gold = self.get_gold_sentiment()
        if not gold:
            return "📊 COT DATA: No Gold positioning data available."
        
        lines = [
            "═" * 50,
            "📊 CFTC COT DATA (Official Government Data)",
            f"Report Date: {gold.get('report_date', 'N/A')}",
            "═" * 50,
            "",
            f"GOLD FUTURES POSITIONING:",
            f"   Open Interest: {gold.get('open_interest', 0):,} contracts",
            "",
            f"   SPECULATORS (Non-Commercial):",
            f"      Long: {gold.get('non_commercial_long', 0):,}",
            f"      Short: {gold.get('non_commercial_short', 0):,}",
            f"      Net: {gold.get('net_positioning', 0):+,} ({gold.get('sentiment', 'N/A')})",
            f"      Long %: {gold.get('speculators_long_pct', 50):.1f}%",
            "",
            f"   COMMERCIALS (Hedgers):",
            f"      Long: {gold.get('commercial_long', 0):,}",
            f"      Short: {gold.get('commercial_short', 0):,}",
            f"      Net: {gold.get('hedgers_net', 0):+,}",
            "",
            f"   ⚠️ Extreme Reading: {'YES' if gold.get('extreme_reading') else 'NO'}",
            "",
            "═" * 50,
            f"Source: {gold.get('source_url', CFTC_URL)}",
            f"Verified: {gold.get('raw_file', 'N/A')}",
            "═" * 50
        ]
        
        return "\n".join(lines)


def run_cot_scraper():
    """Main function to run the COT scraper."""
    logger.info("═" * 60)
    logger.info("CFTC COT SCRAPER - STARTING")
    logger.info("═" * 60)
    
    scraper = COTScraper()
    
    if scraper.fetch():
        scraper.parse()
        scraper.save_parsed()
        
        # Print summary
        gold = scraper.get_gold_sentiment()
        if gold:
            print("\n" + "═" * 60)
            print("🥇 GOLD POSITIONING (XAUUSD)")
            print("═" * 60)
            print(f"   Report Date: {gold['report_date']}")
            print(f"   Speculators Net: {gold['net_positioning']:+,} contracts")
            print(f"   Sentiment: {gold['sentiment']}")
            print(f"   Long %: {gold['speculators_long_pct']:.1f}%")
            print(f"   Extreme Reading: {'⚠️ YES' if gold['extreme_reading'] else 'No'}")
            print("═" * 60)
            print(f"   Raw file: {gold['raw_file']}")
            print("═" * 60)
        
        return scraper.parsed_data
    else:
        logger.error("Failed to fetch COT data")
        return None


if __name__ == "__main__":
    run_cot_scraper()
