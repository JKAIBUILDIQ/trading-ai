"""
Sentiment Data Verification API
===============================
Provides endpoints to verify all sentiment data is REAL.

GET /intel/verify/cot      → Last 10 COT fetches with raw files
GET /intel/sentiment/gold  → Current Gold positioning
GET /intel/raw/{filename}  → Download raw data file

NO RANDOM DATA. EVERYTHING IS AUDITABLE.
"""

from flask import Flask, jsonify, send_file
import os
import json
from datetime import datetime
import glob

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COT_LOG_DIR = os.path.join(BASE_DIR, "..", "logs", "cot")
SENTIMENT_DIR = BASE_DIR


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Sentiment Verification API",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/intel/verify/cot')
def verify_cot():
    """
    Returns last 10 COT fetches with timestamps and raw data links.
    User can manually verify against CFTC website.
    """
    # Find all raw COT files
    raw_files = sorted(
        glob.glob(os.path.join(COT_LOG_DIR, "cot_raw_*.txt")),
        reverse=True
    )[:10]
    
    # Find all parsed COT files
    parsed_files = sorted(
        glob.glob(os.path.join(COT_LOG_DIR, "cot_parsed_*.json")),
        reverse=True
    )[:10]
    
    fetches = []
    for raw_file in raw_files:
        filename = os.path.basename(raw_file)
        # Extract timestamp from filename: cot_raw_YYYY-MM-DD_HHMM.txt
        try:
            timestamp_str = filename.replace("cot_raw_", "").replace(".txt", "")
            fetch_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M")
        except:
            fetch_time = None
        
        # Get file size
        file_size = os.path.getsize(raw_file)
        
        # Check if corresponding parsed file exists
        parsed_filename = filename.replace("cot_raw_", "cot_parsed_").replace(".txt", ".json")
        parsed_path = os.path.join(COT_LOG_DIR, parsed_filename)
        
        gold_data = None
        if os.path.exists(parsed_path):
            try:
                with open(parsed_path) as f:
                    parsed = json.load(f)
                    gold_data = parsed.get("data", {}).get("XAUUSD", {})
            except:
                pass
        
        fetches.append({
            "fetch_time": fetch_time.isoformat() if fetch_time else "unknown",
            "raw_file": filename,
            "raw_file_size": file_size,
            "raw_file_url": f"/intel/raw/cot/{filename}",
            "parsed_file": parsed_filename if os.path.exists(parsed_path) else None,
            "gold_sentiment": gold_data.get("sentiment") if gold_data else None,
            "gold_net_position": gold_data.get("net_positioning") if gold_data else None,
            "gold_long_pct": gold_data.get("speculators_long_pct") if gold_data else None
        })
    
    return jsonify({
        "verification_endpoint": "/intel/verify/cot",
        "purpose": "Audit trail for COT data fetches",
        "source": "CFTC Commitments of Traders",
        "source_url": "https://www.cftc.gov/dea/newcot/deafut.txt",
        "last_10_fetches": fetches,
        "how_to_verify": [
            "1. Download raw file from /intel/raw/cot/{filename}",
            "2. Compare to official CFTC data at source_url",
            "3. Verify Gold row matches our parsed data",
            "4. If ANY mismatch → report as broken"
        ],
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/intel/sentiment/gold')
def get_gold_sentiment():
    """
    Returns current Gold (XAUUSD) positioning data.
    Includes full audit trail.
    """
    # Load latest parsed data
    latest_file = os.path.join(COT_LOG_DIR, "cot_latest.json")
    
    if not os.path.exists(latest_file):
        return jsonify({
            "status": "no_data",
            "message": "No COT data available. Run the scraper first.",
            "scraper_command": "python3 ~/trading_ai/intel/sentiment/cot_scraper.py"
        }), 404
    
    with open(latest_file) as f:
        data = json.load(f)
    
    gold = data.get("data", {}).get("XAUUSD", {})
    
    if not gold:
        return jsonify({
            "status": "no_gold_data",
            "message": "COT data exists but no Gold data found"
        }), 404
    
    return jsonify({
        "symbol": "XAUUSD",
        "source": "CFTC Commitments of Traders",
        "report_date": gold.get("report_date"),
        "fetched_at": data.get("fetched_at"),
        
        "positioning": {
            "speculators_net": gold.get("net_positioning"),
            "speculators_long_pct": gold.get("speculators_long_pct"),
            "speculators_short_pct": gold.get("speculators_short_pct"),
            "hedgers_net": gold.get("hedgers_net"),
            "open_interest": gold.get("open_interest"),
            "sentiment": gold.get("sentiment"),
            "extreme_reading": gold.get("extreme_reading")
        },
        
        "raw_data": {
            "non_commercial_long": gold.get("non_commercial_long"),
            "non_commercial_short": gold.get("non_commercial_short"),
            "commercial_long": gold.get("commercial_long"),
            "commercial_short": gold.get("commercial_short")
        },
        
        "verification": {
            "source_url": gold.get("source_url"),
            "raw_file": gold.get("raw_file"),
            "cftc_code": gold.get("cftc_code"),
            "how_to_verify": "Download raw file and compare to CFTC website"
        }
    })


@app.route('/intel/raw/cot/<filename>')
def get_raw_cot_file(filename):
    """
    Download raw COT data file for verification.
    """
    # Security: only allow specific file patterns
    if not filename.startswith("cot_raw_") or not filename.endswith(".txt"):
        return jsonify({"error": "Invalid filename"}), 400
    
    file_path = os.path.join(COT_LOG_DIR, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(file_path, mimetype='text/plain')


@app.route('/intel/all')
def get_all_sentiment():
    """Get all sentiment data (Gold, Silver, EUR, GBP, JPY)."""
    latest_file = os.path.join(COT_LOG_DIR, "cot_latest.json")
    
    if not os.path.exists(latest_file):
        return jsonify({"status": "no_data"}), 404
    
    with open(latest_file) as f:
        data = json.load(f)
    
    return jsonify({
        "source": "CFTC Commitments of Traders",
        "fetched_at": data.get("fetched_at"),
        "source_url": data.get("source_url"),
        "data": data.get("data", {}),
        "verification": {
            "raw_file": data.get("raw_file"),
            "verify_endpoint": "/intel/verify/cot"
        }
    })


if __name__ == "__main__":
    print("═" * 60)
    print("SENTIMENT VERIFICATION API")
    print("═" * 60)
    print("Endpoints:")
    print("  GET /intel/verify/cot      → Audit trail for COT data")
    print("  GET /intel/sentiment/gold  → Gold positioning")
    print("  GET /intel/raw/cot/{file}  → Download raw data")
    print("  GET /intel/all             → All sentiment data")
    print("═" * 60)
    
    app.run(host='0.0.0.0', port=8095, debug=False)
