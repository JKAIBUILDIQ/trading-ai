"""
War Room API - Command & Control Center for Trading
Now with REAL bot command execution!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import httpx
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Bot API endpoints
GHOST_API = "http://localhost:8036"
PORTFOLIO_API = "http://localhost:8030"
IBKR_GHOST_API = "http://localhost:8035"  # IBKR Ghost sync

# Signal file for bots to monitor
SIGNAL_DIR = Path("/home/jbot/trading_ai/neo/signals")
SIGNAL_DIR.mkdir(exist_ok=True)

COMMAND_FILE = SIGNAL_DIR / "war_room_commands.json"
DEFCON_FILE = SIGNAL_DIR / "defcon_status.json"

# Store analysis history in memory
analysis_history = []

# Current DEFCON level and state
current_defcon = 3
bot_pause_state = {"ghost": False, "casper": False, "all": False}

# ═══════════════════════════════════════════════════════════════════════════════
# DEFCON RULES
# ═══════════════════════════════════════════════════════════════════════════════

DEFCON_RULES = {
    5: {"name": "NORMAL", "position_size_mult": 1.0, "allow_new_entries": True, "hedge_required": False},
    4: {"name": "ELEVATED", "position_size_mult": 0.75, "allow_new_entries": True, "hedge_required": False},
    3: {"name": "ALERT", "position_size_mult": 0.5, "allow_new_entries": True, "hedge_required": False},
    2: {"name": "SEVERE", "position_size_mult": 0.25, "allow_new_entries": False, "hedge_required": True},
    1: {"name": "CRITICAL", "position_size_mult": 0.0, "allow_new_entries": False, "hedge_required": True},
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_defcon_state():
    """Save DEFCON state to file for bots to read"""
    state = {
        "defcon": current_defcon,
        "rules": DEFCON_RULES[current_defcon],
        "pause_state": bot_pause_state,
        "updated_at": datetime.now().isoformat()
    }
    with open(DEFCON_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    return state


def write_command_signal(command: str, bot: str, params: dict):
    """Write command to signal file for bots to pick up"""
    signal = {
        "command": command,
        "target_bot": bot,
        "params": params,
        "defcon": current_defcon,
        "issued_at": datetime.now().isoformat(),
        "executed": False
    }
    
    # Append to command history
    try:
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE, 'r') as f:
                commands = json.load(f)
        else:
            commands = []
    except:
        commands = []
    
    commands.append(signal)
    # Keep last 100 commands
    commands = commands[-100:]
    
    with open(COMMAND_FILE, 'w') as f:
        json.dump(commands, f, indent=2)
    
    return signal


async def call_bot_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Call a bot API endpoint"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method == "GET":
                response = await client.get(endpoint)
            else:
                response = await client.post(endpoint, json=data)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def execute_command_sync(command: str, bot: str, params: dict) -> dict:
    """Execute a command synchronously (no async needed for Flask)"""
    import requests
    
    result = {"command": command, "bot": bot, "status": "pending"}
    
    try:
        if command == "DEFCON":
            # Just update DEFCON level
            level = params.get("level", 3)
            global current_defcon
            current_defcon = level
            save_defcon_state()
            result["status"] = "success"
            result["message"] = f"DEFCON set to {level}"
            
        elif command in ["BUY_DIP", "SCALE_IN"]:
            # Signal for Ghost to scale in
            signal = write_command_signal(command, "ghost", {
                "action": "SCALE_IN",
                "defcon": current_defcon,
                "max_size": DEFCON_RULES[current_defcon]["position_size_mult"]
            })
            
            # Try to call Ghost API
            try:
                resp = requests.post(f"{GHOST_API}/api/neo/signal", json={
                    "action": "BUY",
                    "symbol": params.get("symbol", "XAUUSD"),
                    "source": "WAR_ROOM",
                    "defcon": current_defcon
                }, timeout=10)
                if resp.status_code == 200:
                    result["status"] = "sent"
                    result["api_response"] = resp.json()
                else:
                    result["status"] = "signal_written"
            except:
                result["status"] = "signal_written"
            
            result["message"] = f"Scale-in signal issued at DEFCON {current_defcon}"
            
        elif command in ["CLOSE_PARTIAL", "CLOSE_50"]:
            # Close 50% of positions
            signal = write_command_signal(command, bot, {
                "action": "CLOSE_PARTIAL",
                "percentage": 50
            })
            
            try:
                resp = requests.post(f"{PORTFOLIO_API}/api/portfolio/close-partial", json={
                    "percentage": 50,
                    "source": "WAR_ROOM"
                }, timeout=10)
                result["status"] = "sent" if resp.status_code == 200 else "signal_written"
            except:
                result["status"] = "signal_written"
            
            result["message"] = "Close 50% signal issued"
            
        elif command == "CLOSE_ALL":
            # Emergency close all
            global bot_pause_state
            bot_pause_state["all"] = True
            save_defcon_state()
            
            signal = write_command_signal(command, "all", {
                "action": "CLOSE_ALL",
                "emergency": True
            })
            
            try:
                resp = requests.post(f"{PORTFOLIO_API}/api/portfolio/close-all", json={
                    "source": "WAR_ROOM",
                    "emergency": True
                }, timeout=10)
                result["status"] = "sent" if resp.status_code == 200 else "signal_written"
            except:
                result["status"] = "signal_written"
            
            result["message"] = "EMERGENCY CLOSE ALL issued!"
            
        elif command == "OPEN_HEDGE":
            signal = write_command_signal(command, "casper", {
                "action": "OPEN_HEDGE",
                "defcon": current_defcon
            })
            result["status"] = "signal_written"
            result["message"] = "Hedge signal sent to Casper"
            
        elif command == "CLOSE_HEDGE":
            signal = write_command_signal(command, "casper", {
                "action": "CLOSE_HEDGE"
            })
            result["status"] = "signal_written"
            result["message"] = "Close hedge signal sent"
            
        elif command == "TIGHTEN_STOPS":
            signal = write_command_signal(command, "ghost", {
                "action": "TIGHTEN_STOPS",
                "pips": params.get("pips", 20)
            })
            result["status"] = "signal_written"
            result["message"] = "Tighten stops signal sent"
            
        elif command == "PAUSE_ALL":
            bot_pause_state["all"] = True
            save_defcon_state()
            signal = write_command_signal(command, "all", {"action": "PAUSE"})
            result["status"] = "success"
            result["message"] = "All bots paused"
            
        elif command == "RESUME_ALL":
            bot_pause_state["all"] = False
            save_defcon_state()
            signal = write_command_signal(command, "all", {"action": "RESUME"})
            result["status"] = "success"
            result["message"] = "All bots resumed"
            
        else:
            # Generic command
            signal = write_command_signal(command, bot, params)
            result["status"] = "signal_written"
            result["message"] = f"Command {command} written to signal file"
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/war-room/analyze', methods=['POST'])
def analyze():
    """Analyze a chart screenshot with AI (placeholder - actual analysis via Trading Agents API)"""
    try:
        image_base64 = request.form.get('image_base64', '')
        context = request.form.get('context', '')
        
        # Mock analysis - real analysis goes through Trading Agents API on port 8890
        analysis = {
            "situation": "Analysis available via Trading Agents API (port 8890)",
            "key_levels": "Use /agents/analyze for full multi-agent analysis",
            "recommendation": "Connect via Trading Agents API",
            "confidence": 50,
            "suggested_commands": [],
            "raw_analysis": "This endpoint is for fallback only. Use the Trading Agents API."
        }
        
        analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "analysis": analysis
        })
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/war-room/command', methods=['POST'])
def send_command():
    """Send a command to trading bots - NOW WITH REAL EXECUTION"""
    global current_defcon
    
    try:
        bot = request.form.get('bot', 'all')
        command = request.form.get('command', '')
        params = json.loads(request.form.get('params', '{}'))
        
        # Handle DEFCON commands specially
        if command.startswith('DEFCON_'):
            level = int(command.split('_')[1])
            current_defcon = level
            save_defcon_state()
            
            return jsonify({
                "status": "success",
                "message": f"DEFCON level set to {level} ({DEFCON_RULES[level]['name']})",
                "defcon": level,
                "rules": DEFCON_RULES[level]
            })
        
        # Execute the command
        result = execute_command_sync(command, bot, params)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/war-room/status', methods=['GET'])
def get_status():
    """Get current bot status with REAL data"""
    import requests
    
    try:
        # Try to get real portfolio status
        portfolio_data = {}
        try:
            resp = requests.get(f"{PORTFOLIO_API}/api/portfolio/status", timeout=5)
            if resp.status_code == 200:
                portfolio_data = resp.json()
        except:
            pass
        
        # Try to get Ghost status
        ghost_data = {}
        try:
            resp = requests.get(f"{GHOST_API}/health", timeout=5)
            if resp.status_code == 200:
                ghost_data = resp.json()
        except:
            pass
        
        # Read portfolio state file
        state_file = Path("/home/jbot/trading_ai/neo/portfolio_state.json")
        portfolio_state = {}
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    portfolio_state = json.load(f)
            except:
                pass
        
        # Calculate exposure
        total_exposure = portfolio_state.get("total_exposure", 0)
        positions = portfolio_state.get("positions", [])
        
        return jsonify({
            "defcon": current_defcon,
            "defcon_rules": DEFCON_RULES[current_defcon],
            "pause_state": bot_pause_state,
            "ghost": {
                "total_exposure": f"${total_exposure:,.0f}" if total_exposure else "$0",
                "lots": str(len(positions)),
                "positions": str(len(positions)),
                "status": "ONLINE" if ghost_data else "UNKNOWN"
            },
            "casper": {
                "hedge_active": "OFF",
                "hedge_lots": "0"
            },
            "fomo": {
                "Score": "0",
                "Status": "INACTIVE"
            },
            "portfolio": portfolio_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/war-room/defcon', methods=['POST'])
def set_defcon():
    """Set DEFCON level directly"""
    global current_defcon
    
    try:
        data = request.get_json() or {}
        level = data.get('level', request.form.get('level', 3))
        level = int(level)
        
        if level < 1 or level > 5:
            return jsonify({"error": "DEFCON must be 1-5"}), 400
        
        current_defcon = level
        state = save_defcon_state()
        
        return jsonify({
            "status": "success",
            "defcon": level,
            "rules": DEFCON_RULES[level],
            "state": state
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/war-room/commands', methods=['GET'])
def get_commands():
    """Get recent command history"""
    try:
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE, 'r') as f:
                commands = json.load(f)
            return jsonify({"commands": commands[-20:]})  # Last 20
        return jsonify({"commands": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/war-room/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    return jsonify(analysis_history[-10:])


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "service": "war-room-api",
        "defcon": current_defcon,
        "commands_enabled": True
    })


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize DEFCON state on startup
save_defcon_state()

if __name__ == "__main__":
    print("Starting War Room API on port 8889...")
    print(f"DEFCON Level: {current_defcon}")
    print(f"Signal Directory: {SIGNAL_DIR}")
    app.run(host="0.0.0.0", port=8889, debug=False)
