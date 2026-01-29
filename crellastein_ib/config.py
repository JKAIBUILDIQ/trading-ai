"""
Crellastein IB Configuration

Shared settings for all strategies.
Matches MT5 input parameters.

Author: QUINN001
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GHOST COMMANDER SETTINGS (v020 + v0201 DCA PATCH)
# ═══════════════════════════════════════════════════════════════════════════════

GHOST_SETTINGS = {
    # Strategy ID (Magic Number)
    'strategy_id': 888020,
    'name': 'GHOST_COMMANDER',
    
    # Base Position Sizing
    'base_lots': 0.5,           # Initial lot size
    'max_lots': 1.0,            # Max lot per position
    'max_positions': 10,        # Max total positions
    'max_per_symbol': 5,        # Max DCA levels per symbol
    
    # DCA Ladder Settings (from v0201 PATCH)
    'dca_initial_lots': 0.5,    # Starting position
    'dca_lot_increment': 0.25,  # Add per level
    'dca_max_lot_size': 2.0,    # Cap per position
    'dca_pip_spacing': 30,      # Pips between entries (30, 60, 90, 120)
    'dca_max_levels': 5,        # Max DCA levels
    'dca_min_entry_gap': 20,    # Anti-stack: min 20 pts between positions
    
    # Tiered Take Profits (from average entry)
    'tp1_pips': 30,             # +30 pips = close 30%
    'tp2_pips': 60,             # +60 pips = close 30%
    'tp3_pips': 90,             # +90 pips = close ALL
    'tp1_close_percent': 30,
    'tp2_close_percent': 30,
    
    # Stop Loss
    'stop_loss_points': 150,    # Fixed SL in points
    'use_trailing_stop': True,
    'trailing_start_pips': 20,
    'trailing_distance_pips': 15,
    
    # Regime Detection
    'adx_period': 14,
    'adx_trend_threshold': 25,  # Above = TRENDING
    'atr_period': 14,
    
    # SuperTrend
    'supertrend_period': 10,
    'supertrend_multiplier': 3.0,
    
    # Trading Hours (UTC)
    'trade_start_hour': 0,
    'trade_end_hour': 23,
    
    # Free Roll Runner
    'runner_enabled': True,
    'runner_profit_threshold': 500,  # Min profit to deploy
    'runner_budget_percent': 50,     # % of profits to risk
    'runner_tp_percent': 300,        # 300% target
}

# ═══════════════════════════════════════════════════════════════════════════════
# CASPER SETTINGS (Meta Bot + Drop-Buy Martingale)
# ═══════════════════════════════════════════════════════════════════════════════

CASPER_SETTINGS = {
    # Strategy ID (Magic Number)
    'strategy_id': 8880202,
    'name': 'CASPER',
    
    # Base Position Sizing (Conservative)
    'base_lots': 0.1,           # Smaller base
    'max_lots': 0.2,            # Lower max
    
    # Drop-Buy DCA (Martingale on $10 drops)
    'dropbuy_trigger_pips': 10.0,     # Buy on every $10 drop
    'dropbuy_lot_ladder': [0.5, 0.5, 1.0, 1.0, 2.0],  # 5 levels
    'dropbuy_max_levels': 5,
    'dropbuy_tp_pips': 20.0,          # TP at +$20 from avg entry
    
    # Trailing TP
    'trail_start': 10.0,        # Start locking at +$10
    'trail_distance': 8.0,      # Trail $8 behind
    
    # Meta Bot Score Threshold
    'meta_min_score': 60.0,     # Min composite score to execute
    'meta_check_interval': 30,  # Check every 30 seconds
    
    # Risk Management
    'max_daily_loss': 500,
    'max_positions': 5,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FREE ROLLER SETTINGS (v043)
# ═══════════════════════════════════════════════════════════════════════════════

FREE_ROLLER_SETTINGS = {
    # Strategy ID
    'strategy_id': 888043,
    'name': 'FREE_ROLLER',
    
    # Profit Requirements
    'min_profit_to_roll': 500.0,
    'max_risk_percent': 100.0,  # Risk up to 100% of profits
    'risk_reward_ratio': 2.0,
    
    # Position Management
    'initial_sl_pips': 25.0,
    'tp1_percent': 50.0,        # Take 50% at first target
    'move_to_breakeven': True,  # After TP1
    'free_roll_trail_pips': 30.0,
    
    # Exit Conditions
    'exit_death_cross': True,
    'exit_below_ema50': True,
    'exit_max_drawdown': 50.0,  # % of profit to give back
}

# ═══════════════════════════════════════════════════════════════════════════════
# COUNTERSTRIKE SETTINGS (v041 - Bearish)
# ═══════════════════════════════════════════════════════════════════════════════

COUNTERSTRIKE_SETTINGS = {
    # Strategy ID
    'strategy_id': 888041,
    'name': 'COUNTERSTRIKE',
    
    # Bearish triggers
    'deploy_on_death_cross': True,
    'deploy_on_breakdown': True,
    'breakdown_threshold_pips': 50,
    
    # Position sizing
    'base_lots': 0.3,
    'max_lots': 0.5,
    
    # Short-specific
    'short_tp_pips': 30,
    'short_sl_pips': 50,
}

# ═══════════════════════════════════════════════════════════════════════════════
# IB CONNECTION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

IB_SETTINGS = {
    'host': '100.119.161.65',   # Gringot TWS via Tailscale
    'paper_port': 7497,
    'live_port': 7496,
    'client_id_ghost': 80,
    'client_id_casper': 81,
    'client_id_roller': 82,
    'client_id_counter': 83,
}

# ═══════════════════════════════════════════════════════════════════════════════
# MGC (MICRO GOLD FUTURES) CONTRACT SPECS
# ═══════════════════════════════════════════════════════════════════════════════

MGC_SPECS = {
    'symbol': 'MGC',
    'exchange': 'COMEX',
    'currency': 'USD',
    'multiplier': 10,           # 10 oz per contract
    'tick_size': 0.10,          # $0.10 tick
    'tick_value': 1.00,         # $1 per tick per contract
    'margin_approx': 1200,      # ~$1200 per contract
    
    # Contract months: Feb(G), Apr(J), Jun(M), Aug(Q), Oct(V), Dec(Z)
    'contract_months': [2, 4, 6, 8, 10, 12],
    
    # Conversion from MT5 XAUUSD lots
    # 1 MT5 lot = 100 oz = 10 MGC contracts
    'mt5_lot_to_mgc': 10,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PIPS CONVERSION (Gold)
# ═══════════════════════════════════════════════════════════════════════════════

# MT5 XAUUSD: 1 pip = $0.10 (1 point)
# IB MGC: 1 tick = $0.10 = $1 per contract

def pips_to_dollars(pips: float, contracts: int = 1) -> float:
    """Convert pips to dollars for MGC"""
    return pips * contracts  # 1 pip = $1 per contract

def dollars_to_pips(dollars: float, contracts: int = 1) -> float:
    """Convert dollars to pips for MGC"""
    return dollars / contracts if contracts > 0 else 0
