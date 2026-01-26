#!/usr/bin/env python3
"""
MT5 Risk Analyzer - Quick Portfolio Analysis Tool
=================================================
Agent shortcut: "Run risk simulator" or "Analyze my MT5 positions"

Usage:
  1. User sends MT5 screenshot(s)
  2. Agent extracts positions and updates REALIZED_TRADES / OPEN_POSITIONS below
  3. Run: python3 ~/trading_ai/tools/mt5_risk_analyzer.py

Last Updated: 2026-01-26
"""

import json
from datetime import datetime
from typing import List, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# 📋 UPDATE THESE WITH EXTRACTED DATA FROM MT5 SCREENSHOTS
# ═══════════════════════════════════════════════════════════════════════════════

# Account Info (from MT5 footer)
ACCOUNT = {
    "starting_balance": 100000.00,
    "current_balance": 103551.54,
    "equity": 102626.89,
    "margin": 19315.42,
    "free_margin": 83311.47,
    "margin_level": 531.32,
}

# REALIZED TRADES (from History tab)
# Format: {"symbol", "type", "volume", "profit", "comment"}
REALIZED_TRADES = [
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.45, "profit": -32.85, "comment": "v032|SELL|COUNTER"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.5, "profit": 398.00, "comment": "SELL_SCALP|Manual"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.2, "profit": -3.00, "comment": "NEO|SELL|CASPER"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.45, "profit": 58.95, "comment": "straddle"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.3, "profit": 83.70, "comment": "v033|STRADDLE|SELL"},
    {"symbol": "XAUUSD", "type": "BUY", "volume": 0.3, "profit": -83.10, "comment": "v033|STRADDLE|BUY"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 1.0, "profit": 54.00, "comment": "v033|STRADDLE|SELL"},
    {"symbol": "XAUUSD", "type": "BUY", "volume": 1.0, "profit": -112.00, "comment": "v033|STRADDLE|BUY"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 1.0, "profit": 994.00, "comment": "v033|STRADDLE|SELL TP"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.5, "profit": 751.50, "comment": "straddle"},
    {"symbol": "XAUUSD", "type": "BUY", "volume": 1.0, "profit": 1001.00, "comment": "straddle TP"},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.5, "profit": -623.50, "comment": "v092|XAUUSD|Pos2|1x SL"},
    {"symbol": "XAUUSD", "type": "BUY", "volume": 1.0, "profit": 1000.00, "comment": "v033|STRADDLE|BUY TP"},
    {"symbol": "USDJPY", "type": "BUY", "volume": 0.5, "profit": 64.84, "comment": "v092|USDJPY|Pos1|1x TP"},
]

# OPEN POSITIONS (from Trade tab)
# Format: {"symbol", "type", "volume", "entry", "sl", "tp", "profit"}
OPEN_POSITIONS = [
    {"symbol": "USDJPY", "type": "BUY", "volume": 1.0, "entry": 154.226, "sl": 154.059, "tp": 154.426, "profit": 91.99},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 1.0, "entry": 5069.99, "sl": 5082.52, "tp": 5055.02, "profit": -34.00},
    {"symbol": "USDCHF", "type": "BUY", "volume": 1.25, "entry": 0.77689, "sl": 0.7367, "tp": 0.77951, "profit": 170.32},
    {"symbol": "AUDUSD", "type": "SELL", "volume": 0.44, "entry": 0.69146, "sl": 0.69286, "tp": 0.68968, "profit": 21.12},
    {"symbol": "GBPUSD", "type": "SELL", "volume": 0.44, "entry": 1.36652, "sl": 1.36745, "tp": 1.36469, "profit": 15.84},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 1.0, "entry": 5058.86, "sl": 5108.83, "tp": 5048.83, "profit": -1147.00},
    {"symbol": "USDJPY", "type": "BUY", "volume": 0.37, "entry": 154.027, "sl": 153.498, "tp": 155.376, "profit": 81.73},
    {"symbol": "GBPUSD", "type": "BUY", "volume": 1.4, "entry": 1.36597, "sl": 1.34097, "tp": 1.40597, "profit": 19.60},
    {"symbol": "EURUSD", "type": "SELL", "volume": 0.44, "entry": 1.18576, "sl": 1.18741, "tp": 1.18358, "profit": 17.16},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.44, "entry": 5075.70, "sl": 5086.50, "tp": 5004.28, "profit": 236.28},
    {"symbol": "USDCHF", "type": "SELL", "volume": 0.5, "entry": 0.77697, "sl": 0.78697, "tp": 0.76897, "profit": -70.69},
    {"symbol": "EURUSD", "type": "BUY", "volume": 0.5, "entry": 1.18652, "sl": 1.16152, "tp": 1.22652, "profit": -59.50},
    {"symbol": "XAUUSD", "type": "SELL", "volume": 0.3, "entry": 5078.33, "sl": 5158.68, "tp": 4928.68, "profit": 240.00},
    {"symbol": "XAUUSD", "type": "BUY", "volume": 0.25, "entry": 5090.47, "sl": 5010.47, "tp": 5240.47, "profit": -507.50},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 POINT VALUES FOR P&L CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

POINT_VALUES = {
    "XAUUSD": 1.0,      # $1 per point per lot
    "EURUSD": 10.0,     # $10 per pip per lot
    "GBPUSD": 10.0,
    "USDJPY": 6.5,
    "USDCHF": 11.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,
    "USDCAD": 7.5,
    "GBPJPY": 6.5,
    "EURJPY": 6.5,
    "BTCUSD": 1.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_point_value(symbol: str) -> float:
    return POINT_VALUES.get(symbol.upper(), 10.0)

def calculate_max_profit(pos: Dict) -> float:
    pv = get_point_value(pos['symbol'])
    if pos['type'] == 'BUY':
        return (pos['tp'] - pos['entry']) * pos['volume'] * pv
    else:
        return (pos['entry'] - pos['tp']) * pos['volume'] * pv

def calculate_max_loss(pos: Dict) -> float:
    pv = get_point_value(pos['symbol'])
    if pos['type'] == 'BUY':
        return (pos['entry'] - pos['sl']) * pos['volume'] * pv
    else:
        return (pos['sl'] - pos['entry']) * pos['volume'] * pv

def classify_strategy(comment: str) -> str:
    comment = comment.upper()
    if 'STRADDLE' in comment:
        return 'STRADDLE'
    elif 'SCALP' in comment:
        return 'SCALP'
    elif 'NEO' in comment:
        return 'NEO'
    elif 'GAP' in comment:
        return 'GAP_FILL'
    elif 'SUPPLY' in comment or 'DEMAND' in comment:
        return 'SUPPLY_DEMAND'
    elif 'JACKAL' in comment:
        return 'JACKAL'
    elif 'V092' in comment:
        return 'v092'
    elif 'V091' in comment:
        return 'v091'
    elif 'V033' in comment:
        return 'v033'
    else:
        return 'OTHER'

def analyze_portfolio():
    """Main analysis function"""
    
    print('═' * 70)
    print('💰 MT5 PORTFOLIO RISK ANALYSIS')
    print(f'   Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('═' * 70)
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # ACCOUNT OVERVIEW
    # ─────────────────────────────────────────────────────────────────────
    print('📊 ACCOUNT OVERVIEW')
    print('-' * 70)
    print(f'   Starting Balance:     ${ACCOUNT["starting_balance"]:>12,.2f}')
    print(f'   Current Balance:      ${ACCOUNT["current_balance"]:>12,.2f}')
    print(f'   Equity:               ${ACCOUNT["equity"]:>12,.2f}')
    print(f'   Margin Used:          ${ACCOUNT["margin"]:>12,.2f}')
    print(f'   Free Margin:          ${ACCOUNT["free_margin"]:>12,.2f}')
    print(f'   Margin Level:         {ACCOUNT["margin_level"]:>12.2f}%')
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # REALIZED P&L
    # ─────────────────────────────────────────────────────────────────────
    total_realized = sum(t['profit'] for t in REALIZED_TRADES)
    winners = [t for t in REALIZED_TRADES if t['profit'] > 0]
    losers = [t for t in REALIZED_TRADES if t['profit'] < 0]
    win_total = sum(t['profit'] for t in winners)
    loss_total = sum(t['profit'] for t in losers)
    
    print('═' * 70)
    print('✅ REALIZED P&L (Closed Trades)')
    print('═' * 70)
    print(f'   Total Realized:       ${total_realized:>12,.2f}  ← LOCKED IN')
    print(f'   Winning Trades:       ${win_total:>12,.2f}  ({len(winners)} trades)')
    print(f'   Losing Trades:        ${loss_total:>12,.2f}  ({len(losers)} trades)')
    if len(REALIZED_TRADES) > 0:
        print(f'   Win Rate:             {len(winners)}/{len(REALIZED_TRADES)} = {len(winners)/len(REALIZED_TRADES)*100:.0f}%')
    print()
    
    # Strategy breakdown
    strategies = {}
    for t in REALIZED_TRADES:
        strat = classify_strategy(t.get('comment', ''))
        if strat not in strategies:
            strategies[strat] = {'profit': 0, 'count': 0, 'wins': 0}
        strategies[strat]['profit'] += t['profit']
        strategies[strat]['count'] += 1
        if t['profit'] > 0:
            strategies[strat]['wins'] += 1
    
    print('   By Strategy:')
    for strat, data in sorted(strategies.items(), key=lambda x: x[1]['profit'], reverse=True):
        wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f'      {strat:<15} ${data["profit"]:>8,.2f}  ({data["count"]} trades, {wr:.0f}% WR)')
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # UNREALIZED P&L
    # ─────────────────────────────────────────────────────────────────────
    total_unrealized = sum(p['profit'] for p in OPEN_POSITIONS)
    open_winners = [p for p in OPEN_POSITIONS if p['profit'] > 0]
    open_losers = [p for p in OPEN_POSITIONS if p['profit'] < 0]
    
    print('═' * 70)
    print('📈 UNREALIZED P&L (Open Positions)')
    print('═' * 70)
    print(f'   Total Unrealized:     ${total_unrealized:>12,.2f}  ← AT RISK')
    print(f'   In Profit:            ${sum(p["profit"] for p in open_winners):>12,.2f}  ({len(open_winners)} positions)')
    print(f'   In Loss:              ${sum(p["profit"] for p in open_losers):>12,.2f}  ({len(open_losers)} positions)')
    print()
    
    # Group by symbol
    symbols = {}
    for p in OPEN_POSITIONS:
        sym = p['symbol']
        if sym not in symbols:
            symbols[sym] = {'profit': 0, 'long': 0, 'short': 0, 'max_profit': 0, 'max_loss': 0}
        symbols[sym]['profit'] += p['profit']
        symbols[sym]['max_profit'] += calculate_max_profit(p)
        symbols[sym]['max_loss'] += calculate_max_loss(p)
        if p['type'] == 'BUY':
            symbols[sym]['long'] += p['volume']
        else:
            symbols[sym]['short'] += p['volume']
    
    print('   By Symbol:')
    print(f'   {"Symbol":<10} {"P&L":>10} {"Long":>8} {"Short":>8} {"Net":>12}')
    print('   ' + '-' * 55)
    for sym, data in sorted(symbols.items(), key=lambda x: abs(x[1]['profit']), reverse=True):
        net = data['long'] - data['short']
        direction = '🟢 LONG' if net > 0.1 else '🔴 SHORT' if net < -0.1 else '⚪ HEDGE'
        print(f'   {sym:<10} ${data["profit"]:>9,.2f} {data["long"]:>8.2f} {data["short"]:>8.2f} {direction}')
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # MAX PROFIT / LOSS SCENARIOS
    # ─────────────────────────────────────────────────────────────────────
    total_max_profit = sum(calculate_max_profit(p) for p in OPEN_POSITIONS)
    total_max_loss = sum(calculate_max_loss(p) for p in OPEN_POSITIONS)
    rr_ratio = total_max_profit / total_max_loss if total_max_loss > 0 else 0
    
    print('═' * 70)
    print('🎯 SCENARIO ANALYSIS (Open Positions)')
    print('═' * 70)
    print(f'   If ALL TPs Hit:       ${total_max_profit:>12,.2f}')
    print(f'   If ALL SLs Hit:       ${total_max_loss:>12,.2f}')
    print(f'   Risk/Reward:          1:{rr_ratio:.2f}')
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # NET POSITION
    # ─────────────────────────────────────────────────────────────────────
    net_pnl = total_realized + total_unrealized
    
    print('═' * 70)
    print('💵 NET POSITION')
    print('═' * 70)
    print(f'   Realized:             ${total_realized:>12,.2f}')
    print(f'   Unrealized:           ${total_unrealized:>12,.2f}')
    print(f'   ─────────────────────────────────────────')
    print(f'   NET P&L:              ${net_pnl:>12,.2f}')
    print()
    growth_pct = (net_pnl / ACCOUNT["starting_balance"]) * 100
    print(f'   Account Growth:       ${net_pnl:>12,.2f}  ({growth_pct:+.2f}%)')
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # RISK ALERTS
    # ─────────────────────────────────────────────────────────────────────
    print('═' * 70)
    print('⚠️  RISK ALERTS')
    print('═' * 70)
    
    # Biggest losers
    worst = sorted(OPEN_POSITIONS, key=lambda x: x['profit'])[:3]
    print('   Biggest Drags (Open):')
    for p in worst:
        if p['profit'] < 0:
            print(f'      🔴 {p["symbol"]} {p["type"]} {p["volume"]} lots: ${p["profit"]:,.2f}')
    print()
    
    # Heavy exposure warnings
    print('   Exposure Warnings:')
    for sym, data in symbols.items():
        net = abs(data['long'] - data['short'])
        if net > 1.5:
            direction = 'LONG' if data['long'] > data['short'] else 'SHORT'
            print(f'      ⚠️ {sym}: Heavy {direction} exposure ({net:.2f} lots)')
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # TOP PERFORMERS
    # ─────────────────────────────────────────────────────────────────────
    print('═' * 70)
    print('🏆 TOP PERFORMERS (Open)')
    print('═' * 70)
    best = sorted(OPEN_POSITIONS, key=lambda x: x['profit'], reverse=True)[:3]
    for p in best:
        if p['profit'] > 0:
            print(f'   🟢 {p["symbol"]} {p["type"]} {p["volume"]} lots: +${p["profit"]:,.2f}')
    print()
    
    print('═' * 70)
    print('✅ ANALYSIS COMPLETE')
    print('═' * 70)
    
    # Return summary dict for API use
    return {
        "timestamp": datetime.now().isoformat(),
        "account": ACCOUNT,
        "realized": {
            "total": total_realized,
            "winners": win_total,
            "losers": loss_total,
            "win_rate": len(winners) / len(REALIZED_TRADES) if REALIZED_TRADES else 0,
            "by_strategy": strategies,
        },
        "unrealized": {
            "total": total_unrealized,
            "in_profit": sum(p["profit"] for p in open_winners),
            "in_loss": sum(p["profit"] for p in open_losers),
            "by_symbol": symbols,
        },
        "scenarios": {
            "max_profit": total_max_profit,
            "max_loss": total_max_loss,
            "risk_reward": rr_ratio,
        },
        "net_pnl": net_pnl,
        "growth_percent": growth_pct,
    }


if __name__ == "__main__":
    result = analyze_portfolio()
    
    # Optionally save to JSON
    # with open('/home/jbot/trading_ai/data/risk_analysis.json', 'w') as f:
    #     json.dump(result, f, indent=2, default=str)
