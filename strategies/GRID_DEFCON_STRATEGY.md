# Grid Trading Strategy & DEFCON Integration

## Key Rule
**Grid trading should ONLY be active during DEFCON 4-5 (normal/elevated conditions).**

## DEFCON Grid Settings

| DEFCON | Grid Status | Spacing | Lots | Max Levels | TP |
|--------|-------------|---------|------|------------|-----|
| 🟢 5 | ACTIVE | 1.0× base | 1.0× | 5 | +25 pips |
| 🔵 4 | ACTIVE (cautious) | 1.25× | 0.8× | 4 | +30 pips |
| 🟡 3 | ⏸️ PAUSED | 2.0× | 0.5× | 3 | +15 pips |
| 🟠 2 | ⛔ DISABLED | - | 0 | 0 | Close 30% |
| 🔴 1 | ⛔ EMERGENCY | - | 0 | 0 | Close 50% |

## Optimal Grid Conditions (DEFCON 5)

1. **Range-Bound Market**
   - ADX < 20 (no trend)
   - Price within 100-pip range for 3+ days
   - RSI oscillating 40-60

2. **Shallow Pullback in Uptrend**
   - Price above EMA 20 AND EMA 50
   - Higher highs, higher lows
   - Pullbacks < 50 pips

3. **V-Recovery Setup**
   - News-driven drop (not fundamental)
   - Support level holding
   - Volume spike on recovery

## Dangerous Grid Conditions (DEFCON 1-3)

1. **Trending Against Position**
   - ADX > 25
   - Support levels breaking
   - DXY strengthening

2. **Distribution Top (Bull Trap)**
   - Volume INCREASING on red candles
   - Lower highs within consolidation
   - At ATH or major resistance

3. **Flash Crash**
   - Gap down
   - All entries triggered instantly
   - Max exposure immediately

## Grid Formulas

### Pip Spacing
```
spacing = ATR(14) × 1.5 × DEFCON_multiplier
```

### Lot Sizing
```
total_lots = max_drawdown / (max_levels × spacing × pip_value)
lot_per_level = total_lots / max_levels × DEFCON_multiplier
```

### Take Profit
```
TP = average_entry + (TP_pips × DEFCON_tp_multiplier)
```

## Files
- `mql5/DefconGrid.mqh` - MQL5 grid module
- `mql5/DefconReader.mqh` - DEFCON reader for MT5

---
*Created: 2026-01-30*
*Rule: Grid ONLY active in DEFCON 4-5*
