# Institutional Research Team

## Overview

Three specialized institutional-grade research analysts, each covering a specific basket.

---

## THE TEAM

### 1. NEO - Gold Analyst
**Basket:** XAUUSD, MGC Futures, Precious Metals

**Expertise:**
- Central bank policy and real yields
- Fed impact on gold
- Geopolitical safe-haven flows
- COT positioning data
- DXY correlation

**Prompt:** `/home/jbot/trading_ai/neo/prompts/neo_gold_institutional.txt`

**Use For:**
- Gold market analysis
- MGC futures trading decisions
- Macro-driven precious metals plays

---

### 2. CLAUDIA - BTC Miners & Hyperscaling Analyst  
**Basket:** IREN, CLSK, CIFR (Primary) | MARA, RIOT, WULF (Secondary)

**Expertise:**
- Bitcoin mining economics
- AI/HPC pivot analysis
- BTC price correlation
- Options strategies for miners
- Dilution and capital structure

**Prompt:** `/home/jbot/trading_ai/claudia/prompts/claudia_btc_miners_institutional.txt`

**Use For:**
- BTC miner stock analysis
- Options plays on IREN/CLSK/CIFR
- AI hyperscaling thesis evaluation

---

### 3. META - Technical Analyst
**Basket:** All Assets (Cross-Coverage)

**Expertise:**
- Chart patterns and price action
- Support/resistance identification
- Multi-timeframe analysis
- Entry/exit timing
- Indicator confluence

**Prompt:** `/home/jbot/trading_ai/neo/meta_bot/prompts/meta_technical_institutional.txt`

**Use For:**
- Entry timing for any asset
- Technical setup identification
- Stop loss and TP placement
- Pattern recognition

---

## WORKFLOW

```
1. CLAUDIA analyzes fundamentals of IREN/CLSK/CIFR
   → Produces: Investment thesis, scenarios, risk register

2. NEO analyzes gold macro environment
   → Produces: Trade setup, DCA levels, catalyst calendar

3. META identifies technical entry points
   → Produces: Entry zones, S/R levels, signal strength

4. HUMAN makes final decision with all three inputs
```

---

## OPERATING PRINCIPLES

All analysts follow these rules:
- No generic advice
- No motivational language
- Flag uncertainty explicitly
- Separate facts from assumptions
- Write like capital is at risk

---

## Created: January 29, 2026
