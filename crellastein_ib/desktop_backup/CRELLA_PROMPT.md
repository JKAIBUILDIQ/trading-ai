# Prompt for Desktop Crella Agent

When H100 can't connect to TWS, copy this prompt to Crella on your desktop:

---

## BACKUP TRADE EXECUTION

The H100 couldn't connect to TWS. Please execute these trades locally:

**Current Position:** [POSITION] MGC contracts
**Target Position:** [TARGET] MGC contracts
**Action Needed:** [ACTION] [QUANTITY] MGC contracts

### Execute Commands:

```python
from ib_insync import IB, Future, MarketOrder

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=999)

contract = Future(
    conId=706903676,
    symbol='MGC',
    exchange='COMEX',
    localSymbol='MGCJ6',
)

# Execute the trade
order = MarketOrder('[ACTION]', [QUANTITY])
trade = ib.placeOrder(contract, order)
ib.sleep(5)
print(f"Order status: {trade.orderStatus.status}")

# Check position
for pos in ib.positions():
    if pos.contract.symbol == 'MGC':
        print(f"Position: {pos.position} @ ${pos.avgCost/10:.2f}")

ib.disconnect()
```

### Or run the backup script:

```
cd C:\Users\[USER]\trading_ai\crellastein_ib\desktop_backup
python execute_local.py [action] [quantity]
```

---

## Quick Commands for Crella:

- "Check MGC position" → `python execute_local.py status`
- "Sell 2 MGC" → `python execute_local.py sell 2`
- "Buy 2 MGC" → `python execute_local.py buy 2`
- "Execute pending" → `python execute_local.py`
