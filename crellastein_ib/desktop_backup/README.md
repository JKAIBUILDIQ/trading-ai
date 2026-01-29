# Desktop Backup Agent - Crella Local Execution

When the H100 can't reach TWS, Crella on your desktop can execute trades locally.

## Setup on Desktop

1. Copy this folder to your Windows desktop
2. Install ib_insync: `pip install ib_insync`
3. Run commands manually or have Crella execute them

## Files

- `pending_commands.json` - Commands from H100 that need execution
- `execute_local.py` - Script to execute pending commands locally
- `quick_commands.py` - One-liner trade commands

## Usage

When H100 can't connect, I'll update `pending_commands.json` with the trade.
Then tell Crella: "Execute the pending commands in trading_ai backup folder"

Or run directly:
```
python execute_local.py
```
