"""
BATTLE OPS SCHEDULER
Runs daily operations at scheduled times
Uses APScheduler for reliable scheduling
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.daily_battle_ops import (
    DailyOps,
    run_pre_market,
    run_market_open,
    run_mid_day,
    run_power_hour,
    run_after_hours
)
from ops.learning_engine import LearningEngine

# Timezone for market hours
ET = pytz.timezone('America/New_York')


class BattleOpsScheduler:
    """Scheduler for daily battle operations"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=ET)
        self.is_running = False
        
    def setup_schedules(self):
        """Setup all scheduled jobs"""
        
        # PRE-MARKET: 6:00 AM ET (Mon-Fri)
        self.scheduler.add_job(
            self._run_pre_market,
            CronTrigger(hour=6, minute=0, day_of_week='mon-fri', timezone=ET),
            id='pre_market',
            name='Pre-Market Operations',
            replace_existing=True
        )
        
        # MARKET OPEN: 9:35 AM ET (5 min after open)
        self.scheduler.add_job(
            self._run_market_open,
            CronTrigger(hour=9, minute=35, day_of_week='mon-fri', timezone=ET),
            id='market_open',
            name='Market Open Operations',
            replace_existing=True
        )
        
        # MID-DAY: 12:00 PM ET
        self.scheduler.add_job(
            self._run_mid_day,
            CronTrigger(hour=12, minute=0, day_of_week='mon-fri', timezone=ET),
            id='mid_day',
            name='Mid-Day Operations',
            replace_existing=True
        )
        
        # POWER HOUR: 3:00 PM ET
        self.scheduler.add_job(
            self._run_power_hour,
            CronTrigger(hour=15, minute=0, day_of_week='mon-fri', timezone=ET),
            id='power_hour',
            name='Power Hour Operations',
            replace_existing=True
        )
        
        # AFTER HOURS: 4:30 PM ET
        self.scheduler.add_job(
            self._run_after_hours,
            CronTrigger(hour=16, minute=30, day_of_week='mon-fri', timezone=ET),
            id='after_hours',
            name='After Hours Operations',
            replace_existing=True
        )
        
        # DAILY LEARNING: 8:00 PM ET
        self.scheduler.add_job(
            self._run_learning,
            CronTrigger(hour=20, minute=0, day_of_week='mon-fri', timezone=ET),
            id='daily_learning',
            name='Daily Learning Update',
            replace_existing=True
        )
        
        # WEEKLY SUMMARY: Sunday 8:00 PM ET
        self.scheduler.add_job(
            self._run_weekly_summary,
            CronTrigger(hour=20, minute=0, day_of_week='sun', timezone=ET),
            id='weekly_summary',
            name='Weekly Summary Report',
            replace_existing=True
        )
        
        print("📅 Scheduled jobs:")
        for job in self.scheduler.get_jobs():
            print(f"  - {job.name}: {job.trigger}")
    
    async def _run_pre_market(self):
        """Execute pre-market phase"""
        print(f"[{datetime.now(ET)}] 🌅 Starting Pre-Market Operations...")
        try:
            await run_pre_market()
        except Exception as e:
            print(f"❌ Pre-market error: {e}")
    
    async def _run_market_open(self):
        """Execute market open phase"""
        print(f"[{datetime.now(ET)}] 🔔 Starting Market Open Operations...")
        try:
            await run_market_open()
        except Exception as e:
            print(f"❌ Market open error: {e}")
    
    async def _run_mid_day(self):
        """Execute mid-day phase"""
        print(f"[{datetime.now(ET)}] ☀️ Starting Mid-Day Operations...")
        try:
            await run_mid_day()
        except Exception as e:
            print(f"❌ Mid-day error: {e}")
    
    async def _run_power_hour(self):
        """Execute power hour phase"""
        print(f"[{datetime.now(ET)}] ⚡ Starting Power Hour Operations...")
        try:
            await run_power_hour()
        except Exception as e:
            print(f"❌ Power hour error: {e}")
    
    async def _run_after_hours(self):
        """Execute after hours phase"""
        print(f"[{datetime.now(ET)}] 🌙 Starting After Hours Operations...")
        try:
            await run_after_hours()
        except Exception as e:
            print(f"❌ After hours error: {e}")
    
    async def _run_learning(self):
        """Execute daily learning update"""
        print(f"[{datetime.now(ET)}] 🧠 Running Daily Learning...")
        try:
            engine = LearningEngine()
            engine.update_agent_weights()
            engine.analyze_pattern_success()
            report = engine.generate_training_report()
            
            # Send to Telegram
            ops = DailyOps()
            await ops.send_telegram(f"<pre>{report[:4000]}</pre>")
        except Exception as e:
            print(f"❌ Learning error: {e}")
    
    async def _run_weekly_summary(self):
        """Generate and send weekly summary"""
        print(f"[{datetime.now(ET)}] 📊 Generating Weekly Summary...")
        try:
            engine = LearningEngine()
            recs = engine.get_agent_recommendations()
            
            msg = "<b>📊 WEEKLY AGENT PERFORMANCE SUMMARY</b>\n\n"
            
            for agent, data in recs.items():
                msg += f"<b>{agent.upper()}</b>\n"
                msg += f"  Weight: {data['weight']:.2f}x\n"
                msg += f"  Accuracy: {data['accuracy']:.1%}\n"
                for rec in data.get('recommendations', []):
                    msg += f"  {rec}\n"
                msg += "\n"
            
            ops = DailyOps()
            await ops.send_telegram(msg)
        except Exception as e:
            print(f"❌ Weekly summary error: {e}")
    
    def start(self):
        """Start the scheduler"""
        self.setup_schedules()
        self.scheduler.start()
        self.is_running = True
        print(f"🚀 Battle Ops Scheduler started at {datetime.now(ET)}")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        self.is_running = False
        print("🛑 Battle Ops Scheduler stopped")


async def main():
    """Main entry point"""
    scheduler = BattleOpsScheduler()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received...")
        scheduler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start scheduler
    scheduler.start()
    
    print("\n⏳ Scheduler running. Press Ctrl+C to stop.\n")
    print(f"Current time (ET): {datetime.now(ET)}")
    print(f"Next jobs will run as scheduled.\n")
    
    # Keep running
    while True:
        await asyncio.sleep(60)
        # Heartbeat every hour
        if datetime.now().minute == 0:
            print(f"💓 Scheduler heartbeat: {datetime.now(ET)}")


if __name__ == "__main__":
    asyncio.run(main())
