"""
Stable Signal Manager - Lock signals so they don't flip-flop on refresh!

Generates signal ONCE per day, locks it until next day.
Weekend mode: Friday's signal locked until Monday.
"""
import json
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Optional
import pytz


class StableSignalManager:
    """
    Generate signal ONCE per day, lock it until next day.
    No more flip-flopping on refresh!
    """
    
    SIGNAL_FILE = Path("/home/jbot/trading_ai/data/locked_signal.json")
    ET_TIMEZONE = pytz.timezone('America/New_York')
    
    def __init__(self):
        self.signal_file = self.SIGNAL_FILE
        self.signal_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_et_now(self) -> datetime:
        """Get current time in Eastern timezone"""
        return datetime.now(self.ET_TIMEZONE)
    
    def is_market_open(self) -> bool:
        """Check if US stock market is open"""
        now = self.get_et_now()
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def is_premarket(self) -> bool:
        """Check if it's pre-market hours (4 AM - 9:30 AM ET)"""
        now = self.get_et_now()
        if now.weekday() >= 5:
            return False
        premarket_open = time(4, 0)
        market_open = time(9, 30)
        current_time = now.time()
        return premarket_open <= current_time < market_open
    
    def is_afterhours(self) -> bool:
        """Check if it's after-hours (4 PM - 8 PM ET)"""
        now = self.get_et_now()
        if now.weekday() >= 5:
            return False
        market_close = time(16, 0)
        afterhours_close = time(20, 0)
        current_time = now.time()
        return market_close < current_time <= afterhours_close
    
    def is_weekend(self) -> bool:
        """Check if it's weekend"""
        return self.get_et_now().weekday() >= 5
    
    def get_market_status(self) -> str:
        """Get current market status"""
        if self.is_weekend():
            return "WEEKEND"
        elif self.is_market_open():
            return "OPEN"
        elif self.is_premarket():
            return "PRE-MARKET"
        elif self.is_afterhours():
            return "AFTER-HOURS"
        else:
            return "CLOSED"
    
    def get_signal_date(self) -> str:
        """Get the date for which signal should be valid"""
        now = self.get_et_now()
        
        # If weekend, use Friday's date
        if now.weekday() == 5:  # Saturday
            return (now - timedelta(days=1)).strftime('%Y-%m-%d')
        elif now.weekday() == 6:  # Sunday
            return (now - timedelta(days=2)).strftime('%Y-%m-%d')
        else:
            return now.strftime('%Y-%m-%d')
    
    def get_next_update_time(self) -> str:
        """When will signal next be updated"""
        now = self.get_et_now()
        
        if now.weekday() == 4:  # Friday
            # Next update Monday 6 AM ET
            next_monday = now + timedelta(days=3)
            next_update = next_monday.replace(hour=6, minute=0, second=0, microsecond=0)
        elif now.weekday() >= 5:  # Weekend
            # Next update Monday 6 AM ET
            days_until_monday = 7 - now.weekday()
            next_monday = now + timedelta(days=days_until_monday)
            next_update = next_monday.replace(hour=6, minute=0, second=0, microsecond=0)
        else:
            # Next update tomorrow 6 AM ET
            tomorrow = now + timedelta(days=1)
            next_update = tomorrow.replace(hour=6, minute=0, second=0, microsecond=0)
        
        return next_update.isoformat()
    
    def should_regenerate(self, locked_signal: dict) -> bool:
        """
        Check if we should regenerate the signal.
        
        Regenerate if:
        - Signal is from a different date
        - Market just opened and signal is old
        """
        signal_date = self.get_signal_date()
        
        if locked_signal.get('date') != signal_date:
            return True
        
        # If it's after 6 AM and signal was generated before 6 AM today, regenerate
        now = self.get_et_now()
        if now.hour >= 6 and not self.is_weekend():
            generated_at = locked_signal.get('generated_at', '')
            if generated_at:
                try:
                    gen_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                    gen_time_et = gen_time.astimezone(self.ET_TIMEZONE)
                    
                    # If generated before 6 AM today, regenerate
                    today_6am = now.replace(hour=6, minute=0, second=0, microsecond=0)
                    if gen_time_et < today_6am and now > today_6am:
                        return True
                except:
                    pass
        
        return False
    
    def get_locked_signal(self, force_refresh: bool = False) -> dict:
        """
        Get today's locked signal.
        If doesn't exist or outdated, generate new one.
        """
        signal_date = self.get_signal_date()
        
        # Check if we have a locked signal
        if self.signal_file.exists() and not force_refresh:
            try:
                with open(self.signal_file) as f:
                    locked = json.load(f)
                
                if not self.should_regenerate(locked):
                    # Return existing locked signal with updated status
                    locked['is_locked'] = True
                    locked['is_weekend'] = self.is_weekend()
                    locked['market_status'] = self.get_market_status()
                    locked['next_update'] = self.get_next_update_time()
                    return locked
            except (json.JSONDecodeError, KeyError):
                pass  # Regenerate if file is corrupted
        
        # Generate new signal and lock it
        return self.generate_and_lock_signal(signal_date)
    
    def generate_and_lock_signal(self, signal_date: str) -> dict:
        """Generate signal and lock it for the day"""
        from paper_trading.daily_signals import DailySignalGenerator
        
        generator = DailySignalGenerator()
        signals = generator.generate_daily_signals()
        
        now = self.get_et_now()
        
        locked_signal = {
            'date': signal_date,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'generated_at_et': now.isoformat(),
            'locked_at': datetime.utcnow().isoformat() + 'Z',
            'signals': signals['signals'],
            'summary': signals.get('summary', {}),
            'is_locked': True,
            'is_weekend': self.is_weekend(),
            'market_status': self.get_market_status(),
            'next_update': self.get_next_update_time()
        }
        
        # Save to file
        with open(self.signal_file, 'w') as f:
            json.dump(locked_signal, f, indent=2)
        
        return locked_signal
    
    def force_regenerate(self) -> dict:
        """Force regenerate the signal (admin use only)"""
        signal_date = self.get_signal_date()
        return self.generate_and_lock_signal(signal_date)


# Convenience function
def get_stable_signal(force_refresh: bool = False) -> dict:
    """Get the stable locked signal"""
    manager = StableSignalManager()
    return manager.get_locked_signal(force_refresh)


if __name__ == "__main__":
    # Test
    manager = StableSignalManager()
    print(f"Market status: {manager.get_market_status()}")
    print(f"Is weekend: {manager.is_weekend()}")
    print(f"Signal date: {manager.get_signal_date()}")
    print(f"Next update: {manager.get_next_update_time()}")
    
    signal = manager.get_locked_signal()
    print(f"\nLocked signal for {signal['date']}:")
    print(json.dumps(signal, indent=2)[:500] + "...")
