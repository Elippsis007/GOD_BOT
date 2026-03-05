#!/usr/bin/env python3
"""
Quick script to check current trading session status.
"""

import pytz
import sys
import os
from datetime import datetime

# Add the forex_system directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'forex_system'))

from config.settings import (
    LONDON_OPEN_CET, LONDON_CLOSE_CET,
    NY_OPEN_CET, NY_CLOSE_CET,
    OVERLAP_START, OVERLAP_END,
    TRADER_TIMEZONE
)

def check_session_status():
    """Check current trading session status."""
    timezone = pytz.timezone(TRADER_TIMEZONE)
    current_hour = datetime.now(timezone).hour
    
    print("📊 TRADING SESSION STATUS")
    print("=" * 40)
    print(f"Current time: {datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current hour: {current_hour}:00 CET")
    print()
    
    # Check session status
    is_london = LONDON_OPEN_CET <= current_hour < LONDON_CLOSE_CET
    is_ny = NY_OPEN_CET <= current_hour < NY_CLOSE_CET
    is_overlap = OVERLAP_START <= current_hour < OVERLAP_END
    
    print("Session Status:")
    print(f"  🇬🇧 London Session: {'✅ OPEN' if is_london else '❌ CLOSED'} ({LONDON_OPEN_CET}:00 - {LONDON_CLOSE_CET}:00)")
    print(f"  🇺🇸 New York Session: {'✅ OPEN' if is_ny else '❌ CLOSED'} ({NY_OPEN_CET}:00 - {NY_CLOSE_CET}:00)")
    print(f"  🔄 Overlap Session: {'✅ OPEN' if is_overlap else '❌ CLOSED'} ({OVERLAP_START}:00 - {OVERLAP_END}:00)")
    print()
    
    if is_overlap:
        print("🎯 RECOMMENDED: Overlap session active - best liquidity!")
    elif is_london or is_ny:
        print("⚠️  WARNING: Only single session active - may have fewer signals")
    else:
        print("💤 INFO: No trading sessions active - system waiting")
    
    print()
    print("Next session changes:")
    if current_hour < LONDON_OPEN_CET:
        print(f"  🇬🇧 London opens: {LONDON_OPEN_CET}:00 CET")
    elif current_hour < NY_OPEN_CET:
        print(f"  🇺🇸 New York opens: {NY_OPEN_CET}:00 CET")
    elif current_hour < OVERLAP_END:
        print(f"  🔄 Overlap ends: {OVERLAP_END}:00 CET")
    else:
        print(f"  🌙 All sessions closed - next opens: {LONDON_OPEN_CET}:00 CET tomorrow")

if __name__ == "__main__":
    check_session_status()