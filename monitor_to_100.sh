#!/bin/bash
echo "🔍 CHECKING PROGRESS TOWARD $100 TARGET: $(date)"

python -c "
from bybit_client import BybitClient

client = BybitClient()
balance = client.get_account_balance()

if balance:
    current_equity = float(balance['totalEquity'])
    profit = current_equity - 11.35  # From starting balance
    profit_pct = (profit / 11.35) * 100
    target_remaining = 100 - profit
    target_pct = profit_pct
    
    print(f'💰 Current Equity: \${current_equity:.2f}')
    print(f'📊 Total Profit: \${profit:+.2f} ({profit_pct:+.1f}%)')
    print(f'🎯 Target \$100: \${target_remaining:+.2f} remaining ({target_pct:.1f}%)')
    
    if current_equity >= 100:
        print('🎉 \$100 TARGET REACHED!')
        print('🚀 STOP MONITORING - MISSION ACCOMPLISHED!')
    else:
        print(f'📈 Need additional \${target_remaining:.2f} to reach \$100')
        print(f'🚀 System continues trading toward goal...')
"

echo ""
echo "🎯 SYSTEM STATUS CHECK:"
ps aux | grep "live_calculus_trader.py" | grep -v grep | wc -l | xargs echo "   • Running processes:" || echo "   ❌ SYSTEM STOPPED!"

echo ""
echo "🚀 MONITORING CONTINUES UNTIL \$100 TARGET REACHED..."
