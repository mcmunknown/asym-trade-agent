#!/bin/bash
echo "🔍 CHECKING 24-HOUR PERFORMANCE AT: $(date)"
echo "💰 Current Balance:"
python -c "
from bybit_client import BybitClient
client = BybitClient()
balance = client.get_account_balance()
if balance:
    current = float(balance['totalAvailableBalance'])
    profit = current - 11.35
    pct = (profit / 11.35) * 100
    print(f'   • Current Balance: \${current:.2f}')
    print(f'   • Starting Balance: \$11.35') 
    print(f'   • 24hr Profit: \${profit:.2f}')
    print(f'   • 24hr Return: {pct:.1f}%')
"
echo "📈 24-HOUR PERFORMANCE COMPLETE!"
