"""
Collect Historical Trade Data from Bybit
========================================

Extract last 50 trades with full metadata for analysis.
Calculate actual win rate, avg win/loss, hold times, etc.

Output: historical_trades.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_client import BybitClient
from config import Config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_trade_history(symbols=None, limit=50):
    """Collect trade history from Bybit"""
    
    if symbols is None:
        symbols = Config.TARGET_ASSETS
    
    client = BybitClient()
    all_trades = []
    
    for symbol in symbols:
        logger.info(f"Fetching trades for {symbol}...")
        
        try:
            # Get closed P&L records (completed trades)
            response = client.client.get_closed_pnl(
                category='linear',
                symbol=symbol,
                limit=limit
            )
            
            if response and response.get('retCode') == 0:
                trades = response.get('result', {}).get('list', [])
                logger.info(f"  Found {len(trades)} trades for {symbol}")
                
                for trade in trades:
                    all_trades.append({
                        'symbol': symbol,
                        'order_id': trade.get('orderId', ''),
                        'entry_time': pd.to_datetime(int(trade.get('createdTime', 0)), unit='ms'),
                        'exit_time': pd.to_datetime(int(trade.get('updatedTime', 0)), unit='ms'),
                        'side': trade.get('side', ''),
                        'entry_price': float(trade.get('avgEntryPrice', 0)),
                        'exit_price': float(trade.get('avgExitPrice', 0)),
                        'quantity': float(trade.get('closedSize', 0)),
                        'leverage': float(trade.get('leverage', 1)),
                        'pnl': float(trade.get('closedPnl', 0)),
                        'pnl_percent': float(trade.get('closedPnl', 0)) / float(trade.get('avgEntryPrice', 1)) * 100,
                        'fees': float(trade.get('totalFee', 0)),
                        'net_pnl': float(trade.get('closedPnl', 0)) - float(trade.get('totalFee', 0)),
                    })
                    
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            continue
    
    if not all_trades:
        logger.warning("No trades found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_trades)
    
    # Calculate additional metrics
    df['hold_duration_seconds'] = (df['exit_time'] - df['entry_time']).dt.total_seconds()
    df['win'] = df['net_pnl'] > 0
    df['price_move_percent'] = ((df['exit_price'] - df['entry_price']) / df['entry_price'] * 100).abs()
    
    # Sort by entry time
    df = df.sort_values('entry_time', ascending=False)
    
    return df

def analyze_trades(df):
    """Analyze trade statistics"""
    
    if df is None or len(df) == 0:
        print("No trades to analyze")
        return
    
    print("\n" + "="*70)
    print("TRADE HISTORY ANALYSIS")
    print("="*70)
    
    # Basic stats
    total_trades = len(df)
    wins = df[df['win']].shape[0]
    losses = df[~df['win']].shape[0]
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Wins: {wins}")
    print(f"   Losses: {losses}")
    print(f"   Win Rate: {win_rate:.1%}")
    
    # PnL stats
    total_pnl = df['net_pnl'].sum()
    avg_win = df[df['win']]['net_pnl'].mean() if wins > 0 else 0
    avg_loss = df[~df['win']]['net_pnl'].mean() if losses > 0 else 0
    
    print(f"\n💰 PnL Statistics:")
    print(f"   Total PnL: ${total_pnl:.2f}")
    print(f"   Average Win: ${avg_win:.3f}")
    print(f"   Average Loss: ${avg_loss:.3f}")
    print(f"   Avg Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   Avg Win/Loss Ratio: N/A")
    
    # Expected Value
    ev = win_rate * avg_win + (1 - win_rate) * avg_loss
    print(f"   Expected Value: ${ev:.4f} per trade ({ev/10*100:.2f}% per $10)")
    
    # Break-even win rate
    if avg_loss != 0:
        breakeven_wr = abs(avg_loss) / (avg_win + abs(avg_loss))
        print(f"   Break-even Win Rate: {breakeven_wr:.1%}")
        print(f"   Current vs Break-even: {(win_rate - breakeven_wr)*100:+.1f} percentage points")
    
    # Time stats
    avg_hold_time = df['hold_duration_seconds'].mean()
    median_hold_time = df['hold_duration_seconds'].median()
    
    print(f"\n⏱️  Hold Time Statistics:")
    print(f"   Average: {avg_hold_time:.0f} seconds ({avg_hold_time/60:.1f} minutes)")
    print(f"   Median: {median_hold_time:.0f} seconds ({median_hold_time/60:.1f} minutes)")
    print(f"   Min: {df['hold_duration_seconds'].min():.0f} seconds")
    print(f"   Max: {df['hold_duration_seconds'].max():.0f} seconds")
    
    # Leverage stats
    print(f"\n⚙️  Leverage Statistics:")
    print(f"   Average Leverage: {df['leverage'].mean():.1f}x")
    print(f"   Median Leverage: {df['leverage'].median():.1f}x")
    print(f"   Max Leverage: {df['leverage'].max():.1f}x")
    
    # Per-symbol breakdown
    print(f"\n📈 Per-Symbol Breakdown:")
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        symbol_wins = symbol_df[symbol_df['win']].shape[0]
        symbol_total = len(symbol_df)
        symbol_wr = symbol_wins / symbol_total if symbol_total > 0 else 0
        symbol_pnl = symbol_df['net_pnl'].sum()
        print(f"   {symbol}: {symbol_total} trades, WR={symbol_wr:.1%}, PnL=${symbol_pnl:.2f}")
    
    # Statistical significance
    print(f"\n📐 Statistical Significance:")
    std_error = np.sqrt(win_rate * (1 - win_rate) / total_trades)
    ci_95 = 1.96 * std_error
    print(f"   Win Rate 95% CI: [{(win_rate - ci_95):.1%}, {(win_rate + ci_95):.1%}]")
    print(f"   Sample Size: {total_trades} trades")
    
    if total_trades < 30:
        print(f"   ⚠️  WARNING: Sample size too small (need 30+ trades for statistical validity)")
    
    print("\n" + "="*70)
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'ev': ev,
        'total_trades': total_trades,
        'total_pnl': total_pnl
    }

if __name__ == "__main__":
    print("Collecting trade history from Bybit...")
    print("This may take a few moments...\n")
    
    df = collect_trade_history(limit=100)  # Get more than 50 for safety
    
    if df is not None and len(df) > 0:
        # Save to CSV
        output_file = 'historical_trades.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(df)} trades to {output_file}")
        
        # Analyze
        stats = analyze_trades(df)
        
        # Save summary
        summary_file = 'trade_analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Trade Analysis Summary\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"="*70 + "\n\n")
            f.write(f"Total Trades: {stats['total_trades']}\n")
            f.write(f"Win Rate: {stats['win_rate']:.2%}\n")
            f.write(f"Average Win: ${stats['avg_win']:.4f}\n")
            f.write(f"Average Loss: ${stats['avg_loss']:.4f}\n")
            f.write(f"Expected Value: ${stats['ev']:.4f}\n")
            f.write(f"Total PnL: ${stats['total_pnl']:.2f}\n")
        
        print(f"✅ Saved summary to {summary_file}")
        
    else:
        print("\n❌ No trade data collected. Check Bybit API credentials.")
