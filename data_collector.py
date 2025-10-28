"""
Data Collector - Market Data Aggregation
Simplified synchronous data collection for production trading
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from bybit_client import BybitClient
from config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.bybit_client = BybitClient()
        self.target_assets = Config.TARGET_ASSETS

    def collect_market_data(self, symbol: str) -> Dict:
        """Collect basic market data for a symbol"""
        try:
            # Get basic market data
            market_data = self.bybit_client.get_market_data(symbol)
            if not market_data or market_data.get('lastPrice', '0') == '0':
                logger.warning(f"No market data available for {symbol}")
                return None

            # Get funding rate
            funding_data = self.bybit_client.get_funding_rate(symbol)
            funding_rate = float(funding_data['fundingRate']) if funding_data else 0.0

            # Get open interest
            oi_data = self.bybit_client.get_open_interest(symbol)
            open_interest = float(oi_data['openInterest']) if oi_data else 0.0

            # Get kline data for technical analysis
            klines = self.bybit_client.get_kline_data(symbol, '1h', 200)

            # Combine data
            combined_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_data': market_data,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'klines': klines,
                'technical_indicators': self.calculate_technical_indicators(market_data, klines)
            }

            return combined_data

        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, market_data: Dict, klines: List[Dict]) -> Dict:
        """Calculate basic technical indicators"""
        try:
            if not klines:
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(klines)
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])

            # Calculate moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            current_price = float(market_data.get('lastPrice', 0))

            return {
                'price': current_price,
                'sma_20': float(latest['sma_20']) if pd.notna(latest['sma_20']) else None,
                'sma_50': float(latest['sma_50']) if pd.notna(latest['sma_50']) else None,
                'ema_12': float(latest['ema_12']) if pd.notna(latest['ema_12']) else None,
                'ema_26': float(latest['ema_26']) if pd.notna(latest['ema_26']) else None,
                'rsi': float(latest['rsi']) if pd.notna(latest['rsi']) else None,
                'macd': float(latest['macd']) if pd.notna(latest['macd']) else None,
                'macd_signal': float(latest['macd_signal']) if pd.notna(latest['macd_signal']) else None,
                'macd_histogram': float(latest['macd_histogram']) if pd.notna(latest['macd_histogram']) else None,
                'bb_upper': float(latest['bb_upper']) if pd.notna(latest['bb_upper']) else None,
                'bb_middle': float(latest['bb_middle']) if pd.notna(latest['bb_middle']) else None,
                'bb_lower': float(latest['bb_lower']) if pd.notna(latest['bb_lower']) else None,
                'volume_24h': float(market_data.get('volume24h', 0)),
                'price_change_24h': float(market_data.get('priceChange24h', 0)) * 100,  # Convert to percentage
                'high_24h': float(market_data.get('high24h', 0)),
                'low_24h': float(market_data.get('low24h', 0)),

                # Technical signals
                'price_above_sma_20': current_price > float(latest['sma_20']) if pd.notna(latest['sma_20']) else False,
                'price_above_sma_50': current_price > float(latest['sma_50']) if pd.notna(latest['sma_50']) else False,
                'sma_20_above_sma_50': float(latest['sma_20']) > float(latest['sma_50']) if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']) else False,
                'rsi_oversold': float(latest['rsi']) < 30 if pd.notna(latest['rsi']) else False,
                'rsi_overbought': float(latest['rsi']) > 70 if pd.notna(latest['rsi']) else False,
                'macd_bullish': float(latest['macd']) > float(latest['macd_signal']) if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']) else False,
                'price_near_bb_lower': current_price <= float(latest['bb_lower']) * 1.02 if pd.notna(latest['bb_lower']) else False,
                'price_near_bb_upper': current_price >= float(latest['bb_upper']) * 0.98 if pd.notna(latest['bb_upper']) else False,
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def collect_all_data(self) -> List[Dict]:
        """Collect data for all target assets"""
        all_data = []

        for symbol in self.target_assets:
            try:
                logger.info(f"Collecting data for {symbol}...")

                # Collect market data with technical indicators
                asset_data = self.collect_market_data(symbol)

                if asset_data:
                    # Add additional analysis for asymmetric trading
                    asset_data.update({
                        'market_regime': self._analyze_market_regime(asset_data),
                        'risk_metrics': self._calculate_risk_metrics(asset_data),
                        'catalyst_signals': self._identify_catalysts(asset_data)
                    })

                    all_data.append(asset_data)
                    logger.info(f"✅ Data collected for {symbol}")
                else:
                    logger.warning(f"❌ Failed to collect data for {symbol}")

            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                continue

        logger.info(f"Collected comprehensive data for {len(all_data)} assets")
        return all_data

    def _analyze_market_regime(self, data: Dict) -> Dict:
        """Analyze market regime for asymmetric opportunities"""
        try:
            tech = data.get('technical_indicators', {})
            market = data.get('market_data', {})

            price_change_24h = tech.get('price_change_24h', 0)
            volume_24h = tech.get('volume_24h', 0)
            rsi = tech.get('rsi', 50)

            # Simple regime analysis
            if price_change_24h > 5 and rsi > 50:
                regime = "BULLISH_STRENGTH"
            elif price_change_24h < -5 and rsi < 50:
                regime = "BEARISH_WEAKNESS"
            elif abs(price_change_24h) < 2 and 40 < rsi < 60:
                regime = "NEUTRAL_CONSOLIDATION"
            else:
                regime = "TRANSITIONAL"

            return {
                'regime': regime,
                'price_momentum': price_change_24h,
                'volume_strength': "HIGH" if volume_24h > 1000000 else "LOW",
                'rsi_level': rsi,
                'trend_alignment': self._check_trend_alignment(tech)
            }

        except Exception as e:
            logger.error(f"Error analyzing market regime: {str(e)}")
            return {'regime': 'UNKNOWN'}

    def _calculate_risk_metrics(self, data: Dict) -> Dict:
        """Calculate risk metrics for position sizing"""
        try:
            tech = data.get('technical_indicators', {})
            market = data.get('market_data', {})

            current_price = tech.get('price', 0)
            high_24h = tech.get('high_24h', current_price)
            low_24h = tech.get('low_24h', current_price)

            # Calculate volatility from 24h range
            volatility = ((high_24h - low_24h) / current_price) * 100 if current_price > 0 else 0

            return {
                'daily_volatility': volatility,
                'atr_estimate': volatility * 0.013,  # Rough ATR estimation
                'risk_level': "HIGH" if volatility > 10 else "MEDIUM" if volatility > 5 else "LOW",
                'position_size_risk': min(2.0, volatility / 5),  # Risk-based position sizing
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'risk_level': 'UNKNOWN'}

    def _identify_catalysts(self, data: Dict) -> Dict:
        """Identify potential catalysts for asymmetric moves"""
        try:
            tech = data.get('technical_indicators', {})
            market = data.get('market_data', {})

            catalysts = []

            # Technical catalysts
            if tech.get('macd_bullish'):
                catalysts.append("MACD_BULLISH_CROSS")

            if tech.get('price_near_bb_lower'):
                catalysts.append("OVERSOLD_BB_SQUEEZE")

            if tech.get('rsi_oversold'):
                catalysts.append("RSI_OVERSOLD_REVERSAL")

            # Volume catalysts
            if tech.get('volume_24h', 0) > 2000000:  # High volume
                catalysts.append("HIGH_VOLUME_BREAKOUT")

            # Price catalysts
            if tech.get('price_change_24h', 0) > 8:
                catalysts.append("MOMENTUM_ACCELERATION")

            return {
                'catalysts': catalysts,
                'catalyst_strength': "STRONG" if len(catalysts) >= 3 else "MODERATE" if len(catalysts) >= 1 else "WEAK",
                'breakout_potential': tech.get('price_above_sma_50', False)
            }

        except Exception as e:
            logger.error(f"Error identifying catalysts: {str(e)}")
            return {'catalysts': [], 'catalyst_strength': 'WEAK'}

    def _check_trend_alignment(self, tech: Dict) -> str:
        """Check if multiple indicators align for trend confirmation"""
        try:
            alignment_score = 0

            # Price above moving averages
            if tech.get('price_above_sma_20'):
                alignment_score += 1
            if tech.get('price_above_sma_50'):
                alignment_score += 1

            # SMA alignment
            if tech.get('sma_20_above_sma_50'):
                alignment_score += 1

            # MACD alignment
            if tech.get('macd_bullish'):
                alignment_score += 1

            # Determine alignment strength
            if alignment_score >= 3:
                return "STRONG_BULLISH"
            elif alignment_score >= 2:
                return "MODERATE_BULLISH"
            elif alignment_score <= 1:
                return "WEAK_BEARISH"
            else:
                return "NEUTRAL"

        except Exception as e:
            logger.error(f"Error checking trend alignment: {str(e)}")
            return "UNKNOWN"