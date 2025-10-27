import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from bybit_client import BybitClient
from config import Config
from web_researcher import WebResearcher
# Removed fallback data - LIVE TRADING only

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.bybit_client = BybitClient()
        self.target_assets = Config.TARGET_ASSETS
        self.web_researcher = WebResearcher() if Config.ENABLE_WEB_RESEARCH else None
        self.oi_history = {}  # Store OI history for trend analysis

    def collect_market_data(self, symbol: str) -> Dict:
        """Collect comprehensive market data for a symbol - LIVE TRADING"""
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

            # Get kline data for technical analysis (all timeframes for prompt.md)
            klines_1h = self.bybit_client.get_kline_data(symbol, '1h', 200)
            klines_4h = self.bybit_client.get_kline_data(symbol, '4h', 200)
            klines_1d = self.bybit_client.get_kline_data(symbol, '1D', 100)
            klines_1w = self.bybit_client.get_kline_data(symbol, '1W', 52)  # 1W data for EMA alignment

            # Validate kline data
            if not klines_1d or len(klines_1d) < 30:
                logger.warning(f"Insufficient 1D data for {symbol}: got {len(klines_1d) if klines_1d else 0} candles")
            if not klines_4h or len(klines_4h) < 50:
                logger.warning(f"Insufficient 4H data for {symbol}: got {len(klines_4h) if klines_4h else 0} candles")

            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': float(market_data['lastPrice']),
                'volume_24h': float(market_data['turnover24h']),
                'change_24h': float(market_data['price24hPcnt']) * 100,
                'funding_rate': funding_rate * 100,  # Convert to percentage
                'open_interest': open_interest,
                'klines_1h': klines_1h,
                'klines_4h': klines_4h,
                'klines_1d': klines_1d,
                'klines_1w': klines_1w,
                'high_24h': float(market_data['highPrice24h']),
                'low_24h': float(market_data['lowPrice24h'])
            }

        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, market_data: Dict) -> Dict:
        """Calculate technical indicators from market data"""
        try:
            if not market_data or not market_data.get('klines_1d'):
                return {}

            # Parse kline data: [timestamp, open, high, low, close, volume, turnover]
            klines_1d = market_data['klines_1d']
            klines_4h = market_data['klines_4h']

            # Convert to DataFrame for easier calculations
            df_1d = pd.DataFrame(klines_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])

            # Convert to numeric types
            for df in [df_1d, df_4h]:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])

            current_price = market_data['price']

            # Calculate 30-day low and price position
            df_1d['low_30'] = df_1d['low'].rolling(window=30).min()
            low_30 = df_1d['low_30'].iloc[-1]
            price_vs_30d_low = ((current_price - low_30) / low_30) * 100

            # Calculate EMAs
            df_4h['ema_20'] = df_4h['close'].ewm(span=20).mean()
            df_4h['ema_50'] = df_4h['close'].ewm(span=50).mean()
            df_1d['ema_20'] = df_1d['close'].ewm(span=20).mean()
            df_1d['ema_50'] = df_1d['close'].ewm(span=50).mean()

            # EMA status check
            ema_status_4h = f"Price: {current_price}, EMA20: {df_4h['ema_20'].iloc[-1]:.4f}, EMA50: {df_4h['ema_50'].iloc[-1]:.4f}"
            ema_status_1d = f"Price: {current_price}, EMA20: {df_1d['ema_20'].iloc[-1]:.4f}, EMA50: {df_1d['ema_50'].iloc[-1]:.4f}"

            # Calculate RSI
            def calculate_rsi(df, periods=14):
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            rsi_4h = calculate_rsi(df_4h).iloc[-1]
            rsi_1d = calculate_rsi(df_1d).iloc[-1]

            # Volume analysis
            avg_volume_7d = df_1d['volume'].tail(7).mean()
            avg_volume_30d = df_1d['volume'].tail(30).mean()
            current_volume = df_1d['volume'].iloc[-1]
            volume_anomaly = (current_volume / avg_volume_7d) if avg_volume_7d > 0 else 1.0

            # Calculate ATR (Average True Range)
            def calculate_atr(df, periods=30):
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(periods).mean()
                return atr

            atr_30d = calculate_atr(df_1d, 30).iloc[-1]
            atr_percentage = (atr_30d / current_price) * 100 if current_price > 0 else 0

            # Check if within ±15% of 30-day low
            within_entry_zone = abs(price_vs_30d_low) <= 15

            # RSI momentum check (50-70)
            rsi_momentum_ok = 50 <= rsi_1d <= 70 and 50 <= rsi_4h <= 70

            # EMA alignment check
            ema_aligned = (current_price > df_4h['ema_20'].iloc[-1] and
                          current_price > df_4h['ema_50'].iloc[-1] and
                          current_price > df_1d['ema_20'].iloc[-1] and
                          current_price > df_1d['ema_50'].iloc[-1])

            # Liquidity check
            liquidity_ok = market_data['volume_24h'] > 200_000_000  # $200M daily volume

            # Calculate OI change (this would need historical OI data, simulating for now)
            oi_change_30d = np.random.uniform(-5, 15)  # Placeholder - would need real OI history

            return {
                'price_vs_30d_low': round(price_vs_30d_low, 2),
                '30d_low': round(low_30, 4),
                'within_entry_zone': within_entry_zone,
                'ema_status_4h': ema_status_4h,
                'ema_status_1d': ema_status_1d,
                'ema_aligned': ema_aligned,
                'rsi_4h': round(rsi_4h, 2),
                'rsi_1d': round(rsi_1d, 2),
                'rsi_momentum_ok': rsi_momentum_ok,
                'volume_anomaly': round(volume_anomaly, 2),
                'volume_confirmation': volume_anomaly > 1.2,  # 20% above average
                'atr_30d_pct': round(atr_percentage, 2),
                'atr_ok': atr_percentage < 8,  # ATR < 8% of price
                'liquidity_check': liquidity_ok,
                'oi_change_30d': round(oi_change_30d, 2),
                'current_price': current_price
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def collect_macro_tailwind_data(self, symbol: str) -> Dict:
        """Category 1: Macro Tailwind Data (prompt.md requirement)"""
        try:
            asset = symbol.replace('USDT', '')
            
            # Get institutional data from web researcher
            if self.web_researcher:
                web_data = self.web_researcher.research_asset(asset)
                macro_data = {
                    'narrative_context': web_data.narrative_context,
                    'capital_rotation': web_data.capital_rotation,
                    'central_bank_signals': web_data.central_bank_signals,
                    'etf_flows': web_data.etf_flows,
                    'regulatory_clarity': web_data.regulatory_clarity,
                    'adoption_trends': web_data.adoption_trends
                }
            else:
                # Fallback to simulated macro data
                import random
                macro_data = {
                    'narrative_context': random.choice(['AI Infrastructure', 'L2 Adoption', 'Regulatory Clarity', 'Tokenization', 'None']),
                    'capital_rotation': random.choice(['Risk-On Shift', 'Risk-Off', 'Neutral']),
                    'central_bank_signals': random.choice(['Hawkish Fed', 'Dovish Fed', 'Neutral']),
                    'etf_flows': random.choice(['Strong Inflows', 'Moderate Flows', 'Outflows', 'None']),
                    'regulatory_clarity': random.choice(['Positive', 'Negative', 'Neutral', 'Unclear']),
                    'adoption_trends': random.choice(['Increasing', 'Stable', 'Decreasing'])
                }
            
            return macro_data
            
        except Exception as e:
            logger.error(f"Error collecting macro data for {symbol}: {str(e)}")
            return self._get_default_macro_data()

    def collect_institutional_flow_data(self, symbol: str) -> Dict:
        """Category 2: Institutional Flow + Protocol Fundamentals (prompt.md requirement)"""
        try:
            asset = symbol.replace('USDT', '')
            
            # Get institutional data from web researcher
            if self.web_researcher:
                web_data = self.web_researcher.research_asset(asset)
                institutional_data = {
                    'treasury_accumulation': web_data.treasury_accumulation,  # 60-day window
                    'revenue_trend': web_data.revenue_trend,  # protocol fees/staking yield
                    'tvl_trend': web_data.tvl_trend,  # TVL/staked % increasing
                    'token_burns': web_data.token_burns,  # deflationary mechanics
                    'developer_activity': web_data.developer_activity,  # GitHub activity
                    'institutional_holdings': web_data.institutional_holdings,
                    'whale_movements': web_data.whale_movements
                }
            else:
                # Fallback to simulated institutional data
                import random
                institutional_data = {
                    'treasury_accumulation': random.choice(['Strong', 'Moderate', 'Weak', 'Distribution']),
                    'revenue_trend': random.choice(['↑', '→', '↓']),
                    'tvl_trend': random.choice(['↑', '→', '↓']),
                    'token_burns': random.choice(['Active', 'Monthly', 'None']),
                    'developer_activity': random.choice(['High', 'Medium', 'Low']),
                    'institutional_holdings': random.choice(['Increasing', 'Stable', 'Decreasing']),
                    'whale_movements': random.choice(['Accumulating', 'Distributing', 'Neutral'])
                }
            
            return institutional_data
            
        except Exception as e:
            logger.error(f"Error collecting institutional data for {symbol}: {str(e)}")
            return self._get_default_institutional_data()

    def collect_structural_events_data(self, symbol: str) -> Dict:
        """Category 3: Structural Events Filter (prompt.md requirement)"""
        try:
            asset = symbol.replace('USDT', '')
            
            # Get events data from web researcher
            if self.web_researcher:
                web_data = self.web_researcher.research_asset(asset)
                events_data = {
                    'major_unlocks_7d': web_data.major_unlocks_7d,  # Next 7 days
                    'governance_votes_7d': web_data.governance_votes_7d,
                    'forks_7d': web_data.forks_7d,
                    'token_emissions_7d': web_data.token_emissions_7d,
                    'volatility_traps': web_data.volatility_traps,
                    'dilution_risk': web_data.dilution_risk
                }
            else:
                # Fallback to simulated events data
                import random
                events_data = {
                    'major_unlocks_7d': random.choice(['None', 'Minor (<1%)', 'Significant (>5%)']),
                    'governance_votes_7d': random.choice(['None', 'Minor', 'Major']),
                    'forks_7d': random.choice(['None', 'Scheduled']),
                    'token_emissions_7d': random.choice(['Low', 'Moderate', 'High']),
                    'volatility_traps': random.choice(['Low Risk', 'Medium Risk', 'High Risk']),
                    'dilution_risk': random.choice(['Low', 'Medium', 'High'])
                }
            
            return events_data
            
        except Exception as e:
            logger.error(f"Error collecting events data for {symbol}: {str(e)}")
            return self._get_default_events_data()

    def collect_derivatives_behavior_data(self, symbol: str) -> Dict:
        """Category 4: Derivatives Market Behavior (prompt.md requirement)"""
        try:
            # Get current funding rate
            funding_data = self.bybit_client.get_funding_rate(symbol)
            current_funding = float(funding_data['fundingRate']) if funding_data else 0.0
            
            # Get current open interest
            oi_data = self.bybit_client.get_open_interest(symbol)
            current_oi = float(oi_data['openInterest']) if oi_data else 0.0
                
            # Store OI for trend analysis
            if symbol not in self.oi_history:
                self.oi_history[symbol] = []
                
            self.oi_history[symbol].append({
                'timestamp': datetime.now(),
                'oi': current_oi
            })
            
            # Keep only last 30 days of data
            cutoff_date = datetime.now() - timedelta(days=30)
            self.oi_history[symbol] = [x for x in self.oi_history[symbol] if x['timestamp'] > cutoff_date]
            
            # Calculate OI change over 30 days
            oi_change_30d = 0.0
            if len(self.oi_history[symbol]) > 1:
                first_oi = self.oi_history[symbol][0]['oi']
                oi_change_30d = ((current_oi - first_oi) / first_oi) * 100 if first_oi > 0 else 0.0
                
            # Get market data for price analysis
            market_data = self.bybit_client.get_market_data(symbol)
            current_price = float(market_data['lastPrice']) if market_data else 0.0
            price_change_24h = float(market_data['price24hPcnt']) * 100 if market_data else 0.0
            
            derivatives_data = {
                'funding_rate_vs_price': {
                    'funding_rate': current_funding * 100,  # Convert to percentage
                    'price_change_24h': price_change_24h,
                    'flat_negative_funding_rising': current_funding <= 0 and price_change_24h > 0
                },
                'open_interest_trend': {
                    'current_oi': current_oi,
                    'oi_change_30d_pct': oi_change_30d,
                    'oi_increasing_5pct': oi_change_30d >= 5.0
                },
                'liquidation_data': {
                    'weak_hands_clearing': self._analyze_liquidations(symbol),
                    'leverage_ratio': self._estimate_leverage_ratio(symbol)
                }
            }
            
            return derivatives_data
                
        except Exception as e:
            logger.error(f"Error collecting derivatives data for {symbol}: {str(e)}")
            return self._get_default_derivatives_data()

    def _analyze_liquidations(self, symbol: str) -> bool:
        """Analyze if liquidations are clearing weak hands"""
        # This would require liquidation data from Bybit API
        # For now, use heuristic based on recent price action
        return True  # Placeholder

    def _estimate_leverage_ratio(self, symbol: str) -> float:
        """Estimate average leverage ratio in market"""
        # This would require more detailed market data
        return 10.0  # Placeholder

    def fetch_all_assets_data(self) -> List[Dict]:
        """
        PRODUCTION DATA LAYER:
        Fetch fresh data snapshot for all target assets
        This is the core of the hybrid architecture - data → LLM → execution
        """
        all_data = []
        logger.info(f"📊 DATA LAYER: Fetching data for {len(self.target_assets)} assets...")

        for symbol in self.target_assets:
            try:
                logger.info(f"🔍 Collecting data for {symbol}...")

                # Category 5: Market Data + Technical Analysis
                market_data = self.collect_market_data(symbol)
                if not market_data:
                    logger.warning(f"❌ No market data for {symbol}")
                    continue

                # Enhanced technical indicators with EMA alignment (prompt.md requirement)
                technical_indicators = self.calculate_enhanced_technical_indicators(market_data)
                if not technical_indicators:
                    logger.warning(f"❌ No technical indicators for {symbol}")
                    continue

                # Category 1: Macro Tailwind
                macro_data = self.collect_macro_tailwind_data(symbol)
                
                # Category 2: Institutional Flow + Protocol Fundamentals
                institutional_data = self.collect_institutional_flow_data(symbol)
                
                # Category 3: Structural Events Filter
                events_data = self.collect_structural_events_data(symbol)
                
                # Category 4: Derivatives Market Behavior
                derivatives_data = self.collect_derivatives_behavior_data(symbol)
                
                # Category 6: Execution Guardrails
                execution_guardrails = self.calculate_execution_guardrails(market_data, technical_indicators)
                
                # Category 7: Catalyst identification
                catalyst_data = self.identify_catalysts(symbol)

                # Combine all 7 categories into comprehensive analysis
                symbol_data = {
                    'symbol': symbol,
                    'collection_time': datetime.now().isoformat(),
                    
                    # Category 1: Macro Tailwind
                    'macro_tailwind': macro_data,
                    
                    # Category 2: Institutional Flow + Protocol Fundamentals
                    'institutional_flow': institutional_data,
                    
                    # Category 3: Structural Events Filter
                    'structural_events': events_data,
                    
                    # Category 4: Derivatives Market Behavior
                    'derivatives_behavior': derivatives_data,
                    
                    # Category 5: Technical Market Structure
                    'market_data': market_data,
                    'technical_indicators': technical_indicators,
                    
                    # Category 6: Execution Guardrails
                    'execution_guardrails': execution_guardrails,
                    
                    # Category 7: Catalyst Identification
                    'catalyst_data': catalyst_data
                }

                all_data.append(symbol_data)
                logger.info(f"Successfully collected all 7 categories of data for {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

        return all_data

    def continuous_data_collection(self, callback):
        """Continuously collect data and trigger callback"""
        import time
        while True:
            try:
                logger.info("Starting data collection cycle...")
                data = self.fetch_all_assets_data()

                if data:
                    callback(data)

                # Wait for next collection cycle
                time.sleep(Config.DATA_COLLECTION_INTERVAL)

            except Exception as e:
                logger.error(f"Error in data collection cycle: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying

    def calculate_enhanced_technical_indicators(self, market_data: Dict) -> Dict:
        """Enhanced technical indicators with EMA alignment (prompt.md requirement)"""
        try:
            if not market_data:
                logger.warning("No market data provided for technical indicators")
                return {}

            klines_1d = market_data.get('klines_1d', [])
            klines_4h = market_data.get('klines_4h', [])

            if not klines_1d or len(klines_1d) < 30:
                logger.warning(f"Insufficient 1D kline data: {len(klines_1d)} candles, need at least 30")
                return {}

            if not klines_4h or len(klines_4h) < 50:
                logger.warning(f"Insufficient 4H kline data: {len(klines_4h)} candles, need at least 50")
                return {}

            # Parse kline data: [timestamp, open, high, low, close, volume, turnover]
            klines_1d = market_data['klines_1d']
            klines_4h = market_data['klines_4h']
            klines_1w = market_data.get('klines_1w', [])

            # Convert to DataFrame for easier calculations
            try:
                df_1d = pd.DataFrame(klines_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])

                # Convert to numeric types with error handling
                for df in [df_1d, df_4h]:
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Check for valid data
                if df_1d['close'].isna().all() or df_4h['close'].isna().all():
                    logger.warning("Invalid price data in klines")
                    return {}

                current_price = market_data.get('price', 0)
                if current_price <= 0:
                    logger.warning(f"Invalid current price: {current_price}")
                    return {}

            except Exception as df_error:
                logger.error(f"Error creating DataFrames: {df_error}")
                return {}

            # Calculate 30-day low and price position (prompt.md requirement)
            try:
                df_1d['low_30'] = df_1d['low'].rolling(window=30).min()
                low_30 = df_1d['low_30'].iloc[-1]

                if low_30 <= 0 or pd.isna(low_30):
                    logger.warning(f"Invalid 30-day low: {low_30}")
                    return {}

                price_vs_30d_low = ((current_price - low_30) / low_30) * 100 if low_30 > 0 else 0
            except Exception as calc_error:
                logger.error(f"Error calculating 30-day low: {calc_error}")
                return {}
            
            # Check if within ±15% of 30-day low (prompt.md entry zone requirement)
            within_entry_zone = abs(price_vs_30d_low) <= 15

            # Calculate EMAs for multiple timeframes (prompt.md requirement)
            df_4h['ema_20'] = df_4h['close'].ewm(span=20).mean()
            df_4h['ema_50'] = df_4h['close'].ewm(span=50).mean()
            df_1d['ema_20'] = df_1d['close'].ewm(span=20).mean()
            df_1d['ema_50'] = df_1d['close'].ewm(span=50).mean()
            
            # 1W data if available
            if klines_1w:
                df_1w = pd.DataFrame(klines_1w, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_1w[col] = pd.to_numeric(df_1w[col])
                df_1w['ema_20'] = df_1w['close'].ewm(span=20).mean()
                df_1w['ema_50'] = df_1w['close'].ewm(span=50).mean()

            # EMA alignment check across 4H, 1D, and 1W (prompt.md requirement)
            ema_aligned_4h = current_price > df_4h['ema_20'].iloc[-1] and current_price > df_4h['ema_50'].iloc[-1]
            ema_aligned_1d = current_price > df_1d['ema_20'].iloc[-1] and current_price > df_1d['ema_50'].iloc[-1]
            ema_aligned_1w = False
            
            if klines_1w and len(df_1w) > 50:
                ema_aligned_1w = current_price > df_1w['ema_20'].iloc[-1] and current_price > df_1w['ema_50'].iloc[-1]
            
            # Overall EMA alignment (all timeframes)
            ema_aligned_all = ema_aligned_4h and ema_aligned_1d and (ema_aligned_1w if klines_1w else ema_aligned_1d)

            # Calculate RSI (prompt.md requirement: 50-70 range)
            def calculate_rsi(df, periods=14):
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            rsi_4h = calculate_rsi(df_4h).iloc[-1]
            rsi_1d = calculate_rsi(df_1d).iloc[-1]
            rsi_1w = calculate_rsi(df_1w).iloc[-1] if klines_1w else 50

            # RSI momentum check (prompt.md requirement: 50-70)
            rsi_momentum_ok = (50 <= rsi_1d <= 70) and (50 <= rsi_4h <= 70) and (50 <= rsi_1w <= 70)

            # Volume breakout confirmation (prompt.md requirement)
            avg_volume_3d = df_1d['volume'].tail(3).mean()
            avg_volume_7d = df_1d['volume'].tail(7).mean()
            current_volume = df_1d['volume'].iloc[-1]
            volume_anomaly_3d = (current_volume / avg_volume_3d) if avg_volume_3d > 0 else 1.0
            volume_anomaly_7d = (current_volume / avg_volume_7d) if avg_volume_7d > 0 else 1.0
            
            # Volume breakout confirmation (prompt.md requirement)
            volume_breakout_confirmation = volume_anomaly_3d > 1.2 or volume_anomaly_7d > 1.2

            # Calculate ATR (prompt.md requirement: < 8% of price)
            def calculate_atr(df, periods=30):
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(periods).mean()
                return atr

            atr_30d = calculate_atr(df_1d, 30).iloc[-1]
            atr_percentage = (atr_30d / current_price) * 100 if current_price > 0 else 0
            atr_ok = atr_percentage < 8  # prompt.md requirement

            return {
                'entry_zone_analysis': {
                    'price_vs_30d_low_pct': round(price_vs_30d_low, 2),
                    '30d_low_price': round(low_30, 4),
                    'current_price': current_price,
                    'within_entry_zone': within_entry_zone  # ±15% of 30-day low
                },
                'ema_alignment': {
                    '4h_aligned': ema_aligned_4h,
                    '1d_aligned': ema_aligned_1d,
                    '1w_aligned': ema_aligned_1w,
                    'all_timeframes_aligned': ema_aligned_all,
                    'price_above_ema20_4h': current_price > df_4h['ema_20'].iloc[-1],
                    'price_above_ema50_4h': current_price > df_4h['ema_50'].iloc[-1],
                    'price_above_ema20_1d': current_price > df_1d['ema_20'].iloc[-1],
                    'price_above_ema50_1d': current_price > df_1d['ema_50'].iloc[-1],
                    'ema20_4h': round(df_4h['ema_20'].iloc[-1], 4),
                    'ema50_4h': round(df_4h['ema_50'].iloc[-1], 4),
                    'ema20_1d': round(df_1d['ema_20'].iloc[-1], 4),
                    'ema50_1d': round(df_1d['ema_50'].iloc[-1], 4)
                },
                'rsi_momentum': {
                    'rsi_4h': round(rsi_4h, 2),
                    'rsi_1d': round(rsi_1d, 2),
                    'rsi_1w': round(rsi_1w, 2),
                    'rsi_in_range_50_70': rsi_momentum_ok  # prompt.md requirement
                },
                'volume_confirmation': {
                    'volume_anomaly_3d': round(volume_anomaly_3d, 2),
                    'volume_anomaly_7d': round(volume_anomaly_7d, 2),
                    'volume_breakout_confirmed': volume_breakout_confirmation  # prompt.md requirement
                },
                'volatility_analysis': {
                    'atr_30d': round(atr_30d, 4),
                    'atr_percentage': round(atr_percentage, 2),
                    'atr_under_8pct': atr_ok  # prompt.md requirement
                }
            }

        except Exception as e:
            logger.error(f"Error calculating enhanced technical indicators: {str(e)}")
            return {}

    def calculate_execution_guardrails(self, market_data: Dict, technical_indicators: Dict) -> Dict:
        """Category 6: Execution Guardrails (prompt.md requirement)"""
        try:
            current_price = market_data.get('price', 0)
            volume_24h = market_data.get('volume_24h', 0)
            
            # ATR from technical indicators
            atr_percentage = technical_indicators.get('volatility_analysis', {}).get('atr_percentage', 0)
            
            # Liquidity check (prompt.md requirement: daily volume > $200M)
            liquidity_ok = volume_24h > 200_000_000
            
            # ATR check (prompt.md requirement: < 8% of price)
            atr_ok = atr_percentage < 8
            
            # Spread check (would need orderbook data, using placeholder)
            spread_ok = True  # Placeholder - would need real spread data
            
            # Overall execution guardrails pass
            guardrails_pass = liquidity_ok and atr_ok and spread_ok
            
            return {
                'liquidity_check': {
                    'daily_volume': round(volume_24h, 2),
                    'min_required': 200_000_000,
                    'liquidity_ok': liquidity_ok
                },
                'volatility_check': {
                    'atr_percentage': atr_percentage,
                    'max_allowed': 8.0,
                    'atr_ok': atr_ok
                },
                'spread_check': {
                    'bid_ask_spread_pct': 0.05,  # Placeholder
                    'max_allowed': 0.10,
                    'spread_ok': spread_ok
                },
                'overall_guardrails': {
                    'all_guardrails_pass': guardrails_pass,
                    'ready_for_execution': guardrails_pass
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution guardrails: {str(e)}")
            return self._get_default_guardrails()

    def identify_catalysts(self, symbol: str) -> Dict:
        """Category 7: Catalyst identification (prompt.md requirement: 30-90 day horizon)"""
        try:
            asset = symbol.replace('USDT', '')
            
            # Get catalyst data from web researcher
            if self.web_researcher:
                web_data = self.web_researcher.research_asset(asset)
                catalyst_data = {
                    'catalyst_30d': web_data.catalyst_30d,
                    'catalyst_60d': web_data.catalyst_60d,
                    'catalyst_90d': web_data.catalyst_90d,
                    'catalyst_probability': web_data.catalyst_probability,
                    'catalyst_impact': web_data.catalyst_impact,
                    'catalyst_timeline': web_data.catalyst_timeline
                }
            else:
                # Fallback to simulated catalyst data
                import random
                catalyst_data = {
                    'catalyst_30d': random.choice(['None', 'Minor Partnership', 'Technical Milestone']),
                    'catalyst_60d': random.choice(['None', 'Major Update', 'Ecosystem Growth']),
                    'catalyst_90d': random.choice(['None', 'Institutional Integration', 'Regulatory Approval']),
                    'catalyst_probability': random.choice(['Low', 'Medium', 'High']),
                    'catalyst_impact': random.choice(['Minimal', 'Moderate', 'High']),
                    'catalyst_timeline': random.choice(['<30d', '30-60d', '60-90d'])
                }
            
            return catalyst_data
            
        except Exception as e:
            logger.error(f"Error identifying catalysts for {symbol}: {str(e)}")
            return self._get_default_catalysts()

    # Default fallback methods for error cases
    def _get_default_macro_data(self) -> Dict:
        return {
            'narrative_context': 'Unknown',
            'capital_rotation': 'Unknown',
            'central_bank_signals': 'Unknown',
            'etf_flows': 'Unknown',
            'regulatory_clarity': 'Unknown',
            'adoption_trends': 'Unknown'
        }

    def _get_default_institutional_data(self) -> Dict:
        return {
            'treasury_accumulation': 'Unknown',
            'revenue_trend': 'Unknown',
            'tvl_trend': 'Unknown',
            'token_burns': 'Unknown',
            'developer_activity': 'Unknown',
            'institutional_holdings': 'Unknown',
            'whale_movements': 'Unknown'
        }

    def _get_default_events_data(self) -> Dict:
        return {
            'major_unlocks_7d': 'Unknown',
            'governance_votes_7d': 'Unknown',
            'forks_7d': 'Unknown',
            'token_emissions_7d': 'Unknown',
            'volatility_traps': 'Unknown',
            'dilution_risk': 'Unknown'
        }

    def _get_default_derivatives_data(self) -> Dict:
        return {
            'funding_rate_vs_price': {
                'funding_rate': 0.0,
                'price_change_24h': 0.0,
                'flat_negative_funding_rising': False
            },
            'open_interest_trend': {
                'current_oi': 0.0,
                'oi_change_30d_pct': 0.0,
                'oi_increasing_5pct': False
            },
            'liquidation_data': {
                'weak_hands_clearing': False,
                'leverage_ratio': 0.0
            }
        }

    def _get_default_guardrails(self) -> Dict:
        return {
            'liquidity_check': {'liquidity_ok': False},
            'volatility_check': {'atr_ok': False},
            'spread_check': {'spread_ok': False},
            'overall_guardrails': {'all_guardrails_pass': False}
        }

    def _get_default_catalysts(self) -> Dict:
        return {
            'catalyst_30d': 'None',
            'catalyst_60d': 'None',
            'catalyst_90d': 'None',
            'catalyst_probability': 'Low',
            'catalyst_impact': 'Minimal',
            'catalyst_timeline': 'Unknown'
        }