import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
from ai_client_factory import AIClientFactory, BaseAIProvider, LocalAIProvider
from web_researcher import WebResearcher, research_institutional_data
from config import Config

logger = logging.getLogger(__name__)

class TradingAIClient:
    """OpenRouter GPT-5 AI Client with prompt.md analysis framework"""

    def __init__(self, provider: str = None):
        self.provider = provider or "openrouter"
        self.ai_client = None
        self.web_researcher = None
        self.enable_web_research = True

    async def __aenter__(self):
        # Force OpenRouter with GPT-5, fallback to Local AI
        self.ai_client = await AIClientFactory.get_working_client()
        await self.ai_client.__aenter__()
        
        # Initialize web researcher if enabled
        if self.enable_web_research:
            config = self._get_research_config()
            self.web_researcher = WebResearcher(config)
            logger.info("Web research module initialized for institutional data")
            
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ai_client:
            await self.ai_client.__aexit__(exc_type, exc_val, exc_tb)

    async def analyze_asymmetric_criteria(self, symbol_data: Dict) -> Dict:
        """
        Complete asymmetric analysis using all 7 prompt.md categories
        Returns trading signal based on institutional-grade asymmetric criteria
        """
        if not self.ai_client:
            logger.error("AI client not initialized")
            return self._create_none_signal("AI client not available")

        try:
            # Extract all 7 categories from comprehensive symbol data
            macro_tailwind = symbol_data.get('macro_tailwind', {})
            institutional_flow = symbol_data.get('institutional_flow', {})
            structural_events = symbol_data.get('structural_events', {})
            derivatives_behavior = symbol_data.get('derivatives_behavior', {})
            market_data = symbol_data.get('market_data', {})
            technical_indicators = symbol_data.get('technical_indicators', {})
            execution_guardrails = symbol_data.get('execution_guardrails', {})
            catalyst_data = symbol_data.get('catalyst_data', {})

            # Apply prompt.md 7-category filter system
            filter_results = self._apply_prompt_filters(
                macro_tailwind,          # Category 1
                institutional_flow,      # Category 2
                structural_events,       # Category 3
                derivatives_behavior,     # Category 4
                technical_indicators,    # Category 5
                execution_guardrails,    # Category 6
                catalyst_data           # Category 7
            )

            # Only proceed if ALL categories pass (prompt.md discipline requirement)
            if not filter_results['all_categories_pass']:
                logger.info(f"{symbol_data['symbol']} FAILED filter check: {filter_results['failed_categories']}")
                return self._create_none_signal(f"Failed categories: {', '.join(filter_results['failed_categories'])}")

            # Generate signal with 150% PNL target (prompt.md requirement)
            signal = self._generate_asymmetric_signal(
                symbol_data['symbol'], 
                market_data, 
                technical_indicators,
                catalyst_data
            )

            logger.info(f"ASYMMETRIC SIGNAL GENERATED for {symbol_data['symbol']}: {signal['signal']}")
            logger.info(f"All 7 categories PASSED with confidence {signal['confidence']}%")
            
            return signal

        except Exception as e:
            logger.error(f"Asymmetric analysis error for {symbol_data.get('symbol', 'Unknown')}: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}")

    def _apply_prompt_filters(self, macro, institutional, events, derivatives, technical, guardrails, catalysts) -> Dict:
        """Apply all 7 prompt.md filter categories"""
        failed_categories = []
        
        # Category 1: Macro Tailwind filter
        macro_pass = self._check_macro_tailwind(macro)
        if not macro_pass:
            failed_categories.append('Macro Tailwind')
        
        # Category 2: Institutional Flow filter
        institutional_pass = self._check_institutional_flow(institutional)
        if not institutional_pass:
            failed_categories.append('Institutional Flow')
        
        # Category 3: Structural Events filter
        events_pass = self._check_structural_events(events)
        if not events_pass:
            failed_categories.append('Structural Events')
        
        # Category 4: Derivatives Behavior filter
        derivatives_pass = self._check_derivatives_behavior(derivatives)
        if not derivatives_pass:
            failed_categories.append('Derivatives Behavior')
        
        # Category 5: Technical Market Structure filter
        technical_pass = self._check_technical_structure(technical)
        if not technical_pass:
            failed_categories.append('Technical Structure')
        
        # Category 6: Execution Guardrails filter
        guardrails_pass = self._check_execution_guardrails(guardrails)
        if not guardrails_pass:
            failed_categories.append('Execution Guardrails')
        
        # Category 7: Catalyst filter
        catalyst_pass = self._check_catalysts(catalysts)
        if not catalyst_pass:
            failed_categories.append('Catalyst')
        
        all_pass = len(failed_categories) == 0
        
        return {
            'all_categories_pass': all_pass,
            'failed_categories': failed_categories,
            'category_results': {
                'macro_tailwind': macro_pass,
                'institutional_flow': institutional_pass,
                'structural_events': events_pass,
                'derivatives_behavior': derivatives_pass,
                'technical_structure': technical_pass,
                'execution_guardrails': guardrails_pass,
                'catalysts': catalyst_pass
            }
        }

    def _check_macro_tailwind(self, macro: Dict) -> bool:
        """Category 1: Macro Tailwind check (prompt.md requirement)"""
        if macro.get('narrative_context') in ['None', 'Unknown']:
            return False
        if macro.get('capital_rotation') == 'Risk-Off':
            return False
        if macro.get('regulatory_clarity') == 'Negative':
            return False
        return True

    def _check_institutional_flow(self, institutional: Dict) -> bool:
        """Category 2: Institutional Flow check (prompt.md requirement)"""
        if institutional.get('treasury_accumulation') in ['Weak', 'Distribution', 'Unknown']:
            return False
        if institutional.get('revenue_trend') in ['↓', 'Unknown']:
            return False
        if institutional.get('developer_activity') in ['Low', 'Unknown']:
            return False
        return True

    def _check_structural_events(self, events: Dict) -> bool:
        """Category 3: Structural Events check (prompt.md requirement)"""
        if events.get('major_unlocks_7d') in ['Significant (>5%)', 'Unknown']:
            return False
        if events.get('governance_votes_7d') == 'Major':
            return False
        if events.get('dilution_risk') in ['High', 'Unknown']:
            return False
        return True

    def _check_derivatives_behavior(self, derivatives: Dict) -> bool:
        """Category 4: Derivatives Behavior check (prompt.md requirement)"""
        funding_data = derivatives.get('funding_rate_vs_price', {})
        oi_data = derivatives.get('open_interest_trend', {})
        
        if not funding_data.get('flat_negative_funding_rising', False):
            return False
        if not oi_data.get('oi_increasing_5pct', False):
            return False
        return True

    def _check_technical_structure(self, technical: Dict) -> bool:
        """Category 5: Technical Market Structure check (prompt.md requirement)"""
        entry_zone = technical.get('entry_zone_analysis', {})
        ema_alignment = technical.get('ema_alignment', {})
        rsi_momentum = technical.get('rsi_momentum', {})
        volume_confirmation = technical.get('volume_confirmation', {})
        
        # Check all technical criteria from prompt.md
        if not entry_zone.get('within_entry_zone', False):
            return False
        if not ema_alignment.get('all_timeframes_aligned', False):
            return False
        if not rsi_momentum.get('rsi_in_range_50_70', False):
            return False
        if not volume_confirmation.get('volume_breakout_confirmed', False):
            return False
        
        return True

    def _check_execution_guardrails(self, guardrails: Dict) -> bool:
        """Category 6: Execution Guardrails check (prompt.md requirement)"""
        if not guardrails.get('liquidity_check', {}).get('liquidity_ok', False):
            return False
        if not guardrails.get('volatility_check', {}).get('atr_ok', False):
            return False
        if not guardrails.get('spread_check', {}).get('spread_ok', False):
            return False
        if not guardrails.get('overall_guardrails', {}).get('all_guardrails_pass', False):
            return False
        return True

    def _check_catalysts(self, catalysts: Dict) -> bool:
        """Category 7: Catalyst check (prompt.md requirement: 30-90 day horizon)"""
        if catalysts.get('catalyst_30d') in ['None', 'Unknown']:
            return False
        if catalysts.get('catalyst_probability') in ['Low', 'Unknown']:
            return False
        if catalysts.get('catalyst_impact') in ['Minimal', 'Unknown']:
            return False
        return True

    def _generate_asymmetric_signal(self, symbol: str, market_data: Dict, technical_indicators: Dict, catalysts: Dict) -> Dict:
        """Generate asymmetric trading signal with 150% PNL target (prompt.md requirement)"""
        current_price = market_data.get('price', 0)
        
        # Calculate activation price for 150% PNL with 50-75x leverage
        # For $3 position at 75x leverage = $225 position, need 150% PNL = $4.5 profit
        leverage = 75
        position_value = Config.DEFAULT_TRADE_SIZE * leverage  # $3 * 75 = $225
        target_pnl = Config.DEFAULT_TRADE_SIZE * 1.5  # $4.5 target profit
        
        # Calculate activation price (where 150% PNL achieved)
        # PNL% = (Exit Price - Entry Price) / Entry Price * Leverage
        # 150% = (Exit - Entry) / Entry * 75
        # Exit = Entry * (1 + 150%/75) = Entry * 3.0
        activation_price = current_price * 3.0
        
        # Calculate trailing stop (start at 30%, tighten later)
        trailing_stop_pct = 30.0
        
        # Calculate invalidation level (liquidation protection)
        invalidation_level = current_price * 0.98  # 2% below entry for high leverage
        
        # Build thesis summary from catalysts
        catalyst_30d = catalysts.get('catalyst_30d', 'Strong catalyst identified')
        catalyst_60d = catalysts.get('catalyst_60d', 'Continued momentum expected')
        catalyst_timeline = catalysts.get('catalyst_timeline', '30-60d')
        
        thesis_summary = f"Asymmetric opportunity with {catalyst_30d} catalyst. Timeline: {catalyst_timeline}. Expected 150% PNL with {leverage}x leverage."

        return {
            'signal': 'BUY',
            'confidence': 85,  # High confidence when all 7 categories pass
            'entry_price': current_price,
            'activation_price': activation_price,
            'trailing_stop_pct': trailing_stop_pct,
            'invalidation_level': invalidation_level,
            'thesis_summary': thesis_summary,
            'risk_reward_ratio': '1:5+',  # prompt.md asymmetric requirement
            'leverage': leverage,
            'quantity': position_value / current_price if current_price > 0 else 0,
            'categories_passed': 7,  # All categories passed
            'target_pnl': target_pnl,
            'hold_timeframe': '20-60 days'  # prompt.md timeframe
        }

    def _enhance_market_data(self, market_data: Dict) -> Dict:
        """Enhance market data with OpenRouter GPT-5 requirements"""
        enhanced = market_data.copy()
        
        # Add Bybit-specific data fields for prompt analysis
        enhanced.update({
            'bybit_volume': enhanced.get('volume_24h', 0),
            'bybit_funding_rate': enhanced.get('funding_rate', 0),
            'bybit_open_interest': enhanced.get('open_interest', 0),
            'liquidation_level': enhanced.get('liquidation_level', 0),
            'spread_percentage': enhanced.get('spread_percentage', 0),
            'timestamp': enhanced.get('timestamp', datetime.now().isoformat())
        })
        
        return enhanced

    async def _enhance_fundamentals(self, fundamentals: Dict, symbol: str = None) -> Dict:
        """Enhance fundamentals data for institutional analysis with web research"""
        enhanced = fundamentals.copy()
        
        # Add institutional-grade data fields from existing data
        enhanced.update({
            'treasury_accumulation': enhanced.get('wallet_accumulation', 'Unknown'),
            'revenue_trend': enhanced.get('revenue_trend', 'Unknown'),
            'tvl_trend': enhanced.get('tvl_trend', 'Unknown'),
            'developer_activity': enhanced.get('developer_activity', 'Unknown'),
            'tokenomics_changes': enhanced.get('tokenomics', 'Unknown'),
            'upcoming_events': enhanced.get('events_7d', 'None'),
            'liquidity_metrics': enhanced.get('liquidity', 'Unknown')
        })
        
        # Integrate web research data if available and enabled
        if self.enable_web_research and self.web_researcher and symbol:
            try:
                web_data = await self._get_web_research_data(symbol)
                enhanced.update(web_data)
                logger.info(f"Enhanced fundamentals for {symbol} with web research data")
            except Exception as e:
                logger.warning(f"Web research enhancement failed for {symbol}: {e}")
        
        return enhanced
    
    async def _get_web_research_data(self, symbol: str) -> Dict[str, Any]:
        """Get web research data for the specified symbol"""
        if not self.web_researcher:
            return {}
        
        try:
            # Extract asset symbol from trading pair (remove USDT suffix)
            asset = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            
            research_data = await self.web_researcher.research_asset(asset)
            return research_data.to_dict()
        except Exception as e:
            logger.error(f"Web research failed for {symbol}: {e}")
            return {}
    
    def _get_research_config(self) -> Dict[str, str]:
        """Get configuration for web research from environment/config"""
        return {
            'messari_api_key': getattr(Config, 'MESSARI_API_KEY', None),
            'glassnode_api_key': getattr(Config, 'GLASSNODE_API_KEY', None),
            'tokenterminal_api_key': getattr(Config, 'TOKENTERMINAL_API_KEY', None),
            'arkham_api_key': getattr(Config, 'ARKHAM_API_KEY', None)
        }
    
    async def research_all_assets(self, assets: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Research all target assets or specified assets"""
        if not assets:
            assets = Config.TARGET_ASSETS
        
        # Remove USDT suffix to get asset symbols
        asset_symbols = [asset.replace('USDT', '') for asset in assets]
        
        if not self.web_researcher:
            logger.error("Web researcher not initialized")
            return {}
        
        try:
            research_results = await self.web_researcher.research_multiple_assets(asset_symbols)
            
            # Convert to format expected by TradingAIClient
            formatted_results = {}
            for asset, research_data in research_results.items():
                formatted_results[asset] = research_data.to_dict()
            
            logger.info(f"Completed web research for {len(asset_symbols)} assets")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Web research batch failed: {e}")
            return {}

    def _enhance_technical(self, technical_indicators: Dict) -> Dict:
        """Enhance technical data for prompt analysis"""
        enhanced = technical_indicators.copy()
        
        # Add technical analysis fields for prompt.md criteria
        enhanced.update({
            'rsi_4h': enhanced.get('rsi_4h', enhanced.get('rsi_1d', 50)),
            'rsi_1d': enhanced.get('rsi_1d', 50),
            'rsi_1w': enhanced.get('rsi_1w', 50),
            'ema_20_4h': enhanced.get('ema_20_4h', False),
            'ema_20_1d': enhanced.get('ema_20_1d', False),
            'ema_20_1w': enhanced.get('ema_20_1w', False),
            'ema_50_4h': enhanced.get('ema_50_4h', False),
            'ema_50_1d': enhanced.get('ema_50_1d', False),
            'ema_50_1w': enhanced.get('ema_50_1w', False),
            'atr_30d': enhanced.get('atr_30d', 0),
            'volume_3d_anomaly': enhanced.get('volume_spike', False),
            'volume_7d_anomaly': enhanced.get('volume_breakout', False)
        })
        
        return enhanced

    async def batch_analyze_symbols(self, symbols_data: List[Dict]) -> List[Dict]:
        """Analyze multiple symbols in parallel"""
        if not self.ai_client:
            logger.error("AI client not initialized for batch analysis")
            return [self._create_none_signal("AI client not available") for _ in symbols_data]

        tasks = []

        for symbol_data in symbols_data:
            task = self.analyze_market_conditions(
                symbol_data['market_data'],
                symbol_data['fundamentals'],
                symbol_data['technical_indicators'],
                symbol_data['symbol']
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for {symbols_data[i]['symbol']}: {str(result)}")
                processed_results.append(self._create_none_signal(f"Analysis error: {str(result)}"))
            else:
                processed_results.append(result)

        return processed_results

    def _create_none_signal(self, reason: str) -> Dict:
        """Create a NONE signal when criteria are not met"""
        return {
            'signal': 'NONE',
            'confidence': 0,
            'macro_tailwind': None,
            'fundamental_driver': None,
            'derivative_signal': None,
            'technical_setup': None,
            'catalyst': None,
            'activation_price': None,
            'trailing_stop_pct': None,
            'invalidation_level': None,
            'thesis_summary': f"No trade signal generated: {reason}",
            'risk_reward_ratio': None
        }

# Keep GLMClient as alias for backward compatibility
GLMClient = TradingAIClient