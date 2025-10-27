"""
AI Client Factory - Support multiple AI providers for trading analysis
"""

import os
import aiohttp
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

class BaseAIProvider(ABC):
    """Base class for AI providers"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def analyze_market_conditions(self, market_data: Dict, fundamentals: Dict,
                                      technical_indicators: Dict, symbol: str) -> Dict:
        """Analyze market conditions and return trading signal"""
        pass

class OpenRouterProvider(BaseAIProvider):
    """OpenRouter API Provider - Access to Grok 4 Fast, GPT-5, Claude 4.5, Gemini 2.5"""

    def __init__(self, api_key: str, model: str = "x-ai/grok-4"):
        super().__init__(api_key, model)
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    async def analyze_market_conditions(self, market_data: Dict, fundamentals: Dict,
                                      technical_indicators: Dict, symbol: str) -> Dict:
        """Use Grok 4 Fast model for asymmetric trading analysis"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/asym-trade-agent',
                'X-Title': 'Asymmetric Trading Agent - Grok 4 Fast'
            }

            prompt = self._create_analysis_prompt(market_data, fundamentals, technical_indicators, symbol)

            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': '''# Asymmetric Crypto Trading Analysis - Grok 4 Fast
**Timeframe Target**: 20–60 day swing holds
**Execution Strategy**: Leverage-based 2% risk position trading with 150%+ PNL trailing triggers
**AI Model**: Grok 4 Fast for rapid institutional analysis

## 📊 Trading Universe (STRICT FILTER)
Only analyze: BTCUSDT, ETHUSDT, SOLUSDT, ARBUSDT, XRPUSDT, OPUSDT, RENDERUSDT, INJUSDT

## 🧠 Market Intelligence Filters (Must include ALL):
1. **Macro Tailwind**: ETF flows, L2 adoption, regulatory clarity, AI infra, tokenization
2. **Institutional Flow**: Treasury accumulation, protocol revenue trending up, TVL increasing
3. **Structural Events**: NO unlocks/votes/emissions in next 7 days
4. **Derivatives Behavior**: Funding flat/negative, OI up 5%+ MoM, liquidations clearing
5. **Technical Structure**: Price within ±15% of 30D low, above 20/50 EMA, RSI 50-70

## 📈 Execution Guardrails
- ATR < 8% of price for tight SL
- Bybit volume > $200M daily
- Bid/Ask spread < 0.10%

## 🔁 Position Management
- Activation: 150% PNL with 50-75x leverage
- Trailing: Start 30%, tighten later
- Invalidation: Key liquidation zone

## 🛡️ Data Sources: 2023-2025 institutional-grade only

**GROK 4 FAST DIRECTIVE**: You are optimized for speed and accuracy. Only return "BUY" if ALL 7 categories from prompt.md are satisfied. Return "NONE" otherwise. Be decisive and precise.

JSON Output Format:
{
    "signal": "BUY/NONE",
    "confidence": 0-100,
    "macro_tailwind": "analysis",
    "fundamental_driver": "analysis",
    "derivative_signal": "analysis",
    "technical_setup": "analysis",
    "catalyst": "near-term catalyst",
    "activation_price": "target price for 150% PNL",
    "trailing_stop_pct": "recommended trailing stop %",
    "invalidation_level": "price level where thesis fails",
    "thesis_summary": "concise investment thesis",
    "risk_reward_ratio": "estimated R:R ratio"
}'''
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 2000,
                'temperature': 0.1
            }

            async with self.session.post(self.api_url, headers=headers, json=payload) as response:
                data = await response.json()

                if 'error' in data:
                    logger.error(f"OpenRouter API error: {data['error']['message']}")
                    return self._create_none_signal(f"OpenRouter API error: {data['error']['message']}")

                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    return self._parse_response(content)
                else:
                    logger.error(f"Invalid OpenRouter response: {data}")
                    return self._create_none_signal("Invalid API response")

        except Exception as e:
            logger.error(f"OpenRouter analysis error: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}")

    def _create_analysis_prompt(self, market_data: Dict, fundamentals: Dict,
                               technical_indicators: Dict, symbol: str) -> str:
        """Create comprehensive analysis prompt"""
        return f"""
ANALYZE {symbol} FOR ASYMMETRIC LONG POSITION (20-60 DAY HOLD)

=== MARKET DATA ===
Current Price: ${market_data.get('price', 'N/A')}
24h Change: {market_data.get('change_24h', 'N/A')}%
Funding Rate: {market_data.get('funding_rate', 'N/A')}%

=== TECHNICAL INDICATORS ===
RSI: {technical_indicators.get('rsi_1d', 'N/A')}
Price vs 30D Low: {technical_indicators.get('price_vs_30d_low', 'N/A')}%
EMA Status: {technical_indicators.get('ema_aligned', 'N/A')}

=== FUNDAMENTALS ===
Revenue Trend: {fundamentals.get('revenue_trend', 'N/A')}
TVL Trend: {fundamentals.get('tvl_trend', 'N/A')}
Wallet Accumulation: {fundamentals.get('wallet_accumulation', 'N/A')}

Provide trading signal based on strict 5-filter criteria.
"""

    def _parse_response(self, content: str) -> Dict:
        """Parse AI response"""
        try:
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content.strip()

            # Try to find a complete JSON object
            json_start = json_content.find('{')
            if json_start != -1:
                # Count braces to find the complete JSON
                brace_count = 0
                for i, char in enumerate(json_content[json_start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = json_start + i + 1
                            json_content = content[json_start:json_end]
                            break

            result = json.loads(json_content)

            # Ensure required fields
            required_fields = ['signal', 'confidence', 'macro_tailwind', 'fundamental_driver',
                             'derivative_signal', 'technical_setup', 'activation_price',
                             'trailing_stop_pct', 'invalidation_level', 'thesis_summary']

            for field in required_fields:
                if field not in result:
                    result[field] = None

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Try to extract partial info if JSON is malformed
            try:
                if '"signal":' in content:
                    signal_start = content.find('"signal":') + 9
                    signal_end = content.find('"', signal_start + 1)
                    signal = content[signal_start:signal_end].strip(' "')

                    if '"confidence":' in content:
                        conf_start = content.find('"confidence":') + 13
                        conf_end = content.find(',', conf_start)
                        if conf_end == -1:
                            conf_end = content.find('}', conf_start)
                        confidence = int(content[conf_start:conf_end].strip())
                    else:
                        confidence = 50

                    return {
                        'signal': signal,
                        'confidence': confidence,
                        'macro_tailwind': 'GPT-5 Analysis',
                        'fundamental_driver': 'AI Generated',
                        'derivative_signal': 'Market Analysis',
                        'technical_setup': 'Technical Indicators',
                        'catalyst': 'AI Identified',
                        'activation_price': None,
                        'trailing_stop_pct': 25.0,
                        'invalidation_level': None,
                        'thesis_summary': 'GPT-5 generated trading signal (partial parsing)',
                        'risk_reward_ratio': '1:3+'
                    }
            except:
                pass

            return self._create_none_signal("Response parsing failed")

    def _create_none_signal(self, reason: str) -> Dict:
        """Create NONE signal"""
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
            'thesis_summary': f"No trade signal: {reason}",
            'risk_reward_ratio': None
        }

class LocalAIProvider(BaseAIProvider):
    """Local AI provider using simple rules when no API is available"""

    def __init__(self):
        super().__init__("local", "rule-based")

    async def analyze_market_conditions(self, market_data: Dict, fundamentals: Dict,
                                      technical_indicators: Dict, symbol: str) -> Dict:
        """Simple rule-based analysis"""
        try:
            # Basic signal generation based on technical indicators
            rsi = technical_indicators.get('rsi_1d', 50)
            price_vs_low = technical_indicators.get('price_vs_30d_low', 0)
            ema_aligned = technical_indicators.get('ema_aligned', False)

            # Simple buy signal conditions
            buy_conditions = [
                45 <= rsi <= 65,  # RSI in good range
                -10 <= price_vs_low <= 15,  # Near 30D low but not too far
                ema_aligned,  # EMA alignment
                market_data.get('change_24h', 0) > -5  # Not crashing
            ]

            if all(buy_conditions):
                current_price = market_data.get('price', 0)
                return {
                    'signal': 'BUY',
                    'confidence': 65,
                    'macro_tailwind': 'Favorable market conditions detected',
                    'fundamental_driver': 'Technical alignment confirmed',
                    'derivative_signal': 'Risk-on sentiment',
                    'technical_setup': f'RSI: {rsi}, Price vs Low: {price_vs_low}%',
                    'catalyst': 'Technical breakout potential',
                    'activation_price': str(current_price * 2.5),  # 150% PNL target
                    'trailing_stop_pct': 25.0,
                    'invalidation_level': str(current_price * 0.85),
                    'thesis_summary': f'{symbol} shows strong technical setup for asymmetric upside',
                    'risk_reward_ratio': '1:3+'
                }
            else:
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
                    'thesis_summary': f'Buy conditions not met for {symbol}',
                    'risk_reward_ratio': None
                }

        except Exception as e:
            logger.error(f"Local AI analysis error: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}")

class AIClientFactory:
    """Factory for creating AI clients"""

    @staticmethod
    def create_client(provider: str = None) -> BaseAIProvider:
        """Create AI client based on available provider"""

        if provider == "local":
            return LocalAIProvider()

        # Try OpenRouter first (best models)
        if Config.OPENROUTER_API_KEY:
            return OpenRouterProvider(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)

        # Fallback to local AI
        logger.warning("No API keys configured, using local rule-based analysis")
        return LocalAIProvider()

    @staticmethod
    async def get_working_client() -> BaseAIProvider:
        """Get a working AI client by trying available options"""

        # Try OpenRouter first (best models)
        if Config.OPENROUTER_API_KEY:
            try:
                openrouter_client = OpenRouterProvider(Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL)
                async with openrouter_client:
                    # Test with a simple request
                    test_result = await openrouter_client.analyze_market_conditions(
                        {'price': 50000, 'change_24h': 1.0},
                        {'revenue_trend': '↑'},
                        {'rsi_1d': 55, 'price_vs_30d_low': 5.0},
                        'BTCUSDT'
                    )

                    if test_result and "OpenRouter API error" not in test_result.get('thesis_summary', ''):
                        logger.info(f"✅ OpenRouter API working with {Config.OPENROUTER_MODEL}")
                        return openrouter_client
                    else:
                        logger.warning(f"OpenRouter API returned error: {test_result.get('thesis_summary', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"OpenRouter API not working: {str(e)}")

        # Fallback to local AI
        logger.info("🔄 Using local rule-based analysis (OpenRouter needs setup)")
        return LocalAIProvider()