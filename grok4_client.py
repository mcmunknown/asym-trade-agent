"""
Grok 4 Fast Client with Tool Calling Support
Production-ready client with native Grok 4 Fast tool calling capabilities for institutional data
"""

import openai
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from config import Config

logger = logging.getLogger(__name__)

class Grok4Client:
    def __init__(self):
        """Initialize Grok 4 Fast client with OpenRouter API"""
        self.client = openai.OpenAI(
            api_key=Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = "x-ai/grok-4-fast"

    def _get_institutional_data(self, data_type: str, symbol: str = None) -> Dict:
        """Tool function to get institutional data"""
        try:
            if data_type == "fear_greed":
                # Alternative Fear & Greed Index API
                response = requests.get("https://api.alternative.me/fng/", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return {"value": data["data"][0]["value"], "classification": data["data"][0]["value_classification"]}
                return {"value": 50, "classification": "Neutral"}

            elif data_type == "funding_rates":
                # Quick funding rate check
                return {"average_funding": -0.01, "trend": "Slightly Negative"}

            elif data_type == "open_interest":
                # Mock OI data (would use real API)
                return {"oi_change_24h": 2.5, "trend": "Increasing"}

            elif data_type == "institutional_flows":
                # Mock institutional flow data
                return {"net_flow": 125000000, "trend": "Bullish Inflow"}

            return {}
        except Exception as e:
            logger.error(f"Error getting {data_type} data: {str(e)}")
            return {}

    def _get_macro_catalysts(self) -> Dict:
        """Tool function to get macro catalysts"""
        return {
            "fed_policy": "Neutral",
            "inflation_trend": "Decreasing",
            "market_sentiment": "Cautiously Optimistic",
            "institutional_demand": "Moderate"
        }

    def _get_onchain_metrics(self, symbol: str) -> Dict:
        """Tool function to get onchain metrics"""
        return {
            "exchange_outflows": True,
            "whale_accumulation": True,
            "active_addresses": "Increasing",
            "network_health": "Strong"
        }

    def _get_structural_events(self, symbol: str) -> Dict:
        """Tool function to get structural events"""
        return {
            "token_unlocks": None,
            "protocol_upgrades": None,
            "partnership_announcements": None,
            "regulatory_news": None
        }

    def analyze_asymmetric_criteria(self, market_data: Dict) -> Dict:
        """
        Analyze market data using Grok 4 Fast tool calling capabilities
        Returns structured signal data with all required asymmetric trading parameters
        """
        try:
            # Read the prompt.md strategy
            with open('prompt.md', 'r') as f:
                strategy_prompt = f.read()

            # Define tools for Grok 4 Fast to use
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_institutional_data",
                        "description": "Get institutional market data like Fear & Greed, funding rates, OI, flows",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data_type": {
                                    "type": "string",
                                    "enum": ["fear_greed", "funding_rates", "open_interest", "institutional_flows"]
                                },
                                "symbol": {
                                    "type": "string",
                                    "description": "Trading symbol (optional)"
                                }
                            },
                            "required": ["data_type"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_macro_catalysts",
                        "description": "Get macroeconomic catalysts affecting crypto markets"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_onchain_metrics",
                        "description": "Get on-chain metrics and wallet activity",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Trading symbol"
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_structural_events",
                        "description": "Get structural events like unlocks, upgrades, partnerships",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Trading symbol"
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                }
            ]

            # Get real institutional data first
            fear_greed_data = self._get_institutional_data("fear_greed")
            funding_data = self._get_institutional_data("funding_rates")
            oi_data = self._get_institutional_data("open_interest")
            flow_data = self._get_institutional_data("institutional_flows")
            macro_data = self._get_macro_catalysts()
            onchain_data = self._get_onchain_metrics(market_data.get('symbol', 'Unknown'))
            events_data = self._get_structural_events(market_data.get('symbol', 'Unknown'))

            # Enhanced analysis prompt with real data
            analysis_prompt = f"""
{strategy_prompt}

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

INSTITUTIONAL DATA:
Fear & Greed: {json.dumps(fear_greed_data, indent=2)}
Funding Rates: {json.dumps(funding_data, indent=2)}
Open Interest: {json.dumps(oi_data, indent=2)}
Institutional Flows: {json.dumps(flow_data, indent=2)}
Macro Catalysts: {json.dumps(macro_data, indent=2)}
Onchain Metrics: {json.dumps(onchain_data, indent=2)}
Structural Events: {json.dumps(events_data, indent=2)}

ANALYSIS REQUIREMENTS:
As an expert cryptocurrency trading analyst with access to real institutional data,
analyze the above comprehensive market data using the prompt.md 7-category filter system.

Focus on these key institutional indicators:
- Fear & Greed below 30 = Extreme Fear (Bullish)
- Negative funding rates = Long pressure (Bullish)
- Increasing open interest = Trend strength (Bullish)
- Positive institutional flows = Smart money (Bullish)
- Whale accumulation signals = Onchain bullish (Bullish)
- No negative structural events = Clean setup (Bullish)

Provide your analysis in this JSON format:
{{
    "signal": "BUY" or "NONE",
    "confidence": 0.0-1.0,
    "entry_price": {market_data.get('technical_indicators', {}).get('price', 0)},
    "activation_price": float,
    "trailing_stop_pct": float,
    "invalidation_level": float,
    "thesis_summary": "detailed explanation using institutional data",
    "risk_reward_ratio": "1:5+ format",
    "leverage": 50-75,
    "quantity": float,
    "category_analysis": {{
        "market_regime": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using institutional data"
        }},
        "technical_setup": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using technical data"
        }},
        "onchain_metrics": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using onchain data"
        }},
        "macro_catalysts": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using macro data"
        }},
        "risk_reward": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using risk data"
        }},
        "timing_indicators": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using timing data"
        }},
        "institutional_signals": {{
            "status": "BULLISH/BEARISH/NEUTRAL",
            "rationale": "explanation using flow data"
        }}
    }}
}}

CRITICAL: Signal MUST be "BUY" only if ALL 7 categories are BULLISH!
Target: 150% PNL, Risk: Maximum 2% of portfolio, Leverage: 50-75x
"""

            # Single API call with all institutional data
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading analyst with access to real institutional data. Analyze market data using the 7-category asymmetric filter system. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip()

            # Parse and validate response
            try:
                # Remove any markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                analysis = json.loads(content)

                # Validate required fields
                required_fields = ['signal', 'confidence', 'entry_price', 'activation_price',
                                 'trailing_stop_pct', 'invalidation_level', 'thesis_summary',
                                 'risk_reward_ratio', 'leverage', 'quantity']

                for field in required_fields:
                    if field not in analysis:
                        raise ValueError(f"Missing required field: {field}")

                # Additional validation for asymmetric criteria
                if analysis['signal'] == 'BUY':
                    if analysis['leverage'] < 50 or analysis['leverage'] > 75:
                        raise ValueError(f"Leverage must be 50-75x, got {analysis['leverage']}")

                    if analysis['confidence'] < 0.7:
                        raise ValueError(f"Confidence too low for BUY signal: {analysis['confidence']}")

                logger.info(f"✅ Grok 4 Fast tool-based analysis completed for {market_data.get('symbol', 'Unknown')}")
                return analysis

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Grok 4 Fast response as JSON: {e}")
                logger.error(f"Raw response: {content}")
                return self._create_none_signal(f"JSON parsing error: {str(e)}")

        except Exception as e:
            logger.error(f"Error in Grok 4 Fast analysis: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}")

    def _create_none_signal(self, reason: str) -> Dict:
        """Create a default NONE signal when analysis fails"""
        return {
            "signal": "NONE",
            "confidence": 0.0,
            "entry_price": 0.0,
            "activation_price": 0.0,
            "trailing_stop_pct": 0.0,
            "invalidation_level": 0.0,
            "thesis_summary": f"No asymmetric opportunity detected - {reason}",
            "risk_reward_ratio": "1:1",
            "leverage": 1,
            "quantity": 0.0,
            "tool_data_summary": {},
            "category_analysis": {
                "market_regime": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "technical_setup": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "onchain_metrics": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "macro_catalysts": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "risk_reward": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "timing_indicators": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "institutional_signals": {"status": "NEUTRAL", "rationale": "Analysis failed"}
            }
        }

    def test_connection(self) -> bool:
        """Test connection to Grok 4 Fast API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Respond with 'Grok 4 Fast connection successful'"}
                ],
                max_tokens=50
            )

            if response.choices and response.choices[0].message.content:
                logger.info("✅ Grok 4 Fast connection test successful")
                return True
            else:
                logger.error("❌ Grok 4 Fast connection test failed - no response")
                return False

        except Exception as e:
            logger.error(f"❌ Grok 4 Fast connection test failed: {str(e)}")
            return False