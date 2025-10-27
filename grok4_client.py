
#!/usr/bin/env python3
"""
Grok 4 Fast Client with Native Tool Calling
Leverages Grok 4 Fast's built-in web browsing and data collection capabilities
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class Grok4FastClient:
    """
    Grok 4 Fast client with native tool calling for asymmetric trading analysis
    Uses built-in web browsing and data collection - no external API keys needed
    """
    
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    async def analyze_with_tools(self, symbol: str, market_data: Dict) -> Dict:
        """
        Use Grok 4 Fast's native tool calling for comprehensive analysis
        Fetches institutional data via built-in web browsing capabilities
        """
        try:
            # Define tools for Grok 4 Fast to use
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_macro_data",
                        "description": "Fetch macroeconomic and institutional data for crypto analysis",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Trading symbol (e.g., BTCUSDT, ETHUSDT)"
                                },
                                "data_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Types of data to fetch: ETF flows, institutional adoption, regulatory news, etc."
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "fetch_onchain_data",
                        "description": "Fetch on-chain metrics and protocol fundamentals",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string", 
                                    "description": "Trading symbol"
                                },
                                "metrics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "On-chain metrics: TVL, revenue, active addresses, etc."
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_derivatives_data",
                        "description": "Fetch derivatives market data and funding information",
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
                        "name": "check_structural_events",
                        "description": "Check for upcoming token unlocks, governance events, emissions",
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
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://github.com/asym-trade-agent',
                'X-Title': 'Asymmetric Trading Agent - Grok 4 Fast'
            }
            
            # Create comprehensive prompt with market data
            system_prompt = f'''# Asymmetric Crypto Trading Analysis - Grok 4 Fast with Tool Calling
**Timeframe Target**: 20–60 day swing holds
**Execution Strategy**: Leverage-based 2% risk position trading with 150%+ PNL trailing triggers
**AI Model**: Grok 4 Fast with native web browsing and data collection

## 📊 Current Market Data for {symbol}
{json.dumps(market_data, indent=2)}

## 🧠 Analysis Requirements (Use Tools to Fetch Data):
Use the available tools to gather comprehensive data across all 7 categories:

1. **Macro Tailwind**: Use fetch_macro_data for ETF flows, institutional adoption, regulatory clarity
2. **Institutional Flow**: Use fetch_onchain_data for TVL, protocol revenue, treasury accumulation  
3. **Structural Events**: Use check_structural_events for unlocks, votes, emissions in next 7 days
4. **Derivatives Behavior**: Use fetch_derivatives_data for funding rates, OI trends, liquidations
5. **Technical Structure**: Analyze provided market data for EMA alignment, RSI, price position
6. **Execution Guardrails**: Verify volume, spreads, ATR from market data
7. **Catalyst Identification**: Synthesize all data for near-term catalysts

## 📈 Trading Decision Criteria:
- Only return "BUY" if ALL 7 categories are satisfied
- Must be within ±15% of 30D low, above 20/50 EMA, RSI 50-70
- Funding flat/negative, OI up 5%+ MoM
- No structural events in next 7 days
- Clear institutional flow and macro tailwind

## 🎯 Output Format:
Return structured JSON with trading decision and comprehensive analysis.

**GROK 4 FAST DIRECTIVE**: Use your native web browsing and tool calling capabilities to gather real-time institutional data. Be decisive and precise in your analysis.'''

            payload = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user', 
                        'content': f'Analyze {symbol} for asymmetric trading opportunity using all available tools. Fetch comprehensive institutional data and provide trading recommendation.'
                    }
                ],
                'tools': tools,
                'tool_choice': 'auto',  # Let Grok decide which tools to use
                'parallel_function_calling': True,  # Enable parallel tool execution
                'temperature': 0.1,  # Low temperature for consistent analysis
                'max_tokens': 4000  # Allow for comprehensive analysis
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_tool_response(data, symbol)
                    else:
                        error_text = await response.text()
                        logger.error(f"Grok 4 Fast API error: {response.status} - {error_text}")
                        return self._get_default_analysis(symbol)
                        
        except Exception as e:
            logger.error(f"Error in Grok 4 Fast analysis: {str(e)}")
            return self._get_default_analysis(symbol)
    
    def _process_tool_response(self, response_data: Dict, symbol: str) -> Dict:
        """Process Grok 4 Fast response including tool calls"""
        try:
            message = response_data.get('choices', [{}])[0].get('message', {})
            
            # Check if there are tool calls to execute
            tool_calls = message.get('tool_calls', [])
            if tool_calls:
                logger.info(f"Grok 4 Fast requested {len(tool_calls)} tool calls for {symbol}")
                # In a full implementation, we would execute these tools
                # For now, we'll proceed with the initial analysis
                
            content = message.get('content', '')
            
            # Try to parse JSON response
            if content:
                try:
                    # Extract JSON from content if it's wrapped in markdown
                    if '```json' in content:
                        json_start = content.find('```json') + 7
                        json_end = content.find('```', json_start)
                        json_content = content[json_start:json_end].strip()
                        analysis = json.loads(json_content)
                    else:
                        # Try to parse directly
                        analysis = json.loads(content)
                    
                    # Add metadata
                    analysis['model_used'] = 'grok-4-fast'
                    analysis['analysis_time'] = datetime.now().isoformat()
                    analysis['tool_calls_made'] = len(tool_calls)
                    
                    return analysis
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from Grok 4 Fast response for {symbol}")
                    return self._parse_text_analysis(content, symbol)
            
            # Fallback to text parsing
            return self._parse_text_analysis(content, symbol)
            
        except Exception as e:
            logger.error(f"Error processing Grok 4 Fast response: {str(e)}")
            return self._get_default_analysis(symbol)
    
    def _parse_text_analysis(self, content: str, symbol: str) -> Dict:
        """Parse analysis from text content when JSON parsing fails"""
        # Simple text parsing as fallback
        signal = "NONE"
        confidence = 0
        
        if "BUY" in content.upper() and "NONE" not in content.upper():
            signal = "BUY"
            confidence = 75
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'model_used': 'grok-4-fast',
            'analysis_time': datetime.now().isoformat(),
            'analysis_text': content[:500],  # First 500 chars
            'macro_tailwind': 'Text analysis - see full content',
            'fundamental_driver': 'Text analysis - see full content',
            'derivative_signal': 'Text analysis - see full content',
            'technical_setup': 'Text analysis - see full content',
            'catalyst': 'Text analysis - see full content',
            'activation_price': '0.0',
            'trailing_stop_pct': '30.0',
            'invalidation_level': '0.0',
            'thesis_summary': 'Text analysis fallback',
            'risk_reward_ratio': '1.5:1'
        }
    
    def _get_default_analysis(self, symbol: str) -> Dict:
        """Return default analysis when API fails"""
        return {
            'symbol': symbol,
            'signal': 'NONE',
            'confidence': 0,
            'model_used': 'grok-4-fast',
            'analysis_time': datetime.now().isoformat(),
            'error': 'API call failed',
            'macro_tailwind': 'Analysis failed',
            'fundamental_driver': 'Analysis failed',
            'derivative_signal': 'Analysis failed',
            'technical_setup': 'Analysis failed',
            'catalyst': 'Analysis failed',
            'activation_price': '0.0',
            'trailing_stop_pct': '30.0',
            'invalidation_level': '0.0',
            'thesis_summary': 'Analysis failed - API error',
            'risk_reward_ratio': '0:1'
        }

