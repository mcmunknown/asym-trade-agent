"""
Multi-Model AI Trading Team
Consensus-based trading system using multiple AI models (Grok 4 Fast, Qwen3-Max, DeepSeek V3.1 Terminus)
that collaborate to generate higher-quality signals through majority voting.

Models used:
- x-ai/grok-4-fast (Grok 4 Fast)
- qwen/qwen3-max (Qwen3-Max)
- deepseek/deepseek-chat-v3.1 (DeepSeek Chat V3.1)
"""

import asyncio
import json
import logging
import openai
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class ModelSignal:
    """Individual model's trading signal"""
    model_name: str
    signal: str  # BUY/NONE
    confidence: float
    entry_price: float
    activation_price: float
    trailing_stop_pct: float
    invalidation_level: float
    thesis_summary: str
    risk_reward_ratio: str
    leverage: int
    quantity: float
    category_analysis: Dict
    reasoning: str  # Model's specific reasoning

@dataclass
class ConsensusResult:
    """Consensus result from multiple models"""
    final_signal: str  # BUY/NONE
    consensus_votes: Dict[str, str]  # Model name -> signal
    confidence_avg: float
    thesis_combined: str
    recommended_params: Dict
    disagreement_details: Optional[List[str]] = None

class BaseModelClient(ABC):
    """Abstract base class for all AI model clients"""

    def __init__(self, model_name: str, model_id: str):
        self.model_name = model_name
        self.model_id = model_id
        self.client = openai.OpenAI(
            api_key=Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    @abstractmethod
    async def analyze_market(self, market_data: Dict, institutional_data: Dict) -> ModelSignal:
        """Analyze market data and return trading signal"""
        pass

    def _get_institutional_data(self, data_type: str, symbol: str = None) -> Dict:
        """Get institutional data - same across all models"""
        try:
            if data_type == "fear_greed":
                import requests
                response = requests.get("https://api.alternative.me/fng/", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return {"value": data["data"][0]["value"], "classification": data["data"][0]["value_classification"]}
                return {"value": 50, "classification": "Neutral"}
            elif data_type == "funding_rates":
                return {"average_funding": -0.01, "trend": "Slightly Negative"}
            elif data_type == "open_interest":
                return {"oi_change_24h": 2.5, "trend": "Increasing"}
            elif data_type == "institutional_flows":
                return {"net_flow": 125000000, "trend": "Bullish Inflow"}
            return {}
        except Exception as e:
            logger.error(f"Error getting {data_type} data: {str(e)}")
            return {}

    def _read_strategy_prompt(self) -> str:
        """Read the prompt.md strategy file"""
        try:
            with open('prompt.md', 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading strategy prompt: {str(e)}")
            return "Default trading strategy: Analyze market data for asymmetric opportunities."

    def _create_none_signal(self, reason: str, model_name: str) -> ModelSignal:
        """Create a default NONE signal when analysis fails"""
        return ModelSignal(
            model_name=model_name,
            signal="NONE",
            confidence=0.0,
            entry_price=0.0,
            activation_price=0.0,
            trailing_stop_pct=0.0,
            invalidation_level=0.0,
            thesis_summary=f"No asymmetric opportunity detected - {reason}",
            risk_reward_ratio="1:1",
            leverage=1,
            quantity=0.0,
            category_analysis={
                "market_regime": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "technical_setup": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "onchain_metrics": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "macro_catalysts": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "risk_reward": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "timing_indicators": {"status": "NEUTRAL", "rationale": "Analysis failed"},
                "institutional_signals": {"status": "NEUTRAL", "rationale": "Analysis failed"}
            },
            reasoning=f"Analysis failed: {reason}"
        )

class Grok4FastClient(BaseModelClient):
    """Grok 4 Fast model client - existing implementation"""

    def __init__(self):
        super().__init__("Grok 4 Fast", "x-ai/grok-4-fast")

    async def analyze_market(self, market_data: Dict, institutional_data: Dict) -> ModelSignal:
        """Analyze market using Grok 4 Fast"""
        try:
            strategy_prompt = self._read_strategy_prompt()

            # Grok 4 Fast specific prompt - optimized for speed and real-time analysis
            analysis_prompt = f"""
{strategy_prompt}

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

INSTITUTIONAL DATA:
{json.dumps(institutional_data, indent=2)}

GROK 4 FAST ANALYSIS REQUIREMENTS:
As X AI's real-time analysis model, provide rapid but thorough assessment using:
- Speed-focused technical pattern recognition
- Real-time momentum analysis
- Institutional flow interpretation
- Immediate catalyst identification

Provide your analysis in this JSON format:
{{
    "signal": "BUY", "SELL", or "NONE",
    "confidence": 0.0-1.0,
    "entry_price": {market_data.get('technical_indicators', {}).get('price', 0)},
    "activation_price": float,
    "trailing_stop_pct": float,
    "invalidation_level": float,
    "thesis_summary": "concise but comprehensive analysis",
    "risk_reward_ratio": "1:5+ format (BUY) or 1:1.5-1:3 format (SELL)",
    "leverage": maximum available,
    "quantity": float,
    "reasoning": "Grok 4 Fast specific reasoning highlighting speed and real-time insights",
    "category_analysis": {{
        "market_regime": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}},
        "technical_setup": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}},
        "onchain_metrics": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}},
        "macro_catalysts": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}},
        "risk_reward": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}},
        "timing_indicators": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}},
        "institutional_signals": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "explanation"}}
    }}
}}

CRITICAL CONSENSUS RULES:
- BUY Signal: ALL 7 categories must be BULLISH (RSI 30-50, 1000% PNL targets, 3-day holds)
- SELL Signal: ALL 7 categories must be BEARISH (RSI 70-85, 300-500% PNL targets, 1-2 day holds)
- NONE Signal: Mixed categories or unclear setup
- Crypto Reality: Shorts require quicker exits due to violent bounces!

Focus on speed and actionable insights for immediate trading decisions.
"""

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are Grok 4 Fast, X AI's rapid analysis model. Provide quick but thorough trading signals using real-time data. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            # Enhanced JSON parsing with error recovery
            try:
                # Parse response
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                if not content:
                    raise ValueError("Empty response from Grok 4 Fast")

                # Try to parse JSON directly first
                analysis = json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error for {self.model_name}: {e}")
                logger.warning(f"Raw content (first 500 chars): {content[:500]}...")

                # Enhanced JSON recovery with multiple strategies
                analysis = self._enhanced_json_recovery(content, e, market_data)

            # Validate required fields
            required_fields = ['signal', 'confidence', 'entry_price', 'activation_price',
                             'trailing_stop_pct', 'invalidation_level', 'thesis_summary',
                             'risk_reward_ratio', 'leverage', 'quantity', 'reasoning']

            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            # Validate asymmetric criteria
            if analysis['signal'] == 'BUY':
                if analysis['leverage'] < 1:
                    raise ValueError(f"Leverage must be at least 1x, got {analysis['leverage']}")

            logger.info(f"✅ Grok 4 Fast analysis completed for {market_data.get('symbol', 'Unknown')}")

            return ModelSignal(
                model_name=self.model_name,
                signal=analysis['signal'],
                confidence=analysis['confidence'],
                entry_price=analysis['entry_price'],
                activation_price=analysis['activation_price'],
                trailing_stop_pct=analysis['trailing_stop_pct'],
                invalidation_level=analysis['invalidation_level'],
                thesis_summary=analysis['thesis_summary'],
                risk_reward_ratio=analysis['risk_reward_ratio'],
                leverage=analysis['leverage'],
                quantity=analysis['quantity'],
                category_analysis=analysis.get('category_analysis', {}),
                reasoning=analysis.get('reasoning', 'No specific reasoning provided')
            )

        except Exception as e:
            logger.error(f"Error in Grok 4 Fast analysis: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}", self.model_name)

    def _enhanced_json_recovery(self, content: str, original_error: Exception, market_data: Dict) -> Dict:
        """Enhanced JSON recovery with multiple fallback strategies for Grok 4 Fast responses"""
        import re

        logger.info(f"🔧 Attempting enhanced JSON recovery for {self.model_name}")

        # Strategy 1: Extract JSON from text (existing method)
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                fixed_content = json_match.group(0)
                analysis = json.loads(fixed_content)
                logger.info(f"✅ Strategy 1 successful: Extracted JSON for {self.model_name}")
                return self._validate_and_fix_fields(analysis, market_data)
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {e}")

        # Strategy 2: Fix common comma issues (specifically line 22 column 2 error)
        if "line 22 column 2" in str(original_error) or "Expecting ',' delimiter" in str(original_error):
            try:
                # Add missing commas after common field patterns
                fixed_content = content
                patterns_to_fix = [
                    r'("invalidation_level":\s*[\d.]+)\s*\n',  # Add comma after invalidation_level
                    r'("thesis_summary":\s*"[^"]*")\s*\n',      # Add comma after thesis_summary
                    r'("risk_reward_ratio":\s*"[^"]*")\s*\n',  # Add comma after risk_reward_ratio
                    r'("leverage":\s*[\d.]+)\s*\n',             # Add comma after leverage
                    r'("quantity":\s*[\d.]+)\s*\n',             # Add comma after quantity
                ]

                for pattern in patterns_to_fix:
                    fixed_content = re.sub(pattern, r'\1,\n', fixed_content, flags=re.MULTILINE)

                # Try parsing the fixed JSON
                analysis = json.loads(fixed_content)
                logger.info(f"✅ Strategy 2 successful: Fixed comma delimiter error for {self.model_name}")
                return self._validate_and_fix_fields(analysis, market_data)
            except Exception as e:
                logger.debug(f"Strategy 2 failed: {e}")

        # Strategy 3: Complete incomplete JSON (add missing closing braces/brackets)
        try:
            fixed_content = content

            # Count braces and brackets
            open_braces = content.count('{')
            close_braces = content.count('}')
            open_brackets = content.count('[')
            close_brackets = content.count(']')

            # Add missing closing elements
            fixed_content += '}' * (open_braces - close_braces)
            fixed_content += ']' * (open_brackets - close_brackets)

            # Remove trailing commas before closing braces/brackets
            fixed_content = re.sub(r',(\s*[}\]])', r'\1', fixed_content)

            analysis = json.loads(fixed_content)
            logger.info(f"✅ Strategy 3 successful: Completed incomplete JSON for {self.model_name}")
            return self._validate_and_fix_fields(analysis, market_data)
        except Exception as e:
            logger.debug(f"Strategy 3 failed: {e}")

        # Strategy 4: Extract partial valid JSON and rebuild structure
        try:
            # Extract individual fields using regex patterns
            fields = {}
            field_patterns = {
                'signal': r'"signal":\s*"(\w+)"',
                'confidence': r'"confidence":\s*([\d.]+)',
                'entry_price': r'"entry_price":\s*([\d.]+)',
                'activation_price': r'"activation_price":\s*([\d.]+)',
                'trailing_stop_pct': r'"trailing_stop_pct":\s*([\d.]+)',
                'invalidation_level': r'"invalidation_level":\s*([\d.]+)',
                'thesis_summary': r'"thesis_summary":\s*"([^"]*)"',
                'risk_reward_ratio': r'"risk_reward_ratio":\s*"([^"]*)"',
                'leverage': r'"leverage":\s*([\d.]+)',
                'quantity': r'"quantity":\s*([\d.]+)',
                'reasoning': r'"reasoning":\s*"([^"]*)"'
            }

            for field, pattern in field_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    # Convert to appropriate type
                    value = match.group(1)
                    if field in ['signal']:
                        fields[field] = value.upper()
                    elif field in ['confidence', 'entry_price', 'activation_price', 'trailing_stop_pct',
                                 'invalidation_level', 'leverage', 'quantity']:
                        fields[field] = float(value)
                    else:
                        fields[field] = value

            # If we found enough fields, reconstruct the analysis
            if len(fields) >= 5:  # Minimum fields needed
                analysis = self._validate_and_fix_fields(fields, market_data)
                logger.info(f"✅ Strategy 4 successful: Reconstructed JSON from partial data for {self.model_name}")
                return analysis
        except Exception as e:
            logger.debug(f"Strategy 4 failed: {e}")

        # Strategy 5: Fallback - create default signal but try to extract signal from content
        try:
            signal_match = re.search(r'"signal":\s*"(\w+)"', content, re.IGNORECASE)
            confidence_match = re.search(r'"confidence":\s*([\d.]+)', content)

            signal = signal_match.group(1).upper() if signal_match else "NONE"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0

            # Validate signal value
            if signal not in ["BUY", "SELL", "NONE"]:
                signal = "NONE"
                confidence = 0.0

            analysis = {
                "signal": signal,
                "confidence": confidence,
                "entry_price": market_data.get('technical_indicators', {}).get('price', 0),
                "activation_price": 0.0,
                "trailing_stop_pct": 0.0,
                "invalidation_level": 0.0,
                "thesis_summary": f"Partially recovered signal: {signal} from malformed response",
                "risk_reward_ratio": "1:1",
                "leverage": 1,
                "quantity": 0.0,
                "reasoning": f"Enhanced recovery: extracted {signal} signal with {confidence} confidence"
            }

            logger.info(f"✅ Strategy 5 successful: Partial recovery - {signal} signal for {self.model_name}")
            return analysis
        except Exception as e:
            logger.debug(f"Strategy 5 failed: {e}")

        # Final fallback - complete failure
        logger.warning(f"❌ All JSON recovery strategies failed for {self.model_name}")
        return {
            "signal": "NONE",
            "confidence": 0.0,
            "entry_price": market_data.get('technical_indicators', {}).get('price', 0),
            "activation_price": 0.0,
            "trailing_stop_pct": 0.0,
            "invalidation_level": 0.0,
            "thesis_summary": f"All recovery strategies failed - returning NONE signal",
            "risk_reward_ratio": "1:1",
            "leverage": 1,
            "quantity": 0.0,
            "reasoning": f"Complete JSON recovery failure: {str(original_error)}"
        }

    def _validate_and_fix_fields(self, analysis: Dict, market_data: Dict) -> Dict:
        """Validate and fix fields in recovered JSON"""
        # Ensure all required fields exist with proper types
        current_price = market_data.get('technical_indicators', {}).get('price', 0)

        # Fix signal value
        if 'signal' in analysis:
            signal = str(analysis['signal']).upper()
            if signal not in ["BUY", "SELL", "NONE"]:
                signal = "NONE"
            analysis['signal'] = signal

        # Fix confidence value
        if 'confidence' in analysis:
            try:
                confidence = float(analysis['confidence'])
                analysis['confidence'] = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            except:
                analysis['confidence'] = 0.0

        # Fix numeric fields
        numeric_fields = ['entry_price', 'activation_price', 'trailing_stop_pct',
                         'invalidation_level', 'leverage', 'quantity']
        for field in numeric_fields:
            if field in analysis:
                try:
                    analysis[field] = float(analysis[field])
                except:
                    analysis[field] = 0.0

        # Fix entry_price to use current price if invalid
        if analysis.get('entry_price', 0) <= 0:
            analysis['entry_price'] = current_price

        # Ensure string fields exist
        string_fields = ['thesis_summary', 'risk_reward_ratio', 'reasoning']
        for field in string_fields:
            if field not in analysis or not analysis[field]:
                if field == 'thesis_summary':
                    analysis[field] = f"Recovered signal: {analysis.get('signal', 'NONE')}"
                elif field == 'risk_reward_ratio':
                    analysis[field] = "1:1"
                elif field == 'reasoning':
                    analysis[field] = "JSON recovery and validation completed"

        return analysis

class Qwen3MaxClient(BaseModelClient):
    """Qwen3-Max model client - Alibaba's advanced reasoning model"""

    def __init__(self):
        super().__init__("Qwen3-Max", "qwen/qwen3-max")

    async def analyze_market(self, market_data: Dict, institutional_data: Dict) -> ModelSignal:
        """Analyze market using Qwen3-Max with advanced reasoning"""
        try:
            strategy_prompt = self._read_strategy_prompt()

            # Qwen3-Max specific prompt - optimized for complex reasoning and multi-step analysis
            analysis_prompt = f"""
{strategy_prompt}

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

INSTITUTIONAL DATA:
{json.dumps(institutional_data, indent=2)}

QWEN3-MAX ANALYSIS REQUIREMENTS:
As Alibaba's trillion-parameter reasoning model, provide comprehensive analysis using:
- Complex multi-step logical reasoning
- Cross-market correlation analysis
- Fundamental-technical synthesis
- Risk-adjusted opportunity assessment

Use your advanced reasoning capabilities to:
1. Synthesize multiple data sources coherently
2. Identify subtle market patterns and relationships
3. Evaluate risk-reward through rigorous logical frameworks
4. Consider both immediate and medium-term implications

Provide your analysis in this JSON format:
{{
    "signal": "BUY", "SELL", or "NONE",
    "confidence": 0.0-1.0,
    "entry_price": {market_data.get('technical_indicators', {}).get('price', 0)},
    "activation_price": float,
    "trailing_stop_pct": float,
    "invalidation_level": float,
    "thesis_summary": "comprehensive reasoning-based analysis",
    "risk_reward_ratio": "1:5+ format",
    "leverage": maximum available,
    "quantity": float,
    "reasoning": "Qwen3-Max specific reasoning highlighting complex logical analysis and multi-step deduction",
    "category_analysis": {{
        "market_regime": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "detailed logical explanation"}},
        "technical_setup": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "pattern-based reasoning"}},
        "onchain_metrics": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "causal chain analysis"}},
        "macro_catalysts": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "systematic evaluation"}},
        "risk_reward": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "quantitative reasoning"}},
        "timing_indicators": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "temporal pattern analysis"}},
        "institutional_signals": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "flow-based reasoning"}}
    }}
}}

CRITICAL: Signal MUST be "BUY" only if ALL 7 categories are BULLISH!
Emphasize rigorous reasoning and logical consistency in your analysis.
"""

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are Qwen3-Max, Alibaba's advanced reasoning model. Use your trillion-parameter capabilities for deep logical analysis. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.6,  # Qwen3-Max recommended temperature
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            # Enhanced JSON parsing with error recovery
            try:
                # Parse response
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                if not content:
                    raise ValueError("Empty response from Qwen3-Max")

                # Try to parse JSON directly first
                analysis = json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error for {self.model_name}: {e}")
                logger.warning(f"Raw content: {content[:500]}...")

                # Attempt to fix common JSON issues
                try:
                    # Try to extract JSON from the response if it's embedded in text
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        fixed_content = json_match.group(0)
                        analysis = json.loads(fixed_content)
                        logger.info(f"Successfully extracted JSON for {self.model_name}")
                    else:
                        # If no JSON found, create a default NONE signal
                        analysis = {
                            "signal": "NONE",
                            "confidence": 0.0,
                            "entry_price": market_data.get('technical_indicators', {}).get('price', 0),
                            "activation_price": 0.0,
                            "trailing_stop_pct": 0.0,
                            "invalidation_level": 0.0,
                            "thesis_summary": f"JSON parsing failed - returning NONE signal",
                            "risk_reward_ratio": "1:1",
                            "leverage": 1,
                            "quantity": 0.0,
                            "reasoning": f"JSON parsing error: {str(e)}"
                        }
                        logger.warning(f"Created default NONE signal for {self.model_name} due to JSON parsing failure")

                except Exception as fix_e:
                    logger.error(f"Failed to fix JSON for {self.model_name}: {fix_e}")
                    # Final fallback - create default NONE signal
                    analysis = {
                        "signal": "NONE",
                        "confidence": 0.0,
                        "entry_price": market_data.get('technical_indicators', {}).get('price', 0),
                        "activation_price": 0.0,
                        "trailing_stop_pct": 0.0,
                        "invalidation_level": 0.0,
                        "thesis_summary": f"Complete JSON parsing failure - returning NONE signal",
                        "risk_reward_ratio": "1:1",
                        "leverage": 1,
                        "quantity": 0.0,
                        "reasoning": f"Complete parsing failure: {str(e)}"
                    }

            # Validate required fields
            required_fields = ['signal', 'confidence', 'entry_price', 'activation_price',
                             'trailing_stop_pct', 'invalidation_level', 'thesis_summary',
                             'risk_reward_ratio', 'leverage', 'quantity', 'reasoning']

            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            # Validate asymmetric criteria
            if analysis['signal'] == 'BUY':
                if analysis['leverage'] < 1:
                    raise ValueError(f"Leverage must be at least 1x, got {analysis['leverage']}")

            logger.info(f"✅ Qwen3-Max analysis completed for {market_data.get('symbol', 'Unknown')}")

            return ModelSignal(
                model_name=self.model_name,
                signal=analysis['signal'],
                confidence=analysis['confidence'],
                entry_price=analysis['entry_price'],
                activation_price=analysis['activation_price'],
                trailing_stop_pct=analysis['trailing_stop_pct'],
                invalidation_level=analysis['invalidation_level'],
                thesis_summary=analysis['thesis_summary'],
                risk_reward_ratio=analysis['risk_reward_ratio'],
                leverage=analysis['leverage'],
                quantity=analysis['quantity'],
                category_analysis=analysis.get('category_analysis', {}),
                reasoning=analysis.get('reasoning', 'No specific reasoning provided')
            )

        except Exception as e:
            logger.error(f"Error in Qwen3-Max analysis: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}", self.model_name)

class DeepSeekTerminusClient(BaseModelClient):
    """DeepSeek V3.1-Terminus model client - specialized financial analysis model"""

    def __init__(self):
        super().__init__("DeepSeek Chat V3.1", "deepseek/deepseek-chat-v3.1")

    async def analyze_market(self, market_data: Dict, institutional_data: Dict) -> ModelSignal:
        """Analyze market using DeepSeek V3.1-Terminus with financial specialization"""
        try:
            strategy_prompt = self._read_strategy_prompt()

            # DeepSeek V3.1-Terminus specific prompt - optimized for financial analysis and stability
            analysis_prompt = f"""
{strategy_prompt}

CURRENT MARKET DATA:
{json.dumps(market_data, indent=2)}

INSTITUTIONAL DATA:
{json.dumps(institutional_data, indent=2)}

DEEPSEEK V3.1-TERMINUS ANALYSIS REQUIREMENTS:
As DeepSeek's specialized financial model, provide systematic analysis using:
- Rigorous quantitative assessment
- Risk management focus
- Institutional-grade financial modeling
- Stable and reliable signal generation

Use your financial analysis expertise to:
1. Quantify risk-reward scenarios precisely
2. Model institutional trading behavior
3. Evaluate market microstructure
4. Ensure signal reliability and consistency

Provide your analysis in this JSON format:
{{
    "signal": "BUY", "SELL", or "NONE",
    "confidence": 0.0-1.0,
    "entry_price": {market_data.get('technical_indicators', {}).get('price', 0)},
    "activation_price": float,
    "trailing_stop_pct": float,
    "invalidation_level": float,
    "thesis_summary": "quantitative financial analysis",
    "risk_reward_ratio": "1:5+ format",
    "leverage": maximum available,
    "quantity": float,
    "reasoning": "DeepSeek Terminus specific reasoning highlighting quantitative analysis and risk management",
    "category_analysis": {{
        "market_regime": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "quantitative regime assessment"}},
        "technical_setup": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "statistical technical evaluation"}},
        "onchain_metrics": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "onchain flow quantification"}},
        "macro_catalysts": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "macro impact modeling"}},
        "risk_reward": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "quantitative risk assessment"}},
        "timing_indicators": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "timing probability analysis"}},
        "institutional_signals": {{"status": "BULLISH/BEARISH/NEUTRAL", "rationale": "institutional flow quantification"}}
    }}
}}

CRITICAL: Signal MUST be "BUY" only if ALL 7 categories are BULLISH!
Focus on quantitative precision and risk management in your analysis.
"""

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are DeepSeek V3.1-Terminus, a specialized financial analysis model. Provide systematic, quantitative trading signals with focus on risk management. Respond only with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,  # Lower temperature for financial consistency
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            # Enhanced JSON parsing with error recovery
            try:
                # Parse response
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

                if not content:
                    raise ValueError("Empty response from DeepSeek V3.1-Terminus")

                # Try to parse JSON directly first
                analysis = json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error for {self.model_name}: {e}")
                logger.warning(f"Raw content: {content[:500]}...")

                # Attempt to fix common JSON issues
                try:
                    # Try to extract JSON from the response if it's embedded in text
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        fixed_content = json_match.group(0)
                        analysis = json.loads(fixed_content)
                        logger.info(f"Successfully extracted JSON for {self.model_name}")
                    else:
                        # If no JSON found, create a default NONE signal
                        analysis = {
                            "signal": "NONE",
                            "confidence": 0.0,
                            "entry_price": market_data.get('technical_indicators', {}).get('price', 0),
                            "activation_price": 0.0,
                            "trailing_stop_pct": 0.0,
                            "invalidation_level": 0.0,
                            "thesis_summary": f"JSON parsing failed - returning NONE signal",
                            "risk_reward_ratio": "1:1",
                            "leverage": 1,
                            "quantity": 0.0,
                            "reasoning": f"JSON parsing error: {str(e)}"
                        }
                        logger.warning(f"Created default NONE signal for {self.model_name} due to JSON parsing failure")

                except Exception as fix_e:
                    logger.error(f"Failed to fix JSON for {self.model_name}: {fix_e}")
                    # Final fallback - create default NONE signal
                    analysis = {
                        "signal": "NONE",
                        "confidence": 0.0,
                        "entry_price": market_data.get('technical_indicators', {}).get('price', 0),
                        "activation_price": 0.0,
                        "trailing_stop_pct": 0.0,
                        "invalidation_level": 0.0,
                        "thesis_summary": f"Complete JSON parsing failure - returning NONE signal",
                        "risk_reward_ratio": "1:1",
                        "leverage": 1,
                        "quantity": 0.0,
                        "reasoning": f"Complete parsing failure: {str(e)}"
                    }

            # Validate required fields
            required_fields = ['signal', 'confidence', 'entry_price', 'activation_price',
                             'trailing_stop_pct', 'invalidation_level', 'thesis_summary',
                             'risk_reward_ratio', 'leverage', 'quantity', 'reasoning']

            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")

            # Validate asymmetric criteria
            if analysis['signal'] == 'BUY':
                if analysis['leverage'] < 1:
                    raise ValueError(f"Leverage must be at least 1x, got {analysis['leverage']}")

            logger.info(f"✅ DeepSeek V3.1-Terminus analysis completed for {market_data.get('symbol', 'Unknown')}")

            return ModelSignal(
                model_name=self.model_name,
                signal=analysis['signal'],
                confidence=analysis['confidence'],
                entry_price=analysis['entry_price'],
                activation_price=analysis['activation_price'],
                trailing_stop_pct=analysis['trailing_stop_pct'],
                invalidation_level=analysis['invalidation_level'],
                thesis_summary=analysis['thesis_summary'],
                risk_reward_ratio=analysis['risk_reward_ratio'],
                leverage=analysis['leverage'],
                quantity=analysis['quantity'],
                category_analysis=analysis.get('category_analysis', {}),
                reasoning=analysis.get('reasoning', 'No specific reasoning provided')
            )

        except Exception as e:
            logger.error(f"Error in DeepSeek V3.1-Terminus analysis: {str(e)}")
            return self._create_none_signal(f"Analysis error: {str(e)}", self.model_name)

class MultiModelConsensusEngine:
    """Consensus engine that coordinates multiple AI models for trading decisions"""

    def __init__(self):
        self.grok_client = Grok4FastClient()
        self.qwen_client = Qwen3MaxClient()
        self.deepseek_client = DeepSeekTerminusClient()

        # Track model performance
        self.model_performance = {
            "Grok 4 Fast": {"correct": 0, "total": 0, "accuracy": 0.0},
            "Qwen3-Max": {"correct": 0, "total": 0, "accuracy": 0.0},
            "DeepSeek V3.1-Terminus": {"correct": 0, "total": 0, "accuracy": 0.0}
        }

    async def get_consensus_signal(self, market_data: Dict) -> ConsensusResult:
        """Get consensus signal from all three models using majority voting"""
        symbol = market_data.get('symbol', 'Unknown')

        try:
            logger.info(f"🤖 Starting multi-model consensus analysis for {symbol}")

            # Prepare institutional data for all models
            institutional_data = {
                "fear_greed": self.grok_client._get_institutional_data("fear_greed"),
                "funding_rates": self.grok_client._get_institutional_data("funding_rates"),
                "open_interest": self.grok_client._get_institutional_data("open_interest"),
                "institutional_flows": self.grok_client._get_institutional_data("institutional_flows"),
                "macro_catalysts": {
                    "fed_policy": "Neutral",
                    "inflation_trend": "Decreasing",
                    "market_sentiment": "Cautiously Optimistic",
                    "institutional_demand": "Moderate"
                },
                "onchain_metrics": {
                    "exchange_outflows": True,
                    "whale_accumulation": True,
                    "active_addresses": "Increasing",
                    "network_health": "Strong"
                }
            }

            # Run all models in parallel for efficiency
            tasks = [
                self.grok_client.analyze_market(market_data, institutional_data),
                self.qwen_client.analyze_market(market_data, institutional_data),
                self.deepseek_client.analyze_market(market_data, institutional_data)
            ]

            # Wait for all model responses
            signals = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle any failures
            valid_signals = []
            model_votes = {}
            disagreement_details = []

            for i, signal in enumerate(signals):
                model_names = ["Grok 4 Fast", "Qwen3-Max", "DeepSeek V3.1-Terminus"]
                model_name = model_names[i]

                if isinstance(signal, Exception):
                    logger.error(f"❌ {model_name} failed: {str(signal)}")
                    model_votes[model_name] = "ERROR"
                    disagreement_details.append(f"{model_name}: Analysis failed")
                elif isinstance(signal, ModelSignal):
                    valid_signals.append(signal)
                    model_votes[model_name] = signal.signal

                    if signal.signal == "BUY":
                        logger.info(f"🟢 {model_name}: BUY - Confidence: {signal.confidence:.2f}")
                        logger.info(f"   Reasoning: {signal.reasoning[:100]}...")
                    elif signal.signal == "SELL":
                        logger.info(f"🔴 {model_name}: SELL - Confidence: {signal.confidence:.2f}")
                        logger.info(f"   Reasoning: {signal.reasoning[:100]}...")
                    else:
                        logger.info(f"🔴 {model_name}: NONE - {signal.thesis_summary[:100]}...")
                        disagreement_details.append(f"{model_name}: {signal.thesis_summary[:100]}")
                else:
                    logger.error(f"❌ {model_name} returned invalid response")
                    model_votes[model_name] = "ERROR"
                    disagreement_details.append(f"{model_name}: Invalid response")

            # Calculate consensus using majority voting (2 out of 3)
            final_signal = self._calculate_majority_consensus(model_votes)

            if final_signal == "BUY":
                # Average the parameters from BUY signals
                buy_signals = [s for s in valid_signals if s.signal == "BUY"]
                if buy_signals:
                    consensus_params = self._average_signal_parameters(buy_signals)
                    combined_thesis = self._combine_thesis_statements(buy_signals)
                    avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)

                    logger.info(f"✅ CONSENSUS REACHED: BUY signal for {symbol}")
                    logger.info(f"   Agreement: {len(buy_signals)}/3 models voted BUY")
                    logger.info(f"   Average Confidence: {avg_confidence:.2f}")
                    logger.info(f"   Combined Thesis: {combined_thesis[:200]}...")

                    return ConsensusResult(
                        final_signal="BUY",
                        consensus_votes=model_votes,
                        confidence_avg=avg_confidence,
                        thesis_combined=combined_thesis,
                        recommended_params=consensus_params
                    )
                else:
                    logger.warning(f"⚠️  Unexpected state: BUY consensus but no BUY signals")

            elif final_signal == "SELL":
                # Average the parameters from SELL signals
                sell_signals = [s for s in valid_signals if s.signal == "SELL"]
                if sell_signals:
                    consensus_params = self._average_signal_parameters(sell_signals)
                    combined_thesis = self._combine_thesis_statements(sell_signals)
                    avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)

                    logger.info(f"✅ CONSENSUS REACHED: SELL signal for {symbol}")
                    logger.info(f"   Agreement: {len(sell_signals)}/3 models voted SELL")
                    logger.info(f"   Average Confidence: {avg_confidence:.2f}")
                    logger.info(f"   Combined Thesis: {combined_thesis[:200]}...")

                    return ConsensusResult(
                        final_signal="SELL",
                        consensus_votes=model_votes,
                        confidence_avg=avg_confidence,
                        thesis_combined=combined_thesis,
                        recommended_params=consensus_params
                    )
                else:
                    logger.warning(f"⚠️  Unexpected state: SELL consensus but no SELL signals")

            # No BUY or SELL consensus
            logger.info(f"❌ NO CONSENSUS: {symbol}")
            logger.info(f"   Votes: {model_votes}")

            return ConsensusResult(
                final_signal="NONE",
                consensus_votes=model_votes,
                confidence_avg=0.0,
                thesis_combined=f"No consensus reached for {symbol}",
                recommended_params={},
                disagreement_details=disagreement_details
            )

        except Exception as e:
            logger.error(f"Error in consensus analysis for {symbol}: {str(e)}")
            return ConsensusResult(
                final_signal="NONE",
                consensus_votes={"Error": "CONSENSUS_FAILED"},
                confidence_avg=0.0,
                thesis_combined=f"Consensus analysis failed: {str(e)}",
                recommended_params={},
                disagreement_details=[f"System error: {str(e)}"]
            )

    def _calculate_majority_consensus(self, votes: Dict[str, str]) -> str:
        """Calculate majority consensus (2 out of 3 required for BUY)"""
        buy_count = sum(1 for vote in votes.values() if vote == "BUY")
        total_valid = sum(1 for vote in votes.values() if vote in ["BUY", "NONE"])

        # Need at least 2 out of 3 valid models to agree on BUY
        if buy_count >= 2 and total_valid >= 2:
            return "BUY"
        else:
            return "NONE"

    def _average_signal_parameters(self, signals: List[ModelSignal]) -> Dict:
        """Average parameters from multiple BUY signals"""
        if not signals:
            return {}

        return {
            "entry_price": sum(s.entry_price for s in signals) / len(signals),
            "activation_price": sum(s.activation_price for s in signals) / len(signals),
            "trailing_stop_pct": sum(s.trailing_stop_pct for s in signals) / len(signals),
            "invalidation_level": sum(s.invalidation_level for s in signals) / len(signals),
            "leverage": int(sum(s.leverage for s in signals) / len(signals)),
            "quantity": sum(s.quantity for s in signals) / len(signals),
            "risk_reward_ratio": signals[0].risk_reward_ratio  # Use first one as reference
        }

    def _combine_thesis_statements(self, signals: List[ModelSignal]) -> str:
        """Combine thesis statements from multiple models"""
        if not signals:
            return "No thesis available"

        combined_reasoning = []
        for signal in signals:
            combined_reasoning.append(f"{signal.model_name}: {signal.reasoning}")

        return " | ".join(combined_reasoning)

    def update_model_performance(self, model_name: str, was_correct: bool):
        """Update model performance tracking"""
        if model_name in self.model_performance:
            self.model_performance[model_name]["total"] += 1
            if was_correct:
                self.model_performance[model_name]["correct"] += 1

            # Update accuracy
            total = self.model_performance[model_name]["total"]
            correct = self.model_performance[model_name]["correct"]
            self.model_performance[model_name]["accuracy"] = correct / total if total > 0 else 0.0

    def get_model_performance_report(self) -> Dict:
        """Get performance statistics for all models"""
        return self.model_performance.copy()

    async def test_individual_models(self, test_market_data: Dict) -> Dict:
        """Test each model individually for comparison"""
        institutional_data = {
            "fear_greed": self.grok_client._get_institutional_data("fear_greed"),
            "funding_rates": self.grok_client._get_institutional_data("funding_rates"),
            "open_interest": self.grok_client._get_institutional_data("open_interest"),
            "institutional_flows": self.grok_client._get_institutional_data("institutional_flows")
        }

        results = {}

        # Test each model individually
        models = [
            ("Grok 4 Fast", self.grok_client),
            ("Qwen3-Max", self.qwen_client),
            ("DeepSeek V3.1-Terminus", self.deepseek_client)
        ]

        for model_name, client in models:
            try:
                signal = await client.analyze_market(test_market_data, institutional_data)
                results[model_name] = {
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "thesis": signal.thesis_summary,
                    "reasoning": signal.reasoning
                }
            except Exception as e:
                results[model_name] = {
                    "signal": "ERROR",
                    "confidence": 0.0,
                    "thesis": f"Error: {str(e)}",
                    "reasoning": "Analysis failed"
                }

        return results