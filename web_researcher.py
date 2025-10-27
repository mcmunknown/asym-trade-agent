"""
Comprehensive Web Research Module for Institutional-Grade Crypto Analysis

This module provides web scraping capabilities for gathering institutional-grade data
from multiple sources (Messari, DefiLlama, Glassnode, Token Terminal, etc.) for
deep coin analysis integration with GPT-5 trading framework.
"""

import asyncio
import logging
import time
import json
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import pickle
import os

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ResearchData:
    """Structured data format for institutional research"""
    asset: str
    timestamp: str
    treasury_accumulation: str = "Unknown"
    revenue_trend: str = "Unknown"
    tvl_trend: str = "Unknown"
    developer_activity: str = "Unknown"
    upcoming_events: str = "None"
    wallet_accumulation: str = "Unknown"
    tvl_value: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    active_addresses: int = 0
    network_fees: float = 0.0
    staking_ratio: float = 0.0
    token_burns: float = 0.0
    source_reliability: str = "Unknown"
    raw_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by TradingAIClient"""
        return {
            'treasury_accumulation': self.treasury_accumulation,
            'revenue_trend': self.revenue_trend,
            'tvl_trend': self.tvl_trend,
            'developer_activity': self.developer_activity,
            'upcoming_events': self.upcoming_events,
            'wallet_accumulation': self.wallet_accumulation,
            'source_reliability': self.source_reliability,
            'raw_data': self.raw_data
        }


class RateLimiter:
    """Rate limiting for API requests"""
    
    def __init__(self, calls_per_second: int = 5):
        self.calls_per_second = calls_per_second
        self.last_call = 0
    
    async def wait(self):
        """Wait if necessary to respect rate limits"""
        elapsed = time.time() - self.last_call
        if elapsed < (1.0 / self.calls_per_second):
            await asyncio.sleep((1.0 / self.calls_per_second) - elapsed)
        self.last_call = time.time()


class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    def __init__(self, name: str, rate_limiter: RateLimiter):
        self.name = name
        self.rate_limiter = rate_limiter
        self.session = None
    
    @abstractmethod
    async def fetch_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data for a specific asset"""
        pass
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


class CacheManager:
    """Caching system for API responses"""
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl  # Time to live in seconds
    
    def _get_cache_key(self, source: str, asset: str, endpoint: str) -> str:
        """Generate cache key for data"""
        key_data = f"{source}_{asset}_{endpoint}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, source: str, asset: str, endpoint: str = "main") -> Optional[Dict[str, Any]]:
        """Retrieve cached data if not expired"""
        cache_key = self._get_cache_key(source, asset, endpoint)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            if time.time() - cached_data['timestamp'] > self.ttl:
                return None
            
            return cached_data['data']
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, source: str, asset: str, data: Dict[str, Any], endpoint: str = "main"):
        """Store data in cache"""
        cache_key = self._get_cache_key(source, asset, endpoint)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


class MessariDataSource(DataSource):
    """Messari API data source for institutional metrics"""
    
    def __init__(self, api_key: str = None):
        super().__init__("Messari", RateLimiter(calls_per_second=3))
        self.api_key = api_key
        self.base_url = "https://data.messari.io/api/v1"
    
    async def fetch_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data from Messari API"""
        await self.rate_limiter.wait()
        
        # Check cache first
        cached_data = CacheManager().get("messari", asset)
        if cached_data:
            logger.info(f"Using cached Messari data for {asset}")
            return cached_data
        
        try:
            headers = {
                'X-Messari-API-Key': self.api_key if self.api_key else ''
            }
            
            # Use public endpoint for basic metrics
            url = f"{self.base_url}/assets/{asset}/metrics"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the data
                    CacheManager().set("messari", asset, data)
                    
                    return data
                else:
                    logger.warning(f"Messari API error for {asset}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Messari fetch error for {asset}: {e}")
            return {}


class DefiLlamaDataSource(DataSource):
    """DefiLlama API data source for TVL and protocol data"""
    
    def __init__(self):
        super().__init__("DefiLlama", RateLimiter(calls_per_second=5))
        self.base_url = "https://api.llama.fi"
    
    async def fetch_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data from DefiLlama API"""
        await self.rate_limiter.wait()
        
        # Check cache
        cached_data = CacheManager().get("defillama", asset)
        if cached_data:
            logger.info(f"Using cached DefiLlama data for {asset}")
            return cached_data
        
        try:
            # Get protocol data
            url = f"{self.base_url}/protocol/{asset.lower()}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the data
                    CacheManager().set("defillama", asset, data)
                    
                    return data
                else:
                    # If protocol not found, try to get TVL data from the main endpoint
                    tvl_url = f"{self.base_url}/tvl"
                    async with self.session.get(tvl_url) as tvl_response:
                        if tvl_response.status == 200:
                            tvl_data = await tvl_response.json()
                            # Find the asset in TVL data
                            asset_tvl = next((item for item in tvl_data 
                                            if item.get('name', '').lower() == asset.lower()), None)
                            return asset_tvl or {}
                    return {}
                    
        except Exception as e:
            logger.error(f"DefiLlama fetch error for {asset}: {e}")
            return {}


class GlassnodeDataSource(DataSource):
    """Glassnode API data source for on-chain metrics"""
    
    def __init__(self, api_key: str = None):
        super().__init__("Glassnode", RateLimiter(calls_per_second=2))
        self.api_key = api_key
        self.base_url = "https://api.glassnode.com/v1"
    
    async def fetch_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data from Glassnode API"""
        await self.rate_limiter.wait()
        
        # Check cache
        cached_data = CacheManager().get("glassnode", asset)
        if cached_data:
            logger.info(f"Using cached Glassnode data for {asset}")
            return cached_data
        
        try:
            headers = {
                'X-Api-Key': self.api_key if self.api_key else ''
            }
            
            # Get multiple metrics in parallel
            metrics = ['addresses_active_count', 'fees_volume_sum', 'transactions_count']
            tasks = []
            
            for metric in metrics:
                url = f"{self.base_url}/metrics/{self._map_asset_to_glassnode(asset)}/{metric}"
                task = self._fetch_metric(url, headers)
                tasks.append((metric, task))
            
            results = {}
            for metric, task in tasks:
                try:
                    data = await task
                    results[metric] = data
                except Exception as e:
                    logger.warning(f"Glassnode metric {metric} failed for {asset}: {e}")
                    results[metric] = []
            
            # Cache the data
            CacheManager().set("glassnode", asset, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Glassnode fetch error for {asset}: {e}")
            return {}
    
    async def _fetch_metric(self, url: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch a single metric from Glassnode"""
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []
    
    def _map_asset_to_glassnode(self, asset: str) -> str:
        """Map asset symbol to Glassnode format"""
        mapping = {
            'BTC': 'btc',
            'ETH': 'eth',
            'SOL': 'solana',
            'ARB': 'arbitrum',
            'XRP': 'ripple',
            'OP': 'optimism',
            'RENDER': 'render-token',
            'INJ': 'injective-protocol'
        }
        return mapping.get(asset.upper(), asset.lower())


class TokenTerminalDataSource(DataSource):
    """Token Terminal API data source for protocol metrics"""
    
    def __init__(self, api_key: str = None):
        super().__init__("TokenTerminal", RateLimiter(calls_per_second=3))
        self.api_key = api_key
        self.base_url = "https://api.tokenterminal.com/terminal/api"
    
    async def fetch_data(self, asset: str) -> Dict[str, Any]:
        """Fetch data from Token Terminal API"""
        await self.rate_limiter.wait()
        
        # Check cache
        cached_data = CacheManager().get("tokenterminal", asset)
        if cached_data:
            logger.info(f"Using cached Token Terminal data for {asset}")
            return cached_data
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
            
            # Get protocol fundamentals
            url = f"{self.base_url}/protocols/{self._map_asset_to_tokenterminal(asset)}/fundamental"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the data
                    CacheManager().set("tokenterminal", asset, data)
                    
                    return data
                else:
                    logger.warning(f"Token Terminal API error for {asset}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Token Terminal fetch error for {asset}: {e}")
            return {}
    
    def _map_asset_to_tokenterminal(self, asset: str) -> str:
        """Map asset symbol to Token Terminal format"""
        mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'ARB': 'arbitrum',
            'XRP': 'ripple',
            'OP': 'optimism',
            'RENDER': 'render',
            'INJ': 'injective'
        }
        return mapping.get(asset.upper(), asset.lower())


class ArkhamDataSource(DataSource):
    """Arkham Intelligence data source for wallet analysis"""
    
    def __init__(self, api_key: str = None):
        super().__init__("Arkham", RateLimiter(calls_per_second=3))
        self.api_key = api_key
        self.base_url = "https://api.arkhamintelligence.com"
    
    async def fetch_data(self, asset: str) -> Dict[str, Any]:
        """Fetch wallet analysis data from Arkham"""
        await self.rate_limiter.wait()
        
        # Check cache
        cached_data = CacheManager().get("arkham", asset)
        if cached_data:
            logger.info(f"Using cached Arkham data for {asset}")
            return cached_data
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
            
            # Get smart money flows
            url = f"{self.base_url}/v1/flows/{self._map_asset_to_arkham(asset)}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the data
                    CacheManager().set("arkham", asset, data)
                    
                    return data
                else:
                    logger.warning(f"Arkham API error for {asset}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Arkham fetch error for {asset}: {e}")
            return {}
    
    def _map_asset_to_arkham(self, asset: str) -> str:
        """Map asset symbol to Arkham format"""
        mapping = {
            'BTC': 'btc',
            'ETH': 'eth',
            'SOL': 'sol',
            'ARB': 'arb',
            'XRP': 'xrp',
            'OP': 'op',
            'RENDER': 'rnder',
            'INJ': 'inj'
        }
        return mapping.get(asset.upper(), asset.lower())


class WebResearcher:
    """Main web research orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sources = self._initialize_sources()
        self.cache_manager = CacheManager()
    
    def _initialize_sources(self) -> Dict[str, DataSource]:
        """Initialize all data sources"""
        sources = {}
        
        # Messari
        if self.config.get('messari_api_key'):
            sources['messari'] = MessariDataSource(self.config['messari_api_key'])
        
        # DefiLlama (no API key required)
        sources['defillama'] = DefiLlamaDataSource()
        
        # Glassnode
        if self.config.get('glassnode_api_key'):
            sources['glassnode'] = GlassnodeDataSource(self.config['glassnode_api_key'])
        
        # Token Terminal
        if self.config.get('tokenterminal_api_key'):
            sources['tokenterminal'] = TokenTerminalDataSource(self.config['tokenterminal_api_key'])
        
        # Arkham
        if self.config.get('arkham_api_key'):
            sources['arkham'] = ArkhamDataSource(self.config['arkham_api_key'])
        
        return sources
    
    async def research_asset(self, asset: str) -> ResearchData:
        """Perform comprehensive research on a single asset"""
        logger.info(f"Starting comprehensive research for {asset}")
        
        research_data = ResearchData(
            asset=asset,
            timestamp=datetime.now().isoformat(),
            source_reliability="Mixed"
        )
        
        # Collect data from all sources
        source_data = {}
        
        async with asyncio.TaskGroup() as tg:
            source_tasks = {}
            
            for source_name, source in self.sources.items():
                task = tg.create_task(source.fetch_data(asset))
                source_tasks[source_name] = task
            
            # Wait for all sources to complete
            for source_name, task in source_tasks.items():
                try:
                    source_data[source_name] = await task
                    logger.info(f"Successfully fetched {source_name} data for {asset}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {source_name} data for {asset}: {e}")
                    source_data[source_name] = {}
        
        # Process and aggregate data
        self._process_source_data(research_data, source_data)
        
        logger.info(f"Research completed for {asset}")
        return research_data
    
    def _process_source_data(self, research_data: ResearchData, source_data: Dict[str, Any]):
        """Process and aggregate data from all sources"""
        
        # Process Messari data
        if 'messari' in source_data and source_data['messari']:
            self._process_messari_data(research_data, source_data['messari'])
            research_data.source_reliability = "High"
        
        # Process DefiLlama data
        if 'defillama' in source_data and source_data['defillama']:
            self._process_defillama_data(research_data, source_data['defillama'])
        
        # Process Glassnode data
        if 'glassnode' in source_data and source_data['glassnode']:
            self._process_glassnode_data(research_data, source_data['glassnode'])
        
        # Process Token Terminal data
        if 'tokenterminal' in source_data and source_data['tokenterminal']:
            self._process_tokenterminal_data(research_data, source_data['tokenterminal'])
        
        # Process Arkham data
        if 'arkham' in source_data and source_data['arkham']:
            self._process_arkham_data(research_data, source_data['arkham'])
        
        # Set raw data for reference
        research_data.raw_data = source_data
    
    def _process_messari_data(self, research_data: ResearchData, data: Dict[str, Any]):
        """Process Messari data for institutional metrics"""
        try:
            if 'data' in data:
                metrics = data['data']
                
                # Market cap and volume
                market_data = metrics.get('market_data', {})
                research_data.market_cap = market_data.get('marketcap', {}).get('current', 0)
                research_data.volume_24h = market_data.get('volume_last_24_hours', 0)
                
                # Development activity
                dev_data = metrics.get('developer_activity', {})
                if dev_data:
                    research_data.developer_activity = "High" if dev_data.get('commits_30d', 0) > 100 else "Medium"
                
                # Protocol metrics
                protocol_data = metrics.get('token_metrics', {})
                if protocol_data:
                    research_data.tvl_value = protocol_data.get('supply_circulating', 0)
                    
        except Exception as e:
            logger.warning(f"Error processing Messari data: {e}")
    
    def _process_defillama_data(self, research_data: ResearchData, data: Dict[str, Any]):
        """Process DefiLlama TVL and protocol data"""
        try:
            if isinstance(data, dict):
                # TVL data
                tvl_value = data.get('tvl', 0)
                if isinstance(tvl_value, dict):
                    research_data.tvl_value = tvl_value.get('current', 0)
                else:
                    research_data.tvl_value = tvl_value
                
                # TVL trend (simplified)
                research_data.tvl_trend = "↑" if research_data.tvl_value > 100000000 else "→"
                
        except Exception as e:
            logger.warning(f"Error processing DefiLlama data: {e}")
    
    def _process_glassnode_data(self, research_data: ResearchData, data: Dict[str, Any]):
        """Process Glassnode on-chain metrics"""
        try:
            if 'addresses_active_count' in data:
                addresses_data = data['addresses_active_count']
                if addresses_data and len(addresses_data) > 1:
                    current = addresses_data[-1]['v']
                    previous = addresses_data[-2]['v']
                    change = ((current - previous) / previous) * 100
                    research_data.active_addresses = int(current)
                    
                    if change > 5:
                        research_data.wallet_accumulation = "Strong"
                    elif change > 0:
                        research_data.wallet_accumulation = "Moderate"
                    else:
                        research_data.wallet_accumulation = "Weak"
            
            if 'fees_volume_sum' in data:
                fees_data = data['fees_volume_sum']
                if fees_data and len(fees_data) > 1:
                    research_data.network_fees = fees_data[-1]['v']
                    
        except Exception as e:
            logger.warning(f"Error processing Glassnode data: {e}")
    
    def _process_tokenterminal_data(self, research_data: ResearchData, data: Dict[str, Any]):
        """Process Token Terminal protocol metrics"""
        try:
            if 'fundamentals' in data:
                fundamentals = data['fundamentals']
                
                # Revenue metrics
                revenue = fundamentals.get('revenue', {})
                if revenue:
                    current_revenue = revenue.get('current', 0)
                    previous_revenue = revenue.get('previous_period', 0)
                    if previous_revenue > 0:
                        revenue_growth = ((current_revenue - previous_revenue) / previous_revenue) * 100
                        research_data.revenue_trend = "↑" if revenue_growth > 5 else "→"
                
                # Tokenomics
                tokenomics = fundamentals.get('tokenomics', {})
                if tokenomics:
                    staking_ratio = tokenomics.get('staking_ratio', 0)
                    research_data.staking_ratio = staking_ratio
                    
        except Exception as e:
            logger.warning(f"Error processing Token Terminal data: {e}")
    
    def _process_arkham_data(self, research_data: ResearchData, data: Dict[str, Any]):
        """Process Arkham wallet flow analysis"""
        try:
            if 'flows' in data:
                flows = data['flows']
                
                # Analyze institutional flows
                net_flows = []
                for flow in flows[-30:]:  # Last 30 days
                    if isinstance(flow, dict):
                        net_flow = flow.get('netflow', 0)
                        net_flows.append(net_flow)
                
                if net_flows:
                    avg_flow = sum(net_flows) / len(net_flows)
                    positive_flows = sum(1 for flow in net_flows if flow > 0)
                    flow_ratio = positive_flows / len(net_flows)
                    
                    if avg_flow > 0 and flow_ratio > 0.6:
                        research_data.treasury_accumulation = "Strong"
                    elif avg_flow > 0 and flow_ratio > 0.4:
                        research_data.treasury_accumulation = "Moderate"
                    else:
                        research_data.treasury_accumulation = "Weak"
                        
        except Exception as e:
            logger.warning(f"Error processing Arkham data: {e}")
    
    async def research_multiple_assets(self, assets: List[str]) -> Dict[str, ResearchData]:
        """Research multiple assets in parallel"""
        logger.info(f"Starting parallel research for {len(assets)} assets: {', '.join(assets)}")
        
        research_tasks = []
        for asset in assets:
            task = self.research_asset(asset)
            research_tasks.append(task)
        
        results = await asyncio.gather(*research_tasks, return_exceptions=True)
        
        research_results = {}
        for i, result in enumerate(results):
            asset = assets[i]
            if isinstance(result, Exception):
                logger.error(f"Research failed for {asset}: {result}")
                # Create empty research data for failed assets
                research_results[asset] = ResearchData(
                    asset=asset,
                    timestamp=datetime.now().isoformat(),
                    source_reliability="Failed"
                )
            else:
                research_results[asset] = result
        
        logger.info(f"Completed parallel research for {len(assets)} assets")
        return research_results
    
    def get_research_summary(self, research_data: ResearchData) -> str:
        """Get a human-readable summary of research data"""
        summary = f"""
Asset: {research_data.asset}
Timestamp: {research_data.timestamp}
Source Reliability: {research_data.source_reliability}

Institutional Metrics:
- Treasury Accumulation: {research_data.treasury_accumulation}
- Revenue Trend: {research_data.revenue_trend}
- TVL Trend: {research_data.tvl_trend}
- Developer Activity: {research_data.developer_activity}
- Wallet Accumulation: {research_data.wallet_accumulation}
- Upcoming Events: {research_data.upcoming_events}

Market Metrics:
- TVL Value: ${research_data.tvl_value:,.0f}
- Market Cap: ${research_data.market_cap:,.0f}
- 24h Volume: ${research_data.volume_24h:,.0f}
- Active Addresses: {research_data.active_addresses:,}
- Network Fees: ${research_data.network_fees:,.2f}
- Staking Ratio: {research_data.staking_ratio:.1%}
        """.strip()
        
        return summary


# Integration helper functions
def create_web_researcher_from_config(config_dict: Dict[str, Any]) -> WebResearcher:
    """Create a WebResearcher instance from configuration dictionary"""
    return WebResearcher(config_dict)

async def research_institutional_data(assets: List[str], config: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    """Quick function to get institutional research data for given assets"""
    researcher = create_web_researcher_from_config(config or {})
    research_results = await researcher.research_multiple_assets(assets)
    
    # Convert to format expected by TradingAIClient
    formatted_results = {}
    for asset, research_data in research_results.items():
        formatted_results[asset] = research_data.to_dict()
    
    return formatted_results


if __name__ == "__main__":
    # Example usage and testing
    async def test_web_researcher():
        # Test with sample configuration
        config = {
            'messari_api_key': os.getenv('MESSARI_API_KEY'),
            'glassnode_api_key': os.getenv('GLASSNODE_API_KEY'),
            'tokenterminal_api_key': os.getenv('TOKENTERMINAL_API_KEY'),
            'arkham_api_key': os.getenv('ARKHAM_API_KEY')
        }
        
        researcher = WebResearcher(config)
        
        # Test with BTC
        btc_research = await researcher.research_asset('BTC')
        print(researcher.get_research_summary(btc_research))
        
        # Test multiple assets
        assets = ['BTC', 'ETH', 'SOL']
        multi_research = await researcher.research_multiple_assets(assets)
        
        for asset, data in multi_research.items():
            print(f"\n=== {asset} ===")
            print(data.to_dict())
    
    # Run test if executed directly
    asyncio.run(test_web_researcher())