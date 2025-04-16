from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime, timedelta
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiohttp
import httpx

logger = logging.getLogger(__name__)

class APIRateLimiter:
    """Rate limiter for API calls"""
    def __init__(self):
        self.rate_limits = {
            'newsapi': {'calls': 100, 'period': 60},  # 100 calls per minute
            'bing': {'calls': 100, 'period': 60},
            'newscatcher': {'calls': 100, 'period': 60},
            'brave': {'calls': 100, 'period': 60},
            'serpapi': {'calls': 100, 'period': 60}
        }
        self.call_history = {}

    async def acquire(self, api_name: str) -> bool:
        """Check if we can make an API call"""
        now = datetime.now()
        if api_name not in self.call_history:
            self.call_history[api_name] = []

        # Clean old history
        period = self.rate_limits[api_name]['period']
        self.call_history[api_name] = [
            t for t in self.call_history[api_name]
            if (now - t) < timedelta(seconds=period)
        ]

        if len(self.call_history[api_name]) < self.rate_limits[api_name]['calls']:
            self.call_history[api_name].append(now)
            return True
        return False

class APIManager:
    """Manages API calls with retries and fallbacks"""
    def __init__(self):
        self.rate_limiter = APIRateLimiter()
        self.session = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, httpx.HTTPError))
    )
    async def make_api_call(self, api_name: str, url: str, params: Dict = None, headers: Dict = None) -> Dict[str, Any]:
        """Make API call with rate limiting and retries"""
        if not await self.rate_limiter.acquire(api_name):
            raise Exception(f"Rate limit exceeded for {api_name}")

        try:
            session = await self.get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    raise Exception(f"Rate limit hit for {api_name}")
                    
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"API call failed for {api_name}: {str(e)}")
            raise

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None

# Global API manager instance
api_manager = APIManager()

# Export for use in other modules
async def get_api_manager() -> APIManager:
    return api_manager