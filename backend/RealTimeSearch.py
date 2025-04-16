import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List
from cachetools import TTLCache
import httpx

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from google.generativeai import GenerativeModel

from Weather import get_weather_data  # and get_current_time is defined below
from Ai import initialize_llm
from WebScrapeAndProcess import scraped_data

logger = logging.getLogger(__name__)

# Configure cache with 10 minute TTL
CACHE = TTLCache(maxsize=1000, ttl=600)

def error_response(message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "message": message,
        "type": "error",
        "timestamp": datetime.utcnow().isoformat()
    }

async def fetch_time_from_api(location: str) -> Dict[str, Any]:
    """Fallback timezone lookup using geolocation API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location,
                    "format": "json",
                    "limit": 1
                }
            )
            data = response.json()
            if data:
                lat = data[0]['lat']
                lon = data[0]['lon']
                time_response = await client.get(
                    "https://timeapi.io/api/Time/current/coordinate",
                    params={"latitude": lat, "longitude": lon}
                )
                time_data = time_response.json()
                return {
                    "status": "success",
                    "data": f"🕒 {location.title()}: {time_data['time']} {time_data['timeZone']}",
                    "type": "time",
                    "source": "timeapi.io",
                    "timestamp": datetime.utcnow().isoformat()
                }
            return error_response("Location not found")
    except Exception as e:
        logger.error(f"API time lookup failed: {str(e)}")
        return error_response("Could not determine time for this location")

async def get_current_time(location: str) -> Dict[str, Any]:
    """Get current time for a location with cache and fallback to API."""
    try:
        location = location.strip().lower()
        cache_key = f"time_{location}"
        if cache_key in CACHE:
            return CACHE[cache_key]

        timezone_mappings = {
            'nyc': 'America/New_York',
            'la': 'America/Los_Angeles',
            'chicago': 'America/Chicago',
            'london': 'Europe/London',
            'paris': 'Europe/Paris',
            'berlin': 'Europe/Berlin',
            'tokyo': 'Asia/Tokyo',
            'singapore': 'Asia/Singapore',
            'dubai': 'Asia/Dubai',
            'utc': 'UTC'
        }
        country_zones = {
            'us': ['America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles'],
            'india': ['Asia/Kolkata'],
            'china': ['Asia/Shanghai'],
            'russia': ['Europe/Moscow', 'Asia/Vladivostok']
        }
        if location in country_zones:
            zones = country_zones[location]
            current_times = []
            for zone in zones:
                tz = pytz.timezone(zone)
                current_time = datetime.now(tz)
                current_times.append(f"• {zone.split('/')[-1]}: {current_time.strftime('%I:%M %p %Z')}")
            result = {
                "status": "success",
                "data": "\n".join(current_times),
                "type": "time",
                "source": "timezone_db",
                "timestamp": datetime.utcnow().isoformat()
            }
            CACHE[cache_key] = result
            return result

        tz_name = timezone_mappings.get(location, location)
        try:
            tz = pytz.timezone(tz_name)
            current_time = datetime.now(tz)
            result = {
                "status": "success",
                "data": f"🕒 {tz_name.replace('_', ' ').title()}: {current_time.strftime('%I:%M %p %Z')}",
                "type": "time",
                "source": "timezone_db",
                "timestamp": datetime.utcnow().isoformat()
            }
            CACHE[cache_key] = result
            return result
        except pytz.exceptions.UnknownTimeZoneError:
            return await fetch_time_from_api(location)
    except Exception as e:
        logger.error(f"Time lookup error: {str(e)}", exc_info=True)
        return error_response(f"Time lookup failed: {str(e)}")

async def handle_weather(params: Dict) -> Dict[str, Any]:
    """Handle weather requests using get_weather_data."""
    try:
        location = params["location"]
        return await get_weather_data(location)
    except Exception as e:
        logger.error(f"Weather lookup failed: {str(e)}")
        return error_response("Could not retrieve weather data")

async def handle_time(params: Dict) -> Dict[str, Any]:
    """Handle time requests using get_current_time."""
    location = params.get("location", "UTC")
    return await get_current_time(location)

async def handle_news(params: Dict) -> Dict[str, Any]:
    """Handle news requests using scraped_data."""
    try:
        result = await scraped_data()
        return {"status": "success", "data": result, "type": "news"}
    except Exception as e:
        logger.error(f"News lookup failed: {str(e)}")
        return error_response("Failed to retrieve news")

def validate_request_info(request_info: Dict) -> bool:
    required_fields = {
        "weather": ["location"],
        "time": ["location"],
        "news": ["topic"],
        "stocks": ["symbol"],
        "sports": ["team"],
        "flights": ["number"]
    }
    req_type = request_info.get("type", "").lower()
    if req_type not in required_fields:
        return False
    return all(field in request_info for field in required_fields[req_type])

# Replace your existing fallback_search with the following:
async def fallback_search(query: str) -> Dict[str, Any]:
    """Fallback using the web_search function from WebScrapeAndProcess.py."""
    from WebScrapeAndProcess import web_search
    return await web_search(query)

async def real_time_search(user_prompt: str) -> Dict[str, Any]:
    """Analyze the query and route to the appropriate handler."""
    try:
        logger.info(f"Processing query: {user_prompt}")
        gemini_model = initialize_llm()
        if not gemini_model:
            logger.error("LLM initialization failed")
            return error_response("Service unavailable")
        
        analysis_prompt = f"""
        Analyze the request and respond with JSON. Categories supported:
        - weather (requires location)
        - time (requires location)
        - news (requires topic)
        - stocks (requires ticker symbol)
        - sports (requires team)
        - flights (requires flight number)

        Examples:
        Input: "What's the weather in Tokyo?"
        Output: {{"type": "weather", "location": "Tokyo"}}

        Input: "What's the time in London?"
        Output: {{"type": "time", "location": "London"}}

        Input: "Tell me the latest news about AI"
        Output: {{"type": "news", "topic": "artificial intelligence"}}

        Input: "{user_prompt}"
        """
        try:
            response = gemini_model.generate_content(analysis_prompt)
            response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            request_info = json.loads(response_text)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Analysis failed: {str(e)}")
            return await fallback_search(user_prompt)
        
        if not validate_request_info(request_info):
            return await fallback_search(user_prompt)
        
        handlers = {
            "weather": handle_weather,
            "time": handle_time,
            "news": handle_news,
            # Add other handlers (stocks, sports, flights) if needed.
        }
        handler = handlers.get(request_info["type"].lower(), fallback_search)
        return await handler(request_info)
    except Exception as e:
        logger.error(f"Real-time search failed: {str(e)}", exc_info=True)
        return error_response("Failed to process request")

async def main():
    """Prompt the user for a query and output the result."""
    query = input("Enter your query: ")
    result = await real_time_search(query)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())