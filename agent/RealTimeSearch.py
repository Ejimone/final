import os
import json
import logging
import requests
import asyncio
from datetime import datetime, timedelta
import re
import pytz
import uuid  # Added for session ID generation
from typing import Dict, Any, List, Optional, Union
from cachetools import TTLCache
from pathlib import Path
import aiohttp
import google.generativeai as genai
from Weather import WeatherService
from Ai import initialize_llm
import httpx
import msvcrt
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize cache with TTL
MEMORY_TTL = 300  # 5 minutes
CONVERSATION_MEMORY = {}

# Rate limiting settings
REQUEST_LIMITS = {
    'weather': {'calls': 60, 'period': 60},  # 60 calls per 60 seconds
    'news': {'calls': 30, 'period': 60},     # 30 calls per 60 seconds
    'time': {'calls': 100, 'period': 60},    # 100 calls per 60 seconds
}

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self):
        self.requests = {}
    
    def can_make_request(self, service: str) -> bool:
        if service not in REQUEST_LIMITS:
            return True
            
        now = datetime.now()
        if service not in self.requests:
            self.requests[service] = []
            return True
            
        # Clean up old requests
        cutoff = now - timedelta(seconds=REQUEST_LIMITS[service]['period'])
        self.requests[service] = [
            req_time for req_time in self.requests[service]
            if req_time > cutoff
        ]
        
        # Check if under limit
        if len(self.requests[service]) < REQUEST_LIMITS[service]['calls']:
            self.requests[service].append(now)
            return True
            
        return False

# Initialize rate limiter
rate_limiter = RateLimiter()

class ResponseFormatter:
    """Utility class for formatting API responses consistently"""
    
    @staticmethod
    def format_weather(weather_data: Dict[str, Any]) -> str:
        return (
            f"\n🌍 Current Weather in {weather_data['location']}\n"
            f"🕒 {weather_data['timestamp']}\n"
            f"🌡️ Temperature: {weather_data['temperature']}°C\n"
            f"🌡️ Feels like: {weather_data['feels_like']}°C\n"
            f"💧 Humidity: {weather_data['humidity']}%\n"
            f"🌪️ Wind Speed: {weather_data['wind_speed']} m/s\n"
            f"☁️ Conditions: {weather_data['conditions']}"
        )

    @staticmethod
    def format_time(location: str, time_data: str, timezone: str = None) -> str:
        return (
            f"\n⏰ Current Time in {location.replace('_', ' ').title()}\n"
            f"{'-' * 30}\n"
            f"🕐 {time_data}"
            + (f" {timezone}" if timezone else "")
        )

    @staticmethod
    def format_news(articles: List[Dict]) -> str:
        return "\n\n".join(
            f"📰 {article['title']}\n"
            f"📍 Source: {article.get('source', {}).get('name', 'Unknown')}\n"
            f"🔗 {article['url']}\n"
            f"📝 {article.get('description', '')}"
            for article in articles
        )

    @staticmethod
    def format_web_results(results: List[Dict]) -> str:
        return "\n\n".join(
            f"🔗 {res['title']}\n"
            f"📍 {res['url']}\n"
            f"📝 {res.get('description', '')}"
            for res in results
        )

def get_conversation_memory(session_id: str = "default") -> Dict[str, Any]:
    """Get conversation memory for a session with TTL handling"""
    current_time = datetime.now()
    # Clean up expired memories
    expired = [sid for sid, data in CONVERSATION_MEMORY.items() 
              if (current_time - data["timestamp"]).total_seconds() > MEMORY_TTL]
    for sid in expired:
        del CONVERSATION_MEMORY[sid]
    
    # Initialize or return existing memory
    if (session_id not in CONVERSATION_MEMORY):
        CONVERSATION_MEMORY[session_id] = {
            "messages": [],
            "timestamp": current_time,
            "last_context": None
        }
    else:
        CONVERSATION_MEMORY[session_id]["timestamp"] = current_time
    
    return CONVERSATION_MEMORY[session_id]

def update_conversation_memory(session_id: str, user_input: str, response: Dict[str, Any]):
    """Update conversation memory with new interaction"""
    memory = get_conversation_memory(session_id)
    memory["messages"].append({"role": "user", "content": user_input})
    
    # Store the response and its context
    if response["status"] in ["success", "partial"]:
        response_content = response.get("data", "")
        if isinstance(response_content, dict):
            response_content = "\n".join(f"{k}: {v}" for k, v in response_content.items())
        memory["messages"].append({"role": "assistant", "content": response_content})
        memory["last_context"] = {
            "type": response.get("type"),
            "source": response.get("source"),
            "data": response.get("data")
        }

def get_context_from_memory(session_id: str, user_input: str) -> Optional[Dict[str, Any]]:
    """Get relevant context from conversation memory"""
    memory = get_conversation_memory(session_id)
    
    # Check if the query is a follow-up question
    follow_up_indicators = [
        "what about", "how about", "what's the", "what is the",
        "and the", "is it", "are they", "does it", "do they",
        "there", "here", "now", "today", "tonight", "tomorrow",
        "currently", "right now"
    ]
    
    location_context_words = [
        "temperature", "weather", "humidity", "wind", "rain",
        "hot", "cold", "warm", "cloudy", "sunny", "forecast",
        "time", "timezone", "hour", "morning", "evening", "afternoon"
    ]
    
    is_follow_up = (
        any(user_input.lower().startswith(indicator) for indicator in follow_up_indicators) or
        any(word in user_input.lower() for word in location_context_words)
    )
    
    if is_follow_up and memory["last_context"]:
        return memory["last_context"]
    
    return None

def error_response(message: str) -> Dict[str, Any]:
    """Generate error response"""
    return {
        "status": "error",
        "message": message,
        "type": "error"
    }

async def real_time_search(user_prompt: str, session_id: str = "default") -> Dict[str, Any]:
    """Process real-time information requests with context awareness"""
    try:
        logger.info(f"Processing query: {user_prompt}")
        
        # Check for context from previous conversation
        context = get_context_from_memory(session_id, user_prompt)
        
        # Handle follow-up questions based on context
        if context:
            if context["type"] == "weather":
                location = context.get("data", {}).get("location")
                if location:
                    result = await handle_weather({"location": location})
                    update_conversation_memory(session_id, user_prompt, result)
                    return result
            elif context["type"] == "time":
                location = context.get("data", {}).get("location")
                if location:
                    result = await handle_time({"location": location})
                    update_conversation_memory(session_id, user_prompt, result)
                    return result

        # Analyze the request
        model = initialize_llm()
        analysis_prompt = f"""
        Analyze the request and respond with JSON. Categories:
        - weather (requires location)
        - time (requires timezone/location)
        - news (requires topic)

        Examples:
        "What's the weather in Tokyo?" -> {{"type": "weather", "location": "Tokyo"}}
        "What time is it in London?" -> {{"type": "time", "location": "London"}}
        "Show me news about AI" -> {{"type": "news", "topic": "artificial intelligence"}}

        Request: "{user_prompt}"
        """

        try:
            if hasattr(model, 'generate_content'):
                response = model.generate_content(analysis_prompt)
                clean_response = response.text if hasattr(response, 'text') else str(response)
            else:  # OpenAI client
                response = model.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                clean_response = response.choices[0].message.content

            request_info = json.loads(clean_response)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            result = await fallback_search(user_prompt)
            update_conversation_memory(session_id, user_prompt, result)
            return result

        # Route to appropriate handler
        handlers = {
            "weather": handle_weather,
            "time": handle_time,
            "news": handle_news,
        }

        handler = handlers.get(request_info["type"].lower(), fallback_search)
        result = await handler(request_info if handler != fallback_search else user_prompt)
        
        # Update conversation memory
        update_conversation_memory(session_id, user_prompt, result)
        return result

    except Exception as e:
        logger.error(f"Real-time search failed: {str(e)}", exc_info=True)
        return error_response("Failed to process request")

async def handle_weather(params: Dict) -> Dict[str, Any]:
    """Handle weather requests with rate limiting"""
    if not rate_limiter.can_make_request('weather'):
        return error_response("Rate limit exceeded. Please try again later.")
    
    weather_service = None
    try:
        location = params.get("location")
        if not location:
            return error_response("Location not provided")

        weather_service = WeatherService()
        result = await weather_service.get_weather(location)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "type": "weather",
                "data": ResponseFormatter.format_weather(result["data"]),
                "source": result["source"]
            }
        return result

    except Exception as e:
        logger.error(f"Weather lookup failed: {str(e)}")
        return error_response("Could not retrieve weather data")
    finally:
        if weather_service:
            await weather_service.close()

async def handle_time(params: Dict) -> Dict[str, Any]:
    """Handle time requests with rate limiting"""
    if not rate_limiter.can_make_request('time'):
        return error_response("Rate limit exceeded. Please try again later.")
        
    try:
        location = params.get("location", "UTC")
        
        if location.lower() in ['us', 'usa']:
            zones = {
                'Eastern': 'America/New_York',
                'Central': 'America/Chicago',
                'Mountain': 'America/Denver',
                'Pacific': 'America/Los_Angeles'
            }
            times = []
            for name, zone in zones.items():
                try:
                    tz = pytz.timezone(zone)
                    current = datetime.now(tz)
                    times.append(f"🕐 {name}: {current.strftime('%I:%M %p %Z')}")
                except Exception:
                    continue
            
            if times:
                formatted_result = f"\n⏰ Current Time in USA\n" + "-" * 30 + "\n" + "\n".join(times)
                return {
                    "status": "success",
                    "type": "time",
                    "data": formatted_result,
                    "source": "system"
                }
        
        # Try direct timezone lookup
        try:
            tz = pytz.timezone(location)
            current_time = datetime.now(tz)
            formatted_result = (
                f"\n⏰ Current Time in {location.replace('_', ' ').title()}\n"
                f"{'-' * 30}\n"
                f"🕐 {current_time.strftime('%I:%M %p %Z')}"
            )
            
            return {
                "status": "success",
                "type": "time",
                "data": formatted_result,
                "source": "system"
            }
        except pytz.exceptions.UnknownTimeZoneError:
            return await fetch_time_from_api(location)

    except Exception as e:
        logger.error(f"Time lookup failed: {str(e)}")
        return error_response("Could not retrieve time information")

async def fetch_time_from_api(location: str) -> Dict[str, Any]:
    """Fetch time from external API for locations not in pytz"""
    try:
        async with httpx.AsyncClient() as client:
            # Get coordinates for the location
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1}
            )
            data = response.json()
            
            if data:
                lat = data[0]['lat']
                lon = data[0]['lon']
                
                # Get time for coordinates
                time_response = await client.get(
                    f"https://timeapi.io/api/Time/current/coordinate",
                    params={"latitude": lat, "longitude": lon}
                )
                time_data = time_response.json()
                
                formatted_result = (
                    f"\n⏰ Current Time in {location.title()}\n"
                    f"{'-' * 30}\n"
                    f"🕐 {time_data['time']} {time_data['timeZone']}"
                )
                
                return {
                    "status": "success",
                    "type": "time",
                    "data": formatted_result,
                    "source": "timeapi.io"
                }
            
            return error_response("Location not found")
            
    except Exception as e:
        logger.error(f"Time API lookup failed: {str(e)}")
        return error_response("Could not determine time for this location")

async def handle_news(params: Dict) -> Dict[str, Any]:
    """Handle news requests with rate limiting"""
    if not rate_limiter.can_make_request('news'):
        return error_response("Rate limit exceeded. Please try again later.")
        
    try:
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            return error_response("News service not available")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/news/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": brave_api_key
                },
                params={
                    "q": params["topic"],
                    "count": 3,
                    "freshness": "day"
                }
            )
            
            if response.status_code != 200:
                return error_response("News service error")
                
            results = response.json().get("results", [])
            if not results:
                return error_response("No news found")
            
            return {
                "status": "success",
                "type": "news",
                "data": ResponseFormatter.format_news(results),
                "source": "Brave News"
            }
    except Exception as e:
        logger.error(f"News lookup failed: {str(e)}")
        return error_response("Could not retrieve news")

async def fallback_search(query: str) -> Dict[str, Any]:
    """Fallback to web search when specific handlers fail"""
    try:
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            return error_response("Search service not available")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": brave_api_key
                },
                params={"q": query, "count": 3}
            )
            
            if response.status_code != 200:
                return error_response("Search service error")
                
            results = response.json().get("web", {}).get("results", [])
            if not results:
                return error_response("No results found")
            
            return {
                "status": "success",
                "type": "web",
                "data": ResponseFormatter.format_web_results(results),
                "source": "Brave Search"
            }
    except Exception as e:
        logger.error(f"Fallback search failed: {str(e)}")
        return error_response("Could not retrieve information")

@contextmanager
def file_lock(filename):
    """Cross-platform file locking context manager"""
    if os.name == 'nt':  # Windows
        while True:
            try:
                with open(filename, 'a', encoding='utf-8') as f:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    try:
                        yield f
                    finally:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                break
            except IOError:  # Wait for file to be unlocked
                time.sleep(0.1)
    else:  # Unix-like
        # Import fcntl only on Unix systems
        import fcntl
        with open(filename, 'a', encoding='utf-8') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

async def real_time_result(user_prompt: str, session_id: str = "default") -> Dict[str, Any]:
    """Process user prompt and store history with proper file handling"""
    try:
        result = await real_time_search(user_prompt, session_id)
        
        # Format the history entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory = get_conversation_memory(session_id)
        
        history_text = f"\n{'='*50}\n"
        history_text += f"Session: {session_id}\n"
        history_text += f"Time: {timestamp}\n"
        history_text += f"Query: {user_prompt}\n"
        
        # Add context if this was a follow-up question
        context = memory.get("last_context")
        if context:
            history_text += f"Context: Following up on previous {context['type']} query\n"
        
        history_text += f"Result:\n{result.get('data', result.get('message', ''))}"
        if result.get("source"):
            history_text += f"\nSource: {result['source']}"
        
        history_text += f"\n{'='*50}\n"
        
        # Write to history file with proper locking
        history_file_path = os.path.join(os.path.dirname(__file__), "real_time_history.txt")
        try:
            with file_lock(history_file_path) as f:
                f.write(history_text)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Failed to write to history file: {e}")
            
        # Print the formatted result immediately
        if result["status"] == "success":
            print(f"\n{result['data']}")
        else:
            print(f"\n⚠️ {result['message']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to process request: {str(e)}"
        logger.error(error_msg)
        return error_response(error_msg)

#for testing purposes
if __name__ == "__main__":
    async def main():
        session_id = str(uuid.uuid4())
        print("\nWelcome to Real-Time Search!")
        print("Type 'exit', 'quit', or 'q' to end the session")
        print("Your queries will maintain context for follow-up questions\n")
        
        while True:
            try:
                query = input("\nQuery: ")
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                    
                result = await real_time_result(query, session_id)
                if result["status"] == "success":
                    print(f"\n{result['data']}")
                else:
                    print(f"\n⚠️ {result['message']}")
                print("\n---")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                
    asyncio.run(main())