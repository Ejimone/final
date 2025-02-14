import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import pytz
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
@dataclass(frozen=True)
class WeatherConfig:
    """Immutable weather service configuration"""
    API_BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"
    REQUEST_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    CACHE_TTL: int = 600  # 10 minutes
    MAX_LOCATION_LENGTH: int = 100

class WeatherServiceError(Exception):
    """Base exception for weather service errors"""
    pass

class InvalidLocationError(WeatherServiceError):
    """Exception for invalid location input"""
    pass

class ApiConnectionError(WeatherServiceError):
    """Exception for API connection issues"""
    pass

class WeatherService:
    """Weather service with caching and error handling"""
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            logger.error("OpenWeatherMap API key not found in environment variables")
            raise ValueError("OpenWeatherMap API key not found in environment variables")
        logger.debug(f"Using OpenWeatherMap API key: {self.api_key}")  # Debug log for API key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self._session = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session properly"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather data with caching and enhanced error handling"""
        if not location or not isinstance(location, str):
            print("\n⚠️ Error: Invalid location provided")
            return {
                "status": "error",
                "message": "Invalid location provided"
            }
            
        try:
            session = await self.get_session()
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }

            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    weather_data = self._parse_weather_data(data, location)
                    return {
                        "status": "success",
                        "data": weather_data,
                        "source": "OpenWeatherMap"
                    }
                elif response.status == 401:
                    error_msg = "Weather service authentication failed"
                    print(f"\n⚠️ Error: {error_msg}")
                    return {
                        "status": "error",
                        "message": error_msg
                    }
                else:
                    error_msg = f"Weather service error: {response.status}"
                    print(f"\n⚠️ Error: {error_msg}")
                    return {
                        "status": "error",
                        "message": error_msg
                    }

        except aiohttp.ClientError as e:
            error_msg = f"Network error: {str(e)}"
            print(f"\n⚠️ Error: {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"\n⚠️ Error: {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }

    def _parse_weather_data(self, data: Dict[str, Any], location: str) -> Dict[str, Any]:
        """Parse weather API response into standardized format"""
        return {
            "location": location,
            "temperature": round(data["main"]["temp"], 1),
            "feels_like": round(data["main"]["feels_like"], 1),
            "humidity": data["main"]["humidity"],
            "wind_speed": round(data["wind"]["speed"], 1),
            "conditions": data["weather"][0]["description"].capitalize(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

async def get_weather(location: str) -> Dict[str, Any]:
    """Get weather for a location using WeatherService"""
    weather_service = None
    try:
        weather_service = WeatherService()
        result = await weather_service.get_weather(location)
        return result
    except Exception as e:
        logger.error("Weather lookup error", exc_info=True)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
    finally:
        if weather_service:
            await weather_service.close()

async def interactive_weather_lookup():
    """Interactive weather lookup with enhanced error handling"""
    print("\n" + "=" * 40)
    print("============ Weather Lookup ============")
    print("=" * 40 + "\n")

    try:
        location = input("Enter location (city, country code): ").strip()
        if not location:
            print("⚠️ Location cannot be empty")
            return None

        service = WeatherService()
        try:
            result = await service.get_weather(location)
            
            if result["status"] == "success":
                weather = result["data"]
                print(f"\n🌍 Current Weather in {weather['location']}")
                print(f"🕒 {weather['timestamp']}")
                print(f"🌡️ Temperature: {weather['temperature']}°C")
                print(f"🌡️ Feels like: {weather['feels_like']}°C")
                print(f"💧 Humidity: {weather['humidity']}%")
                print(f"🌪️ Wind Speed: {weather['wind_speed']} m/s")
                print(f"☁️ Conditions: {weather['conditions']}")
            else:
                print(f"\n⚠️ {result['message']}")
                
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {str(e)}")
            logger.error("Interactive lookup error", exc_info=True)
        finally:
            await service.close()

    except KeyboardInterrupt:
        print("\n\nWeather lookup cancelled by user")
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)}")
        logger.error("Weather lookup error", exc_info=True)
    finally:
        print("\nThank you for using Weather Lookup!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(interactive_weather_lookup())