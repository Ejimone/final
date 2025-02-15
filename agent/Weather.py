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

class WeatherService:
    """Production-grade weather data service"""
    
    def __init__(self, config: WeatherConfig = WeatherConfig()):
        self.config = config
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise WeatherServiceError("OPENWEATHER_API_KEY not found in environment variables")
        self.session = None
        self._validate_config()

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    def _validate_config(self) -> None:
        """Validate service configuration"""
        if not self.api_key:
            raise WeatherServiceError("OPENWEATHER_API_KEY not configured")
        if len(self.api_key) != 32:
            raise WeatherServiceError("Invalid API key format")

    async def close(self) -> None:
        """Clean up resources"""
        await self.session.close()

    def _validate_location(self, location: str) -> None:
        """Validate location input"""
        if not location or len(location) > self.config.MAX_LOCATION_LENGTH:
            raise InvalidLocationError("Invalid location format")
        if re.search(r"[^a-zA-Z0-9\s,.-]", location):
            raise InvalidLocationError("Location contains invalid characters")

    @retry(
        stop=stop_after_attempt(WeatherConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather data with comprehensive error handling"""
        try:
            await self._ensure_session()  # Ensure session is available
            self._validate_location(location)
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            
            async with self.session.get(
                self.config.API_BASE_URL,
                params=params,
                timeout=self.config.REQUEST_TIMEOUT
            ) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return self._format_response(response_data)
                elif response.status == 404:
                    return {
                        "status": "error",
                        "message": "Location not found"
                    }
                else:
                    logger.error("API Error: %s", response_data.get('message', 'Unknown error'))
                    return {
                        "status": "error",
                        "message": f"Weather service error: {response_data.get('message', 'Unknown error')}"
                    }

        except aiohttp.ClientError as e:
            logger.error("Network error: %s", str(e))
            return {
                "status": "error",
                "message": "Service temporarily unavailable"
            }
        except json.JSONDecodeError as e:
            logger.error("Invalid API response: %s", str(e))
            return {
                "status": "error",
                "message": "Invalid service response"
            }
        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }

    def _format_response(self, data: Dict) -> Dict[str, Any]:
        """Format API response into standardized format"""
        try:
            # Convert timezone offset (in seconds) to hours
            timezone_offset = data.get('timezone', 0)  # Default to UTC if not provided
            tz = datetime.now(pytz.UTC).astimezone(pytz.FixedOffset(timezone_offset // 60))
            
            return {
                "status": "success",
                "data": {
                    "location": f"{data['name']}, {data.get('sys', {}).get('country', '')}",
                    "temperature": data['main']['temp'],
                    "feels_like": data['main']['feels_like'],
                    "humidity": data['main']['humidity'],
                    "wind_speed": data['wind']['speed'],
                    "conditions": data['weather'][0]['description'].capitalize(),
                    "timestamp": tz.strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "raw_data": data
                },
                "type": "weather"
            }
        except KeyError as e:
            logger.error("Missing data in API response: %s", str(e))
            return {
                "status": "error",
                "message": "Incomplete weather data"
            }
        except Exception as e:
            logger.error("Error formatting response: %s", str(e))
            return {
                "status": "error",
                "message": "Error processing weather data"
            }

async def current_get_weather(service: WeatherService) -> None:
    """Interactive weather lookup interface"""
    print("\n" + "="*40)
    print(" Weather Lookup ".center(40, "="))
    print("="*40 + "\n")
    
    try:
        while True:
            try:
                location = input("Enter location (city, country code): ").strip()
                if not location:
                    print("⚠️ Location cannot be empty")
                    continue
                    
                result = await service.get_weather(location)
                
                if result.get("status") == "success":
                    weather = result["data"]
                    print(f"\n🌍 Current Weather in {weather['location']}")
                    print(f"🕒 {weather['timestamp']}")
                    print(f"🌡️ Temperature: {weather['temperature']}°C")
                    print(f"🌡️ Feels like: {weather['feels_like']}°C")
                    print(f"💧 Humidity: {weather['humidity']}%")
                    print(f"🌪️ Wind Speed: {weather['wind_speed']} m/s")
                    print(f"☁️ Conditions: {weather['conditions']}")
                else:
                    print(f"\n⚠️ {result.get('message', 'Could not retrieve weather data')}")
                
                choice = input("\nCheck another location? (y/N): ").lower()
                if choice != 'y':
                    break
                    
            except InvalidLocationError as e:
                print(f"⚠️ Error: {str(e)}")
            except WeatherServiceError as e:
                print(f"⚠️ Service error: {str(e)}")
            except Exception as e:
                print(f"⚠️ Unexpected error: {str(e)}")
                logger.exception("Interactive lookup error")
                break  # Break on unexpected errors
    finally:
        print("\nThank you for using Weather Lookup!")

# Example usage
async def get_weather():
    service = None
    try:
        # Debug: Print API key (remove in production)
        api_key = os.getenv("OPENWEATHER_API_KEY")
        print(f"API Key found: {bool(api_key)}")
        
        service = WeatherService()
        await current_get_weather(service)
    except WeatherServiceError as e:
        print(f"⚠️ Service initialization failed: {str(e)}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {str(e)}")
    finally:
        if service and service.session:
            await service.close()
            
async def get_weather_data(location: str) -> Dict:
    service = WeatherService()
    try:
        return await service.get_weather(location)
    finally:
        await service.close()
if __name__ == "__main__":
    import asyncio
    asyncio.run(get_weather())


"""
for integration with other services:
"""