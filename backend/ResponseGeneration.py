# voice_assistant/response_generation.py

import logging
from openai import OpenAI
import google.generativeai as genai
from Config import Config
import json
from datetime import datetime
import pytz
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def format_real_time_response(data: Dict[str, Any], query_type: str) -> str:
    """Format real-time information with clear attribution and timestamps"""
    try:
        if not data or "data" not in data:
            return "Could not retrieve current information."

        response_parts = []
        
        # Add timestamp header
        current_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        response_parts.append(f"As of {current_time}:")

        # Format based on data type
        if isinstance(data["data"], str):
            response_parts.append(data["data"])
        elif isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict):
                    # Format news/article data
                    if "title" in item:
                        response_parts.append(f"\n• {item['title']}")
                        if "timestamp" in item:
                            response_parts.append(f"  Published: {item['timestamp']}")
                        if "source" in item:
                            response_parts.append(f"  Source: {item['source']}")
                    # Format other structured data
                    else:
                        for key, value in item.items():
                            response_parts.append(f"\n• {key}: {value}")
        
        # Add source attribution
        if "source" in data:
            response_parts.append(f"\nInformation from: {data['source']}")
            
        # Add freshness warning if needed
        if "timestamp" in data:
            try:
                data_time = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                age = datetime.now(pytz.UTC) - data_time.astimezone(pytz.UTC)
                if age.total_seconds() > 3600:  # Older than 1 hour
                    response_parts.append("\nNote: Some information may not be the most current.")
            except Exception:
                pass

        return "\n".join(response_parts)
    except Exception as e:
        logger.error(f"Error formatting real-time response: {e}")
        return str(data.get("data", "Error formatting response"))

def generate_llm_response(model_type: str, api_key: str, messages: list) -> str:
    """Generate response using the specified model with enhanced real-time handling"""
    try:
        # Check if this is a real-time query
        is_realtime = any(word in messages[-1]['content'].lower() 
                         for word in ['current', 'now', 'latest', 'recent', 'today'])
        
        if model_type.lower() == 'gemini':
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Add real-time context if needed
            if is_realtime:
                current_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                messages[-1]['content'] = f"Current time: {current_time}\n\n{messages[-1]['content']}"
            
            response = model.generate_content(messages[-1]['content'])
            return response.text if hasattr(response, 'text') else str(response)
        else:
            client = OpenAI(api_key=api_key)
            
            # Add real-time context if needed
            if is_realtime:
                current_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                messages.insert(0, {
                    "role": "system",
                    "content": f"Current time: {current_time}. Provide the most up-to-date information available."
                })
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return "Error in generating response"

def generate_response(model: str, api_key: str, chat_history: list, local_model_path: str = None):
    """Generate a response using the specified model with real-time awareness"""
    try:
        # Determine if this is a real-time query
        latest_message = chat_history[-1]['content'] if chat_history else ""
        is_realtime = any(word in latest_message.lower() 
                         for word in ['current', 'now', 'latest', 'recent', 'today'])

        if model.lower() == 'openai':
            return generate_llm_response('openai', api_key, chat_history)
        elif model.lower() in ['groq', 'gemini']:
            return generate_llm_response('gemini', api_key, chat_history)
        elif model.lower() == 'local' and local_model_path:
            # Placeholder for local LLM response generation
            return "Generated response from local model"
        else:
            return "Invalid model specified"
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return "Error in generating response"