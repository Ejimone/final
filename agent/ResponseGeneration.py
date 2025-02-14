# voice_assistant/response_generation.py

import logging
from openai import OpenAI
import google.generativeai as genai
from Config import Config

def generate_llm_response(model_type: str, api_key: str, messages: list) -> str:
    """Generate response using the specified model"""
    try:
        if model_type.lower() == 'gemini':
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(messages[-1]['content'])
            return response.text
        else:  # fallback to OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        return "Error in generating response"

def generate_response(model:str, api_key:str, chat_history:list, local_model_path:str=None):
    """
    Generate a response using the specified model.
    """
    try:
        if model.lower() == 'openai':
            return generate_llm_response('openai', api_key, chat_history)
        elif model.lower() in ['groq', 'gemini']:
            return generate_llm_response('gemini', api_key, chat_history)
        elif model.lower() == 'local' and local_model_path:
            # Placeholder for local LLM response generation
            return "Generated response from local model"
        else:
            raise ValueError("Unsupported response generation model")
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        return "Error in generating response"