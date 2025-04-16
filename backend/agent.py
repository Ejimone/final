import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, elevenlabs
from SendEmail import sendemail, AIService
from datetime import datetime
from WebScrapeAndProcess import web_search, scrape_webpages_with_serpapi, summarize_content, scrape_url, scraped_data
from RealTimeSearch import real_time_search
from Ai import initialize_llm
from typing import Dict, Any, List, Optional, Union
from Todo import TodoManager
from Weather import WeatherService
from ResponseGeneration import generate_llm_response
from Config import Config
import json
import os
import asyncio
import groq
from tenacity import retry, stop_after_attempt, wait_exponential
from RealTimeSearch import real_time_search

# Add this after the imports and before the main code

def format_voice_response(data: Any) -> str:
    """Format any data type into a voice-friendly response"""
    if isinstance(data, list):
        return "Here's what I found:\n" + "\n".join([f"{i+1}. {item}" for i, item in enumerate(data)])
    if isinstance(data, dict):
        # Special handling for todo responses
        if "message" in data and "todo" in data:
            todo = data["todo"]
            return f"{data['message']}. Details: {todo.get('title', '')}, due {todo.get('due_date', 'no date set')}"
        if "message" in data and "todos" in data:
            todos = data["todos"]
            if not todos:
                return "You have no todos."
            todo_list = "\n".join([f"{i+1}. {t.get('title', '')}" for i, t in enumerate(todos)])
            return f"{data['message']}:\n{todo_list}"
        # General dict formatting
        formatted = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in data.items() if v])
        return formatted or "No details available"
    return str(data)

# Initialize logger and environment variables
load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")

# Initialize TodoManager (retain state for todos)
todo_manager = TodoManager()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_prompt_with_groq(prompt: str) -> Dict[str, Any]:
    """Analyze user prompt using Groq API to determine intent"""
    client = groq.Client()
    
    system_prompt = """Analyze the user's request and return a JSON response with the following structure:
    {
        "intent": "email|weather|todo|web_search|real_time|factual|conversation",
        "details": {
            // Specific details based on intent
        },
        "requires_current_info": boolean
    }
    
    For email: Include "to", "subject", "content" if present
    For weather: Include "location"
    For todo: Include "action" (add|list|update|delete), "task", "due_date" if present
    For web_search or real_time: Include "query"
    For factual: Include "query" and "category" (politics|sports|news|general)
    For conversation: Include "topic"
    
    Set requires_current_info to true if the query needs up-to-date information (e.g., current events, prices, news)
    """
    
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=500
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq analysis failed: {str(e)}")
        raise


email_service = AIService()


async def process_user_prompt(prompt: str, session_id: str = "default") -> Dict[str, Any]:
    """Process user prompt and route to the appropriate function"""
    try:
        # Analyze prompt intent using Groq
        analysis = await analyze_prompt_with_groq(prompt)
        intent = analysis.get("intent", "conversation")
        details = analysis.get("details", {})
        requires_current_info = analysis.get("requires_current_info", False)
        
        response = {"status": "success", "data": None, "source": None}
        
        # If current information is required, try real-time retrieval first
        if requires_current_info:
            real_time_info = await real_time_search(prompt)
            if real_time_info["status"] == "success":
                # Format the response for voice
                formatted_response = f"According to {real_time_info.get('source', 'latest sources')}: {
                    real_time_info['data']
                }"
                return {"status": "success", "data": formatted_response}
        
        if intent == "email":
            # Use sendemail directly from SendEmail.py
            if all(k in details for k in ["to", "subject", "content"]):
                email_result = await sendemail(
                    to=details["to"],
                    subject=details["subject"],
                    body=details["content"]
                )
                response["data"] = "Email sent successfully" if email_result else "Failed to send email"
                
                    
        elif intent == "todo":
            action = details.get("action")
            if action == "add":
                # Enhanced add todo with more details
                todo_result = await todo_manager.add_todo(
                    title=details.get("task", ""),
                    description=details.get("description", ""),
                    due_date=details.get("due_date"),
                    priority=details.get("priority", "medium"),
                    tags=details.get("tags", []),
                    reminders=details.get("reminders", []),
                    schedule_meeting=details.get("schedule_meeting", False)
                )
                if todo_result:
                    response["data"] = {
                        "message": "Todo added successfully",
                        "todo": todo_result
                    }
                else:
                    response["data"] = "Failed to add todo"
                    response["status"] = "error"
                    
            elif action == "list":
                # Enhanced list with filtering options
                filters = {
                    "priority": details.get("priority"),
                    "tags": details.get("tags"),
                    "due_date": details.get("due_date"),
                    "completed": details.get("completed", False)
                }
                todos = await todo_manager.list_todos(filters)
                response["data"] = {
                    "message": f"Found {len(todos)} todos",
                    "todos": todos
                }
                
            elif action == "complete":
                todo_id = details.get("task_id")
                if todo_id:
                    success = await todo_manager.complete_todo(todo_id)
                    response["data"] = {
                        "message": "Todo marked complete" if success else "Todo not found",
                        "todo_id": todo_id,
                        "success": success
                    }
                else:
                    response["data"] = "Missing todo ID"
                    response["status"] = "error"
                    
            elif action == "delete":
                todo_id = details.get("task_id")
                if todo_id:
                    success = await todo_manager.delete_todo(todo_id)
                    response["data"] = {
                        "message": "Todo deleted" if success else "Todo not found",
                        "todo_id": todo_id,
                        "success": success
                    }
                else:
                    response["data"] = "Missing todo ID"
                    response["status"] = "error"
                    
            elif action == "update":
                todo_id = details.get("task_id")
                if todo_id:
                    updates = {k: v for k, v in details.items() 
                              if k in ["title", "description", "due_date", "priority", "tags", "reminders"]}
                    success = await todo_manager.update_todo(todo_id, updates)
                    response["data"] = {
                        "message": "Todo updated" if success else "Todo not found",
                        "todo_id": todo_id,
                        "success": success
                    }
                else:
                    response["data"] = "Missing todo ID"
                    response["status"] = "error"
                    
            else:
                response["data"] = f"Unsupported todo action: {action}"
                response["status"] = "error"
                
        
            query = details.get("query", prompt)
            
            # First try real-time search
            try:
                real_time_result = await real_time_search(query)
                if real_time_result["status"] == "success":
                    # Format real-time response
                    formatted_response = format_voice_response({
                        "query": query,
                        "result": real_time_result["data"],
                        "source": real_time_result.get("source", "real-time search"),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    return {
                        "status": "success",
                        "data": formatted_response,
                        "type": "real_time"
                    }
            except Exception as e:
                logger.warning(f"Real-time search failed, falling back to web search: {e}")

            # Fallback to web search
            try:
                web_results = await scraped_data(query)
                verification_prompt = f"""
                Analyze and provide the most up-to-date information for: {query}
                
                Sources:
                {web_results.get('data', '')}
                
                Requirements:
                1. Focus on verified information
                2. Include the timestamp
                3. Mention source reliability
                4. Provide a clear answer
                """
                
                verified_info = await generate_llm_response(
                    "groq", 
                    Config.GROQ_API_KEY,
                    [{"role": "user", "content": verification_prompt}]
                )
                
                formatted_response = format_voice_response({
                    "query": query,
                    "result": verified_info,
                    "source": "web search",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                return {
                    "status": "success",
                    "data": formatted_response,
                    "type": "web_search"
                }
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                return {
                    "status": "error",
                    "data": "Sorry, I couldn't retrieve that information right now."
                }
        else:
            # Handle general conversation queries
            conversation_prompt = f"""
            Provide a natural, conversational response to: {prompt}
            If the query might require current information, mention that the response
            should be verified against recent sources.
            """
            conversation_response = await generate_llm_response("groq", Config.GROQ_API_KEY, [{"role": "user", "content": conversation_prompt}])
            response["data"] = conversation_response
            response["source"] = "conversation"
            
        return response
        
    except Exception as e:
        logger.error(f"Error processing user prompt: {str(e)}")
        return {"status": "error", "data": str(e)}


def prewarm(proc: JobProcess):
    """Preload models and resources for faster task execution."""
    config = Config()
    
    try:
        # Initialize STT
        if config.TRANSCRIPTION_MODEL == "deepgram":
            stt_model = deepgram.STT()
        elif config.TRANSCRIPTION_MODEL == "elevenlabs":
            stt_model = elevenlabs.STT()
        else:
            stt_model = openai.STT(model="whisper-1")

        # Initialize LLM
        if config.RESPONSE_MODEL == "groq":
            llm_model = openai.LLM.with_groq(model=config.GROQ_LLM)
        else:
            llm_model = openai.LLM(model=config.OPENAI_LLM)

        # Initialize TTS with error handling
        try:
            tts_model = deepgram.TTS()  # Use Cartesia instead of Deepgram
        except Exception as tts_error:
            logger.warning(f"Failed to initialize Cartesia TTS, falling back to default: {tts_error}")
            tts_model = openai.TTS(model="tts-1")

        # Update process userdata
        proc.userdata.update({
            "vad": silero.VAD.load(),
            "task_router": process_user_prompt,
            "stt": stt_model,
            "llm": llm_model,
            "tts": tts_model,
            "email_service": email_service
        })
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise



async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice assistant."""
    config = Config()
    system_prompt = (
        "You are a helpful voice assistant created by OpenCode. "
        "Use short and long sentences, conversational responses optimized for voice interaction. "
        "Avoid markdown formatting and special characters. "
        "When handling emails: generate clear subject lines and concise body content. "
        "For web searches: summarize key points clearly. "
        "Maintain a friendly and professional tone in all interactions."
    )

    initial_ctx = llm.ChatContext().append(role="system", text=system_prompt)

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        llm=ctx.proc.userdata["llm"],
        tts=ctx.proc.userdata["tts"],
        chat_ctx=initial_ctx,
    )

    task_router = ctx.proc.userdata["task_router"]

    async def message_handler(text: str):
        """Handle incoming messages and route to appropriate handlers"""
        # Handle email confirmations first
        if "pending_email" in agent.context:
            confirmation = text.lower().strip()
            email_service = ctx.proc.userdata["email_service"]
            
            if "confirm" in confirmation:
                email_data = agent.context["pending_email"]
                result = await email_service.send_email_via_assistant(
                    email_data["to"],
                    email_data["subject"],
                    email_data["content"]
                )
                if result["status"] == "success":
                    await agent.say("Email sent successfully.")
                else:
                    await agent.say("Failed to send email. Please try again later.")
                del agent.context["pending_email"]
                return
                
            elif "cancel" in confirmation:
                await agent.say("Email cancelled.")
                del agent.context["pending_email"]
                return
                
            elif "edit" in confirmation:
                email_data = agent.context["pending_email"]
                try:
                    email_content = await agent.llm.generate(
                        f"Edit the email content: {email_data['content']}"
                    )
                    email_subject = await agent.llm.generate(
                        f"Edit the email subject: {email_data['subject']}"
                    )
                    agent.context["pending_email"].update({
                        "subject": email_subject.strip(),
                        "content": email_content.strip()
                    })
                    await agent.say("Email edited. Would you like to confirm or make more changes?")
                except Exception as e:
                    logger.error(f"Error editing email: {str(e)}")
                    await agent.say("Sorry, I encountered an error while editing the email.")
                return

        # Process normal requests through process_user_prompt
        try:
            response = await process_user_prompt(text)
            
            if response["status"] == "success":
                # If this is an email request, store it for confirmation
                if response.get("type") == "email":
                    agent.context["pending_email"] = {
                        "to": response["data"].get("to"),
                        "subject": response["data"].get("subject"),
                        "content": response["data"].get("content")
                    }
                    await agent.say(
                        "I've prepared the email. Would you like to confirm, edit, or cancel?"
                    )
                else:
                    # For non-email responses, just say the response
                    await agent.say(response["data"], allow_interruptions=True)
            else:
                error_msg = response.get("data", "An unknown error occurred")
                await agent.say(f"Sorry, I encountered an error: {error_msg}")
                
        except asyncio.CancelledError:
            logger.warning("Message handling was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            await agent.say("Sorry, I encountered an unexpected error processing your request.")

    agent.on_message = message_handler
    agent.start(ctx.room, participant)
    await agent.say("Hello! How can I assist you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

