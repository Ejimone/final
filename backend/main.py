import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

# Import from local modules
from SendEmail import AIService, ServiceConfig, setup_logging
from CallAgent import OutboundCaller
# Import dependencies needed for agents
from dotenv import load_dotenv
from livekit import rtc, api
from livekit.agents import (
    AgentSession, 
    Agent, 
    JobContext,
    RoomInputOptions, 
    WorkerOptions,
    cli
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    silero,
    turn_detector,
    noise_cancellation,
    google,
)

# Load environment variables
load_dotenv()
setup_logging()
logger = logging.getLogger("fastapi-app")
logger.setLevel(logging.INFO)

# Initialize the FastAPI app
app = FastAPI(
    title="AI Assistant API",
    description="API for Email, Voice Call, and Agent interactions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI service
ai_service = AIService()

# Pydantic models for request validation
class EmailRequest(BaseModel):
    to: EmailStr
    subject: str
    body: str

class GenerateTextRequest(BaseModel):
    prompt: str

class TimeRequest(BaseModel):
    location: str

class CallRequest(BaseModel):
    phone_number: str
    transfer_to: Optional[str] = None
    name: str = "Customer"
    appointment_time: str = "next Tuesday at 3pm"

class AgentRequest(BaseModel):
    room_name: str

# Dependency to ensure AI service is initialized
async def get_ai_service():
    if not ai_service.gmail_service:
        logger.warning("Gmail service not initialized, attempting to initialize")
        ai_service._initialize_gmail_service()
    return ai_service

# Routes for Email functionality
@app.post("/email/send", response_model=Dict[str, Any])
async def send_email(
    email_request: EmailRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Send an email using the Gmail API."""
    try:
        result = await ai_service.send_email(
            email_request.to,
            email_request.subject,
            email_request.body
        )
        return result
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending email: {str(e)}"
        )

@app.post("/email/generate", response_model=Dict[str, Any])
async def generate_email_content(
    generate_request: GenerateTextRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Generate text content using AI models."""
    try:
        generated_text = await ai_service.generate_text(generate_request.prompt)
        return {
            "status": "success",
            "content": generated_text
        }
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating text: {str(e)}"
        )

@app.post("/time/current", response_model=Dict[str, Any])
async def get_current_time(
    time_request: TimeRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Get current time for a specific location."""
    try:
        time_result = await ai_service.get_current_time(time_request.location)
        return time_result
    except Exception as e:
        logger.error(f"Error getting time: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting time: {str(e)}"
        )

# Routes for CallAgent functionality
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

async def start_outbound_call(call_request: CallRequest):
    """Background task to start an outbound call."""
    try:
        # Create dial info dictionary for the agent
        dial_info = {
            "phone_number": call_request.phone_number,
            "transfer_to": call_request.transfer_to
        }
        
        # Generate a unique room name
        room_name = f"outbound-call-{call_request.phone_number}-{int(asyncio.current_task().get_name())}"
        
        # Create a job context and agent session
        ctx = JobContext(room_name=room_name)
        
        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect()
        
        # Create the agent with the provided information
        agent = OutboundCaller(
            name=call_request.name,
            appointment_time=call_request.appointment_time,
            dial_info=dial_info,
        )
        
        # Set up the agent session
        session = AgentSession(
            turn_detection=turn_detector.EOUModel(),
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            tts=cartesia.TTS(),
            llm=openai.LLM.with_vertex(model="google/gemini-2.0-flash-exp"),
        )
        
        # Start the session
        session_task = asyncio.create_task(
            session.start(
                agent=agent,
                room=ctx.room,
                room_input_options=RoomInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            )
        )
        
        # Dial the user
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=outbound_trunk_id,
                    sip_call_to=dial_info["phone_number"],
                    participant_identity="phone_user",
                    wait_until_answered=True,
                )
            )
            
            # A participant is now available
            participant = await ctx.wait_for_participant(identity="phone_user")
            agent.set_participant(participant)
            
            # Let the session run until completion
            await session_task
            
        except Exception as e:
            logger.error(f"Error creating SIP participant: {e}")
            ctx.shutdown()
            
    except Exception as e:
        logger.error(f"Error starting outbound call: {e}")

@app.post("/call/outbound", response_model=Dict[str, Any])
async def outbound_call(
    call_request: CallRequest,
    background_tasks: BackgroundTasks
):
    """Start an outbound call in the background."""
    try:
        # Check if required environment variables are set
        if not outbound_trunk_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SIP trunk ID not configured"
            )
        
        # Start the call in the background
        background_tasks.add_task(start_outbound_call, call_request)
        
        return {
            "status": "success",
            "message": f"Outbound call to {call_request.phone_number} initiated",
            "details": {
                "name": call_request.name,
                "appointment_time": call_request.appointment_time
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating outbound call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initiating outbound call: {str(e)}"
        )

# Routes for basic voice assistant
class BasicAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def start_voice_assistant(agent_request: AgentRequest):
    """Background task to start a voice assistant."""
    try:
        ctx = JobContext(room_name=agent_request.room_name)
        await ctx.connect()
        
        session = AgentSession(
            llm=google.LLM(model="gemini-2.0-flash"),
            stt=deepgram.STT(model="nova-2"),
            tts=cartesia.TTS()  # Using cartesia instead of elevenlabs since that's imported in CallAgent
        )
        
        await session.start(
            room=ctx.room,
            agent=BasicAssistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
        
    except Exception as e:
        logger.error(f"Error starting voice assistant: {e}")

@app.post("/agent/start", response_model=Dict[str, Any])
async def start_agent(
    agent_request: AgentRequest,
    background_tasks: BackgroundTasks
):
    """Start a voice assistant in the background."""
    try:
        background_tasks.add_task(start_voice_assistant, agent_request)
        
        return {
            "status": "success",
            "message": f"Voice assistant started in room {agent_request.room_name}",
        }
    except Exception as e:
        logger.error(f"Error starting voice assistant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting voice assistant: {str(e)}"
        )

# Root endpoint to provide API information
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Welcome to the AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "email": [
                {"path": "/email/send", "method": "POST", "description": "Send an email using Gmail API"},
                {"path": "/email/generate", "method": "POST", "description": "Generate email content using AI models"}
            ],
            "time": [
                {"path": "/time/current", "method": "POST", "description": "Get current time for a specific location"}
            ],
            "call": [
                {"path": "/call/outbound", "method": "POST", "description": "Start an outbound call for appointment confirmation"}
            ],
            "agent": [
                {"path": "/agent/start", "method": "POST", "description": "Start a voice assistant"}
            ],
            "system": [
                {"path": "/health", "method": "GET", "description": "Health check endpoint"},
                {"path": "/docs", "method": "GET", "description": "API documentation (Swagger UI)"}
            ]
        },
        "documentation_url": "/docs"
    }

# Health check endpoint
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run the FastAPI app using Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
