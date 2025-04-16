from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import asyncio
import os
from SendEmail import AIService
from Todo import TodoManager
from CallAgent import OutboundCaller

# Initialize FastAPI app
app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main-api")

# Initialize shared services
ai_service = AIService()
todo_manager = TodoManager()

# --- Email API Models and Endpoints ---
class EmailRequest(BaseModel):
    to: str
    subject: Optional[str] = None
    body: Optional[str] = None

class TextGenRequest(BaseModel):
    prompt: str

@app.post("/email/send")
async def send_email(req: EmailRequest):
    result = await ai_service.send_email(req.to, req.subject or "", req.body or "")
    return result

@app.post("/email/generate-text")
async def generate_text(req: TextGenRequest):
    text = await ai_service.generate_text(req.prompt)
    return {"text": text}

# --- Todo API Endpoints (example: add/list) ---
class TodoAddRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = None
    priority: Optional[str] = "medium"
    tags: Optional[List[str]] = []
    reminders: Optional[List[str]] = []
    schedule_meeting: Optional[bool] = False

@app.post("/todo/add")
async def add_todo(req: TodoAddRequest):
    todo = await todo_manager.add_todo(
        title=req.title,
        description=req.description,
        due_date=req.due_date,
        priority=req.priority,
        tags=req.tags,
        reminders=req.reminders,
        schedule_meeting=req.schedule_meeting
    )
    return {"todo": todo}

@app.get("/todo/list")
async def list_todos(priority: Optional[str] = None, tags: Optional[str] = None, due_date: Optional[str] = None, completed: Optional[bool] = False):
    filters = {"priority": priority, "tags": tags, "due_date": due_date, "completed": completed}
    todos = await todo_manager.list_todos(filters)
    return {"todos": todos}

# --- Voice Agent/Prompt Processing Endpoint ---
class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"

@app.post("/agent/process-prompt")
async def process_prompt(req: PromptRequest):
    response = await process_user_prompt(req.prompt, req.session_id)
    return response

# --- Outbound Call Agent Example Endpoint (stub) ---
class OutboundCallRequest(BaseModel):
    name: str
    appointment_time: str
    dial_info: Dict[str, Any]

@app.post("/call/outbound")
async def outbound_call(req: OutboundCallRequest, background_tasks: BackgroundTasks):
    # This is a stub. In production, you would trigger the OutboundCaller logic in a background task or job queue.
    background_tasks.add_task(lambda: logger.info(f"Would start outbound call for {req.name} at {req.appointment_time} with {req.dial_info}"))
    return {"status": "started", "details": req.dial_info}

# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok"}
