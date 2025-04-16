import os
import json
import logging
import asyncio
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pytz
from SendEmail import AIService
from Weather import WeatherService
from WebScrapeAndProcess import scrape_webpages_with_serpapi, summarize_content
from RealTimeSearch import real_time_search
from Ai import initialize_llm
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Service configuration parameters"""
    SCOPES: List[str] = None
    
    def __post_init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/calendar',
            'https://www.googleapis.com/auth/calendar.events'
        ]

class PathConfig:
    """Path configuration"""
    BASE_DIR = Path(__file__).parent
    CREDENTIALS_PATH = BASE_DIR / 'credentials.json'
    TOKEN_PATH = BASE_DIR / 'token.json'
    

@dataclass
class TodoItem:
    """Represents a todo item with all necessary details"""
    id: str
    title: str
    description: str
    due_date: Optional[datetime] = None
    priority: str = "medium"  # low, medium, high
    status: str = "pending"   # pending, completed, cancelled
    reminders: List[datetime] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    calendar_event_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.reminders is None:
            self.reminders = []
        if self.tags is None:
            self.tags = []

class TodoManager:
    """Manages todo items with calendar and email integration"""
    
    def __init__(self):
        self.todos: Dict[str, TodoItem] = {}
        self.config = ServiceConfig()
        self._initialize_services()
        self.todo_file = Path(__file__).parent / 'todos.json'
        self.load_todos()
    
    def _initialize_services(self) -> None:
        """Initialize all required services"""
        self.ai_service = AIService()  # This already has retry logic built in
        self.weather_service = WeatherService()
        self._setup_google_calendar()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _setup_google_calendar(self):
        """Set up Google Calendar API connection with retries"""
        creds = None
        logger.info(f"Checking if token file exists at: {PathConfig.TOKEN_PATH}")
        
        if PathConfig.TOKEN_PATH.exists():
            logger.info("Token file exists. Attempting to load credentials from token file.")
            try:
                with open(PathConfig.TOKEN_PATH, 'rb') as token:
                    creds = pickle.load(token)
                logger.info("Credentials loaded from token file.")
            except Exception as e:
                logger.error(f"Error loading credentials from token file: {e}")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials.")
                try:
                    creds.refresh(Request())
                    logger.info("Credentials refreshed successfully.")
                except Exception as e:
                    logger.error(f"Error refreshing credentials: {e}")
                    creds = None
            
            if not creds:
                if not PathConfig.CREDENTIALS_PATH.exists():
                    raise FileNotFoundError(
                        "credentials.json not found. Please download it from Google Cloud Console "
                        "and place it in the same directory as Todo.py"
                    )
                
                logger.info("Using existing credentials from credentials.json")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(PathConfig.CREDENTIALS_PATH),
                        self.config.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    logger.info("New credentials obtained successfully.")
                    
                    # Save the credentials for future use
                    with open(PathConfig.TOKEN_PATH, 'wb') as token:
                        pickle.dump(creds, token)
                    logger.info("New credentials saved to token file.")
                except Exception as e:
                    logger.error(f"Failed to initialize credentials: {e}")
                    raise

        try:
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            logger.info("Calendar service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to build calendar service: {e}")
            raise

    def load_todos(self):
        """Load todos from JSON file"""
        if self.todo_file.exists():
            try:
                with open(self.todo_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        # Convert string dates back to datetime
                        if item.get('due_date'):
                            item['due_date'] = datetime.fromisoformat(item['due_date'])
                        if item.get('created_at'):
                            item['created_at'] = datetime.fromisoformat(item['created_at'])
                        if item.get('updated_at'):
                            item['updated_at'] = datetime.fromisoformat(item['updated_at'])
                        if item.get('reminders'):
                            item['reminders'] = [datetime.fromisoformat(r) for r in item['reminders']]
                        self.todos[item['id']] = TodoItem(**item)
            except Exception as e:
                logger.error(f"Error loading todos: {e}")

    def save_todos(self):
        """Save todos to JSON file"""
        try:
            data = []
            for todo in self.todos.values():
                todo_dict = todo.__dict__.copy()
                # Convert datetime objects to ISO format strings
                if todo_dict.get('due_date'):
                    todo_dict['due_date'] = todo_dict['due_date'].isoformat()
                if todo_dict.get('created_at'):
                    todo_dict['created_at'] = todo_dict['created_at'].isoformat()
                if todo_dict.get('updated_at'):
                    todo_dict['updated_at'] = todo_dict['updated_at'].isoformat()
                if todo_dict.get('reminders'):
                    todo_dict['reminders'] = [r.isoformat() for r in todo_dict['reminders']]
                data.append(todo_dict)
            
            with open(self.todo_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving todos: {e}")

    async def add_todo(self, title: str, description: str, due_date: Optional[str] = None,
                      priority: str = "medium", tags: List[str] = None,
                      reminders: List[str] = None, schedule_meeting: bool = False) -> TodoItem:
        """Add a new todo item with optional calendar event and send to user's email"""
        from uuid import uuid4
        
        # Parse due date if provided
        parsed_due_date = None
        if due_date:
            try:
                parsed_due_date = datetime.fromisoformat(due_date)
            except ValueError:
                logger.error(f"Invalid due date format: {due_date}")
                raise ValueError("Due date must be in ISO format (YYYY-MM-DDTHH:MM:SS)")

        # Create todo item
        todo = TodoItem(
            id=str(uuid4()),
            title=title,
            description=description,
            due_date=parsed_due_date,
            priority=priority,
            tags=tags or [],
            reminders=[datetime.fromisoformat(r) for r in reminders] if reminders else []
        )
        
        # Add calendar event if requested
        if schedule_meeting and parsed_due_date:
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': parsed_due_date.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': (parsed_due_date + timedelta(hours=1)).isoformat(),
                    'timeZone': 'UTC',
                },
                'reminders': {
                    'useDefault': True
                }
            }
            
            try:
                created_event = self.calendar_service.events().insert(
                    calendarId='primary',
                    body=event
                ).execute()
                todo.calendar_event_id = created_event['id']
                logger.info(f"Calendar event created successfully for todo: {title}")
            except Exception as e:
                logger.error(f"Failed to create calendar event: {e}")
        
        self.todos[todo.id] = todo
        self.save_todos()

        # Send todo to user's email
        user_email = os.getenv("USER_EMAIL")
        if user_email:
            subject = f"New Todo: {title}"
            body = f"""
            New Todo Item Created

            Title: {title}
            Description: {description}
            Due Date: {parsed_due_date.strftime('%Y-%m-%d %H:%M') if parsed_due_date else 'No due date'}
            Priority: {priority}
            Status: {todo.status}
            {"Calendar Event: Created" if todo.calendar_event_id else ""}
            """
            try:
                result = await self.ai_service.send_email(user_email, subject, body)
                if result.get("status") == "success":
                    logger.info(f"Todo notification email sent successfully to {user_email}")
                else:
                    logger.error(f"Failed to send todo email: {result.get('message')}")
            except Exception as e:
                logger.error(f"Error sending todo email: {e}")

        return todo

    async def complete_todo(self, todo_id: str) -> bool:
        """Mark a todo as complete and handle related cleanup"""
        if todo_id not in self.todos:
            return False
            
        todo = self.todos[todo_id]
        todo.status = "completed"
        todo.updated_at = datetime.now()
        
        # Remove calendar event if it exists
        if todo.calendar_event_id:
            try:
                self.calendar_service.events().delete(
                    calendarId='primary',
                    eventId=todo.calendar_event_id
                ).execute()
            except Exception as e:
                logger.error(f"Failed to delete calendar event: {e}")
        
        self.save_todos()
        return True

    async def update_todo(self, todo_id: str, **updates) -> Optional[TodoItem]:
        """Update a todo item with new values"""
        if todo_id not in self.todos:
            return None
            
        todo = self.todos[todo_id]
        
        for key, value in updates.items():
            if hasattr(todo, key):
                if key == 'due_date' and value:
                    value = datetime.fromisoformat(value)
                elif key == 'reminders' and value:
                    value = [datetime.fromisoformat(r) for r in value]
                setattr(todo, key, value)
        
        todo.updated_at = datetime.now()
        
        # Update calendar event if it exists and due_date changed
        if todo.calendar_event_id and 'due_date' in updates:
            event = {
                'summary': todo.title,
                'description': todo.description,
                'start': {
                    'dateTime': todo.due_date.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': (todo.due_date + timedelta(hours=1)).isoformat(),
                    'timeZone': 'UTC',
                }
            }
            try:
                self.calendar_service.events().update(
                    calendarId='primary',
                    eventId=todo.calendar_event_id,
                    body=event
                ).execute()
            except Exception as e:
                logger.error(f"Failed to update calendar event: {e}")
        
        self.save_todos()
        return todo

    async def delete_todo(self, todo_id: str) -> bool:
        """Delete a todo item and its calendar event"""
        if todo_id not in self.todos:
            return False
            
        todo = self.todos[todo_id]
        
        # Delete calendar event if it exists
        if todo.calendar_event_id:
            try:
                self.calendar_service.events().delete(
                    calendarId='primary',
                    eventId=todo.calendar_event_id
                ).execute()
            except Exception as e:
                logger.error(f"Failed to delete calendar event: {e}")
        
        del self.todos[todo_id]
        self.save_todos()
        return True

    async def get_todos(self, status: Optional[str] = None, tags: Optional[List[str]] = None,
                       priority: Optional[str] = None) -> List[TodoItem]:
        """Get filtered todo items"""
        todos = self.todos.values()
        
        if status:
            todos = [t for t in todos if t.status == status]
        if tags:
            todos = [t for t in todos if any(tag in t.tags for tag in tags)]
        if priority:
            todos = [t for t in todos if t.priority == priority]
            
        return sorted(todos, key=lambda x: x.due_date if x.due_date else datetime.max)

    async def send_reminder_email(self, todo: TodoItem, to_email: str) -> bool:
        """Send a reminder email for a todo item"""
        subject = f"Reminder: {todo.title}"
        body = f"""
        Task Reminder

        Title: {todo.title}
        Description: {todo.description}
        Due Date: {todo.due_date.strftime('%Y-%m-%d %H:%M') if todo.due_date else 'No due date'}
        Priority: {todo.priority}
        Status: {todo.status}
        
        Please take action on this task.
        """
        
        try:
            result = await self.ai_service.send_email(to_email, subject, body)
            return result.get("status") == "success"
        except Exception as e:
            logger.error(f"Failed to send reminder email: {e}")
            return False

    async def check_reminders(self) -> None:
        """Check and process due reminders"""
        now = datetime.now()
        for todo in self.todos.values():
            if todo.status != "completed" and todo.reminders:
                for reminder_time in todo.reminders:
                    if reminder_time <= now:
                        # Send reminder email (assumes user email is configured)
                        await self.send_reminder_email(todo, os.getenv("USER_EMAIL"))
                        # Remove the processed reminder
                        todo.reminders.remove(reminder_time)
                        self.save_todos()

    async def get_weather_for_meeting(self, todo: TodoItem) -> Optional[Dict[str, Any]]:
        """Get weather forecast for a meeting"""
        if not todo.due_date:
            return None
            
        try:
            # Use the weather service to get forecast
            location = input("which location:\n")  # Replace with actual location if available
            weather_data = await self.weather_service.get_weather(location)
            if weather_data["status"] == "error":
                logger.error(f"Weather service error: {weather_data['message']}")
                return None
            return weather_data
        except Exception as e:
            logger.error(f"Failed to get weather data: {e}")
            return None

    async def get_related_info(self, todo: TodoItem) -> Optional[Dict[str, Any]]:
        """Get related real-time information for a todo item"""
        try:
            # Use WebScrapeAndProcess to get relevant information
            search_query = f"{todo.title} {' '.join(todo.tags)}"
            search_results = await scrape_webpages_with_serpapi(search_query)
            
            if search_results.get("status") == "success":
                # Summarize the results
                gemini_model = initialize_llm()
                summary = await summarize_content(str(search_results.get("data")), gemini_model)
                return {"summary": summary.replace('#', ''), "source": search_results}
                
            return None
        except Exception as e:
            logger.error(f"Failed to get related information: {e}")
            return None

async def main():
    """Main function for testing the TodoManager"""
    todo_manager = TodoManager()
    
    # Test adding a todo
    print("\nAdding a new todo...")
    new_todo = await todo_manager.add_todo(
        title="Important Meeting",
        description="Team sync meeting with project updates",
        due_date="2024-03-01T10:00:00",
        priority="high",
        tags=["meeting", "project"],
        reminders=["2024-03-01T09:00:00"],
        schedule_meeting=True
    )
    print(f"Added todo: {new_todo.title}")

    # Test getting todos
    print("\nGetting all high priority todos...")
    high_priority_todos = await todo_manager.get_todos(priority="high")
    for todo in high_priority_todos:
        print(f"- {todo.title} (Due: {todo.due_date})")

    # Test updating a todo
    if new_todo:
        print("\nUpdating todo...")
        updated_todo = await todo_manager.update_todo(
            new_todo.id,
            description="Updated: Team sync meeting with project updates and demos"
        )
        if updated_todo:
            print(f"Updated todo description: {updated_todo.description}")

    # Test getting weather for a meeting
    if new_todo:
        print("\nGetting weather for meeting...")
        weather = await todo_manager.get_weather_for_meeting(new_todo)
        if weather:
            print(f"Weather forecast: {weather}")

    # Test getting related information
    if new_todo:
        print("\nGetting related information...")
        related_info = await todo_manager.get_related_info(new_todo)
        if related_info:
            print(f"Related information summary: {related_info.get('summary')}")

    # Test completing a todo
    if new_todo:
        print("\nCompleting todo...")
        completed = await todo_manager.complete_todo(new_todo.id)
        print(f"Todo completed: {completed}")

if __name__ == "__main__":
    asyncio.run(main())