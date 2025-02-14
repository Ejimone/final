import os
import asyncio
from typing import Dict, Any, Optional, Union
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI as LangchainOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from SendEmail import AIService as EmailService, sendEmail
from datetime import datetime
import uuid
import json
from Weather import WeatherService, WeatherServiceError
from Rag import RAGProcessor
# Initialize logging
from exceptions import EmailServiceError
from Config import Config
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()



# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize email service with retries
_email_service = EmailService()

async def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Send email using the shared EmailService instance with async wrapper"""
    try:
        message = _email_service.construct_message(to=to, subject=subject, body=body)
        result = _email_service.send_email(message)
        return {
            "status": "success",
            "message_id": result['message_id'],
            "service": result['service']
        }
    except EmailServiceError as e:
        logger.error(f"Email service error: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception(f"Unexpected error sending email: {str(e)}")
        return {"status": "error", "message": "Internal server error"}

def initialize_llm():
    """Initialize LLM with proper fallback handling"""
    try:
        # Try Gemini first
        logger.info("Initializing Gemini model")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        try:
            model = genai.GenerativeModel('gemini-pro')
            # Test the model
            response = model.generate_content("test")
            logger.info("Gemini model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Gemini test failed: {e}")
            raise

    except Exception as e:
        logger.error(f"Primary model initialization failed: {e}, falling back to OpenAI")
        try:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Test the model
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}]
            )
            if response and response.choices:
                logger.info("Fallback to OpenAI successful")
                return openai_client
            raise Exception("OpenAI model test failed")
        except Exception as e:
            logger.error(f"Both model initializations failed: {e}")
            raise

def generate_llm_response(model_type: str, api_key: str, messages: list) -> str:
    """Generate response using the specified model"""
    try:
        if model_type.lower() == 'gemini':
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(messages[-1]['content'])
            return response.text if hasattr(response, 'text') else str(response)
        else:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=Config.OPENAI_LLM,
                messages=messages
            )
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}")
        raise

def create_chain(llm: Any) -> Any:
    """Create conversation chain with proper type hints"""
    prompt_template = PromptTemplate(
        input_variables=["AI_Agent"],
        template="I want you to be an {AI_Agent} assistant, you'll be helping me out with some tasks."
    )
    
    if isinstance(llm, LangchainOpenAI):
        return initialize_agent(
            tools=load_tools(["serpapi", "llm-math"], llm=llm),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory()
        )
    return llm  # Return Gemini model directly

async def create_task(task_type: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """Create and process a task with validation"""
    try:
        llm = initialize_llm()
        if not llm:
            return {"status": "error", "message": "Failed to initialize AI service"}

        task = {
            "task_id": str(uuid.uuid4()),
            "type": task_type,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }

        if task_type.lower() == "email":
            return await handle_email_task(task, task_details, llm)
            
        return {"status": "error", "message": "Unsupported task type"}
        
    except Exception as e:
        logger.error(f"Task creation error: {str(e)}")
        return {"status": "error", "message": "Task processing failed"}

async def handle_email_task(task: Dict[str, Any], details: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """Process email-specific task logic"""
    required_fields = ["to_email", "to_name", "from_name", "purpose"]
    if not all(field in details for field in required_fields):
        return {"status": "error", "message": "Missing required email fields"}

    try:
        # Generate email content using LLM with proper context
        prompt = f"""
        Write a professional email with the following details:
        - From: {details['from_name']}
        - To: {details['to_name']}
        - Purpose: {details['purpose']}

        Requirements:
        1. Start with a proper greeting using the receiver's name
        2. Write a professional but friendly email body based on the purpose
        3. End with a professional signature using the sender's name
        4. Keep the tone warm and professional
        5. Make sure the content is clear and concise
        """

        if isinstance(llm, LangchainOpenAI):
            response = llm.invoke(prompt)
        else:
            response = llm.generate_content(prompt)
            
        email_content = response.text if hasattr(response, 'text') else response
        
        # Generate a suitable subject line
        subject_prompt = f"Generate a professional subject line for an email with this purpose: {details['purpose']}"
        if isinstance(llm, LangchainOpenAI):
            subject_response = llm.invoke(subject_prompt)
        else:
            subject_response = llm.generate_content(subject_prompt)
        
        subject = subject_response.text if hasattr(subject_response, 'text') else subject_response
        subject = subject.strip().strip('"').strip("'")  # Clean up the subject line

        # Send email
        email_result = await send_email(
            to=details['to_email'],
            subject=subject,
            body=email_content
        )

        if email_result["status"] == "error":
            raise EmailServiceError(email_result["message"])

        task["email_result"] = email_result
        task["status"] = "completed"
        return {"status": "success", "task": task}

    except EmailServiceError as e:
        logger.error(f"Email failed: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Email processing error: {str(e)}")
        return {"status": "error", "message": "Email composition failed"}

async def check_services() -> Dict[str, Any]:
    """Initialize and validate core services asynchronously"""
    try:
        llm = initialize_llm()
        if not llm:
            raise RuntimeError("Failed to initialize any AI service")

        # Validate email service
        if not _email_service.check_connection():
            logger.warning("Email service connection check failed")
            
        return {
            "status": "success",
            "message": "Services initialized successfully",
            "services": {
                "llm": "OpenAI" if isinstance(llm, LangchainOpenAI) else "Gemini",
                "email": "Connected" if _email_service.check_connection() else "Not Connected"
            }
        }

    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}")
        return {"status": "error", "message": str(e)}

class AIAssistant:
    """Unified AI Assistant managing multiple services"""
    def __init__(self):
        self.weather_service = None
        self.email_service = None
        self.rag_processor = None
        self.llm = None

    async def initialize_services(self) -> Dict[str, Any]:
        """Initialize all services"""
        try:
            # Initialize LLM
            self.llm = initialize_llm()
            
            # Initialize Email Service
            self.email_service = _email_service
            
            # Initialize Weather Service
            self.weather_service = WeatherService()
            
            # Initialize RAG Processor
            self.rag_processor = RAGProcessor()
            
            return {
                "status": "success",
                "services": {
                    "llm": "OpenAI" if isinstance(self.llm, LangchainOpenAI) else "Gemini",
                    "email": "Connected" if self.email_service.check_connection() else "Not Connected",
                    "weather": "Ready",
                    "rag": "Ready"
                }
            }
        except Exception as e:
            logger.error(f"Service initialization error: {e}")
            return {"status": "error", "message": str(e)}

    async def cleanup(self):
        """Cleanup all services"""
        if self.weather_service:
            await self.weather_service.close()
        if self.rag_processor:
            await self.rag_processor.close()

async def interactive_mode():
    """Enhanced interactive interface with multiple services"""
    print("\n" + "="*50)
    print("AI Assistant Interface".center(50))
    print("="*50 + "\n")

    assistant = AIAssistant()
    try:
        # Initialize all services
        init_result = await assistant.initialize_services()
        if init_result["status"] != "success":
            print(f"⚠️ Service initialization failed: {init_result.get('message', 'Unknown error')}")
            return

        print("✅ Services initialized successfully!")
        print("\nAvailable Services:")
        for service, status in init_result["services"].items():
            print(f"📍 {service.upper()}: {status}")

        while True:
            print("\nAvailable Tasks:")
            print("1. Send AI-generated email")
            print("2. Check weather information")
            print("3. Process documents (RAG)")
            print("4. Ask questions about processed documents")
            print("5. Exit")

            choice = input("\nChoose a task (1-5): ").strip()

            if choice == "5":
                break
            elif choice == "1":
                # Email task
                print("\nEmail Composition:")
                print("-----------------")
                email_details = {
                    "from_name": input("Your name: ").strip(),
                    "to_name": input("Recipient's name: ").strip(),
                    "to_email": input("Recipient's email address: ").strip(),
                    "purpose": input("What is the purpose of this email? (Be specific): ").strip()
                }

                print("\n📝 Generating professional email...")
                result = await create_task("email", email_details)
                
                if result["status"] == "success":
                    print("\n✅ Email sent successfully!")
                    print(f"Message ID: {result['task']['email_result']['message_id']}")
                else:
                    print(f"\n❌ Task failed: {result.get('message', 'Unknown error')}")

            elif choice == "2":
                # Weather task
                print("\nWeather Information:")
                print("------------------")
                location = input("Enter location (city, country code): ").strip()
                
                try:
                    result = await assistant.weather_service.get_weather(location)
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
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")

            elif choice == "3":
                # RAG Document Processing
                print("\nDocument Processing:")
                print("------------------")
                print("Enter document sources (press Enter to skip):")
                
                urls = input("\nEnter URLs (comma-separated): ").split(',')
                urls = [u.strip() for u in urls if u.strip()]
                
                pdf_paths = input("Enter PDF filenames from documents folder: ").split(',')
                pdf_paths = [p.strip() for p in pdf_paths if p.strip()]
                
                if not any([urls, pdf_paths]):
                    print("⚠️ No documents provided")
                    continue
                
                success = await assistant.rag_processor.process_documents(
                    urls=urls,
                    pdf_paths=pdf_paths
                )
                
                if success:
                    print("\n✅ Documents processed successfully!")
                else:
                    print("\n❌ Failed to process documents")

            elif choice == "4":
                # RAG Query
                if not assistant.rag_processor.vector_store:
                    print("\n⚠️ No documents have been processed yet. Please process documents first.")
                    continue
                
                print("\nDocument Query:")
                print("--------------")
                question = input("Enter your question: ").strip()
                
                print("\nSearching documents...")
                result = await assistant.rag_processor.ask_question(question)
                
                if 'error' in result:
                    print(f"\n❌ Error: {result['error']}")
                else:
                    print("\n📝 Answer:")
                    print(result['answer'])
                    if result.get('sources'):
                        print("\n📚 Sources:")
                        for source in result['sources']:
                            print(f"- {source.get('source', 'Unknown')}")

            else:
                print("\n❌ Invalid choice. Please try again.")

            print("\n" + "="*50)
            if input("\nContinue using AI Assistant? (y/N): ").lower() != 'y':
                break

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        logger.exception("Interactive mode error")
    finally:
        print("\nCleaning up services...")
        await assistant.cleanup()
        print("Thank you for using AI Assistant!")

if __name__ == "__main__":
    asyncio.run(interactive_mode())