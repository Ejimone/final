import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Add this at the very top
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Langchain imports
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_community.vectorstores import FAISS

# Other imports
import google.generativeai as genai
import serpapi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessorConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ('\n\n', '\n', '.', ',')
    similarity_threshold: float = 0.5

class RAGProcessor:
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        """Initialize the RAG processor with configuration"""
        load_dotenv()
        self.config = config or DocumentProcessorConfig()
        self._setup_logging()
        self._setup_credentials()
        self.model = self._initialize_model()
        self.vector_store = None
        self.embeddings = self._initialize_embeddings()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Suppress FAISS and other verbose logs
        logging.getLogger('faiss').setLevel(logging.WARNING)
        logging.getLogger('pikepdf._core').setLevel(logging.WARNING)
        logging.getLogger('oauth2client').setLevel(logging.WARNING)
        logging.getLogger('googleapiclient').setLevel(logging.WARNING)

    def _setup_credentials(self) -> None:
        """Setup API credentials"""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY not found in environment")

    def _validate_environment(self) -> None:
        """Validate required environment variables"""
        required_vars = ["SERPAPI_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    def _initialize_model(self) -> Any:
        """Initialize AI model with fallback strategy"""
        try:
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}. Falling back to OpenAI.")
            try:
                return OpenAI(temperature=0.7)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize both Gemini and OpenAI: {e}")

    def _initialize_embeddings(self) -> Any:
        """Initialize embeddings with fallback strategy"""
        try:
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini embeddings: {e}")
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("Neither Gemini nor OpenAI credentials available")
            return OpenAIEmbeddings()

    async def process_documents(self, 
                              urls: Optional[List[str]] = None, 
                              pdf_paths: Optional[List[str]] = None,
                              word_paths: Optional[List[str]] = None,
                              ppt_paths: Optional[List[str]] = None) -> bool:
        """Process documents from multiple sources"""
        docs = []
        
        if urls:
            docs.extend(await self._process_urls(urls))
        if pdf_paths:
            docs.extend(await self._process_files(pdf_paths, "pdf"))
        if word_paths:
            docs.extend(await self._process_files(word_paths, "docx"))
        if ppt_paths:
            docs.extend(await self._process_files(ppt_paths, "ppt"))
            
        if not docs:
            logger.warning("No documents were processed")
            return False
            
        return await self._create_vector_store(docs)

    async def _process_urls(self, urls: List[str]) -> List[Any]:
        """Process documents from URLs"""
        try:
            loop = asyncio.get_event_loop()
            loader = UnstructuredURLLoader(urls=urls)
            return await loop.run_in_executor(None, loader.load)
        except Exception as e:
            logger.error(f"Error loading URLs: {e}")
            return []

    async def _process_files(self, file_paths: List[str], file_type: str) -> List[Any]:
        """Generic file processor for different document types"""
        docs = []
        for path in file_paths:
            full_path = Path("./documents") / path
            if not full_path.exists():
                logger.warning(f"File path does not exist: {full_path}")
                continue
            
            try:
                if file_type == "pdf":
                    loader = PyPDFLoader(str(full_path))
                    load_func = loader.load_and_split
                elif file_type == "docx":
                    loader = UnstructuredWordDocumentLoader(str(full_path))
                    load_func = loader.load
                elif file_type == "ppt":
                    loader = UnstructuredPowerPointLoader(str(full_path))
                    load_func = loader.load
                else:
                    continue

                loop = asyncio.get_event_loop()
                loaded_docs = await loop.run_in_executor(None, load_func)
                docs.extend(loaded_docs)
            except Exception as e:
                logger.error(f"Error loading {file_type} file {full_path}: {e}")
        
        return docs

    async def _create_vector_store(self, docs: List[Any]) -> bool:
        """Create vector store from processed documents"""
        if not docs:
            logger.error("No documents to process")
            return False

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators
            )
            splits = text_splitter.split_documents(docs)
            
            if not splits:
                logger.error("No text splits generated")
                return False

            logger.info(f"Creating vector store with {len(splits)} text splits")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            return True

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False

    async def ask_question(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Query the vector store and generate an answer"""
        if not self.vector_store:
            return {"error": "No documents have been processed yet"}

        try:
            # Add progress indication
            print("Searching through documents...", end='\r')
            
            # Get relevant documents with similarity scores
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            if not docs_with_scores:
                return self._generate_fallback_answer(query)

            print("Generating answer...", end='\r')
            # Process similarity scores
            docs, distances = zip(*docs_with_scores)
            similarities = [1 / (1 + distance) for distance in distances]
            max_similarity = max(similarities)

            # Check similarity threshold
            if max_similarity < self.config.similarity_threshold:
                return await self._handle_low_confidence(query)

            # Generate context-based answer
            context = "\n".join(doc.page_content for doc in docs)
            prompt = f"""Answer the question based only on the following context. 
            If you don't know the answer, say you don't know. Be concise.
            
            Context: {context}
            
            Question: {query}
            
            Answer:"""
            
            response = self.model.generate_content(prompt)
            answer = response.text

            # Check for uncertainty in the answer
            if self._is_uncertain_answer(answer):
                return await self._handle_uncertain_answer(query, answer)

            return {
                "answer": answer,
                "sources": [doc.metadata for doc in docs],
                "confidence": max_similarity,
                "based_on_documents": True
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {"error": str(e)}

    def _generate_fallback_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer when no documents are available"""
        prompt = f"Answer the following question: {query}"
        response = self.model.generate_content(prompt)
        return {
            "answer": f"I couldn't find relevant information. Here's a general answer: {response.text}",
            "sources": [],
            "confidence": 0.0,
            "based_on_documents": False
        }

    async def _handle_low_confidence(self, query: str) -> Dict[str, Any]:
        """Handle cases with low confidence scores"""
        prompt = f"Answer the following question: {query}"
        response = self.model.generate_content(prompt)
        web_results = await self._web_search(query)
        
        answer = f"I'm not sure based on the documents. Here's a general answer: {response.text}"
        if web_results:
            answer += "\n\nHere are some web results that might help:\n"
            answer += "\n".join([f"- {res['title']}: {res['snippet']}" for res in web_results[:3]])
        
        return {
            "answer": answer,
            "sources": [],
            "confidence": 0.0,
            "based_on_documents": False
        }

    def _is_uncertain_answer(self, answer: str) -> bool:
        """Check if the answer contains uncertainty indicators"""
        uncertainty_phrases = [
            "don't know", "not sure", "no information", 
            "not provided", "not mentioned", "I cannot"
        ]
        return any(phrase in answer.lower() for phrase in uncertainty_phrases)

    async def _handle_uncertain_answer(self, query: str, original_answer: str) -> Dict[str, Any]:
        """Handle uncertain answers with web search fallback"""
        web_results = await self._web_search(query)
        answer = original_answer
        
        if web_results:
            answer += "\n\nHere are some web search results that might help:\n"
            answer += "\n".join([f"- {res['title']}: {res['snippet']}" for res in web_results[:3]])
        else:
            answer += "\n\nI couldn't find additional information through web search."
        
        return {
            "answer": answer,
            "sources": [],
            "confidence": 0.0,
            "based_on_documents": False
        }

    async def _web_search(self, query: str) -> List[Dict[str, str]]:
        """Fallback web search when no good answer is found"""
        try:
            search = serpapi.GoogleSearch({
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY")
            })
            results = search.get_dict()
            return [{
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "link": result.get("link")
            } for result in results.get("organic_results", [])]
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a web search and return parsed results.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
        
    Returns:
        List of dictionaries containing search results with 'title' and 'snippet' keys
    
    Raises:
        RequestException: If network request fails
    """
    try:
        # Add delay to respect rate limits
        time.sleep(1)
        
        # Construct search URL
        encoded_query = quote_plus(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        # Send request with headers to mimic browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse results
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = []
        
        # Extract search result divs
        results = soup.find_all('div', class_='g')
        
        for result in results[:num_results]:
            title_elem = result.find('h3')
            snippet_elem = result.find('div', class_='VwiC3b')
            
            if title_elem and snippet_elem:
                search_results.append({
                    'title': title_elem.get_text(),
                    'snippet': snippet_elem.get_text()
                })
                
        return search_results

    except requests.RequestException as e:
        logger.error(f"Web search failed: {str(e)}")
        raise

async def RAG():
    try:
        print("Initializing RAG processor...")
        processor = RAGProcessor()
        
        print("\nDocument Input Options:")
        print("1. For URLs: Enter the full URLs separated by commas")
        print("2. For local files: Enter filenames from the documents folder")
        print("3. Press Enter to skip any option\n")
        
        urls = input("Enter URLs (comma-separated): ").split(',')
        urls = [u.strip() for u in urls if u.strip()]
        
        if urls:
            print("\nProcessing URLs...")
        
        pdf_paths = input("Enter comma-separated PDF filenames from documents folder: ").split(',')
        pdf_paths = [p.strip() for p in pdf_paths if p.strip()]
        
        word_paths = input("Enter comma-separated Word filenames from documents folder: ").split(',')
        word_paths = [p.strip() for p in word_paths if p.strip()]
        
        ppt_paths = input("Enter comma-separated PowerPoint filenames from documents folder: ").split(',')
        ppt_paths = [p.strip() for p in ppt_paths if p.strip()]
        
        if not any([urls, pdf_paths, word_paths, ppt_paths]):
            print("No documents provided for processing")
            return
            
        success = await processor.process_documents(
            urls=urls,
            pdf_paths=pdf_paths,
            word_paths=word_paths,
            ppt_paths=ppt_paths
        )
        
        if not success:
            print("\n❌ Failed to process documents")
            return
        
        print("\n✅ Documents processed successfully!")
        print("\nYou can now ask questions about the documents.")
        print("Type 'exit' to quit the program.\n")
            
        while True:
            question = input("\n🔍 Ask a question: ").strip()
            if question.lower() == 'exit':
                break
                
            result = await processor.ask_question(question)
            
            if 'error' in result:
                print(f"\n❌ Error: {result['error']}")
            else:
                print("\n📝 Answer:")
                print(result['answer'])
                
                if result.get('sources'):
                    print("\n📚 Sources:")
                    for source in result['sources']:
                        print(f"- {source.get('source', 'Unknown')}")
            
            print("\n" + "="*50)

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        print("\nThank you for using the RAG processor!")

if __name__ == "__main__":
    asyncio.run(RAG())