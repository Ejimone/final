import os
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, Any

import re
from Ai import initialize_llm
import asyncio
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import re

def clean_text(text: str) -> str:
    """Clean scraped text by removing extra whitespace and special characters"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
    return text.strip()

async def scrape_webpages_with_serpapi(query: str) -> Dict[str, Any]:
    """Scrape webpages using SerpAPI and extract real‑time content"""
    try:
        params = {
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google",
            "q": query,
            "num": 5,
            "gl": "us"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            raw_results = response.json()
            # Extract relevant URLs from the search results
            urls = [result.get("link") for result in raw_results.get("organic_results", [])]
            scraped_data = []

            for url in urls:
                try:
                    # Use the generic scrape_url to extract the page's main content
                    content = await scrape_url(url)
                    if content:
                        scraped_data.append({"url": url, "content": content})
                    else:
                        logger.warning(f"No content could be extracted from {url}.")
                except Exception as e:
                    logger.error(f"Error scraping {url}: {str(e)}")

            return {
                "status": "success",
                "data": scraped_data
            } if scraped_data else {"status": "error", "message": "No content scraped."}
        else:
            return {"status": "error", "message": f"Error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        return {"status": "error", "message": str(e)}



async def scrape_url(url: str) -> str:
    """Scrape content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
        
        # Extract main content
        main_content = ""
        
        # Try to find main content container
        content_tags = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'(content|article|post|entry)'))
        if content_tags:
            main_content = content_tags[0].get_text()
        else:
            # Fallback to paragraphs
            paragraphs = soup.find_all('p')
            main_content = ' '.join(p.get_text() for p in paragraphs)
        
        return clean_text(main_content)
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

async def summarize_content(content: str, gemini_model) -> str:
    """Summarize content using Gemini"""
    try:
        summary_prompt = f"""
        Summarize this content concisely and clearly:
        {content[:10000]}  # Limit content length to avoid token limits
        
        Provide:
        1. Key Points
        2. Important Facts
        3. Relevant Dates
        4. Additional Context
        
        Format in markdown for readability.
        """
        
        summary = gemini_model.generate_content(summary_prompt)
        return summary.text.strip()
    except Exception as e:
        print(f"Error summarizing content: {str(e)}")
        return content[:1000] + "..."  # Fallback to truncated content

async def web_search(query: str) -> Dict[str, Any]:
    """Perform web search, scrape URLs, and summarize content"""
    try:
        # Direct API call to SerpAPI
        params = {
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google",
            "q": query,
            "num": 5,
            "gl": "us"
        }
        
        search_response = requests.get("https://serpapi.com/search", params=params)
        raw_results = search_response.json()
        
        print(f"Search query: {query}")  # Debug log
        
        if "error" in raw_results:
            return {
                "status": "success",
                "data": f"Search API error: {raw_results['error']}",
                "type": "search"
            }
        
        search_results = raw_results.get("organic_results", [])
        
        if not search_results:
            return {
                "status": "success",
                "data": f"No search results found for query: {query}",
                "type": "search"
            }
        
        gemini_model = initialize_llm()
        if not gemini_model:
            return {
                "status": "error",
                "message": "Could not initialize Gemini model"
            }
        
        # Process each result individually for detailed summaries
        detailed_summaries = []
        for result in search_results[:3]:
            try:
                url = result.get('link', '')
                title = result.get('title', 'No title')
                snippet = result.get('snippet', '')
                
                # Scrape content from URL
                content = await scrape_url(url) if url else ""
                
                # Generate detailed summary for each source
                source_prompt = f"""
                Provide a comprehensive analysis of this content about {query}:

                Title: {title}
                URL: {url}
                Content: {content if content else snippet}

                Requirements:
                1. Write at least 250 words
                2. Include specific details, facts, and figures
                3. Mention dates and relevant context
                4. Analyze the significance of the information
                5. Include any controversies or different perspectives
                6. Explain how this information relates to {query}

                Format the response with these sections:
                1. Main Points (detailed explanation)
                2. Key Facts and Figures
                3. Context and Background
                4. Analysis and Implications
                5. Related Developments
                """
                
                source_summary = gemini_model.generate_content(source_prompt)
                detailed_summaries.append({
                    "title": title,
                    "url": url,
                    "summary": source_summary.text.strip()
                })
                
            except Exception as e:
                print(f"Error processing result {url}: {str(e)}")
                continue
        
        # Generate comprehensive overview
        overview_prompt = f"""
        Create a comprehensive overview of {query} based on all these sources:

        {[summary['summary'] for summary in detailed_summaries]}

        Requirements:
        1. Write at least 300 words
        2. Synthesize information from all sources
        3. Highlight key themes and patterns
        4. Include contrasting viewpoints if any
        5. Provide temporal context (recent vs historical information)
        6. Explain broader implications

        Format with these sections:
        1. Executive Summary
        2. Detailed Analysis
        3. Key Developments
        4. Implications and Future Outlook
        """
        
        overview_response = gemini_model.generate_content(overview_prompt)
        overview = overview_response.text.strip()
        
        # Format the final output
        formatted_result = f"""
# Comprehensive Analysis: {query}

{overview}

## Detailed Source Summaries

"""
        # Add individual source summaries
        for idx, summary in enumerate(detailed_summaries, 1):
            formatted_result += f"""
### Source {idx}: {summary['title']}
[Link to original article]({summary['url']})

{summary['summary']}

---
"""
        
        return {
            "status": "success",
            "data": formatted_result,
            "type": "search"
        }
        
    except Exception as e:
        print(f"Search error: {str(e)}")  # Debug log
        return {
            "status": "success",
            "data": f"""
# Search Error

⚠️ An error occurred while searching for "{query}":
- Error details: {str(e)}

Please try again with a different search query.
            """,
            "type": "search"
        }




async def scraped_data():
    query = input("what do you want to search for? ")
    result = await scrape_webpages_with_serpapi(query)
    print(result)  # Optionally keep for debugging
    return result

if __name__ == "__main__":
    asyncio.run(scraped_data())
