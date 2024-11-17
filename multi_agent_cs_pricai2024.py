import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from chromadb.utils import embedding_functions
import openai
import time
from typing import List, Set, Dict
import re
from crewai import Agent, Task, Crew, Process
from langchain.tools import BaseTool, Tool
from pydantic import BaseModel, Field
from typing import Any, Optional

class VectorDBRAGToolInput(BaseModel):
    """Schema for input to the VectorDBRAGTool."""
    query: str = Field(
        ...,
        description="The query to search for in the vector database"
    )

class VectorDBRAGTool(BaseTool):
    """Tool for querying the vector database with RAG capabilities."""
    name: str = Field(default="vector_db_rag_tool", description="Tool for querying the vector database")
    description: str = Field(default="Use this tool to query the vector database for relevant information")
    collection: Any = Field(default=None, description="ChromaDB collection instance")
    args_schema: type[BaseModel] = VectorDBRAGToolInput

    def __init__(self, collection):
        super().__init__()
        self.collection = collection

    def _run(self, query: str) -> str:
        # Query the vector database
        results = self.collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # Combine retrieved documents
        context = "\n\n".join(results['documents'][0])
        return context
    
    async def _arun(self, query: str) -> str:
        """Async implementation of the tool."""
        return self._run(query)
    
def create_vector_db_tool(collection) -> Tool:
    """Create a LangChain Tool wrapper for the vector database."""
    return Tool(
        name="vector_database_search",
        description="Search the vector database for relevant information about PRICAI 2024",
        func=lambda q: VectorDBRAGTool(collection=collection)._run(q),
        args_schema=VectorDBRAGToolInput
    )

class ScrapingConfig:
    """Configuration for web scraping parameters"""
    def __init__(self):
        # Increase default pages to cover more content
        self.MAX_PAGES = 100  
        
        # Add delay between requests to be polite to the server
        self.DELAY_BETWEEN_REQUESTS = 1  # seconds

        self.PRIORITY_URLS = [
            "https://pricai.org/2024/index.php/registration",
        ]
        

class WebScraperRAG:
    def __init__(self, openai_api_key: str, base_url: str, persist_directory: str = "./chroma_db"):
        """Initialize the RAG system with OpenAI API key and base URL to scrape."""
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.persist_directory = persist_directory
        self.config = ScrapingConfig()
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        
        # Initialize ChromaDB with OpenAI embeddings
        # self.chroma_client = chromadb.Client()
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.chroma_client.get_collection(
                name="web_content",
                embedding_function=self.openai_ef
            )
            print("Found existing collection with data")
        except ValueError:
            print("Creating new collection")
            self.collection = self.chroma_client.create_collection(
                name="web_content",
                embedding_function=self.openai_ef
            )
        
        self.visited_urls: Set[str] = set()

    def collection_has_data(self) -> bool:
        """Check if the collection has any documents."""
        return len(self.collection.get()['ids']) > 0

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain and is valid."""
        parsed_url = urlparse(url)
        return (
            parsed_url.netloc == self.base_domain
            and not url.endswith(('.pdf', '.jpg', '.png', '.gif'))
            and '#' not in url
        )

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def scrape_page(self, url: str) -> Dict[str, str]:
        """Scrape a single page and return its content and links."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Extract text content
            text = self.clean_text(soup.get_text())
            
            # Find all links
            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self.is_valid_url(absolute_url):
                    links.append(absolute_url)
            
            return {
                'content': text,
                'links': links
            }
        
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {'content': '', 'links': []}

    def scrape_website(self, max_pages: int = 30):
        """Recursively scrape website starting from base_url."""
        if max_pages is None:
            max_pages = self.config.MAX_PAGES

        urls_to_visit = self.config.PRIORITY_URLS.copy()
        if self.base_url not in urls_to_visit:
            urls_to_visit.insert(0, self.base_url)
        page_count = 0
        
        while urls_to_visit and page_count < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            print(f"Scraping {current_url} ({page_count + 1}/{max_pages})")
            page_data = self.scrape_page(current_url)
            
            if page_data['content']:
                # Add content to ChromaDB
                self.collection.add(
                    documents=[page_data['content']],
                    metadatas=[{
                        "url": current_url,
                        "title": self.extract_page_title(current_url),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }],
                    ids=[f"doc_{page_count}"]
                )
                
                self.visited_urls.add(current_url)
                
                # Prioritize links from conference domain
                new_links = [url for url in page_data['links'] 
                           if url not in self.visited_urls and 
                           url not in urls_to_visit]
                
                # Prioritize URLs that likely contain important information
                priority_links = [url for url in new_links 
                                if any(keyword in url.lower() 
                                      for keyword in ['tutorial', 'workshop', 'paper', 
                                                    'registration', 'program', 'keynote'])]
                
                # Add priority links first
                urls_to_visit.extend(priority_links)
                
                # Add remaining links
                remaining_links = [url for url in new_links if url not in priority_links]
                urls_to_visit.extend(remaining_links)
                
                page_count += 1
            
            # Polite delay between requests
            time.sleep(self.config.DELAY_BETWEEN_REQUESTS)
        print(f"\nScraping completed:")
        print(f"- Total pages scraped: {page_count}")
        print(f"- Priority pages scraped: {len(set(self.config.PRIORITY_URLS) & self.visited_urls)}")

    def extract_page_title(self, url: str) -> str:
        """Extract the title of the webpage."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ''
            return title.strip()
        except Exception:
            return url

    def get_collection(self):
        """Return the ChromaDB collection for use in tools."""
        return self.collection

def create_crew_with_rag(openai_api_key: str, persist_directory: str = "./chroma_db"):
    # Initialize RAG system
    rag_system = WebScraperRAG(
        openai_api_key=openai_api_key,
        base_url="https://pricai.org/2024/index.php",
        persist_directory=persist_directory
    )
    
    # Only scrape if the collection is empty
    if not rag_system.collection_has_data():
        print("No existing data found. Starting web scraping...")
        rag_system.scrape_website(max_pages=30)
        print(f"Scraped {len(rag_system.visited_urls)} pages")
    else:
        print("Using existing data from vector database")
    
    # Create vector database query tool
    vector_tool = VectorDBRAGTool(rag_system.get_collection())
    
    # Create enhanced agents with more personable characteristics
    support_agent = Agent(
        role="Senior Conference Support Representative",
        goal="Provide warm, personalized, and comprehensive support while making every attendee feel valued",
        backstory=(
            "You are a dedicated PRICAI 2024 Conference Organizer with years of experience in making "
            "academic conferences welcoming and accessible to all participants. You take pride in providing "
            "detailed, friendly responses that anticipate attendees' needs. You always address people by name "
            "and maintain a warm, professional tone. You have extensive knowledge of the conference through "
            "the vector database and are passionate about helping attendees have the best possible experience."
        ),
        tools=[vector_tool],
        allow_delegation=True,
        verbose=True
    )

    registration_agent = Agent(
        role="Registration Experience Specialist",
        goal="Create a smooth and welcoming registration experience while building excitement for PRICAI 2024",
        backstory=(
            "As the Registration Experience Specialist for PRICAI 2024, you combine technical expertise "
            "with a warm, welcoming approach. You understand that registration is often attendees' first "
            "interaction with the conference, so you strive to make it memorable and positive. You're known "
            "for your ability to explain complex registration processes clearly while maintaining a friendly, "
            "encouraging tone. You always personalize your responses and ensure attendees feel supported "
            "throughout their registration journey. Please note that for registration information, "
            "you must refer to official website at https://pricai.org/2024/index.php/registration."
            "Don't make hallucination about this crucial information, such as pricing for attendance."
        ),
        tools=[vector_tool],
        verbose=True
    )

    support_quality_assurance_agent = Agent(
        role="Attendee Experience Guardian",
        goal="Ensure every response is not only accurate but also engaging and helpful",
        backstory=(
            "You are passionate about creating exceptional experiences for PRICAI 2024 attendees. "
            "Your role goes beyond just checking facts - you ensure responses are warm, clear, and "
            "anticipate follow-up questions. You have a keen eye for detail and always think about "
            "how to make information more accessible and engaging. You believe in the power of "
            "personalized communication and ensure every response reflects the conference's "
            "welcoming spirit. After answer question, you can warm regards or best regards your name for more polite."
        ),
        tools=[vector_tool],
        verbose=True
    )

    # Create enhanced tasks with more personalized output requirements
    inquiry_resolution = Task(
        description=(
            "Our valued attendee {person} from {customer} has reached out with this inquiry:\n"
            "{inquiry}\n\n"
            "Use the vector database tool to find relevant information and craft a warm, "
            "personalized response that addresses them by name and shows genuine interest "
            "in their participation at PRICAI 2024."
        ),
        expected_output=(
            "A warm, personalized response that:\n"
            "1. Addresses the attendee by name\n"
            "2. Expresses appreciation for their interest\n"
            "3. Provides comprehensive information in a friendly tone\n"
            "4. Includes relevant follow-up resources or contact points\n"
            "5. Ends with an encouraging note about their participation in PRICAI 2024"
        ),
        agent=support_agent
    )

    registration_inquiry_resolution = Task(
        description=(
            "Our potential attendee {person} from {customer} has a registration-related question:\n"
            "{inquiry}\n\n"
            "Craft a welcoming response that makes their registration process smooth and "
            "builds excitement for their participation in PRICAI 2024."
        ),
        expected_output=(
            "A friendly and detailed response that:\n"
            "1. Warmly greets them by name\n"
            "2. Provides clear registration guidance\n"
            "3. Highlights relevant conference benefits\n"
            "4. Offers additional assistance if needed\n"
            "5. Expresses enthusiasm about their potential participation"
        ),
        agent=registration_agent
    )

    quality_assurance_review = Task(
        description=(
            "Review the response prepared for {person} from {customer} regarding:\n"
            "{inquiry}\n\n"
            "Ensure it maintains our high standards for both accuracy and engagement."
        ),
        expected_output=(
            "A verified response that is:\n"
            "1. Technically accurate and complete\n"
            "2. Warm and personalized\n"
            "3. Clear and accessible\n"
            "4. Proactive in addressing potential follow-up questions\n"
            "5. Aligned with our commitment to exceptional attendee experience\n"
            "6. End with your name as a Senior Support Representative"
        ),
        agent=support_quality_assurance_agent
    )

    # Create crew
    crew = Crew(
        agents=[support_agent, registration_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, registration_inquiry_resolution, quality_assurance_review],
        verbose=True,
        memory=True
    )

    return crew

# Example usage
def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment variables")
    
    # Create persist_directory if it doesn't exist
    persist_directory = "./chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    
    crew = create_crew_with_rag(openai_api_key, persist_directory)
    
    # inputs = {
    #     "customer": "Kalbe Digital Lab",
    #     "person": "Adhi Setiawan",
    #     "inquiry": "I need know about PRICAI 2024 conference, specifically "
    #               "is there any topics that explain about synthetic data generation "
    #               "in tutorials program at PRICAI 2024?"
    # }

    # inputs = {
    #     "customer": "Kalbe Digital Lab",
    #     "person": "Adhi Setiawan",
    #     "inquiry": "I need know about PRICAI 2024 conference, specifically "
    #               "how to register and how about pricing for general attendance for early registration, late, and onsite "
    #               "that want to attent at PRICAI 2024? also can you recommend hotel near venue?"
    # }
    
    inputs = {
        "customer": "Kalbe Digital Lab",
        "person": "Adhi Setiawan",
        "inquiry": "can you explain more detail about diffusion model? maybe history, theoritical, conceptual, and short implementation using pytorch?"
    }
    result = crew.kickoff(inputs=inputs)
    print("\nFinal Result:", result)

if __name__ == "__main__":
    main()