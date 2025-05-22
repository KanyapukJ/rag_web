import os
import json
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
import aiohttp
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Load environment variables
load_dotenv()

# Initialize Ollama models
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Initialize LLM
llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.1)

# Initialize embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


@dataclass
class ProcessedChunk:
    """Represents a processed text chunk with metadata and embedding."""
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """Split text into chunks, respecting paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        # Try to break at paragraph
        paragraph_break = chunk.rfind("\n\n")
        if paragraph_break > chunk_size * 0.3:
            end = start + paragraph_break + 2
        # Or try sentence break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


async def get_title_and_content(chunk: str, url: str) -> Dict[str, str]:
    """Extract a title using Ollama LLM and return the original chunk as content."""
    system_prompt = """วิเคราะห์เนื้อหาต่อไปนี้และสร้างชื่อเรื่องที่กระชับ (ไม่เกิน 10-15 คำ) เป็นภาษาไทย
    ชื่อเรื่องควรจับใจความสำคัญของเนื้อหา
    ตอบกลับเป็น JSON เท่านั้น โดยมีคีย์ 'title'
    ตัวอย่าง: {"title": "วิธีการดูแลผู้ป่วยโรคเบาหวาน"}
    ห้ามใส่คำอธิบายหรือข้อความอื่นใดนอกจาก JSON"""

    title = f"เนื้อหาจาก {urlparse(url).netloc}"
    try:
        loop = asyncio.get_event_loop()
        prompt_for_title = f"{system_prompt}\n\nเนื้อหา:\n{chunk[:1500]}..."
        
        response = await loop.run_in_executor(None, lambda: llm.invoke(prompt_for_title))
        
        json_str = response.strip()
        if not json_str.startswith('{'):
            start = json_str.find('{')
            if start >= 0:
                json_str = json_str[start:]
            else:
                print(f"Warning: No JSON found in title generation response for {url}. Raw: {response}")
        
        if json_str.startswith('{') and json_str.endswith('}'):
            try:
                extracted_json = json.loads(json_str)
                title = extracted_json.get("title", title)
            except json.JSONDecodeError as e:
                print(f"Warning: JSONDecodeError for title generation on {url}: {e}. Raw: {json_str}")
        else:
            print(f"Warning: Malformed JSON for title generation on {url}. Raw: {json_str}")

    except Exception as e:
        print(f"Error getting title: {e}. Using default title for {url}.")

    return {
        "title": title,
        "summary": chunk
    }


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama asynchronously."""
    try:
        # Use aembed_query if available and it's a native async method
        result = await embeddings.aembed_query(text) 
        return result
    except Exception as e:
        print(f"Error getting embedding for text snippet: {text[:100]}... Error: {e}")
        default_dim = 1024 
        try:
            if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'show'): # Check for Ollama specific client details
                pass # Placeholder if direct dim query is not straightforward
        except:
            pass # Ignore errors trying to get a more specific dimension
        return [0.0] * default_dim


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted_data = await get_title_and_content(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": urlparse(url).netloc,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted_data["title"],
        summary=extracted_data["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(collection, chunk: ProcessedChunk):
    """Insert a processed chunk into ChromaDB asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, # Use default ThreadPoolExecutor
            lambda: collection.add(
                documents=[chunk.content],
                embeddings=[chunk.embedding],
                metadatas=[
                    {
                        "url": chunk.url,
                        "chunk_number": chunk.chunk_number,
                        "title": chunk.title,
                        "summary": chunk.summary,
                        **chunk.metadata,
                    }
                ],
                ids=[f"{chunk.url}_{chunk.chunk_number}"],
            )
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting chunk for {chunk.url} (chunk {chunk.chunk_number}): {e}")


async def process_and_store_document(collection, url: str, content: str):
    """Process a document and store its chunks in ChromaDB."""
    chunks = chunk_text(content)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk(collection, chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)
    return len(processed_chunks)


async def fetch_url(session, url: str) -> Optional[str]:
    """Fetch URL content using aiohttp."""
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                html = await response.text()
                return html
            else:
                print(f"Error fetching {url}: Status {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_text_from_html(html: str) -> str:
    """Extract clean text content from HTML."""
    soup = BeautifulSoup(html, "lxml")
    
    # Remove script and style elements
    for element in soup(["script", "style", "header", "footer", "nav"]):
        element.decompose()
    
    # Get text and clean it
    text = soup.get_text(separator="\n")
    
    # Remove empty lines and excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n\n".join(lines)
    
    return text


def extract_links_from_html(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML content."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Convert relative links to absolute
        full_url = urljoin(base_url, href)
        # Keep only links from the same domain
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.append(full_url)
    
    return list(set(links))  # Remove duplicates


async def crawl_website(collection, start_url: str, max_pages: int = 100):
    """Crawl website starting from start_url and store content in ChromaDB."""
    print(f"Starting crawl from {start_url}")
    visited_urls = set()
    queued_urls = {start_url}
    processed_count = 0
    
    async with aiohttp.ClientSession() as session:
        while queued_urls and len(visited_urls) < max_pages:
            # Get next URL to process
            current_url = queued_urls.pop()
            if current_url in visited_urls:
                continue
                
            print(f"Crawling {current_url}")
            visited_urls.add(current_url)
            
            # Fetch and process content
            html = await fetch_url(session, current_url)
            if html:
                # Extract text content
                text_content = extract_text_from_html(html)
                if len(text_content) > 500:  # Only process non-empty pages
                    chunks_added = await process_and_store_document(collection, current_url, text_content)
                    processed_count += 1
                    print(f"Processed {current_url} - Added {chunks_added} chunks")
                
                # Extract links and add to queue
                new_links = extract_links_from_html(html, current_url)
                for link in new_links:
                    if link not in visited_urls:
                        queued_urls.add(link)
                        
            # Pause to be respectful to the server
            await asyncio.sleep(1)
    
    print(f"Crawl complete. Processed {processed_count} pages.")
    return processed_count 