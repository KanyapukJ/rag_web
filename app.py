import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from db import init_collection
from crawl import crawl_website
from utils import get_db_stats, query_rag_system

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize ChromaDB collection
collection = init_collection()

# CSS for styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e3f2fd;
    }
    .source-item {
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: #f9f9f9;
        border-left: 2px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crawling" not in st.session_state:
        st.session_state.crawling = False
    if "crawl_status" not in st.session_state:
        st.session_state.crawl_status = ""
    if "stats" not in st.session_state:
        st.session_state.stats = None # Or an empty dict: {"doc_count": 0, "urls": [], "domains": [], "last_updated": "N/A"}

async def initialize_async_session_state():
    """Initialize parts of session state that require async operations."""
    if st.session_state.stats is None: # Only fetch if not already populated
        st.session_state.stats = await get_db_stats(collection)


async def render_chat():
    """Render the chat interface."""
    st.title("Agnos Health RAG Chatbot")
    
    # Display existing messages
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            
            # Display sources if available
            if "sources" in msg and msg["sources"]:
                with st.expander("View sources"):
                    for source in msg["sources"]:
                        st.markdown(f"**{source['title']}**")
                        st.markdown(f"URL: [{source['url']}]({source['url']})")
                        st.markdown(f"Summary: {source['summary']}")
                        st.divider()
    
    # Chat input
    if query := st.chat_input("Ask a question about Agnos Health..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare chat history for the RAG system
                chat_history_for_rag = []
                # Take last N messages, excluding the current user query which is passed separately
                # Example: take last 4 messages (2 turns of user/assistant)
                recent_messages = st.session_state.messages[:-1] # Exclude current query message
                for msg in recent_messages[-4:]:
                    chat_history_for_rag.append({"role": msg["role"], "content": msg["content"]})
                
                # Query the RAG system, now passing chat_history
                result = await query_rag_system(collection, query, chat_history=chat_history_for_rag)
                answer = result["answer"]
                sources = result["sources"]
                
                st.markdown(answer)
                
                # Display sources if available
                if sources:
                    with st.expander("View sources"):
                        for source in sources:
                            st.markdown(f"**{source['title']}**")
                            st.markdown(f"URL: [{source['url']}]({source['url']})")
                            st.markdown(f"Summary: {source['summary']}")
                            st.divider()
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })


async def sidebar_crawl_controls():
    """Render the sidebar crawl controls."""
    st.sidebar.title("Data Management")
    
    # Show database stats
    if st.session_state.stats:
        st.sidebar.subheader("Current Database")
        st.sidebar.metric("Documents", st.session_state.stats["doc_count"])
        st.sidebar.write(f"Last updated: {st.session_state.stats.get('last_updated', 'Unknown')}")
        st.sidebar.write(f"Sources: {', '.join(st.session_state.stats['domains'])}")
        
        with st.sidebar.expander("Crawled URLs"):
            if "urls" in st.session_state.stats and st.session_state.stats["urls"]:
                for url_entry in st.session_state.stats["urls"]:
                    st.sidebar.write(f"- {url_entry}") # Assuming url_entry is a string
            else:
                st.sidebar.write("No URLs found in stats.")
    else:
        st.sidebar.warning("No data in the database. Please crawl a website.")
    
    st.sidebar.divider()
    
    # Crawl controls
    st.sidebar.subheader("Crawl Website")
    target_url = st.sidebar.text_input("Website URL", os.getenv("TARGET_URL", "https://www.agnoshealth.com/forums"))
    max_pages = st.sidebar.slider("Max Pages to Crawl", 5, 200, 50)
    
    if st.sidebar.button("Start Crawling"):
        st.session_state.crawling = True
        st.session_state.crawl_status = "Starting crawl..."
        
        # Run the crawl in a background task
        st.sidebar.warning("Crawling in progress, please wait...")
        
        # Actually run the crawl - this will block the UI until complete
        try:
            with st.sidebar.status("Crawling website...") as status_ui:
                st.session_state.crawl_status = "Crawling website..."
                pages_processed = await crawl_website(collection, target_url, max_pages)
                status_ui.update(label=f"Crawl complete! Processed {pages_processed} pages.", state="complete")
                
            # Update stats after crawl
            st.session_state.crawling = False
            st.session_state.stats = await get_db_stats(collection)
            st.sidebar.success(f"Crawl complete! Processed {pages_processed} pages.")
            if st.sidebar.button("Refresh Data Display"):
                st.experimental_rerun()
            
        except Exception as e:
            st.session_state.crawling = False
            st.sidebar.error(f"Error during crawl: {str(e)}")
            print(f"Crawl Error: {e}")
    
    # Status update
    if st.session_state.crawling:
        st.sidebar.info(st.session_state.crawl_status)


async def main():
    """Main application function."""
    initialize_session_state()
    await initialize_async_session_state()
    
    await sidebar_crawl_controls()
    await render_chat()


if __name__ == "__main__":
    asyncio.run(main()) 