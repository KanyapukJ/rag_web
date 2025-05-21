# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Agnos Health that automatically crawls and indexes content from the Agnos health forum.

## Features

- **Web Crawling**: Automatically crawls the Agnos health forum to extract content
- **Vector Database**: Stores and indexes content in ChromaDB for efficient retrieval
- **RAG System**: Uses Ollama with llama3.2 for LLM and bge-m3 for embeddings to provide accurate answers based on forum content
- **Chat Interface**: Clean Streamlit interface for interacting with the RAG system

## Architecture

The system consists of the following components:

1. **Web Crawler**: Scrapes content from the Agnos health forum, processes it, and stores it in a vector database
2. **ChromaDB**: Vector database that stores document content and embeddings
3. **RAG Pipeline**: Retrieves relevant documents and generates answers using Ollama models
4. **Streamlit UI**: Provides an intuitive chat interface for users

## Requirements

- Python 3.9+
- Ollama installed and running locally (or accessible via network). Ensure the Ollama application is active.
- Models `llama3.2` (or your chosen LLM) and `bge-m3` (or your chosen embedding model) downloaded in Ollama.
- Internet connection for web crawling and API access.
- `nest_asyncio` for running asyncio code within Streamlit.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag_web.git
cd rag_web
```

2. Create and activate a Conda environment (recommended):
```bash
conda create -n rag_web python=3.9
conda activate rag_web
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the `rag_web` directory by copying `env.example` and update your configuration:
```bash
cp env.example .env
# Then edit .env with your details
```
Example `.env` content:
```
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2
EMBEDDING_MODEL=bge-m3
TARGET_URL=https://www.agnoshealth.com/forums # Default website to crawl
```

5. Download the required models in Ollama:
   Open the Ollama application, then in your terminal:
```bash
ollama pull llama3.2
ollama pull bge-m3
```
   Verify the models are listed by running `ollama list` in your terminal.

## Usage

### Web Interface

1. Ensure your Ollama application is running and the specified models are available.
2. Navigate to the `rag_web` directory.
3. Activate your conda environment if you created one:
```bash
conda activate rag_web
```
4. Start the application:
```bash
streamlit run app.py
```

5. Open your browser at the URL provided by Streamlit (typically http://localhost:8501 or similar).

6. Use the sidebar to:
    - **Crawl Website**: Enter a URL and max pages, then click "Start Crawling".
    - View database statistics.

7. Once crawling is complete, you can ask questions in the chat interface. The RAG system will use the crawled content to generate answers.

## Project Structure

```
rag_web/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── env.example             # Environment variables template
├── .env                    # Your local environment variables (after copying from env.example)
├── README.md               # This file
│
├── crawl/                  # Web crawling module
│   ├── __init__.py
│   └── crawler.py          # Web crawler implementation (example, actual might differ)
│
├── db/                     # Database module
│   ├── __init__.py
│   └── database.py         # ChromaDB integration (example, actual might differ)
│
└── utils/                  # Utility functions
    ├── __init__.py
    └── rag_utils.py        # RAG functionality (example, actual might differ)
```

## Customization

- Modify `env.example` to change default settings
- Adjust `utils/rag_utils.py` to customize RAG behavior
- Edit `crawl/crawler.py` to modify web crawling behavior

## Notes

- The first crawl may take some time depending on the website size
- Be respectful when crawling websites by limiting request frequency

## License

This project is proprietary and confidential. All rights reserved to Agnos Health Co. Ltd. 