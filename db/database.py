import chromadb
from chromadb.config import Settings

def get_chroma_client():
    """Initialize and return a ChromaDB client with persistent storage."""
    return chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(
            allow_reset=True, 
            anonymized_telemetry=False, 
            is_persistent=True
        ),
    )

def init_collection(collection_name="agnos_health_data"):
    """Initialize or get a collection from ChromaDB."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # We'll provide embeddings separately
    ) 