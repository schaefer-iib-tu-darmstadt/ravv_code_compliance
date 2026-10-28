"""Qdrant client (embedded local store if QDRANT_LOCAL_PATH is set) and collection helpers."""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def create_qdrant_client() -> QdrantClient:
    local_path = os.getenv("QDRANT_LOCAL_PATH")
    if local_path:
        return QdrantClient(path=local_path)
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("Qdrant configuration missing: Please set QDRANT_URL and QDRANT_API_KEY in .env.")
    return QdrantClient(url=url, api_key=api_key, prefer_grpc=False, https=True, timeout=300)

def collection_is_empty(client: QdrantClient, name: str) -> bool:
    if not client.collection_exists(collection_name=name):
        return True
    return client.count(collection_name=name, exact=False).count == 0

def delete_collection_if_exists(client: QdrantClient, name: str) -> None:
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass