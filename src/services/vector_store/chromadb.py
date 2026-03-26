from typing import Dict
import chromadb
from chromadb.config import Settings
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ChromadbStore:

    _instance: Optional[chromadb.Client] = None

    def __init__(self, persist_directory: str = "./chroma_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        if ChromadbStore._instance is None:
            ChromadbStore._instance = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB initialized at: {self.persist_directory}")

        self.client = ChromadbStore._instance
        self._collections: Dict[str, chromadb.Collection] = {}
    
    def create_collection(self, pipeline_id: str) -> chromadb.Collection:

        collection_name = f"{pipeline_id}"
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine", 
                "pipeline_id": collection_name
            })

        self._collections[collection_name] = collection
        print(f"Collection ready: {collection_name} | docs: {collection.count()}")
        # logger.info(f"Collection ready: {collection_name} | docs: {collection.count()}")
        return collection

    def store(self, collection_name: str, embeddings: list[list[float]], chunks: list[str], doc_id: str) -> int:

        if not embeddings or not chunks:
            print(f"Empty embeddings or chunks for {collection_name}")
           # logger.warning(f"Empty embeddings or chunks for {pipeline_id}")
            return 0

        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings ({len(embeddings)}) must match chunks ({len(chunks)})")

        collection = self._collections.get(collection_name)

        if not collection:
            raise ValueError(f"Collection '{collection_name}' not initialized. Call get_or_create_collection() first.")

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        collection.add(embeddings=embeddings, documents=chunks, ids=ids)

        print(f"Stored {len(chunks)} chunks → {collection_name}")
        #logger.info(f"Stored {len(chunks)} chunks → {collection_name}")
        return len(chunks)

    # def query(
    #     self,
    #     session_id: str,
    #     pipeline_id: str,
    #     query_embedding: list[float],
    #     n_results: int = 5,
    # ) -> dict:

    #     collection_key = f"{session_id}_{pipeline_id}"
    #     collection = self._collections.get(collection_key)

    #     if not collection:
    #         raise ValueError(f"Collection '{collection_key}' not found.")

    #     return collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=n_results,
    #         include=["documents", "metadatas", "distances"]
    #     )

    # def reset_session(self, session_id: str) -> None:
    #     """ Wipe all pipeline collections for a given session. """
    #     to_delete = [key for key in self._collections if key.startswith(session_id)]
    #     for key in to_delete:
    #         self.client.delete_collection(name=key)
    #         del self._collections[key]
    #     logger.info(f"Cleared {len(to_delete)} collections for session: {session_id}")