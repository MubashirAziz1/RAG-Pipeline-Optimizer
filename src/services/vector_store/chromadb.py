import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Literal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromadbStore:
    """
    ChromaDB store that manages multiple collections for different RAG pipelines.
    
    """
    
    def __init__(self, persist_directory: str = "./chromadb_data"):
        """
        Initialize ChromaDB client with persistent storage.
        
        :param persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Track active collections
        self.collections: Dict[str, chromadb.Collection] = {}
        
        logger.info(f"✅ ChromaDB initialized at {self.persist_directory}")
    
    def initialize_pipeline_collection(self, pipeline_id: str) -> chromadb.Collection:
        """
        Initialize or get a collection for a specific pipeline.
        
        """
        collection_name = f"pipeline_{pipeline_id.lower()}"
        
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "pipeline_id": pipeline_id,
                }
            )
            
            self.collections[pipeline_id] = collection
            
            logger.info(f"✅ Collection '{collection_name}' initialized - ")
            
            return collection
        
        except Exception as e:
            logger.error(f"Failed to initialize collection for pipeline {pipeline_id}: {e}")
            raise
    
    def store(self, pipeline_id: str, embeddings: List[List[float]], chunks: List[str], doc_id: str,) -> int:
        """
        Store embeddings and chunks in a specific pipeline's collection.
        
        """
        if not embeddings or not chunks:
            logger.warning(f"Empty embeddings or chunks provided for pipeline {pipeline_id}")
            return 0
        
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings count ({len(embeddings)}) must match documents count ({len(chunks)})")
        
        # Get the collection for this pipeline
        if pipeline_id not in self.collections:
            raise ValueError(
                f"Pipeline {pipeline_id} not initialized. "
                f"Call initialize_pipeline_collection() first."
            )
        
        collection = self.collections[pipeline_id]
        
        # Generate unique IDs for each chunk
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        try:
            # Store in ChromaDB
            collection.add(embeddings=embeddings, documents=chunks, ids=ids)
            
            logger.info(
                f"✅ Stored {len(chunks)} chunks in Pipeline {pipeline_id} collection "
                f"(dimension: {len(embeddings[0])})"
            )
            
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Failed to store embeddings in pipeline {pipeline_id}: {e}")
            raise
    
    # def query(
    #     self,
    #     pipeline_id: str,
    #     query_embedding: List[float],
    #     top_k: int = 5,
    #     where: Optional[Dict] = None
    # ) -> Dict:
    #     """
    #     Query a specific pipeline's collection.
        
    #     :param pipeline_id: Pipeline to query
    #     :param query_embedding: Query embedding vector
    #     :param top_k: Number of results to return
    #     :param where: Optional metadata filter
    #     :return: Query results
    #     """
    #     if pipeline_id not in self.collections:
    #         raise ValueError(f"Pipeline {pipeline_id} not initialized")
        
    #     collection = self.collections[pipeline_id]
        
    #     try:
    #         results = collection.query(
    #             query_embeddings=[query_embedding],
    #             n_results=top_k,
    #             where=where
    #         )
            
    #         logger.debug(f"Retrieved {len(results['documents'][0])} results from Pipeline {pipeline_id}")
            
    #         return results
        
    #     except Exception as e:
    #         logger.error(f"Query failed for pipeline {pipeline_id}: {e}")
    #         raise
    