import cohere
from sentence_transformers import SentenceTransformer

import logging
from typing import List

logger = logging.getLogger(__name__)


class CohereEmbeddingsClient:
    """Client for Cohere embeddings API.
     
      Uses Cohere v3 embedding model with 1024 dimensions """

    def __init__(self, embedding_model: str, api_key: str, input_type: str):
        """
        Initialize Cohere embedding client.
            
        :param embedding_model: Embedding model name
        :param api_key: Cohere API key 
        """

        self.api_key = api_key
        if not self.api_key:
            logging.info("Cohere API Key is missing and required.")
            raise ValueError("Cohere API key required.")
        
        self.embedding_model = embedding_model
        self.input_type = input_type
        self.cohere_client = cohere.Client(self.api_key )


    def embed_passage(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Cohere embedding model (embed-english-v3.0)"""
        try:
            logging.info("Embedding the text chunks using Cohere model")
            
            response = self.cohere_client.embed(
                texts=texts,
                model=self.embedding_model,
                input_type=self.input_type
            )

            logger.info(f"Successfully embedded {len(texts)} passages using cohere model. ")
            return response.embeddings
        
        except Exception as e:
            logger.error(f"Unexpected error in Cohere embed_passages: {e}")
            raise


class BertEmbeddingsClient:
    """Client for bert embeddings.
     
      Uses all-MiniLM-L6-v2 embedding model with 384 dimensions """

    def __init__(self, embedding_model: str):
        """
        Initialize Bert embedding client.
            
        :param embedding_model: Embedding model name 
        """

        self.embedding_model = SentenceTransformer(embedding_model)

    def embed_passage(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Bert embedding model (all-MiniLM-L6-v2)"""
        try:
            logging.info("Embedding the text chunks using Bert model")
            
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True)

            logger.info(f"Successfully embedded {len(texts)} passages using Bert Model into 384 dimensions.")
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Unexpected error in Bert embed_passages: {e}")
            raise







