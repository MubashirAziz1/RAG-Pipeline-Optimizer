import cohere

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
        self.cohere_client = cohere.Client(api_key)


    def embed_passage(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Cohere embedding model (embed-english-v3.0)"""
        try:
            logging.info("Embedding the text chunks using Cohere model")
            
            response = self.cohere_client.embed(
                texts=texts,
                model=self.embedding_model,
                input_type=self.input_type
            )

            logger.info(f"Successfully embedded {len(texts)} passages")
            return response.embeddings
        
        except Exception as e:
            logger.error(f"Unexpected error in embed_passages: {e}")
            raise






