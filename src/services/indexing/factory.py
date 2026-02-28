from typing import Optional
from src.config import Settings, get_settings
from .embedding import CohereEmbeddingsClient
from .embedding import BertEmbeddingsClient

import logging

logger = logging.getLogger(__name__)


def make_cohere_embeddings_service(
    settings: Optional[Settings] = None,
    input_type: str = "search_document"
) -> CohereEmbeddingsClient:
    """
    Factory function to create Cohere embeddings service.

    """
    if settings is None:
        settings = get_settings()

    # Get configuration from settings
    api_key = settings.cohere_api_key
    model_name = 'embed-english-v3.0'

    logger.info(f"Creating Cohere embeddings service with model: {model_name}")

    return CohereEmbeddingsClient(
        embedding_model=model_name,
        api_key=api_key,
        input_type=input_type
    )


def make_bert_embeddings_service() -> BertEmbeddingsClient:
    """
    Factory function to create BERT embeddings service.

    """

    model_name = 'all-MiniLM-L6-v2'

    logger.info(f"Creating BERT embeddings service with model: {model_name}")

    return BertEmbeddingsClient(embedding_model=model_name)

