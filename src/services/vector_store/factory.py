from .chromadb import ChromadbStore
import logging

logger = logging.getLogger(__name__)


def make_chromadb_service() -> ChromadbStore:
    """
    Factory function to create Chromadb service.

    """
    return ChromadbStore()


