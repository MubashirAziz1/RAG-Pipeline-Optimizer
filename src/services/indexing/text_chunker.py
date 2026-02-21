import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextChunker:
    """Service for chunking text into overlapping segments."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize text chunker.

        :chunk_size: Target number of words per chunk
        :overlap_size: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")
        
        # Initialize the Langchain Recusrsive Character Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(
            f"Text chunker initialized: chunk_size={chunk_size}, overlap_size={chunk_overlap}. "
        )
        
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Text chunking using Langchain Recusrsive Character Splitting.

        :text: Extracted text to split
        """

        logger.info(
            f"Chunking text ........ "
        )
        
        chunks = self.text_splitter.split_text(text)
        
        logger.info(
            f"Created {len(chunks)} chunks"
        )
        
        return chunks
