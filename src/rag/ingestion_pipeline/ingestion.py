from src.services.pdf_parser.factory  import make_pdf_parser_service
from src.services.indexing.text_chunker import TextChunker
from src.services.indexing.factory import make_cohere_embeddings_service
from src.services.vector_store.factory import make_chromadb_service

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class IngestionPipeline():

    def __init__(self, chunker, embedder, vector_store, name : str):

        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.name = name
    
    def ingest(self, raw_text: str):
        print(f"\n🚀 {self.name} begins ........")

        chunks = self.chunker.chunk_text(raw_text)
        embeddings = self.embedder.embed_passage(chunks)

        self.vector_store.store(
            collection_name = self.name,
            chunks=chunks,
            embeddings=embeddings,
            doc_id=self.name
        )

        print(f"✅ {self.name} completed")




