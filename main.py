import os
import sys

from src.services.pdf_parser.factory  import make_pdf_parser_service
from src.services.indexing.text_chunker import TextChunker
from src.services.indexing.factory import make_cohere_embeddings_service
from src.services.vector_store.factory import make_chromadb_service
from src.rag.ingestion_pipeline.ingestion import IngestionPipeline

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_ROOT / "data"   # drop your PDFs in RAG_Pipeline_Optimizer/data/



def prompt_for_pdf() -> Path:
    """
    Lists every .pdf inside <project_root>/data/ and lets the user pick
    one by number.  Falls back to manual path entry if folder is empty.
    """

    PDF_FOLDER.mkdir(parents=True, exist_ok=True)   # create data/ if missing

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if pdf_files:
        print(f"\n📁  PDFs found in '{PDF_FOLDER}':\n")
        for i, f in enumerate(pdf_files, start=1):
            size_kb = f.stat().st_size / 1024
            print(f"   [{i}] {f.name}  ({size_kb:.1f} KB)")
        print(f"   [0] Enter a custom path instead")

        while True:
            choice = input("\n   Select a file number: ").strip()
            if choice == "0":
                break                               # fall through to manual entry
            if choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
                selected = pdf_files[int(choice) - 1].resolve()
                print(f"      ✅  Selected: {selected.name}")
                return selected
            print(f"   ⚠️  Please enter a number between 0 and {len(pdf_files)}.")
    else:
        print(f"\n   ℹ️  No PDFs found in '{PDF_FOLDER}'.")
        print(f"      Drop your PDF files there, or enter a path manually below.")

    # ── Manual fallback ────────────────────────────────────────────────────────
    while True:
        raw = input("\n📄  Enter the full path to your PDF: ").strip().strip('"').strip("'")

        if not raw:
            print("   ⚠️  No path entered. Please try again.")
            continue

        pdf_path = Path(raw).resolve()

        if pdf_path.suffix.lower() != ".pdf":
            print("   ⚠️  File must be a .pdf — please try again.")
            continue

        if not pdf_path.exists():
            print(f"   ⚠️  File not found: '{pdf_path}' — please try again.")
            continue

        return pdf_path


# ── Step 2: Ask what the user wants to query ───────────────────────────────────
# def prompt_for_query() -> str:
#     """Prompt the user for a search / question to run against the document."""
#     while True:
#         query = input("\n🔍  Enter your question or search query: ").strip()
#         if query:
#             return query
#         print("   ⚠️  Query cannot be empty. Please try again.")




#     # # ── Query / Retrieval ──────────────────────────────────────────────────────
#     # print("\n" + "─" * 60)
#     # print(f"🔎  Running query: \"{query}\"")
#     # print("─" * 60)

#     # query_embedding = embedder.embed([query])[0]     # embed the user query
#     # results = chroma.query(
#     #     query_embeddings=[query_embedding],
#     #     n_results=3,                                 # top-3 relevant chunks
#     # )

#     # print("\n📌  Top relevant chunks:\n")
#     # documents = results.get("documents", [[]])[0]
#     # distances = results.get("distances", [[]])[0]

#     # if not documents:
#     #     print("   No results found.")
#     # else:
#     #     for rank, (doc, dist) in enumerate(zip(documents, distances), start=1):
#     #         print(f"  [{rank}] (distance: {dist:.4f})")
#     #         print(f"       {doc[:300].strip()}{'...' if len(doc) > 300 else ''}")
#     #         print()


# # ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("       RAG Pipeline — PDF Upload & Query")
    print("=" * 60)

    try:
        chroma_client = make_chromadb_service()
        pipeline = chroma_client.create_collection('Pipeline_A')
        pdf_path = prompt_for_pdf()
        parser = make_pdf_parser_service()
        raw_text = parser.parse_pdf(pdf_path)

        ingest_docs = IngestionPipeline(TextChunker(), make_cohere_embeddings_service(), chroma_client, name='Pipeline_A')
        ingest_docs.ingest(raw_text)


    except KeyboardInterrupt:
        print("\n\n👋  Interrupted by user. Exiting.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌  Pipeline error: {e}")
        raise