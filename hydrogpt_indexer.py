"""Build a Chroma vector index from PDFs in ./data."""
import shutil
from importlib import import_module
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _resolve_hf_embeddings_class():
    for module_name in ["langchain_huggingface", "langchain_community.embeddings"]:
        try:
            return getattr(import_module(module_name), "HuggingFaceEmbeddings")
        except Exception:
            continue
    raise ImportError("Could not import HuggingFaceEmbeddings from langchain_huggingface or langchain_community.")


def _resolve_chroma_class():
    for module_name in ["langchain_chroma", "langchain_community.vectorstores"]:
        try:
            return getattr(import_module(module_name), "Chroma")
        except Exception:
            continue
    raise ImportError("Could not import Chroma from langchain_chroma or langchain_community.")


HuggingFaceEmbeddings = _resolve_hf_embeddings_class()
Chroma = _resolve_chroma_class()

DATA_DIR  = Path("./data")
DB_DIR    = Path("./hydro_db")

# Lightweight embedding model for faster local indexing/inference on Windows.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keep technical entities like SSP585/Epoch 3 intact across chunk boundaries.
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 200


def build_index(force: bool = False) -> None:
    if DB_DIR.exists():
        if not force:
            print(f"[INFO] Index already exists at '{DB_DIR}'. Use force=True to rebuild.")
            return
        print(f"[INFO] Deleting old index at '{DB_DIR}'…")
        shutil.rmtree(DB_DIR)

    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDFs found in '{DATA_DIR}'. Add your PDF files there and re-run."
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[1/4] Loading PDFs from '{DATA_DIR}'…")
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()
    print(f"      Loaded {len(documents)} pages from {len(set(d.metadata['source'] for d in documents))} files.")

    # ── Split ─────────────────────────────────────────────────────────────────
    print(f"[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Filter out very short chunks (e.g. page headers/footers)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 80]
    print(f"      Created {len(chunks)} usable chunks.")

    # ── Embed & Store ─────────────────────────────────────────────────────────
    print(f"[3/4] Building embeddings with '{EMBED_MODEL}'…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # important for cosine similarity
    )

    print(f"[4/4] Writing ChromaDB index to '{DB_DIR}'…")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    count = vectorstore._collection.count()
    print(f"\n✓ Index built successfully — {count} vectors stored in '{DB_DIR}'.\n")

    # Quick sanity check
    test_results = vectorstore.similarity_search("temperature precipitation", k=3)
    print(f"Sanity check — top 3 chunks for 'temperature precipitation':")
    for i, doc in enumerate(test_results, 1):
        src = Path(doc.metadata.get("source", "?")).name
        print(f"  {i}. [{src}] {doc.page_content[:120].strip()}…")


if __name__ == "__main__":
    build_index(force=True)
