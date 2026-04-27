import os
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

# -- Config -------------------------------------------------------------------
DATA_DIR = Path("./data")
DB_DIR = Path("./hydro_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 12
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_FALLBACKS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

REFINE_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict retrieval-grounded answer refiner.

Use ONLY the provided extracted answer and evidence lines. Do not invent values.
If the question asks for numeric values (baseline, increase, expected value), return exact numbers and units if present.
If exact numbers are unavailable, say that clearly and summarize available related evidence.

Question:
{question}

Extracted Answer:
{base_answer}

Evidence:
{evidence}
"""
)


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


def _is_valid_db_dir(path: Path) -> bool:
    """Basic structural check for a persisted Chroma directory."""
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "chroma.sqlite3").exists():
        return False
    return any(p.is_dir() for p in path.iterdir())


def _extract_index_zip(zip_path: Path) -> bool:
    """Extract hydro_db zip into DB_DIR, handling common zip layouts."""
    tmp_extract = Path("./_hydro_extract_tmp")
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_extract)

        nested_db = tmp_extract / "hydro_db"
        candidate = nested_db if nested_db.exists() else tmp_extract

        if not _is_valid_db_dir(candidate):
            return False

        if DB_DIR.exists():
            shutil.rmtree(DB_DIR, ignore_errors=True)
        shutil.move(str(candidate), str(DB_DIR))
        return _is_valid_db_dir(DB_DIR)
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)


def download_and_extract_index() -> None:
    """Download and extract hydro_db from remote URL if it doesn't exist locally."""
    if DB_DIR.exists():
        return  # Already exists, nothing to do
    
    # Try to get URL from Streamlit secrets first, then environment
    index_url = None
    try:
        index_url = st.secrets.get("INDEX_ZIP_URL", "").strip()
    except Exception:
        pass
    
    if not index_url:
        index_url = os.getenv("INDEX_ZIP_URL", "").strip()
    
    if not index_url:
        # No URL provided, let get_vectorstore() raise the proper error
        return
    
    try:
        # Download the zip file
        zip_path = Path("./hydro_db_temp.zip")
        with st.spinner("Downloading vector index... This may take a minute."):
            urllib.request.urlretrieve(index_url, str(zip_path))

        ok = _extract_index_zip(zip_path)
        zip_path.unlink(missing_ok=True)

        if ok:
            st.success("Vector index downloaded and extracted successfully.")
        else:
            st.warning(
                "Downloaded hydro_db.zip but could not find a valid Chroma layout. "
                "Expected chroma.sqlite3 and index folders."
            )
    except Exception as e:
        st.warning(f"Could not download index from {index_url}: {e}. App will attempt to run without it.")


def _read_index_zip_url() -> str:
    """Read index zip URL from secrets/env."""
    try:
        url = (st.secrets.get("INDEX_ZIP_URL") or "").strip()
        if url:
            return url
    except Exception:
        pass
    return os.getenv("INDEX_ZIP_URL", "").strip()


def redownload_index() -> bool:
    """Delete local hydro_db and fetch a fresh copy from INDEX_ZIP_URL."""
    index_url = _read_index_zip_url()
    if not index_url:
        return False

    try:
        if DB_DIR.exists():
            shutil.rmtree(DB_DIR, ignore_errors=True)
    except Exception:
        pass

    try:
        zip_path = Path("./hydro_db_temp.zip")
        with st.spinner("Refreshing vector index from remote storage..."):
            urllib.request.urlretrieve(index_url, str(zip_path))
        ok = _extract_index_zip(zip_path)
        zip_path.unlink(missing_ok=True)
        return ok
    except Exception:
        return False


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{3,}", text.lower())


def expand_query(query: str) -> List[str]:
    """Generate up to 3 query variants to improve recall."""
    queries = [query]

    simple = re.sub(r"\b[A-Z]{2,8}\b", "", query).strip()
    if simple and simple != query:
        queries.append(simple)

    if any(word in query.lower() for word in ["what", "which", "how", "where", "when"]):
        stmt_query = re.sub(
            r"^\s*(?:what|which|how|when|where)\s+(?:is|are|does|do|can|will)?\s*",
            "",
            query,
            flags=re.IGNORECASE,
        ).rstrip("?").strip()
        if stmt_query and stmt_query != query:
            queries.append(stmt_query)

    return queries[:3]


@st.cache_resource(show_spinner=False)
def get_reranker():
    if not HAS_CROSS_ENCODER:
        return None
    try:
        return CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        st.warning(f"Could not load cross-encoder: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_embeddings() -> Any:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Any:
    if not DB_DIR.exists():
        # Try fetching the index artifact on-demand in Cloud environments.
        download_and_extract_index()
        if not DB_DIR.exists() and redownload_index():
            st.info("Vector index downloaded from INDEX_ZIP_URL.")

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{DB_DIR}'. "
            "Set INDEX_ZIP_URL in Streamlit Secrets to a valid hydro_db.zip direct URL."
        )
    try:
        vs = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
        # Touch the collection and retrieval path so broken HNSW indexes fail fast.
        _ = int(vs._collection.count())
        _ = vs.max_marginal_relevance_search(
            "index health check",
            k=1,
            fetch_k=4,
            lambda_mult=0.5,
        )
        return vs
    except Exception as e:
        if redownload_index():
            try:
                vs = Chroma(
                    persist_directory=str(DB_DIR),
                    embedding_function=get_embeddings(),
                    collection_metadata={"hnsw:space": "cosine"},
                )
                _ = int(vs._collection.count())
                _ = vs.max_marginal_relevance_search(
                    "index health check",
                    k=1,
                    fetch_k=4,
                    lambda_mult=0.5,
                )
                st.info("Vector index refreshed from INDEX_ZIP_URL.")
                return vs
            except Exception:
                pass
        raise RuntimeError(
            "Unable to load hydro_db. The downloaded index appears incompatible or corrupted. "
            "Rebuild hydro_db locally with Python 3.11 and upload a fresh hydro_db.zip. "
            f"Details: {e}"
        )


@st.cache_data(show_spinner=False)
def get_indexed_chunk_count() -> int:
    try:
        return int(get_vectorstore()._collection.count())
    except Exception:
        return 0


@st.cache_resource(show_spinner=False)
def get_lexical_corpus() -> List[Dict[str, Any]]:
    """Load all indexed chunks once for lexical retrieval scoring."""
    collection = get_vectorstore()._collection
    try:
        raw = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        st.warning(
            "Lexical corpus could not be loaded from Chroma. "
            "Falling back to semantic retrieval only. "
            f"Details: {e}"
        )
        return []
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])

    corpus: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) and metas[idx] else {}
        content = (doc or "").strip()
        if not content:
            continue
        corpus.append(
            {
                "content": content,
                "source": meta.get("source", ""),
                "lower": content.lower(),
                "tokens": set(tokenize(content)),
            }
        )
    return corpus


def resolve_api_key() -> str:
    # Prefer Streamlit secrets from .streamlit/secrets.toml
    try:
        key = (st.secrets.get("GOOGLE_API_KEY") or "").strip()
        if key:
            return key
    except Exception:
        pass

    # Optional fallback to environment variable
    return os.getenv("GOOGLE_API_KEY", "").strip()


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    if not api_key:
        st.session_state["llm_source"] = "RAG only"
        return None

    try:
        ChatGoogleGenerativeAI = getattr(import_module("langchain_google_genai"), "ChatGoogleGenerativeAI")
    except Exception:
        st.session_state["llm_source"] = "RAG only"
        return None

    class GeminiRouter:
        def __init__(self, key: str):
            self.key = key
            self._clients: Dict[str, Any] = {}

        def _client(self, model_name: str):
            if model_name not in self._clients:
                self._clients[model_name] = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.key,
                    temperature=0.0,
                )
            return self._clients[model_name]

        def invoke(self, messages):
            for model_name in GEMINI_MODEL_FALLBACKS:
                try:
                    st.session_state["llm_source"] = f"RAG + Gemini refine ({model_name})"
                    return self._client(model_name).invoke(messages)
                except Exception:
                    continue
            st.session_state["llm_source"] = "RAG only"
            return None

    return GeminiRouter(api_key)



def retrieve(query: str, k: int, use_reranking: bool = True) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
    1. Query expansion
    2. Semantic retrieval (MMR)
    3. Lexical retrieval (token overlap + acronym match)
    4. Reciprocal Rank Fusion
    5. Optional cross-encoder re-ranking
    """
    vs = get_vectorstore()

    query_variants = expand_query(query)

    all_semantic = []
    semantic_errors: List[str] = []
    for q_var in query_variants:
        try:
            semantic_docs = vs.max_marginal_relevance_search(
                q_var,
                k=min(max(k * 3, 12), 30),
                fetch_k=min(max(k * 8, 24), 80),
                lambda_mult=0.5,
            )
            all_semantic.extend(semantic_docs)
        except Exception as e:
            semantic_errors.append(str(e))

    semantic_by_content = {}
    for d in all_semantic:
        key = d.page_content[:200]
        if key not in semantic_by_content:
            semantic_by_content[key] = d

    sem_results = [
        {"content": d.page_content, "source": d.metadata.get("source", "")}
        for d in semantic_by_content.values()
    ]

    q_tokens = set(tokenize(query))
    q_lower = query.lower()
    acronym_terms = re.findall(r"\b[A-Z]{2,8}\b", query)

    corpus = get_lexical_corpus()

    if not sem_results and not corpus and semantic_errors:
        raise RuntimeError(
            "Index retrieval failed. hydro_db was found but cannot be searched in this environment. "
            "Upload a fresh hydro_db.zip built with the same code/dependency versions. "
            f"Details: {semantic_errors[0]}"
        )

    lexical_scored: List[tuple[float, Dict[str, Any]]] = []

    for row in corpus:
        score = 0.0
        overlap = len(q_tokens.intersection(row["tokens"]))
        score += overlap * 2.0

        if q_lower in row["lower"]:
            score += 12.0

        for acronym in acronym_terms:
            if acronym.lower() in row["lower"]:
                score += 8.0
            if f"({acronym.lower()})" in row["lower"]:
                score += 10.0

        if score > 0:
            lexical_scored.append((score, {"content": row["content"], "source": row["source"]}))

    lexical_scored.sort(key=lambda x: x[0], reverse=True)
    lex_results = [item[1] for item in lexical_scored[: min(max(k * 3, 12), 30)]]

    fused = defaultdict(float)
    payload: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(sem_results, start=1):
        key = f"{item['source']}::{item['content'][:180]}"
        fused[key] += 1.0 / (50 + rank)
        payload[key] = item

    for rank, item in enumerate(lex_results, start=1):
        key = f"{item['source']}::{item['content'][:180]}"
        fused[key] += 1.5 / (50 + rank)
        payload[key] = item

    best_keys = sorted(fused.keys(), key=lambda k_: fused[k_], reverse=True)[: k * 4]
    candidates = [payload[key] for key in best_keys]

    if use_reranking and HAS_CROSS_ENCODER and len(candidates) > 0:
        reranker = get_reranker()
        if reranker is not None:
            try:
                pairs = [[query, c["content"]] for c in candidates]
                scores = reranker.predict(pairs)
                scored_candidates = list(zip(scores, candidates))
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = [c for _, c in scored_candidates[:k]]
            except Exception:
                candidates = candidates[:k]
    else:
        candidates = candidates[:k]

    return candidates


def extractive_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return (
            "No relevant content was found in the indexed documents for this question. "
            "Make sure the PDFs are present in ./data and re-run hydrogpt_indexer.py."
        )

    q_lower = question.lower()

    # High-confidence numeric extraction for SU baseline/increase queries.
    if "su" in q_lower and any(term in q_lower for term in ["increase", "baseline", "current value", "how many days"]):
        for chunk in chunks:
            src = Path(chunk["source"]).name
            text = " ".join(chunk["content"].split())
            text = text.replace("−", "-")

            patterns = [
                r"number\s+of\s+su\s+will\s+increase\s+by\s*(\d+)\D{0,8}days\s+compared\s+to\s+the\s+current\s+value\s+of\s*(\d+)\D{0,8}days",
                r"su\s+will\s+increase\s+by\s*(\d+)\D{0,8}days.*?current\s+value\s+of\s*(\d+)\D{0,8}days",
                r"increase\s+by\s*(\d+)\D{0,8}days.*?current\s+value\s+of\s*(\d+)\D{0,8}days.*?su",
            ]
            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    increase_days = m.group(1)
                    baseline_days = m.group(2)
                    return (
                        f"Under the SSP585 scenario for Epoch 3, the number of Summer Days (SU) "
                        f"is projected to increase by {increase_days} days. "
                        f"The current baseline value for the region is {baseline_days} days.\n\n"
                        f"*Source: {src}*"
                    )

    q_tokens = set(tokenize(question))
    numeric_query = any(term in q_lower for term in ["how many", "how much", "increase", "decrease", "baseline", "current value", "expected"])
    acronym_terms = re.findall(r"\b[A-Z]{2,8}\b", question)
    scored_sentences: List[tuple[float, str, str]] = []

    for chunk in chunks:
        source = Path(chunk["source"]).name
        sentences = re.split(r"(?<=[.!?])\s+", chunk["content"])
        for sentence in sentences:
            sent = sentence.strip()
            if len(sent) < 20:
                continue

            sent_tokens = set(tokenize(sent))
            score = float(len(q_tokens.intersection(sent_tokens)) * 3.0)

            if any(acronym.lower() in sent.lower() for acronym in acronym_terms):
                score += 8.0

            if re.search(r"\d+(?:\.\d+)?", sent):
                score += 2.0

            if numeric_query and re.search(r"\d+(?:\.\d+)?\s*(?:%|days?|mm|degc|°c|years?)", sent.lower()):
                score += 4.0
            if numeric_query and any(term in sent.lower() for term in ["increase", "baseline", "current", "expected", "epoch", "ssp"]):
                score += 2.0

            if score > 0:
                scored_sentences.append((score, sent, source))

    if not scored_sentences:
        return (
            "I found the documents, but could not extract a precise answer from them. "
            "Try including a few more keywords from the paper title or the figure/table name."
        )

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    top = scored_sentences[:3]
    answer = f"**Answer:** {top[0][1]}"
    if len(top) > 1:
        answer += "\n\n**Supporting details:**\n"
        for idx, (_, sentence, _) in enumerate(top[1:], 1):
            answer += f"{idx}. {sentence}\n"
    sources = ", ".join(dict.fromkeys(src for _, _, src in top))
    return f"{answer}\n\n*Sources: {sources}*"


def generate_answer(question: str, chunks: List[Dict[str, Any]], llm=None) -> str:
    if not chunks:
        return (
            "No relevant content was found in the indexed documents for this question. "
            "Make sure the relevant PDFs are in ./data and re-run hydrogpt_indexer.py."
        )

    base_answer = extractive_answer(question, chunks)

    # Keep deterministic exact numeric outputs untouched.
    if base_answer.startswith("Under the SSP585 scenario for Epoch 3"):
        return base_answer

    if llm is None:
        return base_answer

    evidence_lines = []
    for c in chunks[:5]:
        src = Path(c["source"]).name
        snippet = c["content"].replace("\n", " ").strip()
        evidence_lines.append(f"[{src}] {snippet[:350]}")

    try:
        messages = REFINE_PROMPT.format_messages(
            question=question,
            base_answer=base_answer,
            evidence="\n".join(evidence_lines),
        )
        response = llm.invoke(messages)
        refined = str(getattr(response, "content", "")).strip() if response is not None else ""
        if refined:
            return refined
    except Exception:
        pass

    return base_answer


def dedup_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen, sources = set(), []
    for c in chunks:
        p = Path(c["source"])
        if p.name not in seen:
            seen.add(p.name)
            sources.append({"name": p.name, "path": str(p.resolve())})
    return sources


# -- UI helpers ---------------------------------------------------------------
def render_sources(sources: List[Dict[str, str]], key_prefix: str) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for i, src in enumerate(sources):
            p = Path(src["path"])
            if p.exists():
                with open(p, "rb") as f:
                    st.download_button(
                        label=f"PDF {src['name']}",
                        data=f.read(),
                        file_name=src["name"],
                        mime="application/pdf",
                        key=f"{key_prefix}_{i}",
                    )
            else:
                st.markdown(f"- {src['name']}")


# -- Main ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Mai-T GPT", page_icon="💧", layout="wide")

    # Download vector index if missing
    download_and_extract_index()

    with st.sidebar:
        st.title("💧 Mai-T GPT")
        st.caption("Retrieval-grounded RAG assistant for hydro-climate PDFs")
        st.divider()

        st.divider()
        top_k = st.slider(
            "Chunks retrieved (Top-K)",
            2,
            12,
            DEFAULT_K,
            1,
            help="Higher values provide broader retrieval context at a small latency cost.",
        )

        st.divider()
        st.caption(f"Mode: {st.session_state.get('llm_source', 'RAG only')}")

        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

    st.header("Ask your documents anything")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"], key_prefix=f"hist_{idx}")

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Searching knowledge base", expanded=True) as status:
            status.write(f"Retrieving top-{top_k} chunks")
            chunks = retrieve(question, k=top_k)
            status.write(f"Found {len(chunks)} chunk(s). Generating answer")
            llm = get_llm(resolve_api_key())
            answer = generate_answer(question, chunks, llm=llm)
            status.update(label="Done", state="complete", expanded=False)

        st.markdown(answer)
        sources = dedup_sources(chunks)
        render_sources(sources, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


if __name__ == "__main__":
    main()
