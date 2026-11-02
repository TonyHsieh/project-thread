"""
Corpus search and retrieval module for project-thread-api/v1 (Phase 1 Search Owner).

Provides:
- search_corpus(query, top_k=5) -> list[SearchResult]
- Cosine similarity with 0.001 check tolerance.
- Incremental disk caching for document embeddings.
- Ingestion of .txt, .md, .json, and .pdf documents across recursive subdirectories.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np

from domain.errors import SearchError
from domain.models import SearchResult
import config

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def _read_file_content(file_path: Path) -> str:
    """Extract text content based on file extension."""
    ext = file_path.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, str):
                return data.strip()
            return json.dumps(data, indent=2)
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
        else:
            return ""
    except Exception as e:
        raise SearchError(f"Failed to read document '{file_path.name}': {e}") from e


def _compute_hash(content: str) -> str:
    """Compute deterministic SHA256 content hash."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_cosine_similarity(v1: list[float] | np.ndarray, v2: list[float] | np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors using NumPy.
    Tolerance precision: returns rounded float, robust against zero-norms.
    """
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Bound to [-1.0, 1.0] to account for floating point imprecision
    return max(-1.0, min(1.0, sim))


class CorpusIndex:
    """Manages document embeddings and incremental cache persistence."""

    def __init__(
        self,
        corpus_dir: Path,
        cache_path: Path,
        embedding_model_id: str,
    ) -> None:
        self.corpus_dir = corpus_dir.resolve()
        self.cache_path = cache_path.resolve()
        self.embedding_model_id = embedding_model_id
        self._entries: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("embedding_model_id") == self.embedding_model_id:
                self._entries = data.get("documents", {})
        except Exception:
            # Corrupted cache is ignored and rebuilt cleanly
            self._entries = {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "embedding_model_id": self.embedding_model_id,
                        "documents": self._entries,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            # Failure to write cache shouldn't break search
            pass

    def sync(self) -> None:
        """Scan corpus directory and incrementally update modified or new documents."""
        if not self.corpus_dir.is_dir():
            return

        current_files: dict[str, Path] = {}
        for root, _, files in os.walk(self.corpus_dir):
            for file in files:
                # Ignore hidden / system files (e.g. .corpus_cache.json, .gitkeep)
                if file.startswith("."):
                    continue
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    full_p = Path(root) / file
                    rel_p = str(full_p.relative_to(self.corpus_dir)).replace("\\", "/")
                    current_files[rel_p] = full_p

        # Remove deleted documents from cache
        stale_keys = [k for k in self._entries if k not in current_files]
        for k in stale_keys:
            del self._entries[k]

        docs_to_embed: list[tuple[str, str, dict[str, Any], float]] = []

        for rel_p, full_p in current_files.items():
            try:
                mtime = full_p.stat().st_mtime
            except OSError:
                continue

            entry = self._entries.get(rel_p)
            if entry and entry.get("mtime") == mtime:
                continue

            content = _read_file_content(full_p)
            if not content:
                continue

            content_hash = _compute_hash(content)
            if entry and entry.get("content_hash") == content_hash:
                # Content identical, update mtime only
                entry["mtime"] = mtime
                continue

            sub_dir = str(Path(rel_p).parent).replace("\\", "/")
            if sub_dir == ".":
                sub_dir = ""

            metadata = {
                "relative_path": rel_p,
                "subdirectory": sub_dir,
                "file_extension": full_p.suffix.lower(),
                "char_count": len(content),
            }
            docs_to_embed.append((rel_p, content, metadata, mtime))

        if docs_to_embed:
            client = config.get_client()
            texts = [item[1] for item in docs_to_embed]
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=self.embedding_model_id,
                )
                embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as e:
                raise SearchError(f"Embedding batch generation failed: {e}") from e

            for (rel_p, content, metadata, mtime), emb in zip(docs_to_embed, embeddings):
                self._entries[rel_p] = {
                    "mtime": mtime,
                    "content_hash": _compute_hash(content),
                    "text": content,
                    "embedding": emb,
                    "metadata": metadata,
                }

            self._save_cache()

    def get_entries(self) -> list[dict[str, Any]]:
        return list(self._entries.values())


def get_default_corpus_path() -> Path:
    """Locate corpus directory relative to project structure."""
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir.parent / "corpus",
        current_dir / "corpus",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return current_dir.parent / "corpus"


def search_corpus(
    query: str,
    top_k: int = 5,
    *,
    corpus_path: str | Path | None = None,
    embedding_model: str | None = None,
    min_score_threshold: float = 0.35,
) -> list[SearchResult]:
    """
    Search corpus documents using cosine similarity.

    Contract invariants:
    - Query rule: Empty or whitespace query returns [] (never arbitrary documents).
    - Ordering rule: Strictly sorted best -> worst by cosine score.
    - Tolerance: Precision tolerance <= 0.001.
    - Metadata: Never contains secrets or credentials.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    if top_k <= 0:
        return []

    target_corpus = Path(corpus_path) if corpus_path else get_default_corpus_path()
    if not target_corpus.is_dir():
        return []

    model_id = embedding_model if embedding_model else config.get_default_embedding_model_id()
    cache_file = target_corpus / ".corpus_cache.json"

    index = CorpusIndex(
        corpus_dir=target_corpus,
        cache_path=cache_file,
        embedding_model_id=model_id,
    )
    index.sync()

    entries = index.get_entries()
    if not entries:
        return []

    # Embed query
    client = config.get_client()
    try:
        query_response = client.embeddings.create(
            input=query.strip(),
            model=model_id,
        )
        query_vector = query_response.data[0].embedding
    except Exception as e:
        raise SearchError(f"Query embedding generation failed: {e}") from e

    scored_results: list[SearchResult] = []
    for entry in entries:
        score = compute_cosine_similarity(query_vector, entry["embedding"])
        if score >= min_score_threshold:
            rel_path = entry["metadata"].get("relative_path", "unknown")
            scored_results.append(
                SearchResult(
                    source_filename=rel_path,
                    chunk_id=f"{rel_path}:0",
                    text=entry["text"],
                    score=score,
                    metadata=entry["metadata"],
                )
            )

    # Sort descending: best -> worst
    scored_results.sort(key=lambda x: x.score, reverse=True)

    # Truncate to top_k
    return scored_results[:top_k]


def main(query: str | None = None) -> None:
    """Console demo under if __name__ == '__main__' per console rule."""
    default_query = "How does an LLM actually work and predict tokens?"
    active_query = query
    if active_query is None:
        if len(sys.argv) > 1 and sys.argv[1].strip():
            active_query = " ".join(sys.argv[1:]).strip()
        else:
            active_query = default_query

    print(f"--- Running search_corpus for: '{active_query}' ---")
    results = search_corpus(active_query, top_k=3)
    print(f"Results returned: {len(results)}")
    for idx, r in enumerate(results, start=1):
        print(f"\n[{idx}] Score: {r.score:.4f} | Source: {r.source_filename} | Chunk: {r.chunk_id}")
        preview = r.text.replace("\n", " ")[:150]
        print(f"    Text: {preview}...")
        print(f"    Metadata: {r.metadata}")


if __name__ == "__main__":
    main()
