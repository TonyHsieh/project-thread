# `search_corpus`

**Module:** [`app/search.py`](../app/search.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`search_corpus` provides semantic vector retrieval over the project `corpus/` directory using pure Python and NumPy cosine similarity. It recursively ingests documents (`.txt`, `.md`, `.json`, `.pdf`), maintains an incremental disk cache of vector embeddings (`.corpus_cache.json`), and returns rank-ordered [`SearchResult`](domain_models.md#searchresult) objects.

---

## 2. Function Signature

```python
def search_corpus(
    query: str,
    top_k: int = 5,
    *,
    corpus_path: str | Path | None = None,
    embedding_model: str | None = None,
    min_score_threshold: float = 0.0,
) -> list[SearchResult]:
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | `str` | *required* | The search text or query string. |
| `top_k` | `int` | `5` | Maximum number of ranked results to return. |
| `corpus_path` | `str \| Path \| None` | `None` | Path to corpus folder. Defaults to `corpus/` relative to project structure. |
| `embedding_model`| `str \| None` | `None` | Embedding model ID. Defaults to `config.get_default_embedding_model_id()` (`text-embedding-3-small`). |
| `min_score_threshold` | `float` | `0.0` | Minimum cosine similarity score required for inclusion. |

### Return Value
Returns a list of [`SearchResult`](domain_models.md#searchresult) objects ordered from best to worst.

---

## 3. Invariants & Rules
1. **Query Rule**:
   An empty or whitespace-only query string (`""`, `"   "`) immediately returns an empty list `[]`. It never returns arbitrary documents.
2. **Ordering Rule**:
   Results are strictly sorted in descending score order (`best -> worst`).
3. **Tolerance Rule**:
   Cosine similarity calculations are computed via normalized NumPy dot product with check tolerance within $0.001$:
   $$\text{sim}(a, b) = \frac{a \cdot b}{\|a\|_2 \|b\|_2}$$
4. **Incremental Disk Caching**:
   Embeddings and SHA-256 content hashes are cached in `.corpus_cache.json`. When files are unchanged on disk, zero embedding API calls are triggered. Deleted files are purged from the cache automatically.
5. **Phase 1 Whole-Document Granularity**:
   In Phase 1, `chunk_id` is set to `f"{source_filename}:0"`, reserving 500-token/50-overlap chunking for the Phase 5 classic RAG indexer.
6. **Metadata Guard**:
   `metadata` includes `relative_path`, `subdirectory`, `file_extension`, and `char_count`. It never contains system paths or credentials.

---

## 4. CLI Execution & Standalone Usage

```bash
# Run default query
python app/search.py

# Run with custom query argument
python app/search.py "transformers and self-attention"
```

### Python Usage Example
```python
import search

results = search.search_corpus("How do transformers work?", top_k=3)
for r in results:
    print(f"[{r.score:.4f}] {r.source_filename} (chunk: {r.chunk_id})")
    print(f"Text preview: {r.text[:120]}...\n")
```

