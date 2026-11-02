"""
Unit, invariant, and boundary tests for app/search.py (Phase 1 Search Owner).

Covers:
- Query rule: Empty or whitespace queries return [] (never arbitrary documents)
- Ordering rule: Strictly sorted best -> worst
- Cosine check tolerance: Verified within 0.001
- Ingestion: Recursion, subdirectories, and metadata sanitation
- Incremental caching: Avoids re-embedding when content hash is unchanged
- Failure modes: SearchError raised on API embedding failures
"""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure app directory is in sys.path
_app_dir = Path(__file__).resolve().parent.parent.parent / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from domain.errors import SearchError
from domain.models import SearchResult
import search


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity math and 0.001 tolerance check."""

    def test_identical_vectors(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        sim = search.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 1.0, delta=0.001)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = search.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, delta=0.001)

    def test_known_angle_vector(self):
        v1 = [1.0, 0.0]
        v2 = [1.0, 1.0]
        expected = 1.0 / np.sqrt(2.0)  # ~0.7071
        sim = search.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, expected, delta=0.001)

    def test_zero_vector(self):
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        sim = search.compute_cosine_similarity(v1, v2)
        self.assertEqual(sim, 0.0)


class TestSearchQueryRules(unittest.TestCase):
    """Test query boundaries and invariants."""

    def test_empty_and_whitespace_query_returns_empty_list(self):
        queries = ["", "   ", "\t\n", "   \r\n  "]
        for q in queries:
            with self.subTest(query=repr(q)):
                results = search.search_corpus(q)
                self.assertEqual(results, [])

    def test_zero_or_negative_top_k_returns_empty_list(self):
        self.assertEqual(search.search_corpus("valid query", top_k=0), [])
        self.assertEqual(search.search_corpus("valid query", top_k=-5), [])


class TestSearchExecution(unittest.TestCase):
    """Test corpus ingestion, ranking, caching, and domain contracts with mock embeddings."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus_path = Path(self.temp_dir.name)

        # Create sample files
        (self.corpus_path / "doc1.txt").write_text("Introduction to AI and machine learning.", encoding="utf-8")
        (self.corpus_path / "notes.md").write_text("# Deep Learning\nTransformers and self-attention.", encoding="utf-8")

        sub_dir = self.corpus_path / "sub"
        sub_dir.mkdir()
        (sub_dir / "data.json").write_text(json.dumps({"topic": "Data pipelines and retrieval"}), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("config.get_client")
    def test_search_corpus_ordering_and_metadata(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock embeddings: query closer to doc1 than doc2 or data.json
        # doc1 -> [1.0, 0.0], notes.md -> [0.8, 0.2], sub/data.json -> [0.1, 0.9]
        def fake_embeddings_create(input, model):
            if isinstance(input, str):
                # Query embedding
                return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])
            else:
                # Batch document embeddings
                emb_map = {
                    "Introduction": [1.0, 0.0],  # Sim = 1.0 (doc1.txt)
                    "Transformers": [0.7, 0.7],  # Sim = 0.707 (notes.md)
                    "Data": [0.0, 1.0],          # Sim = 0.0 (sub/data.json)
                }
                data_items = []
                for idx, text in enumerate(input):
                    for key, vec in emb_map.items():
                        if key in text:
                            data_items.append(SimpleNamespace(index=idx, embedding=vec))
                            break
                    else:
                        data_items.append(SimpleNamespace(index=idx, embedding=[0.5, 0.5]))
                return SimpleNamespace(data=data_items)

        mock_client.embeddings.create.side_effect = fake_embeddings_create

        results = search.search_corpus(
            query="AI learning",
            top_k=5,
            corpus_path=self.corpus_path,
        )

        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], SearchResult)

        # Invariant: Ordering strictly best -> worst
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

        # Invariant: Best match is doc1.txt with cosine ~1.0
        self.assertEqual(results[0].source_filename, "doc1.txt")
        self.assertAlmostEqual(results[0].score, 1.0, delta=0.001)
        self.assertEqual(results[0].chunk_id, "doc1.txt:0")

        # Invariant: Subdirectory metadata is populated
        sub_results = [r for r in results if "sub" in r.source_filename]
        if sub_results:
            self.assertEqual(sub_results[0].metadata["subdirectory"], "sub")

    @patch("config.get_client")
    def test_incremental_cache_avoids_re_embedding(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[0.1, 0.2]),
                SimpleNamespace(index=1, embedding=[0.3, 0.4]),
                SimpleNamespace(index=2, embedding=[0.5, 0.6]),
            ]
        )

        # First run: builds cache
        index = search.CorpusIndex(
            corpus_dir=self.corpus_path,
            cache_path=self.corpus_path / ".corpus_cache.json",
            embedding_model_id="test-model",
        )
        index.sync()
        initial_call_count = mock_client.embeddings.create.call_count
        self.assertGreater(initial_call_count, 0)

        # Second run without file changes: uses cache, no embedding calls
        index2 = search.CorpusIndex(
            corpus_dir=self.corpus_path,
            cache_path=self.corpus_path / ".corpus_cache.json",
            embedding_model_id="test-model",
        )
        index2.sync()
        self.assertEqual(mock_client.embeddings.create.call_count, initial_call_count)

    @patch("config.get_client")
    def test_search_error_on_api_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("Embedding service unavailable")

        with self.assertRaises(SearchError):
            search.search_corpus("Any query", corpus_path=self.corpus_path)


if __name__ == "__main__":
    unittest.main()
