
import unittest
from unittest.mock import patch

from evaluation import (
    evaluate_retrieval,
    filter_relevant_chunks,
)

from rag_pipeline import rag_pipeline


class TestRetrievalEvaluation(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {
                "document_id": "doc_1",
                "source_name": "ml.txt",
                "chunk_id": 0,
                "text": "Machine learning uses data.",
                "distance": 0.15
            },
            {
                "document_id": "doc_1",
                "source_name": "ml.txt",
                "chunk_id": 1,
                "text": "Neural networks are ML models.",
                "distance": 0.35
            },
            {
                "document_id": "doc_2",
                "source_name": "python.txt",
                "chunk_id": 0,
                "text": "Python is a programming language.",
                "distance": 0.75
            }
        ]

    def test_filter_relevant_chunks(self):
        relevant = filter_relevant_chunks(
            self.chunks,
            distance_threshold=0.5
        )

        self.assertEqual(len(relevant), 2)
        self.assertEqual(relevant[0]["chunk_id"], 0)
        self.assertEqual(relevant[1]["chunk_id"], 1)

    def test_retrieval_evaluation(self):
        evaluation = evaluate_retrieval(
            self.chunks,
            distance_threshold=0.5
        )

        self.assertTrue(evaluation.is_relevant)
        self.assertEqual(evaluation.retrieved_count, 3)
        self.assertEqual(evaluation.accepted_count, 2)
        self.assertEqual(evaluation.best_distance, 0.15)

    def test_weak_retrieval(self):
        weak_chunks = [
            {"distance": 0.7},
            {"distance": 0.9}
        ]

        evaluation = evaluate_retrieval(
            weak_chunks,
            distance_threshold=0.5
        )

        self.assertFalse(evaluation.is_relevant)
        self.assertEqual(evaluation.accepted_count, 0)

    def test_empty_retrieval(self):
        evaluation = evaluate_retrieval([])

        self.assertFalse(evaluation.is_relevant)
        self.assertIsNone(evaluation.best_distance)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            filter_relevant_chunks(
                self.chunks,
                distance_threshold=-0.1
            )

    @patch("rag_pipeline.generate_answer")
    @patch("rag_pipeline.retrieve_chunks")
    def test_weak_context_skips_llm(
        self,
        mock_retrieve,
        mock_generate
    ):
        mock_retrieve.return_value = [
            {
                "document_id": "doc_1",
                "source_name": "ml.txt",
                "chunk_id": 0,
                "text": "Unrelated content",
                "distance": 0.9
            }
        ]

        result = rag_pipeline(
            "What is the weather on Mars?",
            distance_threshold=0.5
        )

        self.assertEqual(
            result["status"],
            "insufficient_context"
        )
        self.assertEqual(result["sources"], [])

        mock_generate.assert_not_called()

    @patch("rag_pipeline.generate_answer")
    @patch("rag_pipeline.retrieve_chunks")
    def test_relevant_context_generates_answer(
        self,
        mock_retrieve,
        mock_generate
    ):
        mock_retrieve.return_value = self.chunks
        mock_generate.return_value = "ML uses data. [1]"

        result = rag_pipeline(
            "What is machine learning?",
            distance_threshold=0.5
        )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["answer"], "ML uses data. [1]")

        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
