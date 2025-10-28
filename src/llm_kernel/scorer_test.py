"""Tests for scorer module."""

from typing import Any
from unittest import mock

from absl.testing import absltest

from llm_kernel import scorer


class MockDataObject:
    """Mock data object with id attribute."""

    def __init__(self, id_name: str):
        self.id = id_name


class KernelScorerTests(absltest.TestCase):
    """Tests for KernelScorer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = mock.MagicMock()
        self.config = scorer.KernelConfig(max_workers=1)

    def test_scorer_initialization(self):
        """Test KernelScorer initialization."""
        mock_signature_class = mock.MagicMock()
        mock_signature_class.__name__ = "TestSignature"

        scorer_instance = scorer.KernelScorer(
            client=self.mock_client,
            signature_class=mock_signature_class,
            config=self.config,
        )
        self.assertIsNotNone(scorer_instance)

    @mock.patch("llm_kernel.scorer.tqdm")
    def test_score_pairs(self, mock_tqdm):
        """Test score_pairs method."""
        mock_tqdm.return_value.__enter__ = mock.MagicMock(
            return_value=mock.MagicMock()
        )
        mock_tqdm.return_value.__exit__ = mock.MagicMock(return_value=False)

        mock_signature_class = mock.MagicMock()
        mock_signature_class.__name__ = "TestSignature"
        scorer_instance = scorer.KernelScorer(
            client=self.mock_client,
            signature_class=mock_signature_class,
            config=self.config,
        )

        # Create mock objects with ids
        obj1 = MockDataObject("obj1")
        obj2 = MockDataObject("obj2")
        obj3 = MockDataObject("obj3")

        # Create pairs
        pairs = [(obj1, obj2), (obj2, obj3)]

        # Mock score_pair to return different scores
        scores = {("obj1", "obj2"): 0.8, ("obj2", "obj3"): 0.6}
        responses = {
            ("obj1", "obj2"): mock.MagicMock(spec=["similarity_score"]),
            ("obj2", "obj3"): mock.MagicMock(spec=["similarity_score"]),
        }
        responses[("obj1", "obj2")].similarity_score = "0.8"
        responses[("obj2", "obj3")].similarity_score = "0.6"

        def mock_score_pair(x, y):
            key = (x.id, y.id)
            return scores[key], responses[key]

        scorer_instance.score_pair = mock_score_pair

        # Collect results from the iterator
        results = list(scorer_instance.score_pairs(pairs))

        # Verify results
        self.assertEqual(len(results), 2)

        # Extract scores and ids from results
        result_map = {(r["x"], r["y"]): r for r in results}

        self.assertEqual(result_map[("obj1", "obj2")]["similarity_score"], 0.8)
        self.assertEqual(result_map[("obj2", "obj3")]["similarity_score"], 0.6)
        self.assertIn("x", result_map[("obj1", "obj2")])
        self.assertIn("y", result_map[("obj1", "obj2")])
        self.assertIn("similarity_score", result_map[("obj1", "obj2")])
        self.assertIn("response", result_map[("obj1", "obj2")])


if __name__ == "__main__":
    absltest.main()
