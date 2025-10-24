"""Tests for scorer module."""

from unittest import mock

from absl.testing import absltest

from llm_kernel import scorer


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


if __name__ == "__main__":
    absltest.main()
