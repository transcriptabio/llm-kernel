"""Tests for data_models module."""

from absl.testing import absltest

from llm_kernel import data_models


class RNASeqTests(absltest.TestCase):
    """Tests for RNASeq data model."""

    def test_rna_creation(self):
        """Test creating RNASeq instance with valid data."""
        rna = data_models.RNASeq(
            id="sample1",
            gene_names=["GENE1", "GENE2"],
            lfc=[1.0, -0.5],
            fdr=[0.01, 0.05],
        )
        self.assertEqual(rna.id, "sample1")
        self.assertEqual(len(rna.gene_names), 2)
        self.assertEqual(len(rna.lfc), 2)
        self.assertEqual(len(rna.fdr), 2)

    def test_list_length_validation(self):
        """Test that mismatched list lengths raise ValueError."""
        with self.assertRaises(ValueError):
            data_models.RNASeq(
                id="sample1",
                gene_names=["GENE1", "GENE2"],
                lfc=[1.0],
                fdr=[0.01, 0.05],
            )

    def test_similarity_score_bounds(self):
        """Test similarity score validation."""
        rna_x = data_models.RNASeq(
            id="x", gene_names=["GENE1"], lfc=[1.0], fdr=[0.01]
        )
        rna_y = data_models.RNASeq(
            id="y", gene_names=["GENE1"], lfc=[0.5], fdr=[0.02]
        )

        # Valid scores
        data_models.RNASeqPair(x=rna_x, y=rna_y, similarity_score=0.8)

        # Invalid scores
        with self.assertRaises(Exception):  # pydantic.ValidationError
            data_models.RNASeqPair(x=rna_x, y=rna_y, similarity_score=1.5)


class TextTests(absltest.TestCase):
    """Tests for Text data model."""

    def test_text_creation(self):
        """Test creating Text instance with valid data."""
        text = data_models.Text(id="desc1", text="Cell proliferation response")
        self.assertEqual(text.id, "desc1")
        self.assertEqual(text.text, "Cell proliferation response")

    def test_text_pair_similarity_score_bounds(self):
        """Test TextPair similarity score validation."""
        text_x = data_models.Text(id="x", text="Sample text")
        text_y = data_models.Text(id="y", text="Another text")

        # Valid scores
        data_models.TextPair(x=text_x, y=text_y, similarity_score=1.0)

        # Invalid scores
        with self.assertRaises(Exception):  # pydantic.ValidationError
            data_models.TextPair(x=text_x, y=text_y, similarity_score=1.5)


if __name__ == "__main__":
    absltest.main()
