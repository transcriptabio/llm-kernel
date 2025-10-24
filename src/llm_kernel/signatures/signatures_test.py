"""Tests for signatures module."""

from absl.testing import absltest

from llm_kernel import data_models
from llm_kernel.signatures import kernels


class KernelSignatureTests(absltest.TestCase):
    """Tests for kernel signatures."""

    def test_format_data_rna(self):
        """Test format_data method for RNA data."""
        rna_data = data_models.RNASeq(
            id="test",
            gene_names=["GENE1", "GENE2"],
            lfc=[1.0, -0.5],
            fdr=[0.01, 0.05],
        )

        formatted = kernels.RNAvsRNAKernelSignature.format_data(rna_data)
        self.assertIsInstance(formatted, str)
        self.assertIn("GENE1", formatted)

    def test_format_data_text(self):
        """Test format_data method for text data."""
        text_data = data_models.Text(id="test", text="Cell proliferation")

        formatted = kernels.TextvsTextKernelSignature.format_data(text_data)
        self.assertEqual(formatted, "Cell proliferation")

    def test_text_vs_rna_format_data(self):
        """Test TextvsRNAKernelSignature format_data with both types."""
        # Test with RNA data
        rna_data = data_models.RNASeq(
            id="test", gene_names=["GENE1"], lfc=[1.0], fdr=[0.01]
        )
        formatted = kernels.TextvsRNAKernelSignature.format_data(rna_data)
        self.assertIn("GENE1", formatted)

        # Test with text data
        text_data = data_models.Text(id="test", text="Cell proliferation")
        formatted = kernels.TextvsRNAKernelSignature.format_data(text_data)
        self.assertEqual(formatted, "Cell proliferation")


if __name__ == "__main__":
    absltest.main()
