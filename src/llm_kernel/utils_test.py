"""Tests for utils module."""

import anndata
import numpy as np
from absl.testing import absltest

from llm_kernel import utils


class AdataToSignaturesTests(absltest.TestCase):
    """Tests for adata_to_signatures function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create minimal mock AnnData object
        n_genes, n_samples = 10, 2
        X = np.random.randn(n_samples, n_genes)

        self.adata = anndata.AnnData(X=X)
        self.adata.var_names = [f"GENE_{i}" for i in range(n_genes)]
        self.adata.obs_names = [f"sample_{i}" for i in range(n_samples)]

        # Add hgnc_symbol column to var dataframe (required by adata_to_signatures)
        self.adata.var["hgnc_symbol"] = [f"GENE_{i}" for i in range(n_genes)]

        self.adata.layers["X"] = np.random.randn(n_samples, n_genes) * 2
        self.adata.layers["q_value"] = np.random.rand(n_samples, n_genes) * 0.01

    def test_adata_to_signatures_returns_list(self):
        """Test that adata_to_signatures returns a list of signatures."""
        signatures = utils.adata_to_signatures(
            self.adata,
            lfc_threshold=0.0,
            fdr_threshold=0.1,
            max_genes=5,
            min_genes=1,
        )

        self.assertIsInstance(signatures, list)
        self.assertGreater(len(signatures), 0)

    def test_adata_to_signatures_with_formatting(self):
        """Test adata_to_signatures with number formatting."""
        signatures = utils.adata_to_signatures(
            self.adata,
            lfc_threshold=0.0,
            fdr_threshold=0.05,
            max_genes=5,
            min_genes=1,
            format_numbers=True,
        )

        self.assertIsInstance(signatures, list)
        if signatures:
            self.assertIsInstance(signatures[0].lfc[0], float)
        else:
            self.assertEqual(len(signatures), 0)


if __name__ == "__main__":
    absltest.main()
