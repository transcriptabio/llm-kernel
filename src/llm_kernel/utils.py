"""Data processing utilities for kernel scoring operations.

Provides utility functions for converting between data formats and preprocessing
biological datasets. Handles AnnData to kernel signature conversion and data
formatting.

Typical usage example:

  signatures = adata_to_signatures(
      adata,
      lfc_threshold=0.5,
      fdr_threshold=0.05,
      max_genes=1000
  )
"""

import itertools

import anndata
import numpy as np
from absl import logging

from llm_kernel import data_models


def generate_all_pairs(
    signatures: list[data_models.RNASeq],
) -> list[tuple[data_models.RNASeq, data_models.RNASeq]]:
    """Generate all unique pairs including self-pairs."""
    return list(itertools.combinations(signatures, 2)) + [
        (s, s) for s in signatures
    ]


def generate_reference_pairs(
    query_signatures: list[data_models.RNASeq],
    reference_signature: data_models.RNASeq,
) -> list[tuple[data_models.RNASeq, data_models.RNASeq]]:
    """Generate pairs of each query signature against a reference."""
    return [(query, reference_signature) for query in query_signatures]


def _format_mixed(arr: np.ndarray, precision: int = 4) -> np.ndarray:
    """Format numbers with sci notation except when exponent==0 (then fixed).
    Returns array[str]. Precision is significant digits.
    """
    arr = np.asarray(arr, dtype=float)

    def _one(x: float) -> str:
        s = np.format_float_scientific(
            float(x), precision=precision, unique=False, exp_digits=2
        )  # e.g., -1.7800e+00
        mant, _, exp = s.partition("e")
        try:
            if int(exp) == 0:
                # Use general format with significant digits; no trailing zeros/exponent.
                return format(float(x), f".{precision}g")
        except ValueError:
            pass
        return s

    return np.vectorize(_one, otypes=[float])(arr)


def adata_to_signatures(
    adata: anndata.AnnData,
    lfc_threshold: float = 0.5,
    fdr_threshold: float = 0.05,
    max_genes: int = 1000,
    min_genes: int = 50,
    lfc_layer: str = "X",
    fdr_layer: str = "q_value",
    precision: int = 2,
    random_state: int | None = 77,
    format_numbers: bool = True,
) -> list[data_models.RNASeq]:
    """Build per-sample transcriptome signatures from AnnData.

    Args:
      adata: AnnData object with samples in obs and genes in var.
      lfc_threshold: Minimum absolute log2 fold change to include.
      fdr_threshold: Maximum FDR to include.
      max_genes: Maximum number of genes per signature.
      min_genes: Minimum number of genes per signature.
      lfc_layer: Layer name for log2 fold changes (use 'X' for main matrix).
      fdr_layer: Layer name for FDR values.
      precision: Number of significant digits for formatting (if format_numbers=True).
      random_state: Random seed for gene ordering.
      format_numbers: If True, format numbers as strings; if False, keep as floats.

    Returns:
      List of RNASeq data models, one per sample.
    """
    rng = (
        np.random.default_rng(random_state)
        if random_state is not None
        else None
    )
    signatures = []

    # Get gene names with fallback to var_names
    gene_names_all = adata.var.get("hgnc_symbol", adata.var_names)
    gene_names_all = gene_names_all.fillna(
        adata.var_names.to_series()
    ).to_numpy()

    for sample in adata.obs_names:
        # Extract data for this sample
        lfc_values = (
            adata[sample, :].X
            if lfc_layer == "X"
            else adata[sample, :].layers[lfc_layer]
        ).flatten()
        fdr_values = adata[sample, :].layers[fdr_layer].flatten()

        # Filter genes by thresholds
        keep_mask = (np.abs(lfc_values) >= lfc_threshold) & (
            fdr_values <= fdr_threshold
        )
        if not np.any(keep_mask):
            continue

        lfc_filtered = lfc_values[keep_mask]
        fdr_filtered = fdr_values[keep_mask]
        genes_filtered = gene_names_all[keep_mask]

        # Check minimum gene count
        if len(lfc_filtered) < min_genes:
            logging.warning(
                "Sample %s has less than %d genes, skipping.", sample, min_genes
            )
            continue

        # Select top genes by absolute log2 fold change
        top_n = min(max_genes, len(lfc_filtered))
        gene_order = np.argsort(np.abs(lfc_filtered))[::-1][:top_n]

        # Apply random ordering if specified
        if rng is not None:
            gene_order = rng.permutation(gene_order)

        # Format data based on preference
        if format_numbers:
            lfc_formatted = _format_mixed(
                lfc_filtered[gene_order], precision
            ).tolist()
            fdr_formatted = _format_mixed(
                fdr_filtered[gene_order], precision
            ).tolist()
        else:
            lfc_formatted = lfc_filtered[gene_order].astype(float).tolist()
            fdr_formatted = fdr_filtered[gene_order].astype(float).tolist()

        # Create signature
        signatures.append(
            data_models.RNASeq(
                id=sample,
                gene_names=genes_filtered[gene_order].tolist(),
                lfc=lfc_formatted,
                fdr=fdr_formatted,
            )
        )

    return signatures
