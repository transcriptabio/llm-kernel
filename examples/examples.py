"""Comprehensive examples demonstrating kernel scoring usage patterns.

Provides practical examples of kernel scoring for different types of pairwise
comparisons. Demonstrates basic usage, custom signatures,
and data model creation.

Typical usage example:

  kernel_scorer = create_rna_vs_rna_scorer()
  rna_x = data_models.RNASeq(id='sample_x', gene_names=['GENE1'], lfc=[1.0], fdr=[0.01])
  rna_y = data_models.RNASeq(id='sample_y', gene_names=['GENE1'], lfc=[0.8], fdr=[0.02])
  score, response = kernel_scorer.score_pair(rna_x, rna_y)
"""

import dspy
from google import genai

from llm_kernel import data_models, scorer
from llm_kernel.signatures import individual, kernels, pairwise


def default_kernel_config():
    """Default kernel configuration."""
    return scorer.KernelConfig(
        model_name="gemini/gemini-2.5-pro", thinking_budget=8000, max_workers=5
    )


def default_client():
    """Default client configuration."""
    return genai.Client(
        vertexai=True, project="YOUR_PROJECT_ID", location="YOUR_LOCATION"
    )


def create_rna_vs_rna_scorer(client=None, config=None):
    """Factory function for RNA vs RNA scorer."""
    return scorer.KernelScorer(
        client=default_client() if not client else client,
        signature_class=kernels.RNAvsRNAKernelSignature,
        config=default_kernel_config() if not config else config,
    )


def create_text_vs_text_scorer(client=None, config=None):
    """Factory function for Text vs Text scorer."""
    return scorer.KernelScorer(
        client=default_client() if not client else client,
        signature_class=kernels.TextvsTextKernelSignature,
        config=default_kernel_config() if not config else config,
    )


def create_text_vs_rna_scorer(client=None, config=None):
    """Factory function for Text vs RNA scorer."""
    return scorer.KernelScorer(
        client=default_client() if not client else client,
        signature_class=kernels.TextvsRNAKernelSignature,
        config=default_kernel_config() if not config else config,
    )


def create_custom_rna_vs_rna_scorer(client=None, config=None):
    """Factory function for RNA vs RNA with custom analysis signatures."""

    # Create a custom signature with complex analysis types
    class CustomRNAvsRNAKernelSignature(kernels.RNAvsRNAKernelSignature):
        x_analysis: individual.FunctionalAnalysis = dspy.OutputField(
            description="Functional analysis of the first RNA profile"
        )
        y_analysis: individual.PathwayAnalysis = dspy.OutputField(
            description="Pathway analysis of the second RNA profile"
        )
        pairwise_analysis: pairwise.SemanticAnalysis = dspy.OutputField(
            description="Semantic comparative analysis of both RNA profiles"
        )

    return scorer.KernelScorer(
        client=default_client() if not client else client,
        signature_class=CustomRNAvsRNAKernelSignature,
        config=default_kernel_config() if not config else config,
    )


def create_pathway_focused_rna_scorer(client=None, config=None):
    """Factory function for RNA vs RNA with pathway-focused analysis."""

    # Create a custom signature with pathway-focused analysis types
    class PathwayRNAvsRNA(kernels.RNAvsRNAKernelSignature):
        x_analysis: individual.PathwayAnalysis = dspy.OutputField(
            description="Pathway analysis of the first RNA profile"
        )
        y_analysis: individual.PathwayAnalysis = dspy.OutputField(
            description="Pathway analysis of the second RNA profile"
        )
        pairwise_analysis: pairwise.FunctionalAnalysis = dspy.OutputField(
            description="Functional comparative analysis of both RNA profiles"
        )

    return scorer.KernelScorer(
        client=default_client() if not client else client,
        signature_class=PathwayRNAvsRNA,
        config=default_kernel_config() if not config else config,
    )


def example_rna_vs_rna():
    """Example: RNA vs RNA comparison using base class pattern."""
    kernel_scorer = create_rna_vs_rna_scorer()

    # Create some RNA signatures (example)
    rna_x = data_models.RNASeq(
        id="sample_x",
        gene_names=["GENE1", "GENE2"],
        lfc=[1.0, -0.5],
        fdr=[0.01, 0.05],
    )
    rna_y = data_models.RNASeq(
        id="sample_y",
        gene_names=["GENE1", "GENE2"],
        lfc=[0.8, -0.3],
        fdr=[0.02, 0.03],
    )

    score, response = kernel_scorer.score_pair(rna_x, rna_y)
    print(f"RNA vs RNA similarity: {score}")


def example_text_vs_text():
    """Example: Text vs Text comparison using base class pattern."""
    kernel_scorer = create_text_vs_text_scorer()

    # Score text pairs
    text_x = data_models.Text(
        id="text_x", text="Rapid growth and proliferation of cells"
    )
    text_y = data_models.Text(
        id="text_y", text="Programmed death and growth inhibition of cells"
    )

    score, response = kernel_scorer.score_pair(text_x, text_y)
    print(f"Text vs Text similarity: {score}")


def example_text_vs_rna():
    """Example: Text vs RNA comparison using base class pattern."""
    kernel_scorer = create_text_vs_rna_scorer()

    # Score text vs RNA
    text_x = data_models.Text(
        id="text_x", text="Inflammatory response and immune activation"
    )
    rna_y = data_models.RNASeq(
        id="inflammatory_sample",
        gene_names=["IL1B", "TNF", "IFNG"],
        lfc=[2.1, 1.8, 1.5],
        fdr=[0.001, 0.002, 0.01],
    )

    score, response = kernel_scorer.score_pair(text_x, rna_y)
    print(f"Text vs RNA similarity: {score}")


def example_kernel_pattern():
    """Example: Using the kernel pattern with custom analysis signatures."""
    kernel_scorer = create_custom_rna_vs_rna_scorer()

    # Create RNA signatures
    x = data_models.RNASeq(
        id="sample_x",
        gene_names=["IL1B", "TNF", "IFNG"],
        lfc=[1.0, -0.5, 2.1],
        fdr=[0.01, 0.05, 0.001],
    )
    y = data_models.RNASeq(
        id="sample_y",
        gene_names=["IL1B", "TNF", "IFNG"],
        lfc=[0.8, -0.3, 1.8],
        fdr=[0.02, 0.03, 0.002],
    )

    score, response = kernel_scorer.score_pair(x, y)
    print(f"Custom analysis similarity: {score}\n")
    print(f"X analysis: {response.x_analysis}\n")
    print(f"Y analysis: {response.y_analysis}\n")
    print(f"Pairwise analysis: {response.pairwise_analysis}")


if __name__ == "__main__":
    print("RNA vs RNA example:")
    example_rna_vs_rna()

    print("\nText vs Text example:")
    example_text_vs_text()

    print("\nText vs RNA example:")
    example_text_vs_rna()

    print("\nKernel pattern example:")
    example_kernel_pattern()
