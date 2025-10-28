"""Core kernel signatures for pairwise data comparisons.

Defines the main DSPy signatures that orchestrate pairwise comparisons between
different types of data. Combines individual and pairwise analysis
components for similarity scoring.

Typical usage example:

  class CustomRNAvsRNAKernelSignature(kernels.RNAvsRNAKernelSignature):
    x_analysis: individual.FunctionalAnalysis = dspy.OutputField(
        description='Functional analysis of the first RNA profile')
    y_analysis: individual.PathwayAnalysis = dspy.OutputField(
        description='Pathway analysis of the second RNA profile')
    pairwise_analysis: pairwise.SemanticAnalysis = dspy.OutputField(
        description='Semantic comparative analysis of both RNA profiles')
"""

import dspy

from analysis.kernel import data_models
from analysis.kernel.signatures import individual
from analysis.kernel.signatures import pairwise


class KernelSignatureBase(dspy.Signature):
    """Base kernel signature with standard structure.

    Subclasses should override x and y fields with specific descriptions
    for their data types and formats (e.g., RNA-seq profiles, text descriptions).

    Standard structure:
    - Input: x, y (the two objects to compare)
    - Output: x_analysis, y_analysis, pairwise_analysis, similarity_score
    """

    x: str = dspy.InputField(description="Object X to analyze")
    y: str = dspy.InputField(description="Object Y to analyze")

    x_analysis: individual.Analysis = dspy.OutputField(
        description="Analysis of object X"
    )
    y_analysis: individual.Analysis = dspy.OutputField(
        description="Analysis of object Y"
    )
    pairwise_analysis: pairwise.Analysis = dspy.OutputField(
        description="Structured comparative analysis between object X and object Y."
        "Should identify salient points of similarity and difference, and provide "
        "a concise qualitative assessment of their relationship. "
        "If the available evidence does not indicate clear alignment or opposition,"
        'the relationship should be characterized as "inconclusive" or "unrelated".'
    )
    similarity_score: float = dspy.OutputField(
        description=(
            "Continuous similarity metric ranging from -1.0 to 1.0. "
            "+1.0 represents strong alignment or high similarity; "
            "0.0 represents little or no meaningful relationship, or inconclusive evidence; "
            "-1.0 represents strong, systematic opposition or inverse correspondence."
            "Absence of overlap or unrelated characteristics should yield a score near 0, not a strong negative value."
        )
    )

    @staticmethod
    def format_data(data):
        """Convert data model to format expected by this kernel.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement format_data method"
        )


class RNAvsRNAKernelSignature(KernelSignatureBase):
    """RNA vs RNA kernel signature."""

    x: str = dspy.InputField(
        description="RNA-seq transcriptome profile X: "
        "[(HGNC_symbol, log2_fold_change, FDR), ...]"
    )
    y: str = dspy.InputField(
        description="RNA-seq transcriptome profile Y: "
        "[(HGNC_symbol, log2_fold_change, FDR), ...]"
    )

    @staticmethod
    def format_data(data: data_models.RNASeq) -> str:
        """Convert RNASeq data model to format expected by this kernel.

        Args:
          data: RNASeq data model to convert.

        Returns:
          String representation of the RNA-seq data in kernel format.
        """
        return str(list(zip(data.gene_names, data.lfc, data.fdr)))


class TextvsTextKernelSignature(KernelSignatureBase):
    """Text vs Text kernel signature."""

    x: str = dspy.InputField(description="Text description X")
    y: str = dspy.InputField(description="Text description Y")

    @staticmethod
    def format_data(data: data_models.Text) -> str:
        """Convert Text data model to format expected by this kernel.

        Args:
          data: Text data model to convert.

        Returns:
          String representation of the text data.
        """
        return data.text


class TextvsRNAKernelSignature(KernelSignatureBase):
    """Text vs RNA kernel signature."""

    x: str = dspy.InputField(
        description="Text description X of biological process or phenotype"
    )
    y: str = dspy.InputField(
        description="RNA-seq transcriptome profile Y: "
        "[(HGNC_symbol, log2_fold_change, FDR), ...]"
    )

    @staticmethod
    def format_data(
        data: data_models.Text | data_models.RNASeq,
    ) -> str | list[tuple[str, float, float]]:
        """Convert data model to format expected by this kernel."""
        if isinstance(data, data_models.Text):
            return data.text
        elif isinstance(data, data_models.RNASeq):
            return str(list(zip(data.gene_names, data.lfc, data.fdr)))
