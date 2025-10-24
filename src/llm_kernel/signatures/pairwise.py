"""Analysis signatures for pairwise comparisons between data objects.

Defines DSPy signatures that compare two data objects and identify
similarities, differences, and relationships. Used as components in kernel
signatures for pairwise similarity scoring.

Typical usage example:

  class BasicAnalysis(base.Analysis):
    similarities: list[str] = dspy.OutputField(description='Key similarities')
    differences: list[str] = dspy.OutputField(description='Key differences')
    relationship_summary: str = dspy.OutputField(description='Summary of the relationship between the objects')
"""

import dspy


class Analysis(dspy.Signature):
    """Comparative analysis between two objects."""

    similarities: list[str] = dspy.OutputField(
        description="Key similarities identified between the objects"
    )
    differences: list[str] = dspy.OutputField(
        description="Key differences identified between the objects"
    )
    relationship_summary: str = dspy.OutputField(
        description="Summary of the relationship between the objects"
    )


class FunctionalAnalysis(Analysis):
    """Functional comparative analysis between two objects."""

    shared_pathways: list[str] = dspy.OutputField(
        description="Pathways shared between the objects"
    )
    functional_similarities: list[str] = dspy.OutputField(
        description="Key functional similarities between the objects"
    )
    functional_differences: list[str] = dspy.OutputField(
        description="Key functional differences between the objects"
    )
    functional_comparison: str = dspy.OutputField(
        description="Summary of functional comparison"
    )


class SemanticAnalysis(Analysis):
    """Semantic comparative analysis between two objects."""

    conceptual_similarities: list[str] = dspy.OutputField(
        description="Key conceptual similarities between the objects"
    )
    conceptual_differences: list[str] = dspy.OutputField(
        description="Key conceptual differences between the objects"
    )
    conceptual_comparison: str = dspy.OutputField(
        description="Summary of conceptual comparison"
    )
