"""Analysis signatures for individual analysis of objects.

Defines DSPy signatures that analyze single data objects independently.
Used as components in pairwise comparisons.

Typical usage example:

  class BasicAnalysis(base.Analysis):
    summary: str = dspy.OutputField(description='Summary analysis of the object')
    key_findings: list[str] = dspy.OutputField(
        description='Key findings from the analysis')
"""

import dspy


class Analysis(dspy.Signature):
    """Analysis of a single data object."""

    summary: str = dspy.OutputField(
        description="Summary analysis of the object"
    )
    key_findings: list[str] = dspy.OutputField(
        description="Key findings from the analysis"
    )


class FunctionalAnalysis(Analysis):
    """Causal reasoning about the mechanisms driving a bulk RNA-seq
    transcriptome profile."""

    primary_targets: list[str] = dspy.OutputField(
        description="Hypothesized direct molecular targets of the perturbation"
        "(e.g., kinases, receptors, TFs)."
    )
    secondary_targets: list[str] = dspy.OutputField(
        description="Possible additional targets due to polypharmacology or indirect effects."
    )
    upstream_regulators: list[str] = dspy.OutputField(
        description="Regulatory genes or factors inferred "
        "to mediate the response."
    )
    affected_pathways: list[str] = dspy.OutputField(
        description="Biological pathways, modules, or programs "
        "activated or repressed."
    )
    functional_themes: list[str] = dspy.OutputField(
        description="High-level biological programs "
        "(e.g., cell cycle arrest, stress response, immune activation)."
    )
    causal_reasoning: str = dspy.OutputField(
        description="Narrative explanation of the inferred "
        "mechanism and biological impact."
    )


class SemanticAnalysis(Analysis):
    """Semantic analysis of text descriptions."""

    key_concepts: list[str] = dspy.OutputField(
        description="Key biological concepts and processes involved"
    )
    biological_domains: list[str] = dspy.OutputField(
        description="Biological domains or systems involved"
    )
    molecular_entities: list[str] = dspy.OutputField(
        description="Specific molecular entities (genes, proteins, pathways) involved"
    )
    phenotypic_effects: list[str] = dspy.OutputField(
        description="Phenotypic or functional effects described"
    )


class PathwayAnalysis(Analysis):
    """Detailed pathway analysis for individual objects."""

    enriched_pathways: list[str] = dspy.OutputField(
        description="Significantly enriched biological pathways"
    )
    pathway_categories: list[str] = dspy.OutputField(
        description="Categories of pathways (metabolic, signaling, etc.)"
    )
    pathway_interactions: list[str] = dspy.OutputField(
        description="Interactions between pathways"
    )
    functional_enrichment: str = dspy.OutputField(
        description="Overall functional enrichment assessment"
    )
