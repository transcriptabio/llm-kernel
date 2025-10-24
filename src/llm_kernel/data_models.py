"""Pydantic data models for type-safe data representation.

Defines core data structures for the kernel scoring pipeline with automatic
validation and type safety. Supports RNA-seq profiles and text descriptions.

Typical usage example:

  rna_data = RNASeq(
      id='sample1',
      gene_names=['GENE1', 'GENE2'],
      lfc=[1.0, -0.5],
      fdr=[0.01, 0.05]
  )
  text_data = Text(id='description1', text='Cell proliferation response')
"""

import pydantic


class RNASeq(pydantic.BaseModel):
    """RNA-seq transcriptome signature."""

    id: str = pydantic.Field(description="ID of the sample")
    gene_names: list[str] = pydantic.Field(description="List of gene names")
    lfc: list[float] = pydantic.Field(description="List of log2 fold changes")
    fdr: list[float] = pydantic.Field(
        description="List of FDR-corrected p-values"
    )

    @pydantic.model_validator(mode="after")
    def validate_list_lengths(self) -> "RNASeq":
        """Validate that all lists have the same length."""
        if not (len(self.gene_names) == len(self.lfc) == len(self.fdr)):
            raise ValueError("All input lists must have the same length.")
        return self


class RNASeqPair(pydantic.BaseModel):
    """RNA-seq transcriptome pair."""

    x: RNASeq = pydantic.Field(description="RNA-seq transcriptome profile X")
    y: RNASeq = pydantic.Field(description="RNA-seq transcriptome profile Y")
    similarity_score: float = pydantic.Field(
        description="Similarity score", ge=-1, le=1
    )


class Text(pydantic.BaseModel):
    """Text-based signature."""

    id: str = pydantic.Field(description="ID of the sample")
    text: str = pydantic.Field(description="Text content")


class TextPair(pydantic.BaseModel):
    """Text-based pair."""

    x: Text = pydantic.Field(description="Text content X")
    y: Text = pydantic.Field(description="Text content Y")
    similarity_score: float = pydantic.Field(
        description="Similarity score", ge=-1, le=1
    )
