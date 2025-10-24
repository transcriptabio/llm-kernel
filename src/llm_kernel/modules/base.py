"""Base DSPy modules for pairwise kernel comparisons.

Provides foundational DSPy modules for building kernel scoring pipelines.
Handles orchestration of LLM calls for pairwise analysis.

Typical usage example:

  module = PairwiseKernelModule(RNAvsRNAKernelSignature)
  result = module.forward(x='RNA profile 1', y='RNA profile 2')
"""

import dspy


class PairwiseKernelModule(dspy.Module):
    """Base module for pairwise kernel scoring.

    A thin wrapper around dspy.Predict that handles any two-input signature.
    The signature must have a similarity_score output field.
    """

    def __init__(self, signature_class: type[dspy.Signature]):
        """Initialize with a dspy.Signature class.

        Args:
          signature_class: The signature class that defines the input/output schema.
        """
        super().__init__()
        self.kernel = dspy.Predict(signature_class)

    def forward(self, **kwargs):
        """Forward pass for kernel scoring.

        Args:
          **kwargs: Input fields that match the signature's input fields.

        Returns:
          The signature's output, which must include similarity_score.
        """
        return self.kernel(**kwargs)


class MultistepPairwiseKernelModule(dspy.Module):
    """Multistep pairwise kernel module for pairwise comparisons."""

    def __init__(
        self,
        x_analysis_signature: type[dspy.Signature],
        y_analysis_signature: type[dspy.Signature],
        pairwise_analysis_signature: type[dspy.Signature],
    ):
        """Initialize with a dspy.Signature class.

        Args:
          x_analysis_signature: The signature class that defines the input/output schema for the x analysis.
          y_analysis_signature: The signature class that defines the input/output schema for the y analysis.
          pairwise_analysis_signature: The signature class that defines the input/output schema for the comparative analysis.
        """
        super().__init__()

        class SimilarityScoreSignature(dspy.Signature):
            pairwise_analysis: str = dspy.InputField(
                description="Comparative analysis of object X and object Y"
            )
            similarity_score: float = dspy.OutputField(
                description="Similarity score between -1.0 and 1.0"
            )

        self.x_analysis = dspy.Predict(x_analysis_signature)
        self.y_analysis = dspy.Predict(y_analysis_signature)
        self.pairwise_analysis = dspy.Predict(pairwise_analysis_signature)
        self.similarity_score = dspy.Predict(SimilarityScoreSignature)

    def forward(self, x, y):
        x_analysis = self.x_analysis(x)
        y_analysis = self.y_analysis(y)
        pairwise_analysis = self.pairwise_analysis(x_analysis, y_analysis)
        similarity_score = self.similarity_score(pairwise_analysis)
        return similarity_score
