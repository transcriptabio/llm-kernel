"""Base signature classes for kernel analysis."""

import dspy


class Analysis(dspy.Signature):
    """Base class for analysis Signatures.

    This provides a common interface for all analysis Signatures.
    """
