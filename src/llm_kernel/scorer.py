"""Core scoring engine for LLM-based pairwise similarity analysis.

Provides the main KernelScorer class that orchestrates LLM calls for pairwise
similarity scoring. Handles data formatting, parallel processing, and response
parsing. Works with any DSPy signature class.

Typical usage example:

  config = KernelConfig(model_name='gemini/gemini-2.5-pro', max_workers=5)
  scorer = KernelScorer(client=client, signature_class=RNAvsRNAKernelSignature, config=config)
  score, response = scorer.score_pair(rna_x, rna_y)
"""

import typing
from concurrent import futures

import dspy
import pydantic
from absl import logging
from google import genai
from tqdm import tqdm

from llm_kernel.modules import base


class KernelConfig(pydantic.BaseModel):
    """Configuration for kernel scoring operations."""

    model_name: str = "gemini/gemini-2.5-flash"
    thinking_budget: int = 0
    max_workers: int = 5


class KernelScorer:
    """Kernel-based similarity scorer for pairwise comparisons."""

    def __init__(
        self,
        client: genai.Client,
        signature_class: type[dspy.Signature],
        config: KernelConfig | None = None,
    ):
        """Initialize the kernel scorer.

        Args:
          client: Google AI client for LLM calls.
          signature_class: dspy.Signature class for the kernel.
          config: Optional configuration for model settings.
        """
        self.client = client
        self.config = config or KernelConfig()

        # Configure dspy with the client
        lm = dspy.LM(
            self.config.model_name,
            max_tokens=self.config.thinking_budget,
            cache=False,
        )
        dspy.configure(
            lm=lm, adapter=dspy.JSONAdapter(), cache=False, track_usage=True
        )

        # Initialize kernel module with the signature
        self.kernel_module = base.PairwiseKernelModule(signature_class)
        self.signature_class = signature_class
        logging.info(
            "Using signature-based kernel module: %s", signature_class.__name__
        )

    def score_pair(self, x, y) -> tuple[float, dspy.Prediction]:
        """Score similarity between two pre-formatted objects.

        Args:
          x: Data model object for first object '
          '(use kernel signature\'s format_data method).
          y: Data model object for second object '
          '(use kernel signature\'s format_data method).

        Returns:
          Tuple of (similarity_score, full_response).
        """
        try:
            # format data
            x = self.signature_class.format_data(x)
            y = self.signature_class.format_data(y)
            # Use dspy module to get response with pre-formatted inputs
            response = self.kernel_module(x=x, y=y)
            # Extract similarity score
            score = float(response.similarity_score)
            return score, response

        except Exception as e:
            x_name = getattr(x, "sample_name", str(x)[:50])
            y_name = getattr(y, "sample_name", str(y)[:50])
            logging.error(
                "Scoring failed for %s vs %s: %s", x_name, y_name, str(e)
            )
            return float("nan"), f"Failed: {e!s}"

    def score_pairs(self, pairs: list[tuple]) -> typing.Iterator[dict]:
        """Score a list of object pairs, yielding results as they complete.

        Args:
          pairs: List of (x, y) tuples to compare.
        """

        def _work(pair):
            x, y = pair
            score, response = self.score_pair(x, y)
            x_name = getattr(x, "sample_name", str(x)[:50])
            y_name = getattr(y, "sample_name", str(y)[:50])
            return {
                "x": x_name,
                "y": y_name,
                "similarity_score": score,
                "response": response,
            }

        with futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as ex:
            futs = {ex.submit(_work, p): p for p in pairs}

            with tqdm(
                total=len(pairs), desc="Scoring pairs", unit="pair"
            ) as pbar:
                for fut in futures.as_completed(futs):
                    try:
                        yield fut.result()
                    except Exception as e:
                        x, y = futs[fut]
                        x_name = getattr(x, "sample_name", str(x)[:50])
                        y_name = getattr(y, "sample_name", str(y)[:50])
                        logging.error(
                            "Error processing %s vs %s: %s",
                            x_name,
                            y_name,
                            str(e),
                        )
                        yield {
                            "x": x_name,
                            "y": y_name,
                            "similarity_score": float("nan"),
                            "response": f"Error: {e!s}",
                        }
                    finally:
                        pbar.update(1)
