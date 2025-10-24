# LLM Kernel

A framework for LLM-based pairwise similarity scoring across different data types.

📄 **NeurIPS AI for Science paper:** [*LLM Kernel: a framework for verifiable evaluation of scientific data interpretations*](https://openreview.net/forum?id=Rq5wDa8Obf)

## Table of Contents

- [Module Overview](#module-overview)
- [Data Flow](#data-flow)
- [Files](#files)
- [Usage](#usage)
- [Install](#install)
- [Current limitations](#current-limitations)
- [Citation](#citation)

## Module Overview

This is a simple framework built on [`DSPy`](http://dspy.ai/) that captures the core concepts
of __LLM Kernel__.

- **Signatures** define analysis components for individual data objects or pairwise comparisons
- **Kernels** compose signatures into complete workflows and define input data structure
- **Scorer** executes kernels with any data type through the defined input structure

### Structure

- **Data Layer** (`data_models.py`) - Pydantic models for type-safe data representation
- **Analysis Layer** (`signatures/`) - DSPy signatures defining comparison workflows and analysis components
- **Execution Layer** (`scorer.py`) - LLM orchestration with parallel processing and error handling  
- **Support Layer** (`utils.py`, `examples.py`) - Data preprocessing and usage patterns

## Data Flow

```markdown
Input Data → Data Models → Signature Formatting → LLM Analysis → Similarity Score
     ↓              ↓              ↓                    ↓              ↓
  Raw Data    Type Validation   String Format    Structured Output   Float Score
```

## Files

- `data_models.py` - Pydantic models for type-safe data representations (currently RNA-seq and text data)
- `scorer.py` - Core LLM-based pairwise similarity scoring engine with parallel processing
- `signatures/kernels.py` - DSPy signatures defining pairwise comparison workflows for different data types
- `signatures/individual.py` - Analysis components for single object evaluation (functional, semantic, pathway analysis)
- `signatures/pairwise.py` - Analysis components for comparating similarities and differences between objects
- `examples.py` - Usage examples for building kernels with different analysis components
- `utils.py` - AnnData to kernel signature conversion and data preprocessing utilities

## Usage

Read and run examples via `python examples/example.py`. The script relies on the Gemini API so please configure to your use case. The DSPy framework relies on `LiteLLM` so you can swap out common model API providers, see more [in the DSPy docs](https://dspy.ai/learn/programming/language_models/).

```python
from analysis.kernel import data_models, scorer
from analysis.kernel.signatures import kernels

# Create scorer
scorer = scorer.KernelScorer(
    client=client,
    signature_class=kernels.RNAvsRNAKernelSignature
)

# Create data models
rna_x = data_models.RNASeq(id='sample1', gene_names=['GENE1'], lfc=[1.0], fdr=[0.01])
rna_y = data_models.RNASeq(id='sample2', gene_names=['GENE1'], lfc=[0.8], fdr=[0.02])

# Score similarity
score, response = scorer.score_pair(rna_x, rna_y)
```

## Install

```bash
git clone https://github.com/transcriptabio/llm-kernel
cd llm-kernel
make install
```

### Run Tests

```bash
make dev
make test
```

## Current limitations

### 1. Opinionated comparison model

The `KernelSignatureBase` enforces a specific reasoning pattern:

1. **Independent analysis** of object X (`x_analysis`)
2. **Independent analysis** of object Y (`y_analysis`)
3. **Pairwise analysis** comparing X and Y (`pairwise_analysis`)

This is a "limited" path of reasoning / comparison, but I think is sensible and typically has better results than a "freeform" method.

### 2. Single LLM call architecture

The library is built around a `KernelSignatureBase` DSPy module with a single LLM call, which means:

- All analysis patterns in `signatures/` must be composed into hierarchical `Signatures` at input
- Individual and pairwise analysis signatures solely define `OutputField`'s
- Future multi-step programs (e.g., `MultistepPairwiseKernelModule`) would require redesigning the analysis signatures to include proper `InputField` definitions for composition

## Citation

```bibtex
@misc{llm-kernel2025,
  title={LLM Kernel: a framework for verifiable evaluation of scientific data interpretations},
  author={Connell, William and Guin, Drishti and Mellina, Clayton},
  year={2025},
  howpublished={NeurIPS AI for Science Workshop},
  url={https://openreview.net/forum?id=Rq5wDa8Obf},
  note={Software: \url{https://github.com/transcriptabio/llm-kernel}}
}
```
