# Method-Based Retrieval-Augmented Generation for Mathematical Reasoning

## Project Overview

This research project investigates whether **generalized problem-solving methods** extracted from worked examples can enhance the mathematical reasoning capabilities of small language models through Retrieval-Augmented Generation (RAG). Unlike traditional RAG approaches that retrieve similar examples or factual knowledge, this work explores retrieving abstract reasoning patterns and strategies.


## Research Objectives


The core research question: **Can externalized, generalized solution methods improve mathematical problem-solving performance when retrieved during inference for small language models?**


We evaluate this hypothesis through:

- **5 distinct prompt engineering strategies** targeting different cognitive dimensions

- **Semantic similarity-based retrieval** using FAISS vector search

- **Comprehensive evaluation** on 1,319 GSM8K test problems

- **Stratified analysis** comparing performance with and without retrieved methods


## Experimental Design


### Dataset: GSM8K (Grade School Math 8K)


- **Training Set**: 7,473 grade-school level math word problems

- **Selected Samples**: 200 problems chosen via stratified sampling for diversity

- **Test Set**: 1,319 problems for evaluation

- **Variant**: Socratic version with detailed step-by-step explanations

 
### Method Extraction Pipeline


**Step 1: Stratified Sampling**

- Selected 200 representative problems from 7,473 training examples

- Ensured diversity across:

  - **Complexity levels**: Simple (2-3 steps) to Complex (8+ steps)

  - **Mathematical concepts**: Time, money, fractions, percentages, geometry, etc.

  - **Problem schemas**: Repeated groups, sequential operations, aggregation, comparison



**Step 2: Multi-Dimensional Prompt Engineering**
 

Five prompt strategies extract different aspects of problem-solving knowledge:



| Prompt Type | Target | Description |

|-------------|--------|-------------|

| **Step-by-Step** | Procedural Knowledge | Ordered algorithmic sequence without specific numbers |

| **Conceptual** | Mathematical Principles | Underlying concepts and "why" the solution works |

| **Pattern-Based** | Structural Templates | Problem structure and schema recognition |

| **Error-Aware** | Common Pitfalls | Correct approaches + typical mistakes to avoid |

| **Metacognitive** | Strategic Thinking | High-level planning and reasoning strategies |

 

**Step 3: Method Generation**

- Used **Turkish-Gemma-9b-T1 (Cosmos-T1)** reasoning model

- Generated 1,000 total methods (200 problems × 5 prompt types)

- Methods abstracted from specific numerical values to enable transfer


**Step 4: Semantic Indexing**

- Embedded methods using **sentence-transformers/all-MiniLM-L6-v2**

- Created FAISS indices for efficient similarity search

- 384-dimensional vector space representation


### Retrieval Configuration


- **Top-K**: 3 methods per question

- **Similarity Threshold**: 40% (cosine similarity)

  - Balances coverage (70-80%) with quality

  - Filters extremely poor matches while maintaining sufficient retrieval

- **Coverage Analysis**:

  - 60% threshold → <5% coverage (too restrictive)

  - 50% threshold → 25-30% coverage (limited)

  - 45% threshold → ~50% coverage (moderate)

  - 40% threshold → 70-80% coverage (selected)

 
### Evaluation Framework
 

**Model**: Gemma-3-1B-it


**Conditions**:

1. Baseline (no retrieval)

2. Step-by-Step RAG

3. Conceptual RAG

4. Pattern-Based RAG

5. Error-Aware RAG

6. Metacognitive RAG
 

**Metrics**:

- Overall accuracy across all 1,319 test questions

- Stratified accuracy: with-methods vs. without-methods

- Retrieval statistics: usage rate, average similarity, methods per question


## Key Findings

### Overall Performance
 

| Condition | Accuracy | Δ from Baseline |

|-----------|----------|-----------------|

| Baseline | 48.22% | --- |

| Conceptual | 48.37% | +0.15% |

| Pattern-Based | 46.55% | -1.67% |

| Error-Aware | 45.56% | -2.66% |

| Step-by-Step | 44.81% | -3.41% |

| Metacognitive | 44.66% | -3.56% |

 

### Critical Discovery: The Without-Methods Paradox
 

**Stratified Performance Reveals Unexpected Pattern**:


Questions **without** retrieved methods (failed 40% threshold) consistently outperform those **with** methods:

- Without-methods accuracy: **50-52%**

- With-methods accuracy: **42-48%**

- Both exceed baseline: **48.22%**

 

**Interpretation**:

1. **Method Quality**: Cosmos-T1 generated methods may not optimally serve small models

2. **Question Characteristics**: Filtered questions may be structurally simpler or better suited to model's internal reasoning

3. **Interference Effect**: Abstract guidance can confuse 1B models even when semantically relevant

 

### Threshold-Coverage Trade-off


**Empirical evidence that RAG infrastructure is beneficial**:

- Higher thresholds (50%, 60%) → Lower coverage → **Worse overall accuracy**

- Despite potentially "better quality" methods, system performs worse when fewer questions receive guidance

- 40% threshold achieves optimal balance: sufficient coverage with quality filtering

 

### Prompt Strategy Insights
 

- **Pattern-Based**: Minimal interference (+0.39% with vs. without methods)

  - Structural guidance compatible with small model reasoning

- **Conceptual**: High coverage (81.8%) but moderate interference (-3.52%)

  - Abstract principles may exceed model capacity

- **Step-by-Step & Metacognitive**: Largest negative impact (-8.90%, -7.82%)

  - Procedural and strategic instructions confuse small models

 
