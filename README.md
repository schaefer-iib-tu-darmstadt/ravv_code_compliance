# RAVV Code Compliance

**From False Compliance to Conservative Rejection: Orchestrated Retrieval-Augmented Generation for Multimodal Engineering Compliance Checking**

This repository contains the code and data to reproduce the experiments of the paper on the [DesignQA](https://github.com/anniedoris/design_qa) benchmark.
It implements in-context retrieval (ICR), basic and advanced vector RAG, LLM-extracted and structure-aware graph RAG,
retrieval-augmented visual verification (RAVV), and the router that assigns each question to one of these components.

### Setup

First, clone this repository and install the locked environment with [uv](https://docs.astral.sh/uv/) (Python 3.13):

```
git clone https://github.com/schaefer-iib-tu-darmstadt/ravv_code_compliance.git && cd ravv_code_compliance
uv sync
```

Next, add the DesignQA benchmark questions, images and metrics, which are not redistributed here:

```
git clone https://github.com/anniedoris/design_qa.git
cp -r design_qa/dataset design_qa/eval .
```

Finally, copy `.env.example` to `.env` and fill in the keys of the model providers you want to use (GWDG/SAIA, OpenAI, OpenRouter) and of a Qdrant instance.
Setting `QDRANT_LOCAL_PATH` uses an embedded local Qdrant instead.

### Running the Evaluation

To evaluate a pipeline on the DesignQA subtasks, run:

```
uv run python -m main
```

Pipelines and subtasks are selected at the top of `main.py`, with `SELECTED_PIPELINES` taking names from `src/strategies.py` and `BENCHMARKS_TO_RUN` taking `retrieval`, `compilation`, `definition`, `presence`, `dimension`, `functional`, `dimension_synthetic` or `functional_synthetic`.
`TEST_RUN = False` runs the full question sets, and `N_RUNS` sets the number of independent runs of the compliance subtasks.
Each run writes `results/{category}/{benchmark}/{pipeline}/{run_id}/results.csv` with the columns `ground_truth` and `model_prediction`, which `eval/full_evaluation.py` from DesignQA scores directly.
Vector indices are built from the processed rulebook on first use.

The structure-aware graph needs no model call and reproduces its rule compilation score offline:

```
uv run python scripts/run_compilation_eval.py structured_graph_rag
```

## Contents

* **Preprocessing:** the LlamaParse Markdown of the 2024 FSAE rulebook, the 200-question retrieval subsample and the routing questions (`data/processed/`), and the hierarchical parser (`src/rag/parser/`)
* **Graph construction:** the structure-aware clause graph (`src/rag/graph/structured_graph.py`) and the LLM-extracted graphs (`src/pipelines/graph_rag.py`), with the extracted GPT-5-mini graphs in `data/graphs/`
* **Prompts and routing templates:** `src/prompts.py`, the vision prompts in `src/pipelines/`, and the routing prompt in `src/orchestration/orchestration_tests.py`
* **Configurations:** all pipeline configurations in `src/strategies.py`, endpoint settings in `config/ai_config.yaml`, and locked package versions in `uv.lock`
* **Evaluation and analysis:** `main.py`, `src/evaluation/` and `scripts/`
* **Synthetic violations:** question and label files in `dataset_synthetic/`; the images are rebuilt from the DesignQA images by the two `build_synthetic_*` scripts, which draw the labels in the Windows Arial font

## Reproducing the Paper

| Experiment | Configurations | Scripts |
|---|---|---|
| Rule retrieval | `*_no_rag`, `*_basic_rag_small_chunks[_md]`, `*_advanced_hierarchical`, `gpt-oss_ablation_*` | `compute_retrieval_metrics.py`, `bootstrap_retrieval_ci.py`, `analyze_basic_rag_failure.py`, `analyze_abstention.py` |
| Rule compilation | `structured_graph_rag*`, `gpt-5-mini_simple_graph_rag_mem`, `gpt-5-mini_schema_graph_rag` | `run_compilation_eval.py`, `bootstrap_compilation_ci.py`, `analyze_structured_graph_errors.py`, `analyze_graph_construction_coverage.py`, `run_structure_perturbation.py` |
| Rule compliance | `gemma_no_rag_vision`, `*2gemma_retrieval2vision`, `*2qwen_retrieval2vision` | `compute_compliance_scores.py`, `bootstrap_compliance_ci.py`, `bootstrap_ci.py`, `analyze_ravv_confusion.py`, `analyze_three_way_decision.py`, `analyze_violation_strata.py`, `analyze_hallucination_examples.py` |
| Synthetic violations | `or_gemma3_no_rag_vision`, `or_gpt2gemma3_retrieval2vision` | `build_synthetic_violations.py`, `build_synthetic_functional.py`, `run_synthetic_violations.py`, `analyze_synthetic_gemma3_openrouter.py` |
| Routing | | `build_routing_full_dataset.py`, `uv run python -m src.orchestration.orchestration_tests`, `analyze_routing_error_propagation.py` |
| Latency and tokens | `gpt-oss_*` text configurations, `structured_graph_rag`, `or_*` vision configurations | `measure_latency_cost.py`, `analyze_latency_cost.py` |

The analysis scripts run offline on the result CSVs of the evaluation runs, which are not part of this repository.
Each script names its input files at the top, either a run directory under `results/` or a renamed copy under `results_for_publication/`.

## Reproducibility

Open-weight models run at temperature 0.0, and the direct chat-completion calls in the vision pipelines and the router fix seed 10.
The 200-question retrieval subsample and the bootstrap intervals (10,000 resamples) use seed 42, the cluster bootstrap of the synthetic violations uses seed 7, and the structure perturbation runs over seeds 0 to 4.
Hosted endpoints change over time, and OpenRouter calls are pinned to one hosting provider per model (`src/utils/llm_api/client.py`), so LLM outputs may not reproduce exactly.
The structure-aware graph and all offline analyses are unaffected.
The reported runs of the LLM-extracted graphs built each graph and queried it in one process, with vector retrieval over the embedded graph nodes; loading a graph from `data/graphs/` restores its triplets but not these node embeddings.

## Citation

The citation will be added on publication.
