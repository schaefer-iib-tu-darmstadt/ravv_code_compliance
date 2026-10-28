"""Routing accuracy of the orchestrator prompt, scored against each question's DesignQA subtask.

Reports overall and macro (per-subtask) accuracy; questions whose API call fails after
retries are excluded. Run: ``uv run python -m src.orchestration.orchestration_tests``.
"""

import re
import time
from pathlib import Path

import pandas as pd

from eval.metrics.metrics import character_string_no_space
from src.utils.llm_api.client import create_gwdg_client
from src.utils.llm_api.chat_completions import llm_chat_completion_message_content
from src.utils.logging_helpers import set_logging_level

ROUTINGSYSTEMPROMPT = """You are a routing agent. Classify each user question into one of the following *mutually exclusive* categories.

Before selecting a category, reason briefly:
- What information sources are required? (single vs. multiple)
- Is any image provided or referenced?
Then output only the final line: Category: … --- Query: …

1. Vector-RAG
   - The question asks for a *single, specific piece of information* (lookup, definition, direct quote).
   - Only use this if **exactly one rule, paragraph, or text fragment** must be retrieved.
   - Input is **text only**.
   - Respond with:
     Category: Vector-RAG --- Query: [short paraphrase of what to retrieve]

2. Graph-RAG
   - The question requires *collecting, comparing, or aggregating* multiple rules, sections, or facts.
   - The query involves *lists, overviews, or summaries* of several items.
   - Input is **text only**.
   - Respond with:
     Category: Graph-RAG --- Query: [short paraphrase of the multi-step information request]

3. Retrieval-to-Vision
   - Use only when the question explicitly asks to verify *compliance* with a rule, instruction, or numerical requirement based on image content.
   - Do NOT use this category for simple visual identification, detection, or naming tasks.
   - Respond with:
     Category: Retrieval-to-Vision --- Query: [Query]

4. Other
   - Use for image-related tasks that do not involve rule or instruction compliance, such as identifying or naming components, confirming visual presence, or counting elements.
"""

category_map = {
    "vector-rag": {"retrieval"},
    "graph-rag": {"compilation"},
    "retrieval-to-vision": {"functional", "dimension"},
    "other": {"definition", "presence"},
}

MODEL_LIST = ["openai-gpt-oss-120b"]
N = None
FULL_SET = True
MAX_RETRIES = 3


def route_question(client, model_name: str, question: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm_chat_completion_message_content(
                client,
                model=model_name,
                messages=[
                    {"role": "system", "content": ROUTINGSYSTEMPROMPT},
                    {"role": "user", "content": question},
                ],
            )
        except Exception as exc:
            print(f"[API-Error] routing call failed for model {model_name} "
                  f"(attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
    return None


def main():
    set_logging_level("INFO")
    client = create_gwdg_client()

    csv_name = "routing_questions_full.csv" if FULL_SET else "routing_questions.csv"
    df = pd.read_csv(f"data/processed/{csv_name}")
    rows = df if N is None else df.head(N)

    output_dir = Path("results/agentic_orchestration") / ("full_set" if FULL_SET else "pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_LIST:
        results = []

        for row in rows.itertuples(index=False):
            question = row.question
            ground_truth = row.task_type

            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")

            response = route_question(client, model_name, question)
            print(f"Response: {response}")

            match = re.search(r"Category:\s*(.*?)\s*---", response, flags=re.IGNORECASE) if response else None
            raw_category = match.group(1).strip() if match else None
            normalized_category = character_string_no_space(raw_category.lower()) if raw_category else None

            expected_tasks = category_map.get(normalized_category, set())
            is_correct = ground_truth in expected_tasks

            results.append(
                {
                    "question": question,
                    "image": getattr(row, "image", None),
                    "ground_truth": ground_truth,
                    "response": response,
                    "predicted_category_raw": raw_category,
                    "predicted_category_normalized": normalized_category,
                    "expected_tasks": ", ".join(sorted(expected_tasks)) if expected_tasks else "",
                    "is_correct": is_correct,
                    "api_failed": response is None,
                }
            )

            print(f"Predicted Category: {raw_category or 'N/A'}")
            print(f"Match: {'yes' if is_correct else 'no'}")

        if not results:
            continue

        results_df = pd.DataFrame(results)
        n_failed = int(results_df["api_failed"].sum())
        scored_df = results_df[~results_df["api_failed"]]
        total = len(scored_df)
        correct = scored_df["is_correct"].sum()
        overall_accuracy = correct / total if total else 0.0

        per_task = (
            scored_df.groupby("ground_truth")["is_correct"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "correct", "count": "total"})
        )
        per_task["accuracy"] = per_task["correct"] / per_task["total"]
        macro_accuracy = per_task["accuracy"].mean() if not per_task.empty else 0.0

        csv_path = output_dir / f"{model_name}.csv"
        txt_path = output_dir / f"{model_name}.txt"

        results_df.to_csv(csv_path, index=False)

        summary_lines = [
            f"Model: {model_name}",
            f"Total: {correct}/{total} ({overall_accuracy:.1%})",
            f"Macro avg (across tasks): {macro_accuracy:.1%}",
            f"API failures (excluded from accuracy): {n_failed}",
            "",
            "Results per task:",
        ]
        summary_lines.extend(
            f"- {task}: {row.correct}/{row.total} ({row.accuracy:.1%})"
            for task, row in per_task.iterrows()
        )

        txt_path.write_text("\n".join(summary_lines), encoding="utf-8")

        print(f"Accuracy: {correct}/{total} ({overall_accuracy:.0%})")
        print(f"Results saved to {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
