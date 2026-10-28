"""Prompt templates of the evaluated pipelines."""

from llama_index.core import PromptTemplate

QA_PROMPT = PromptTemplate(
    "You are a specialized assistant for the Formula SAE student engineering team.\n"
    "Your task is to answer questions about the FSAE rules with high precision.\n"
    "\n"
    "INSTRUCTIONS:\n"
    "- Use ONLY the context below to answer\n"
    "- Do NOT use prior knowledge\n"
    "- If context lacks the answer, state: 'The provided context does not contain the answer to this question.'\n"
    "- Output ONLY the exact rule text\n"
    "- Do NOT add introductions, explanations, or phrases like 'Rule X states:'\n"
    "\n"
    "---------------------\n"
    "CONTEXT:\n{context_str}\n"
    "---------------------\n"
    "QUESTION: {query_str}\n"
    "ANSWER:"
)

GRAPH_COMPILATION_PROMPT = PromptTemplate(
    "You are a specialized assistant for the Formula SAE student engineering team.\n"
    "Your task is to extract rule numbers from a knowledge graph structure.\n"
    "\n"
    "The context below contains:\n"
    "- RULE nodes with IDs (e.g., 'T.7.1', 'EV.4.5')\n"
    "- TOPIC nodes (e.g., 'Aerodynamics', 'Brake System')\n"
    "- Relations:\n"
    "  * (RULE)-[CONTAINS_TOPIC]->(TOPIC): Rules that discuss a topic\n"
    "  * (RULE)-[HAS_SUBRULE]->(RULE): Parent-child rule hierarchy\n"
    "  * (RULE)-[CITES]->(RULE): Cross-references between rules\n"
    "\n"
    "TASK:\n"
    "Extract ALL unique rule IDs that are:\n"
    "1. Connected to the topic term(s) from the question via CONTAINS_TOPIC relation\n"
    "2. Connected via HAS_SUBRULE (include parent rules AND all their subrules)\n"
    "3. Connected via CITES (include all rules that are cited)\n"
    "\n"
    "CRITICAL REQUIREMENTS:\n"
    "- Output format: Comma-separated list with exactly ONE space after each comma\n"
    "- Example: T.7.1, T.7.1.1, T.7.2, V.1.4\n"
    "- Include ONLY the rule IDs from the context - no explanations\n"
    "- Include parent rules even if only subrules are in context (e.g., if T.7.1.1 is found, include T.7.1)\n"
    "- Do NOT include letter suffixes (e.g., T.7.1.2a → just T.7.1.2)\n"
    "- Sort numerically by prefix (A, B, T, V, EV, IC, etc.) then by number\n"
    "- If no rules found, output: 'No rules found'\n"
    "\n"
    "---------------------\n"
    "CONTEXT:\n{context_str}\n"
    "---------------------\n"
    "QUESTION: {query_str}\n"
    "RULE IDs:"
)


VISION_STAGE2_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on provided rule text "
    "and engineering drawings."
)
