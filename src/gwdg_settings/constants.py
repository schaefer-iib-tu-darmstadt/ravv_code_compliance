"""GWDG connection defaults (adapted from
https://gitlab-ce.gwdg.de/hpc-team-public/chat-ai-llamaindex-examples)."""
import os

from dotenv import load_dotenv

load_dotenv()

GWDG_API_KEY = os.getenv("GWDG_API_KEY")
LLM_API_ENDPOINT_KEY = os.getenv("GWDG_API_BASE", "https://chat-ai.academiccloud.de/v1/")
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
EMBEDDING_MODEL_NAME = "e5-mistral-7b-instruct"
