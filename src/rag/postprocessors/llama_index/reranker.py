"""RankGPT reranking (top-8) with the pipeline LLM."""

from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core import Settings


def rerank_rankgpt_top8():
    llm = Settings.llm

    if hasattr(llm, 'temperature'):
        if llm.temperature != 0.0:
            print(f"Using temperature=0.0 copy of LLM for RankGPT reranker "
                  f"(generation LLM keeps temperature={llm.temperature})")
            llm = llm.model_copy(update={"temperature": 0.0})
    else:
        print("LLM does not have a temperature attribute; cannot enforce determinism for RankGPT reranker")

    return RankGPTRerank(top_n=8, llm=llm)
