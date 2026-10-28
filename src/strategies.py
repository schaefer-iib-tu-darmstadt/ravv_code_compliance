"""Named configurations of all evaluated pipelines; "type" selects the class in src/pipeline_factory.py.

Model names are the provider's model ids (GWDG by default, "openai"/"openrouter" via llm_provider).
"""

from src.rag.parser.llama_parser.markdown_parser import create_two_stage_h1_chapter_parser
from src.rag.postprocessors.llama_index.reranker import rerank_rankgpt_top8

PATHS = {
    "pdf_rules": "dataset/docs/FSAE_Rules_2024_V1.pdf",
    "txt_rules": "dataset/docs/rules_pdfplumber1.txt",
    "md_rules": "data/processed/FSAE_Rules_2024_V1_llama_agentic_plus.md",
}

PARSER_CONFIGS = {
    "small_chunks": {
        "class": "SentenceSplitter",
        "kwargs": {"chunk_size": 250, "chunk_overlap": 50, "separator": "\n\n"}
    },
    "hierarchical_h1": create_two_stage_h1_chapter_parser
}

RETRIEVER_CONFIGS = {
    "dense_top15": {
        "vector_store_query_mode": "default",
        "similarity_top_k": 15
    },
    "dense_top40": {
        "vector_store_query_mode": "default",
        "similarity_top_k": 40
    },
}

PIPELINE_CONFIGS = {
    "gemma_no_rag": {
        "type": "no_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "llm": {
            "llm_model": "gemma-3-27b-it",
            "temperature": 0.0
        }
    },
    "mistral_no_rag": {
        "type": "no_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "llm": {
            "llm_model": "mistral-large-instruct",
            "temperature": 0.0
        }
    },
    "gpt-oss_no_rag": {
        "type": "no_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },
    "qwen3_no_rag": {
        "type": "no_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "llm": {
            "llm_model": "qwen3-235b-a22b",
            "temperature": 0.0
        }
    },
    "gpt-5_no_rag": {
        "type": "no_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "llm": {
            "llm_model": "gpt-5",
            "llm_provider": "openai",
            "temperature": 0.0
        }
    },

    "gemma_basic_rag_small_chunks": {
        "type": "basic_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "embed_model": "text-embedding-3-large",
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top15"]
        },
        "llm": {
            "llm_model": "gemma-3-27b-it",
            "temperature": 0.0
        }
    },
    "gemma_basic_rag_small_chunks_md": {
        "type": "basic_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "embed_model": "text-embedding-3-large",
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top15"]
        },
        "llm": {
            "llm_model": "gemma-3-27b-it",
            "temperature": 0.0
        }
    },
    "mistral_basic_rag_small_chunks": {
        "type": "basic_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "embed_model": "text-embedding-3-large",
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top15"]
        },
        "llm": {
            "llm_model": "mistral-large-instruct",
            "temperature": 0.0
        }
    },
    "mistral_basic_rag_small_chunks_md": {
        "type": "basic_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "embed_model": "text-embedding-3-large",
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top15"]
        },
        "llm": {
            "llm_model": "mistral-large-instruct",
            "temperature": 0.0
        }
    },
    "gpt-oss_basic_rag_small_chunks": {
        "type": "basic_rag",
        "general": {
            "data_path": PATHS["txt_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "embed_model": "text-embedding-3-large",
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top15"]
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },
    "gpt-oss_basic_rag_small_chunks_md": {
        "type": "basic_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "embed_model": "text-embedding-3-large",
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top15"]
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },

    "gemma_advanced_hierarchical": {
        "type": "advanced_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8]
        },
        "llm": {
            "llm_model": "gemma-3-27b-it",
            "temperature": 0.0
        }
    },
    "mistral_advanced_hierarchical": {
        "type": "advanced_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8]
        },
        "llm": {
            "llm_model": "mistral-large-instruct",
            "temperature": 0.0
        }
    },
    "gpt-oss_advanced_hierarchical": {
        "type": "advanced_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8]
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },

    "gpt-oss_ablation_no_parent_exp": {
        "type": "advanced_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "index_source_pipeline": "gpt-oss_advanced_hierarchical",
            "skip_parent_expansion": True,
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },
    "gpt-oss_ablation_no_reranking": {
        "type": "advanced_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [],
            "index_source_pipeline": "gpt-oss_advanced_hierarchical",
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },
    "gpt-oss_ablation_flat_e5": {
        "type": "advanced_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["small_chunks"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "skip_parent_expansion": True,
        },
        "llm": {
            "llm_model": "openai-gpt-oss-120b",
            "temperature": 0.0
        }
    },

    "gemma_no_rag_vision": {
        "type": "no_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "llm": {
            "llm_model": "gemma-3-27b-it", 
            "temperature": 0.0,
        }
    },

    "gpt2gemma_retrieval2vision": {
        "type": "advanced_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "llm_model": "openai-gpt-oss-120b",
        },
        "llm": {
            "llm_model": "gemma-3-27b-it",
            "temperature": 0.0,
        }
    },
    "gemma2gemma_retrieval2vision": {
        "type": "advanced_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "llm_model": "gemma-3-27b-it",
        },
        "llm": {
            "llm_model": "gemma-3-27b-it",
            "temperature": 0.0,
        }
    },
    "gpt2qwen_retrieval2vision": {
        "type": "advanced_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "llm_model": "openai-gpt-oss-120b",
        },
        "llm": {
            "llm_model": "qwen2.5-vl-72b-instruct",
            "temperature": 0.0,
        }
    },
    "mistral2qwen_retrieval2vision": {
        "type": "advanced_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "llm_model": "mistral-large-instruct",
        },
        "llm": {
            "llm_model": "qwen2.5-vl-72b-instruct",
            "temperature": 0.0,
        }
    },

    "or_gemma3_no_rag_vision": {
        "type": "no_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "llm": {
            "llm_model": "google/gemma-3-27b-it",
            "llm_provider": "openrouter",
            "temperature": 0.0,
        }
    },
    "or_gpt2gemma3_retrieval2vision": {
        "type": "advanced_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "llm_model": "openai/gpt-oss-120b",
            "llm_provider": "openrouter",
            "index_source_pipeline": "or_ravv_hierarchical_e5",
        },
        "llm": {
            "llm_model": "google/gemma-3-27b-it",
            "llm_provider": "openrouter",
            "temperature": 0.0,
        }
    },
    "or_gemma2gemma3_retrieval2vision": {
        "type": "advanced_rag_vision",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "parser": PARSER_CONFIGS["hierarchical_h1"],
            "hybrid_search": False,
            "embed_model": "e5-mistral-7b-instruct",
            "fastembed_sparse_model": None,
            "retriever_kwargs": RETRIEVER_CONFIGS["dense_top40"],
            "node_postprocessor": [rerank_rankgpt_top8],
            "llm_model": "google/gemma-3-27b-it",
            "llm_provider": "openrouter",
            "index_source_pipeline": "or_ravv_hierarchical_e5",
        },
        "llm": {
            "llm_model": "google/gemma-3-27b-it",
            "llm_provider": "openrouter",
            "temperature": 0.0,
        }
    },

    "gpt-5-mini_simple_graph_rag_mem": {
        "type": "graph_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "embed_model": "text-embedding-3-large",
            "embed_provider": "openai",
            "similarity_top_k": 20,
            "path_depth": 2,
        },
        "llm": {
            "llm_model": "gpt-5-mini",
            "temperature": 0.0,
            "llm_provider": "openai",
        },
        "graph": {
            "store": "memory",
            "extractor": "simple",
            "override_graph": False,
            "persist_dir": "data/graphs/simple_graph_gpt-5-mini",
        }
    },
    "gpt-5-mini_schema_graph_rag": {
        "type": "graph_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "retrieval": {
            "embed_model": "text-embedding-3-large",
            "embed_provider": "openai",
            "similarity_top_k": 20,
            "path_depth": 2,
        },
        "llm": {
            "llm_model": "gpt-5-mini",
            "temperature": 0.0,
            "llm_provider": "openai",
        },
        "graph": {
            "store": "memory",
            "extractor": "schema",
            "override_graph": False,
            "persist_dir": "data/graphs/schema_graph_gpt-5-mini",
        }
    },

    "structured_graph_rag": {
        "type": "structured_graph_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "graph": {
            "generation": "deterministic",
            "use_subtree": True,
            "use_crossref": True,
        },
    },
    "structured_graph_rag_no_crossref": {
        "type": "structured_graph_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "graph": {
            "generation": "deterministic",
            "use_subtree": True,
            "use_crossref": False,
        },
    },
    "structured_graph_rag_seeds_only": {
        "type": "structured_graph_rag",
        "general": {
            "data_path": PATHS["md_rules"],
        },
        "graph": {
            "generation": "deterministic",
            "use_subtree": False,
            "use_crossref": False,
        },
    },
}
