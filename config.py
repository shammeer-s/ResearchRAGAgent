# --- Model Configuration ---
OLLAMA_MODEL = "mistral"
OLLAMA_CLIENT = None  # Will be initialized in the agent files

# --- Agent Prompts ---

# 1. Router Agent Prompt
ROUTER_PROMPT = """
You are a master query router. Your job is to classify the user's query into one of the following categories:
- 'academic_research': For queries related to scientific papers, technical concepts, algorithms, or academic topics.
- 'general_search': For all other queries, such as news, opinions, or general knowledge.

Analyze the following query and respond with *only* the category name and nothing else.

Query: "{query}"
"""

# 2. Synthesis Agent Prompt
SYNTHESIS_PROMPT = """
You are an expert research assistant. Your task is to answer the user's query based *only* on the provided research snippets.
Do not use any prior knowledge.

You must follow these rules:
1.  Analyze the user's query: "{query}"
2.  Review the research snippets below. Each snippet is tagged with a [Source X].
3.  Write a comprehensive, synthesized answer to the user's query.
4.  You **MUST** cite the information you use by including the [Source X] tag directly after the sentence or fact it supports.
5.  If the provided snippets do not contain enough information to answer the query, state that clearly.
6.  Do not invent information.

Research Snippets:
{context}

Answer:
"""

# --- Tool Configuration ---
SEARCH_RESULT_COUNT = 5
ACADEMIC_SEARCH_SITES = [
    "site:arxiv.org",
    "site:jmlr.org",
    "site:dl.acm.org",
    "site:semanticscholar.org",
    "site:neurips.cc"
]