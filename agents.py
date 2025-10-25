import ollama
import re
from prompts import (
    OLLAMA_MODEL, ROUTER_PROMPT, CRITIC_PROMPT,
    SYNTHESIS_PROMPT, FEEDBACK_PROMPT, QA_PROMPT
)
from tools import get_search_results

client = ollama.Client()

def _call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Helper function to call the Ollama API."""
    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Error: Could not get response from model."

# --- AGENT DEFINITIONS ---

def run_router_agent(query: str) -> str:
    """Classifies the query."""
    prompt = ROUTER_PROMPT.format(query=query)
    decision = _call_llm(prompt)

    if 'academic_research' in decision:
        return 'academic_research'
    elif 'code_search' in decision:
        return 'code_search'
    else:
        return 'general_search'

def run_search_agent(query: str, search_type: str) -> list[dict]:
    """Runs the specified search."""
    return get_search_results(query, search_type)

def run_rag_agent(query: str, retriever) -> str:
    """Gets relevant code snippets from the vector store."""
    if not retriever:
        return "No code retriever available."

    docs = retriever.invoke(query)

    context = "\n---\n".join([
        f"File: {doc.metadata.get('source', 'unknown')}\n\n{doc.page_content}"
        for doc in docs
    ])
    return context

def run_critic_agent(query: str, search_results: list[dict]) -> tuple[list[dict], str]:
    """
    Reads all search snippets and returns only the most relevant ones.
    """
    if not search_results:
        return [], "No search results to critique."

    snippet_text = ""
    for res in search_results:
        snippet_text += f"[{res['index']}] {res['snippet']}\n\n"

    prompt = CRITIC_PROMPT.format(query=query, snippets=snippet_text)
    reasoning = f"Critiquing {len(search_results)} snippets..."

    response = _call_llm(prompt)

    if response.lower() == "none":
        return [], "Critic found no relevant snippets."

    try:
        # Extract indices (e.g., "1, 3, 5")
        indices_to_keep = [int(i) for i in re.findall(r'\d+', response)]
        if not indices_to_keep:
            return [], "Critic response was unparsable."

        # Filter the original list
        final_results = [
            res for res in search_results if res['index'] in indices_to_keep
        ]
        reasoning = f"Critic selected snippets: {', '.join(map(str, indices_to_keep))}"
        return final_results, reasoning

    except Exception as e:
        print(f"Critic parsing error: {e}. Returning all results.")
        return search_results, "Critic agent failed, using all snippets."


def run_synthesis_agent(query: str, research_context: str, code_context: str, feedback: str = None) -> str:
    """Generates the final report."""

    feedback_section = ""
    if feedback:
        feedback_section = f"""
        ---
        Your previous attempt failed. You received this feedback:
        "{feedback}"
        Please generate a new, improved report based on this feedback.
        ---
        """

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        feedback_section=feedback_section,
        research_context=research_context,
        code_context=code_context
    )

    return _call_llm(prompt, temperature=0.1)

def run_feedback_agent(query: str, report: str) -> str:
    """Runs the supervisor agent to check report quality."""
    prompt = FEEDBACK_PROMPT.format(query=query, report=report)
    return _call_llm(prompt)

def run_qa_agent(report: str, question: str) -> str:
    """Answers follow-up questions about the report."""
    prompt = QA_PROMPT.format(report=report, question=question)
    return _call_llm(prompt, temperature=0.0)