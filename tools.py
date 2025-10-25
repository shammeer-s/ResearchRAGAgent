from ddgs import DDGS
from config import ACADEMIC_SEARCH_SITES, SEARCH_RESULT_COUNT

def get_search_results(query: str, search_type: str) -> list[dict]:
    """
    Performs a web search using DuckDuckGo and returns structured results.
    """
    print(f"[Tool] Running search type: {search_type}")

    if search_type == "academic_research":
        # Fix: Corrected typo from ACADEMWINIC... to ACADEMIC...
        site_query = " OR ".join(ACADEMIC_SEARCH_SITES)
        query = f"{query} ({site_query})"

    results = []
    with DDGS() as ddgs:

        # Fix: Changed 'keywords=query' to be the first positional argument
        ddgs_results = ddgs.text(
            query,
            region='wt-wt',
            safesearch='off',
            timelimit=None,
            max_results=SEARCH_RESULT_COUNT
        )

        if not ddgs_results:
            return []

        for i, r in enumerate(ddgs_results):
            results.append({
                "index": i + 1,
                "title": r['title'],
                "snippet": r['body'],
                "source_url": r['href']
            })
    return results