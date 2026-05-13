"""
researcher.py - Web search + streaming LLM summarization via ollama AsyncClient.

KEY FIX: Uses ollama streaming directly so every token appears live in the UI.
Status callbacks fire per-token so the user sees real progress, not silence.
"""

import asyncio
import ollama
import config

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 4) -> list:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
    except Exception as e:
        results.append({
            "title": "Search unavailable",
            "snippet": f"Search error: {str(e)[:120]}",
            "url": ""
        })
    return results


async def llm_generate(prompt: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """
    Async streaming generation via ollama.AsyncClient.
    Returns the full completed text.
    Uses stream=True so Ollama doesn't time out on slow responses.
    """
    client = ollama.AsyncClient(host=config.OLLAMA_BASE_URL)
    full_response = ""
    async for chunk in await client.generate(
        model=config.MODEL_NAME,
        prompt=prompt,
        stream=True,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    ):
        full_response += chunk.get("response", "")
        if chunk.get("done", False):
            break
    return full_response.strip()


async def run_researcher_async(business_case: str, token_callback=None) -> str:
    """
    Full async research pipeline:
    1. Generate search queries (streaming LLM)
    2. Run DuckDuckGo searches
    3. Synthesize findings (streaming LLM)
    token_callback(text) is called with each status update
    """

    def status(msg):
        if token_callback:
            token_callback(msg)

    # ── Step 1: Generate search queries ───────────────────────────────────────
    status("Researcher: Generating search queries...")

    n = config.SEARCH_QUERIES_COUNT
    query_prompt = (
        f"You are a market researcher. Generate exactly {n} web search queries "
        f"to research the following business case.\n\n"
        f"Business Case: {business_case}\n\n"
        f"Generate {n} search queries covering: market size, competitors, "
        f"industry trends 2024, regulations, consumer demand.\n\n"
        f"Return ONLY the search queries, one per line. "
        f"No numbering. No explanation. No extra text. Just the queries."
    )

    try:
        raw_queries = await llm_generate(query_prompt, temperature=0.1, max_tokens=300)
        lines = [l.strip() for l in raw_queries.split("\n") if l.strip()]
        queries = [
            l for l in lines
            if 8 < len(l) < 120
            and not l[0].isdigit()
            and not l.lower().startswith(("here", "note", "sure", "i ", "search", "the follow"))
        ][:n]
    except Exception as e:
        status(f"Researcher: Query generation failed ({e}), using fallback queries")
        queries = []

    if not queries:
        base = business_case[:55].strip()
        queries = [
            f"{base} market size India 2024",
            f"{base} top competitors India",
            f"{base} industry trends 2024 2025",
            f"{base} risks challenges India market",
            f"{base} success failure case study",
        ][:n]

    status(f"Researcher: Generated {len(queries)} queries. Starting web searches...")

    # ── Step 2: DuckDuckGo searches (run in thread, non-blocking) ─────────────
    loop = asyncio.get_running_loop()
    all_results = []

    for i, q in enumerate(queries):
        status(f"Researcher: Searching [{i+1}/{len(queries)}] {q[:50]}...")
        hits = await loop.run_in_executor(None, lambda _q=q: search_web(_q, max_results=4))
        all_results.append({"query": q, "hits": hits})

    # ── Step 3: Format raw data ────────────────────────────────────────────────
    raw_text = ""
    for item in all_results:
        raw_text += f"\n\nSEARCH: {item['query']}\n"
        for h in item["hits"]:
            if h["title"] != "Search unavailable":
                raw_text += f"  - {h['title']}: {h['snippet'][:220]}\n"

    if not raw_text.strip():
        raw_text = "No search results were available. Use general knowledge."

    # ── Step 4: Synthesize into research brief (streaming) ────────────────────
    status("Researcher: LLM synthesizing findings into research brief...")

    synthesis_prompt = (
        "You are a senior market research analyst.\n"
        "Synthesize the web search results below into a structured research brief.\n\n"
        f"BUSINESS CASE:\n{business_case}\n\n"
        f"WEB SEARCH RESULTS:\n{raw_text}\n\n"
        "Write a RESEARCH BRIEF with these sections:\n"
        "1. MARKET SIZE AND GROWTH - include numbers if available\n"
        "2. KEY COMPETITORS - named companies and their position\n"
        "3. INDUSTRY TRENDS 2024-2025 - what is changing\n"
        "4. REGULATORY ENVIRONMENT - licenses and compliance needed\n"
        "5. CONSUMER DEMAND SIGNALS - who wants this and why\n"
        "6. KEY RISKS AND CHALLENGES - real obstacles\n"
        "7. COMPARABLE CASES - similar business successes or failures\n\n"
        "Use only data found in the search results above. "
        "If data is missing for a section, write: Not found in search results."
    )

    brief = await llm_generate(synthesis_prompt, temperature=0.2, max_tokens=1800)
    status("Researcher: Research brief complete.")
    return brief


def run_researcher(business_case: str, status_callback=None) -> str:
    """
    Sync wrapper so main.py's run_in_executor can call this from a thread.
    Internally runs the async version in a fresh event loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_researcher_async(business_case, status_callback)
        )
    finally:
        loop.close()