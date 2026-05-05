"""
researcher.py — Searches the web using DuckDuckGo and summarizes findings.
No API key required. Uses LangChain + Ollama.
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from duckduckgo_search import DDGS
import config


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a DuckDuckGo search and return results."""
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
        results.append({"title": "Search error", "snippet": str(e), "url": ""})
    return results


def build_search_queries(business_case: str, llm) -> list[str]:
    """Use LLM to generate targeted search queries for the business case."""
    prompt = PromptTemplate(
        input_variables=["business_case"],
        template="""You are a market researcher. Given this business case, generate exactly {num} distinct web search queries to gather real market data.

Business Case: {business_case}

Generate search queries covering:
1. Market size and growth statistics
2. Key competitors and market leaders
3. Recent industry trends (2023-2025)
4. Regulatory or legal aspects
5. Consumer demand and behavior data

Return ONLY the queries, one per line, no numbering, no explanation."""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(business_case=business_case, num=config.SEARCH_QUERIES_COUNT)

    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return queries[:config.SEARCH_QUERIES_COUNT]


def run_researcher(business_case: str, status_callback=None) -> str:
    """
    Main researcher function.
    1. Generate smart search queries via LLM
    2. Run actual DuckDuckGo searches
    3. Summarize findings into a research brief
    """

    if status_callback:
        status_callback("🔍 Researcher: Initializing LLM...")

    llm = OllamaLLM(
        model=config.MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.RESEARCHER_TEMP,
        num_predict=config.MAX_TOKENS,
    )

    # Step 1: Generate search queries
    if status_callback:
        status_callback("🔍 Researcher: Generating search queries...")

    queries = build_search_queries(business_case, llm)

    if status_callback:
        status_callback(f"🔍 Researcher: Generated {len(queries)} search queries")

    # Step 2: Execute searches
    all_search_results = []
    for i, query in enumerate(queries):
        if status_callback:
            status_callback(f"🔍 Researcher: Searching [{i+1}/{len(queries)}]: {query}")

        results = search_web(query, max_results=4)
        all_search_results.append({
            "query": query,
            "results": results
        })

    # Step 3: Format raw search data
    raw_data = ""
    for item in all_search_results:
        raw_data += f"\n\n### Search: {item['query']}\n"
        for r in item["results"]:
            raw_data += f"- **{r['title']}**: {r['snippet']}\n  Source: {r['url']}\n"

    # Step 4: Summarize into a structured research brief
    if status_callback:
        status_callback("🔍 Researcher: Synthesizing findings into research brief...")

    summary_prompt = PromptTemplate(
        input_variables=["business_case", "raw_data"],
        template="""You are a senior market research analyst. You have gathered the following web search results about a business case. 
Synthesize this into a structured, factual research brief.

BUSINESS CASE:
{business_case}

RAW SEARCH DATA:
{raw_data}

Write a comprehensive RESEARCH BRIEF with these sections:
1. MARKET SIZE & GROWTH (include specific numbers where available)
2. KEY COMPETITORS (named companies, their strengths)  
3. INDUSTRY TRENDS (2023-2025, what's changing)
4. REGULATORY ENVIRONMENT (rules, licenses, compliance)
5. CONSUMER DEMAND SIGNALS (who wants this, why)
6. RISKS & MARKET CHALLENGES (real obstacles)
7. COMPARABLE SUCCESS/FAILURE CASES (real examples)

Be specific. Use numbers from the search data. Flag when data is uncertain.
Do not fabricate statistics. If data was not found, say so."""
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    brief = summary_chain.run(business_case=business_case, raw_data=raw_data)

    if status_callback:
        status_callback("✅ Researcher: Brief complete.")

    return brief.strip()