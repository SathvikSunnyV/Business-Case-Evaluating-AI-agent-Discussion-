"""
judge.py - Synthesizes debate into final verdict using ollama streaming.
"""

import asyncio
import ollama
import config


async def llm_generate(prompt: str, temperature: float = 0.3, max_tokens: int = 2500) -> str:
    client = ollama.AsyncClient(host=config.OLLAMA_BASE_URL)
    full_response = ""
    async for chunk in await client.generate(
        model=config.MODEL_NAME,
        prompt=prompt,
        stream=True,
        options={"temperature": temperature, "num_predict": max_tokens}
    ):
        full_response += chunk.get("response", "")
        if chunk.get("done", False):
            break
    return full_response.strip()


async def run_judge_async(
    business_case: str,
    research_brief: str,
    debate_history: list,
    status_callback=None
) -> str:
    if status_callback:
        status_callback("Judge: Reading full debate transcript...")

    transcript = ""
    for item in debate_history:
        transcript += f"\n\n=== ROUND {item['round']} ===\n"
        transcript += f"\nADVOCATE:\n{item['advocate']}\n"
        transcript += f"\nCRITIC:\n{item['critic']}\n"

    prompt = (
        "You are an elite business strategy consultant and investment committee chair.\n"
        "You observed a structured debate between an Advocate and a Critic, backed by real research.\n\n"
        f"BUSINESS CASE:\n{business_case}\n\n"
        f"MARKET RESEARCH:\n{research_brief}\n\n"
        f"DEBATE TRANSCRIPT:\n{transcript}\n\n"
        "Write the final assessment report with EXACTLY these sections:\n\n"
        "## EXECUTIVE SUMMARY\n"
        "3-4 sentences: verdict and top 3 reasons.\n\n"
        "## VERDICT\n"
        "State clearly: RECOMMEND / CONDITIONAL RECOMMEND / DO NOT RECOMMEND\n"
        "Viability Score: X/10 with one-sentence justification.\n\n"
        "## MARKET OPPORTUNITY ANALYSIS\n"
        "What the research data says about market size, timing, and demand.\n\n"
        "## CRITICAL RISKS AND RED FLAGS\n"
        "The Critic's strongest points that cannot be ignored.\n\n"
        "## COMPETITIVE LANDSCAPE\n"
        "Who they compete against and what moats exist or are missing.\n\n"
        "## FINANCIAL VIABILITY\n"
        "Investment required, realistic revenue timeline, break-even estimate.\n\n"
        "## STRATEGIC RECOMMENDATIONS\n"
        "5 specific actions to maximize success. "
        "If not recommending, state what must change to make this viable.\n\n"
        "## FINAL VERDICT JUSTIFICATION\n"
        "2-3 paragraphs synthesizing both sides and explaining your verdict."
    )

    if status_callback:
        status_callback("Judge: Writing final report... (takes 1-3 minutes)")

    result = await llm_generate(prompt, temperature=config.JUDGE_TEMP, max_tokens=2500)

    if status_callback:
        status_callback("Judge: Report complete.")

    return result


def run_judge(
    business_case: str,
    research_brief: str,
    debate_history: list,
    status_callback=None
) -> str:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_judge_async(business_case, research_brief, debate_history, status_callback)
        )
    finally:
        loop.close()