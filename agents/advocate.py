"""
advocate.py - Argues FOR the business case using ollama streaming.
"""

import asyncio
import ollama
import config

PERSONA = (
    "You are a sharp, experienced business advocate and investment analyst. "
    "Build the STRONGEST POSSIBLE CASE in favor of this business opportunity. "
    "Argue with conviction using market data, strategic logic, and real-world analogies. "
    "Be persuasive but grounded. Write clear punchy paragraphs. No bullet lists."
)


async def llm_generate(prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
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


async def run_advocate_async(
    business_case: str,
    research_brief: str,
    round_num: int,
    critic_previous: str = "",
    status_callback=None
) -> str:
    if status_callback:
        status_callback(f"Advocate: Writing Round {round_num} argument...")

    if critic_previous:
        prompt = (
            f"{PERSONA}\n\n"
            f"BUSINESS CASE:\n{business_case}\n\n"
            f"RESEARCH DATA:\n{research_brief}\n\n"
            f"CRITIC'S PREVIOUS ARGUMENT (counter this directly):\n{critic_previous}\n\n"
            f"ROUND {round_num} TASK: Counter the Critic's specific objections using "
            f"research data and logic. Then reinforce your strongest points. "
            f"Write 3-4 substantial paragraphs. No bullet lists."
        )
    else:
        prompt = (
            f"{PERSONA}\n\n"
            f"BUSINESS CASE:\n{business_case}\n\n"
            f"RESEARCH DATA:\n{research_brief}\n\n"
            f"ROUND 1 TASK: Make the strongest case FOR this business. "
            f"Cover: market opportunity, timing, financial viability, manageable risks. "
            f"Write 3-4 substantial paragraphs backed by the research data. No bullet lists."
        )

    result = await llm_generate(prompt, temperature=config.ADVOCATE_TEMP, max_tokens=config.MAX_TOKENS)

    if status_callback:
        status_callback(f"Advocate: Round {round_num} complete.")

    return result


def run_advocate(
    business_case: str,
    research_brief: str,
    round_num: int,
    critic_previous: str = "",
    status_callback=None
) -> str:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_advocate_async(business_case, research_brief, round_num, critic_previous, status_callback)
        )
    finally:
        loop.close()