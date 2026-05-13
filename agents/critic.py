"""
critic.py - Stress-tests the business case using ollama streaming.
"""

import asyncio
import ollama
import config

PERSONA = (
    "You are a skeptical, battle-hardened business analyst and risk consultant. "
    "Think like a VC who has seen hundreds of pitches fail. "
    "Stress-test this idea ruthlessly. Surface every real risk, flaw, and market challenge. "
    "Be surgical and specific, not cynically dismissive. "
    "Write in forceful clear prose. No bullet lists."
)


async def llm_generate(prompt: str, temperature: float = 0.6, max_tokens: int = 1500) -> str:
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


async def run_critic_async(
    business_case: str,
    research_brief: str,
    round_num: int,
    advocate_current: str = "",
    status_callback=None
) -> str:
    if status_callback:
        status_callback(f"Critic: Writing Round {round_num} counter-argument...")

    if advocate_current:
        prompt = (
            f"{PERSONA}\n\n"
            f"BUSINESS CASE:\n{business_case}\n\n"
            f"RESEARCH DATA:\n{research_brief}\n\n"
            f"ADVOCATE'S ARGUMENT (counter this directly):\n{advocate_current}\n\n"
            f"ROUND {round_num} TASK: Directly dismantle the Advocate's specific claims. "
            f"Use research data, failure cases, and hard market realities. "
            f"Identify the single biggest risk they ignored. "
            f"Write 3-4 substantial paragraphs. No bullet lists."
        )
    else:
        prompt = (
            f"{PERSONA}\n\n"
            f"BUSINESS CASE:\n{business_case}\n\n"
            f"RESEARCH DATA:\n{research_brief}\n\n"
            f"ROUND 1 TASK: Make the strongest case AGAINST this business. "
            f"Cover: market risks, financial concerns, execution challenges, failure precedents. "
            f"Write 3-4 substantial paragraphs. Be specific and honest. No bullet lists."
        )

    result = await llm_generate(prompt, temperature=config.CRITIC_TEMP, max_tokens=config.MAX_TOKENS)

    if status_callback:
        status_callback(f"Critic: Round {round_num} complete.")

    return result


def run_critic(
    business_case: str,
    research_brief: str,
    round_num: int,
    advocate_current: str = "",
    status_callback=None
) -> str:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_critic_async(business_case, research_brief, round_num, advocate_current, status_callback)
        )
    finally:
        loop.close()