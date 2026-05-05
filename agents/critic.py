"""
critic.py — Stress-tests the business case, surfaces real risks.
Directly counters the Advocate's arguments each round.
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config


CRITIC_SYSTEM = """You are a skeptical, battle-hardened business analyst and risk consultant.
You think like a VC who has seen 500 pitches — most of which failed.
Your role is to stress-test business ideas ruthlessly — not to destroy them, but to surface every real risk, flaw, and market challenge.
You use market data, historical failure cases, competitive dynamics, and economic realities.
You are surgical and specific — not cynically dismissive.
Challenge: unrealistic assumptions, capital burn, market saturation, regulatory barriers, competitive moats, execution difficulty.
Write in forceful, clear prose backed by research data."""


def run_critic(
    business_case: str,
    research_brief: str,
    round_num: int,
    advocate_current: str = "",
    status_callback=None,
) -> str:

    if status_callback:
        status_callback(f"🔴 Critic: Building Round {round_num} counter-argument...")

    llm = OllamaLLM(
        model=config.MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.CRITIC_TEMP,
        num_predict=config.MAX_TOKENS,
    )

    if advocate_current:
        prompt = PromptTemplate(
            input_variables=["business_case", "research_brief", "advocate_current", "round_num"],
            template="""{system}

BUSINESS CASE:
{business_case}

MARKET RESEARCH DATA (use this to find risks and failure patterns):
{research_brief}

THE ADVOCATE JUST ARGUED:
{advocate_current}

ROUND {round_num} — Your task:
1. Directly dismantle the Advocate's specific claims with counter-evidence
2. Highlight the risks, assumptions, or gaps they glossed over
3. Cite real-world failures or market data that contradict their optimism
4. Identify the single biggest threat to this business that they ignored

Write 3-4 substantial paragraphs. Be specific, data-driven, and unflinching.""".replace("{system}", CRITIC_SYSTEM)
        )
        result = LLMChain(llm=llm, prompt=prompt).run(
            business_case=business_case,
            research_brief=research_brief,
            advocate_current=advocate_current,
            round_num=round_num,
        )
    else:
        prompt = PromptTemplate(
            input_variables=["business_case", "research_brief"],
            template="""{system}

BUSINESS CASE:
{business_case}

MARKET RESEARCH DATA (use this to find risks):
{research_brief}

ROUND 1 — Opening critique:
Make the strongest case AGAINST or identifying critical risks. Cover:
1. The most dangerous market/competitive risks
2. Financial viability concerns (capital, margins, burn rate)
3. Execution challenges specific to this type of business
4. Historical precedents of similar failures

Write 3-4 substantial paragraphs. Be specific and honest.""".replace("{system}", CRITIC_SYSTEM)
        )
        result = LLMChain(llm=llm, prompt=prompt).run(
            business_case=business_case,
            research_brief=research_brief,
        )

    if status_callback:
        status_callback(f"✅ Critic: Round {round_num} argument ready.")

    return result.strip()