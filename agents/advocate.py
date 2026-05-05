"""
advocate.py — Argues FOR the business case using research data.
Responds to Critic's previous arguments in subsequent rounds.
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config


ADVOCATE_SYSTEM = """You are a sharp, experienced business advocate and investment analyst.
Your role is to build the STRONGEST POSSIBLE CASE in favor of a business opportunity.
You argue with conviction — using market data, strategic logic, and real-world analogies.
You are persuasive but grounded. You briefly acknowledge risks only to explain why they are manageable.
Write in clear, punchy paragraphs. Argue like a seasoned investor pitching to a committee.
Use specific data points from the research wherever possible."""


def run_advocate(
    business_case: str,
    research_brief: str,
    round_num: int,
    critic_previous: str = "",
    status_callback=None,
) -> str:

    if status_callback:
        status_callback(f"📈 Advocate: Building Round {round_num} argument...")

    llm = OllamaLLM(
        model=config.MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.ADVOCATE_TEMP,
        num_predict=config.MAX_TOKENS,
    )

    if critic_previous:
        # Subsequent rounds: respond to critic's specific points
        prompt = PromptTemplate(
            input_variables=["business_case", "research_brief", "critic_previous", "round_num"],
            template="""{system}

BUSINESS CASE:
{business_case}

MARKET RESEARCH DATA (use this to support arguments):
{research_brief}

THE CRITIC'S PREVIOUS ARGUMENT (address these directly):
{critic_previous}

ROUND {round_num} — Your task:
1. Directly counter the Critic's specific objections with data and logic
2. Strengthen your core argument for why this business makes sense
3. Point to market evidence that supports viability
4. End with your strongest point

Write 3-4 substantial paragraphs. Be direct and confident.""".replace("{system}", ADVOCATE_SYSTEM)
        )
        result = LLMChain(llm=llm, prompt=prompt).run(
            business_case=business_case,
            research_brief=research_brief,
            critic_previous=critic_previous,
            round_num=round_num,
        )
    else:
        # Round 1: Opening argument
        prompt = PromptTemplate(
            input_variables=["business_case", "research_brief"],
            template="""{system}

BUSINESS CASE:
{business_case}

MARKET RESEARCH DATA (use this to support arguments):
{research_brief}

ROUND 1 — Opening argument:
Make the strongest case FOR pursuing this business. Cover:
1. Why the market opportunity is real and significant
2. Why the timing is right
3. What makes this viable (financials, demand, execution)
4. Why risks are manageable

Write 3-4 substantial paragraphs backed by the research data.""".replace("{system}", ADVOCATE_SYSTEM)
        )
        result = LLMChain(llm=llm, prompt=prompt).run(
            business_case=business_case,
            research_brief=research_brief,
        )

    if status_callback:
        status_callback(f"✅ Advocate: Round {round_num} argument ready.")

    return result.strip()