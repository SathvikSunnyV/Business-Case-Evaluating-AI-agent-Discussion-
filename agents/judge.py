"""
judge.py — Reads the full debate + research and produces a final verdict report.
Structured like a real investment committee output.
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config


def run_judge(
    business_case: str,
    research_brief: str,
    debate_history: list[dict],
    status_callback=None,
) -> str:

    if status_callback:
        status_callback("⚖️ Judge: Reading full debate transcript...")

    llm = OllamaLLM(
        model=config.MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.JUDGE_TEMP,
        num_predict=2500,
    )

    # Build debate transcript
    transcript = ""
    for item in debate_history:
        transcript += f"\n\n=== ROUND {item['round']} ===\n"
        transcript += f"\nADVOCATE:\n{item['advocate']}\n"
        transcript += f"\nCRITIC:\n{item['critic']}\n"

    prompt = PromptTemplate(
        input_variables=["business_case", "research_brief", "transcript"],
        template="""You are an elite business strategy consultant and investment committee chair.
You have observed a structured debate between an Advocate and a Critic about a business case, backed by real market research.

Your job: produce a DEFINITIVE, BALANCED, ACTIONABLE business assessment report.

Rules:
- Be brutally honest — not diplomatic fluff
- Draw on actual data from the research
- Acknowledge the strongest points from both sides
- Reach a CLEAR VERDICT: RECOMMEND / CONDITIONAL RECOMMEND / DO NOT RECOMMEND
- If conditional, specify exact conditions the business must meet
- Write like a McKinsey partner — sharp, structured, no fluff

═══════════════════════════════════════
BUSINESS CASE:
{business_case}

MARKET RESEARCH:
{research_brief}

DEBATE TRANSCRIPT:
{transcript}
═══════════════════════════════════════

Now write the final report with EXACTLY these sections:

## EXECUTIVE SUMMARY
(3-4 sentences: verdict + top 3 reasons)

## VERDICT
State clearly: RECOMMEND / CONDITIONAL RECOMMEND / DO NOT RECOMMEND
Viability Score: X/10 — (one sentence justification)

## MARKET OPPORTUNITY ANALYSIS
(What the data actually says about the market size, timing, demand)

## CRITICAL RISKS & RED FLAGS
(The Critic's strongest points that must be taken seriously)

## COMPETITIVE LANDSCAPE
(Who they'll fight, what moats exist or don't)

## FINANCIAL VIABILITY
(Investment required, realistic revenue timeline, break-even estimate)

## STRATEGIC RECOMMENDATIONS
(If proceeding: 5 specific actions to maximize success)
(If not proceeding: what would need to change for this to be viable)

## FINAL VERDICT JUSTIFICATION
(2-3 paragraphs explaining why you reached this verdict, synthesizing both sides)"""
    )

    if status_callback:
        status_callback("⚖️ Judge: Writing final report...")

    report = LLMChain(llm=llm, prompt=prompt).run(
        business_case=business_case,
        research_brief=research_brief,
        transcript=transcript,
    )

    if status_callback:
        status_callback("✅ Judge: Final report complete.")

    return report.strip()