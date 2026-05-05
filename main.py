"""
main.py — FastAPI server orchestrating all agents with Server-Sent Events (SSE).
Streams progress to the frontend in real time.
"""

import asyncio
import json
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import uvicorn

from agents.researcher import run_researcher
from agents.advocate import run_advocate
from agents.critic import run_critic
from agents.judge import run_judge
import config

app = FastAPI(title="AgentDebate")


# ─── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = any(config.MODEL_NAME.split(":")[0] in m for m in models)
        return {
            "status": "ok",
            "ollama": "running",
            "model": config.MODEL_NAME,
            "model_ready": model_ready,
            "available_models": models,
        }
    except Exception as e:
        return {
            "status": "error",
            "ollama": "not running",
            "message": str(e),
            "fix": f"Run: ollama serve && ollama pull {config.MODEL_NAME}",
        }


# ─── Main debate SSE endpoint ──────────────────────────────────────────────────
@app.get("/api/debate")
async def debate(
    case: str = Query(..., description="Business case to debate"),
    rounds: int = Query(default=config.DEFAULT_ROUNDS, ge=1, le=3),
):
    async def event_generator():
        def send(event: str, data: dict):
            return {"event": event, "data": json.dumps(data)}

        try:
            # ── Phase 1: Research ─────────────────────────────────────────────
            yield send("phase", {
                "phase": "research",
                "message": "🔍 Researcher agent searching the web for live market data..."
            })
            await asyncio.sleep(0.1)

            status_events = []

            def status_cb(msg: str):
                status_events.append(msg)

            # Run researcher in thread (blocking LangChain calls)
            loop = asyncio.get_event_loop()

            research_brief = await loop.run_in_executor(
                None, lambda: run_researcher(case, status_cb)
            )

            # Flush status events
            for msg in status_events:
                yield send("status", {"message": msg})
                await asyncio.sleep(0.05)
            status_events.clear()

            yield send("research_complete", {"content": research_brief})
            await asyncio.sleep(0.1)

            # ── Phase 2: Debate Rounds ────────────────────────────────────────
            yield send("phase", {
                "phase": "debate",
                "message": f"⚔️ Starting debate — {rounds} round(s)..."
            })

            debate_history = []
            last_critic_arg = ""
            last_advocate_arg = ""

            for round_num in range(1, rounds + 1):
                yield send("round_start", {"round": round_num, "totalRounds": rounds})
                await asyncio.sleep(0.1)

                # Advocate
                yield send("agent_thinking", {
                    "agent": "advocate",
                    "round": round_num,
                    "message": f"📈 Advocate building Round {round_num} argument..."
                })

                advocate_arg = await loop.run_in_executor(
                    None,
                    lambda r=round_num, c=last_critic_arg: run_advocate(
                        case, research_brief, r, c, status_cb
                    )
                )
                last_advocate_arg = advocate_arg

                for msg in status_events:
                    yield send("status", {"message": msg})
                    await asyncio.sleep(0.05)
                status_events.clear()

                yield send("advocate_argument", {"round": round_num, "content": advocate_arg})
                await asyncio.sleep(0.1)

                # Critic
                yield send("agent_thinking", {
                    "agent": "critic",
                    "round": round_num,
                    "message": f"🔴 Critic countering in Round {round_num}..."
                })

                critic_arg = await loop.run_in_executor(
                    None,
                    lambda r=round_num, a=last_advocate_arg: run_critic(
                        case, research_brief, r, a, status_cb
                    )
                )
                last_critic_arg = critic_arg

                for msg in status_events:
                    yield send("status", {"message": msg})
                    await asyncio.sleep(0.05)
                status_events.clear()

                yield send("critic_argument", {"round": round_num, "content": critic_arg})
                await asyncio.sleep(0.1)

                debate_history.append({
                    "round": round_num,
                    "advocate": advocate_arg,
                    "critic": critic_arg,
                })

                yield send("round_complete", {"round": round_num})
                await asyncio.sleep(0.1)

            # ── Phase 3: Judge ────────────────────────────────────────────────
            yield send("phase", {
                "phase": "judgment",
                "message": "⚖️ Judge synthesizing full debate into final report..."
            })

            final_report = await loop.run_in_executor(
                None,
                lambda: run_judge(case, research_brief, debate_history, status_cb)
            )

            for msg in status_events:
                yield send("status", {"message": msg})
                await asyncio.sleep(0.05)
            status_events.clear()

            yield send("final_report", {"content": final_report})
            yield send("complete", {"message": "Analysis complete!"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield send("error", {"message": str(e)})

    return EventSourceResponse(event_generator())


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  🤖 AgentDebate — Free Multi-Agent Business Engine")
    print("═" * 55)
    print(f"  Model  : {config.MODEL_NAME}")
    print(f"  Ollama : {config.OLLAMA_BASE_URL}")
    print(f"  UI     : http://localhost:8000")
    print("═" * 55)
    print("\n  Make sure Ollama is running: ollama serve")
    print(f"  Make sure model is pulled:  ollama pull {config.MODEL_NAME}\n")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)