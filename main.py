"""
main.py - FastAPI server with full SSE streaming.

Architecture change: agents now use ollama AsyncClient streaming directly,
so they run in the same event loop as FastAPI. No more run_in_executor.
Users see live token output as the LLM generates, not just silence.
"""

import asyncio
import json
import traceback as tb
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from agents.researcher import run_researcher_async
from agents.advocate import run_advocate_async
from agents.critic import run_critic_async
from agents.judge import run_judge_async
import config

app = FastAPI(title="AgentDebate")
BASE_DIR = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health():
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
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
            "ollama": "not_running",
            "model": config.MODEL_NAME,
            "message": str(e),
        }


@app.get("/api/test-llm")
async def test_llm():
    """Quick smoke-test: send a tiny prompt to Ollama and return result."""
    import ollama
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_BASE_URL)
        full = ""
        async for chunk in await client.generate(
            model=config.MODEL_NAME,
            prompt="What is 2+2? Answer in one word.",
            stream=True,
            options={"num_predict": 10}
        ):
            full += chunk.get("response", "")
            if chunk.get("done", False):
                break
        return {"status": "ok", "model": config.MODEL_NAME, "response": full.strip()}
    except Exception as e:
        tb.print_exc()
        return {"status": "error", "error": type(e).__name__, "message": str(e)}


@app.get("/api/debate")
async def debate(
    case: str = Query(...),
    rounds: int = Query(default=config.DEFAULT_ROUNDS, ge=1, le=3),
):
    async def event_generator():
        def send(event: str, data: dict):
            return {"event": event, "data": json.dumps(data, ensure_ascii=False)}

        # Status messages from agents go through this queue
        status_q: asyncio.Queue = asyncio.Queue()

        def cb(msg: str):
            """Called by agent code to report progress."""
            status_q.put_nowait(msg)

        async def drain_status():
            """Yield all queued status messages."""
            while not status_q.empty():
                yield send("status", {"message": status_q.get_nowait()})

        try:
            # ── Phase 1: Research ──────────────────────────────────────────────
            yield send("phase", {
                "phase": "research",
                "message": "Researcher is searching the web for live market data..."
            })
            await asyncio.sleep(0.05)

            # Run directly as async - no run_in_executor needed
            research = await run_researcher_async(case, cb)

            async for ev in drain_status():
                yield ev
                await asyncio.sleep(0.02)

            yield send("research_complete", {"content": research})
            await asyncio.sleep(0.05)

            # ── Phase 2: Debate Rounds ─────────────────────────────────────────
            yield send("phase", {
                "phase": "debate",
                "message": f"Starting debate — {rounds} round(s)..."
            })
            await asyncio.sleep(0.05)

            history = []
            prev_critic = ""
            prev_advocate = ""

            for r in range(1, rounds + 1):
                yield send("round_start", {"round": r, "total": rounds})
                await asyncio.sleep(0.05)

                # Advocate
                yield send("agent_thinking", {
                    "agent": "advocate", "round": r,
                    "message": f"Advocate writing Round {r} argument..."
                })
                await asyncio.sleep(0.05)

                adv = await run_advocate_async(case, research, r, prev_critic, cb)
                prev_advocate = adv

                async for ev in drain_status():
                    yield ev
                    await asyncio.sleep(0.02)

                yield send("advocate_argument", {"round": r, "content": adv})
                await asyncio.sleep(0.05)

                # Critic
                yield send("agent_thinking", {
                    "agent": "critic", "round": r,
                    "message": f"Critic writing Round {r} counter-argument..."
                })
                await asyncio.sleep(0.05)

                crit = await run_critic_async(case, research, r, prev_advocate, cb)
                prev_critic = crit

                async for ev in drain_status():
                    yield ev
                    await asyncio.sleep(0.02)

                yield send("critic_argument", {"round": r, "content": crit})
                await asyncio.sleep(0.05)

                history.append({"round": r, "advocate": adv, "critic": crit})
                yield send("round_complete", {"round": r})
                await asyncio.sleep(0.05)

            # ── Phase 3: Judge ─────────────────────────────────────────────────
            yield send("phase", {
                "phase": "judgment",
                "message": "Judge reading full debate and writing final report..."
            })
            await asyncio.sleep(0.05)

            report = await run_judge_async(case, research, history, cb)

            async for ev in drain_status():
                yield ev
                await asyncio.sleep(0.02)

            yield send("final_report", {"content": report})
            yield send("done", {"message": "Analysis complete!"})

        except Exception as e:
            tb.print_exc()
            yield send("server_error", {
                "message": f"{type(e).__name__}: {e}\n\nCheck terminal for full traceback."
            })
            yield send("done", {"message": "stopped due to error"})

    return EventSourceResponse(event_generator(), ping=20)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AgentDebate - Free Multi-Agent Business Engine")
    print("=" * 60)
    print(f"  Model  : {config.MODEL_NAME}")
    print(f"  Ollama : {config.OLLAMA_BASE_URL}")
    print(f"  UI     : http://localhost:8000")
    print(f"  Test   : http://localhost:8000/api/test-llm")
    print("=" * 60)
    print(f"\n  Run diagnose first: python diagnose.py")
    print(f"  Ollama must be running: ollama serve")
    print(f"  Model must be pulled: ollama pull {config.MODEL_NAME}\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)