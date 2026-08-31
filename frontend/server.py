"""Local web chat demo for recording — wraps the real Agent, not a mock.

Not part of the evaluated pipeline and not meant to run on a judge's
machine: judging happens off an uploaded video, so this only needs to work
reliably here, for one person, in one browser tab, while recording a take.
It drives the exact same `Agent.reset`/`Agent.respond` contract that
`scripts/chat.py` and the official evaluator use.

Run from the repo root:
    python -m uvicorn frontend.server:app --port 8000
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import Agent
from src.orchestrator import MAX_TURNS

STATIC_DIR = Path(__file__).resolve().parent / "static"
CATALOG_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample_catalog.jsonl")
TOP_K = 10

DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": [],
    "summary": "",
}

print(f"Loading catalog from {CATALOG_PATH} ...")
agent = Agent(CATALOG_PATH)
print(f"Loaded {len(agent.catalog)} products.")

_turns: dict[str, int] = {}

app = FastAPI(title="Shopping Copilot — Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SessionRequest(BaseModel):
    profile: dict | None = None


class MessageRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session")
def create_session(body: SessionRequest) -> dict:
    profile = {**DEFAULT_PROFILE, **(body.profile or {})}
    session_id = uuid4().hex[:8]
    agent.reset(session_id, profile)
    _turns[session_id] = 0
    return {"session_id": session_id, "max_turns": MAX_TURNS, "profile": profile}


@app.post("/api/message")
def send_message(body: MessageRequest) -> dict:
    turn = _turns.get(body.session_id, 0) + 1
    if turn > MAX_TURNS:
        return {"session_complete": True, "max_turns": MAX_TURNS, "turn": MAX_TURNS}

    _turns[body.session_id] = turn
    response = agent.respond(body.session_id, body.message, turn, TOP_K)

    recommendations = []
    for rec in response.get("recommendations") or []:
        pid = rec["parent_asin"]
        product = agent.catalog.by_id.get(pid)
        recommendations.append(
            {
                "parent_asin": pid,
                "title": product.title if product else "(unknown product)",
                "price": product.price if product else None,
                "average_rating": product.average_rating if product else None,
                "rating_number": product.rating_number if product else None,
                "store": product.store if product else None,
            }
        )

    return {
        "turn": turn,
        "max_turns": MAX_TURNS,
        "message": response.get("message", ""),
        "ask_attribute": response.get("ask_attribute"),
        "recommendations": recommendations,
        "session_complete": turn >= MAX_TURNS,
    }
