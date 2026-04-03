"""
api/server.py — FastAPI REST interface for SupportEnv.

Exposes the environment over HTTP so the dashboard and external
evaluators can interact with it without running Python directly.

Endpoints:
  POST /env/reset        → Observation
  POST /env/step         → StepResult
  GET  /env/state        → Full state dict (includes ground-truth)
  GET  /env/grade        → GradeReport
  GET  /health           → {"status": "ok"}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.environment import SupportEnv
from src.models import Action, TaskName

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Request bodies
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


# ─────────────────────────────────────────────
# App + CORS
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SupportEnv API starting up…")
    yield
    logger.info("SupportEnv API shutting down…")


app = FastAPI(
    title="SupportEnv API",
    description="AI Customer Support Ticket Resolution RL Environment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard as static files at root
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

# ─────────────────────────────────────────────
# State — one env instance per server process
# ─────────────────────────────────────────────

_env: Optional[SupportEnv] = None


def _get_env() -> SupportEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /env/reset first.")
    return _env


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env_ready": _env is not None}


@app.post("/env/reset")
def reset(req: ResetRequest):
    global _env
    try:
        task = TaskName(req.task)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid task '{req.task}'. Choose: easy, medium, hard")
    
    _env = SupportEnv(task=task, seed=req.seed)
    obs = _env.reset()
    logger.info("ENV RESET — task=%s seed=%d", req.task, req.seed)
    return obs.model_dump()


@app.post("/env/step")
def step(action: Action):
    env = _get_env()
    if env._done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /env/reset to start a new one.")
    
    # Compliance: Handles tuple return (obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/env/state")
def state():
    return _get_env().state()


@app.get("/env/grade")
def grade():
    env = _get_env()
    report = env.grade()
    return {
        "score": round(report.score, 4),
        "summary": report.summary,
        "breakdown": report.breakdown,
    }


@app.get("/env/config")
def config():
    env = _get_env()
    cfg = env.config
    return {
        "task": cfg.name,
        "max_steps": cfg.max_steps,
        "ticket_count": cfg.ticket_count,
        "allowed_actions": [a.value for a in cfg.allowed_actions],
        "description": cfg.description,
    }
