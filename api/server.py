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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

class TicketRequest(BaseModel):
    text: str
    category: str
    priority: str
    persona: str = "polite"
    requires_escalation: bool = False


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

# ── WebSocket Manager ───────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str | dict):
        import json
        if isinstance(message, dict):
            message = json.dumps(message)
        
        # Avoid issues with connections closing during loop
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

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

@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard/")

@app.get("/health")
def health():
    return {"status": "ok", "env_ready": _env is not None}


@app.post("/env/reset")
async def reset(req: ResetRequest):
    global _env
    try:
        task = TaskName(req.task)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid task '{req.task}'. Choose: easy, medium, hard")
    
    _env = SupportEnv(task=task, seed=req.seed)
    obs = _env.reset()
    logger.info("ENV RESET — task=%s seed=%d", req.task, req.seed)

    await manager.broadcast({
        "type": "event",
        "event": "reset",
        "task": req.task,
        "seed": req.seed,
        "message": f"Environment reset: {req.task.upper()} (seed {req.seed})"
    })

    return obs.model_dump()


@app.post("/env/step")
async def step(action: Action):
    env = _get_env()
    if env._done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /env/reset to start a new one.")
    
    # Compliance: Handles tuple return (obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    
    await manager.broadcast({
        "type": "event",
        "event": "step",
        "step": env._step_count,
        "action": action.action_type.value,
        "ticket": action.ticket_id,
        "reward": float(reward),
        "done": done,
        "message": f"[STEP {env._step_count}] {action.action_type.value.upper()} on #{action.ticket_id} | Reward: {reward:+.3f}"
    })

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.post("/env/ticket")
async def add_custom_ticket(req: TicketRequest):
    global _env
    if _env is None:
        # Auto-initialize if user tries to inject before reset
        _env = SupportEnv(task="easy", seed=42)
        _env.reset()
        logger.info("Auto-initializing environment for custom ticket injection")
    
    env = _env
    ticket = env.add_ticket(
        text=req.text,
        true_category=req.category,
        true_priority=req.priority,
        persona=req.persona,
        requires_escalation=req.requires_escalation
    )
    
    await manager.broadcast({
        "type": "event",
        "event": "ticket_added",
        "ticket": ticket.model_dump(),
        "message": f"New ticket injected: #{ticket.id} (Correct label: {ticket.true_category.value})"
    })
    
    return {"status": "ok", "ticket_id": ticket.id}


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


# ── WebSocket Endpoint ──────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Initial greeting
        await websocket.send_text(json.dumps({
            "type": "system",
            "message": "📡 Live-Stream Terminal Connected"
        }))
        while True:
            # Just keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

import json
