---
title: SupportEnv AgentOps Benchmark
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🎯 SupportEnv — Dynamic AgentOps Benchmark for LLM Agents

> **A production-ready, reproducible, stateful reinforcement learning environment for evaluating LLM agents on multi-step customer support workflows.**

[![OpenEnv v2 Compliant](https://img.shields.io/badge/OpenEnv-v2%20Compliant-brightgreen)](https://openenv.ai)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces%20Ready-blue)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](./Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Why SupportEnv?

Most LLM agent benchmarks evaluate static, single-step tasks. Real-world enterprise applications require **multi-step reasoning under evolving state** — exactly what SupportEnv tests.

SupportEnv is built as an **AgentOps benchmark**: a structured, reproducible environment that measures an agent's ability to:

- 📋 **Classify** ambiguous customer intent across 5 categories
- ⚖️ **Prioritize** tickets correctly under urgency pressure
- 💬 **Generate relevant responses** with required keywords, avoiding banned phrases
- 🚨 **Escalate** judiciously (false escalations are penalized)
- 🔗 **Handle dependencies** (some tickets block others)
- 🔄 **Avoid loops** (repeated actions incur dense negative reward)

The environment evolves during an episode: priorities auto-escalate, user tone worsens on bad responses, and partial observability hides ground truth from the agent.

---

## 🏗️ Architecture

```
src/
├── environment.py      # step/reset/state/grade/replay
├── models.py           # Pydantic v2 data contracts
├── generator.py        # Procedural, seed-based ticket generation
├── rewards.py          # Dense reward shaping
├── graders.py          # Deterministic keyword-based graders
├── tasks.py            # 3 task configs with success criteria
├── database.py         # SQLite persistence (episodes/actions/metrics)
├── supabase_client.py  # Optional Supabase analytics
├── dynamics.py         # Priority escalation, tone worsening, dependencies

inference.py            # Baseline agent ([START]/[STEP]/[END] format)
openenv.yaml            # Full OpenEnv v2 specification
Dockerfile              # HuggingFace Spaces compatible
api/server.py           # FastAPI REST interface
```

---

## 🎯 Tasks

| Task | Difficulty | Actions | Tickets | Max Steps | Target Score |
|------|-----------|---------|---------|-----------|--------------|
| **Easy** | 🟢 Low | classify | 5 | 15 | ≥ 80% |
| **Medium** | 🟡 Medium | classify, prioritize | 6 | 25 | ≥ 75% |
| **Hard** | 🔴 High | classify, prioritize, respond, escalate, close | 7 | 40 | ≥ 70% |

### Task 1 — Classification (Easy)
Correctly classify 5 tickets into: `billing | technical | general | complaint | positive`.  
Score = fraction of tickets correctly classified.

### Task 2 — Response Generation (Medium)
Classify and prioritize 6 tickets.  
Score = `0.5 × classification_rate + 0.5 × priority_rate`.

### Task 3 — Full Workflow (Hard)
Full pipeline with all 5 actions. Response quality is graded by:
- Required keywords present in response text
- Absence of forbidden phrases
- Correct escalation decisions
- Respect for ticket dependencies

---

## 📐 Action Space

```python
Action(
    action_type: "classify" | "prioritize" | "respond" | "escalate" | "close",
    ticket_id: int,
    content: str | None  # category | priority | response text | null
)
```

## 👁️ Observation Space

```python
Observation(
    tickets: List[{
        "id": int,
        "text": str,
        "persona": "angry" | "polite" | "confused",
        "category": str | None,     # agent-assigned
        "priority": str | None,     # agent-assigned
        "status": "open" | "closed" | "escalated",
        "urgency_hint": "urgent" | "normal",  # partial observability
        "dependencies": List[int],
    }],
    step_count: int,
    cumulative_reward: float,
    last_action_error: bool,
    task: str,
    info: {
        "tickets_resolved": int,
        "efficiency": float,
        "loops_detected": int,
        "priority_misses": int,
    }
)
```

---

## ⚡ Reward Function

| Event | Reward |
|-------|--------|
| Correct classification | **+0.30** |
| Correct prioritization | **+0.30** |
| Relevant response (keywords match) | **+0.50** |
| No forbidden phrases bonus | **+0.20** |
| Correct escalation | **+0.30** |
| Close fully-resolved ticket | **+0.20** |
| Wrong classification/priority | **-0.30** |
| Irrelevant response | **-0.30** |
| Incorrect escalation | **-0.30** |
| **Repeated action (loop)** | **-0.50** |
| **Invalid action / unknown ticket** | **-1.00** |

---

## 🧪 Graders

All graders are **deterministic** — no LLM, no randomness:

| Signal | Weight |
|--------|--------|
| Category match | **+0.3** |
| Required keywords in response | **+0.5** |
| No forbidden phrases | **+0.2** |

Score ∈ [0.0, 1.0]

---

## 🔄 Dynamic State

- **Priority Escalation**: Each ticket has an `escalation_step` threshold. If the step count exceeds it and the ticket is unresolved, its priority increases automatically.
- **Tone Worsening**: User persona degrades (`polite → confused → angry`) if the agent provides bad/irrelevant responses.
- **Ticket Dependencies**: In the Hard task, some tickets block others — an agent must resolve prerequisite tickets first.
- **Partial Observability**: Ground-truth labels (`true_category`, `true_priority`, `requires_escalation`) are hidden from agent observations. Only `urgency_hint` is exposed.

---

## 🗄️ Database

Every episode is fully logged to SQLite (`support_env.db`):

| Table | Contents |
|-------|---------|
| `episodes` | Task, seed, agent_mode, score, steps, timestamps |
| `tickets` | End-of-episode ticket snapshots |
| `actions` | Every step with action, reward, reason |
| `metrics` | efficiency, loops_detected, priority_misses, final_score |

### Episode Replay

```python
from src.environment import SupportEnv

env = SupportEnv(task="hard", seed=42)
env.reset()
# ... run episode ...
env.grade()

# Replay any past episode
actions = env.replay(episode_id=1)
```

---

## 🚀 Quickstart

### Option 1: Local Python

```bash
git clone <your-repo>
cd hackathon-project
pip install -r requirements.txt

# Run smoke test
python smoke_test.py

# Run baseline agent (rule-based, no API key needed)
python inference.py --task all --agent rule

# Run with LLM (requires HF_TOKEN)
HF_TOKEN=your_token python inference.py --task hard --agent llm
```

### Option 2: Docker (Recommended)

```bash
# Build
docker build -t support-env .

# Run API server
docker run -p 7860:7860 support-env

# Run baseline agent inside container
docker exec -it $(docker ps -q --filter ancestor=support-env) \
    python inference.py --task all --agent rule

# Run with LLM
docker run -p 7860:7860 \
  -e HF_TOKEN="your_token" \
  -e MODEL_NAME="meta-llama/Llama-3-70b-instruct" \
  support-env
```

### Option 3: FastAPI REST

```bash
uvicorn api.server:app --port 7860

# Reset
curl -X POST http://localhost:7860/env/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard", "seed": 42}'

# Step
curl -X POST http://localhost:7860/env/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "ticket_id": 1, "content": "billing"}'

# Grade
curl http://localhost:7860/env/grade
```

---

## ⚙️ Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `SUPPORT_ENV_DB` | `support_env.db` | SQLite database path |
| `ENABLE_SUPABASE` | `false` | Enable Supabase analytics |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service key |
| `HF_TOKEN` / `API_KEY` | — | API key for LLM inference |
| `API_BASE_URL` | HF Router | LLM API base URL |
| `MODEL_NAME` | Llama-3-70b | Model to use for LLM agent |

---

## 📊 Baseline Results (Rule-Based Agent, seed=42)

| Task | Score | Notes |
|------|-------|-------|
| **Easy** | ~80% | Strong keyword matching |
| **Medium** | ~72% | Priority inference is heuristic |
| **Hard** | ~65% | Response quality limited by templates |
| **Composite** | ~72% | Reproducible across runs |

---

## 🧬 Dynamic Ticket Generation

Tickets are generated procedurally — not sampled from a fixed bank:

- **Seed-controlled**: `seed=42` always produces the same episode
- **Categories**: billing, technical, general, complaint, positive (balanced round-robin)
- **Personas**: angry, polite, confused (affects text tone and rewards)
- **Noise**: Realistic typos and casing errors injected at configurable levels
- **Metadata**: Each ticket includes `required_keywords`, `forbidden_phrases`, and `resolution_steps`

---

## 🔬 Running Tests

```bash
python -m pytest tests/ -v
python smoke_test.py
```

---

## 📄 License

MIT. Built for the OpenEnv Hackathon 2026.
