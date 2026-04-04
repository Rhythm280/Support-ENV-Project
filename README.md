---
title: SupportEnv AgentOps Benchmark
emoji: 🎯
colorFrom: cyan
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
---

# 🎯 SupportEnv — AgentOps & Live Observability Benchmark

> **A production-ready, interactive AgentOps platform for evaluating LLM agents on multi-step customer support workflows. Built with FastAPI, WebSockets, and a premium Glassmorphism UI.**

[![OpenEnv v2 Compliant](https://img.shields.io/badge/OpenEnv-v2%20Compliant-brightgreen)](https://openenv.ai)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces%20Ready-blue)](https://huggingface.co/spaces/Joy2801/AgentOps-Support-Benchmark)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](./Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Live AgentOps Dashboard

SupportEnv is no longer just a CLI tool. It is now a full-stack **Observability Platform** for AI Agents.

### 🖥️ Interactive Monitoring
- **Live Stream Terminal**: Watch the agent's internal thoughts and tool calls in real-time via a dedicated WebSocket-powered CRT terminal.
- **Visual Ticket Board**: Track ticket states (Open, Closed, Escalated) and agent assignments as they happen.
- **Reward Sparklines**: Real-time performance tracking with dynamic "Profit & Loss" style visualizations for cumulative rewards.

### 📩 Manual Ticket Injection
Test your agent's edge cases by **injecting custom tickets** directly from the dashboard.
- Define custom user queries.
- Set ground-truth categories, priorities, and escalation requirements.
- Validate how the agent handles specific, user-defined scenarios in real-time.

---

## 🧠 Core Evaluation Logic

SupportEnv measures an agent's ability to handle **stateful transitions** and **business logic**:

- 📋 **Intent Classification**: 5 balanced categories (billing, technical, general, etc.).
- ⚖️ **Dynamic Prioritization**: Urgency levels that escalate automatically the longer a ticket stays open.
- 💬 **Precision Response**: Keyword matching, Persona-aware tone (angry/polite), and **Forbidden Phrase** avoidance.
- 🚨 **Escalation Management**: Judges whether the agent correctly involves Tier 2 support or fails to handle a baseline request.
- 🔄 **Loop Prevention**: Built-in UI safety to stop "Model-in-a-Loop" compute waste.

---

## 🏗️ Architecture

```text
dashboard/              # Glassmorphism UI (HTML/CSS/JS)
api/
└── server.py           # FastAPI, WebSockets, & ENV State Manager
src/
├── environment.py      # Core RL logic: step, reset, state
├── models.py           # Strict Pydantic v2 data contracts
├── generator.py        # Seed-based procedural ticket generation
├── rewards.py          # Dense reward shaping engine
├── graders.py          # Deterministic, non-stochastic grading
├── database.py         # SQLite persistence for every action
└── dynamics.py         # Priority leaks & user tone evolution
```

---

## ⚡ Deployment & Running

### Option 1: Hugging Face Spaces (Recommended for Showcase)
This project is pre-configured for Hugging Face Spaces using Docker.

1.  **Create a New Space**: Choose the **Docker** SDK.
2.  **Add Secrets**: In your Space settings, add these variables:
    *   `HF_TOKEN`: Your Hugging Face API token (required for LLM inference).
    *   `MODEL_NAME`: e.g., `meta-llama/Llama-3-70b-instruct`.
3.  **Push your code**:
    ```bash
    git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
    git push hf main --force
    ```

### Option 2: Docker (Local Desktop)
Run the entire platform locally in a isolated container:
```bash
# 1. Build
docker build -t support-env .

# 2. Run with Persistent Data
docker run -p 7860:7860 \
  -v $(pwd)/data:/home/user/app/data \
  -e HF_TOKEN="your_token_here" \
  support-env
```
Access the dashboard at **[http://localhost:7860](http://localhost:7860)**.

### Option 3: GitHub & Local Development
If you want to contribute or modify the logic:
```bash
# 1. Clone & Setup
git clone https://github.com/Rhythm280/Support-ENV-Project.git
cd Support-ENV-Project
pip install -r requirements.txt

# 2. Start the Backend
uvicorn api.server:app --port 7860 --reload
```

---

## 🛠️ Troubleshooting

- **Port 7860 in use**: If you see `Address already in use`, run `kill -9 $(lsof -t -i:7860)` or use a different port: `--port 8080`.
- **Database Locked**: If two processes touch the DB at once, SQLite might lock. Restart the server; WAL mode is enabled to prevent this.
- **WebSocket Disconnected**: Ensure you are accessing the dashboard via `localhost` or a secure `https` URL if deployed.

---

## 📐 Actions & Rewards

### Action Space
Agents interact via a structured JSON bridge:
```json
{
  "action_type": "classify",
  "ticket_id": 1,
  "content": "billing"
}
```

### Reward Shaping
| Event | Reward |
|-------|--------|
| Correct Label | **+0.30** |
| Resolved & Closed | **+0.70** |
| Wrong Label | **-0.30** |
| **Action Loop** | **-0.50** |
| **Invalid Action** | **-1.00** |

---

## 🤗 Deployment to Hugging Face
SupportEnv is optimized for **Hugging Face Spaces**. 
1. The `Dockerfile` uses a non-root user (UID 1000) for security.
2. The UI is built to handle the HF reverse-proxy seamlessly.
3. Live results are persisted in a persistent SQLite volume.

---

## 🔬 Testing & Baselines
- **Easy Mode**: ~80% Score (Baseline: Keyword Matching)
- **Hard Mode**: ~65% Score (Baseline: Template Logic)

Run local tests:
```bash
python -m pytest tests/ -v
```

---

## 📄 License
MIT. Built for the **OpenEnv Hackathon 2026**.
