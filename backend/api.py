"""
api.py

FastAPI HTTP server — sits next to main.py, imports the same stack.

Run with:
    cd backend
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.agent import Agent
from core.tools import FileCreationTool
from core.tools.registry import ToolRegistry
from core.tools.system_control_tool import SystemControlTool
from core.tools.ui_automation_tool import UIAutomationTool
from execution.executor import ToolExecutor


# ── Build agent once at startup ─────────────────────────────────────────── #

def _build_agent() -> Agent:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY not set — add it to backend/.env")
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(UIAutomationTool())
    registry.register(FileCreationTool())
    registry.register(SystemControlTool())

    executor = ToolExecutor(registry)
    agent    = Agent(registry=registry, executor=executor, api_key=api_key)
    logger.info("Agent ready — tools: %s", registry.list_names())
    return agent


agent = _build_agent()


# ── FastAPI app ──────────────────────────────────────────────────────────── #

app = FastAPI(title="Gladden API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────── #

class InstructionRequest(BaseModel):
    instruction: str

class ExecuteResponse(BaseModel):
    success: bool
    output:  Any        = None
    error:   str | None = None
    events:  list[dict] = []


# ── Routes ───────────────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {"status": "ok", "tools": agent._registry.list_names()}


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: InstructionRequest):
    if not req.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction must not be empty.")

    collected_events: list[dict] = []

    def _collect(event: dict) -> None:
        collected_events.append(event)

    # Patch executor for this request only, then always restore
    original = agent._executor.execute

    def _patched(tool_name: str, **kwargs):
        return original(tool_name, event_callback=_collect, **kwargs)

    agent._executor.execute = _patched

    try:
        result = agent.run(req.instruction)
    finally:
        agent._executor.execute = original

    return ExecuteResponse(
        success=result.success,
        output=result.output,
        error=result.error,
        events=collected_events,
    )