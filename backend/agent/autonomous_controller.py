"""
agent/autonomous_controller.py

AutonomousController — OpenClaw-style goal-driven execution loop.

Architecture
------------
This module sits alongside the existing Agent and adds an optional
autonomous execution mode.  It reuses:
  - ToolExecutor  (execution gateway — unchanged)
  - ToolRegistry  (tool discovery — unchanged)
  - Groq SDK      (same client as Agent)
  - perception.py (lightweight screen capture)

It does NOT modify any existing tool, executor, or registry code.

Modes
-----
Deterministic (default):  Agent.run(instruction)  — existing behaviour.
Autonomous (new):         AutonomousController.run(goal)
                          Triggered when instruction starts with "AUTO:".

Execution loop
--------------
for step in range(max_steps):
    1. observe()   — optional screenshot + description
    2. get_next_action(goal, context)  — ask LLM for next tool call
    3. execute_step()  — run via ToolExecutor
    4. is_goal_achieved(goal, context) — check if we're done
    5. update context with result
    6. break on goal achieved / failure / max steps

Safety constraints
------------------
- Only ui_automation and system_control tools are allowed in auto mode.
- Shell execution is never permitted (enforced by tool layer).
- Hard cap of MAX_AUTONOMOUS_STEPS (default 8).
- Every step emits structured events via event_callback.

Event schema (stable)
---------------------
{
    "type":      "info"|"status"|"error",
    "stage":     "<stage_name>",
    "message":   "<description>",
    "tool":      "autonomous_controller",
    "timestamp": "<ISO-8601>",
    "step":      <int>   # present on step-level events
}
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from groq import Groq

from core.tools.base import ToolResult
from core.tools.registry import ToolRegistry
from execution.executor import ToolExecutor

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

MAX_AUTONOMOUS_STEPS: int = 8
_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Tools allowed in autonomous mode — subset of all registered tools
_ALLOWED_AUTO_TOOLS = frozenset({"ui_automation", "system_control"})


# ── Event helpers ─────────────────────────────────────────────────────────── #

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(
    type_: str,
    stage: str,
    message: str,
    step: Optional[int] = None,
) -> dict:
    ev: dict[str, Any] = {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      "autonomous_controller",
        "timestamp": _utc_now(),
    }
    if step is not None:
        ev["step"] = step
    return ev


def _emit(callback: EventCallback, event: dict) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning("autonomous event_callback raised: %s", exc)


# ── Prompt templates ──────────────────────────────────────────────────────── #

_STEP_GEN_SYSTEM = """\
You are an autonomous desktop agent.  Your job is to decide the SINGLE NEXT
ACTION needed to make progress toward a high-level goal.

You control a Windows desktop via two tools:

1. ui_automation — visible UI workflows:
   Workflows: create_file_notepad, create_file_vscode, send_email_browser,
              send_whatsapp_desktop, send_whatsapp_advanced,
              play_youtube_video, linkedin_action, code_workflow_cpp,
              launch_application, take_screenshot

2. system_control — secure OS operations:
   Domains: process (list/inspect/kill)
            system  (cpu_usage/memory_usage/disk_usage/uptime)
            filesystem (list_directory/file_info/create_directory/rename_file/delete_file)

RULES:
- Choose the SIMPLEST single action that advances the goal.
- Prefer ui_automation workflows over low-level system_control calls.
- If the goal is already achieved (last_result shows success), output DONE.
- Never invent new tools or workflows.

RESPONSE FORMAT — respond ONLY with a valid JSON object, no markdown, no preamble:

If a step is needed:
{
  "tool": "<tool_name>",
  "arguments": { ... },
  "reasoning": "<1 sentence why this step is needed>"
}

If the goal is achieved:
{
  "tool": "DONE",
  "arguments": {},
  "reasoning": "<why goal is complete>"
}

If something is wrong and you cannot proceed:
{
  "tool": "ABORT",
  "arguments": {},
  "reasoning": "<why you cannot continue>"
}
"""

_STEP_GEN_USER_TEMPLATE = """\
GOAL: {goal}

STEP: {step} of {max_steps}

CONTEXT:
{context_text}

What is the single next action?
"""

_GOAL_CHECK_SYSTEM = """\
You are evaluating whether a high-level goal has been achieved.

Reply with ONLY a JSON object:
{
  "achieved": true | false,
  "confidence": "high" | "medium" | "low",
  "reason": "<1 sentence>"
}
"""

_GOAL_CHECK_USER_TEMPLATE = """\
GOAL: {goal}

LAST STEP RESULT:
  success: {success}
  output: {output}

EXECUTION HISTORY:
{history}

Has the goal been fully achieved?
"""


# ── AutonomousController ──────────────────────────────────────────────────── #

class AutonomousController:
    """
    Goal-driven iterative execution loop.

    Parameters
    ----------
    registry    : ToolRegistry
    executor    : ToolExecutor
    groq_client : groq.Groq
    model_name  : str
    max_steps   : int   hard cap on loop iterations (default MAX_AUTONOMOUS_STEPS)
    use_perception : bool  whether to capture screenshots each step (default False)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        groq_client: Groq,
        model_name: str = _DEFAULT_MODEL,
        max_steps: int = MAX_AUTONOMOUS_STEPS,
        use_perception: bool = False,
    ) -> None:
        self._registry       = registry
        self._executor       = executor
        self._client         = groq_client
        self._model          = model_name
        self._max_steps      = max_steps
        self._use_perception = use_perception

    # ================================================================== #
    #  Public entry point                                                  #
    # ================================================================== #

    def run(
        self,
        goal: str,
        event_callback: EventCallback = None,
    ) -> ToolResult:
        """
        Execute *goal* autonomously.

        Returns a ToolResult summarising the overall outcome.
        """
        goal = goal.strip()
        if not goal:
            return ToolResult(success=False, error="Goal must not be empty.")

        _emit(event_callback, _event(
            "info", "autonomous_started",
            f"Autonomous mode activated. Goal: {goal!r}  max_steps={self._max_steps}",
        ))

        context: dict[str, Any] = {
            "goal":     goal,
            "step":     0,
            "history":  [],   # list of {step, tool, arguments, success, output}
            "last_result": None,
            "screen_desc": None,
        }

        final_result: ToolResult = ToolResult(
            success=False,
            error="Max steps reached without achieving goal.",
        )

        for step_idx in range(1, self._max_steps + 1):
            context["step"] = step_idx

            # ── 1. Observe ──────────────────────────────────────────── #
            if self._use_perception:
                context["screen_desc"] = self._observe(event_callback, step_idx)

            # ── 2. Generate next action ─────────────────────────────── #
            _emit(event_callback, _event(
                "info", "step_generated",
                f"Step {step_idx}: asking LLM for next action…",
                step=step_idx,
            ))

            action = self._get_next_action(goal, context)

            if action is None:
                _emit(event_callback, _event(
                    "error", "step_generation_failed",
                    "LLM did not return a valid action.  Stopping.",
                    step=step_idx,
                ))
                final_result = ToolResult(
                    success=False,
                    error="LLM failed to produce a valid action.",
                )
                break

            tool_name = action.get("tool", "")
            arguments = action.get("arguments", {})
            reasoning = action.get("reasoning", "")

            _emit(event_callback, _event(
                "info", "step_generated",
                f"Step {step_idx}: tool={tool_name!r}  reasoning={reasoning!r}",
                step=step_idx,
            ))

            # ── 3. Check for terminal actions ───────────────────────── #
            if tool_name == "DONE":
                _emit(event_callback, _event(
                    "status", "goal_achieved",
                    f"LLM declared goal achieved: {reasoning}",
                    step=step_idx,
                ))
                last = context["last_result"]
                final_result = ToolResult(
                    success=True,
                    output=f"Goal achieved after {step_idx - 1} step(s). {reasoning}",
                    metadata={"steps_taken": step_idx - 1, "history": context["history"]},
                )
                break

            if tool_name == "ABORT":
                _emit(event_callback, _event(
                    "error", "autonomous_aborted",
                    f"LLM aborted: {reasoning}",
                    step=step_idx,
                ))
                final_result = ToolResult(
                    success=False,
                    error=f"Autonomous loop aborted by LLM: {reasoning}",
                    metadata={"steps_taken": step_idx, "history": context["history"]},
                )
                break

            # ── 4. Safety: only allowed tools ──────────────────────── #
            if tool_name not in _ALLOWED_AUTO_TOOLS:
                msg = (
                    f"Tool {tool_name!r} is not permitted in autonomous mode. "
                    f"Allowed: {sorted(_ALLOWED_AUTO_TOOLS)}"
                )
                _emit(event_callback, _event(
                    "error", "tool_not_allowed", msg, step=step_idx,
                ))
                final_result = ToolResult(success=False, error=msg)
                break

            # ── 5. Execute step ─────────────────────────────────────── #
            _emit(event_callback, _event(
                "info", "step_started",
                f"Step {step_idx}: executing {tool_name!r} …",
                step=step_idx,
            ))

            step_result = self._executor.execute(
                tool_name,
                event_callback=event_callback,
                **arguments,
            )

            context["last_result"] = step_result
            context["history"].append({
                "step":      step_idx,
                "tool":      tool_name,
                "arguments": arguments,
                "reasoning": reasoning,
                "success":   step_result.success,
                "output":    str(step_result.output or step_result.error or ""),
            })

            _emit(event_callback, _event(
                "status" if step_result.success else "error",
                "step_completed",
                (
                    f"Step {step_idx} {'✓' if step_result.success else '✗'}: "
                    f"{step_result.output or step_result.error}"
                ),
                step=step_idx,
            ))

            if not step_result.success:
                _emit(event_callback, _event(
                    "error", "autonomous_stopped",
                    f"Step {step_idx} failed.  Stopping autonomous loop.",
                    step=step_idx,
                ))
                final_result = ToolResult(
                    success=False,
                    error=f"Step {step_idx} failed: {step_result.error}",
                    metadata={"steps_taken": step_idx, "history": context["history"]},
                )
                break

            # ── 6. Check goal completion ────────────────────────────── #
            if self._is_goal_achieved(goal, context, event_callback, step_idx):
                _emit(event_callback, _event(
                    "status", "goal_achieved",
                    f"Goal achieved after {step_idx} step(s).",
                    step=step_idx,
                ))
                final_result = ToolResult(
                    success=True,
                    output=f"Goal achieved after {step_idx} step(s).",
                    metadata={"steps_taken": step_idx, "history": context["history"]},
                )
                break

            final_result = ToolResult(
                success=True,
                output=f"Goal completed in {step_idx} step(s).",
                metadata={"steps_taken": step_idx, "history": context["history"]},
            )

        else:
            # loop exhausted without break
            _emit(event_callback, _event(
                "error", "autonomous_stopped",
                f"Max steps ({self._max_steps}) reached.",
            ))

        _emit(event_callback, _event(
            "status" if final_result.success else "error",
            "autonomous_stopped",
            f"Autonomous run finished — success={final_result.success}",
        ))

        return final_result

    # ================================================================== #
    #  Step generation                                                     #
    # ================================================================== #

    def _get_next_action(
        self,
        goal: str,
        context: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Ask the LLM to return the next action as JSON.

        Returns a parsed dict or None on failure.
        """
        context_lines: list[str] = []

        if context["last_result"] is not None:
            r = context["last_result"]
            context_lines.append(f"last_step_success: {r.success}")
            context_lines.append(f"last_step_output:  {r.output or r.error}")

        if context.get("screen_desc"):
            context_lines.append(f"screen_description: {context['screen_desc']}")

        if context["history"]:
            context_lines.append("steps_completed:")
            for h in context["history"]:
                context_lines.append(
                    f"  step {h['step']}: {h['tool']} → "
                    f"{'ok' if h['success'] else 'fail'} — {h['output'][:80]}"
                )

        context_text = "\n".join(context_lines) if context_lines else "(no context yet)"

        user_prompt = _STEP_GEN_USER_TEMPLATE.format(
            goal=goal,
            step=context["step"],
            max_steps=self._max_steps,
            context_text=context_text,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _STEP_GEN_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("AutonomousController._get_next_action Groq call failed: %s", exc)
            return None

        return self._parse_json_response(raw)

    # ================================================================== #
    #  Goal completion check                                               #
    # ================================================================== #

    def _is_goal_achieved(
        self,
        goal: str,
        context: dict[str, Any],
        callback: EventCallback,
        step: int,
    ) -> bool:
        """
        Ask the LLM whether the goal has been achieved.

        Falls back to a heuristic (last step succeeded + ≥1 step done)
        if the LLM call fails.
        """
        last = context["last_result"]
        history = context["history"]

        if not history:
            return False

        history_text = "\n".join(
            f"  step {h['step']}: {h['tool']} → {'ok' if h['success'] else 'fail'} — {h['output'][:80]}"
            for h in history
        )

        user_prompt = _GOAL_CHECK_USER_TEMPLATE.format(
            goal=goal,
            success=last.success if last else False,
            output=str(last.output or last.error or "") if last else "",
            history=history_text,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GOAL_CHECK_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,
                max_tokens=128,
            )
            raw = response.choices[0].message.content or ""
            parsed = self._parse_json_response(raw)
            if parsed and isinstance(parsed.get("achieved"), bool):
                achieved = parsed["achieved"]
                reason   = parsed.get("reason", "")
                _emit(callback, _event(
                    "info", "goal_check",
                    f"Goal check: achieved={achieved}  reason={reason!r}",
                    step=step,
                ))
                return achieved
        except Exception as exc:  # noqa: BLE001
            logger.warning("AutonomousController._is_goal_achieved LLM call failed: %s", exc)

        # Heuristic fallback: goal achieved if last step succeeded
        return bool(last and last.success)

    # ================================================================== #
    #  Perception                                                          #
    # ================================================================== #

    def _observe(self, callback: EventCallback, step: int) -> Optional[str]:
        """
        Capture screen and generate a brief text description.
        Returns None if perception is disabled or capture fails.
        """
        _emit(callback, _event(
            "info", "observation_collected",
            f"Step {step}: capturing screen observation…",
            step=step,
        ))
        try:
            from agent.perception import describe_screen
            description = describe_screen(self._client, self._model)
            _emit(callback, _event(
                "info", "observation_collected",
                f"Step {step}: {description[:100]}",
                step=step,
            ))
            return description
        except Exception as exc:  # noqa: BLE001
            logger.warning("Perception failed at step %d: %s", step, exc)
            return None

    # ================================================================== #
    #  JSON parsing helper                                                 #
    # ================================================================== #

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict[str, Any]]:
        """Extract and parse the first JSON object from *text*."""
        # Strip markdown fences if present
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if brace:
                text = brace.group(0)

        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as exc:
            logger.warning("AutonomousController: JSON parse failed: %s  raw=%r", exc, text[:120])

        return None

    def __repr__(self) -> str:
        return (
            f"<AutonomousController model={self._model!r} "
            f"max_steps={self._max_steps} "
            f"perception={self._use_perception}>"
        )