"""ReAct-style agent loop over RAG context + tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from smartops.agent.tools import TOOL_REGISTRY, run_tool, tool_descriptions
from smartops.core.async_utils import run_sync
from smartops.core.logging import get_logger
from smartops.llm.client import LLMClient
from smartops.llm.prompts import FINAL_ANSWER_PROMPT, REACT_SYSTEM, REACT_USER
from smartops.rag.retriever import Retriever, format_context
from smartops.rag.store import RetrievedChunk

logger = get_logger(__name__)

ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*Action:\s*(?P<action>[A-Za-z_]+)\s*Action Input:\s*(?P<input>.*)",
    re.IGNORECASE | re.DOTALL,
)

# Tolerant patterns for noisy model output (markdown fences, bullets, casing)
ACTION_LOOSE_RE = re.compile(
    r"(?:\*{0,2}|#{1,3}\s*|[-*]\s*)?Action\*{0,2}\s*[:\-]\s*`?(?P<action>[A-Za-z_]+)`?",
    re.IGNORECASE,
)
INPUT_LOOSE_RE = re.compile(
    r"(?:\*{0,2}|#{1,3}\s*|[-*]\s*)?Action\s*Input\*{0,2}\s*[:\-]\s*`?(?P<input>[^\n`]*)`?",
    re.IGNORECASE,
)
THOUGHT_LOOSE_RE = re.compile(
    r"(?:\*{0,2}|#{1,3}\s*|[-*]\s*)?Thought\*{0,2}\s*[:\-]\s*(?P<thought>[^\n]+)",
    re.IGNORECASE,
)

HOST_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|local|internal|dev|test)\b",
    re.IGNORECASE,
)

FINISH_ACTIONS = {"finish", "final", "answer", "none"}


@dataclass
class AgentTrace:
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    used_fallback_llm: bool = False
    steps: int = 0


@dataclass
class AgentResult:
    answer: str
    trace: AgentTrace


class ReActAgent:
    def __init__(self, llm: LLMClient, retriever: Retriever, max_steps: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max(1, int(max_steps))

    async def run(self, question: str, model: str, top_k: int) -> AgentResult:
        trace = AgentTrace()
        # Chroma / embedding encode is blocking — keep it off the event loop
        chunks = await run_sync(self.retriever.retrieve, question, top_k)
        trace.retrieved = chunks
        context = format_context(chunks)

        system = REACT_SYSTEM.format(tool_descriptions=tool_descriptions())
        user = REACT_USER.format(context=context, question=question)
        react_prompt = f"{system}\n\n{user}"
        seen_calls: set[tuple[str, str]] = set()

        for step in range(self.max_steps):
            trace.steps = step + 1
            raw, fallback = await self.llm.generate(model, react_prompt, temperature=0.1)
            trace.used_fallback_llm = trace.used_fallback_llm or fallback
            parsed = self._parse_react(raw)

            if parsed is None:
                logger.warning("react_parse_failed", raw=raw[:400])
                break

            thought, action, action_input = parsed
            action_norm = action.lower().strip()
            action_input = self._normalize_action_input(action_norm, action_input, question)

            trace.thought = thought
            trace.action = action_norm
            trace.action_input = action_input

            if action_norm in FINISH_ACTIONS:
                break

            if action_norm not in TOOL_REGISTRY:
                logger.warning("react_unknown_action", action=action_norm)
                break

            # Guard: never run check_server_status without a hostname
            if action_norm == "check_server_status" and not action_input:
                logger.info("react_skip_status_without_host")
                break

            call_key = (action_norm, action_input)
            if call_key in seen_calls:
                logger.info("react_duplicate_tool_skipped", action=action_norm, input=action_input)
                break
            seen_calls.add(call_key)

            tool_result = run_tool(action_norm, action_input)
            trace.tool_results.append(
                {
                    "name": tool_result.name,
                    "input": tool_result.input,
                    "ok": tool_result.ok,
                    "output": tool_result.output,
                }
            )

            observation = json.dumps(tool_result.output, ensure_ascii=True)
            react_prompt = (
                f"{react_prompt}\n\nAssistant:\nThought: {thought}\n"
                f"Action: {action_norm}\nAction Input: {action_input or 'none'}\n\n"
                f"Observation: {observation}\n\n"
                "You now have enough information. Reply with Action: finish and Action Input: none."
            )

        tool_blob = (
            json.dumps(trace.tool_results, indent=2, ensure_ascii=True)
            if trace.tool_results
            else "No tools were used."
        )
        final_prompt = FINAL_ANSWER_PROMPT.format(
            context=context,
            tool_results=tool_blob,
            question=question,
        )
        answer, fallback = await self.llm.generate(model, final_prompt, temperature=0.2)
        trace.used_fallback_llm = trace.used_fallback_llm or fallback

        return AgentResult(answer=answer, trace=trace)

    @staticmethod
    def _normalize_action_input(action: str, action_input: str, question: str) -> str:
        value = (action_input or "").strip()
        if value.lower() in {"none", "null", "n/a", "-"}:
            value = ""

        # Strip quotes / trailing punctuation from model output
        value = value.strip().strip("\"'`").rstrip(".,;")

        if action == "check_server_status":
            if value and HOST_RE.search(value):
                match = HOST_RE.search(value)
                return match.group(0).lower() if match else value.lower()
            # Recover hostname from the user question when model forgot Action Input
            match = HOST_RE.search(question or "")
            return match.group(0).lower() if match else ""

        return value

    @staticmethod
    def _strip_fences(text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    @classmethod
    def _parse_react(cls, text: str) -> tuple[str, str, str] | None:
        cleaned = cls._strip_fences(text or "")
        match = ACTION_RE.search(cleaned)
        if match:
            return (
                match.group("thought").strip(),
                match.group("action").strip(),
                match.group("input").strip().splitlines()[0].strip(),
            )

        action_m = ACTION_LOOSE_RE.search(cleaned) or re.search(
            r"Action:\s*([A-Za-z_]+)", cleaned, re.I
        )
        if not action_m:
            return None
        input_m = INPUT_LOOSE_RE.search(cleaned) or re.search(
            r"Action Input:\s*(.*)", cleaned, re.I
        )
        thought_m = THOUGHT_LOOSE_RE.search(cleaned) or re.search(
            r"Thought:\s*(.*)", cleaned, re.I
        )
        action_gd = action_m.groupdict()
        action = action_gd.get("action") or action_m.group(1)
        action_input = "none"
        if input_m:
            input_gd = input_m.groupdict()
            action_input = input_gd.get("input") or input_m.group(1)
        thought = ""
        if thought_m:
            thought_gd = thought_m.groupdict()
            thought = thought_gd.get("thought") or thought_m.group(1)
        return (thought.strip(), str(action).strip(), (action_input or "none").strip())
