"""Feasibility phase handler."""

import logging
import concurrent.futures
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterator

from app.agent.formatters import format_constraints, format_risk_assessment
from app.agent.models import ConversationState, Phase, RiskAssessment
from app.agent.prompts import get_phase_prompt
from app.agent.utils import get_current_date_context
from app.agent.web_search import web_search

if TYPE_CHECKING:
    from app.agent.ai_client import AIClient

logger = logging.getLogger(__name__)

# Background thread pool for fire-and-forget JSON structuring
_bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _parse_risk_bg(
    client: "AIClient",
    system_prompt: str,
    full_response: str,
    state: ConversationState,
) -> None:
    """Background task: parse streamed text into structured RiskAssessment."""
    try:
        structured_prompt = (
            f"Provide the structured RiskAssessment JSON for: {full_response}"
        )
        risk = client.chat_structured(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": structured_prompt},
            ],
            RiskAssessment,
            temperature=0.1,
        )
        state.risk_assessment = risk
        logger.info("Background risk assessment structuring completed")
    except Exception:
        logger.exception("Background risk assessment structuring failed")


def _quick_high_risk_check(text: str) -> bool:
    """Fast heuristic: check if response text mentions high risk indicators."""
    lowered = text.lower()
    high_indicators = [
        "high risk",
        "strongly advise against",
        "not recommended",
        "dangerous",
        "severe warning",
        "travel advisory",
        "do not travel",
        "extreme caution",
        "life-threatening",
    ]
    return any(indicator in lowered for indicator in high_indicators)


def _gather_research(
    client: "AIClient",
    state: ConversationState,
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
) -> str:
    """Gather current feasibility research via direct web search (Tavily first)."""
    del client, language_code  # Maintained signature compatibility.
    constraints_text = format_constraints(state)
    date_context = get_current_date_context()
    current_year = datetime.now().year
    destination = (
        state.constraints.destination if state.constraints else state.destination
    )
    month_or_season = (
        state.constraints.month_or_season if state.constraints else None
    ) or f"{current_year}"

    if not destination:
        return ""

    queries = [
        (
            "travel_advisory",
            f"{destination} travel advisory restrictions safety {current_year}",
        ),
        (
            "weather",
            f"{destination} weather {month_or_season} {current_year} travel conditions",
        ),
        (
            "infrastructure",
            f"{destination} transport strikes airport disruption infrastructure updates {current_year}",
        ),
    ]

    lines = [
        "Feasibility Research:",
        date_context,
        constraints_text,
    ]

    for label, query in queries:
        if on_tool_call:
            on_tool_call("web_search", {"query": query})
        results = web_search(query, num_results=2)
        if not results or "error" in results[0]:
            lines.append(f"\n[{label}] Query: {query}\n- No reliable results found.")
            continue

        lines.append(f"\n[{label}] Query: {query}")
        for item in results:
            title = item.get("title", "").strip()
            snippet = item.get("snippet", "").strip()
            url = item.get("url", "").strip()
            lines.append(f"- {title}: {snippet} ({url})")

    return "\n".join(lines)


def _append_research_when_ready(
    future: concurrent.futures.Future,
    search_results: list[str],
) -> None:
    """Background callback to persist feasibility research once completed."""
    try:
        result = future.result()
        if result and result not in search_results:
            search_results.append(result)
            logger.info(
                "Feasibility research appended asynchronously for planning reuse."
            )
    except Exception:
        logger.exception("Deferred feasibility research collection failed")


def run_feasibility_check(
    client: "AIClient",
    state: ConversationState,
    search_results: list[str],
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
) -> tuple[str, bool]:
    """Run feasibility check and return risk assessment."""
    search_response = _gather_research(
        client, state, on_tool_call=on_tool_call, language_code=language_code
    )
    search_results.append(search_response)

    system_prompt = get_phase_prompt("feasibility", language_code)
    constraints_text = format_constraints(state)

    assessment_prompt = f"""Based on the information gathered, provide a structured risk assessment for this trip:

{constraints_text}

Research findings:
{search_response}

Provide a risk assessment for each category."""

    assessment_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": assessment_prompt},
    ]

    risk = client.chat_structured(assessment_messages, RiskAssessment, temperature=0.3)
    state.risk_assessment = risk

    response = format_risk_assessment(risk)
    has_high_risk = _check_high_risk(risk)

    if has_high_risk:
        response += "\n\nThis trip has some real risks. Want to go ahead anyway, or should we look at alternatives?"
        state.awaiting_confirmation = True
    else:
        state.phase = Phase.ASSUMPTIONS

    state.add_message("assistant", response)
    return response, has_high_risk


def run_feasibility_check_stream(
    client: "AIClient",
    state: ConversationState,
    search_results: list[str],
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
) -> Iterator[str]:
    """Run feasibility check with token streaming."""
    # Start research immediately so planning can reuse it.
    # Keep first-token latency low by bounding how long we wait here.
    search_response = ""
    research_future = _bg_executor.submit(
        _gather_research,
        client,
        state,
        on_tool_call,
        language_code,
    )
    try:
        search_response = research_future.result(timeout=2)
    except concurrent.futures.TimeoutError:
        logger.info(
            "Feasibility research still running; continuing stream immediately."
        )
        research_future.add_done_callback(
            lambda fut: _append_research_when_ready(fut, search_results)
        )
    except Exception:
        logger.exception("Feasibility research failed; continuing without blocking.")
    if search_response:
        search_results.append(search_response)

    parse_system_prompt = get_phase_prompt("feasibility", language_code)
    system_prompt = parse_system_prompt
    constraints_text = format_constraints(state)

    assessment_prompt = f"""Based on the information gathered, provide a detailed feasibility assessment and risk analysis for this trip:

{constraints_text}

Research findings:
{search_response}

Be specific about weather, route, health, and infrastructure. Include a clear conclusion on whether it's safe and recommended.

OUTPUT FORMAT:
- Return traveler-facing markdown with concise bullets and short paragraphs.
- Highlight key labels in bold where useful.
- Do NOT output raw JSON or schema-shaped key/value objects."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": assessment_prompt},
    ]

    full_response = ""
    for token in client.chat_stream(messages, temperature=0.3):
        full_response += token
        yield token

    # Fire-and-forget: parse risk assessment in background
    _bg_executor.submit(
        _parse_risk_bg, client, parse_system_prompt, full_response, state
    )

    # Quick heuristic check from streamed text (no LLM call needed)
    has_high_risk = _quick_high_risk_check(full_response)

    if has_high_risk:
        extra = "\n\nThis trip has some real risks. Want to go ahead anyway, or should we look at alternatives?"
        state.awaiting_confirmation = True
        yield extra
        full_response += extra
    else:
        state.phase = Phase.ASSUMPTIONS

    state.add_message("assistant", full_response)


def _check_high_risk(risk: RiskAssessment) -> bool:
    """Helper to check if any risk category is HIGH."""
    return any(
        [
            risk.season_weather.value == "HIGH",
            risk.route_accessibility.value == "HIGH",
            risk.altitude_health.value == "HIGH",
            risk.infrastructure.value == "HIGH",
        ]
    )
