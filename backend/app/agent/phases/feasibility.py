"""Feasibility phase handler."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterator, Tuple

from app.agent.formatters import format_constraints, format_risk_assessment
from app.agent.models import ConversationState, Phase, RiskAssessment
from app.agent.prompts import get_phase_prompt
from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.agent.utils import get_current_date_context

if TYPE_CHECKING:
    from app.agent.ai_client import AIClient

logger = logging.getLogger(__name__)


def _gather_research(
    client: "AIClient",
    state: ConversationState,
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
) -> str:
    """Helper to gather current research info via web search."""
    system_prompt = get_phase_prompt("feasibility", language_code)
    constraints_text = format_constraints(state)
    date_context = get_current_date_context()

    search_prompt = f"""You need to evaluate the feasibility of this trip:

{date_context}

{constraints_text}

Before providing your assessment, search for current information about:
1. Current travel advisories or restrictions for the destination
2. Weather/seasonal conditions for the specified travel period
3. Any recent infrastructure or accessibility issues

IMPORTANT: Use the CURRENT YEAR ({datetime.now().year}) in your search queries, not past years.

Use the web_search tool to gather this information, then provide your risk assessment."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": search_prompt},
    ]

    return client.chat_with_tools(
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_executor=execute_tool,
        temperature=0.7,
        max_tool_calls=1,
        on_tool_call=on_tool_call,
    )


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
    on_status: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Run feasibility check with token streaming."""
    if on_status:
        on_status("Analyzing risks...")

    search_response = _gather_research(
        client, state, on_tool_call=on_tool_call, language_code=language_code
    )
    search_results.append(search_response)

    system_prompt = get_phase_prompt("feasibility", language_code)
    constraints_text = format_constraints(state)

    # To provide token-by-token streaming of the assessment, we'll ask the AI
    # to provide a natural language summary based on the research.
    assessment_prompt = f"""Based on the information gathered, provide a detailed feasibility assessment and risk analysis for this trip:

{constraints_text}

Research findings:
{search_response}

Be specific about weather, route, health, and infrastructure. Include a clear conclusion on whether it's safe and recommended."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": assessment_prompt},
    ]

    full_response = ""
    for token in client.chat_stream(messages, temperature=0.3):
        full_response += token
        yield token

    # Yield a transition to mask the pause
    transition = "\n\nJust double-checking the safety details..."
    full_response += transition
    yield transition

    # Show status before blocking step
    if on_status:
        on_status("Verifying details...")

    # After streaming the text, we MUST still do the structured call to update state
    # This happens in the background from the user's perspective.
    structured_prompt = f"Provide the structured RiskAssessment JSON for: {full_response}"
    risk = client.chat_structured(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": structured_prompt},
        ],
        RiskAssessment,
        temperature=0.1,
    )
    state.risk_assessment = risk
    has_high_risk = _check_high_risk(risk)

    if has_high_risk:
        extra = "\n\nThis trip has some real risks. Want to go ahead anyway, or should we look at alternatives?"
        state.awaiting_confirmation = True
        yield extra
        full_response += extra
    else:
        state.phase = Phase.ASSUMPTIONS
        extra = "\n\nEverything looks good to go!"
        yield extra
        full_response += extra

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
