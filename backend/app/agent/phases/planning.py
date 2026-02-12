"""Planning phase handler."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterator

from app.agent.formatters import format_constraints, format_plan
from app.agent.models import ConversationState, Phase, TravelPlan
from app.agent.prompts import get_phase_prompt
from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.agent.utils import detect_budget_currency, get_current_date_context

if TYPE_CHECKING:
    from app.agent.ai_client import AIClient

logger = logging.getLogger(__name__)


def _gather_planning_research(
    client: "AIClient",
    state: ConversationState,
    search_results: list[str],
    user_interests: list[str],
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
) -> str:
    """Helper to gather planning-specific info via web search."""
    system_prompt = get_phase_prompt("planning", language_code)
    constraints_text = format_constraints(state)
    assumptions_text = ""
    if state.assumptions:
        assumptions_text = "\n\nConfirmed Assumptions:\n"
        for a in state.assumptions.assumptions:
            assumptions_text += f"• {a}\n"

    search_context = ""
    if search_results:
        search_context = "\n\nPrevious research findings:\n" + "\n".join(
            search_results[-3:]
        )

    interests_text = ""
    if user_interests:
        interests_text = "\n\nUser's specific interests to incorporate:\n"
        for interest in user_interests:
            interests_text += f"• {interest}\n"

    date_context = get_current_date_context()
    budget_currency = detect_budget_currency(state)

    research_prompt = f"""Generate a day-by-day itinerary for this trip:

{date_context}

{constraints_text}{assumptions_text}{interests_text}{search_context}

PREVIOUS RESEARCH is provided above. Do NOT re-search for information already available there.

Only search for information NOT already covered. Typical gaps:
- Specific attraction entry fees
- Average meal costs
- Offbeat spots matching interests

IMPORTANT:
- Use the CURRENT YEAR ({datetime.now().year}) in all search queries.
- ALL prices must be in {budget_currency}.

Use web_search to find current prices for gaps only, then return the findings."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": research_prompt},
    ]

    return client.chat_with_tools(
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_executor=execute_tool,
        temperature=0.7,
        max_tool_calls=1,
        on_tool_call=on_tool_call,
    )


def generate_plan(
    client: "AIClient",
    state: ConversationState,
    search_results: list[str],
    user_interests: list[str],
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
) -> str:
    """Generate the travel itinerary (non-streaming)."""
    planning_research = _gather_planning_research(
        client,
        state,
        search_results,
        user_interests,
        on_tool_call=on_tool_call,
        language_code=language_code,
    )
    search_results.append(planning_research)

    system_prompt = get_phase_prompt("planning", language_code)
    constraints_text = format_constraints(state)
    assumptions_text = ""
    if state.assumptions:
        assumptions_text = "\n\nConfirmed Assumptions:\n"
        for a in state.assumptions.assumptions:
            assumptions_text += f"• {a}\n"

    interests_text = ""
    if user_interests:
        interests_text = "\n\nUser's interests:\n" + "\n".join(user_interests)

    budget_currency = detect_budget_currency(state)

    plan_prompt = f"""Create a structured day-by-day itinerary based on this information:

{constraints_text}{assumptions_text}{interests_text}

Research findings (use these for accurate cost estimates):
{planning_research}

CURRENCY: ALL prices MUST be in {budget_currency}."""

    plan_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": plan_prompt},
    ]

    plan = client.chat_structured(plan_messages, TravelPlan, temperature=0.7)
    state.current_plan = plan
    state.phase = Phase.REFINEMENT

    response = format_plan(plan)
    response += "\n\n---\nWant me to tweak anything? I can make it safer, faster, more comfortable, or change the base location. Or if you're happy with it, we're done!"

    state.add_message("assistant", response)
    return response


def generate_plan_stream(
    client: "AIClient",
    state: ConversationState,
    search_results: list[str],
    user_interests: list[str],
    on_tool_call: Callable[[str, dict], None] | None = None,
    language_code: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Generate the travel itinerary with token streaming."""

    # Only do the expensive research phase when we have NO prior search results.
    # Earlier phases (feasibility) already gathered pricing and context.
    if not search_results:
        if on_status:
            on_status("Researching prices and attractions...")
        planning_research = _gather_planning_research(
            client,
            state,
            search_results,
            user_interests,
            on_tool_call=on_tool_call,
            language_code=language_code,
        )
        search_results.append(planning_research)

    if on_status:
        on_status("Writing your itinerary...")

    system_prompt = get_phase_prompt("planning", language_code)
    constraints_text = format_constraints(state)
    assumptions_text = ""
    if state.assumptions:
        assumptions_text = "\n\nConfirmed Assumptions:\n"
        for a in state.assumptions.assumptions:
            assumptions_text += f"• {a}\n"

    interests_text = ""
    if user_interests:
        interests_text = "\n\nUser's interests:\n" + "\n".join(user_interests)

    budget_currency = detect_budget_currency(state)

    # Combine all prior research into the prompt
    research_context = "\n\n".join(search_results[-5:]) if search_results else "No prior research available."

    plan_prompt = f"""Create a detailed day-by-day itinerary based on this information:

{constraints_text}{assumptions_text}{interests_text}

Research findings (use these for accurate cost estimates):
{research_context}

CURRENCY: ALL prices MUST be in {budget_currency}.

Format the itinerary nicely with markdown headers, bullet points, daily totals, and a budget breakdown at the end.

IMPORTANT:
- Do NOT list "Breakfast", "Lunch", or "Dinner" as separate bullet points unless it's a specific famous restaurant or food experience.
- Focus on specific places to visit, things to do, and venues.
- Include clear transport details between locations.
- For each day, include specific tips or notes for the places visited (e.g., best photo spots, hidden gems, or practical advice)."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": plan_prompt},
    ]

    full_response = ""
    # Stream tokens to the client in real-time
    for token in client.chat_stream(messages, temperature=0.7):
        full_response += token
        yield token

    # Update state
    state.phase = Phase.REFINEMENT

    extra = "\n\n---\nWant me to tweak anything? I can make it safer, faster, more comfortable, or change the base location. Or if you're happy with it, we're done!"
    yield extra
    full_response += extra
    state.add_message("assistant", full_response)


