"""LangGraph orchestration for the travel agent."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.ai_client import AIClient
from app.agent.models import ConversationState, InitialExtraction, Phase
from app.agent.phases import (
    assumptions,
    clarification,
    feasibility,
    planning,
    refinement,
)
from app.agent.sanitizer import sanitize_input

logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict):
    action: str
    input: dict[str, Any]
    agent_state: ConversationState
    search_results: list[str]
    user_interests: list[str]
    initial_extraction: Optional[InitialExtraction]
    response: str
    has_high_risk: bool


def build_agent_graph(
    client: AIClient,
    fast_client: AIClient,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
):
    """Build the LangGraph workflow used by the TravelAgent."""

    def route_action(state: AgentGraphState) -> str:
        return state["action"]

    def node_start(state: AgentGraphState) -> AgentGraphState:
        prompt = state["input"].get("prompt", "")
        response, extracted = clarification.handle_start(
            client, state["agent_state"], prompt
        )
        state["response"] = response
        state["initial_extraction"] = extracted
        state["has_high_risk"] = False
        return state

    def node_clarify(state: AgentGraphState) -> AgentGraphState:
        answers = state["input"].get("answers", "")
        constraints = clarification.process_clarification(
            client,
            state["agent_state"],
            answers,
            state.get("initial_extraction"),
        )
        state["agent_state"].constraints = constraints
        state["agent_state"].phase = Phase.FEASIBILITY

        response, has_high_risk = feasibility.run_feasibility_check(
            fast_client,
            state["agent_state"],
            state["search_results"],
            on_tool_call,
        )
        state["response"] = response
        state["has_high_risk"] = has_high_risk
        return state

    def node_proceed(state: AgentGraphState) -> AgentGraphState:
        proceed = bool(state["input"].get("proceed", True))
        agent_state = state["agent_state"]
        awaiting = agent_state.awaiting_confirmation
        agent_state.awaiting_confirmation = False

        if awaiting and not proceed:
            state["response"] = (
                "Totally fair. You might want to check out the alternatives "
                "I mentioned, or we can adjust your dates/destination. What do you think?"
            )
            return state

        if not proceed and not awaiting:
            state["response"] = (
                "No problem. Share any changes you'd like and I'll recalibrate."
            )
            return state

        agent_state.phase = Phase.ASSUMPTIONS
        state["response"] = assumptions.generate_assumptions(
            fast_client, agent_state
        )
        state["has_high_risk"] = False
        return state

    def node_assumptions(state: AgentGraphState) -> AgentGraphState:
        confirmed = bool(state["input"].get("confirmed", True))
        modifications = state["input"].get("modifications")
        additional_interests = state["input"].get("additional_interests")
        agent_state = state["agent_state"]
        agent_state.awaiting_confirmation = False

        user_modifications = modifications
        if additional_interests:
            user_modifications = (
                f"{user_modifications or ''}\nAdditional interests: {additional_interests}"
            )

        if not confirmed and user_modifications:
            result = sanitize_input(user_modifications)
            user_modifications = result.text
            if result.injection_detected:
                logger.warning(
                    "Possible prompt injection in modifications: %s", result.flags
                )

            state["user_interests"].append(user_modifications)
            if agent_state.constraints:
                agent_state.constraints.interests.append(user_modifications)

            agent_state.add_message(
                "user", f"Adjustments needed: {user_modifications}"
            )

            search_results = assumptions.search_for_interests(
                fast_client, agent_state, user_modifications, on_tool_call
            )
            if search_results:
                state["search_results"].append(search_results)

            assumptions.update_assumptions_with_interests(
                fast_client, agent_state, user_modifications, state["search_results"]
            )

        agent_state.phase = Phase.PLANNING
        state["response"] = planning.generate_plan(
            client,
            agent_state,
            state["search_results"],
            state["user_interests"],
            on_tool_call,
        )
        state["has_high_risk"] = False
        return state

    def node_refine(state: AgentGraphState) -> AgentGraphState:
        refinement_type = state["input"].get("refinement_type", "")
        state["response"] = refinement.refine_plan(
            client, state["agent_state"], refinement_type
        )
        state["has_high_risk"] = False
        return state

    graph = StateGraph(AgentGraphState)
    graph.add_node("router", lambda s: s)
    graph.add_node("start", node_start)
    graph.add_node("clarify", node_clarify)
    graph.add_node("proceed", node_proceed)
    graph.add_node("assumptions", node_assumptions)
    graph.add_node("refine", node_refine)

    graph.add_conditional_edges(
        "router",
        route_action,
        {
            "start": "start",
            "clarify": "clarify",
            "proceed": "proceed",
            "assumptions": "assumptions",
            "refine": "refine",
        },
    )

    graph.add_edge("start", END)
    graph.add_edge("clarify", END)
    graph.add_edge("proceed", END)
    graph.add_edge("assumptions", END)
    graph.add_edge("refine", END)

    graph.set_entry_point("router")
    return graph.compile()
