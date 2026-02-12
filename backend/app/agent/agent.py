"""TravelAgent - Main orchestrator for the travel planning conversation."""

import logging
from typing import Callable, Optional, Iterator

from app.agent.ai_client import AIClient, DEFAULT_MODEL, FAST_MODEL
from app.agent.graph import build_agent_graph
from app.agent.models import ConversationState, Phase
from app.agent.phases import (
    clarification,
    feasibility,
    planning,
    refinement,
)

logger = logging.getLogger(__name__)


class TravelAgent:
    """Orchestrates the constraint-first travel planning conversation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        fast_model: Optional[str] = FAST_MODEL,
        on_search: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        language_code: Optional[str] = None,
    ):
        """Initialize the travel agent.

        Args:
            api_key: OpenRouter API key. Uses OPENROUTER_API_KEY env var if not provided.
            model: Model to use via OpenRouter.
            on_search: Optional callback when a web search is performed.
            language_code: Optional user's preferred language code (e.g., 'fr', 'es').
        """
        self.client = AIClient(api_key=api_key, model=model)
        self.fast_client = AIClient(
            api_key=api_key,
            model=fast_model or DEFAULT_MODEL,
        )
        self.state = ConversationState()
        self.on_search = on_search
        self.search_results: list[str] = []  # Store search results for context
        self.user_interests: list[str] = []  # Store user interests/adjustments
        self._initial_extraction = None
        self.on_status = on_status
        self._last_status: Optional[str] = None
        self.language_code = language_code  # Store user's preferred language
        self._graph = build_agent_graph(
            self.client, self.fast_client, self._handle_tool_call, language_code
        )

    def _run_graph(self, action: str, payload: dict) -> dict:
        state = {
            "action": action,
            "input": payload,
            "agent_state": self.state,
            "search_results": self.search_results,
            "user_interests": self.user_interests,
            "initial_extraction": self._initial_extraction,
            "response": "",
            "has_high_risk": False,
            "language_code": self.language_code,
        }
        result = self._graph.invoke(state)
        self.state = result["agent_state"]
        self.search_results = result["search_results"]
        self.user_interests = result["user_interests"]
        self._initial_extraction = result.get("initial_extraction")
        return result

    def _emit_status(self, message: str) -> None:
        if not self.on_status:
            return
        if message == self._last_status:
            return
        self._last_status = message
        self.on_status(message)

    def _handle_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Handle tool call notifications."""
        if tool_name == "web_search" and self.on_search:
            query = arguments.get("query", "")
            self.on_search(query)
        if tool_name == "web_search":
            query = (arguments.get("query", "") or "").lower()
            if "flight" in query or "flights" in query:
                self._emit_status("Searching flights...")
            elif "hotel" in query or "hostel" in query:
                self._emit_status("Finding stays...")
            elif (
                "transport" in query
                or "metro" in query
                or "train" in query
                or "pass" in query
            ):
                self._emit_status("Estimating local transport...")
            elif "meal" in query or "food" in query:
                self._emit_status("Estimating meal costs...")
            elif "entry fee" in query or "ticket" in query or "attraction" in query:
                self._emit_status("Checking activity costs...")
            else:
                self._emit_status("Researching local details...")

    def start(self, user_prompt: str) -> str:
        """Start a new travel planning conversation.

        Extracts everything possible from the initial prompt and only asks
        for what's still missing.

        Args:
            user_prompt: User's initial prompt (e.g., "Plan a trip from Mumbai to Japan in March, 7 days, solo").

        Returns:
            Clarification questions for missing info, or request for origin/destination.
        """
        result = self._run_graph("start", {"prompt": user_prompt})
        return result["response"]

    def process_clarification(self, answers: str) -> tuple[str, bool]:
        """Process user's answers to clarification questions.

        Merges answers with any info already extracted from the initial prompt.

        Args:
            answers: User's answers to the clarification questions.

        Returns:
            Tuple of (response text, has_high_risk).
        """
        result = self._run_graph("clarify", {"answers": answers})
        return result["response"], bool(result.get("has_high_risk"))

    def confirm_proceed(self, proceed: bool) -> str:
        """Handle user's decision to proceed despite high risk.

        Args:
            proceed: Whether user wants to proceed despite risks.

        Returns:
            Next phase response.
        """
        result = self._run_graph("proceed", {"proceed": proceed})
        return result["response"]

    def proceed_to_assumptions(self) -> str:
        """Move to assumptions phase after feasibility check."""
        result = self._run_graph("proceed", {"proceed": True})
        return result["response"]

    def _generate_assumptions(self) -> str:
        """Generate and present assumptions before planning.

        Returns:
            Assumptions text for user confirmation.
        """
        result = self._run_graph("proceed", {"proceed": True})
        return result["response"]

    def confirm_assumptions(
        self,
        confirmed: bool,
        adjustments: Optional[str] = None,
        modifications: Optional[str] = None,
        additional_interests: Optional[str] = None,
    ) -> str:
        """Handle user's confirmation of assumptions.

        Args:
            confirmed: Whether user confirms the assumptions.
            adjustments: Any adjustments the user wants to make (deprecated, use modifications).
            modifications: Any modifications the user wants to make.
            additional_interests: Additional interests to incorporate.

        Returns:
            Generated plan or request for clarification.
        """
        self._emit_status("Researching current prices...")
        result = self._run_graph(
            "assumptions",
            {
                "confirmed": confirmed,
                "modifications": modifications or adjustments,
                "additional_interests": additional_interests,
            },
        )
        return result["response"]

    def _generate_plan(self) -> str:
        """Generate the travel itinerary.

        Returns:
            Day-by-day travel plan.
        """
        self._emit_status("Researching current prices...")
        result = self._run_graph("assumptions", {"confirmed": True})
        return result["response"]

    def refine_plan(self, refinement_type: str) -> str:
        """Refine the plan based on user's choice.

        Args:
            refinement_type: Type of refinement requested.

        Returns:
            Refined plan.
        """
        result = self._run_graph("refine", {"refinement_type": refinement_type})
        return result["response"]

    # ==================== STREAMING METHODS ====================

    def start_stream(self, user_prompt: str) -> Iterator[str]:
        """Start a new trip planning conversation with token streaming."""
        self._emit_status("Understanding your request...")
        return clarification.handle_start_stream(
            self.client, self.state, user_prompt, self.language_code
        )

    def process_clarification_stream(self, answers: str) -> Iterator[str]:
        """Process clarification answers with token streaming."""
        self._emit_status("Analyzing your answers...")
        return clarification.process_clarification_stream(
            self.client,
            self.state,
            answers,
            self._initial_extraction,
            self.language_code,
            on_status=self._emit_status,
        )

    def confirm_proceed_stream(self, proceed: bool) -> Iterator[str]:
        """Handle proceed decision with token streaming."""
        from app.agent.phases import assumptions

        if not proceed:
            yield "Trip planning cancelled. Let me know if you'd like to try something else."
            return

        self.state.phase = Phase.ASSUMPTIONS
        self._emit_status("Planning your trip...")

        yield from assumptions.generate_assumptions_stream(
            self.fast_client, self.state, self.language_code
        )

    def confirm_assumptions_stream(
        self,
        confirmed: bool,
        modifications: Optional[str] = None,
        additional_interests: Optional[str] = None,
    ) -> Iterator[str]:
        """Handle assumptions confirmation with token streaming."""
        from app.agent.phases import assumptions

        if not confirmed and not modifications:
            yield "Please let me know what changes you'd like to make."
            return

        # If user has modifications or additional interests, we might need more research
        if modifications or additional_interests:
            self._emit_status("Researching your preferences...")
            interests = f"{modifications or ''} {additional_interests or ''}".strip()
            self.user_interests.append(interests)

            # In streaming mode, we'll first generate updated assumptions then proceed to plan
            # Or we can just jump to plan generation with these interests incorporated.
            # Let's jump to plan generation for a smoother experience.

        self._emit_status("Creating your itinerary...")
        self.state.phase = Phase.PLANNING
        yield from planning.generate_plan_stream(
            self.client,
            self.state,
            self.search_results,
            self.user_interests,
            on_tool_call=self._handle_tool_call,
            language_code=self.language_code,
            on_status=self._emit_status,
        )

    def refine_plan_stream(self, refinement_type: str) -> Iterator[str]:
        """Refine the plan with token streaming."""
        self._emit_status(f"Making it {refinement_type}...")
        return refinement.refine_plan_stream(
            self.client, self.state, refinement_type, self.language_code
        )
