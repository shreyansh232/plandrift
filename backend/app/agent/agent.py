"""TravelAgent - Main orchestrator for the travel planning conversation."""

import logging
from typing import Callable, Optional

from app.agent.ai_client import AIClient, DEFAULT_MODEL
from app.agent.graph import build_agent_graph
from app.agent.models import ConversationState

logger = logging.getLogger(__name__)


class TravelAgent:
    """Orchestrates the constraint-first travel planning conversation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        on_search: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the travel agent.

        Args:
            api_key: OpenRouter API key. Uses OPENROUTER_API_KEY env var if not provided.
            model: Model to use via OpenRouter.
            on_search: Optional callback when a web search is performed.
        """
        self.client = AIClient(api_key=api_key, model=model)
        self.state = ConversationState()
        self.on_search = on_search
        self.search_results: list[str] = []  # Store search results for context
        self.user_interests: list[str] = []  # Store user interests/adjustments
        self._initial_extraction = None
        self.on_status = on_status
        self._last_status: Optional[str] = None
        self._graph = build_agent_graph(self.client, self._handle_tool_call)

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
            elif "transport" in query or "metro" in query or "train" in query or "pass" in query:
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
