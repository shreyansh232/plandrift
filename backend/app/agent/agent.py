"""TravelAgent - Main orchestrator for the travel planning conversation."""

import concurrent.futures
import logging
from typing import Callable, Optional, Iterator

from app.agent.ai_client import AIClient, DEFAULT_MODEL, FAST_MODEL
from app.agent.graph import build_agent_graph
from app.agent.models import ConversationState, Phase
from app.agent.phases import (
    clarification,
    planning,
    refinement,
)
from app.agent.image_search import search_destination_images
from app.agent.flight_search import search_flight_costs
from app.agent.hotel_search import search_hotel_costs
from app.agent.train_search import search_train_costs, should_search_trains

logger = logging.getLogger(__name__)

# Background executor for image search
_img_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


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
        vibe: Optional[str] = None,
    ):
        """Initialize the travel agent.

        Args:
            api_key: OpenRouter API key. Uses OPENROUTER_API_KEY env var if not provided.
            model: Model to use via OpenRouter.
            on_search: Optional callback when a web search is performed.
            language_code: Optional user's preferred language code (e.g., 'fr', 'es').
            vibe: Optional aesthetic/vibe for the trip (e.g., "Cyberpunk", "Wes Anderson").
        """
        self.client = AIClient(api_key=api_key, model=model)
        self.fast_client = AIClient(
            api_key=api_key,
            model=fast_model or DEFAULT_MODEL,
        )
        self.state = ConversationState()
        if vibe:
            self.state.vibe = vibe
        self.on_search = on_search
        self.search_results: list[str] = []  # Store search results for context
        self.user_interests: list[str] = []  # Store user interests/adjustments
        self._initial_extraction = None
        self.on_status = on_status
        self._last_status: Optional[str] = None
        self.language_code = language_code  # Store user's preferred language
        self.destination_images: list[dict] = []
        self._image_search_future: Optional[concurrent.futures.Future] = None
        self.destination_images: list[dict] = []
        self._image_search_future: Optional[concurrent.futures.Future] = None
        self._flight_search_future: Optional[concurrent.futures.Future] = None
        self._hotel_search_future: Optional[concurrent.futures.Future] = None
        self._train_search_future: Optional[concurrent.futures.Future] = None
        self._flight_costs: str = ""
        self._hotel_costs: str = ""
        self._train_costs: dict = {}
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

    def _start_image_search(self, destination: str) -> None:
        """Kick off background image search for the destination."""
        if self._image_search_future is not None:
            return  # Already searching
        logger.info(f"[AGENT] Starting background image search for: {destination}")
        self._image_search_future = _img_executor.submit(
            search_destination_images, destination, 6
        )

    def get_destination_images(self) -> list[dict]:
        """Get cached destination images, waiting up to 5s if search is still running."""
        if self.destination_images:
            return self.destination_images
        if self._image_search_future is None:
            return []
        try:
            self.destination_images = self._image_search_future.result(timeout=5)
        except Exception as e:
            logger.warning(f"[AGENT] Image search failed: {e}")
            self.destination_images = []
        return self.destination_images

    def _start_flight_search(
        self, origin: str, destination: str, date_context: str | None = None
    ) -> None:
        """Kick off background flight cost search."""
        if self._flight_search_future is not None or not origin or not destination:
            return
        msg = f"[AGENT] Starting background flight search: {origin} -> {destination}"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")
        self._flight_search_future = _img_executor.submit(
            search_flight_costs, origin, destination, date_context
        )

    def get_flight_costs(self) -> str:
        """Get cached flight costs, waiting up to 6s if search is still running."""
        if self._flight_costs:
            return self._flight_costs
        if self._flight_search_future is None:
            return ""
        try:
            self._flight_costs = self._flight_search_future.result(timeout=6)
        except Exception as e:
            logger.warning(f"[AGENT] Flight search failed/timeout: {e}")
            self._flight_costs = ""
        return self._flight_costs

    def _start_hotel_search(
        self,
        destination: str,
        date_context: str | None = None,
        budget: str | None = None,
        preferences: str | None = None,
    ) -> None:
        """Kick off background hotel cost search."""
        if self._hotel_search_future is not None or not destination:
            return
        msg = f"[AGENT] Starting background hotel search: {destination}"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")  # Blue color for visibility
        self._hotel_search_future = _img_executor.submit(
            search_hotel_costs, destination, date_context, budget, preferences
        )

    def get_hotel_costs(self) -> str:
        """Get cached hotel costs, waiting up to 6s if search is still running."""
        if self._hotel_costs:
            return self._hotel_costs
        if self._hotel_search_future is None:
            return ""
        try:
            self._hotel_costs = self._hotel_search_future.result(timeout=6)
        except Exception as e:
            logger.warning(f"[AGENT] Hotel search failed/timeout: {e}")
            self._hotel_costs = ""
        return self._hotel_costs

    def _start_train_search(
        self,
        origin: str,
        destination: str,
        date_context: str | None = None,
        budget: str | None = None,
    ) -> None:
        """Kick off background train cost search for Indian routes."""
        if self._train_search_future is not None or not origin or not destination:
            return
        # Only search trains for Indian domestic routes
        if not should_search_trains(origin, destination):
            logger.info("[AGENT] Skipping train search - not an Indian domestic route")
            return
        msg = f"[AGENT] Starting background train search: {origin} -> {destination}"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")  # Blue color for visibility
        self._train_search_future = _img_executor.submit(
            search_train_costs, origin, destination, date_context, budget
        )

    def get_train_costs(self) -> dict:
        """Get cached train costs, waiting up to 6s if search is still running."""
        if self._train_costs:
            return self._train_costs
        if self._train_search_future is None:
            return {}
        try:
            self._train_costs = self._train_search_future.result(timeout=6)
        except Exception as e:
            logger.warning(f"[AGENT] Train search failed/timeout: {e}")
            self._train_costs = {}
        return self._train_costs

    def _warm_price_searches(
        self,
        origin: str | None,
        destination: str | None,
        date_ctx: str | None = None,
        budget: str | None = None,
        preferences: str | None = None,
    ) -> None:
        """Start transport/stay searches as soon as required constraints are known."""
        if origin and destination:
            self._start_flight_search(origin, destination, date_ctx)
            self._start_train_search(origin, destination, date_ctx, budget)
        if destination:
            self._start_hotel_search(destination, date_ctx, budget, preferences)

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
        # Kick off background image search if destination was extracted
        if self.state.destination:
            self._start_image_search(self.state.destination)
        # Kick off flight search if origin and destination are known
        if self.state.origin and self.state.destination:
            # Try to get date context from extraction, or default to generic
            date_ctx = (
                self._initial_extraction.month_or_season
                if self._initial_extraction
                else None
            )
            self._start_flight_search(
                self.state.origin, self.state.destination, date_ctx
            )
            # Also start train search for Indian routes
            budget = (
                self._initial_extraction.budget if self._initial_extraction else None
            )
            self._start_train_search(
                self.state.origin, self.state.destination, date_ctx, budget
            )

        # Kick off hotel search if destination is known
        if self.state.destination:
            # Try to get date/budget context from extraction
            date_ctx = (
                self._initial_extraction.month_or_season
                if self._initial_extraction
                else None
            )
            budget = (
                self._initial_extraction.budget if self._initial_extraction else None
            )
            preferences = (
                " ".join(self._initial_extraction.interests)
                if self._initial_extraction and self._initial_extraction.interests
                else None
            )
            self._start_hotel_search(
                self.state.destination, date_ctx, budget, preferences
            )
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
        train_costs = self.get_train_costs()
        if train_costs and train_costs.get("summary"):
            if train_costs["summary"] not in self.search_results:
                self.search_results.append(train_costs["summary"])
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
        stream = clarification.handle_start_stream(
            self.client, self.state, user_prompt, self.language_code
        )
        started = False
        for token in stream:
            yield token
            # After first token, destination should be extracted — start image search
            if not started:
                if self.state.destination:
                    self._start_image_search(self.state.destination)

                # Also start flight search if we happen to have origin immediately
                if self.state.origin and self.state.destination:
                    date_ctx = (
                        self._initial_extraction.month_or_season
                        if self._initial_extraction
                        else None
                    )
                    self._start_flight_search(
                        self.state.origin, self.state.destination, date_ctx
                    )
                    # Also start train search for Indian routes
                    budget = (
                        self._initial_extraction.budget
                        if self._initial_extraction
                        else None
                    )
                    self._start_train_search(
                        self.state.origin, self.state.destination, date_ctx, budget
                    )

                # Also start hotel search if we happen to have destination immediately
                if self.state.destination:
                    date_ctx = (
                        self._initial_extraction.month_or_season
                        if self._initial_extraction
                        else None
                    )
                    budget = (
                        self._initial_extraction.budget
                        if self._initial_extraction
                        else None
                    )
                    preferences = (
                        " ".join(self._initial_extraction.interests)
                        if self._initial_extraction
                        and self._initial_extraction.interests
                        else None
                    )
                    self._start_hotel_search(
                        self.state.destination, date_ctx, budget, preferences
                    )
                started = True

    def process_clarification_stream(self, answers: str) -> Iterator[str]:
        """Process clarification answers with token streaming."""
        self._emit_status("Analyzing your answers...")

        def _start_searches_on_constraints(constraints) -> None:
            date_ctx = constraints.month_or_season
            budget = constraints.budget or (
                self._initial_extraction.budget if self._initial_extraction else None
            )
            preferences: list[str] = []
            if constraints.interests:
                preferences.extend(constraints.interests)
            if constraints.vibe:
                preferences.append(constraints.vibe)
            pref_str = " ".join(preferences) if preferences else None
            self._warm_price_searches(
                constraints.origin,
                constraints.destination,
                date_ctx=date_ctx,
                budget=budget,
                preferences=pref_str,
            )

        stream = clarification.process_clarification_stream(
            self.fast_client,
            self.state,
            answers,
            self._initial_extraction,
            self.language_code,
            search_results=self.search_results,
            on_constraints_extracted=_start_searches_on_constraints,
        )
        for token in stream:
            yield token

        # After clarification, check for origin/dest in constraints or extraction
        origin = self.state.origin or (
            self.state.constraints and self.state.constraints.origin
        )
        destination = self.state.destination or (
            self.state.constraints and self.state.constraints.destination
        )

        if origin and destination:
            # Get date context
            date_ctx = None
            if self.state.constraints and self.state.constraints.month_or_season:
                date_ctx = self.state.constraints.month_or_season
            elif self._initial_extraction and self._initial_extraction.month_or_season:
                date_ctx = self._initial_extraction.month_or_season

            # Also start train search for Indian routes
            budget = None
            if self.state.constraints and self.state.constraints.budget:
                budget = self.state.constraints.budget
            elif self._initial_extraction and self._initial_extraction.budget:
                budget = self._initial_extraction.budget
            self._start_flight_search(origin, destination, date_ctx)
            self._start_train_search(origin, destination, date_ctx, budget)

        # Trigger hotel search if we have destination (possibly from clarification)
        if destination:
            date_ctx = None
            if self.state.constraints and self.state.constraints.month_or_season:
                date_ctx = self.state.constraints.month_or_season
            elif self._initial_extraction and self._initial_extraction.month_or_season:
                date_ctx = self._initial_extraction.month_or_season

            budget = None
            if self.state.constraints and self.state.constraints.budget:
                budget = self.state.constraints.budget
            elif self._initial_extraction and self._initial_extraction.budget:
                budget = self._initial_extraction.budget

            preferences = []
            if self.state.constraints and self.state.constraints.interests:
                preferences.extend(self.state.constraints.interests)
            if self.state.constraints and self.state.constraints.vibe:
                preferences.append(self.state.constraints.vibe)

            pref_str = " ".join(preferences) if preferences else None

            self._warm_price_searches(
                origin,
                destination,
                date_ctx=date_ctx,
                budget=budget,
                preferences=pref_str,
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
        import time

        if not confirmed and not modifications:
            yield "Please let me know what changes you'd like to make."
            return

        # Wait-guard: background assumptions parse may still be running
        if not self.state.assumptions:
            for _ in range(20):  # Wait up to 10s
                if self.state.assumptions:
                    break
                time.sleep(0.5)

        # If user has modifications or additional interests, we might need more research
        if modifications or additional_interests:
            self._emit_status("Researching your preferences...")
            interests = f"{modifications or ''} {additional_interests or ''}".strip()
            self.user_interests.append(interests)

        self._emit_status("Creating your itinerary...")
        self.state.phase = Phase.PLANNING

        # Get train costs and add to search results if available
        train_costs = self.get_train_costs()
        if train_costs and train_costs.get("summary"):
            if train_costs["summary"] not in self.search_results:
                self.search_results.append(train_costs["summary"])

        yield from planning.generate_plan_stream(
            self.client,
            self.state,
            self.search_results,
            self.user_interests,
            on_tool_call=self._handle_tool_call,
            language_code=self.language_code,
            flight_costs=self.get_flight_costs(),
            hotel_costs=self.get_hotel_costs(),
            train_costs=train_costs,
        )

    def refine_plan_stream(self, refinement_type: str) -> Iterator[str]:
        """Refine the plan with token streaming."""
        self._emit_status(f"Making it {refinement_type}...")
        yield from refinement.refine_plan_stream(
            self.client, self.state, refinement_type, self.language_code
        )
