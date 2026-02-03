"""TravelAgent - Main orchestrator for the travel planning conversation."""

from datetime import datetime
from typing import Callable, Optional

from app.agent.models import (
    Assumptions,
    ConversationState,
    Phase,
    RiskAssessment,
    TravelConstraints,
    TravelPlan,
)
from app.agent.openai_client import OpenAIClient
from app.agent.prompts import get_phase_prompt
from app.agent.tools import TOOL_DEFINITIONS, execute_tool


class TravelAgent:
    """Orchestrates the constraint-first travel planning conversation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        on_search: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the travel agent.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            model: OpenAI model to use.
            on_search: Optional callback when a web search is performed.
        """
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.state = ConversationState()
        self.on_search = on_search
        self.search_results: list[str] = []  # Store search results for context
        self.user_interests: list[str] = []  # Store user interests/adjustments

    def _get_current_date_context(self) -> str:
        """Get current date context for prompts."""
        now = datetime.now()
        return f"Today's date: {now.strftime('%B %d, %Y')} (Year: {now.year})"

    def _handle_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Handle tool call notifications."""
        if tool_name == "web_search" and self.on_search:
            query = arguments.get("query", "")
            self.on_search(query)

    def start(self, origin: str, destination: str) -> str:
        """Start a new travel planning conversation.

        Args:
            origin: The starting location (traveling from).
            destination: The travel destination (traveling to).

        Returns:
            Clarification questions to ask the user.
        """
        self.state = ConversationState()
        self.state.origin = origin
        self.state.destination = destination
        self.state.phase = Phase.CLARIFICATION

        # Build initial prompt
        system_prompt = get_phase_prompt("clarification")
        user_message = f"I want to plan a trip from {origin} to {destination}."

        self.state.add_message("system", system_prompt)
        self.state.add_message("user", user_message)

        # Get clarification questions
        messages = self.state.get_openai_messages()
        response = self.client.chat(messages, temperature=0.3)

        self.state.add_message("assistant", response)
        return response

    def process_clarification(self, answers: str) -> tuple[str, bool]:
        """Process user's answers to clarification questions.

        Args:
            answers: User's answers to the clarification questions.

        Returns:
            Tuple of (response text, needs_confirmation).
            If needs_confirmation is True, user should confirm before proceeding.
        """
        self.state.add_message("user", answers)

        # Extract constraints from answers
        extraction_prompt = f"""Based on the user's answers, extract their travel constraints.
User's origin: {self.state.origin}
User's destination: {self.state.destination}
User's answers: {answers}

Extract the following if mentioned:
- Month or season of travel
- Total trip duration (days)
- Solo or group travel
- Budget level
- Comfort with rough conditions"""

        messages = [
            {
                "role": "system",
                "content": "Extract travel constraints from user input.",
            },
            {"role": "user", "content": extraction_prompt},
        ]

        constraints = self.client.chat_structured(
            messages, TravelConstraints, temperature=0.1
        )
        constraints.origin = self.state.origin
        constraints.destination = self.state.destination
        self.state.constraints = constraints

        # Move to feasibility phase
        self.state.phase = Phase.FEASIBILITY
        return self._run_feasibility_check()

    def _run_feasibility_check(self) -> tuple[str, bool]:
        """Run feasibility check and return risk assessment.

        Returns:
            Tuple of (response text, has_high_risk).
        """
        system_prompt = get_phase_prompt("feasibility")

        constraints_text = self._format_constraints()

        # First, gather current information via web search
        date_context = self._get_current_date_context()
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

        # Use chat with tools to gather current information
        search_response = self.client.chat_with_tools(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            temperature=0.3,
            on_tool_call=self._handle_tool_call,
        )

        # Store search context for later phases
        self.search_results.append(search_response)

        # Now get structured risk assessment with the gathered information
        assessment_prompt = f"""Based on the information gathered, provide a structured risk assessment for this trip:

{constraints_text}

Research findings:
{search_response}

Provide a risk assessment for each category."""

        assessment_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": assessment_prompt},
        ]

        risk = self.client.chat_structured(
            assessment_messages, RiskAssessment, temperature=0.3
        )
        self.state.risk_assessment = risk

        # Format response
        response = self._format_risk_assessment(risk)

        has_high_risk = any(
            [
                risk.season_weather.value == "HIGH",
                risk.route_accessibility.value == "HIGH",
                risk.altitude_health.value == "HIGH",
                risk.infrastructure.value == "HIGH",
            ]
        )

        if has_high_risk:
            response += "\n\n⚠️ HIGH RISK DETECTED. Do you want to proceed anyway, or consider alternatives?"
            self.state.awaiting_confirmation = True
        else:
            # Auto-proceed to assumptions if no high risk
            self.state.phase = Phase.ASSUMPTIONS

        self.state.add_message("assistant", response)
        return response, has_high_risk

    def confirm_proceed(self, proceed: bool) -> str:
        """Handle user's decision to proceed despite high risk.

        Args:
            proceed: Whether user wants to proceed despite risks.

        Returns:
            Next phase response.
        """
        self.state.awaiting_confirmation = False

        if not proceed:
            return "Understood. Consider the safer alternatives mentioned above, or let me know if you'd like to modify your constraints."

        self.state.phase = Phase.ASSUMPTIONS
        return self._generate_assumptions()

    def proceed_to_assumptions(self) -> str:
        """Move to assumptions phase after feasibility check."""
        self.state.phase = Phase.ASSUMPTIONS
        return self._generate_assumptions()

    def _generate_assumptions(self) -> str:
        """Generate and present assumptions before planning.

        Returns:
            Assumptions text for user confirmation.
        """
        system_prompt = get_phase_prompt("assumptions")

        constraints_text = self._format_constraints()
        risk_text = ""
        if self.state.risk_assessment:
            risk_text = f"\nRisk Assessment: Overall feasible = {self.state.risk_assessment.overall_feasible}"

        user_message = f"""Based on these constraints, list the assumptions for planning:

{constraints_text}{risk_text}

List all assumptions explicitly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        assumptions = self.client.chat_structured(
            messages, Assumptions, temperature=0.3
        )
        self.state.assumptions = assumptions

        response = "**Assumptions for Planning:**\n\n"
        for assumption in assumptions.assumptions:
            response += f"• {assumption}\n"

        if assumptions.uncertain_assumptions:
            response += "\n**Uncertain (please confirm):**\n"
            for uncertain in assumptions.uncertain_assumptions:
                response += f"• [?] {uncertain}\n"

        response += "\nPlease confirm these assumptions are correct, or let me know what to adjust."
        self.state.awaiting_confirmation = True
        self.state.add_message("assistant", response)
        return response

    def confirm_assumptions(
        self, confirmed: bool, adjustments: Optional[str] = None, modifications: Optional[str] = None, additional_interests: Optional[str] = None
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
        self.state.awaiting_confirmation = False

        # Use modifications if provided, otherwise use adjustments for backward compatibility
        user_modifications = modifications or adjustments
        if additional_interests:
            user_modifications = f"{user_modifications or ''}\nAdditional interests: {additional_interests}"

        if not confirmed and user_modifications:
            # Store user interests/adjustments for later use
            self.user_interests.append(user_modifications)
            if self.state.constraints:
                self.state.constraints.interests.append(user_modifications)

            self.state.add_message("user", f"Adjustments needed: {user_modifications}")

            # Search for events/activities based on user interests
            search_results = self._search_for_interests(user_modifications)
            if search_results:
                self.search_results.append(search_results)

            # Re-generate assumptions with adjustments
            return self._generate_assumptions_with_interests(user_modifications)

        self.state.phase = Phase.PLANNING
        return self._generate_plan()

    def _search_for_interests(self, interests: str) -> str:
        """Search for events/activities based on user interests.

        Args:
            interests: User's stated interests.

        Returns:
            Search results for the interests.
        """
        destination = self.state.destination or ""
        month = ""
        if self.state.constraints and self.state.constraints.month_or_season:
            month = self.state.constraints.month_or_season

        date_context = self._get_current_date_context()
        search_prompt = f"""The user wants to find specific activities/events at their destination.

{date_context}

Destination: {destination}
Travel period: {month}
User interests: {interests}

Search for:
1. Upcoming events matching their interests (conferences, meetups, festivals, etc.)
2. Popular venues or locations for these activities
3. Booking requirements or ticket prices

IMPORTANT: Use the CURRENT YEAR ({datetime.now().year}) in your search queries. Search for events in {datetime.now().year}, not past years.

Use web_search to find current/upcoming events and activities."""

        messages = [
            {
                "role": "system",
                "content": "You are a travel research assistant. Search for events and activities matching user interests.",
            },
            {"role": "user", "content": search_prompt},
        ]

        return self.client.chat_with_tools(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            temperature=0.3,
            on_tool_call=self._handle_tool_call,
        )

    def _generate_assumptions_with_interests(self, interests: str) -> str:
        """Generate assumptions incorporating user's stated interests.

        Args:
            interests: User's stated interests.

        Returns:
            Assumptions text for user confirmation.
        """
        system_prompt = get_phase_prompt("assumptions")

        constraints_text = self._format_constraints()
        risk_text = ""
        if self.state.risk_assessment:
            risk_text = f"\nRisk Assessment: Overall feasible = {self.state.risk_assessment.overall_feasible}"

        # Include search results for interests
        interest_research = ""
        if self.search_results:
            interest_research = (
                f"\n\nResearch on user interests:\n{self.search_results[-1]}"
            )

        user_message = f"""Based on these constraints and the user's specific interests, list the assumptions for planning:

{constraints_text}{risk_text}

USER'S SPECIFIC INTERESTS (MUST incorporate):
{interests}
{interest_research}

IMPORTANT: The user specifically mentioned these interests. You MUST include assumptions about incorporating these into the plan.

List all assumptions explicitly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        assumptions = self.client.chat_structured(
            messages, Assumptions, temperature=0.3
        )
        self.state.assumptions = assumptions

        response = "**Assumptions for Planning:**\n\n"
        for assumption in assumptions.assumptions:
            response += f"• {assumption}\n"

        if assumptions.uncertain_assumptions:
            response += "\n**Uncertain (please confirm):**\n"
            for uncertain in assumptions.uncertain_assumptions:
                response += f"• [?] {uncertain}\n"

        response += "\nPlease confirm these assumptions are correct, or let me know what to adjust."
        self.state.awaiting_confirmation = True
        self.state.add_message("assistant", response)
        return response

    def _generate_plan(self) -> str:
        """Generate the travel itinerary.

        Returns:
            Day-by-day travel plan.
        """
        system_prompt = get_phase_prompt("planning")

        constraints_text = self._format_constraints()
        assumptions_text = ""
        if self.state.assumptions:
            assumptions_text = "\n\nConfirmed Assumptions:\n"
            for a in self.state.assumptions.assumptions:
                assumptions_text += f"• {a}\n"

        # Include previous search context
        search_context = ""
        if self.search_results:
            search_context = "\n\nPrevious research findings:\n" + "\n".join(
                self.search_results[-3:]
            )

        # Include user interests
        interests_text = ""
        if self.user_interests:
            interests_text = "\n\nUser's specific interests to incorporate:\n"
            for interest in self.user_interests:
                interests_text += f"• {interest}\n"

        # First gather planning-specific information including prices
        date_context = self._get_current_date_context()
        research_prompt = f"""Generate a day-by-day itinerary for this trip:

{date_context}

{constraints_text}{assumptions_text}{interests_text}{search_context}

Before creating the plan, search for:
1. Current flight prices from origin to destination
2. Hotel/accommodation prices in the destination
3. Transportation costs (local transport, taxis, etc.)
4. Entry fees and activity costs for attractions
5. Average meal costs in the destination
6. Any events/activities matching user interests with dates and ticket prices

IMPORTANT: Use the CURRENT YEAR ({datetime.now().year}) in all search queries. We are in {datetime.now().year}, not 2024.

Use web_search to find current prices and information, then create the itinerary with FULL COST ESTIMATES."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": research_prompt},
        ]

        # Gather current planning information
        planning_research = self.client.chat_with_tools(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            temperature=0.5,
            max_tool_calls=8,
            on_tool_call=self._handle_tool_call,
        )

        # Now generate structured plan with gathered info
        plan_prompt = f"""Create a structured day-by-day itinerary based on this information:

{constraints_text}{assumptions_text}{interests_text}

Research findings (use these for accurate cost estimates):
{planning_research}

REQUIREMENTS:
1. Commit to ONE specific route
2. Include COST ESTIMATE for EVERY activity
3. Include daily totals (accommodation + meals + transport + activities)
4. Include complete BUDGET BREAKDOWN at the end
5. If user mentioned specific interests (tech events, etc.), include relevant events with dates and costs

For costs, use the currency appropriate for the destination. Provide realistic estimates based on research."""

        plan_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": plan_prompt},
        ]

        plan = self.client.chat_structured(plan_messages, TravelPlan, temperature=0.7)
        self.state.current_plan = plan

        response = self._format_plan(plan)
        self.state.phase = Phase.REFINEMENT

        response += "\n\n---\n**Refinement Options:**\n"
        response += "1. Make it safer\n"
        response += "2. Make it faster\n"
        response += "3. Reduce travel hours\n"
        response += "4. Increase comfort\n"
        response += "5. Change base location\n"
        response += (
            "\nWhich refinement would you like, or are you satisfied with this plan?"
        )

        self.state.add_message("assistant", response)
        return response

    def refine_plan(self, refinement_type: str) -> str:
        """Refine the plan based on user's choice.

        Args:
            refinement_type: Type of refinement requested.

        Returns:
            Refined plan.
        """
        if not self.state.current_plan:
            return "No plan to refine. Please complete the planning phase first."

        current_plan_text = self._format_plan(self.state.current_plan)

        system_prompt = get_phase_prompt("refinement")

        user_message = f"""Current plan:
{current_plan_text}

User requested refinement: {refinement_type}

Apply this refinement and regenerate the affected parts of the plan.
Maintain the same format. Explain what changed and why."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Get refined plan
        plan = self.client.chat_structured(messages, TravelPlan, temperature=0.7)
        self.state.current_plan = plan

        response = f"**Plan refined for: {refinement_type}**\n\n"
        response += self._format_plan(plan)

        response += "\n\n---\nWould you like any other refinements?"
        self.state.add_message("user", f"Refine: {refinement_type}")
        self.state.add_message("assistant", response)
        return response

    def _format_constraints(self) -> str:
        """Format constraints for prompts."""
        c = self.state.constraints
        if not c:
            return f"From: {self.state.origin}\nTo: {self.state.destination}"

        lines = [f"From: {c.origin}", f"To: {c.destination}"]
        if c.month_or_season:
            lines.append(f"Season/Month: {c.month_or_season}")
        if c.duration_days:
            lines.append(f"Duration: {c.duration_days} days")
        if c.solo_or_group:
            lines.append(f"Travel type: {c.solo_or_group}")
        if c.budget:
            lines.append(f"Budget: {c.budget}")
        if c.comfort_level:
            lines.append(f"Comfort level: {c.comfort_level}")
        if c.interests:
            lines.append(f"Interests: {', '.join(c.interests)}")
        return "\n".join(lines)

    def _format_risk_assessment(self, risk: RiskAssessment) -> str:
        """Format risk assessment for display."""

        def risk_emoji(level: str) -> str:
            if level == "LOW":
                return "🟢"
            elif level == "MEDIUM":
                return "🟡"
            return "🔴"

        lines = ["**Risk Assessment:**\n"]
        lines.append(
            f"{risk_emoji(risk.season_weather.value)} Season & Weather: {risk.season_weather.value}"
        )
        lines.append(
            f"{risk_emoji(risk.route_accessibility.value)} Route Accessibility: {risk.route_accessibility.value}"
        )
        lines.append(
            f"{risk_emoji(risk.altitude_health.value)} Altitude & Health: {risk.altitude_health.value}"
        )
        lines.append(
            f"{risk_emoji(risk.infrastructure.value)} Infrastructure: {risk.infrastructure.value}"
        )

        if risk.warnings:
            lines.append("\n**Warnings:**")
            for warning in risk.warnings:
                lines.append(f"⚠️ {warning}")

        if risk.alternatives:
            lines.append("\n**Safer Alternatives:**")
            for alt in risk.alternatives:
                lines.append(f"→ {alt}")

        return "\n".join(lines)

    def _format_plan(self, plan: TravelPlan) -> str:
        """Format travel plan for display."""
        lines = [f"**{plan.summary}**\n"]
        lines.append(f"📍 Route: {plan.route}\n")

        if plan.acclimatization_notes:
            lines.append(f"🏔️ Acclimatization: {plan.acclimatization_notes}\n")

        if plan.buffer_days > 0:
            lines.append(f"⏳ Buffer days included: {plan.buffer_days}\n")

        lines.append("---\n")

        for day in plan.days:
            lines.append(f"**Day {day.day}: {day.title}**")

            # Activities with costs
            for activity in day.activities:
                cost_str = (
                    f" ({activity.cost_estimate})" if activity.cost_estimate else ""
                )
                lines.append(f"  • {activity.activity}{cost_str}")
                if activity.cost_notes:
                    lines.append(f"    ↳ {activity.cost_notes}")

            if day.travel_time:
                travel_cost = f" - {day.travel_cost}" if day.travel_cost else ""
                lines.append(f"  🚗 Travel: {day.travel_time}{travel_cost}")

            if day.accommodation:
                acc_cost = (
                    f" ({day.accommodation_cost}/night)"
                    if day.accommodation_cost
                    else ""
                )
                lines.append(f"  🏨 Stay: {day.accommodation}{acc_cost}")

            if day.meals_cost:
                lines.append(f"  🍽️ Meals: ~{day.meals_cost}")

            if day.day_total:
                lines.append(f"  💰 **Day Total: {day.day_total}**")

            lines.append(f"  💡 Why: {day.reasoning}")

            if day.notes:
                lines.append(f"  📝 Note: {day.notes}")
            lines.append("")

        # Budget breakdown
        if plan.budget_breakdown:
            b = plan.budget_breakdown
            lines.append("---\n")
            lines.append("## 💰 Budget Breakdown\n")
            lines.append("| Category | Estimated Cost |")
            lines.append("|----------|----------------|")
            lines.append(f"| ✈️ Flights | {b.flights} |")
            lines.append(f"| 🏨 Accommodation | {b.accommodation} |")
            lines.append(f"| 🚗 Local Transport | {b.local_transport} |")
            lines.append(f"| 🍽️ Meals | {b.meals} |")
            lines.append(f"| 🎯 Activities | {b.activities} |")
            lines.append(f"| 📦 Miscellaneous | {b.miscellaneous} |")
            lines.append(f"| **TOTAL** | **{b.total}** |")
            if b.notes:
                lines.append(f"\n📝 {b.notes}")

        return "\n".join(lines)
