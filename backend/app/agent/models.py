"""Pydantic models for conversation state and LLM responses."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    """Conversation phases for the travel planning agent."""

    CLARIFICATION = "clarification"
    FEASIBILITY = "feasibility"
    ASSUMPTIONS = "assumptions"
    PLANNING = "planning"
    REFINEMENT = "refinement"


class RiskLevel(str, Enum):
    """Risk assessment levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskAssessment(BaseModel):
    """Risk assessment for a travel plan."""

    season_weather: RiskLevel = Field(
        description="Risk from season and weather conditions"
    )
    route_accessibility: RiskLevel = Field(description="Risk from route accessibility")
    altitude_health: RiskLevel = Field(
        description="Risk from altitude and health stress"
    )
    infrastructure: RiskLevel = Field(
        description="Risk from infrastructure and connectivity"
    )
    overall_feasible: bool = Field(description="Whether the trip is feasible overall")
    warnings: list[str] = Field(
        default_factory=list, description="Specific warnings for the user"
    )
    alternatives: list[str] = Field(
        default_factory=list, description="Safer alternatives if high risk"
    )


class ClarificationQuestions(BaseModel):
    """Structured clarification questions from the agent."""

    questions: list[str] = Field(
        description="List of clarification questions to ask the user",
        min_length=1,
        max_length=5,
    )


class TravelConstraints(BaseModel):
    """User's travel constraints extracted from clarification answers."""

    origin: str = Field(description="Starting location / traveling from")
    destination: str = Field(description="Travel destination / traveling to")
    month_or_season: Optional[str] = Field(
        default=None, description="Month or season of travel"
    )
    duration_days: Optional[int] = Field(
        default=None, description="Total trip duration including travel"
    )
    solo_or_group: Optional[str] = Field(
        default=None, description="Solo or group travel"
    )
    budget: Optional[str] = Field(default=None, description="Budget level or range")
    comfort_level: Optional[str] = Field(
        default=None, description="Comfort with rough conditions: low/medium/high"
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Specific interests or activities the traveler wants (e.g., tech events, hiking, food tours)",
    )


class Assumptions(BaseModel):
    """Assumptions the agent is making before planning."""

    assumptions: list[str] = Field(description="List of assumptions being made")
    uncertain_assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions that are uncertain and need confirmation",
    )


class ActivityCost(BaseModel):
    """Cost estimate for a single activity."""

    activity: str = Field(description="Activity name/description")
    cost_estimate: str = Field(
        description="Estimated cost (e.g., '$50', '₹2000', 'Free')"
    )
    cost_notes: Optional[str] = Field(
        default=None,
        description="Notes about the cost (e.g., 'booking required', 'varies by season')",
    )


class DayPlan(BaseModel):
    """Plan for a single day."""

    day: int = Field(description="Day number")
    title: str = Field(description="Brief title for the day")
    activities: list[ActivityCost] = Field(
        description="Activities for the day with cost estimates"
    )
    reasoning: str = Field(description="Why the day is structured this way")
    travel_time: Optional[str] = Field(
        default=None, description="Expected travel time if applicable"
    )
    travel_cost: Optional[str] = Field(
        default=None, description="Cost of travel/transport for the day"
    )
    accommodation: Optional[str] = Field(default=None, description="Where to stay")
    accommodation_cost: Optional[str] = Field(
        default=None, description="Cost of accommodation per night"
    )
    meals_cost: Optional[str] = Field(
        default=None, description="Estimated cost for meals for the day"
    )
    day_total: Optional[str] = Field(
        default=None, description="Total estimated cost for the day"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes or warnings"
    )


class BudgetBreakdown(BaseModel):
    """Budget breakdown for the trip."""

    flights: str = Field(description="Estimated flight costs")
    accommodation: str = Field(description="Total accommodation costs")
    local_transport: str = Field(description="Local transportation costs")
    meals: str = Field(description="Total meal costs")
    activities: str = Field(description="Activities and entrance fees")
    miscellaneous: str = Field(description="Buffer for miscellaneous expenses")
    total: str = Field(description="Total estimated trip cost")
    currency: str = Field(description="Currency used for estimates")
    notes: Optional[str] = Field(
        default=None, description="Important notes about the budget"
    )


class TravelPlan(BaseModel):
    """Complete travel itinerary."""

    summary: str = Field(description="Brief summary of the trip")
    route: str = Field(description="The committed route")
    days: list[DayPlan] = Field(description="Day-by-day itinerary")
    buffer_days: int = Field(default=0, description="Number of buffer days included")
    acclimatization_notes: Optional[str] = Field(
        default=None, description="Acclimatization logic if applicable"
    )
    budget_breakdown: Optional[BudgetBreakdown] = Field(
        default=None, description="Detailed budget breakdown for the trip"
    )


class RefinementOptions(BaseModel):
    """Refinement options offered after plan generation."""

    options: list[str] = Field(
        description="Available refinement options",
        default=[
            "Make it safer",
            "Make it faster",
            "Reduce travel hours",
            "Increase comfort",
            "Change base location",
        ],
    )


class Message(BaseModel):
    """A single message in the conversation."""

    role: str = Field(description="Message role: system, user, or assistant")
    content: str = Field(description="Message content")


class ConversationState(BaseModel):
    """Complete state of the travel planning conversation."""

    phase: Phase = Field(
        default=Phase.CLARIFICATION, description="Current conversation phase"
    )
    origin: Optional[str] = Field(default=None, description="Starting location")
    destination: Optional[str] = Field(default=None, description="Travel destination")
    constraints: Optional[TravelConstraints] = Field(
        default=None, description="User's travel constraints"
    )
    risk_assessment: Optional[RiskAssessment] = Field(
        default=None, description="Risk assessment results"
    )
    assumptions: Optional[Assumptions] = Field(
        default=None, description="Assumptions for planning"
    )
    current_plan: Optional[TravelPlan] = Field(
        default=None, description="Generated travel plan"
    )
    messages: list[Message] = Field(
        default_factory=list, description="Full conversation history"
    )
    awaiting_confirmation: bool = Field(
        default=False, description="Waiting for user confirmation"
    )

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

    def get_openai_messages(self) -> list[dict]:
        """Get messages in OpenAI API format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
