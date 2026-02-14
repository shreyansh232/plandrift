"""Utility functions for the travel agent."""

from datetime import datetime
from typing import Optional

from app.agent.models import ConversationState


def get_current_date_context() -> str:
    """Get current date context for prompts.

    Returns:
        Formatted date string with year.
    """
    now = datetime.now()
    return f"Today's date: {now.strftime('%B %d, %Y')} (Year: {now.year})"


def detect_budget_currency(state: ConversationState, current_input: Optional[str] = None) -> str:
    """Detect the user's preferred currency from their budget string or current input.

    Args:
        state: Current conversation state.
        current_input: Optional current user input to check.

    Returns:
        Currency code string (e.g., 'INR', 'USD', 'EUR').
    """
    # Check current input first as it's the most recent preference
    search_targets = []
    if current_input:
        search_targets.append(current_input.upper())
    
    if state.constraints and state.constraints.budget:
        search_targets.append(state.constraints.budget.upper())
    
    # If no budget info at all, check full conversation history for currency symbols
    if not search_targets:
        for msg in reversed(state.messages):
            if msg.role == "user":
                search_targets.append(msg.content.upper())

    currency_map = {
        ("INR", "₹", "LAKH", "RUPEE", "RS"): "INR",
        ("USD", "$", "DOLLAR"): "USD",
        ("EUR", "€", "EURO"): "EUR",
        ("JPY", "¥", "YEN"): "JPY",
        ("GBP", "£", "POUND"): "GBP",
        ("THB", "BAHT"): "THB",
        ("AUD", "A$"): "AUD",
        ("CAD", "C$"): "CAD",
        ("SGD", "S$"): "SGD",
    }
    
    for target in search_targets:
        for keywords, code in currency_map.items():
            if any(kw in target for kw in keywords):
                return code
                
    return "USD"
