"""Web search and other tools for the travel planning agent."""

import json
import logging
from typing import Any

# Import the multi-provider web search
from app.agent.web_search import (
    web_search,
    execute_tool,
    format_search_results,
)

logger = logging.getLogger(__name__)


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": """Search the web for current travel information.

Use this when you need:
- Flight prices (search for "round trip flights <origin> to <dest> <month> <year> price")
- Hotel/hostel prices (search for "<dest> hotel prices per night <month> <year>")
- Activity costs, entry fees, transport costs
- Visa requirements, travel advisories, weather
- Event dates, ticket prices

TIPS FOR BETTER PRICE RESULTS:
- Include the currency (e.g. "INR", "USD", "JPY")
- Include the month and year
- Include "price" or "cost" in the query
- Search for specific sites: "skyscanner", "booking.com", "google flights"
- Example: "round trip flights Mumbai to Tokyo March 2026 price INR skyscanner"
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific: include origin, destination, dates, currency, and 'price'.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 8, max 10)",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


# Re-export for backward compatibility
__all__ = ["TOOL_DEFINITIONS", "web_search", "execute_tool", "format_search_results"]
