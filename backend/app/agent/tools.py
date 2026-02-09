"""Web search and other tools for the travel planning agent."""

import json
from typing import Any

from ddgs import DDGS


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


def web_search(query: str, num_results: int = 8) -> list[dict[str, str]]:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        num_results: Number of results to return (max 10).

    Returns:
        List of search results with title, url, and snippet.
    """
    num_results = min(num_results, 10)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments.

    Returns:
        JSON string of tool results.
    """
    if tool_name == "web_search":
        results = web_search(
            query=arguments.get("query", ""),
            num_results=arguments.get("num_results", 5),
        )
        return json.dumps(results, indent=2)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def format_search_results(results: list[dict[str, str]]) -> str:
    """Format search results for display.

    Args:
        results: List of search result dicts.

    Returns:
        Formatted string for display.
    """
    if not results:
        return "No results found."

    if "error" in results[0]:
        return f"Search error: {results[0]['error']}"

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   {r['snippet']}")
        lines.append(f"   Source: {r['url']}")
        lines.append("")

    return "\n".join(lines)
