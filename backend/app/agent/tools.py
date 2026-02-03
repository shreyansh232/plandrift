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
            "description": """Search the web for current travel information. Use this tool when you need:
- Current visa requirements or travel restrictions
- Recent travel advisories or safety warnings
- Current weather conditions or seasonal information
- Recent infrastructure changes (road closures, new routes)
- Current prices for flights, hotels, or activities
- Latest reviews or recommendations
- Permit requirements and booking procedures
- Festival or event dates""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and include the destination/route and what information you need.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def web_search(query: str, num_results: int = 5) -> list[dict[str, str]]:
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
