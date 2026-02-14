"""Multi-provider web search with failover: Tavily → DuckDuckGo.

Note: Brave Search API is commented out for now - can be enabled later.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import requests
from ddgs import DDGS
from dotenv import load_dotenv

from app.cache import cache

# Load .env from project root (parent of backend/)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger = logging.getLogger(__name__)
    logger.debug(f"[WEB SEARCH] Loaded .env from {env_path}")
else:
    logger = logging.getLogger(__name__)
    logger.warning(f"[WEB SEARCH] .env not found at {env_path}")

# API Keys from environment
# BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")  # Commented out for now
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    logger.warning("[WEB SEARCH] TAVILY_API_KEY not found in environment!")


# def brave_search(
#     query: str, num_results: int = 5, timeout: int = 3
# ) -> list[dict[str, str]] | None:
#     """Search using Brave Search API.
#
#     Args:
#         query: Search query string.
#         num_results: Number of results to return.
#         timeout: Request timeout in seconds.
#
#     Returns:
#         List of search results or None if failed.
#     """
#     if not BRAVE_API_KEY:
#         logger.debug("[BRAVE] No API key configured, skipping")
#         return None
#
#     try:
#         logger.info(f"[BRAVE] Searching: {query}")
#         response = requests.get(
#             "https://api.search.brave.com/res/v1/web/search",
#             headers={
#                 "X-Subscription-Token": BRAVE_API_KEY,
#                 "Accept": "application/json",
#             },
#             params={
#                 "q": query,
#                 "count": num_results,
#             },
#             timeout=timeout,
#         )
#         response.raise_for_status()
#         data = response.json()
#
#         results = []
#         for item in data.get("web", {}).get("results", []):
#             results.append(
#                 {
#                     "title": item.get("title", ""),
#                     "url": item.get("url", ""),
#                     "snippet": item.get("description", ""),
#                 }
#             )
#
#         logger.info(f"[BRAVE] Found {len(results)} results for: {query}")
#         return results if results else None
#
#     except requests.Timeout:
#         logger.warning(f"[BRAVE] Timeout for query: {query}")
#         return None
#     except Exception as e:
#         logger.error(f"[BRAVE] Error for query '{query}': {str(e)}")
#         return None


@cache.memoize(expire=86400)  # Cache for 24 hours
def tavily_search(
    query: str, num_results: int = 5, timeout: int = 3
) -> list[dict[str, str]] | None:
    """Search using Tavily API.

    Args:
        query: Search query string.
        num_results: Number of results to return.
        timeout: Request timeout in seconds.

    Returns:
        List of search results or None if failed.
    """
    if not TAVILY_API_KEY:
        logger.debug("[TAVILY] No API key configured, skipping")
        return None

    try:
        logger.info(f"[TAVILY] Searching: {query}")
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",  # Fast mode
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )

        logger.info(f"[TAVILY] Found {len(results)} results for: {query}")
        return results if results else None

    except requests.Timeout:
        logger.warning(f"[TAVILY] Timeout for query: {query}")
        return None
    except Exception as e:
        logger.error(f"[TAVILY] Error for query '{query}': {str(e)}")
        return None


@cache.memoize(expire=86400)  # Cache for 24 hours
def ddgs_search(
    query: str, num_results: int = 5, timeout: int = 5
) -> list[dict[str, str]] | None:
    """Search using DuckDuckGo.

    Args:
        query: Search query string.
        num_results: Number of results to return.
        timeout: Request timeout in seconds.

    Returns:
        List of search results or None if failed.
    """
    num_results = min(num_results, 5)

    try:
        logger.info(f"[DDGS] Searching: {query}")

        def _run() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=num_results))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            results_raw = future.result(timeout=timeout)

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results_raw
        ]

        logger.info(f"[DDGS] Found {len(results)} results for: {query}")
        return results if results else None

    except TimeoutError:
        logger.warning(f"[DDGS] Timeout for query: {query}")
        return None
    except Exception as e:
        logger.error(f"[DDGS] Error for query '{query}': {str(e)}")
        return None


def web_search(query: str, num_results: int = 3) -> list[dict[str, str]]:
    """Search the web using multiple providers with failover.

    Order: Tavily → DuckDuckGo
    (Note: Brave is commented out for now)

    Args:
        query: Search query string.
        num_results: Number of results to return (max 10).

    Returns:
        List of search results with title, url, and snippet.
    """
    logger.info(f"[WEB SEARCH] Query: {query}")
    num_results = min(num_results, 3)

    # Try Tavily first (primary)
    results = tavily_search(query, num_results)
    if results:
        logger.info(f"[WEB SEARCH] Using TAVILY results for: {query}")
        return results

    # Fallback to DuckDuckGo
    results = ddgs_search(query, num_results)
    if results:
        logger.info(f"[WEB SEARCH] Using DDGS results for: {query}")
        return results

    # All failed
    logger.error(f"[WEB SEARCH] All providers failed for query: {query}")
    return [{"error": "All search providers failed. Proceed with estimates."}]


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Tool arguments.

    Returns:
        JSON string of tool results.
    """
    if tool_name == "web_search":
        query = arguments.get("query", "")
        logger.info(f"[TOOL CALL] Executing {tool_name} with query: {query}")
        results = web_search(
            query=query,
            num_results=arguments.get("num_results", 5),
        )
        return json.dumps(results, indent=2)

    logger.warning(f"[TOOL CALL] Unknown tool requested: {tool_name}")
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
