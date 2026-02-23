"""Flight cost search using shared web search providers.

Searches for average flight costs between origin and destination
to provide a baseline estimate for the planning phase.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

from app.agent.web_search import web_search

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)


def search_flight_costs(
    origin: str, destination: str, date_context: str | None = None
) -> str:
    """Search for estimated flight costs.

    Args:
        origin: Origin city/airport.
        destination: Destination city/airport.
        date_context: Month/season/dates of travel.

    Returns:
        Summary string of search results constraints cost info.
    """
    try:
        current_year = datetime.now().year
        date_str = date_context if date_context else f"{current_year}"
        query = (
            f"round trip flight cost from {origin} to {destination} {date_str} price"
        )
        msg = f"[FLIGHT SEARCH] Searching: {query}"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")  # Blue color for visibility

        def _run() -> list[dict]:
            # Uses Tavily as primary provider and DDG as fallback.
            return web_search(query, num_results=4)

        future = _executor.submit(_run)
        results = future.result(timeout=6)
        if results and "error" in results[0]:
            return ""

        if not results:
            return ""

        # Format results into a context string
        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("snippet", "")
            snippets.append(f"- {title}: {body}")

        summary = "\n".join(snippets)
        msg = f"[FLIGHT SEARCH] Found {len(results)} results"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")
        return f"Flight Cost Estimates Research ({origin} -> {destination}):\n{summary}"

    except FuturesTimeoutError:
        msg = "[FLIGHT SEARCH] Timeout"
        logger.warning(msg)
        print(f"\n\033[94m{msg}\033[0m")
        return ""
    except Exception as e:
        msg = f"[FLIGHT SEARCH] Error: {e}"
        logger.error(msg)
        print(f"\n\033[94m{msg}\033[0m")
        return ""
