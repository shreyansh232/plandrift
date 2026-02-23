"""Hotel cost search using shared web search providers.

Searches for average hotel costs in destination
to provide a baseline estimate for the planning phase.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

from app.agent.web_search import web_search

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)


def search_hotel_costs(
    destination: str,
    date_context: str | None = None,
    budget: str | None = None,
    preferences: str | None = None,
) -> str:
    """Search for estimated hotel costs.

    Args:
        destination: Destination city/location.
        date_context: Month/season/dates of travel.
        budget: Total trip budget.
        preferences: specific preferences like 'hostel', 'luxury', etc.

    Returns:
        Summary string of search results constraints cost info.
    """
    try:
        current_year = datetime.now().year
        date_str = date_context if date_context else f"{current_year}"

        # Determine accommodation type
        accommodation_type = "hotel"
        if preferences and "hostel" in preferences.lower():
            accommodation_type = "hostel"

        # Construct query
        # "Search for hotels with average pricing"
        query = f"average {accommodation_type} prices in {destination} {date_str}"

        # If we have a budget, we might want to check for "best value" or similar if budget is low
        # But generally "average price" gives good baseline data.
        if budget:
            pass  # logic to parse budget is complex, relying on 'average' as per instructions

        msg = f"[HOTEL SEARCH] Searching: {query}"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")  # Blue color for visibility

        def _run() -> list[dict]:
            # Uses Tavily as primary provider and DDG as fallback.
            return web_search(query, num_results=5)

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
        msg = f"[HOTEL SEARCH] Found {len(results)} results"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")

        context_msg = f"Hotel/Accommodation Cost Estimates Research ({destination}):\n"
        if budget:
            context_msg += f"(Context: User budget is {budget}, usually ~30% is spent on accommodation)\n"

        return f"{context_msg}{summary}"

    except FuturesTimeoutError:
        msg = "[HOTEL SEARCH] Timeout"
        logger.warning(msg)
        print(f"\n\033[94m{msg}\033[0m")
        return ""
    except Exception as e:
        msg = f"[HOTEL SEARCH] Error: {e}"
        logger.error(msg)
        print(f"\n\033[94m{msg}\033[0m")
        return ""
