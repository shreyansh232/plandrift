"""Train cost search helpers for Indian Railways routes."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from statistics import median
from typing import Optional
from urllib.parse import urlparse

from app.agent.web_search import web_search

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)

_RUPEE_PREFIX_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_RUPEE_SUFFIX_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:₹|rs\.?|inr)",
    re.IGNORECASE,
)
_TRAIN_NUMBER_PATTERN = re.compile(r"\b\d{5}\b")
_MULTISPACE_PATTERN = re.compile(r"\s+")
_TRUSTED_TRAIN_DOMAINS = {
    "irctc.co.in",
    "indianrail.gov.in",
    "ixigo.com",
    "confirmtkt.com",
    "trainman.in",
    "redbus.in",
    "railmitra.com",
}

# Common Indian city/station mappings for validation
INDIAN_CITIES = {
    "delhi",
    "mumbai",
    "kolkata",
    "chennai",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "pune",
    "ahmedabad",
    "jaipur",
    "lucknow",
    "kanpur",
    "nagpur",
    "indore",
    "thane",
    "bhopal",
    "visakhapatnam",
    "pimpri",
    "patna",
    "vadodara",
    "ghaziabad",
    "ludhiana",
    "agra",
    "nashik",
    "faridabad",
    "meerut",
    "rajkot",
    "kalyan",
    "vasai",
    "varanasi",
    "srinagar",
    "aurangabad",
    "dhanbad",
    "amritsar",
    "navi mumbai",
    "allahabad",
    "prayagraj",
    "ranchi",
    "coimbatore",
    "jabalpur",
    "gwalior",
    "vijayawada",
    "jodhpur",
    "madurai",
    "raipur",
    "kota",
    "guwahati",
    "chandigarh",
    "solapur",
    "hubli",
    "tiruchirappalli",
    "mysore",
    "tiruppur",
    "gurgaon",
    "gurugram",
    "aligarh",
    "jalandhar",
    "bareilly",
    "merut",
    "dehradun",
    "shimla",
    "manali",
    "goa",
    "udaipur",
    "jaisalmer",
    "pushkar",
    "ajmer",
    "bikaner",
    "munnar",
    "kochi",
    "cochin",
    "alleppey",
    "kovalam",
    "trivandrum",
    "thiruvananthapuram",
    "kozhikode",
    "calicut",
    "darjeeling",
    "gangtok",
    "shillong",
    "kohima",
    "imphal",
    "leh",
    "ladakh",
    "gulmarg",
    "pahalgam",
    "rishikesh",
    "haridwar",
    "mussoorie",
    "khajuraho",
    "orchha",
    "ujjain",
    "puri",
    "bhubaneswar",
    "konark",
    "cuttack",
    "tanjore",
    "thanjavur",
    "kanyakumari",
    "rameswaram",
    "mahabalipuram",
    "pondicherry",
    "puducherry",
    "hampi",
    "mangalore",
    "udupi",
    # Airport codes (common ones for India)
    "bom",
    "del",
    "maa",
    "ccu",
    "hyd",
    "blr",
    "pnq",
    "amd",
    "jai",
    "lko",
    "cok",
    "trv",
    "ccj",
    "ixc",
    "gaa",
    "sxe",
}


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.replace("www.", "").lower()


def _is_trusted_train_source(url: str) -> bool:
    domain = _extract_domain(url)
    return any(
        domain == trusted_domain or domain.endswith(f".{trusted_domain}")
        for trusted_domain in _TRUSTED_TRAIN_DOMAINS
    )


def _sanitize_snippet(text: str) -> str:
    cleaned = _TRAIN_NUMBER_PATTERN.sub("", text or "")
    cleaned = _MULTISPACE_PATTERN.sub(" ", cleaned).strip(" -:|")
    return cleaned


def _is_indian_city(location: str) -> bool:
    """Check if a location is likely in India."""
    if not location:
        return False

    location_lower = location.lower()

    for city in INDIAN_CITIES:
        if city in location_lower:
            return True

    for indicator in ("india", "bharat", "hindustan"):
        if indicator in location_lower:
            return True

    return False


def _extract_numeric_price(price_str: str) -> Optional[float]:
    """Extract numeric price value from a price string."""
    if not price_str:
        return None

    cleaned = price_str.lower().replace(",", "").strip()
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs|lacs)\b", cleaned)
    if lakh_match:
        return float(lakh_match.group(1)) * 100_000

    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)\b", cleaned)
    if crore_match:
        return float(crore_match.group(1)) * 10_000_000

    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", cleaned)
    if k_match:
        return float(k_match.group(1)) * 1_000

    numbers = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def _extract_rupee_prices(text: str) -> list[float]:
    prices: list[float] = []
    working_text = (text or "").replace(",", " ")
    for pattern in (_RUPEE_PREFIX_PATTERN, _RUPEE_SUFFIX_PATTERN):
        for match in pattern.findall(working_text):
            try:
                value = float(str(match).replace(",", ""))
            except ValueError:
                continue
            if 50 <= value <= 100_000:
                prices.append(value)
    # Preserve ordering and remove duplicates
    return list(dict.fromkeys(prices))


def _is_within_budget(train_cost: Optional[float], budget_str: Optional[str]) -> bool:
    """Check if train cost is within the user's budget."""
    if train_cost is None or not budget_str:
        return True

    budget_amount = _extract_numeric_price(budget_str)
    if budget_amount is None:
        return True

    max_transport_budget = budget_amount * 0.4
    return train_cost <= max_transport_budget


def search_train_costs(
    origin: str,
    destination: str,
    date_context: str | None = None,
    budget: str | None = None,
    train_class: str | None = None,
) -> dict:
    """Search for estimated train costs between Indian cities."""
    origin_is_indian = _is_indian_city(origin)
    destination_is_indian = _is_indian_city(destination)
    is_indian_route = origin_is_indian and destination_is_indian

    result = {
        "summary": "",
        "within_budget": True,
        "estimated_cost": None,
        "is_indian_route": is_indian_route,
        "trusted_sources": 0,
        "sources_scanned": 0,
    }

    if not is_indian_route:
        logger.info(
            "[TRAIN SEARCH] Route %s -> %s is not Indian domestic, skipping train search",
            origin,
            destination,
        )
        return result

    try:
        current_year = datetime.now().year
        date_str = date_context if date_context else f"{current_year}"
        class_str = f"{train_class} " if train_class else ""
        query = (
            f"Indian Railways {origin} to {destination} train fare "
            f"{class_str}{date_str} IRCTC"
        )

        msg = f"[TRAIN SEARCH] Searching: {query}"
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")

        def _run() -> list[dict]:
            # Uses Tavily as primary provider and DDG as fallback.
            return web_search(query, num_results=8)

        future = _executor.submit(_run)
        results = future.result(timeout=6)
        if results and "error" in results[0]:
            results = []
        result["sources_scanned"] = len(results)

        if not results:
            msg = f"[TRAIN SEARCH] No results found for {origin} to {destination}"
            logger.warning(msg)
            print(f"\n\033[94m{msg}\033[0m")
            result["summary"] = f"No train data found for {origin} -> {destination}"
            return result

        candidates: list[dict] = []
        for raw in results:
            title = (raw.get("title") or "").strip()
            body = (raw.get("snippet") or "").strip()
            href = (raw.get("url") or "").strip()
            domain = _extract_domain(href)
            combined_text = " ".join(part for part in (title, body) if part)
            candidates.append(
                {
                    "title": _sanitize_snippet(title),
                    "body": _sanitize_snippet(body),
                    "url": href,
                    "domain": domain,
                    "trusted": _is_trusted_train_source(href),
                    "prices": _extract_rupee_prices(combined_text),
                }
            )

        trusted_candidates = [
            candidate for candidate in candidates if candidate["trusted"]
        ]
        selected_candidates = trusted_candidates or candidates
        result["trusted_sources"] = len(trusted_candidates)

        costs_found: list[float] = []
        for candidate in selected_candidates:
            costs_found.extend(candidate["prices"])
        if not costs_found:
            for candidate in candidates:
                costs_found.extend(candidate["prices"])

        estimated_cost = None
        if costs_found:
            estimated_cost = float(median(costs_found))
            result["estimated_cost"] = estimated_cost

        within_budget = _is_within_budget(estimated_cost, budget)
        result["within_budget"] = within_budget

        snippets: list[str] = []
        for candidate in selected_candidates[:4]:
            source = candidate["domain"] or "search result"
            snippet = candidate["body"] or candidate["title"]
            if not snippet:
                continue
            snippet = snippet[:220].rstrip()
            fares = candidate["prices"]
            fare_hint = ""
            if fares:
                low = min(fares)
                high = max(fares)
                fare_hint = (
                    f" (observed fare: ₹{low:.0f})"
                    if low == high
                    else f" (observed fares: ₹{low:.0f}-₹{high:.0f})"
                )
            url_hint = f" ({candidate['url']})" if candidate["url"] else ""
            snippets.append(f"- {source}: {snippet}{fare_hint}{url_hint}")

        trust_note = (
            "Trusted railway sources found."
            if trusted_candidates
            else "Limited trusted railway sources; verify options on IRCTC before booking."
        )
        budget_note = ""
        if estimated_cost and not within_budget:
            budget_note = (
                f"\n\nBudget Alert: Estimated train fare around ₹{estimated_cost:.0f} "
                "may exceed 40% of your stated budget."
            )

        snippet_block = (
            "\n".join(snippets) if snippets else "No reliable fare snippets found."
        )
        result["summary"] = (
            f"Train Cost Estimates ({origin} -> {destination} - Indian Railways):\n"
            f"{snippet_block}\n\n{trust_note}"
            "\nUse these as fare benchmarks; do not assume a specific train number "
            "unless verified on official sources."
            f"{budget_note}"
        )

        msg = (
            f"[TRAIN SEARCH] Found {len(results)} results, "
            f"trusted={len(trusted_candidates)}, within_budget={within_budget}"
        )
        logger.info(msg)
        print(f"\n\033[94m{msg}\033[0m")
        return result

    except FuturesTimeoutError:
        msg = "[TRAIN SEARCH] Timeout"
        logger.warning(msg)
        print(f"\n\033[94m{msg}\033[0m")
        result["summary"] = "Train search timed out"
        return result
    except Exception as e:
        msg = f"[TRAIN SEARCH] Error: {e}"
        logger.error(msg)
        print(f"\n\033[94m{msg}\033[0m")
        result["summary"] = f"Error searching train costs: {e}"
        return result


def should_search_trains(origin: str, destination: str) -> bool:
    """Determine if train search should be performed for this route."""
    return _is_indian_city(origin) and _is_indian_city(destination)


def get_train_assumption_note(
    origin: str, destination: str, budget: str | None = None
) -> str | None:
    """Generate an assumption note about train travel for the assumptions phase."""
    if not should_search_trains(origin, destination):
        return None

    note = (
        f"For travel between {origin} and {destination} within India, "
        "considering Indian Railways train options as an alternative to flights"
    )
    if budget:
        note += f", with budget constraint of {budget}"
    return note
