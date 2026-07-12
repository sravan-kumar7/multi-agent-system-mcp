from __future__ import annotations

import asyncio
import json
import logging
import operator
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Iterable, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from mcp_client import load_all_tools


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("travel_graph")


# =============================================================================
# Environment and model configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """
    Create the Groq client lazily.

    Lazy initialization prevents the entire Streamlit app from crashing during
    module import when a deployment secret is missing or temporarily unavailable.
    Agent-level fallback handling can then return a useful travel plan.
    """
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Community "
            "Cloud secrets or the local .env file."
        )

    model_name = os.getenv("GROQ_MODEL", GROQ_MODEL).strip() or GROQ_MODEL

    _llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_retries=2,
    )
    return _llm


# =============================================================================
# LangGraph state
# =============================================================================

class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    destination: str
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    final_response: str
    llm_calls: int
    errors: list[str]


# =============================================================================
# Shared tool cache
# =============================================================================

_loaded_tool_data: dict[str, Any] | None = None
_tool_lock = asyncio.Lock()


async def get_agent_tools() -> dict[str, Any]:
    """Load and cache all MCP/search tools once per Python process."""
    global _loaded_tool_data

    if _loaded_tool_data is not None:
        return _loaded_tool_data

    async with _tool_lock:
        if _loaded_tool_data is None:
            logger.info("Loading MCP and search tools...")
            _loaded_tool_data = await load_all_tools()

    return _loaded_tool_data


# =============================================================================
# Generic helpers
# =============================================================================

def get_tool_schema(tool: Any) -> dict[str, Any]:
    """Return a tool's JSON input schema when available."""
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema:
            return args_schema.model_json_schema()
    except Exception:
        logger.debug("Could not read args_schema", exc_info=True)

    try:
        tool_args = getattr(tool, "args", None)
        if tool_args:
            return {"type": "object", "properties": tool_args}
    except Exception:
        logger.debug("Could not read tool.args", exc_info=True)

    return {}


def find_tool(
    tools: list[Any],
    preferred_names: Iterable[str],
    keywords: Iterable[str],
) -> Any | None:
    """Find a tool by exact preferred name, then by keyword."""
    preferred = {name.lower() for name in preferred_names}

    for tool in tools:
        if str(getattr(tool, "name", "")).lower() in preferred:
            return tool

    lowered_keywords = [keyword.lower() for keyword in keywords]
    for tool in tools:
        tool_name = str(getattr(tool, "name", "")).lower()
        if any(keyword in tool_name for keyword in lowered_keywords):
            return tool

    return tools[0] if tools else None


def safe_json_loads(value: str) -> Any:
    """Parse JSON text when possible; otherwise return the original string."""
    text = value.strip()
    if not text:
        return ""

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return value


def normalize_tool_result(value: Any) -> Any:
    """
    Convert MCP/LangChain content blocks into normal Python objects.

    Handles shapes such as:
        [{"type": "text", "text": "{\"city\":\"Tokyo\"}"}]
        {"content": [{"type": "text", "text": "..."}]}
        AIMessage(content=[{"type": "text", "text": "..."}])
    """
    if value is None:
        return None

    if hasattr(value, "content"):
        return normalize_tool_result(getattr(value, "content"))

    if isinstance(value, str):
        parsed = safe_json_loads(value)
        if parsed is value:
            return value.strip()
        return normalize_tool_result(parsed)

    if isinstance(value, list):
        normalized_items: list[Any] = []

        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                normalized_items.append(
                    normalize_tool_result(item.get("text", ""))
                )
            else:
                normalized_items.append(normalize_tool_result(item))

        if len(normalized_items) == 1:
            return normalized_items[0]

        return normalized_items

    if isinstance(value, dict):
        # Common wrapper used by MCP/LangChain.
        if "content" in value and len(value) <= 4:
            content = normalize_tool_result(value["content"])
            other_fields = {
                key: normalize_tool_result(item)
                for key, item in value.items()
                if key != "content"
            }
            if not other_fields:
                return content
            return {"content": content, **other_fields}

        return {
            str(key): normalize_tool_result(item)
            for key, item in value.items()
        }

    return value



def extract_mcp_text(value: Any) -> str:
    """
    Extract clean Markdown/text from MCP and LangChain content blocks.

    This prevents raw wrappers such as
    ``[{"type": "text", "text": "..."}]`` from reaching Streamlit.
    """
    if value is None:
        return ""

    if hasattr(value, "content"):
        return extract_mcp_text(getattr(value, "content"))

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = [extract_mcp_text(item) for item in value]
        return "\n\n".join(part for part in parts if part).strip()

    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return str(value.get("text", "")).strip()

        if "text" in value:
            return str(value.get("text", "")).strip()

        if "content" in value:
            return extract_mcp_text(value.get("content"))

        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        except Exception:
            return str(value).strip()

    return str(value).strip()


def looks_like_markdown(value: str) -> bool:
    """Return True when a tool already returned ready-to-render Markdown."""
    text = value.lstrip()
    return (
        text.startswith("#")
        or "| Metric |" in text
        or "| Date and time |" in text
        or "### Current conditions" in text
        or "### Upcoming forecast" in text
    )


async def invoke_llm_markdown(
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_seconds: int = 90,
) -> str:
    """Call Groq safely and require a non-empty Markdown response."""
    response = await asyncio.wait_for(
        get_llm().ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        ),
        timeout=timeout_seconds,
    )

    content = clean_markdown(response)

    if not content:
        raise RuntimeError("The language model returned an empty response.")

    return content


def extract_trip_duration(user_query: str, default: int = 5) -> int:
    """Extract a sensible trip duration from the request."""
    patterns = (
        r"\b(\d{1,2})\s*[- ]?\s*day",
        r"\bfor\s+(\d{1,2})\s+days?\b",
    )

    for pattern in patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            return max(1, min(int(match.group(1)), 30))

    return default


def build_fallback_itinerary(
    destination: str,
    user_query: str,
) -> str:
    """Create a useful itinerary even when the LLM provider is unavailable."""
    days = extract_trip_duration(user_query)
    day_templates = [
        ("Arrival and orientation", "Check in, explore the local area and recover from travel."),
        ("Signature landmarks", "Visit the destination's most important landmarks and central districts."),
        ("Culture and history", "Explore museums, heritage areas, temples, monuments or historic neighbourhoods."),
        ("Local experiences", "Try local food, markets, neighbourhood walks and community experiences."),
        ("Nature or day trip", "Take a practical day trip or visit parks, viewpoints or nearby attractions."),
        ("Shopping and leisure", "Keep a flexible day for shopping, cafés, entertainment and rest."),
        ("Final highlights and departure", "Complete any missed activities, pack and travel to the airport early."),
    ]

    lines = [
        f"## 🗺️ {days}-day plan for {destination}",
        "",
        "> This practical fallback itinerary was generated because final LLM "
        "synthesis was temporarily unavailable. Confirm opening hours and bookings.",
        "",
    ]

    for index in range(days):
        title, detail = day_templates[min(index, len(day_templates) - 1)]

        if index >= len(day_templates):
            title = f"Flexible exploration day {index + 1}"
            detail = (
                "Choose activities based on weather, energy level and any attractions "
                "that still need to be covered."
            )

        lines.extend(
            [
                f"### Day {index + 1} — {title}",
                "",
                f"- **Morning:** {detail}",
                "- **Afternoon:** Continue with nearby attractions to minimise travel time.",
                "- **Evening:** Enjoy local dining and keep the schedule flexible.",
                "- **Transport:** Prefer public transport or official taxi services.",
                "- **Tip:** Reserve timed-entry attractions in advance.",
                "",
            ]
        )

    lines.extend(
        [
            "## 💰 Budget guidance",
            "",
            "- Treat all prices as estimates until live booking pages are checked.",
            "- Keep a 10–15% contingency for transport, taxes and unexpected costs.",
            "",
            "## 📄 Before travelling",
            "",
            "- Verify passport validity, visa rules and entry requirements.",
            "- Confirm travel insurance and emergency contact details.",
            "- Recheck flight, hotel and weather information before departure.",
        ]
    )

    return "\n".join(lines)


def build_fallback_final_plan(
    destination: str,
    user_query: str,
    itinerary: str,
) -> str:
    """Create a presentable final response if final LLM synthesis fails."""
    days = extract_trip_duration(user_query)

    return f"""# ✈️ Your Travel Plan: {destination}

## ✅ Recommended approach

Use a balanced {days}-day plan with centrally located accommodation, public
transport where practical and advance booking for major attractions.

## 📌 Trip at a glance

| Item | Recommendation |
|---|---|
| Destination | **{destination}** |
| Duration | **{days} days** |
| Planning style | Balanced and flexible |
| Booking priority | Flights, accommodation and timed-entry attractions |
| Safety margin | Keep 10–15% of the budget as contingency |

## 🧭 Day-wise plan

{itinerary}

## ⚠️ Verify before payment

1. Live flight schedule and final fare
2. Hotel taxes, cancellation conditions and location
3. Passport, visa and entry requirements
4. Travel insurance
5. Latest weather forecast

## 🚀 Next actions

1. Compare flight options.
2. Shortlist accommodation in a convenient area.
3. Reserve major attractions.
4. Save offline copies of confirmations.
5. Recheck weather and transport shortly before departure.
"""


def compact_json(value: Any, limit: int = 12_000) -> str:
    """Serialize normalized data for LLM context without exposing it to users."""
    normalized = normalize_tool_result(value)

    if isinstance(normalized, str):
        text = normalized
    else:
        try:
            text = json.dumps(
                normalized,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        except Exception:
            text = str(normalized)

    return text[:limit]


def clean_markdown(text: Any) -> str:
    """Normalize model output and remove accidental fenced wrappers."""
    value = str(getattr(text, "content", text) or "").strip()

    if value.startswith("```markdown"):
        value = value[len("```markdown"):].strip()
    elif value.startswith("```"):
        value = value[3:].strip()

    if value.endswith("```"):
        value = value[:-3].strip()

    return value


def safe_number(value: Any, digits: int = 1) -> str:
    """Format numeric values while tolerating strings and missing data."""
    try:
        number = float(value)
        rounded = round(number, digits)
        if rounded.is_integer():
            return str(int(rounded))
        return str(rounded)
    except (TypeError, ValueError):
        return "Not available"


def title_case_condition(value: Any) -> str:
    text = str(value or "Not available").strip()
    return text[:1].upper() + text[1:]


def source_domain(url: str) -> str:
    """Return a clean domain for display."""
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return "Source"


# =============================================================================
# Destination extraction
# =============================================================================

def clean_location_text(location: str) -> str:
    location = re.sub(
        r"^(destination|location|city)\s*:\s*",
        "",
        location.strip(),
        flags=re.IGNORECASE,
    )
    location = re.sub(r"[.!?]+$", "", location)
    return location.strip(" ,")


async def extract_destination(user_query: str) -> str:
    """
    Extract the destination without returning the entire travel request.

    Deterministic parsing is attempted first so requests such as
    ``Plan a 7-day Japan trip from Hyderabad`` resolve to ``Tokyo, Japan``
    even when the LLM provider is temporarily unavailable.
    """
    query = " ".join(user_query.strip().split())

    country_defaults = {
        "japan": "Tokyo, Japan",
        "france": "Paris, France",
        "italy": "Rome, Italy",
        "thailand": "Bangkok, Thailand",
        "united arab emirates": "Dubai, United Arab Emirates",
        "uae": "Dubai, United Arab Emirates",
        "indonesia": "Bali, Indonesia",
        "singapore": "Singapore",
        "united kingdom": "London, United Kingdom",
        "uk": "London, United Kingdom",
        "united states": "New York, United States",
        "usa": "New York, United States",
        "australia": "Sydney, Australia",
        "south korea": "Seoul, South Korea",
        "switzerland": "Zurich, Switzerland",
        "maldives": "Malé, Maldives",
    }

    city_aliases = {
        "tokyo": "Tokyo, Japan",
        "kyoto": "Kyoto, Japan",
        "osaka": "Osaka, Japan",
        "paris": "Paris, France",
        "london": "London, United Kingdom",
        "dubai": "Dubai, United Arab Emirates",
        "rome": "Rome, Italy",
        "bangkok": "Bangkok, Thailand",
        "bali": "Bali, Indonesia",
        "new york": "New York, United States",
        "singapore": "Singapore",
        "sydney": "Sydney, Australia",
        "seoul": "Seoul, South Korea",
        "cape town": "Cape Town, South Africa",
        "visakhapatnam": "Visakhapatnam, India",
        "hyderabad": "Hyderabad, India",
        "bengaluru": "Bengaluru, India",
        "bangalore": "Bengaluru, India",
    }

    lowered = query.lower()

    # Prefer explicitly named destination cities.
    for alias, resolved in city_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            # Avoid selecting a city that is clearly the origin.
            origin_match = re.search(
                rf"\bfrom\s+{re.escape(alias)}\b",
                lowered,
            )
            destination_match = re.search(
                rf"\b(?:to|in|visit|trip\s+to)\s+{re.escape(alias)}\b",
                lowered,
            )

            if destination_match or not origin_match:
                return resolved

    # Handle country-style requests such as "Japan trip from Hyderabad".
    for country, resolved in country_defaults.items():
        if re.search(rf"\b{re.escape(country)}\b", lowered):
            return resolved

    explicit_patterns = (
        r"\b(?:travel|fly|go|trip)\s+to\s+([A-Za-z][A-Za-z\s,.'-]+?)(?=\s+(?:from|for|under|with|on)\b|$)",
        r"\bto\s+([A-Za-z][A-Za-z\s,.'-]+?)(?=\s+(?:from|for|under|with|on)\b|$)",
        r"\bin\s+([A-Za-z][A-Za-z\s,.'-]+?)(?=\s+(?:for|under|with|on)\b|$)",
        r"\bvisit\s+([A-Za-z][A-Za-z\s,.'-]+?)(?=\s+(?:for|under|with|on)\b|$)",
    )

    for expression in explicit_patterns:
        match = re.search(expression, query, flags=re.IGNORECASE)
        if match:
            candidate = clean_location_text(match.group(1))
            if candidate and len(candidate) <= 80:
                return candidate

    extraction_prompt = f"""
Extract only the primary destination from the travel request below.

Request:
{query}

Return only a city and country, for example:
Tokyo, Japan

Never return the full request. If only a country is given, choose its most
appropriate major tourist gateway city.
"""

    try:
        destination = await invoke_llm_markdown(
            "You extract one precise destination location.",
            extraction_prompt,
            timeout_seconds=35,
        )
        destination = clean_location_text(destination.splitlines()[0])

        if destination and len(destination) <= 80:
            return destination

    except Exception:
        logger.warning("LLM destination extraction failed", exc_info=True)

    raise ValueError(
        "Could not identify a destination. Include a city or country, "
        "for example: 'Tokyo, Japan'."
    )


# =============================================================================
# Weather parsing and formatting
# =============================================================================

def find_first_mapping(value: Any) -> dict[str, Any]:
    """Find the first dictionary inside a nested structure."""
    normalized = normalize_tool_result(value)

    if isinstance(normalized, dict):
        if "content" in normalized:
            nested = find_first_mapping(normalized["content"])
            if nested:
                return nested
        return normalized

    if isinstance(normalized, list):
        for item in normalized:
            found = find_first_mapping(item)
            if found:
                return found

    return {}


def locate_forecast_list(value: Any) -> list[dict[str, Any]]:
    """Locate a forecast array in a nested MCP response."""
    normalized = normalize_tool_result(value)

    if isinstance(normalized, dict):
        for key in ("forecast", "list", "items", "data", "hourly", "daily"):
            candidate = normalized.get(key)
            if isinstance(candidate, list):
                return [
                    item for item in candidate
                    if isinstance(item, dict)
                ]

        for item in normalized.values():
            found = locate_forecast_list(item)
            if found:
                return found

    if isinstance(normalized, list):
        if normalized and all(isinstance(item, dict) for item in normalized):
            return normalized
        for item in normalized:
            found = locate_forecast_list(item)
            if found:
                return found

    return []


def weather_icon(condition: str) -> str:
    lowered = condition.lower()
    if "thunder" in lowered:
        return "⛈️"
    if "rain" in lowered or "drizzle" in lowered:
        return "🌧️"
    if "snow" in lowered:
        return "❄️"
    if "clear" in lowered or "sun" in lowered:
        return "☀️"
    if "cloud" in lowered or "overcast" in lowered:
        return "☁️"
    if "mist" in lowered or "fog" in lowered or "haze" in lowered:
        return "🌫️"
    return "🌤️"


def readable_datetime(value: Any) -> str:
    """Convert common API timestamps into a compact readable label."""
    text = str(value or "").strip()
    if not text:
        return "Upcoming"

    normalized = text.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%d %b, %I:%M %p")
    except ValueError:
        return text


def format_weather_markdown(
    current_result: Any,
    forecast_result: Any,
    destination: str,
) -> str:
    """Create deterministic Markdown from common OpenWeather/MCP shapes."""
    current = find_first_mapping(current_result)
    forecast_items = locate_forecast_list(forecast_result)

    main_data = current.get("main") if isinstance(current.get("main"), dict) else {}
    wind_data = current.get("wind") if isinstance(current.get("wind"), dict) else {}
    weather_data = current.get("weather")

    weather_entry: dict[str, Any] = {}
    if (
        isinstance(weather_data, list)
        and weather_data
        and isinstance(weather_data[0], dict)
    ):
        weather_entry = weather_data[0]
    elif isinstance(weather_data, dict):
        weather_entry = weather_data

    city = (
        current.get("city")
        or current.get("location")
        or current.get("name")
        or destination
    )

    temperature = next(
        (
            value
            for value in (
                current.get("temperature_c"),
                current.get("temperature"),
                current.get("temp"),
                main_data.get("temp"),
            )
            if value is not None
        ),
        None,
    )
    feels_like = next(
        (
            value
            for value in (
                current.get("feels_like_c"),
                current.get("feels_like"),
                main_data.get("feels_like"),
            )
            if value is not None
        ),
        None,
    )
    humidity = next(
        (
            value
            for value in (
                current.get("humidity"),
                main_data.get("humidity"),
            )
            if value is not None
        ),
        None,
    )
    condition = next(
        (
            value
            for value in (
                current.get("condition"),
                current.get("description"),
                weather_entry.get("description"),
                weather_entry.get("main"),
            )
            if value not in (None, "")
        ),
        "Not available",
    )
    wind = next(
        (
            value
            for value in (
                current.get("wind_speed"),
                current.get("wind_mps"),
                wind_data.get("speed"),
            )
            if value is not None
        ),
        None,
    )

    condition_text = title_case_condition(condition)
    icon = weather_icon(condition_text)

    lines = [
        f"## {icon} Weather in {city}",
        "",
        "### Current conditions",
        "",
        "| Metric | Reading |",
        "|---|---|",
        f"| 🌡️ Temperature | **{safe_number(temperature)}°C** |",
        f"| 🤗 Feels like | **{safe_number(feels_like)}°C** |",
        f"| {icon} Condition | **{condition_text}** |",
        f"| 💧 Humidity | **{safe_number(humidity, 0)}%** |",
        f"| 💨 Wind speed | **{safe_number(wind)} m/s** |",
    ]

    if forecast_items:
        lines.extend(
            [
                "",
                "### Upcoming forecast",
                "",
                "| Date and time | Temperature | Conditions |",
                "|---|---:|---|",
            ]
        )

        for item in forecast_items[:8]:
            item_main = (
                item.get("main")
                if isinstance(item.get("main"), dict)
                else {}
            )
            item_weather = item.get("weather")
            item_weather_entry: dict[str, Any] = {}

            if (
                isinstance(item_weather, list)
                and item_weather
                and isinstance(item_weather[0], dict)
            ):
                item_weather_entry = item_weather[0]
            elif isinstance(item_weather, dict):
                item_weather_entry = item_weather

            timestamp = (
                item.get("datetime")
                or item.get("date")
                or item.get("time")
                or item.get("dt_txt")
            )
            item_temperature = next(
                (
                    value
                    for value in (
                        item.get("temperature"),
                        item.get("temperature_c"),
                        item.get("temp"),
                        item_main.get("temp"),
                    )
                    if value is not None
                ),
                None,
            )
            item_condition = next(
                (
                    value
                    for value in (
                        item.get("condition"),
                        item.get("description"),
                        item_weather_entry.get("description"),
                        item_weather_entry.get("main"),
                    )
                    if value not in (None, "")
                ),
                "Not available",
            )
            condition_label = title_case_condition(item_condition)
            lines.append(
                f"| {readable_datetime(timestamp)} | "
                f"**{safe_number(item_temperature)}°C** | "
                f"{weather_icon(condition_label)} {condition_label} |"
            )

    recommendations: list[str] = []
    lowered = condition_text.lower()

    if "rain" in lowered:
        recommendations.append(
            "Carry a compact umbrella or lightweight rain jacket."
        )

    try:
        if humidity is not None and float(humidity) >= 70:
            recommendations.append(
                "Expect humid conditions; wear breathable clothing and stay hydrated."
            )
    except (TypeError, ValueError):
        pass

    try:
        if temperature is not None and float(temperature) >= 28:
            recommendations.append(
                "Use sunscreen and avoid prolonged outdoor activity at midday."
            )
        elif temperature is not None and float(temperature) <= 12:
            recommendations.append(
                "Pack warm layers, especially for mornings and evenings."
            )
    except (TypeError, ValueError):
        pass

    if not recommendations:
        recommendations.append(
            "Check the forecast again before departure because conditions can change."
        )

    lines.extend(["", "### Traveller advice", ""])
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(
        [
            "",
            "> Weather values come from the live weather tool when available. "
            "Recheck close to departure for the latest conditions.",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Hotel parsing and formatting
# =============================================================================

def extract_search_results(value: Any) -> list[dict[str, Any]]:
    """Extract Tavily-style search results from nested tool output."""
    normalized = normalize_tool_result(value)

    if isinstance(normalized, dict):
        results = normalized.get("results")
        if isinstance(results, list):
            return [
                item for item in results
                if isinstance(item, dict)
            ]

        if {"title", "url"} & set(normalized):
            return [normalized]

        for item in normalized.values():
            found = extract_search_results(item)
            if found:
                return found

    if isinstance(normalized, list):
        direct = [
            item for item in normalized
            if isinstance(item, dict)
            and ("title" in item or "url" in item)
        ]
        if direct:
            return direct

        for item in normalized:
            found = extract_search_results(item)
            if found:
                return found

    return []


def format_search_sources(results: list[dict[str, Any]], limit: int = 5) -> str:
    """Create a compact source section without exposing raw search JSON."""
    lines: list[str] = []

    for result in results[:limit]:
        title = str(result.get("title") or "Travel source").strip()
        url = str(result.get("url") or "").strip()

        if url:
            lines.append(f"- [{title}]({url}) — {source_domain(url)}")

    if not lines:
        return ""

    return "\n".join(["### Sources consulted", "", *lines])


def fallback_hotel_markdown(
    destination: str,
    results: list[dict[str, Any]],
) -> str:
    """Readable fallback when hotel summarization by the LLM fails."""
    lines = [
        f"## 🏨 Hotel research for {destination}",
        "",
        "Live hotel search completed, but a structured recommendation "
        "could not be generated. These are the most relevant sources:",
        "",
    ]

    for index, result in enumerate(results[:5], start=1):
        title = str(result.get("title") or f"Hotel source {index}").strip()
        content = str(result.get("content") or "").strip()
        url = str(result.get("url") or "").strip()

        lines.append(f"### {index}. {title}")
        if content:
            lines.append(content[:500].strip())
        if url:
            lines.append(f"[Open source]({url})")
        lines.append("")

    lines.append(
        "> Prices and ratings change frequently. Confirm the final amount, "
        "taxes and cancellation policy before booking."
    )

    return "\n".join(lines)


# =============================================================================
# Dynamic tool invocation
# =============================================================================

def build_weather_tool_input(
    tool: Any,
    user_query: str,
    destination: str,
) -> dict[str, Any]:
    """Build weather arguments using only the normalized destination."""
    del user_query

    schema = get_tool_schema(tool)
    properties = schema.get("properties", {})
    lookup = {str(name).lower(): name for name in properties}

    parts = [part.strip() for part in destination.split(",", maxsplit=1)]
    city = parts[0]
    country = parts[1] if len(parts) > 1 else ""

    for field in (
        "location",
        "place",
        "destination",
        "location_name",
        "address",
    ):
        if field in lookup:
            return {lookup[field]: destination}

    city_field = lookup.get("city") or lookup.get("city_name")
    country_field = lookup.get("country") or lookup.get("country_name")
    if city_field:
        payload: dict[str, Any] = {city_field: city}
        if country_field and country:
            payload[country_field] = country
        return payload

    for field in ("query", "prompt", "question", "request", "text"):
        if field in lookup:
            return {lookup[field]: destination}

    required = schema.get("required", [])
    if not required:
        return {"location": destination}

    raise ValueError(
        "Could not build a normalized location input for weather tool "
        f"'{getattr(tool, 'name', 'unknown')}'."
    )


def extract_origin(user_query: str) -> str:
    """Extract the origin following the word 'from'."""
    match = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z\s,.'-]+?)"
        r"(?=\s+(?:to|for|under|with|on|during|in)\b|$)",
        user_query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    return clean_location_text(match.group(1))


def airport_code_for_location(location: str) -> str:
    """Return a practical IATA gateway for common portfolio test locations."""
    lowered = location.lower()
    airport_codes = {
        "hyderabad": "HYD",
        "bengaluru": "BLR",
        "bangalore": "BLR",
        "visakhapatnam": "VTZ",
        "delhi": "DEL",
        "mumbai": "BOM",
        "chennai": "MAA",
        "kolkata": "CCU",
        "tokyo": "NRT",
        "japan": "NRT",
        "paris": "CDG",
        "london": "LHR",
        "dubai": "DXB",
        "rome": "FCO",
        "bangkok": "BKK",
        "bali": "DPS",
        "singapore": "SIN",
        "sydney": "SYD",
        "seoul": "ICN",
        "new york": "JFK",
        "cape town": "CPT",
    }

    for name, code in airport_codes.items():
        if name in lowered:
            return code
    return ""


def build_flight_tool_input(
    tool: Any,
    user_query: str,
    destination: str,
) -> dict[str, Any]:
    """Build AviationStack arguments from the tool schema."""
    schema = get_tool_schema(tool)
    properties = schema.get("properties", {})
    lookup = {str(name).lower(): name for name in properties}
    required = [str(item) for item in schema.get("required", [])]

    origin = extract_origin(user_query)
    origin_code = airport_code_for_location(origin)
    destination_code = airport_code_for_location(destination)

    payload: dict[str, Any] = {}

    field_values = {
        "query": user_query,
        "prompt": user_query,
        "request": user_query,
        "question": user_query,
        "search": user_query,
        "route": f"{origin} to {destination}".strip(),
        "origin": origin,
        "from": origin,
        "departure": origin,
        "departure_city": origin,
        "destination": destination,
        "to": destination,
        "arrival": destination,
        "arrival_city": destination,
        "dep_iata": origin_code,
        "departure_iata": origin_code,
        "origin_iata": origin_code,
        "arr_iata": destination_code,
        "arrival_iata": destination_code,
        "destination_iata": destination_code,
        "limit": 10,
    }

    for normalized_name, value in field_values.items():
        actual_name = lookup.get(normalized_name)
        if actual_name and value not in ("", None):
            payload[actual_name] = value

    missing_required = [
        field
        for field in required
        if field not in payload
    ]
    if missing_required:
        raise ValueError(
            f"Unsupported required fields for '{getattr(tool, 'name', 'unknown')}': "
            + ", ".join(missing_required)
        )

    return payload


async def invoke_flight_tool(
    tool: Any,
    user_query: str,
    destination: str,
) -> Any:
    """Invoke an AviationStack tool with schema-aware arguments."""
    payload = build_flight_tool_input(tool, user_query, destination)
    return await tool.ainvoke(payload)


async def invoke_search_tool(tool: Any, query: str) -> Any:
    """Invoke Tavily or another search tool across schema variants."""
    schema = get_tool_schema(tool)
    properties = schema.get("properties", {})
    lookup = {str(name).lower(): name for name in properties}

    for field in ("query", "search_query", "q", "input", "text", "prompt"):
        if field in lookup:
            return await tool.ainvoke({lookup[field]: query})

    if not properties:
        try:
            return await tool.ainvoke({"query": query})
        except Exception:
            return await tool.ainvoke(query)

    required = schema.get("required", [])
    if not required:
        return await tool.ainvoke({"query": query})

    raise ValueError(
        f"Could not build search input for '{getattr(tool, 'name', 'unknown')}'."
    )


def tool_text_is_failure(text: str) -> bool:
    """Detect friendly MCP error Markdown so it is not treated as live data."""
    lowered = text.lower()
    markers = (
        "information unavailable",
        "service is not configured",
        "request timed out",
        "connection unavailable",
        "could not connect",
        "invalid, inactive",
        "no live weather information",
        "unexpected weather-service error",
    )
    return any(marker in lowered for marker in markers)


# =============================================================================
# Agent prompts
# =============================================================================

FLIGHT_AGENT_PROMPT = """
You are a senior international flight advisor creating a polished travel report.

USER REQUEST
{query}

NORMALIZED AVIATION TOOL DATA
{flight_data}

Return clean Markdown only.

Required structure:

## ✈️ Flight guidance
### Route overview
A compact table with origin, destination, likely airports and estimated duration.

### Airline and connection options
Use concise bullets. Mention only information supported by the data or clearly
label general guidance as an estimate.

### Fare guidance
Give a sensible estimate only when justified. Never present an estimate as a
live fare.

### Booking strategy
Provide 3 to 5 practical recommendations.

### Important note
Explain any data limitation in one short blockquote.

Never output JSON, Python objects, content blocks, tool metadata, IDs, scores,
request IDs or raw API responses.
"""

HOTEL_AGENT_PROMPT = """
You are a premium hotel research specialist.

USER REQUEST
{user_query}

DESTINATION
{destination}

NORMALIZED SEARCH RESULTS
{hotel_data}

Create attractive Markdown only.

Required structure:

## 🏨 Recommended stays in {destination}

Start with one sentence explaining the selection.

For 3 to 5 useful options, use:

### 1. Hotel or property name
| Detail | Information |
|---|---|
| 📍 Area | ... |
| 💰 Typical price | ... or "Check live price" |
| ⭐ Rating | ... or "Not verified" |
| 👤 Best for | ... |

**Why it stands out**
- ...
- ...

**Nearby**
- ...

[Check availability](source URL)

Then include:

### How to choose
Provide practical area and booking advice.

> Prices, availability and ratings can change. Verify taxes, cancellation terms
> and the final total before payment.

Rules:
- Use only grounded details from the supplied search results.
- If a result is a travel article rather than an actual hotel, use it only for
  area guidance, not as a hotel recommendation.
- Never expose raw JSON, result scores, request IDs, content blocks or tool data.
"""

ITINERARY_PROMPT = """
You are a world-class itinerary designer.

USER REQUEST
{user_query}

DESTINATION
{destination}

FLIGHT GUIDANCE
{flight_results}

HOTEL GUIDANCE
{hotel_results}

WEATHER GUIDANCE
{weather_results}

Return polished Markdown only.

Required structure:

## 🗺️ Trip overview
A compact summary table with destination, suggested duration, travel style and
budget positioning.

## 🗓️ Day-by-day itinerary
Create realistic daily sections with:
- Morning
- Afternoon
- Evening
- Approximate local transport guidance
- One practical tip

## 💰 Budget framework
Provide a clear estimate table, marking every amount as approximate.

## 🎒 Packing checklist
Tailor it to the destination and weather.

## 📄 Before you travel
List documents and requirements the traveller must verify through official
sources.

## 💡 Smart travel tips
Provide concise, high-value advice.

Do not repeat raw tool output. Do not invent exact live prices, schedules,
weather values or legal requirements.
"""

FINAL_AGENT_PROMPT = """
You are the final quality-control and synthesis agent for an AI travel planner.

USER REQUEST
{user_query}

DESTINATION
{destination}

FLIGHT REPORT
{flight_results}

HOTEL REPORT
{hotel_results}

WEATHER REPORT
{weather_results}

ITINERARY
{itinerary}

Produce an executive-quality final travel plan in clean Markdown.

Use this structure:

# ✈️ Your AI Travel Plan: {destination}

A short personalised introduction.

## ✅ Best overall recommendation
Summarise the ideal route, stay area, travel pace and key weather consideration.

## 📌 Trip at a glance
Use a concise table.

## 🧭 Recommended plan
Summarise the strongest itinerary choices without duplicating every detail.

## 💰 Budget snapshot
Use approximate ranges and clearly label estimates.

## ⚠️ Important checks before booking
Include live fare, hotel availability, visa/passport, insurance and weather
verification where relevant.

## 🚀 Next actions
Provide a numbered checklist of what the traveller should do next.

Never output JSON, code, internal state, tool names, IDs, scores or raw API data.
"""


# =============================================================================
# Agent nodes
# =============================================================================

async def flight_agent(state: TravelState) -> dict[str, Any]:
    """Collect aviation data, with Tavily fallback when MCP is unavailable."""
    query = state["user_query"]
    destination = state.get("destination") or await extract_destination(query)
    errors = list(state.get("errors", []))

    try:
        tool_data = await get_agent_tools()
        flight_tools = tool_data.get("flight_tools", [])
        fallback_tools = tool_data.get("flight_fallback_tools", [])

        outputs: list[dict[str, Any]] = []

        for tool in flight_tools[:4]:
            tool_name = str(getattr(tool, "name", "flight_tool"))
            try:
                result = await invoke_flight_tool(tool, query, destination)
                normalized = normalize_tool_result(result)
                if normalized not in (None, "", [], {}):
                    outputs.append(
                        {
                            "source": "AviationStack MCP",
                            "tool": tool_name,
                            "data": normalized,
                        }
                    )
            except Exception as exc:
                logger.warning("Flight tool %s failed: %s", tool_name, exc)

        if not outputs and fallback_tools:
            fallback_query = (
                f"Current flight routes and typical fare guidance to "
                f"{destination}. User request: {query}. "
                "Use reputable airline and flight-search sources. "
                "Do not claim a fare is live unless explicitly supported."
            )
            fallback_result = await invoke_search_tool(
                fallback_tools[0],
                fallback_query,
            )
            search_results = extract_search_results(fallback_result)

            if search_results:
                outputs.append(
                    {
                        "source": "Web-search fallback",
                        "data": [
                            {
                                "title": item.get("title"),
                                "url": item.get("url"),
                                "content": str(item.get("content") or "")[:1200],
                            }
                            for item in search_results[:8]
                        ],
                    }
                )

        if not outputs:
            tool_error = tool_data.get("tool_errors", {}).get("flight")
            raise RuntimeError(
                tool_error
                or "No aviation or fallback flight information was available."
            )

        flight_results = await invoke_llm_markdown(
            (
                "You turn aviation and travel-search data into concise, "
                "accurate, traveller-friendly Markdown. Clearly label estimates."
            ),
            FLIGHT_AGENT_PROMPT.format(
                query=query,
                flight_data=compact_json(outputs),
            ),
        )

    except Exception as exc:
        logger.exception("Flight agent failed")
        errors.append(f"Flight Agent: {exc}")
        flight_results = f"""## ✈️ Flight guidance to {destination}

Live airline inventory could not be verified during this run.

### Recommended search approach

- Compare the same route on official airline websites and a trusted flight-search platform.
- Check one-stop options as well as direct flights.
- Compare the final total including baggage, seat selection and payment fees.
- Avoid treating an estimated fare as a guaranteed live price.

> The Hotel, Weather and Itinerary agents continued building the trip.
"""

    return {
        "destination": destination,
        "flight_results": flight_results,
        "messages": [AIMessage(content="Flight report completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "errors": errors,
    }


async def hotel_agent(state: TravelState) -> dict[str, Any]:
    """Research hotels and return readable recommendations."""
    user_query = state["user_query"]
    destination = state.get("destination") or await extract_destination(user_query)
    errors = list(state.get("errors", []))

    try:
        tool_data = await get_agent_tools()
        hotel_tools = tool_data.get("hotel_tools", [])

        if not hotel_tools:
            raise RuntimeError("No Tavily hotel search tool was loaded.")

        hotel_tool = hotel_tools[0]
        hotel_query = (
            f"Best hotels and accommodation in {destination} for: {user_query}. "
            "Prioritize actual hotel/property pages and reliable travel sources. "
            "Find property name, neighborhood, price guidance, rating, amenities, "
            "nearby landmarks and source URL."
        )

        result = await invoke_search_tool(hotel_tool, hotel_query)
        search_results = extract_search_results(result)

        if not search_results:
            raise RuntimeError("Hotel search returned no structured results.")

        compact_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": str(item.get("content") or "")[:1200],
            }
            for item in search_results[:8]
        ]

        response = await get_llm().ainvoke(
            [
                SystemMessage(
                    content=(
                        "You create grounded hotel recommendations from web "
                        "search results without exposing raw search data."
                    )
                ),
                HumanMessage(
                    content=HOTEL_AGENT_PROMPT.format(
                        user_query=user_query,
                        destination=destination,
                        hotel_data=compact_json(compact_results),
                    )
                ),
            ]
        )

        hotel_results = clean_markdown(response)
        sources = format_search_sources(search_results)

        if sources and "### Sources consulted" not in hotel_results:
            hotel_results = f"{hotel_results}\n\n{sources}"

    except Exception as exc:
        logger.exception("Hotel agent failed")
        errors.append(f"Hotel Agent: {exc}")

        try:
            search_results
        except UnboundLocalError:
            search_results = []

        if search_results:
            hotel_results = fallback_hotel_markdown(
                destination,
                search_results,
            )
        else:
            hotel_results = (
                f"## 🏨 Hotel research for {destination}\n\n"
                "Live hotel search is temporarily unavailable. "
                "The rest of the trip plan was still generated.\n\n"
                "> Compare properties by neighbourhood, total price including "
                "taxes, guest reviews and cancellation policy."
            )

    return {
        "destination": destination,
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel report completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "errors": errors,
    }


async def weather_agent(state: TravelState) -> dict[str, Any]:
    """
    Fetch current weather and forecast and preserve clean MCP Markdown.

    The updated weather MCP server returns ready-to-render Markdown. Older
    dictionary responses remain supported through ``format_weather_markdown``.
    """
    user_query = state["user_query"]
    destination = state.get("destination") or await extract_destination(user_query)
    errors = list(state.get("errors", []))

    current_result: Any = None
    forecast_result: Any = None

    try:
        tool_data = await get_agent_tools()
        weather_tools = tool_data.get("weather_tools", [])

        if not weather_tools:
            tool_error = tool_data.get("tool_errors", {}).get("weather")
            raise RuntimeError(tool_error or "No OpenWeather MCP tools were loaded.")

        current_tool = find_tool(
            weather_tools,
            preferred_names=(
                "get_current_weather",
                "get_weather",
                "current_weather",
                "weather",
            ),
            keywords=("current", "weather", "temperature"),
        )
        forecast_tool = find_tool(
            weather_tools,
            preferred_names=(
                "get_forecast",
                "weather_forecast",
                "forecast",
            ),
            keywords=("forecast",),
        )

        sections: list[str] = []

        if current_tool:
            current_input = build_weather_tool_input(
                current_tool,
                user_query,
                destination,
            )
            if not current_input:
                raise ValueError("Could not build current-weather tool input.")

            current_result = await current_tool.ainvoke(current_input)
            current_text = extract_mcp_text(current_result)

            if current_text and not tool_text_is_failure(current_text):
                sections.append(current_text)
            elif current_text:
                errors.append(
                    "Weather MCP current conditions returned an unavailable response."
                )

        if forecast_tool and forecast_tool is not current_tool:
            forecast_input = build_weather_tool_input(
                forecast_tool,
                user_query,
                destination,
            )

            if forecast_input:
                forecast_result = await forecast_tool.ainvoke(forecast_input)
                forecast_text = extract_mcp_text(forecast_result)

                if forecast_text and not tool_text_is_failure(forecast_text):
                    sections.append(forecast_text)
                elif forecast_text:
                    errors.append(
                        "Weather MCP forecast returned an unavailable response."
                    )

        if not sections:
            raise RuntimeError("Weather tools returned no readable information.")

        combined = "\n\n---\n\n".join(sections).strip()

        if looks_like_markdown(combined):
            weather_results = combined
        else:
            weather_results = format_weather_markdown(
                current_result=current_result,
                forecast_result=forecast_result,
                destination=destination,
            )

        if tool_text_is_failure(weather_results):
            raise RuntimeError(
                "Weather provider returned an unavailable response."
            )

        unavailable_count = weather_results.lower().count("not available")
        if unavailable_count >= 2:
            raise RuntimeError(
                "Weather response did not contain enough valid measurements."
            )

    except Exception as exc:
        logger.exception("Weather agent failed")
        errors.append(f"Weather Agent: {exc}")
        weather_results = f"""## 🌦️ Weather in {destination}

Live weather values could not be retrieved for this run.

### 🎒 Traveller advice

- Check the latest forecast shortly before departure.
- Pack flexible layers and a compact umbrella.
- Keep indoor alternatives for weather-sensitive activities.

> The rest of the travel workflow continued normally.
"""

    return {
        "destination": destination,
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather report completed.")],
        "llm_calls": state.get("llm_calls", 0),
        "errors": errors,
    }


async def itinerary_agent(state: TravelState) -> dict[str, Any]:
    """Combine specialist reports into a detailed, resilient itinerary."""
    errors = list(state.get("errors", []))
    destination = state.get("destination") or "the destination"

    try:
        prompt = ITINERARY_PROMPT.format(
            user_query=state["user_query"],
            destination=destination,
            flight_results=state.get("flight_results", "")[:6000],
            hotel_results=state.get("hotel_results", "")[:6000],
            weather_results=state.get("weather_results", "")[:5000],
        )

        itinerary = await invoke_llm_markdown(
            (
                "You create realistic, structured international itineraries. "
                "Return polished Markdown only."
            ),
            prompt,
        )

    except Exception as exc:
        logger.exception("Itinerary agent failed")
        errors.append(f"Itinerary Agent: {exc}")
        itinerary = build_fallback_itinerary(
            destination,
            state["user_query"],
        )

    return {
        "itinerary": itinerary,
        "messages": [AIMessage(content="Itinerary completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "errors": errors,
    }


async def final_agent(state: TravelState) -> dict[str, Any]:
    """Produce the polished executive summary displayed by Streamlit."""
    errors = list(state.get("errors", []))
    destination = state.get("destination") or "the destination"

    try:
        prompt = FINAL_AGENT_PROMPT.format(
            user_query=state["user_query"],
            destination=destination,
            flight_results=state.get("flight_results", "")[:4500],
            hotel_results=state.get("hotel_results", "")[:4500],
            weather_results=state.get("weather_results", "")[:3500],
            itinerary=state.get("itinerary", "")[:9000],
        )

        final_response = await invoke_llm_markdown(
            (
                "You are the final editorial and quality-control agent for a "
                "premium AI travel planner. Return polished Markdown only."
            ),
            prompt,
        )

    except Exception as exc:
        logger.exception("Final agent failed")
        errors.append(f"Final Agent: {exc}")
        final_response = build_fallback_final_plan(
            destination=destination,
            user_query=state["user_query"],
            itinerary=state.get("itinerary", "")
            or build_fallback_itinerary(destination, state["user_query"]),
        )

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "errors": errors,
    }


# =============================================================================
# Build LangGraph
# =============================================================================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


# =============================================================================
# Command-line test
# =============================================================================

async def command_line_test() -> None:
    """Run the graph without Streamlit for local verification."""
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    user_input = input("Enter travel request: ").strip()
    if not user_input:
        print("Travel request cannot be empty.")
        return

    destination = await extract_destination(user_input)

    initial_state: TravelState = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "destination": destination,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0,
        "errors": [],
    }

    result = await app.ainvoke(initial_state, config=config)

    print("\n" + "=" * 80)
    print("FINAL TRAVEL PLAN")
    print("=" * 80 + "\n")
    print(result.get("final_response", result.get("itinerary", "")))

    if result.get("errors"):
        print("\n" + "-" * 80)
        print("NON-FATAL AGENT WARNINGS")
        print("-" * 80)
        for error in result["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    asyncio.run(command_line_test())