from __future__ import annotations

import asyncio
import json
import logging
import operator
import os
import re
import uuid
from datetime import datetime
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

load_dotenv(override=False)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Create the Groq client lazily so importing main.py never crashes."""
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env file or "
            "Streamlit Community Cloud secrets."
        )

    _llm = ChatGroq(
        model=GROQ_MODEL,
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


async def get_agent_tools() -> dict[str, Any]:
    """Load and cache MCP/search tools without import-time side effects.

    Importing mcp_client lazily prevents the whole Streamlit application from
    failing when an optional MCP dependency or server is unavailable.
    """
    global _loaded_tool_data

    if _loaded_tool_data is not None:
        return _loaded_tool_data

    try:
        from mcp_client import load_all_tools

        logger.info("Loading MCP and search tools...")
        loaded = await load_all_tools()
        _loaded_tool_data = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.exception("MCP/search tool loading failed")
        _loaded_tool_data = {
            "flight_tools": [],
            "hotel_tools": [],
            "weather_tools": [],
            "tool_loading_error": str(exc),
        }

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
    """Normalize model/tool output into clean display-ready Markdown."""
    normalized = normalize_tool_result(text)

    if isinstance(normalized, dict):
        for key in (
            "final_response",
            "final_answer",
            "travel_plan",
            "output",
            "answer",
            "content",
            "text",
        ):
            if normalized.get(key):
                return clean_markdown(normalized[key])
        value = json.dumps(normalized, ensure_ascii=False, default=str)
    elif isinstance(normalized, list):
        parts = [clean_markdown(item) for item in normalized]
        value = "\n\n".join(part for part in parts if part)
    else:
        value = str(normalized or "")

    value = value.replace("\\n", "\n").replace("\\t", " ")
    value = value.strip()

    if value.startswith("```markdown"):
        value = value[len("```markdown"):].strip()
    elif value.startswith("```"):
        value = value[3:].strip()

    if value.endswith("```"):
        value = value[:-3].strip()

    return value.strip().strip('"').strip("'")


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
    """Extract the primary destination from a worldwide travel request."""
    extraction_prompt = f"""
Extract the main destination city and country from this travel request.

Request:
{user_query}

Return only one location in this format:
City, Country

Rules:
- If both origin and destination exist, return the destination.
- Prefer a city rather than only a country.
- Do not add explanation or markdown.
"""

    try:
        response = await get_llm().ainvoke(
            [
                SystemMessage(
                    content="You extract precise travel destinations."
                ),
                HumanMessage(content=extraction_prompt),
            ]
        )
        destination = clean_location_text(clean_markdown(response))
        if destination and len(destination) <= 100:
            return destination
    except Exception:
        logger.warning("LLM destination extraction failed", exc_info=True)

    patterns = [
        r"\bto\s+([A-Za-z][A-Za-z\s,.'-]+?)(?:\s+from\s+|\s+for\s+|\s+under\s+|$)",
        r"\bin\s+([A-Za-z][A-Za-z\s,.'-]+?)(?:\s+for\s+|\s+under\s+|$)",
        r"\bvisit\s+([A-Za-z][A-Za-z\s,.'-]+?)(?:\s+for\s+|\s+under\s+|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            return clean_location_text(match.group(1))

    return clean_location_text(user_query[:100])


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
    """Create deterministic, attractive Markdown from weather MCP data."""
    current = find_first_mapping(current_result)
    forecast_items = locate_forecast_list(forecast_result)

    city = (
        current.get("city")
        or current.get("location")
        or current.get("name")
        or destination
    )
    temperature = (
        current.get("temperature_c")
        or current.get("temperature")
        or current.get("temp")
    )
    feels_like = (
        current.get("feels_like_c")
        or current.get("feels_like")
    )
    humidity = current.get("humidity")
    condition = (
        current.get("condition")
        or current.get("weather")
        or current.get("description")
        or "Not available"
    )
    wind = (
        current.get("wind_speed")
        or current.get("wind")
        or current.get("wind_mps")
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
            timestamp = (
                item.get("datetime")
                or item.get("date")
                or item.get("time")
                or item.get("dt_txt")
            )
            item_temperature = (
                item.get("temperature")
                or item.get("temperature_c")
                or item.get("temp")
            )
            item_condition = (
                item.get("weather")
                or item.get("condition")
                or item.get("description")
                or "Not available"
            )
            condition_label = title_case_condition(item_condition)
            lines.append(
                f"| {readable_datetime(timestamp)} | "
                f"**{safe_number(item_temperature)}°C** | "
                f"{weather_icon(condition_label)} {condition_label} |"
            )

    lowered = condition_text.lower()
    recommendations: list[str] = []

    if "rain" in lowered or any(
        "rain" in str(item.get("weather", "")).lower()
        for item in forecast_items[:8]
    ):
        recommendations.append(
            "Carry a compact umbrella or lightweight rain jacket."
        )
    if humidity is not None:
        try:
            if float(humidity) >= 70:
                recommendations.append(
                    "Expect humid conditions; choose breathable clothing and stay hydrated."
                )
        except (TypeError, ValueError):
            pass
    try:
        if temperature is not None and float(temperature) >= 28:
            recommendations.append(
                "Use sunscreen and avoid extended outdoor activity during the hottest hours."
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
            "> Weather values are live tool data when available. "
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
    """Build arguments dynamically from a weather tool's schema."""
    schema = get_tool_schema(tool)
    properties = schema.get("properties", {})
    lookup = {str(name).lower(): name for name in properties}

    for field in ("query", "prompt", "question", "request", "text"):
        if field in lookup:
            return {lookup[field]: user_query}

    for field in (
        "location",
        "city",
        "place",
        "city_name",
        "location_name",
        "destination",
        "address",
    ):
        if field in lookup:
            return {lookup[field]: destination}

    city_field = lookup.get("city")
    country_field = lookup.get("country")

    if city_field:
        parts = [
            item.strip()
            for item in destination.split(",", maxsplit=1)
        ]
        payload: dict[str, Any] = {city_field: parts[0]}
        if country_field and len(parts) > 1:
            payload[country_field] = parts[1]
        return payload

    return {}


async def invoke_flight_tool(tool: Any, user_query: str) -> Any:
    """Invoke different AviationStack tool schemas safely."""
    schema = get_tool_schema(tool)
    properties = schema.get("properties", {})

    if not properties:
        return await tool.ainvoke({})

    lookup = {str(name).lower(): name for name in properties}

    for field in (
        "query",
        "prompt",
        "request",
        "question",
        "route",
        "search",
    ):
        if field in lookup:
            return await tool.ainvoke({lookup[field]: user_query})

    required = schema.get("required", [])
    if not required:
        return await tool.ainvoke({})

    raise ValueError(
        f"Could not build input for flight tool '{tool.name}'."
    )


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
    """Collect aviation data and produce a polished flight report."""
    query = state["user_query"]
    errors = list(state.get("errors", []))

    try:
        tool_data = await get_agent_tools()
        flight_tools = tool_data.get("flight_tools", [])

        if not flight_tools:
            raise RuntimeError("No AviationStack MCP tools were loaded.")

        outputs: list[dict[str, Any]] = []

        for tool in flight_tools[:4]:
            tool_name = str(getattr(tool, "name", "flight_tool"))
            try:
                result = await invoke_flight_tool(tool, query)
                outputs.append(
                    {
                        "tool": tool_name,
                        "data": normalize_tool_result(result),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Flight tool %s failed: %s",
                    tool_name,
                    exc,
                )
                outputs.append(
                    {
                        "tool": tool_name,
                        "status": "unavailable",
                    }
                )

        response = await get_llm().ainvoke(
            [
                SystemMessage(
                    content=(
                        "You turn aviation data into concise, accurate, "
                        "traveller-friendly Markdown."
                    )
                ),
                HumanMessage(
                    content=FLIGHT_AGENT_PROMPT.format(
                        query=query,
                        flight_data=compact_json(outputs),
                    )
                ),
            ]
        )
        flight_results = clean_markdown(response)

    except Exception as exc:
        logger.exception("Flight agent failed")
        errors.append(f"Flight Agent: {exc}")
        flight_results = (
            "## ✈️ Flight guidance\n\n"
            "Live aviation information is temporarily unavailable. "
            "The remaining agents continued building the trip.\n\n"
            "> Compare routes and prices directly with airlines or a trusted "
            "flight-search platform before booking."
        )

    return {
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

        result = await hotel_tool.ainvoke({"query": hotel_query})
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
    """Fetch and deterministically format weather data."""
    user_query = state["user_query"]
    destination = state.get("destination") or await extract_destination(user_query)
    errors = list(state.get("errors", []))

    current_result: Any = None
    forecast_result: Any = None

    try:
        tool_data = await get_agent_tools()
        weather_tools = tool_data.get("weather_tools", [])

        if not weather_tools:
            raise RuntimeError("No OpenWeather MCP tools were loaded.")

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

        if current_tool:
            current_input = build_weather_tool_input(
                current_tool,
                user_query,
                destination,
            )
            if current_input:
                current_result = await current_tool.ainvoke(current_input)

        if forecast_tool and forecast_tool is not current_tool:
            forecast_input = build_weather_tool_input(
                forecast_tool,
                user_query,
                destination,
            )
            if forecast_input:
                forecast_result = await forecast_tool.ainvoke(forecast_input)

        if current_result is None and forecast_result is None:
            raise RuntimeError("Weather tools returned no usable data.")

        weather_results = format_weather_markdown(
            current_result=current_result,
            forecast_result=forecast_result,
            destination=destination,
        )

    except Exception as exc:
        logger.exception("Weather agent failed")
        errors.append(f"Weather Agent: {exc}")
        weather_results = (
            f"## 🌦️ Weather in {destination}\n\n"
            "Live weather information is temporarily unavailable.\n\n"
            "### Traveller advice\n\n"
            "- Check an official weather service shortly before departure.\n"
            "- Pack flexible layers suitable for changing conditions.\n"
            "- Keep indoor alternatives for weather-sensitive activities."
        )

    return {
        "destination": destination,
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather report completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "errors": errors,
    }


async def itinerary_agent(state: TravelState) -> dict[str, Any]:
    """Combine all specialist reports into a detailed itinerary."""
    errors = list(state.get("errors", []))

    try:
        response = await get_llm().ainvoke(
            [
                SystemMessage(
                    content=(
                        "You create realistic, structured international "
                        "itineraries using supplied specialist reports."
                    )
                ),
                HumanMessage(
                    content=ITINERARY_PROMPT.format(
                        user_query=state["user_query"],
                        destination=state.get("destination", "the destination"),
                        flight_results=state.get("flight_results", ""),
                        hotel_results=state.get("hotel_results", ""),
                        weather_results=state.get("weather_results", ""),
                    )
                ),
            ]
        )
        itinerary = clean_markdown(response)

    except Exception as exc:
        logger.exception("Itinerary agent failed")
        errors.append(f"Itinerary Agent: {exc}")
        itinerary = (
            "## 🗓️ Suggested itinerary\n\n"
            "A detailed day-by-day itinerary could not be generated. "
            "Use the flight, hotel and weather sections above to continue "
            "planning manually."
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

    try:
        response = await get_llm().ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are the final editorial and quality-control agent. "
                        "Produce polished travel plans without raw tool output."
                    )
                ),
                HumanMessage(
                    content=FINAL_AGENT_PROMPT.format(
                        user_query=state["user_query"],
                        destination=state.get("destination", "the destination"),
                        flight_results=state.get("flight_results", ""),
                        hotel_results=state.get("hotel_results", ""),
                        weather_results=state.get("weather_results", ""),
                        itinerary=state.get("itinerary", ""),
                    )
                ),
            ]
        )
        final_response = clean_markdown(response)

    except Exception as exc:
        logger.exception("Final agent failed")
        errors.append(f"Final Agent: {exc}")
        final_response = state.get("itinerary", "") or (
            "# ✈️ Your AI Travel Plan\n\n"
            "The specialist reports were generated, but final synthesis "
            "is temporarily unavailable."
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
# Public application entry point
# =============================================================================

async def run_travel_planner(
    query: str,
    thread_id: str | None = None,
) -> str:
    """Run the complete multi-agent workflow and return final Markdown.

    This is the stable integration contract used by frontend.py, tests, and
    future API layers. Individual tool failures are handled inside each agent,
    so the workflow returns the best available plan whenever possible.
    """
    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("Travel request cannot be empty.")

    request_thread_id = thread_id or str(uuid.uuid4())
    destination = await extract_destination(clean_query)

    initial_state: TravelState = {
        "messages": [HumanMessage(content=clean_query)],
        "user_query": clean_query,
        "destination": destination,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "final_response": "",
        "llm_calls": 0,
        "errors": [],
    }

    config = {"configurable": {"thread_id": request_thread_id}}

    try:
        result = await app.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Travel planner workflow failed")
        raise RuntimeError(
            "The travel workflow could not start. Verify GROQ_API_KEY and "
            "the installed LangGraph/LangChain dependencies. "
            f"Technical cause: {exc}"
        ) from exc

    final_response = clean_markdown(
        result.get("final_response")
        or result.get("itinerary")
        or result.get("hotel_results")
        or result.get("flight_results")
        or result.get("weather_results")
    )

    if not final_response:
        warnings = result.get("errors", [])
        warning_text = "; ".join(str(item) for item in warnings)
        raise RuntimeError(
            "The workflow completed without a displayable travel plan."
            + (f" Non-fatal agent warnings: {warning_text}" if warning_text else "")
        )

    return final_response


def run_travel_planner_sync(
    query: str,
    thread_id: str | None = None,
) -> str:
    """Synchronous convenience wrapper for scripts and simple clients."""
    return asyncio.run(run_travel_planner(query, thread_id=thread_id))


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

    result = await run_travel_planner(
        user_input,
        thread_id=config["configurable"]["thread_id"],
    )

    print("\n" + "=" * 80)
    print("FINAL TRAVEL PLAN")
    print("=" * 80 + "\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(command_line_test())