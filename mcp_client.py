from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch

logger = logging.getLogger("mcp_client")

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

AVIATIONSTACK_PROJECT_DIR = BASE_DIR / "aviationstack-mcp"
AVIATIONSTACK_SRC = AVIATIONSTACK_PROJECT_DIR / "src"
AVIATIONSTACK_PACKAGE_DIR = AVIATIONSTACK_SRC / "aviationstack_mcp"

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"

MCP_STARTUP_TIMEOUT_SECONDS = 60
TAVILY_MAX_RESULTS = 8

load_dotenv(ENV_FILE, override=False)

_TOOL_CACHE: dict[str, Any] | None = None


def get_environment_value(*names: str) -> str | None:
    """Return the first non-empty configured environment variable."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def get_aviationstack_api_key() -> str | None:
    """Return the AviationStack API key using supported variable aliases."""
    return get_environment_value(
        "AVIATION_STACK_API_KEY",
        "AVIATIONSTACK_API_KEY",
    )


def get_openweather_api_key() -> str | None:
    """Return the OpenWeather API key using supported variable aliases."""
    return get_environment_value(
        "OPENWEATHER_API_KEY",
        "OPEN_WEATHER_API_KEY",
    )


def get_tavily_api_key() -> str | None:
    """Return the Tavily API key."""
    return get_environment_value("TAVILY_API_KEY")


def build_aviationstack_server_config(api_key: str) -> dict[str, Any]:
    """Build the AviationStack MCP stdio server configuration."""
    child_environment = os.environ.copy()
    existing_pythonpath = child_environment.get("PYTHONPATH", "").strip()

    pythonpath_parts = [str(AVIATIONSTACK_SRC)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    child_environment.update(
        {
            "AVIATION_STACK_API_KEY": api_key,
            "AVIATIONSTACK_API_KEY": api_key,
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "PYTHONUNBUFFERED": "1",
        }
    )

    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "aviationstack_mcp", "mcp", "run"],
        "env": child_environment,
        "cwd": str(AVIATIONSTACK_PROJECT_DIR),
    }


def build_weather_server_config(api_key: str) -> dict[str, Any]:
    """Build the custom OpenWeather MCP stdio server configuration."""
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "OPENWEATHER_API_KEY": api_key,
            "OPEN_WEATHER_API_KEY": api_key,
            "PYTHONUNBUFFERED": "1",
        }
    )

    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(WEATHER_SERVER_PATH)],
        "env": child_environment,
        "cwd": str(BASE_DIR),
    }


def get_tool_searchable_text(tool: Any) -> str:
    """Return normalized tool name and description text."""
    name = str(getattr(tool, "name", "") or "")
    description = str(getattr(tool, "description", "") or "")
    return f"{name} {description}".lower()


def classify_tools(tools: list[Any], keywords: tuple[str, ...]) -> list[Any]:
    """Return tools whose name or description contains a target keyword."""
    return [
        tool
        for tool in tools
        if any(
            keyword in get_tool_searchable_text(tool)
            for keyword in keywords
        )
    ]


def classify_flight_tools(tools: list[Any]) -> list[Any]:
    """Return tools related to aviation and flights."""
    return classify_tools(
        tools,
        (
            "flight",
            "aviation",
            "airport",
            "airline",
            "arrival",
            "departure",
            "route",
            "schedule",
        ),
    )


def classify_weather_tools(tools: list[Any]) -> list[Any]:
    """Return tools related to current weather and forecasts."""
    return classify_tools(
        tools,
        (
            "weather",
            "forecast",
            "temperature",
            "humidity",
            "rain",
            "wind",
            "climate",
        ),
    )


async def _get_server_tools(
    client: MultiServerMCPClient,
    server_name: str,
) -> list[Any]:
    """Load tools from one MCP server with compatibility fallbacks."""
    try:
        tools = await client.get_tools(server_name=server_name)
    except TypeError:
        tools = await client.get_tools()

    return list(tools or [])


async def load_flight_tools(
) -> tuple[MultiServerMCPClient | None, list[Any], str | None]:
    """Load AviationStack MCP flight tools."""
    api_key = get_aviationstack_api_key()

    if not api_key:
        return (
            None,
            [],
            "AVIATION_STACK_API_KEY is missing.",
        )

    required_paths = {
        "AviationStack project directory": AVIATIONSTACK_PROJECT_DIR,
        "AviationStack source directory": AVIATIONSTACK_SRC,
        "AviationStack package directory": AVIATIONSTACK_PACKAGE_DIR,
        "AviationStack module entry point": (
            AVIATIONSTACK_PACKAGE_DIR / "__main__.py"
        ),
    }

    for label, path in required_paths.items():
        if not path.exists():
            return None, [], f"{label} was not found: {path}"

    try:
        client = MultiServerMCPClient(
            {
                "aviationstack": build_aviationstack_server_config(api_key),
            }
        )

        tools = await asyncio.wait_for(
            _get_server_tools(client, "aviationstack"),
            timeout=MCP_STARTUP_TIMEOUT_SECONDS,
        )

        flight_tools = classify_flight_tools(tools)

        if not flight_tools and tools:
            logger.warning(
                "Tool classification found no explicit flight keyword; "
                "using all AviationStack tools."
            )
            flight_tools = tools

        if not flight_tools:
            return (
                client,
                [],
                "AviationStack MCP started but returned no tools.",
            )

        logger.info(
            "Loaded %d AviationStack tool(s): %s",
            len(flight_tools),
            ", ".join(
                str(getattr(tool, "name", "unknown"))
                for tool in flight_tools
            ),
        )
        return client, flight_tools, None

    except asyncio.TimeoutError:
        logger.exception("AviationStack MCP startup timed out")
        return (
            None,
            [],
            (
                "AviationStack MCP startup exceeded "
                f"{MCP_STARTUP_TIMEOUT_SECONDS} seconds."
            ),
        )
    except Exception as exc:
        logger.exception("Could not load AviationStack MCP tools")
        return (
            None,
            [],
            f"AviationStack MCP loading failed: {type(exc).__name__}: {exc}",
        )


async def load_weather_tools(
) -> tuple[MultiServerMCPClient | None, list[Any], str | None]:
    """Load custom OpenWeather MCP tools."""
    api_key = get_openweather_api_key()

    if not api_key:
        return None, [], "OPENWEATHER_API_KEY is missing."

    if not WEATHER_SERVER_PATH.is_file():
        return (
            None,
            [],
            f"Weather MCP server file was not found: {WEATHER_SERVER_PATH}",
        )

    try:
        client = MultiServerMCPClient(
            {
                "weather": build_weather_server_config(api_key),
            }
        )

        tools = await asyncio.wait_for(
            _get_server_tools(client, "weather"),
            timeout=MCP_STARTUP_TIMEOUT_SECONDS,
        )

        weather_tools = classify_weather_tools(tools)

        if not weather_tools and tools:
            logger.warning(
                "Tool classification found no explicit weather keyword; "
                "using all weather-server tools."
            )
            weather_tools = tools

        if not weather_tools:
            return client, [], "Weather MCP started but returned no tools."

        logger.info(
            "Loaded %d weather tool(s): %s",
            len(weather_tools),
            ", ".join(
                str(getattr(tool, "name", "unknown"))
                for tool in weather_tools
            ),
        )
        return client, weather_tools, None

    except asyncio.TimeoutError:
        logger.exception("Weather MCP startup timed out")
        return (
            None,
            [],
            (
                "Weather MCP startup exceeded "
                f"{MCP_STARTUP_TIMEOUT_SECONDS} seconds."
            ),
        )
    except Exception as exc:
        logger.exception("Could not load weather MCP tools")
        return (
            None,
            [],
            f"Weather MCP loading failed: {type(exc).__name__}: {exc}",
        )


def create_tavily_search_tool(
    *,
    tool_name: str,
    description: str,
) -> TavilySearch:
    """Create a named Tavily search tool with version-safe arguments."""
    api_key = get_tavily_api_key()
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured.")

    os.environ["TAVILY_API_KEY"] = api_key

    common_arguments: dict[str, Any] = {
        "max_results": TAVILY_MAX_RESULTS,
        "topic": "general",
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
    }

    try:
        return TavilySearch(
            name=tool_name,
            description=description,
            **common_arguments,
        )
    except (TypeError, ValueError):
        logger.warning(
            "Installed langchain-tavily version rejected custom tool "
            "metadata; creating TavilySearch with compatible arguments."
        )
        return TavilySearch(**common_arguments)


def load_flight_fallback_tools() -> tuple[list[Any], str | None]:
    """Load Tavily as a flight-route research fallback."""
    try:
        tool = create_tavily_search_tool(
            tool_name="search_flight_routes",
            description=(
                "Research flight routes, airlines, likely connections, "
                "airport options, and estimated fare ranges. Results are "
                "web research and must not be presented as live fares."
            ),
        )
        logger.info("Loaded Tavily flight fallback tool")
        return [tool], None
    except Exception as exc:
        logger.exception("Could not create Tavily flight fallback tool")
        return (
            [],
            f"Tavily flight fallback failed: {type(exc).__name__}: {exc}",
        )


def load_hotel_tools() -> tuple[list[Any], str | None]:
    """Load Tavily for grounded hotel research."""
    try:
        tool = create_tavily_search_tool(
            tool_name="search_hotels",
            description=(
                "Research hotels, neighbourhoods, indicative nightly "
                "prices, reviews, amenities, and cancellation guidance."
            ),
        )
        logger.info("Loaded Tavily hotel research tool")
        return [tool], None
    except Exception as exc:
        logger.exception("Could not create Tavily hotel search tool")
        return [], f"Tavily hotel tool failed: {type(exc).__name__}: {exc}"


def _copy_tool_data(tool_data: dict[str, Any]) -> dict[str, Any]:
    """Return a safe shallow copy of cached tool collections."""
    copied = dict(tool_data)
    for key in (
        "all_mcp_tools",
        "all_tools",
        "flight_tools",
        "flight_fallback_tools",
        "weather_tools",
        "hotel_tools",
    ):
        copied[key] = list(tool_data.get(key, []))

    copied["clients"] = dict(tool_data.get("clients", {}))
    copied["tool_errors"] = dict(tool_data.get("tool_errors", {}))
    copied["tool_status"] = dict(tool_data.get("tool_status", {}))
    copied["client"] = copied["clients"]
    return copied


async def load_all_tools(*, force_reload: bool = False) -> dict[str, Any]:
    """Load and cache all MCP and web-search tools independently."""
    global _TOOL_CACHE

    if _TOOL_CACHE is not None and not force_reload:
        return _copy_tool_data(_TOOL_CACHE)

    flight_result, weather_result = await asyncio.gather(
        load_flight_tools(),
        load_weather_tools(),
    )

    flight_client, flight_tools, flight_error = flight_result
    weather_client, weather_tools, weather_error = weather_result

    flight_fallback_tools, flight_fallback_error = (
        load_flight_fallback_tools()
    )
    hotel_tools, hotel_error = load_hotel_tools()

    clients = {
        "aviationstack": flight_client,
        "weather": weather_client,
    }

    tool_errors = {
        "flight": flight_error,
        "flight_fallback": flight_fallback_error,
        "weather": weather_error,
        "hotel": hotel_error,
    }

    all_mcp_tools = [*flight_tools, *weather_tools]
    all_tools = [
        *flight_tools,
        *flight_fallback_tools,
        *weather_tools,
        *hotel_tools,
    ]

    result = {
        "client": clients,
        "clients": clients,
        "all_mcp_tools": all_mcp_tools,
        "all_tools": all_tools,
        "flight_tools": flight_tools,
        "flight_fallback_tools": flight_fallback_tools,
        "weather_tools": weather_tools,
        "hotel_tools": hotel_tools,
        "tool_errors": tool_errors,
        "tool_status": {
            "aviationstack_available": bool(flight_tools),
            "flight_fallback_available": bool(flight_fallback_tools),
            "weather_available": bool(weather_tools),
            "hotel_search_available": bool(hotel_tools),
        },
    }

    _TOOL_CACHE = result
    return _copy_tool_data(result)


def clear_tool_cache() -> None:
    """Clear cached tool metadata for tests or manual reloads."""
    global _TOOL_CACHE
    _TOOL_CACHE = None


def print_tool_group(title: str, tools: list[Any]) -> None:
    """Print a diagnostic tool group without exposing credentials."""
    print(f"\n{title}:")
    if not tools:
        print("- No tools loaded")
        return

    for tool in tools:
        name = str(getattr(tool, "name", "unknown"))
        print(f"- {name}")


async def diagnostic_test() -> None:
    """Run a complete, credential-safe tool-loading diagnostic."""
    print("=" * 76)
    print("MULTI-AGENT TRAVEL SYSTEM - MCP CLIENT DIAGNOSTIC")
    print("=" * 76)

    print(f"\nPython executable: {sys.executable}")
    print(f"Project directory: {BASE_DIR}")
    print(f"AviationStack source: {AVIATIONSTACK_SRC}")
    print(f"Weather server: {WEATHER_SERVER_PATH}")

    print("\nEnvironment variables:")
    print(
        "- AviationStack API key: "
        + ("Configured" if get_aviationstack_api_key() else "Missing")
    )
    print(
        "- OpenWeather API key: "
        + ("Configured" if get_openweather_api_key() else "Missing")
    )
    print(
        "- Tavily API key: "
        + ("Configured" if get_tavily_api_key() else "Missing")
    )

    tool_data = await load_all_tools(force_reload=True)

    print_tool_group(
        "AviationStack flight tools",
        tool_data["flight_tools"],
    )
    print_tool_group(
        "Tavily flight fallback tools",
        tool_data["flight_fallback_tools"],
    )
    print_tool_group(
        "Weather tools",
        tool_data["weather_tools"],
    )
    print_tool_group(
        "Hotel tools",
        tool_data["hotel_tools"],
    )

    print("\nTool availability:")
    for name, available in tool_data["tool_status"].items():
        print(f"- {name}: {'Available' if available else 'Unavailable'}")

    print("\nTool errors:")
    errors = {
        provider: error
        for provider, error in tool_data["tool_errors"].items()
        if error
    }
    if not errors:
        print("- None")
    else:
        for provider, error in errors.items():
            print(f"- {provider}: {error}")

    print("\n" + "=" * 76)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    asyncio.run(diagnostic_test())