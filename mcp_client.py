import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch


# =========================================================
# Project configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
AVIATIONSTACK_SRC = BASE_DIR / "aviationstack-mcp" / "src"
WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"

load_dotenv(BASE_DIR / ".env")


AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

OPENWEATHER_API_KEY = (
    os.getenv("OPENWEATHER_API_KEY")
    or os.getenv("OPEN_WEATHER_API_KEY")
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =========================================================
# Validation
# =========================================================

def validate_configuration() -> None:
    errors: list[str] = []

    if not AVIATION_STACK_API_KEY:
        errors.append("AVIATION_STACK_API_KEY is missing")

    if not OPENWEATHER_API_KEY:
        errors.append("OPENWEATHER_API_KEY is missing")

    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is missing")

    if not AVIATIONSTACK_SRC.is_dir():
        errors.append(
            f"AviationStack source directory is missing: "
            f"{AVIATIONSTACK_SRC}"
        )

    if not WEATHER_SERVER_PATH.is_file():
        errors.append(
            f"Weather server file is missing: "
            f"{WEATHER_SERVER_PATH}"
        )

    if errors:
        raise RuntimeError(
            "Configuration problems:\n- " + "\n- ".join(errors)
        )


# =========================================================
# MCP client
# =========================================================

def create_mcp_client() -> MultiServerMCPClient:
    validate_configuration()

    return MultiServerMCPClient(
        {
            "aviationstack": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    "-m",
                    "aviationstack_mcp",
                    "mcp",
                    "run",
                ],
                "env": {
                    **os.environ,
                    "AVIATION_STACK_API_KEY": (
                        AVIATION_STACK_API_KEY
                    ),
                    "PYTHONPATH": str(AVIATIONSTACK_SRC),
                },
            },

            "weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    str(WEATHER_SERVER_PATH)
                ],
                "env": {
                    **os.environ,
                    "OPENWEATHER_API_KEY": (
                        OPENWEATHER_API_KEY
                    ),
                },
            },
        }
    )


# =========================================================
# Hotel tool
# =========================================================

def create_hotel_tool() -> TavilySearch:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is missing")

    return TavilySearch(
        max_results=8,
        topic="general",
        search_depth="advanced",
    )


# =========================================================
# Tool loading
# =========================================================

async def load_all_tools() -> dict[str, Any]:
    client = create_mcp_client()

    mcp_tools = await client.get_tools()
    hotel_tool = create_hotel_tool()

    flight_tools = []
    weather_tools = []

    for tool in mcp_tools:
        tool_name = tool.name.lower()

        if any(
            keyword in tool_name
            for keyword in [
                "flight",
                "aviation",
                "airport",
                "airline",
            ]
        ):
            flight_tools.append(tool)

        if any(
            keyword in tool_name
            for keyword in [
                "weather",
                "forecast",
                "temperature",
            ]
        ):
            weather_tools.append(tool)

    return {
        "client": client,
        "all_mcp_tools": mcp_tools,
        "flight_tools": flight_tools,
        "weather_tools": weather_tools,
        "hotel_tools": [hotel_tool],
    }