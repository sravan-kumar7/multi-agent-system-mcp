import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# Project configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"

load_dotenv(BASE_DIR / ".env")

OPENWEATHER_API_KEY = (
    os.getenv("OPENWEATHER_API_KEY")
    or os.getenv("OPEN_WEATHER_API_KEY")
)


# =========================================================
# Configuration validation
# =========================================================

def validate_configuration() -> None:
    if not WEATHER_SERVER_PATH.is_file():
        raise FileNotFoundError(
            "Weather MCP server file was not found:\n"
            f"{WEATHER_SERVER_PATH}"
        )

    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OpenWeather API key is missing.\n\n"
            "Add this line to your .env file:\n"
            "OPENWEATHER_API_KEY=your_actual_api_key"
        )


# =========================================================
# Tool schema handling
# =========================================================

def get_tool_schema(tool: Any) -> dict[str, Any]:
    try:
        args_schema = getattr(tool, "args_schema", None)

        if args_schema:
            return args_schema.model_json_schema()

    except Exception:
        pass

    try:
        tool_args = getattr(tool, "args", None)

        if tool_args:
            return {
                "type": "object",
                "properties": tool_args,
            }

    except Exception:
        pass

    return {}


# =========================================================
# Select the most suitable weather tool
# =========================================================

def select_weather_tool(tools: list[Any]) -> Any:
    preferred_exact_names = [
        "get_current_weather",
        "get_weather",
        "current_weather",
        "weather",
        "get_forecast",
        "weather_forecast",
        "forecast",
    ]

    for preferred_name in preferred_exact_names:
        for tool in tools:
            if tool.name.lower() == preferred_name:
                return tool

    weather_keywords = [
        "weather",
        "forecast",
        "temperature",
        "climate",
    ]

    for tool in tools:
        tool_name = tool.name.lower()

        if any(keyword in tool_name for keyword in weather_keywords):
            return tool

    return tools[0]


# =========================================================
# Natural-language prompt handling
# =========================================================

def clean_location(location: str) -> str:
    """
    Remove trailing punctuation and weather-related words.
    """

    location = location.strip()

    location = re.sub(
        r"[?.!]+$",
        "",
        location,
    )

    location = re.sub(
        r"\b(today|now|currently|right now|please)\b",
        "",
        location,
        flags=re.IGNORECASE,
    )

    location = re.sub(r"\s+", " ", location)

    return location.strip(" ,.-")


def extract_location_from_prompt(prompt: str) -> str:
    """
    Extract a location from common weather prompts.

    Examples:
    - What is the weather in New York, USA?
    - Show weather for Tokyo, Japan
    - Temperature at Paris, France
    - London, United Kingdom
    """

    prompt = prompt.strip()

    if not prompt:
        raise ValueError("The weather prompt cannot be empty.")

    patterns = [
        r"(?:weather|temperature|forecast|climate)\s+(?:in|at|for)\s+(.+)",
        r"(?:show|give|tell|check|find|get)\s+"
        r"(?:me\s+)?(?:the\s+)?"
        r"(?:current\s+)?(?:weather|temperature|forecast)\s+"
        r"(?:in|at|for)\s+(.+)",
        r"(?:what(?:'s| is)\s+the\s+)"
        r"(?:current\s+)?(?:weather|temperature|forecast)\s+"
        r"(?:in|at|for)\s+(.+)",
        r"(?:how(?:'s| is)\s+the\s+weather\s+)"
        r"(?:in|at|for)\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            prompt,
            flags=re.IGNORECASE,
        )

        if match:
            location = clean_location(match.group(1))

            if location:
                return location

    # When the user enters only a place, use it directly.
    return clean_location(prompt)


def get_weather_prompt() -> tuple[str, str]:
    print("\nEnter a weather question for any location worldwide.")
    print("\nExamples:")
    print("  What is the weather in New York, USA?")
    print("  Show current weather for Tokyo, Japan")
    print("  Check weather in Cape Town, South Africa")
    print("  What is the temperature in Paris, France?")
    print("  Weather in Visakhapatnam, India")
    print("  London, United Kingdom")
    print("  51.5074,-0.1278")

    prompt = input("\nWeather prompt: ").strip()

    if not prompt:
        raise ValueError("Weather prompt cannot be empty.")

    location = extract_location_from_prompt(prompt)

    if not location:
        raise ValueError(
            "A location could not be extracted from the prompt."
        )

    return prompt, location


# =========================================================
# Coordinates support
# =========================================================

def parse_coordinates(
    location: str,
) -> tuple[float, float] | None:

    try:
        parts = [
            part.strip()
            for part in location.split(",")
        ]

        if len(parts) != 2:
            return None

        latitude = float(parts[0])
        longitude = float(parts[1])

        if not -90 <= latitude <= 90:
            return None

        if not -180 <= longitude <= 180:
            return None

        return latitude, longitude

    except ValueError:
        return None


def build_coordinate_input(
    schema: dict[str, Any],
    location: str,
) -> dict[str, Any]:

    coordinates = parse_coordinates(location)

    if coordinates is None:
        return {}

    latitude, longitude = coordinates

    properties = schema.get("properties", {})

    property_lookup = {
        str(name).lower(): name
        for name in properties
    }

    latitude_field = (
        property_lookup.get("latitude")
        or property_lookup.get("lat")
    )

    longitude_field = (
        property_lookup.get("longitude")
        or property_lookup.get("lon")
        or property_lookup.get("lng")
    )

    if latitude_field and longitude_field:
        return {
            latitude_field: latitude,
            longitude_field: longitude,
        }

    return {}


# =========================================================
# Dynamic tool-input creation
# =========================================================

def build_weather_input(
    schema: dict[str, Any],
    prompt: str,
    location: str,
) -> dict[str, Any]:

    properties = schema.get("properties", {})

    property_lookup = {
        str(name).lower(): name
        for name in properties
    }

    # Prefer passing the complete natural-language prompt
    # when the MCP tool has a query or prompt field.
    prompt_fields = [
        "query",
        "prompt",
        "question",
        "text",
        "request",
    ]

    for field_name in prompt_fields:
        if field_name in property_lookup:
            actual_name = property_lookup[field_name]

            return {
                actual_name: prompt
            }

    # Otherwise pass the extracted location.
    location_fields = [
        "location",
        "city",
        "place",
        "city_name",
        "location_name",
        "destination",
        "address",
    ]

    for field_name in location_fields:
        if field_name in property_lookup:
            actual_name = property_lookup[field_name]

            return {
                actual_name: location
            }

    # Some tools have city and country as separate parameters.
    city_field = property_lookup.get("city")
    country_field = property_lookup.get("country")

    if city_field:
        parts = [
            item.strip()
            for item in location.split(",", maxsplit=1)
        ]

        weather_input: dict[str, Any] = {
            city_field: parts[0]
        }

        if country_field and len(parts) > 1:
            weather_input[country_field] = parts[1]

        return weather_input

    return {}


# =========================================================
# Main program
# =========================================================

async def main() -> None:
    validate_configuration()

    print("=" * 72)
    print("GLOBAL WEATHER MCP PROMPT TEST")
    print("=" * 72)

    print(f"\nUsing Python:\n{sys.executable}")
    print(f"\nWeather MCP server:\n{WEATHER_SERVER_PATH}")

    prompt, location = get_weather_prompt()

    print(f"\nDetected location: {location}")

    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    str(WEATHER_SERVER_PATH)
                ],
                "env": {
                    **os.environ,
                    "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
                },
            }
        }
    )

    try:
        tools = await client.get_tools()

        if not tools:
            raise RuntimeError(
                "No weather tools were returned by the MCP server."
            )

        print("\nAvailable Weather Tools:\n")

        for index, tool in enumerate(tools, start=1):
            print(f"{index}. {tool.name}")

        weather_tool = select_weather_tool(tools)

        print("\nSelected Weather Tool:")
        print(weather_tool.name)

        schema = get_tool_schema(weather_tool)

        print("\nTool Input Schema:\n")
        print(
            json.dumps(
                schema,
                indent=2,
                default=str,
            )
        )

        # First check whether coordinates were entered.
        weather_input = build_coordinate_input(
            schema,
            location,
        )

        # Otherwise create input from the prompt/location.
        if not weather_input:
            weather_input = build_weather_input(
                schema,
                prompt,
                location,
            )

        if not weather_input:
            raise ValueError(
                "The program could not construct the weather tool "
                "input from the schema printed above."
            )

        print("\nWeather Tool Input:\n")
        print(
            json.dumps(
                weather_input,
                indent=2,
                default=str,
            )
        )

        print(f"\nFetching weather for: {location}\n")

        result = await weather_tool.ainvoke(
            weather_input
        )

        print("=" * 72)
        print("WEATHER RESULT")
        print("=" * 72)
        print(result)

        print(
            "\nThe Weather MCP server and dynamic global "
            "weather prompt are working successfully."
        )

    except Exception as error:
        print("\nWeather MCP test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Reason: {error}")

        print(
            "\nCheck the following:\n"
            "1. OPENWEATHER_API_KEY is valid.\n"
            "2. custom_weather_mcp_server.py exists.\n"
            "3. The server does not print normal text to stdout.\n"
            "4. Enter a city or place, not only a large country.\n"
            "5. Include the country for cities with duplicate names.\n"
            "6. Check the tool schema printed above."
        )

        raise


if __name__ == "__main__":
    asyncio.run(main())