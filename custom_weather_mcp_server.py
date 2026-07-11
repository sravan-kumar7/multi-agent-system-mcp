"""
Custom OpenWeather MCP Server

Features:
- Worldwide city support
- Current weather
- Five-entry forecast
- Clean Markdown output
- Async HTTP requests
- API-key validation
- Timeout and network error handling
- Friendly error messages
- Streamlit Cloud and local deployment compatibility
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


# =============================================================================
# Environment configuration
# =============================================================================

load_dotenv(override=False)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT_SECONDS = 15.0


# =============================================================================
# MCP server
# =============================================================================

mcp = FastMCP("Worldwide Weather Server")


# =============================================================================
# Helper functions
# =============================================================================

def validate_api_key() -> str | None:
    """
    Return a readable error when the OpenWeather API key is missing.
    """

    if OPENWEATHER_API_KEY:
        return None

    return (
        "## ⚠️ Weather service is not configured\n\n"
        "The `OPENWEATHER_API_KEY` environment variable is missing.\n\n"
        "Add the key to your local `.env` file or Streamlit Cloud secrets."
    )


def normalize_city(city: str) -> str:
    """
    Clean and validate a city or worldwide location supplied by the agent.
    """

    cleaned_city = " ".join(city.strip().split())

    if not cleaned_city:
        raise ValueError(
            "Please provide a city, such as `Tokyo, Japan` or "
            "`London, United Kingdom`."
        )

    if len(cleaned_city) > 150:
        raise ValueError("The supplied location is too long.")

    return cleaned_city


def format_number(value: Any, decimal_places: int = 1) -> str:
    """
    Safely format API numeric values.
    """

    try:
        number = round(float(value), decimal_places)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except (TypeError, ValueError):
        return "Not available"


def weather_icon(condition: str) -> str:
    """
    Return an emoji suitable for an OpenWeather condition.
    """

    normalized = condition.lower()

    if "thunder" in normalized:
        return "⛈️"

    if "rain" in normalized or "drizzle" in normalized:
        return "🌧️"

    if "snow" in normalized:
        return "❄️"

    if "clear" in normalized:
        return "☀️"

    if "cloud" in normalized or "overcast" in normalized:
        return "☁️"

    if (
        "mist" in normalized
        or "fog" in normalized
        or "haze" in normalized
        or "smoke" in normalized
    ):
        return "🌫️"

    return "🌤️"


def readable_condition(condition: Any) -> str:
    """
    Convert conditions such as 'overcast clouds' into 'Overcast clouds'.
    """

    text = str(condition or "Not available").strip()

    if not text:
        return "Not available"

    return text[0].upper() + text[1:]


def format_forecast_datetime(value: str) -> str:
    """
    Convert OpenWeather's date format into a human-readable label.

    Example:
    2026-07-11 15:00:00 -> 11 Jul, 03:00 PM
    """

    try:
        parsed = datetime.strptime(
            value,
            "%Y-%m-%d %H:%M:%S",
        )

        return parsed.strftime("%d %b, %I:%M %p")

    except (TypeError, ValueError):
        return str(value)


def create_weather_advice(
    temperature: Any,
    humidity: Any,
    condition: str,
    wind_speed: Any,
) -> list[str]:
    """
    Generate practical advice using only returned weather values.
    """

    advice: list[str] = []
    normalized_condition = condition.lower()

    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError):
        temperature_value = None

    try:
        humidity_value = float(humidity)
    except (TypeError, ValueError):
        humidity_value = None

    try:
        wind_value = float(wind_speed)
    except (TypeError, ValueError):
        wind_value = None

    if "rain" in normalized_condition or "drizzle" in normalized_condition:
        advice.append(
            "Carry a compact umbrella or lightweight waterproof jacket."
        )

    if "thunder" in normalized_condition:
        advice.append(
            "Avoid exposed outdoor areas during thunderstorms and "
            "keep an indoor backup plan."
        )

    if "snow" in normalized_condition:
        advice.append(
            "Wear insulated footwear and allow extra travel time."
        )

    if temperature_value is not None:
        if temperature_value >= 30:
            advice.append(
                "Wear breathable clothing, use sunscreen and stay hydrated."
            )

        elif temperature_value >= 24:
            advice.append(
                "Light, breathable clothing should be comfortable."
            )

        elif temperature_value <= 10:
            advice.append(
                "Pack warm layers, especially for mornings and evenings."
            )

        elif temperature_value <= 18:
            advice.append(
                "Carry a light jacket or sweater."
            )

    if humidity_value is not None and humidity_value >= 70:
        advice.append(
            "Humidity is high, so plan short breaks and carry drinking water."
        )

    if wind_value is not None and wind_value >= 10:
        advice.append(
            "Strong winds may affect outdoor plans and local transport."
        )

    if not advice:
        advice.append(
            "Conditions appear manageable, but check the forecast again "
            "before outdoor activities."
        )

    return advice


def openweather_error_message(
    status_code: int,
    payload: Any,
    city: str,
) -> str:
    """
    Convert OpenWeather errors into readable Markdown.
    """

    message = ""

    if isinstance(payload, dict):
        message = str(payload.get("message", "")).strip()

    if status_code == 401:
        explanation = (
            "The OpenWeather API key is invalid, inactive or not yet activated."
        )

    elif status_code == 404:
        explanation = (
            f"The location **{city}** could not be found. "
            "Try a more specific value such as `Tokyo, Japan`."
        )

    elif status_code == 429:
        explanation = (
            "The OpenWeather request limit has been reached. "
            "Please wait and try again."
        )

    elif status_code >= 500:
        explanation = (
            "OpenWeather is temporarily unavailable. Please try again shortly."
        )

    else:
        explanation = message or "The weather request could not be completed."

    return (
        "## ⚠️ Weather information unavailable\n\n"
        f"{explanation}\n\n"
        f"**Requested location:** {city}"
    )


async def request_openweather(
    endpoint: str,
    city: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Make an asynchronous request to OpenWeather.

    Returns:
        (response_data, error_markdown)
    """

    api_key_error = validate_api_key()

    if api_key_error:
        return None, api_key_error

    url = f"{OPENWEATHER_BASE_URL}/{endpoint}"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        timeout = httpx.Timeout(
            REQUEST_TIMEOUT_SECONDS,
            connect=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                params=params,
            )

        try:
            payload = response.json()

        except ValueError:
            payload = {}

        if response.status_code != 200:
            return (
                None,
                openweather_error_message(
                    response.status_code,
                    payload,
                    city,
                ),
            )

        if not isinstance(payload, dict):
            return (
                None,
                (
                    "## ⚠️ Weather information unavailable\n\n"
                    "OpenWeather returned an unexpected response format."
                ),
            )

        return payload, None

    except httpx.TimeoutException:
        return (
            None,
            (
                "## ⏳ Weather request timed out\n\n"
                "OpenWeather did not respond within the expected time. "
                "Please try again."
            ),
        )

    except httpx.RequestError as error:
        return (
            None,
            (
                "## 🌐 Weather service connection failed\n\n"
                "The application could not connect to OpenWeather.\n\n"
                f"**Technical reason:** {error}"
            ),
        )

    except Exception as error:
        return (
            None,
            (
                "## ⚠️ Unexpected weather error\n\n"
                "The weather request could not be completed.\n\n"
                f"**Technical reason:** {error}"
            ),
        )


# =============================================================================
# MCP tools
# =============================================================================

@mcp.tool()
async def get_current_weather(city: str) -> str:
    """
    Get current weather for any worldwide city or location.

    Examples:
    - Tokyo, Japan
    - London, United Kingdom
    - New York, United States
    - Dubai, United Arab Emirates
    """

    try:
        city = normalize_city(city)

    except ValueError as error:
        return (
            "## ⚠️ Invalid weather location\n\n"
            f"{error}"
        )

    data, error = await request_openweather(
        endpoint="weather",
        city=city,
    )

    if error:
        return error

    if not data:
        return (
            "## ⚠️ Weather information unavailable\n\n"
            "No weather information was returned."
        )

    main_data = data.get("main", {})
    weather_data = data.get("weather", [])
    wind_data = data.get("wind", {})
    system_data = data.get("sys", {})

    weather_entry = (
        weather_data[0]
        if isinstance(weather_data, list) and weather_data
        else {}
    )

    resolved_city = str(data.get("name") or city)

    country_code = str(system_data.get("country") or "").strip()

    display_location = (
        f"{resolved_city}, {country_code}"
        if country_code
        else resolved_city
    )

    temperature = main_data.get("temp")
    feels_like = main_data.get("feels_like")
    humidity = main_data.get("humidity")
    pressure = main_data.get("pressure")
    condition = readable_condition(
        weather_entry.get("description")
    )
    wind_speed = wind_data.get("speed")

    icon = weather_icon(condition)

    advice = create_weather_advice(
        temperature=temperature,
        humidity=humidity,
        condition=condition,
        wind_speed=wind_speed,
    )

    advice_markdown = "\n".join(
        f"- {item}"
        for item in advice
    )

    return f"""## {icon} Weather in {display_location}

### Current conditions

| Metric | Reading |
|---|---|
| 🌡️ Temperature | **{format_number(temperature)}°C** |
| 🤗 Feels like | **{format_number(feels_like)}°C** |
| {icon} Condition | **{condition}** |
| 💧 Humidity | **{format_number(humidity, 0)}%** |
| 💨 Wind speed | **{format_number(wind_speed)} m/s** |
| 🧭 Air pressure | **{format_number(pressure, 0)} hPa** |

### 🎒 Traveller advice

{advice_markdown}

> Weather conditions may change. Check again shortly before outdoor activities.
"""


@mcp.tool()
async def get_forecast(city: str) -> str:
    """
    Get the upcoming five forecast periods for any worldwide city.
    """

    try:
        city = normalize_city(city)

    except ValueError as error:
        return (
            "## ⚠️ Invalid forecast location\n\n"
            f"{error}"
        )

    data, error = await request_openweather(
        endpoint="forecast",
        city=city,
    )

    if error:
        return error

    if not data:
        return (
            "## ⚠️ Forecast unavailable\n\n"
            "No forecast information was returned."
        )

    city_data = data.get("city", {})

    resolved_city = str(
        city_data.get("name")
        or city
    )

    country_code = str(
        city_data.get("country")
        or ""
    ).strip()

    display_location = (
        f"{resolved_city}, {country_code}"
        if country_code
        else resolved_city
    )

    forecast_items = data.get("list", [])

    if not isinstance(forecast_items, list) or not forecast_items:
        return (
            f"## ⚠️ Forecast unavailable for {display_location}\n\n"
            "OpenWeather returned no forecast periods."
        )

    table_rows: list[str] = []
    conditions_seen: list[str] = []

    for item in forecast_items[:5]:
        if not isinstance(item, dict):
            continue

        main_data = item.get("main", {})
        weather_data = item.get("weather", [])

        weather_entry = (
            weather_data[0]
            if isinstance(weather_data, list) and weather_data
            else {}
        )

        timestamp = format_forecast_datetime(
            str(item.get("dt_txt") or "")
        )

        temperature = main_data.get("temp")
        feels_like = main_data.get("feels_like")

        condition = readable_condition(
            weather_entry.get("description")
        )

        conditions_seen.append(condition)

        icon = weather_icon(condition)

        table_rows.append(
            f"| {timestamp} | "
            f"**{format_number(temperature)}°C** | "
            f"{format_number(feels_like)}°C | "
            f"{icon} {condition} |"
        )

    if not table_rows:
        return (
            f"## ⚠️ Forecast unavailable for {display_location}\n\n"
            "No usable forecast entries were returned."
        )

    combined_conditions = " ".join(conditions_seen).lower()

    forecast_advice: list[str] = []

    if "rain" in combined_conditions or "drizzle" in combined_conditions:
        forecast_advice.append(
            "Rain is expected during at least one forecast period. "
            "Carry an umbrella."
        )

    if "thunder" in combined_conditions:
        forecast_advice.append(
            "Thunderstorms may affect outdoor activities. "
            "Keep an indoor backup plan."
        )

    if "snow" in combined_conditions:
        forecast_advice.append(
            "Snow may slow transport. Allow additional travel time."
        )

    if not forecast_advice:
        forecast_advice.append(
            "The upcoming conditions appear relatively stable, "
            "but recheck before departure."
        )

    forecast_advice_markdown = "\n".join(
        f"- {item}"
        for item in forecast_advice
    )

    rows_markdown = "\n".join(table_rows)

    return f"""## 📅 Upcoming forecast for {display_location}

| Date and time | Temperature | Feels like | Conditions |
|---|---:|---:|---|
{rows_markdown}

### 🧳 Forecast-based advice

{forecast_advice_markdown}

> Forecast data is shown in the location's reported time sequence and may change.
"""


# =============================================================================
# Server entry point
# =============================================================================

if __name__ == "__main__":
    mcp.run()