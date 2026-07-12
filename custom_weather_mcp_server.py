from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT_SECONDS = 20.0

load_dotenv(ENV_FILE, override=False)

mcp = FastMCP("Worldwide Weather Server")


def get_environment_value(*names: str) -> str | None:
    """Return the first non-empty environment variable."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def get_openweather_api_key() -> str | None:
    """Return the OpenWeather API key using supported aliases."""
    return get_environment_value(
        "OPENWEATHER_API_KEY",
        "OPEN_WEATHER_API_KEY",
    )


def validate_api_key() -> str | None:
    """Return safe Markdown when the OpenWeather key is unavailable."""
    if get_openweather_api_key():
        return None

    return (
        "## ⚠️ Weather service is not configured\n\n"
        "The OpenWeather API key is missing.\n\n"
        "Add `OPENWEATHER_API_KEY` to the local `.env` file or "
        "Streamlit Community Cloud secrets."
    )


def first_non_empty(*values: str | None) -> str:
    """Return the first non-empty location candidate."""
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def normalize_location(
    *,
    city: str = "",
    location: str = "",
    place: str = "",
    destination: str = "",
    city_name: str = "",
    location_name: str = "",
    query: str = "",
) -> str:
    """
    Select and normalize a location.

    Specific fields are preferred before the generic query field.
    The caller must pass only a normalized destination in query.
    """
    selected = first_non_empty(
        location,
        city,
        place,
        destination,
        city_name,
        location_name,
        query,
    )

    cleaned = " ".join(selected.split())

    if not cleaned:
        raise ValueError(
            "Provide a location such as `Tokyo, Japan` or "
            "`London, United Kingdom`."
        )

    if len(cleaned) > 120:
        raise ValueError(
            "The supplied location is too long. Send only the city or "
            "city-and-country name."
        )

    return cleaned


def format_number(value: Any, decimal_places: int = 1) -> str | None:
    """Safely format numeric values, returning None when invalid."""
    try:
        number = round(float(value), decimal_places)
    except (TypeError, ValueError):
        return None

    if number.is_integer():
        return str(int(number))

    return str(number)


def readable_condition(value: Any) -> str | None:
    """Return a clean human-readable weather condition."""
    text = str(value or "").strip()
    if not text:
        return None
    return text[:1].upper() + text[1:]


def weather_icon(condition: str) -> str:
    """Return an emoji suitable for a weather condition."""
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
    if any(
        item in normalized
        for item in ("mist", "fog", "haze", "smoke")
    ):
        return "🌫️"

    return "🌤️"


def format_forecast_datetime(value: Any) -> str:
    """Convert an OpenWeather forecast timestamp to a readable label."""
    text = str(value or "").strip()

    try:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return parsed.strftime("%d %b, %I:%M %p")
    except ValueError:
        return text or "Upcoming period"


def create_weather_advice(
    *,
    temperature: Any,
    humidity: Any,
    condition: str,
    wind_speed: Any,
) -> list[str]:
    """Generate practical traveller advice from live measurements."""
    advice: list[str] = []
    normalized = condition.lower()

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

    if "rain" in normalized or "drizzle" in normalized:
        advice.append(
            "Carry a compact umbrella or lightweight waterproof jacket."
        )

    if "thunder" in normalized:
        advice.append(
            "Keep an indoor alternative and avoid exposed outdoor areas "
            "during thunderstorms."
        )

    if "snow" in normalized:
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
            advice.append("Carry a light jacket or sweater.")

    if humidity_value is not None and humidity_value >= 70:
        advice.append(
            "High humidity is expected, so carry water and plan short breaks."
        )

    if wind_value is not None and wind_value >= 10:
        advice.append(
            "Strong winds may affect outdoor plans and local transport."
        )

    if not advice:
        advice.append(
            "Conditions appear manageable, but recheck before outdoor plans."
        )

    return advice


def openweather_error_message(
    *,
    status_code: int,
    payload: Any,
    location: str,
) -> str:
    """Convert OpenWeather errors into safe traveller-facing Markdown."""
    api_message = ""
    if isinstance(payload, dict):
        api_message = str(payload.get("message", "") or "").strip()

    if status_code == 401:
        explanation = (
            "The OpenWeather API key is invalid, inactive, or still awaiting "
            "activation."
        )
    elif status_code == 404:
        explanation = (
            f"The location **{location}** could not be found. Try a more "
            "specific destination such as `Tokyo, Japan`."
        )
    elif status_code == 429:
        explanation = (
            "The OpenWeather request limit has been reached. Try again later."
        )
    elif status_code >= 500:
        explanation = (
            "OpenWeather is temporarily unavailable. Try again shortly."
        )
    else:
        explanation = api_message or (
            "The weather request could not be completed."
        )

    return (
        "## ⚠️ Weather information unavailable\n\n"
        f"{explanation}\n\n"
        f"**Requested location:** {location}"
    )


async def request_openweather(
    *,
    endpoint: str,
    location: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Request OpenWeather data asynchronously."""
    api_key_error = validate_api_key()
    if api_key_error:
        return None, api_key_error

    api_key = get_openweather_api_key()
    if not api_key:
        return None, api_key_error

    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",
    }

    try:
        timeout = httpx.Timeout(
            REQUEST_TIMEOUT_SECONDS,
            connect=10.0,
        )

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                f"{OPENWEATHER_BASE_URL}/{endpoint}",
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
                    status_code=response.status_code,
                    payload=payload,
                    location=location,
                ),
            )

        if not isinstance(payload, dict):
            return (
                None,
                (
                    "## ⚠️ Weather information unavailable\n\n"
                    "The weather provider returned an unexpected response."
                ),
            )

        return payload, None

    except httpx.TimeoutException:
        return (
            None,
            (
                "## ⏳ Weather request timed out\n\n"
                "The weather provider did not respond in time. Try again."
            ),
        )
    except httpx.RequestError:
        return (
            None,
            (
                "## 🌐 Weather connection unavailable\n\n"
                "The application could not connect to the weather provider. "
                "Check the internet connection and try again."
            ),
        )
    except Exception:
        return (
            None,
            (
                "## ⚠️ Weather information unavailable\n\n"
                "An unexpected weather-service error occurred. Try again."
            ),
        )


def build_current_weather_markdown(
    *,
    requested_location: str,
    data: dict[str, Any],
) -> str:
    """Build clean current-weather Markdown from OpenWeather data."""
    main_data = data.get("main")
    weather_data = data.get("weather")
    wind_data = data.get("wind")
    system_data = data.get("sys")

    main_data = main_data if isinstance(main_data, dict) else {}
    wind_data = wind_data if isinstance(wind_data, dict) else {}
    system_data = system_data if isinstance(system_data, dict) else {}

    weather_entry: dict[str, Any] = {}
    if isinstance(weather_data, list) and weather_data:
        first_entry = weather_data[0]
        if isinstance(first_entry, dict):
            weather_entry = first_entry

    resolved_city = str(data.get("name") or requested_location).strip()
    country_code = str(system_data.get("country") or "").strip()
    display_location = (
        f"{resolved_city}, {country_code}"
        if country_code
        else resolved_city
    )

    temperature = format_number(main_data.get("temp"))
    feels_like = format_number(main_data.get("feels_like"))
    humidity = format_number(main_data.get("humidity"), 0)
    pressure = format_number(main_data.get("pressure"), 0)
    wind_speed = format_number(wind_data.get("speed"))
    condition = readable_condition(weather_entry.get("description"))

    required_values = (
        temperature,
        feels_like,
        humidity,
        wind_speed,
        condition,
    )
    if any(value is None for value in required_values):
        return (
            f"## 🌦️ Weather in {display_location}\n\n"
            "Live weather was received, but the provider response did not "
            "contain all required measurements.\n\n"
            "Check an official weather source before departure."
        )

    icon = weather_icon(condition)
    advice = create_weather_advice(
        temperature=main_data.get("temp"),
        humidity=main_data.get("humidity"),
        condition=condition,
        wind_speed=wind_data.get("speed"),
    )
    advice_markdown = "\n".join(f"- {item}" for item in advice)

    pressure_row = ""
    if pressure is not None:
        pressure_row = f"| 🧭 Air pressure | **{pressure} hPa** |\n"

    return f"""## {icon} Weather in {display_location}

### Current conditions

| Metric | Reading |
|---|---|
| 🌡️ Temperature | **{temperature}°C** |
| 🤗 Feels like | **{feels_like}°C** |
| {icon} Condition | **{condition}** |
| 💧 Humidity | **{humidity}%** |
| 💨 Wind speed | **{wind_speed} m/s** |
{pressure_row}
### 🎒 Traveller advice

{advice_markdown}

> Live weather can change. Recheck shortly before outdoor activities.
"""


@mcp.tool()
async def get_current_weather(
    city: str = "",
    location: str = "",
    place: str = "",
    destination: str = "",
    city_name: str = "",
    location_name: str = "",
    query: str = "",
) -> str:
    """Get current weather for a normalized worldwide destination."""
    try:
        normalized_location = normalize_location(
            city=city,
            location=location,
            place=place,
            destination=destination,
            city_name=city_name,
            location_name=location_name,
            query=query,
        )
    except ValueError as error:
        return f"## ⚠️ Invalid weather location\n\n{error}"

    data, error = await request_openweather(
        endpoint="weather",
        location=normalized_location,
    )

    if error:
        return error

    if not data:
        return (
            f"## 🌦️ Weather in {normalized_location}\n\n"
            "No live weather information was returned."
        )

    return build_current_weather_markdown(
        requested_location=normalized_location,
        data=data,
    )


@mcp.tool()
async def get_forecast(
    city: str = "",
    location: str = "",
    place: str = "",
    destination: str = "",
    city_name: str = "",
    location_name: str = "",
    query: str = "",
) -> str:
    """Get the next five OpenWeather forecast periods."""
    try:
        normalized_location = normalize_location(
            city=city,
            location=location,
            place=place,
            destination=destination,
            city_name=city_name,
            location_name=location_name,
            query=query,
        )
    except ValueError as error:
        return f"## ⚠️ Invalid forecast location\n\n{error}"

    data, error = await request_openweather(
        endpoint="forecast",
        location=normalized_location,
    )

    if error:
        return error

    if not data:
        return (
            f"## 📅 Forecast for {normalized_location}\n\n"
            "No forecast information was returned."
        )

    city_data = data.get("city")
    city_data = city_data if isinstance(city_data, dict) else {}

    resolved_city = str(
        city_data.get("name") or normalized_location
    ).strip()
    country_code = str(city_data.get("country") or "").strip()
    display_location = (
        f"{resolved_city}, {country_code}"
        if country_code
        else resolved_city
    )

    forecast_items = data.get("list")
    if not isinstance(forecast_items, list) or not forecast_items:
        return (
            f"## 📅 Forecast for {display_location}\n\n"
            "The provider returned no usable forecast periods."
        )

    rows: list[str] = []
    conditions_seen: list[str] = []

    for item in forecast_items[:5]:
        if not isinstance(item, dict):
            continue

        main_data = item.get("main")
        weather_data = item.get("weather")
        main_data = main_data if isinstance(main_data, dict) else {}

        weather_entry: dict[str, Any] = {}
        if isinstance(weather_data, list) and weather_data:
            first_entry = weather_data[0]
            if isinstance(first_entry, dict):
                weather_entry = first_entry

        temperature = format_number(main_data.get("temp"))
        feels_like = format_number(main_data.get("feels_like"))
        condition = readable_condition(
            weather_entry.get("description")
        )

        if (
            temperature is None
            or feels_like is None
            or condition is None
        ):
            continue

        conditions_seen.append(condition)
        rows.append(
            f"| {format_forecast_datetime(item.get('dt_txt'))} | "
            f"**{temperature}°C** | {feels_like}°C | "
            f"{weather_icon(condition)} {condition} |"
        )

    if not rows:
        return (
            f"## 📅 Forecast for {display_location}\n\n"
            "The provider returned forecast data without usable measurements."
        )

    combined_conditions = " ".join(conditions_seen).lower()
    forecast_advice: list[str] = []

    if "rain" in combined_conditions or "drizzle" in combined_conditions:
        forecast_advice.append(
            "Rain appears in the forecast. Carry a compact umbrella."
        )
    if "thunder" in combined_conditions:
        forecast_advice.append(
            "Keep an indoor backup plan for possible thunderstorms."
        )
    if "snow" in combined_conditions:
        forecast_advice.append(
            "Allow extra travel time because snow may affect transport."
        )
    if not forecast_advice:
        forecast_advice.append(
            "Conditions appear relatively stable, but recheck before travel."
        )

    rows_markdown = "\n".join(rows)
    advice_markdown = "\n".join(
        f"- {item}" for item in forecast_advice
    )

    return f"""## 📅 Upcoming forecast for {display_location}

| Date and time | Temperature | Feels like | Conditions |
|---|---:|---:|---|
{rows_markdown}

### 🧳 Forecast-based advice

{advice_markdown}

> Forecasts can change. Recheck shortly before departure.
"""


if __name__ == "__main__":
    mcp.run()