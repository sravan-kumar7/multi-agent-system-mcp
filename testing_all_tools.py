import asyncio

from mcp_client import load_all_tools


async def main() -> None:
    print("=" * 72)
    print("TESTING ALL AGENT TOOLS")
    print("=" * 72)

    result = await load_all_tools()

    print("\nFlight tools:")

    if result["flight_tools"]:
        for tool in result["flight_tools"]:
            print(f"- {tool.name}")
    else:
        print("- No flight tools detected")

    print("\nWeather tools:")

    if result["weather_tools"]:
        for tool in result["weather_tools"]:
            print(f"- {tool.name}")
    else:
        print("- No weather tools detected")

    print("\nHotel tools:")

    for tool in result["hotel_tools"]:
        print(f"- {tool.name}")

    print("\nAll agent tools loaded successfully.")


if __name__ == "__main__":
    asyncio.run(main())