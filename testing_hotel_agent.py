import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_tavily import TavilySearch


# =========================================================
# Project configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =========================================================
# Validation
# =========================================================

def validate_configuration() -> None:
    if not TAVILY_API_KEY:
        raise ValueError(
            "Tavily API key is missing.\n\n"
            "Add this line to your .env file:\n"
            "TAVILY_API_KEY=your_actual_tavily_api_key"
        )


# =========================================================
# Read destination and travel preferences
# =========================================================

def get_hotel_request() -> tuple[str, str]:
    print("=" * 72)
    print("GLOBAL HOTEL AGENT TEST")
    print("=" * 72)

    print("\nEnter a hotel request for any destination worldwide.")
    print("\nExamples:")
    print("  Find budget hotels in London, United Kingdom")
    print("  Show luxury hotels in Dubai near Burj Khalifa")
    print("  Find hotels in Tokyo under $150 per night")
    print("  Best family hotels in Paris, France")
    print("  Hotels near Times Square, New York")
    print("  Beach resorts in Bali, Indonesia")

    prompt = input("\nHotel prompt: ").strip()

    if not prompt:
        raise ValueError("Hotel prompt cannot be empty.")

    return prompt, prompt


# =========================================================
# Main test
# =========================================================

async def main() -> None:
    validate_configuration()

    prompt, query = get_hotel_request()

    hotel_search_tool = TavilySearch(
        max_results=8,
        topic="general",
        search_depth="advanced",
    )

    print("\nSearching for hotel information...\n")

    try:
        result = await hotel_search_tool.ainvoke(
            {
                "query": (
                    f"{query}. Provide useful hotel options with hotel name, "
                    "exact area or neighborhood, approximate nightly price, "
                    "rating, important amenities, nearby landmarks, "
                    "and booking or official source."
                )
            }
        )

        print("=" * 72)
        print("HOTEL SEARCH RESULT")
        print("=" * 72)
        print(result)

        print(
            "\nThe global Hotel Agent test completed successfully."
        )

    except Exception as error:
        print("\nHotel Agent test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Reason: {error}")

        print(
            "\nCheck the following:\n"
            "1. TAVILY_API_KEY is present and valid.\n"
            "2. Internet access is available.\n"
            "3. langchain-tavily is installed.\n"
            "4. The destination or hotel request is clear.\n"
            "5. Try including city and country for better accuracy."
        )

        raise


if __name__ == "__main__":
    asyncio.run(main())