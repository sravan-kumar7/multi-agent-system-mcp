# LangGraph Multi-Agent Travel Booking System
# Streamlit-compatible version using in-memory checkpointing

import asyncio
import operator
import os
import uuid
from typing import Annotated, TypedDict

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

from mcp_client import (
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    tavily_mcp_search,
    weather_mcp_search,
)


# -------------------------------------------------------------------
# Environment variables
# -------------------------------------------------------------------

# Loads .env locally.
# On Streamlit Cloud, secrets are supplied as environment variables.
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it in Streamlit Cloud "
        "under Manage app → Settings → Secrets."
    )


# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)


# -------------------------------------------------------------------
# LangGraph state
# -------------------------------------------------------------------

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


# -------------------------------------------------------------------
# Flight agent
# -------------------------------------------------------------------

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState) -> dict:
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel flight planner."
                ),
                HumanMessage(content=prompt),
            ]
        )

        flight_data = response.content

    except Exception as error:
        flight_data = (
            "Flight information is currently unavailable. "
            f"Reason: {error}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight recommendations generated")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------------------
# Hotel agent
# -------------------------------------------------------------------

def hotel_agent(state: TravelState) -> dict:
    query = f"Best hotels for {state['user_query']}"

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as error:
        hotel_results = (
            "Hotel information is currently unavailable. "
            f"Reason: {error}"
        )

    return {
        "hotel_results": str(hotel_results),
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------------------
# Weather agent
# -------------------------------------------------------------------

def weather_agent(state: TravelState) -> dict:
    try:
        city = extract_destination(state["user_query"])

        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as error:
        weather_results = (
            "Weather information is currently unavailable. "
            f"Reason: {error}"
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(content="Weather information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------------------
# Itinerary agent
# -------------------------------------------------------------------

def itinerary_agent(state: TravelState) -> dict:
    prompt = f"""
Create a detailed and practical travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Information:
{state['weather_results']}

Provide:

1. Travel summary
2. Recommended flight guidance
3. Recommended hotels
4. Day-wise itinerary
5. Weather advice
6. Estimated budget guidance
7. Important travel tips
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel planner."
                ),
                HumanMessage(content=prompt),
            ]
        )

        itinerary = response.content
        response_message = response

    except Exception as error:
        itinerary = (
            "The itinerary could not be generated. "
            f"Reason: {error}"
        )
        response_message = AIMessage(content=itinerary)

    return {
        "itinerary": itinerary,
        "messages": [response_message],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------------------
# Build LangGraph
# -------------------------------------------------------------------

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)


# -------------------------------------------------------------------
# In-memory checkpointing
# -------------------------------------------------------------------

# This replaces:
#
# _conn = psycopg.connect(DATABASE_URL)
# checkpointer = PostgresSaver(_conn)
# checkpointer.setup()
#
# MemorySaver works without PostgreSQL.
# Data may be lost when Streamlit restarts.

checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)


# -------------------------------------------------------------------
# Command-line testing
# -------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    user_input = input("Enter travel request: ")

    initial_state: TravelState = {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "user_query": user_input,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "llm_calls": 0,
    }

    result = app.invoke(
        initial_state,
        config=config,
    )

    print("\nFINAL RESPONSE:\n")
    print(result["itinerary"])