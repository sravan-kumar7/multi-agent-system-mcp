
import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

BASE_DIR = r"C:\multi-agent-system-with-mcp"
PYTHON_EXE = rf"{BASE_DIR}\langgraph_env2\Scripts\python.exe"

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },

        "aviationstack": {
            "transport": "stdio",
            "command": PYTHON_EXE,
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "env": {
                "AVIATIONSTACK_API_KEY": AVIATIONSTACK_API_KEY,
                "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY,
            },
        },

        "weather": {
            "transport": "stdio",
            "command": PYTHON_EXE,
            "args": [
                rf"{BASE_DIR}\custom_weather_mcp_server.py",
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }
)

search_tool = None
aviation_tools = {}
weather_tool = None
forecast_tool = None


async def initialize_mcp():
    global search_tool, aviation_tools, weather_tool, forecast_tool

    if search_tool is not None and aviation_tools and weather_tool is not None:
        return

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")
    for tool in tools:
        print(tool.name)

    search_tool = next(
        tool for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name in ["list_airports", "list_airlines"]
    }

    weather_tool = next(
        tool for tool in tools
        if tool.name == "get_current_weather"
    )

    forecast_tool = next(
        tool for tool in tools
        if tool.name == "get_forecast"
    )


async def tavily_mcp_search(query: str):
    await initialize_mcp()

    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )

    return result


async def aviation_mcp_call(tool_name: str, tool_args: dict = None):
    await initialize_mcp()

    tool = aviation_tools.get(tool_name)

    if not tool:
        return f"{tool_name} tool unavailable"

    result = await tool.ainvoke(tool_args or {})

    return result


async def get_airports():
    return await aviation_mcp_call("list_airports")


async def get_airlines():
    return await aviation_mcp_call("list_airlines")


async def weather_mcp_search(city: str):
    await initialize_mcp()

    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


async def forecast_mcp_search(city: str):
    await initialize_mcp()

    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def extract_destination(query: str):
    prompt = f"""
Extract only the destination city or country.

Query:
{query}

Return only destination name.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


async def main():
    await initialize_mcp()


if __name__ == "__main__":
    asyncio.run(main())