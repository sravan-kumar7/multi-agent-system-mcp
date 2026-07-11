import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


# Project root:
# C:\multi-agent-system-with-mcp
BASE_DIR = Path(__file__).resolve().parent

# AviationStack Python source folder:
# C:\multi-agent-system-with-mcp\aviationstack-mcp\src
AVIATIONSTACK_SRC = BASE_DIR / "aviationstack-mcp" / "src"

# Load values from .env
load_dotenv(BASE_DIR / ".env")

# Support either environment-variable spelling
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)


def validate_configuration() -> None:
    """Validate required files and environment variables."""

    if not AVIATION_STACK_API_KEY:
        raise ValueError(
            "AviationStack API key is missing. Add this to your .env file:\n"
            "AVIATION_STACK_API_KEY=your_actual_api_key"
        )

    if not AVIATIONSTACK_SRC.exists():
        raise FileNotFoundError(
            f"AviationStack source folder was not found:\n"
            f"{AVIATIONSTACK_SRC}"
        )

    package_folder = AVIATIONSTACK_SRC / "aviationstack_mcp"

    if not package_folder.exists():
        raise FileNotFoundError(
            f"aviationstack_mcp package was not found:\n"
            f"{package_folder}"
        )


async def main() -> None:
    validate_configuration()

    print(f"Using Python: {sys.executable}")
    print(f"AviationStack source: {AVIATIONSTACK_SRC}")

    client = MultiServerMCPClient(
        {
            "aviationstack": {
                "transport": "stdio",

                # Automatically uses the active environment locally
                # and Streamlit's Python when deployed.
                "command": sys.executable,

                "args": [
                    "-m",
                    "aviationstack_mcp",
                    "mcp",
                    "run",
                ],

                "env": {
                    **os.environ,
                    "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY,

                    # Allows Python to locate aviationstack_mcp/src
                    "PYTHONPATH": str(AVIATIONSTACK_SRC),
                },
            }
        }
    )

    try:
        tools = await client.get_tools()

        print("\nAvailable AviationStack tools:\n")

        if not tools:
            print("No tools were returned by the AviationStack MCP server.")
            return

        for index, tool in enumerate(tools, start=1):
            print(f"{index}. {tool.name}")

        print("\nAviationStack MCP server is working successfully.")

    except Exception as error:
        print("\nAviationStack MCP server failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Reason: {error}")
        raise


if __name__ == "__main__":
    asyncio.run(main())