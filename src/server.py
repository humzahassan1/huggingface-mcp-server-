import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Get HuggingFace token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Create the MCP server
mcp = FastMCP("huggingface-mcp-server")

@mcp.tool()
async def search_models(query: str, limit: int = 5) -> str:
    """Search HuggingFace Hub for machine learning models by keyword."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://huggingface.co/api/models",
            params={"search": query, "limit": limit, "sort": "downloads"},
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        models = response.json()
        
        if not models:
            return f"No models found for '{query}'"
        
        results = []
        for model in models:
            results.append(
                f"- {model.get('id', 'Unknown')}\n"
                f"  Downloads: {model.get('downloads', 'N/A')}\n"
                f"  Tags: {', '.join(model.get('tags', [])[:5])}\n"
                f"  Pipeline: {model.get('pipeline_tag', 'N/A')}"
            )
        
        return f"Found {len(models)} models for '{query}':\n\n" + "\n\n".join(results)

if __name__ == "__main__":
    mcp.run(transport="stdio")