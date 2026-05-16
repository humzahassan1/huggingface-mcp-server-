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
    async with httpx.AsyncClient(follow_redirects=True) as client:
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


    
@mcp.tool()
async def get_model_info(model_id: str) -> str:
    """Get detailed information about a specific HuggingFace model."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            f"https://huggingface.co/api/models/{model_id}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )

        if response.status_code == 404:
            return f"Model '{model_id}' not found."
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"

        model = response.json()

        return (
            f"Model: {model.get('id', 'Unknown')}\n"
            f"Pipeline: {model.get('pipeline_tag', 'N/A')}\n"
            f"Downloads: {model.get('downloads', 'N/A')}\n"
            f"Likes: {model.get('likes', 'N/A')}\n"
            f"Tags: {', '.join(model.get('tags', []))}\n"
            f"License: {model.get('cardData', {}).get('license', 'N/A') if model.get('cardData') else 'N/A'}\n"
            f"Last Modified: {model.get('lastModified', 'N/A')}\n"
            f"URL: https://huggingface.co/{model.get('id', '')}"
        )


@mcp.tool()
async def search_datasets(query: str, limit: int = 5) -> str:
    """Search HuggingFace Hub for datasets by keyword."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            "https://huggingface.co/api/datasets",
            params={"search": query, "limit": limit, "sort": "downloads"},
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )

        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"

        datasets = response.json()

        if not datasets:
            return f"No datasets found for '{query}'"

        results = []
        for ds in datasets:
            results.append(
                f"- {ds.get('id', 'Unknown')}\n"
                f"  Downloads: {ds.get('downloads', 'N/A')}\n"
                f"  Tags: {', '.join(ds.get('tags', [])[:5])}"
            )

        return f"Found {len(datasets)} datasets for '{query}':\n\n" + "\n\n".join(results)


@mcp.tool()
async def get_dataset_info(dataset_id: str) -> str:
    """Get detailed information about a specific HuggingFace dataset."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(
            f"https://huggingface.co/api/datasets/{dataset_id}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )

        if response.status_code == 404:
            return f"Dataset '{dataset_id}' not found."
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"

        ds = response.json()

        return (
            f"Dataset: {ds.get('id', 'Unknown')}\n"
            f"Downloads: {ds.get('downloads', 'N/A')}\n"
            f"Likes: {ds.get('likes', 'N/A')}\n"
            f"Tags: {', '.join(ds.get('tags', []))}\n"
            f"Last Modified: {ds.get('lastModified', 'N/A')}\n"
            f"URL: https://huggingface.co/datasets/{ds.get('id', '')}"
        )

if __name__ == "__main__":
    mcp.run(transport="stdio")