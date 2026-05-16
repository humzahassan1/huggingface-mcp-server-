# huggingface-mcp-server-
MCP server for searching HuggingFace models, datasets, and running inference
# 🤗 HuggingFace MCP Server

An MCP (Model Context Protocol) server that connects AI agents to the HuggingFace ecosystem. Search models, explore datasets, and get detailed metadata — all through a standardized tool interface.

Built for the [Dedalus Labs](https://dedaluslabs.ai) MCP marketplace.

## Tools

| Tool | Description |
|------|-------------|
| `search_models` | Search HuggingFace Hub for ML models by keyword, sorted by downloads |
| `get_model_info` | Get detailed info about a specific model (downloads, likes, tags, license, etc.) |
| `search_datasets` | Search HuggingFace Hub for datasets by keyword |
| `get_dataset_info` | Get detailed info about a specific dataset |

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/humzahassan1/huggingface-mcp-server-.git
cd huggingface-mcp-server-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Then edit `.env` and add your HuggingFace API token (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):


### 4. Run the server

```bash
python src/server.py
```

## Example Usage

Once connected to an MCP client (like Dedalus), an agent can:

- **"Search for text generation models"** → calls `search_models("text generation")`
- **"Tell me about the BERT model"** → calls `get_model_info("bert-base-uncased")`
- **"Find sentiment analysis datasets"** → calls `search_datasets("sentiment analysis")`
- **"Get details on the IMDB dataset"** → calls `get_dataset_info("imdb")`

## Tech Stack

- **Python** — core language
- **Dedalus MCP** — MCP server framework
- **httpx** — async HTTP client
- **HuggingFace API** — model and dataset metadata
