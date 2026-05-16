import os
import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

response = httpx.get(
    "https://huggingface.co/api/models",
    params={"search": "text-generation", "limit": 3, "sort": "downloads"},
    headers={"Authorization": f"Bearer {token}"},
)

print(response.status_code)
for model in response.json():
    print(f"- {model['id']} ({model.get('downloads', 'N/A')} downloads)")