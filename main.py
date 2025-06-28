from fastapi import FastAPI, Request
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

@app.post("/summarize")
async def summarize(request: Request):
    body = await request.json()

    # Support both content string or raw DraftJS JSON
    if "content" in body and isinstance(body["content"], str):
        content = body["content"]
    elif "blocks" in body and isinstance(body["blocks"], list):
        # Extract text fields from DraftJS structure
        content = "\n".join(block.get("text", "") for block in body["blocks"] if block.get("text"))
    else:
        return {"error": "No valid content or blocks provided"}

    if not content.strip():
        return {"error": "No content to summarize"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "Summarize this text for a business newsletter"},
                    {"role": "user", "content": content}
                ],
                "temperature": 0.5
            }
        )

    if response.status_code != 200:
        return {"error": "Groq API error", "details": response.text}

    return {
        "summary": response.json()["choices"][0]["message"]["content"]
    }


