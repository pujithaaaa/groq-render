from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

class SummaryRequest(BaseModel):
    content: str

@app.post("/summarize")
async def summarize(data: SummaryRequest):
    if not data.content:
        return {"error": "No content provided"}

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
                    {"role": "user", "content": data.content}
                ],
                "temperature": 0.5
            }
        )

        if response.status_code != 200:
            return {"error": "Groq API error", "details": response.text}

        groq_result = response.json()
        return {
            "summary": groq_result["choices"][0]["message"]["content"]
        }

