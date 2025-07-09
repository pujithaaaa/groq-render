from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class SummaryRequest(BaseModel):
    name: str
    content: str

# Root route for browser testing
@app.get("/")
def home():
    return {"status": "Groq summarizer is running ✅"}

# POST endpoint to generate summary
@app.post("/summarize")
async def summarize(data: SummaryRequest):
    if not data.content:
        return {"error": "No content provided"}

    prompt = (
        f"You are an assistant that writes polished newsletter summaries for *quarterly individual achievements*.\n"
        f"The person's name is {data.name}.\n\n"
        "The input will contain *flat, field-wise labeled text* (e.g., **Go-Lives:**, **Feedback:**, **Achievements:**, etc.) describing this individual's accomplishments.\n\n"
        "Generate a professional summary that:\n"
        "- Begins with the individual's name (e.g., '**Pujitha's Q2 Highlights:**')\n"
        "- Uses bullet points\n"
        "- Clearly labels each section using the field headers from the input (e.g., **Go-Lives:**)\n"
        "- Skips any sections that are empty or irrelevant\n"
        "- Keeps the length under 270 words\n"
        "- Writes in third-person point of view (focus only on the individual)\n"
        "- Does NOT mention or refer to 'we', 'our team', 'our', or 'us'\n"
        "- Does NOT group content under abstract categories like 'Successes', 'Looking Ahead', or 'Achievements'. Use the provided field labels only\n"
        "- Does NOT include markdown formatting in the output (use plain text only — no **bold**, no *italics*)\n\n"
        "Treat all content as if it's attributed solely to the individual named above. Do not imply team involvement."
    )

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
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": data.content}
                ],
                "temperature": 0.5
            }
        )

        if response.status_code != 200:
            return {"error": "Groq API error", "details": response.text}

        groq_result = response.json()
        return {"summary": groq_result["choices"][0]["message"]["content"]}
