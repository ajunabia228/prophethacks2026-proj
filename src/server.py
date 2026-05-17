import os
import json
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI(title="ProphetHacks 2026 Forecast Agent")

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to assign a probability to EVERY possible outcome of the given event.

CALIBRATION GUIDELINES:
- Consider base rates, current standings, recent form, and any relevant context.
- Probabilities MUST sum to 1 across all outcomes for a single event.
- Extremes (p < 0.05 or p > 0.95) require very strong evidence.
- For multi-outcome events (e.g. league winners), most outcomes should have low probability.

Respond with ONLY valid JSON in this exact format, no other text:
{
  "probabilities": [
    {
        "market": "<outcome name>",
        "probability": <float 0.01-0.99>
    },
    ...
  ]
}"""


def get_probabilities(event: dict) -> list:
    outcomes = event.get("outcomes") or []
    outcomes_list = "\n".join(f"- {o}" for o in outcomes)

    prompt = f"""
Event: {event.get('title')}
Category: {event.get('category')}
Close time: {event.get('close_time')}
Description: {event.get('description', '')}
Rules: {event.get('rules', '')}

Possible outcomes (probabilities must sum to 1):
{outcomes_list}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    data = json.loads(text)

    # Normalize probabilities to ensure they sum to 1
    total = sum(item["probability"] for item in data["probabilities"])
    normalized = [
        {
            "market": item["market"],
            "probability": round(item["probability"] / total, 4)
        }
        for item in data["probabilities"]
    ]

    return normalized


@app.post("/predict")
async def predict(event: dict):
    probabilities = get_probabilities(event)
    return {"probabilities": probabilities}