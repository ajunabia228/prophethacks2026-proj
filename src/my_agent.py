import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

#- Probabilities do NOT need to sum to 1 (each outcome is scored independently as YES/NO).

SYSTEM_PROMPT = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to assign a probability to EVERY possible outcome of the given event.

CALIBRATION GUIDELINES:
- Consider base rates, current standings, recent form, and any relevant context.
- Extremes (p < 0.05 or p > 0.95) require very strong evidence.
- For multi-outcome events (e.g. league winners), most outcomes should have low probability.
- Probabilities should sum close to 1, it will not be exact as you MUST round probabilities to 2 decimal places and assign 0.01 to very unlikely outcomes.
- Do NOT include marlet tickers, timestamps, or any information that is not the market or its probability.

Respond with ONLY valid JSON in this exact format, NO other text:
{
  "probabilities": [
    {
        "market": "<outcome name>", 
        "probability": <float 0.01-0.99>
    },
    ...
  ]
}"""


def predict(event: dict) -> dict:
    outcomes = event.get("outcomes") or []
    outcomes_list = "\n".join(f"- {o}" for o in outcomes)

    prompt = f"""
Event: {event['title']}
Category: {event['category']}
Close time: {event['close_time']}
Description: {event.get('description', '')}
Rules: {event.get('rules', '')}

Possible outcomes (assign a probability to each):
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

    #prob_map = {
    #    item["market"].strip().lower(): item["probability"]
    #    for item in data["probabilities"]
    #}

    #yes_outcome = outcomes[0] if outcomes else None
    #if yes_outcome:
    #    p_yes = prob_map.get(yes_outcome.strip().lower(), 0.5)
    #else:
    #    p_yes = 0.5

    #p_yes = max(0.01, min(0.99, float(p_yes)))

    probs = data["probabilities"]

    total = sum(item["probability"] for item in probs)
    for item in probs:
        item["probability"] = round(item["probability"] / total, 4)

    # Fix rounding error so sum is exactly 1.0
    diff = round(1.0 - sum(item["probability"] for item in probs), 4)
    if diff != 0:
        largest = max(probs, key=lambda x: x["probability"])
        largest["probability"] = round(largest["probability"] + diff, 4)

    return {"probabilities": probs}