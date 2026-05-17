import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to assign a probability to EVERY possible outcome of the given event.

CALIBRATION GUIDELINES:
- Consider base rates, current standings, recent form, and any relevant context.
- Probabilities do NOT need to sum to 1 (each outcome is scored independently as YES/NO).
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
        ],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    data = json.loads(text)

    prob_map = {
        item["market"].strip().lower(): item["probability"]
        for item in data["probabilities"]
    }

    yes_outcome = outcomes[0] if outcomes else None
    if yes_outcome:
        p_yes = prob_map.get(yes_outcome.strip().lower(), 0.5)
    else:
        p_yes = 0.5

    p_yes = max(0.01, min(0.99, float(p_yes)))

    return {
    "market": yes_outcome,
    "p_yes": p_yes,
    "rationale": " | ".join(
        f"{item['market']}: {item['probability']}"
        for item in data["probabilities"]
    )
}