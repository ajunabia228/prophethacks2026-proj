import os
import json
import requests
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "predictions_cache.json")

SPORT_MAP = {
    "KXNBA": "basketball_nba",
    "KXMLB": "baseball_mlb",
    "KXNHL": "icehockey_nhl",
    "KXPREMIERLEAGUE": "soccer_epl",
    "KXFACUP": "soccer_fa_cup",
    "KXBUNDESLIGA": "soccer_germany_bundesliga",
    "KXEUROLEAGUE": "basketball_euroleague",
    "KXATP": "tennis_atp_french_open",
}

CATEGORY_HINTS = {
    "Sports": """
- If betting odds are provided, use implied probabilities as your primary anchor.
- Adjust slightly for recent form, injuries, home advantage, or playoff context.
- For championship/outright markets, only teams still in contention should have meaningful probability.
""",
    "Economics": """
- Search for the most recent actual value and analyst consensus forecast for this metric.
- Consider the trend over the last 3 months and any relevant macro context.
- For range-bucket questions, concentrate probability mass around the consensus forecast.
- Use a roughly normal distribution — outcomes further from consensus get exponentially less probability.
- Do not spread probability evenly; most mass should be on 2-4 adjacent buckets.
""",
    "Entertainment": """
- Search for current betting odds, fan polls, prediction markets, or critic consensus.
- For award shows, consider who the frontrunner is based on prior wins this season.
- For release/timing questions, search for any official announcements or credible rumors.
- Concentrate probability on 1-3 frontrunners; don't spread evenly.
""",
}

BASE_SYSTEM_PROMPT = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to assign a probability to EVERY possible outcome of the given event.

CALIBRATION GUIDELINES:
- Probabilities must sum to 1.0 exactly.
- Do not spread probability evenly — concentrate mass where evidence points.
- Minimum probability for any outcome is 0.01.
- Extremes (p > 0.90 for a single outcome) require very strong evidence.

{category_hint}

Respond with ONLY valid JSON in this exact format, NO other text:
{{
  "probabilities": [
    {{
        "market": "<outcome name>",
        "probability": <float 0.01-0.99>
    }},
    ...
  ]
}}"""


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_sport_key(event_ticker: str) -> str | None:
    for prefix, sport_key in SPORT_MAP.items():
        if event_ticker.startswith(prefix):
            return sport_key
    return None


def fetch_odds(sport_key: str) -> str:
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us,uk",
            "markets": "h2h,outrights",
            "oddsFormat": "decimal",
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return ""
        games = resp.json()
        if not games:
            return ""

        lines = []
        for game in games[:5]:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            lines.append(f"Match: {away} vs {home}")
            for bookmaker in game.get("bookmakers", [])[:2]:
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        implied = round(1 / outcome["price"], 3)
                        lines.append(
                            f"  {outcome['name']}: {outcome['price']} decimal odds (implied {implied})"
                        )
        return "\n".join(lines)
    except Exception:
        return ""


def predict(event: dict) -> dict:
    ticker = event.get("market_ticker", "")

    # Return cached result if available
    cache = load_cache()
    if ticker in cache:
        return cache[ticker]

    outcomes = event.get("outcomes") or []
    outcomes_list = "\n".join(f"- {o}" for o in outcomes)
    category = event.get("category", "")

    # Build category-aware system prompt
    category_hint = CATEGORY_HINTS.get(
        category, "- Use all available context to make calibrated estimates."
    )
    system_prompt = BASE_SYSTEM_PROMPT.format(category_hint=category_hint)

    # Detect binary yes/no
    is_binary = len(outcomes) == 2 and set(o.lower() for o in outcomes) == {"yes", "no"}

    # Fetch odds for sports events
    odds_context = ""
    if category == "Sports" and ODDS_API_KEY:
        sport_key = get_sport_key(event.get("event_ticker", ""))
        if sport_key:
            odds_data = fetch_odds(sport_key)
            if odds_data:
                odds_context = f"\nCURRENT BETTING ODDS (use implied probabilities as anchor):\n{odds_data}\n"

    prompt = f"""
Event: {event['title']}
Category: {category}
Close time: {event['close_time']}
Description: {event.get('description', '')}
Rules: {event.get('rules', '')}
{odds_context}
Possible outcomes (assign a probability to each):
{outcomes_list}
"""

    if is_binary:
        prompt += "\nThis is a binary YES/NO question. Assign probability to Yes based on how likely the described event is to occur, and the remainder to No. Research current information to inform your estimate."

    try:
        response = client.chat.completions.create(
            model="perplexity/sonar",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)
        probs = data["probabilities"]

    except Exception:
        # Fallback to uniform distribution if anything fails
        n = len(outcomes)
        probs = [{"market": o, "probability": round(1.0 / n, 4)} for o in outcomes]

    # Normalize
    total = sum(item["probability"] for item in probs)
    for item in probs:
        item["probability"] = round(item["probability"] / total, 4)

    # Fix rounding error so sum is exactly 1.0
    diff = round(1.0 - sum(item["probability"] for item in probs), 4)
    if diff != 0:
        largest = max(probs, key=lambda x: x["probability"])
        largest["probability"] = round(largest["probability"] + diff, 4)

    result = {"probabilities": probs}

    # Cache and return
    cache[ticker] = result
    save_cache(cache)

    return result