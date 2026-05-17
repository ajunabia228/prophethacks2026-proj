import os
import json
import time
import requests
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "predictions_cache.json")
CACHE_TTL_HOURS = 24  # Refresh predictions older than this

SPORT_MAP = {
    "KXNBA": "basketball_nba",
    "KXMLB": "baseball_mlb",
    "KXNHL": "icehockey_nhl",
    "KXPREMIERLEAGUE": "soccer_epl",
    "KXFACUP": "soccer_fa_cup",
    "KXBUNDESLIGA": "soccer_germany_bundesliga",
    "KXEUROLEAGUE": "basketball_euroleague",
    "KXATP": "tennis_atp",
    "KXWTA": "tennis_wta",
    "KXITF": "tennis_itf",
    "KXMLS": "soccer_usa_mls",
    "KXLALIGA": "soccer_spain_la_liga",
    "KXSERIEA": "soccer_italy_serie_a",
    "KXLIGUE1": "soccer_france_ligue_one",
    "KXCHAMPIONS": "soccer_uefa_champs_league",
    "KXNFL": "americanfootball_nfl",
    "KXNCAA": "americanfootball_ncaaf",
    "KXNBA2": "basketball_nba",
    "KXWNBA": "basketball_wnba",
    "KXPGA": "golf_pga_championship_winner",
    "KXMASTERS": "golf_masters_tournament_winner",
    "KXUFC": "mma_mixed_martial_arts",
    "KXBOXING": "boxing_boxing",
    "KXCRICKET": "cricket_icc_world_cup_2027_winner",
    "KXNASCAR": "motorsport_constructor_championship_winner",
    "KXF1": "motorsport_constructor_championship_winner",
    "KXPREMDARTS": None,
    "KXDARTS": None,
}

CATEGORY_HINTS = {
    "Sports": """
- If betting odds are provided, use implied probabilities as your primary anchor.
- Adjust slightly for recent form, injuries, home advantage, or playoff context.
- For championship/outright markets, only teams or players still in contention should have meaningful probability.
- For head-to-head matchups, consider current form, head-to-head record, and venue.
""",
    "Economics": """
- Search for the most recent actual value and analyst consensus forecast for this metric.
- Consider the trend over the last 3 months and any relevant macro context (inflation, Fed policy, trade conditions).
- For range-bucket questions, concentrate probability mass around the consensus forecast.
- Use a roughly normal distribution — outcomes further from consensus get exponentially less probability.
- Do not spread probability evenly; most mass should be on 2-4 adjacent buckets.
- For central bank decisions, check current market pricing (OIS swaps, futures) for rate expectations.
""",
    "Entertainment": """
- Search for current betting odds, fan polls, prediction markets, or critic consensus.
- For award shows, consider who the frontrunner is based on prior wins this season and critics' picks.
- For release/timing questions, search for any official announcements or credible rumors.
- For chart/streaming questions, consider recent momentum and historical baselines.
- Concentrate probability on 1-3 frontrunners; don't spread evenly.
""",
    "Politics": """
- Search for the most recent polling averages, prediction market prices, and expert forecasts.
- Consider structural factors: incumbency advantage, economic conditions, historical base rates.
- For electoral questions, weight recent polls heavily but account for historical polling error.
- Do not anchor too heavily on a single poll — use aggregates where possible.
- Concentrate probability on realistic outcomes; fringe outcomes get 0.01-0.03.
""",
    "Elections": """
- Search for the most recent polling averages, prediction market prices, and expert forecasts.
- Consider structural factors: incumbency advantage, economic conditions, historical base rates.
- For electoral questions, weight recent polls heavily but account for historical polling error.
- Do not anchor too heavily on a single poll — use aggregates where possible.
- Concentrate probability on realistic outcomes; fringe outcomes get 0.01-0.03.
""",
    "Science": """
- Search for the current state of the research or event in question.
- Consider historical base rates for similar events (e.g. rocket launches, clinical trial success rates).
- For binary outcomes, anchor on expert consensus and recent developments.
- Be conservative — scientific and technical outcomes are often uncertain.
""",
    "Technology": """
- Search for the latest news, official announcements, and analyst expectations.
- For product release questions, check official roadmaps and credible leaks.
- For market/adoption questions, consider current trends and historical growth rates.
- Weight official sources heavily over speculation.
""",
    "Finance": """
- Search for current market pricing, analyst consensus, and recent trends.
- For asset price questions, consider implied volatility and futures pricing.
- Use a distribution that reflects genuine uncertainty — financial outcomes are hard to predict.
- Concentrate mass around the current market consensus with fat tails.
""",
    "Weather": """
- Search for the latest meteorological forecasts from official sources (NOAA, ECMWF).
- Weight the most recent forecast model runs heavily.
- For extreme event questions, use historical base rates as a prior.
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


def is_cache_fresh(entry: dict) -> bool:
    cached_at = entry.get("cached_at", 0)
    age_hours = (time.time() - cached_at) / 3600
    return age_hours < CACHE_TTL_HOURS


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


def research_event(event: dict) -> str:
    """Two-step research for unknown or complex event types."""
    try:
        research_prompt = f"""Research this forecasting question and provide the most relevant current information:

Title: {event['title']}
Description: {event.get('description', '')}
Close time: {event['close_time']}

Provide:
1. Current status or most recent relevant data point
2. Expert consensus or market expectations if available
3. Key factors that would influence the outcome
4. Any recent developments that are relevant

Be concise and factual. Do not assign probabilities."""

        response = client.chat.completions.create(
            model="perplexity/sonar",
            max_tokens=500,
            messages=[{"role": "user", "content": research_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def predict(event: dict) -> dict:
    ticker = event.get("market_ticker", "")

    # Return cached result if fresh
    cache = load_cache()
    if ticker in cache and is_cache_fresh(cache[ticker]):
        return {"probabilities": cache[ticker]["probabilities"]}

    outcomes = event.get("outcomes") or []
    outcomes_list = "\n".join(f"- {o}" for o in outcomes)
    category = event.get("category", "")

    # Build category-aware system prompt
    category_hint = CATEGORY_HINTS.get(category)
    known_category = category_hint is not None
    if not category_hint:
        category_hint = "- Use all available context and current research to make calibrated estimates.\n- Concentrate probability mass on the most likely outcomes based on evidence."
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

    # Two-step research for unknown categories or complex events
    research_context = ""
    if not known_category or (known_category and not odds_context and category != "Sports"):
        research = research_event(event)
        if research:
            research_context = f"\nCURRENT RESEARCH:\n{research}\n"

    prompt = f"""
Event: {event['title']}
Category: {category}
Close time: {event['close_time']}
Description: {event.get('description', '')}
Rules: {event.get('rules', '')}
{odds_context}{research_context}
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

    # Cache with timestamp
    cache[ticker] = {
        "probabilities": probs,
        "cached_at": time.time(),
    }
    save_cache(cache)

    return {"probabilities": probs}