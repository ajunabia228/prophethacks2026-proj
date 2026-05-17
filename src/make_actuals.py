import json

with open("../sample_sets/resolved.json") as f:
    events = json.load(f)

actuals = {}
for event in events:
    resolved = event.get("resolved_outcome")
    if resolved and resolved.get("value"):
        actuals[event["market_ticker"]] = resolved["value"][0]

with open("../data/actuals.json", "w") as f:
    json.dump(actuals, f, indent=2)

print(f"Done. {len(actuals)} actuals written.")