import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")
from my_agent import predict

with open("../sample_sets/resolved.json") as f:
    events = json.load(f)

# Test just one event
events = [e for e in events if e["market_ticker"] == "KXITFWMATCH-26MAY12NAJEBS"]

predictions = []
for event in events:
    print(f"Predicting {event['market_ticker']}...")
    result = predict(event)
    print(json.dumps(result, indent=2))
    predictions.append({
        "market_ticker": event["market_ticker"],
        **result
    })

output = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "predictions": predictions
}

with open("../data/test_pro.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Done.")