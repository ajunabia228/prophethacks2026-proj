import json
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, ".")
from my_agent import predict

with open("../sample_sets/resolved.json") as f:
    events = json.load(f)

predictions = []
for event in events:
    print(f"Predicting {event['market_ticker']}...")
    result = predict(event)
    predictions.append({
        "market_ticker": event["market_ticker"],
        **result
    })

output = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "predictions": predictions
}

with open("../data/resolved_predictions.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Done. {len(predictions)} predictions written.")