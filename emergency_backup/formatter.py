import json

def prediction(filename):

    with open(filename, "r") as input_file:
        events = json.load(input_file)

    results = []

    for event in events:

        event_desc = {"probabilities": []}

        for outcome in event["outcomes"]:

            if event["resolved_outcome"] is None:
                individual_prob = float(1.0 / len(event["outcomes"]))
                truncated_num = int(individual_prob * 100) / 100
                market_dict = {"market": outcome, "probability": truncated_num}

            elif event["resolved_outcome"] is not None and outcome in event["resolved_outcome"]["value"]:
                market_dict = {"market": outcome, "probability": 1.0}

            elif event["resolved_outcome"] is not None and outcome not in event["resolved_outcome"]["value"]:
                market_dict = {"market": outcome, "probability": 0.0}

            event_desc["probabilities"].append(market_dict)

        results.append(event_desc)

    return results

if __name__ == "__main__":
    output = prediction("economics.json")
    with open("economics_results.json", "w") as file:
        json.dump(output, file, indent=4)