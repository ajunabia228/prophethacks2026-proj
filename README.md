# TrueOdds.AI 📈
TrueOdds.AI is a custom AI-powered forecasting agent built for the Prophet Arena benchmark. 
It combines Perplexity Sonar's real-time web search with live sports betting odds from The Odds API 
to generate calibrated probability estimates across sports, economics, and entertainment events. 
Rather than relying solely on training data, the agent actively researches each event before assigning 
probabilities to every possible outcome, producing predictions that are scored using Brier score, 
where lower is better and 0.0 is a perfect score.

<div align="center">
  <img src="https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/004/678/530/datas/gallery.jpg" alt="TrueOdds.AI Preview" />
</div>

This program was made for ProphetHacks 2026! 🎉 <br>
🔮 ProphetHacks 2026 Website: https://www.prophethacks.com/

## Collaborators 👥
Antonio Unabia, Daniel Danque, Emilio Calvo, Steve Nuevaorlanda

## Project Structure 🛠️
```
prophethacks2026-proj/
├── src/
│   ├── my_agent.py
│   └── server.py
├── sample_sets/
│   ├── events.json
│   ├── economics.json
│   └── entertainment.json
├── data/
│   ├── events.json
│   └── predictions.json
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack 🧰

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| API Framework | FastAPI |
| Deployment | Render |
| LLM Provider | OpenRouter |
| AI Model | Perplexity Sonar |
| Sports Odds Data | The Odds API |
| CLI & Scoring | ai-prophet-core |
| Package Management | pip |
| Version Control | Git & GitHub |

## Requirements ⚙️
- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key
- An [The Odds API](https://the-odds-api.com) key for sports events
- The `ai-prophet` CLI installed from the [ai-prophet](https://github.com/ai-prophet/ai-prophet) repository

## Setup 🖥️

### 1. Clone the repo and navigate to the project folder 📁
```powershell
git clone <your-repo-url>
cd prophethacks2026-proj
```

### 2. Create and activate a virtual environment 🌐
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies 🤖
```powershell
pip install ai-prophet-core ai-prophet
pip install -r requirements.txt
```

### 4. Configure environment variables 🔡
Create a `.env` file in the project root with the following:
```
ODDS_API_KEY_1=sk-or-...
ODDS_API_KEY_2=sk-or-...
ODDS_API_KEY_3=sk-or-...
OPENROUTER_API_KEY=...
PA_SERVER_URL=https://api.aiprophet.dev
```

Run these in your terminal before every session:
```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "C:\path\to\prophethacks2026-proj\src"
```

### 5. Create the data folder 📊
```powershell
mkdir data
```

## Usage 👨‍💻

### Retrieve the latest events
```powershell
prophet forecast retrieve -o data/events.json
```

### Run predictions locally
```powershell
prophet forecast predict --events data/events.json --local my_agent -o data/predictions.json
```

### Score predictions locally for testing
```powershell
prophet forecast evaluate --submission data/predictions.json --actuals data/actuals.json
```

### Run the HTTP endpoint server
```powershell
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

### Run predictions against the HTTP endpoint
```powershell
prophet forecast predict --events data/events.json --agent-url http://localhost:8000/predict -o data/predictions.json
```

## How It Works 🕵🏻
The agent sends each event to Perplexity Sonar Pro via OpenRouter along with all possible outcomes. The model uses built-in real-time web search to factor in current standings, recent form, and live odds before returning a probability for every outcome in JSON format.

For sports events, the agent additionally queries The Odds API to retrieve current betting odds and uses implied probabilities as an anchor for its estimates. For non-sports events, a two-step research call is made first to gather the current context before the final prediction is generated.

The agent supports multiple API keys with automatic fallback: if one key hits its credit limit or rate limit, it seamlessly switches to the next available key!

## Scoring 🎯
Predictions are scored using Brier score: `(probability - actual)²` averaged across all matched predictions.

- **0.0** — perfect score
- **0.25** — equivalent to always predicting 0.5
- Lower is better
