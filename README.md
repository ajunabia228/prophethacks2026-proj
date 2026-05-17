# ProphetHacks 2026 Forecasting Agent 📈

A custom forecasting agent built for the Prophet Arena benchmark. The agent uses an LLM via OpenRouter to generate calibrated probability estimates for binary-outcome events, scored using Brier score.

## Collaborators 👥

Antonio Unabia, Daniel Danque, Emilio Calvo, Steve Nuevaorlanda

## Project Structure 🛠️

prophethacks2026-proj/ <br>
├── src/ <br>
│   └── my_agent.py <br>
└── data/ <br>
├── events.json <br>
└── predictions.json <br>

## Requirements ⚙️

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier works)
- The `ai-prophet` CLI installed from the [ai-prophet](https://github.com/ai-prophet) repository

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
pip install ai-prophet-core ai-prophet openai
```

### 4. Set environment variables 🔡

Run these in your terminal before every session:

```powershell
$env:PYTHONUTF8 = "1"
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:PA_SERVER_URL = "https://api.aiprophet.dev"
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

### Run predictions

```powershell
prophet forecast predict --events data/events.json --local my_agent -o data/predictions.json
```

## How It Works 🕵🏻

The agent sends each event to an LLM (via OpenRouter) along with all possible outcomes. The model returns a probability for every outcome in JSON format. The agent then extracts the probability for the YES outcome (always the first entry in the outcomes list) and returns it as `p_yes`.

The placeholder model used during testing was `openai/gpt-oss-120b:free`, which is available on OpenRouter's free tier.

## Scoring 🎯

Predictions are scored using Brier score: `(probability - actual)²` averaged across all matched predictions.

- **0.0** — perfect score
- **0.25** — equivalent to always predicting 0.5
- Lower is better
