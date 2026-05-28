# PSMF Agent

PSMF Agent is a Gemini-powered coaching assistant for PSMF / fat-loss tracking. It includes:

- a terminal chat entrypoint in `main.py`
- a Telegram bot entrypoint in `telegram_bot.py`
- a Streamlit demo UI in `app.py`
- local memory/profile management
- local RAG indexing over the bundled PSMF protocol, food database, training guide, and symptom matrix

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your Gemini API key and, if using Telegram, your bot token:

```bash
GEMINI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
```

## Run

Terminal:

```bash
python main.py
```

Streamlit demo:

```bash
streamlit run app.py
```

Telegram bot:

```bash
python telegram_bot.py
```

## Data And Secrets

Do not commit `.env`, local vector indexes, virtual environments, caches, or `user_profiles.json`. These files may contain secrets, generated state, or personal health data.
