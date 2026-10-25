# Observable Incident Triage Agent

FastAPI service for the ga5-incident-agent/v2 contract.

## Environment variables

- AI_API_KEY: Groq API key. Configure it in Render Environment.
- AI_BASE_URL: Defaults to https://api.groq.com/openai/v1.
- OPENAI_MODEL: Defaults to openai/gpt-oss-20b.
- DATABASE_PATH: SQLite database file path.

Never commit API keys or credentials.

## Run locally

Create and activate a virtual environment, then run:

    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Health check:

    http://127.0.0.1:8000/healthz

## Deploy

Deploy this repository as a Render Web Service.

Configure AI_API_KEY in Render's Environment settings.

The service exposes:

- POST /v2/incidents
- GET /v2/incidents/{runId}
- POST /v2/incidents/{runId}/receipts
- GET /healthz

## Security

The agent should diagnose an incident before dispatching a response.
Diagnostic receipts must be evaluated before an effect is dispatched.
Destructive actions must follow the supplied approval policy.
Do not expose sensitive incident fields in public responses or telemetry.
