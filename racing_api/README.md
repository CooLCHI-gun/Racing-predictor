# Racing Predictor Flask API

Production-style Flask backend for racing predictions with modular architecture, validation, consistent JSON responses, and test coverage.

This API is intentionally built with realistic mock data first so you can plug in your existing predictor logic later without changing route contracts.

## Features

- Flask application factory pattern
- Blueprint-based route organization
- Service layer for business logic and data access
- Pydantic schema validation for query params and output models
- Consistent success/error JSON response format
- Global error handlers for 400/404/500 flows
- CORS enabled for `/api/*` routes (Base44-ready)
- Gunicorn-compatible for deployment platforms like Railway
- Pytest suite covering core endpoints

## Project Structure

```text
racing_api/
  app/
    __init__.py
    config.py
    extensions.py
    api/
      __init__.py
      health.py
      predictions.py
      results.py
      summary.py
    services/
      prediction_service.py
      result_service.py
      summary_service.py
    schemas/
      prediction_schema.py
      result_schema.py
      summary_schema.py
    utils/
      responses.py
      errors.py
  tests/
    conftest.py
    test_health.py
    test_predictions.py
    test_results.py
    test_summary.py
  run.py
  requirements.txt
  .env.example
  README.md
  Procfile
  .gitignore
```

## API Endpoints

- `GET /api/health`
- `GET /api/predictions`
- `GET /api/predictions/<race_id>`
- `GET /api/results`
- `GET /api/results/<race_id>`
- `GET /api/summary`

### Query Parameters

- `GET /api/predictions?date=2026-04-05`
- `GET /api/predictions?venue=ST`
- `GET /api/results?status=completed`
- `GET /api/results?venue=HV`
- `GET /api/summary?window=7d`

## Response Convention

Success response shape:

```json
{
  "success": true,
  "message": "Request successful",
  "timestamp": "2026-04-05T07:00:00+00:00",
  "data": {},
  "meta": {}
}
```

Error response shape:

```json
{
  "success": false,
  "error": {
    "code": "validation_error",
    "message": "window must be in format like 7d, 30d, or all"
  },
  "timestamp": "2026-04-05T07:00:00+00:00"
}
```

## Setup (Windows PowerShell in VS Code)

From the workspace root (`Racing-predictor`):

```powershell
cd .\racing_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Copy environment file:

```powershell
Copy-Item .env.example .env
```

## Run Locally

```powershell
python .\run.py
```

Default URL:

- `http://127.0.0.1:5000/api/health`

## Example URLs to Test

- `http://127.0.0.1:5000/api/health`
- `http://127.0.0.1:5000/api/predictions`
- `http://127.0.0.1:5000/api/predictions/ST_R1_2026-04-05`
- `http://127.0.0.1:5000/api/predictions?venue=HV`
- `http://127.0.0.1:5000/api/results?status=completed`
- `http://127.0.0.1:5000/api/results/ST_R1_2026-04-05`
- `http://127.0.0.1:5000/api/summary?window=7d`

## Run Tests

```powershell
pytest -q
```

## Deployment Readiness (Railway/Gunicorn)

- `Procfile` uses: `web: gunicorn run:app`
- App entry point is `run.py` with exported `app` object
- Environment variables are loaded via `python-dotenv`

Typical production env setup:

- `FLASK_ENV=production`
- `FLASK_RUN_HOST=0.0.0.0`
- `FLASK_RUN_PORT` provided by platform
- `CORS_ALLOWED_ORIGINS` set to your frontend domain(s)

## Replacing Mock Data With Real Predictor Logic

Current mock data lives in service modules:

- `app/services/prediction_service.py`
- `app/services/result_service.py`
- `app/services/summary_service.py`

To integrate your real system safely:

1. Keep route contracts unchanged (`/api/*` and JSON schema shapes).
2. Replace service internals with calls to your existing project scripts/modules.
3. Map your real fields to schema output models.
4. Keep validation in `app/schemas/*` so frontend contract remains stable.
5. Keep response helpers in `app/utils/responses.py` to preserve API consistency.

This lets your Base44 frontend integrate once and remain stable while backend logic evolves.
