# Spam Detector

Full-stack spam detection app with:
- a FastAPI backend for text and `.eml` classification
- a React + Vite frontend UI

## Project structure

- `server/` - ML training and FastAPI inference API
- `client/` - React frontend
- `test-data/` - sample email data used for UI/testing

## Prerequisites

- Python 3.10+
- Node.js 18+ (or newer LTS)

## Backend setup (`server`)

1. Install dependencies:
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare data and train model:
   ```bash
   python scripts/download_data.py
   python train.py
   ```
3. Start API:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

API base URL: `http://127.0.0.1:8000`

## Frontend setup (`client`)

1. Install dependencies:
   ```bash
   cd client
   npm install
   ```
2. Start dev server:
   ```bash
   npm run dev
   ```

Frontend URL: `http://127.0.0.1:5173`

## Environment variables

- Frontend (`client/.env`)
  - `VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`)
- Backend (optional)
  - `MODEL_PATH`
  - `SPAM_THRESHOLD`
  - `CORS_ORIGINS`

## API endpoints

- `POST /health`
- `POST /classify` with JSON body: `{ "text": "..." }`
- `POST /classify/eml` with multipart file upload (`.eml`)

## Deploy on Render

This repo includes a Render blueprint file: `render.yaml`.

1. Push latest code to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select this GitHub repo and deploy.

Render creates two services:
- `spam-detector-api` (FastAPI backend)
- `spam-detector-web` (Vite static frontend)

### Important after deploy

- If your Render service names differ from:
  - `spam-detector-api`
  - `spam-detector-web`
  update `render.yaml` env vars accordingly:
  - backend `CORS_ORIGINS`
  - frontend `VITE_API_BASE_URL`
- Free plan services may sleep after inactivity, so first request can be slow.
