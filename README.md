# SpudGuard — Full-Stack Potato Disease Classification

Classify potato leaf images into **Early Blight**, **Late Blight**, or **Healthy** using a TensorFlow/Keras model, with **Google Sign-In** and **AI-powered treatment advice** via Google Gemini.

## Architecture

```
┌─────────────────┐     Google OAuth      ┌──────────────────┐
│  React Frontend │ ◄────────────────────► │  Google Identity  │
│  (port 3000)    │                        └──────────────────┘
└────────┬────────┘
         │ JWT + REST
         ▼
┌─────────────────┐     Gemini API        ┌──────────────────┐
│  FastAPI Backend│ ◄────────────────────► │  Google Gemini   │
│  (port 8000)    │                        └──────────────────┘
└────────┬────────┘
         │ TensorFlow
         ▼
┌─────────────────┐
│  model_v1.keras │
└─────────────────┘
```

## Features

- **Google OAuth 2.0** — Sign in with Google; JWT session tokens protect all API routes
- **Disease classification** — Upload a leaf image; CNN model returns label + confidence
- **AI treatment advice** — Gemini generates symptoms, treatment, and prevention tips (falls back to built-in guidance if no API key)
- **React + Material-UI** — Drag-and-drop upload, results dashboard, user profile in navbar

## Prerequisites

- Python 3.10+
- Node.js 16+
- [Google Cloud Console](https://console.cloud.google.com/) OAuth 2.0 Web Client ID
- [Google AI Studio](https://aistudio.google.com/apikey) Gemini API key (optional but recommended)

## 1. Google OAuth setup

1. Go to [Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
2. Create an **OAuth 2.0 Client ID** of type **Web application**
3. Add authorized JavaScript origins:
   - `http://localhost:3000`
4. Copy the **Client ID** — use the same value in both `api/.env` and `frontend/.env`

## 2. Environment configuration

**Backend** (`api/.env`):

```bash
cd api
copy .env.example .env
```

Edit `api/.env`:

```env
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
JWT_SECRET_KEY=a-long-random-secret-string
GEMINI_API_KEY=your-gemini-api-key
FRONTEND_URL=http://localhost:3000
```

**Frontend** (`frontend/.env`):

```bash
cd frontend
copy .env.example .env
```

Edit `frontend/.env`:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
```

## 3. Install dependencies

```bash
# Backend
cd api
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install

# Optional: install root runner
cd ..
npm install
```

## 4. Run the project

### Option A — Run both together (from repo root)

```bash
npm start
```

### Option B — Run separately (two terminals)

**Terminal 1 — API** (must run from `api/` so the model path resolves):

```bash
cd api
python main.py
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm start
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/ping` | No | Health check |
| `POST` | `/auth/google` | No | Exchange Google ID token for JWT |
| `GET` | `/auth/me` | Yes | Current user profile |
| `POST` | `/predict` | Yes | Upload leaf image → classification |
| `POST` | `/ai/advice` | Yes | Get AI treatment advice for a diagnosis |

## Repository layout

| Path | Purpose |
|------|---------|
| `api/` | FastAPI backend — auth, prediction, AI advice |
| `api/auth.py` | Google token verification + JWT |
| `api/ai_service.py` | Gemini integration + fallback advice |
| `frontend/` | React app with Google Sign-In |
| `frontend/src/context/AuthContext.js` | Auth state & API client |
| `saved_models/` | Trained `model_v1.keras` weights |
| `training/` | Jupyter notebook + PlantVillage dataset |

## Training

See `training/training.ipynb` to train on the PlantVillage potato subset and export to `saved_models/model_v1.keras`.
