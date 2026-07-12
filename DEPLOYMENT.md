# Deployment — How SpudGuard is Deployed on AWS

This document describes exactly how this project is deployed, so it can be reproduced, redeployed, or handed off to someone else.

## Stack overview

| Component | Service | Why |
|---|---|---|
| Backend (FastAPI + TensorFlow model) | **AWS Elastic Beanstalk** (Docker platform) | Runs the container, handles restarts/health checks, no manual server management |
| HTTPS for the backend | **Amazon CloudFront** | Elastic Beanstalk's default domain is HTTP-only; CloudFront sits in front of it for free HTTPS |
| Frontend (React) | **AWS Amplify Hosting** | Static hosting with HTTPS built in, deployed by uploading a build zip |
| Authentication | **Google OAuth 2.0** | Frontend gets a Google ID token, backend verifies it and issues its own JWT |
| AI treatment advice | **Google Gemini API** | Called from the backend, falls back to built-in guidance if no key is set |

```
Browser
  │
  ▼
AWS Amplify (React build, HTTPS)
  │  REACT_APP_API_URL = CloudFront domain
  ▼
Amazon CloudFront (HTTPS → HTTP)
  │
  ▼
AWS Elastic Beanstalk (Docker container: FastAPI + TensorFlow)
```

---

## Part 1: Backend on Elastic Beanstalk

### 1.1 Dockerfile

The backend is containerized with a single-stage `Dockerfile` in `api/`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=model_v1.keras
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key points:
- `MODEL_PATH=model_v1.keras` works because a copy of the model lives directly inside `api/` (`api/model_v1.keras`), so no separate `saved_models/` folder needs to be copied into the image.
- The `CMD` runs `main:app`, **not** `api.main:app` — the app's modules (`main.py`, `auth.py`, `config.py`, `ai_service.py`, `user_store.py`) all import each other with flat imports (e.g. `from config import get_settings`), not as a Python package. Any `api.`-prefixed import breaks the container at startup (see Troubleshooting below).

A matching `.dockerignore` keeps secrets and junk out of the image:
```
.env
.env.example
__pycache__/
*.pyc
users.json
model_v1.h5
```

### 1.2 Packaging the source bundle

Elastic Beanstalk's Docker platform deploys from a zip file. The zip must have `Dockerfile` **at its root** — not nested inside an `api/` folder:

```bash
cd Potato-Disease-Classification/api
zip -r ../api-deploy.zip . -x "__pycache__/*" ".env" "*.pyc"
```

The `.` (zip the *contents* of the current folder) is what keeps the structure flat. Zipping the parent folder instead (e.g. right-clicking `api/` from outside it) nests everything one level too deep and breaks the deployment.

### 1.3 Creating the environment (AWS Console)

1. Elastic Beanstalk → **Create application**
   - Platform: **Docker**
   - Application code: **Upload your code** → the zip from 1.2
2. **Service role** and **EC2 instance profile**: created via the "Create role" links in the wizard (uses AWS's default managed policies for Elastic Beanstalk)
3. **Environment type**: Single instance (no load balancer — sufficient for this project's traffic)
4. **Environment properties** (Configuration → Software → Edit):
   - `GOOGLE_CLIENT_ID`
   - `JWT_SECRET_KEY`
   - `GEMINI_API_KEY`
   - `FRONTEND_URL` (set to the Amplify URL once it exists)
   - `CORS_ORIGINS` (same as `FRONTEND_URL`)

### 1.4 Redeploying after a code change

```bash
cd api
zip -r ../api-deploy.zip . -x "__pycache__/*" ".env" "*.pyc"
```
Then in the console: environment → **Upload and deploy** → select the new zip → **Deploy**.

---

## Part 2: HTTPS via CloudFront

Elastic Beanstalk's default domain (`*.elasticbeanstalk.com`) only serves plain HTTP. Since the frontend is served over HTTPS (via Amplify), browsers block the frontend from calling an HTTP API — this is "mixed content" blocking, and it fails silently (no popup, just failed requests).

**Fix:** put CloudFront in front of the Elastic Beanstalk domain to get free HTTPS.

1. CloudFront → **Create distribution**
2. **Origin type**: Other (the EB domain isn't S3, Load Balancer, or API Gateway, so this is the correct choice — it accepts any publicly resolvable domain)
3. **Origin domain**: the EB domain, no protocol prefix, e.g. `spudgrd-env.eba-xxxxxx.eu-north-1.elasticbeanstalk.com`
4. **Origin path**: left empty (the app is served at the root, not a subpath)
5. **Origin protocol**: HTTP only (the EB backend itself has no SSL certificate — CloudFront terminates HTTPS for the browser and talks HTTP to the origin)
6. **WAF**: not enabled (adds cost, unnecessary for this project's traffic level)
7. Everything else left at defaults

Once deployed (~5 minutes), CloudFront gives a domain like `https://d123abc456.cloudfront.net` — this is what the frontend's `REACT_APP_API_URL` points to, not the raw EB domain.

---

## Part 3: Frontend on AWS Amplify

1. Point the frontend at the CloudFront URL, in `frontend/.env`:
   ```env
   REACT_APP_API_URL=https://d123abc456.cloudfront.net
   REACT_APP_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   ```
2. Build:
   ```bash
   cd frontend
   npm install
   npm run build
   ```
3. Zip the **contents** of `build/`:
   ```bash
   cd build && zip -r ../build.zip . && cd ..
   ```
4. Amplify console → **Host a web app** → **Deploy without a Git provider** → upload `build.zip`

Amplify gives a URL like `https://main.d1a2b3c4.amplifyapp.com`.

---

## Part 4: Wiring it all together

1. **Google Cloud Console** → OAuth Client → **Authorized JavaScript origins** → add the Amplify URL
2. **Elastic Beanstalk** → Configuration → Software → Environment properties → update:
   - `FRONTEND_URL` = Amplify URL
   - `CORS_ORIGINS` = Amplify URL
   → **Apply**
3. Open the Amplify URL, sign in with Google, upload a leaf photo, confirm prediction + AI advice both return correctly

---

## Tearing it down

```
Elastic Beanstalk → environment → Actions → Terminate environment
Elastic Beanstalk → Applications → Delete application
CloudFront → distribution → Disable, then Delete (once disabled)
Amplify → app → App settings → General → Delete app
```
