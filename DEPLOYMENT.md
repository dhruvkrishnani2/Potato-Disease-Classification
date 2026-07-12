# Deploying SpudGuard — Render (API) + Vercel (frontend)

Two independent deployments, wired together with env vars. Do the API first
since the frontend needs its final URL.

## 0. Push to GitHub

Render and Vercel both deploy from a GitHub repo.

```bash
git add .
git commit -m "Add Render deployment config"
git push origin main
```

## 1. Deploy the API on Render

1. Go to https://dashboard.render.com → **New +** → **Web Service**
2. Connect your GitHub repo (grant Render access if first time)
3. Render should auto-detect `render.yaml` at the repo root and offer to
   create the `spudguard-api` service from it (a "Blueprint"). If it doesn't
   pick it up automatically, create the service manually with:
   - **Runtime:** Docker
   - **Dockerfile path:** `./Dockerfile`
   - **Docker build context:** `.` (repo root)
   - **Instance type:** Free (fine for a demo/capstone; cold-starts after
     15 min idle — bump to Starter if you need it always warm for a viva)
4. Set environment variables on the service (Render → your service →
   **Environment**):

   | Key | Value |
   |---|---|
   | `GOOGLE_CLIENT_ID` | same client ID from `api/.env` |
   | `JWT_SECRET_KEY` | let Render auto-generate (already set in `render.yaml`), or paste your own long random string |
   | `GEMINI_API_KEY` | your Gemini key |
   | `FRONTEND_URL` | your future Vercel URL, e.g. `https://spudguard.vercel.app` (you can update this after step 2) |
   | `CORS_ORIGINS` | same as `FRONTEND_URL` — comma-separate if you add more origins later |

5. Deploy. First build takes a while (TensorFlow is a large dependency).
6. Once live, confirm the API is up:

   ```bash
   curl https://spudguard-api.onrender.com/ping
   ```

   Expect `{"status":"alive","service":"SpudGuard API"}`.

## 2. Deploy the frontend on Vercel

1. Go to https://vercel.com/new and import the same GitHub repo
2. Set **Root Directory** to `frontend`
3. Framework preset: Create React App (Vercel should detect this automatically)
4. Add environment variables (Vercel → Project → **Settings → Environment Variables**):

   | Key | Value |
   |---|---|
   | `REACT_APP_API_URL` | your Render URL, e.g. `https://spudguard-api.onrender.com` |
   | `REACT_APP_GOOGLE_CLIENT_ID` | same client ID as the backend |

5. Deploy. Vercel gives you a URL like `https://spudguard.vercel.app`.
6. `vercel.json` (already in `frontend/`) handles SPA routing rewrites, so
   client-side routes won't 404 on refresh.

## 3. Close the loop

Now that both URLs are final:

1. **Update Render env vars** — set `FRONTEND_URL` and `CORS_ORIGINS` to the
   real Vercel URL from step 2, then redeploy (or Render will do it
   automatically on env var save).
2. **Update Google OAuth origins** — in
   [Google Cloud Console → Credentials](https://console.cloud.google.com/apis/credentials),
   edit your OAuth 2.0 Web Client and add the Vercel URL to
   **Authorized JavaScript origins**.
3. Reload the Vercel app and confirm Google Sign-In and `/predict` both work
   end-to-end.

## Notes

- The model (`saved_models/model_v1.keras`, ~2.2 MB) is baked into the
  Docker image at build time — no external storage needed for a project
  this size.
- `api/model_v1.keras` and `api/model_v1.h5` in the repo look like leftover
  duplicates (the app loads from `MODEL_PATH=../saved_models/model_v1.keras`
  by default). You can delete them to slim the image; not required.
- Free-tier Render web services spin down after 15 minutes of inactivity;
  the first request after that takes ~30-60s to cold-start. Mention this if
  you're demoing live in a viva, or upgrade the plan beforehand.
- Redeploys: pushing to `main` auto-redeploys both Render and Vercel once
  connected — no CI/CD YAML needed for this simpler split (unlike the
  Cloud Run path, which used `.github/workflows/deploy-cloudrun.yml`).
