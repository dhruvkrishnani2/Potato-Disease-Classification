Deployment steps for GCP Cloud Run

Prerequisites
- A Google Cloud project and a service account with Cloud Run and Cloud Build permissions.
- Create a JSON key for the service account and add it to the GitHub repo secrets as `GCP_SA_KEY`.
- Add `GCP_PROJECT_ID` secret with your project id.
- (Optional) Add `CLOUD_RUN_REGION` secret (default `us-central1`).

How it works
- On push to `main`, the GitHub Action will build a Docker image with `gcloud builds submit`, push it to Container Registry, and deploy to Cloud Run as the `spudguard` service.

Local manual deploy (if you prefer)
1. Build image locally:

```bash
docker build -t gcr.io/<PROJECT_ID>/spudguard:latest .
```

2. Push and deploy with gcloud:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud builds submit --tag gcr.io/<PROJECT_ID>/spudguard:latest
gcloud run deploy spudguard --image gcr.io/<PROJECT_ID>/spudguard:latest --platform managed --region us-central1 --allow-unauthenticated
```

Notes
- Ensure your model files (in `api/` or `saved_models/`) are included in the repository; they are copied into the container by the included `Dockerfile`.
- Set any runtime environment variables (JWT secret, Google keys, GEMINI key) in Cloud Run service settings after deployment.
