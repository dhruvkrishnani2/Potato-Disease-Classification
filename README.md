# Potato disease classification

Classify potato leaf images into **Early Blight**, **Late Blight**, or **Healthy** using a TensorFlow/Keras model trained on the [PlantVillage](https://arxiv.org/abs/1511.08060) potato subset. This repo includes a FastAPI service, a React web UI, an optional Streamlit app, training notebooks, and sample Google Cloud–style deployment code.

## Features

- REST API (`POST /predict`) for image upload and JSON predictions with confidence scores
- React + Material-UI frontend that posts images to the API (configure `REACT_APP_API_URL`)
- Jupyter notebook pipeline for training and exporting `saved_models/model_v1.keras`
- Optional `gcp/` helpers for bucket-hosted models (reference implementation)

## Repository layout

| Path | Purpose |
|------|---------|
| `api/` | FastAPI app (`main.py`), API dependencies (`requirements.txt`), `runtime.txt` for Python version on hosts that use it |
| `saved_models/` | Trained model (`model_v1.keras`) used by the API |
| `frontend/` | Create React App UI (axios → `/predict`) |
| `app1.py` | Streamlit demo (expects a Keras model path—align with your local `.keras` / `.h5` file) |
| `training/` | `training.ipynb` and expected `PlantVillage/` data layout for potato classes |
| `gcp/` | Example Cloud Functions–style code loading a model from Google Cloud Storage |

## Prerequisites

- **Python** 3.10+ recommended (see `api/runtime.txt` for one deployment target)
- **Node.js** 16+ for the React app
- Trained weights: ensure `saved_models/model_v1.keras` is present (train with `training/training.ipynb` or copy from your artifacts)

## Quick start — API

Install dependencies and start the server **from the `api` directory** so the model path `../saved_models/model_v1.keras` resolves correctly.

```bash
cd api
pip install -r requirements.txt
python main.py
```

The app listens on `http://localhost:8000` by default.

Alternatively, with uvicorn explicitly:

```bash
cd api
uvicorn main:app --host localhost --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ping` | Health check |
| `POST` | `/predict` | Multipart form field `file`: leaf image |

Example response:

```json
{
  "class": "Early Blight",
  "confidence": 0.97
}
```

CORS is enabled for `http://localhost` and `http://localhost:3000` (see `api/main.py`).

## Quick start — React frontend

```bash
cd frontend
npm install
```

Copy the example environment file and point it at your API (include the `/predict` path):

```bash
copy .env.example .env
```

On Linux or macOS, use `cp` instead of `copy`. Set `REACT_APP_API_URL` to your predict URL, e.g. `http://localhost:8000/predict`.

```bash
npm start
```

The dev server runs on port 3000 and will call the API when you drop in an image.

## Streamlit (`app1.py`)

From the repository root (after installing Streamlit and TensorFlow):

```bash
pip install streamlit tensorflow pillow numpy
streamlit run app1.py
```

Update the `load_model()` path in `app1.py` if your weights live at `saved_models/model_v1.keras` instead of `api/model_v1.h5`.

## Training

1. Obtain the PlantVillage dataset and arrange the potato classes under `training/PlantVillage/` as in `training/training.ipynb` (e.g. `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`).
2. Open `training/training.ipynb` in Jupyter, run the cells, and export the trained model to `saved_models/model_v1.keras` (or adjust the API load path to match your export).

Training uses 256×256 inputs and typical Keras image augmentation; see the notebook for batch size, epochs, and architecture details.

## Google Cloud (`gcp/`)

The `gcp/` folder contains example code that downloads a model from a GCS bucket and runs inference. Update bucket names, object paths, and TensorFlow versions to match your project; treat it as a starting point, not a turnkey deploy.
