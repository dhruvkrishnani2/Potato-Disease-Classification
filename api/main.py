import os
from io import BytesIO

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from ai_service import get_ai_advice
from auth import (
    GoogleAuthRequest,
    PasswordLoginRequest,
    RegisterRequest,
    User,
    create_access_token,
    get_current_user,
    user_from_record,
    verify_google_token,
)
from user_store import authenticate_user, register_user
from config import get_settings

settings = get_settings()
app = FastAPI(title="SpudGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(settings.model_path)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


class AdviceRequest(BaseModel):
    disease: str
    confidence: float


def read_file_as_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))


@app.get("/")
async def root():
    return {
        "message": "SpudGuard API is running.",
        "frontend": settings.frontend_url,
        "docs": "http://localhost:8000/docs",
        "health": "/ping",
    }


@app.get("/ping")
async def ping():
    return {"status": "alive", "service": "SpudGuard API"}


@app.post("/auth/google", response_model=AuthResponse)
async def google_auth(body: GoogleAuthRequest):
    user = verify_google_token(body.credential)
    token = create_access_token(user)
    return AuthResponse(access_token=token, user=user)


@app.post("/auth/register", response_model=AuthResponse)
async def register(body: RegisterRequest):
    try:
        record = register_user(body.username, body.password, body.name, body.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    user = user_from_record(record, record["sub"])
    token = create_access_token(user)
    return AuthResponse(access_token=token, user=user)


@app.post("/auth/login", response_model=AuthResponse)
async def login(body: PasswordLoginRequest):
    record = authenticate_user(body.username, body.password)
    if not record:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    user = user_from_record(record, record["sub"])
    token = create_access_token(user)
    return AuthResponse(access_token=token, user=user)


@app.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        "class": predicted_class,
        "confidence": confidence,
        "user": current_user.email,
    }


@app.post("/ai/advice")
async def ai_advice(
    body: AdviceRequest,
    current_user: User = Depends(get_current_user),
):
    result = get_ai_advice(body.disease, body.confidence)
    return {
        "disease": body.disease,
        "confidence": body.confidence,
        "advice": result["advice"],
        "source": result["source"],
        "user": current_user.email,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(app, host="0.0.0.0", port=port)
