from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests
from google.oauth2 import id_token
from pydantic import BaseModel

from api.config import get_settings

security = HTTPBearer()


class User(BaseModel):
    email: str
    name: str
    picture: Optional[str] = None
    sub: str


class GoogleAuthRequest(BaseModel):
    credential: str


class PasswordLoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str = ""
    email: str = ""


def user_from_record(record: dict, sub: str) -> User:
    return User(
        email=record["email"],
        name=record["name"],
        picture=record.get("picture"),
        sub=sub,
    )


def verify_google_token(token: str) -> User:
    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google Client ID is not configured on the server.",
        )

    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            settings.google_client_id,
        )
        return User(
            email=idinfo["email"],
            name=idinfo.get("name", idinfo["email"]),
            picture=idinfo.get("picture"),
            sub=idinfo["sub"],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credential.",
        ) from exc


def create_access_token(user: User) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": user.sub,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    settings = get_settings()
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return User(
            sub=payload["sub"],
            email=payload["email"],
            name=payload["name"],
            picture=payload.get("picture"),
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please sign in again.",
        ) from exc
