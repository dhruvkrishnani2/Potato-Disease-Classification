import json
from pathlib import Path

import bcrypt

USERS_FILE = Path(__file__).parent / "users.json"


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def register_user(username: str, password: str, name: str = "", email: str = "") -> dict:
    users = _load_users()
    key = username.strip().lower()
    if not key or len(password) < 6:
        raise ValueError("Username required and password must be at least 6 characters.")
    if key in users:
        raise ValueError("Username already taken.")

    record = {
        "password_hash": _hash_password(password),
        "name": name.strip() or username.strip(),
        "email": email.strip() or f"{key}@spudguard.local",
        "sub": f"local:{key}",
        "picture": None,
    }
    users[key] = record
    _save_users(users)
    return record


def authenticate_user(username: str, password: str) -> dict | None:
    users = _load_users()
    key = username.strip().lower()
    record = users.get(key)
    if not record or not _verify_password(password, record["password_hash"]):
        return None
    return record
