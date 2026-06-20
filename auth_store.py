"""Small local authentication store used by the demo Streamlit dashboard.

This module is intentionally dependency-free. Passwords are stored as salted
PBKDF2 hashes, and legacy plaintext records are upgraded after a successful
login.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app_config import LEGACY_USERS_FILE, USERS_FILE


PBKDF2_ITERATIONS = 260_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_users(path: Path = USERS_FILE) -> Dict[str, Dict[str, Any]]:
    source_path = path
    if path == USERS_FILE and not path.exists() and LEGACY_USERS_FILE.exists():
        source_path = LEGACY_USERS_FILE
    if not source_path.exists():
        return {}

    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the user database: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("The user database must contain a JSON object.")
    if source_path != path:
        for record in data.values():
            if isinstance(record, dict) and isinstance(record.get("password"), str):
                password = record.pop("password")
                record.update(_password_record(password))
        try:
            _save_users(data, path)
        except OSError as exc:
            raise ValueError(f"Could not migrate the user database: {exc}") from exc
    return data


def _save_users(users: Dict[str, Dict[str, Any]], path: Path = USERS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(users, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _hash_password(
    password: str,
    *,
    salt: Optional[bytes] = None,
    iterations: int = PBKDF2_ITERATIONS,
) -> Tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return salt.hex(), digest.hex()


def _password_record(password: str) -> Dict[str, Any]:
    salt, digest = _hash_password(password)
    return {
        "password_hash": digest,
        "password_salt": salt,
        "password_iterations": PBKDF2_ITERATIONS,
    }


def _verify_hashed_password(password: str, record: Dict[str, Any]) -> bool:
    try:
        salt = bytes.fromhex(str(record["password_salt"]))
        iterations = int(record.get("password_iterations", PBKDF2_ITERATIONS))
        expected = str(record["password_hash"])
    except (KeyError, TypeError, ValueError):
        return False

    _, actual = _hash_password(password, salt=salt, iterations=iterations)
    return hmac.compare_digest(actual, expected)


def validate_signup(username: str, email: str, password: str) -> Optional[str]:
    if not USERNAME_PATTERN.fullmatch(username):
        return "Username must be 3-32 characters using letters, numbers, '.', '_' or '-'."
    if not EMAIL_PATTERN.fullmatch(email):
        return "Enter a valid email address."
    if len(password) < 8:
        return "Password must contain at least 8 characters."
    return None


def signup_user(
    username: str,
    email: str,
    password: str,
    *,
    path: Path = USERS_FILE,
) -> Tuple[bool, str]:
    username = username.strip()
    email = email.strip()

    validation_error = validate_signup(username, email, password)
    if validation_error:
        return False, validation_error

    try:
        users = _load_users(path)
    except ValueError as exc:
        return False, str(exc)

    if username in users:
        return False, "User already exists."

    users[username] = {
        "email": email,
        **_password_record(password),
    }
    try:
        _save_users(users, path)
    except OSError as exc:
        return False, f"Could not save the user database: {exc}"
    return True, "Signup successful. Please sign in."


def signin_user(username: str, password: str, *, path: Path = USERS_FILE) -> bool:
    username = username.strip()
    try:
        users = _load_users(path)
    except ValueError:
        return False

    record = users.get(username)
    if not isinstance(record, dict):
        return False

    if "password_hash" in record:
        return _verify_hashed_password(password, record)

    # Migrate records created by older versions of the dashboard.
    legacy_password = record.get("password")
    if isinstance(legacy_password, str) and hmac.compare_digest(legacy_password, password):
        record.pop("password", None)
        record.update(_password_record(password))
        try:
            _save_users(users, path)
        except OSError:
            return False
        return True

    return False
