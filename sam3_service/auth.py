import os
import hashlib
import json
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, Header, HTTPException

from .config import API_KEYS_FILE


class ApiKeyManager:
    """Manage API keys in a local JSON file using hashed storage."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.RLock()
        self._store = self._load_store()

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _load_store(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return {"version": 1, "keys": []}

        try:
            with self.file_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if "keys" not in data or not isinstance(data["keys"], list):
                return {"version": 1, "keys": []}
            return data
        except Exception:
            return {"version": 1, "keys": []}

    def _save_store(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(self._store, fp, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.file_path)

    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": record["id"],
            "name": record["name"],
            "role": record["role"],
            "enabled": record["enabled"],
            "created_at": record["created_at"],
            "expires_at": record.get("expires_at"),
            "key_prefix": record.get("key_prefix"),
        }

    def has_admin(self) -> bool:
        with self._lock:
            return any(k.get("role") == "admin" and k.get("enabled", True) for k in self._store["keys"])

    def upsert_admin_key(self, raw_key: str, name: str = "env-admin") -> None:
        key_hash = self._hash_key(raw_key)
        with self._lock:
            existing = next((k for k in self._store["keys"] if k["key_hash"] == key_hash), None)
            if existing:
                existing["enabled"] = True
                existing["role"] = "admin"
                existing["name"] = name
            else:
                record = {
                    "id": f"key_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "role": "admin",
                    "key_hash": key_hash,
                    "key_prefix": f"{raw_key[:8]}...{raw_key[-4:]}",
                    "enabled": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": None,
                }
                self._store["keys"].append(record)
            self._save_store()

    def create_key(
        self,
        name: str,
        role: Literal["client", "admin"] = "client",
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw_key = f"sam3_{secrets.token_urlsafe(32)}"
        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()

        record = {
            "id": f"key_{uuid.uuid4().hex[:12]}",
            "name": name,
            "role": role,
            "key_hash": self._hash_key(raw_key),
            "key_prefix": f"{raw_key[:8]}...{raw_key[-4:]}",
            "enabled": True,
            "created_at": now.isoformat(),
            "expires_at": expires_at,
        }

        with self._lock:
            self._store["keys"].append(record)
            self._save_store()

        return {
            **self._sanitize_record(record),
            "api_key": raw_key,
        }

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._sanitize_record(record) for record in self._store["keys"]]

    def delete_key(self, key_id: str, protect_last_admin: bool = True) -> bool:
        with self._lock:
            index = next((idx for idx, item in enumerate(self._store["keys"]) if item["id"] == key_id), None)
            if index is None:
                return False

            record = self._store["keys"][index]
            if protect_last_admin and record.get("role") == "admin" and record.get("enabled", True):
                other_enabled_admin = any(
                    item.get("id") != key_id and item.get("role") == "admin" and item.get("enabled", True)
                    for item in self._store["keys"]
                )
                if not other_enabled_admin:
                    raise ValueError("Cannot delete the last enabled admin key")

            self._store["keys"].pop(index)
            self._save_store()
            return True

    def validate_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        key_hash = self._hash_key(raw_key)

        with self._lock:
            record = next(
                (
                    item
                    for item in self._store["keys"]
                    if item.get("enabled", True) and item.get("key_hash") == key_hash
                ),
                None,
            )

        if not record:
            return None

        expires_at = record.get("expires_at")
        if expires_at:
            try:
                expires_at_dt = datetime.fromisoformat(expires_at)
                if datetime.now(timezone.utc) > expires_at_dt:
                    return None
            except Exception:
                return None

        return self._sanitize_record(record)


api_key_manager = ApiKeyManager(API_KEYS_FILE)


def extract_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()

    if authorization:
        auth_value = authorization.strip()
        if auth_value.lower().startswith("bearer "):
            return auth_value[7:].strip()

    return None


async def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    raw_key = extract_api_key(authorization, x_api_key)
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Use Authorization: Bearer <api_key> or X-API-Key header.",
        )

    metadata = api_key_manager.validate_key(raw_key)
    if not metadata:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    return metadata


async def require_admin_key(key_metadata: Dict[str, Any] = Depends(require_api_key)) -> Dict[str, Any]:
    if key_metadata.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin API key required")
    return key_metadata
