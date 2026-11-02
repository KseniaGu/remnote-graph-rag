"""Durable local limits for explicitly requested evaluation calls."""

import asyncio
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend.utils.chat_limits import (
    EvaluationBudgetExceeded as EvaluationGuardExceeded,
)
from backend.utils.chat_limits import (
    EvaluationBudgetUnavailable as EvaluationGuardUnavailable,
)


class EvaluationGuard:
    def __init__(
        self, path: Path, *, quota: dict, max_attempts: int, max_web_attempts: int = 0
    ):
        self.path = Path(path).resolve()
        self.quota = quota
        self.max_attempts = max_attempts
        self.max_web_attempts = max_web_attempts
        self.allow_tavily = max_web_attempts > 0
        self.attempts = 0
        self.web_attempts = 0
        self._lock = None
        self._clients = []

    def _observation(self):
        try:
            if set(self.quota) != {
                "observed_at",
                "provider",
                "model",
                "remaining_requests",
                "demo_reserve",
                "daily_batch_limit",
            }:
                raise ValueError
            observed = datetime.fromisoformat(self.quota["observed_at"])
            if observed.tzinfo is None:
                raise ValueError
            observed = observed.astimezone(UTC)
            now = datetime.now(UTC)
            if observed.date() != now.date() or not timedelta(
                0
            ) <= now - observed <= timedelta(minutes=15):
                raise ValueError
            if (
                self.quota["provider"] != "openrouter"
                or not isinstance(self.quota["model"], str)
                or not self.quota["model"]
            ):
                raise ValueError
            for key in ("remaining_requests", "demo_reserve", "daily_batch_limit"):
                if type(self.quota[key]) is not int or self.quota[key] < 0:
                    raise ValueError
            if (
                type(self.max_attempts) is not int
                or self.max_attempts <= 0
                or type(self.max_web_attempts) is not int
                or self.max_web_attempts < 0
            ):
                raise ValueError
            return observed
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationGuardUnavailable(
                "invalid or stale quota observation"
            ) from exc

    def __enter__(self):
        observed = self._observation()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(".lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"version": "request-guard-v1", "days": {}}
            )
            if self.state.get("version") != "request-guard-v1" or not isinstance(
                self.state.get("days"), dict
            ):
                raise EvaluationGuardUnavailable("invalid evaluation guard ledger")
            self.day = observed.date().isoformat()
            day = self.state["days"].setdefault(
                self.day,
                {
                    "reservations": [],
                    "ceiling": self.quota["daily_batch_limit"],
                    "observations": {},
                },
            )
            if (
                not isinstance(day.get("reservations"), list)
                or type(day.get("ceiling")) is not int
                or not isinstance(day.get("observations"), dict)
            ):
                raise EvaluationGuardUnavailable("invalid evaluation guard day")
            timestamps = [datetime.fromisoformat(item) for item in day["reservations"]]
            if any(
                item.tzinfo is None or item.date().isoformat() != self.day
                for item in timestamps
            ):
                raise EvaluationGuardUnavailable("invalid evaluation reservations")
            identity = hashlib.sha256(
                json.dumps(
                    {
                        key: self.quota[key]
                        for key in ("observed_at", "provider", "remaining_requests")
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            baseline = day["observations"].setdefault(
                identity, sum(item <= observed for item in timestamps)
            )
            if type(baseline) is not int or not 0 <= baseline <= len(timestamps):
                raise EvaluationGuardUnavailable("invalid observation baseline")
            day["ceiling"] = min(
                day["ceiling"],
                self.quota["daily_batch_limit"],
                baseline
                + max(0, self.quota["remaining_requests"] - self.quota["demo_reserve"]),
            )
            self._persist()
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def register_clients(self, sync_client, async_client):
        self._clients.append((sync_client, async_client))

    async def _close_clients(self):
        while self._clients:
            sync_client, async_client = self._clients.pop()
            try:
                sync_client.close()
            finally:
                await async_client.aclose()

    def _unlock(self):
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __exit__(self, *_):
        try:
            if self._clients:
                asyncio.run(self._close_clients())
        finally:
            self._unlock()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *_):
        try:
            await self._close_clients()
        finally:
            self._unlock()

    def _persist(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False, encoding="utf-8"
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.state, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def reserve(self, kind: str, *, provider: str = "", model: str = ""):
        if self._lock is None:
            raise EvaluationGuardUnavailable("evaluation guard is not locked")
        observed = self._observation()
        if observed.date().isoformat() != self.day:
            raise EvaluationGuardUnavailable("quota day changed")
        if kind == "web_search":
            if not self.allow_tavily:
                raise EvaluationGuardUnavailable("web access not enabled")
            return
        if kind == "web_attempt":
            if self.web_attempts >= self.max_web_attempts:
                raise EvaluationGuardExceeded("web attempt limit reached")
            self.web_attempts += 1
            return
        if (
            kind not in {"llm", "judge"}
            or provider != self.quota["provider"]
            or model != self.quota["model"]
        ):
            raise EvaluationGuardUnavailable("provider or model differs from quota")
        day = self.state["days"][self.day]
        if (
            self.attempts >= self.max_attempts
            or len(day["reservations"]) >= day["ceiling"]
        ):
            raise EvaluationGuardExceeded("provider attempt limit reached")
        day["reservations"].append(datetime.now(UTC).isoformat())
        self._persist()
        self.attempts += 1

    def provenance(self):
        return {"provider_attempts": self.attempts, "web_attempts": self.web_attempts}
