"""Durable, evaluation-only reservations for the approved portfolio campaign."""

from __future__ import annotations

import contextvars
import difflib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_BUDGET_LEDGER = Path("data/evaluation/portfolio_budget/ledger.json")
DAILY_POLICY = {"llm_attempts_per_utc_day": 50, "demo_headroom_requests": 20}
CAPS = {
    "B0": {"llm": 33, "judge": 5, "web_search": 3, "web_attempt": 6},
    "final": {"llm": 59, "judge": 5, "web_search": 7, "web_attempt": 14},
    "overall": {"llm": 92, "judge": 10, "web_search": 10, "web_attempt": 20},
}


def campaign_ledger(checkout: Path, *, override: Path | None = None) -> Path:
    """Selects the original canonical ledger, including a restored snapshot binding."""
    git = checkout.resolve() / ".git"
    if git.is_file():
        text = git.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise EvaluationBudgetUnavailable("cannot locate shared campaign ledger")
        git = (checkout / text.removeprefix("gitdir: ")).resolve()
        common = git / "commondir"
        if not common.is_file():
            raise EvaluationBudgetUnavailable("cannot locate shared Git directory")
        git = (git / common.read_text(encoding="utf-8").strip()).resolve()
    if not git.is_dir() or git.name != ".git":
        raise EvaluationBudgetUnavailable("cannot locate shared campaign ledger")
    marker = git / "portfolio-restored-source"
    canonical = git.parent / DEFAULT_BUDGET_LEDGER
    if marker.exists():
        try:
            binding = json.loads(marker.read_text(encoding="utf-8"))["canonical_ledger"]
            if not isinstance(binding, str) or not Path(binding).is_absolute():
                raise ValueError
            canonical = Path(binding).resolve()
        except (ValueError, KeyError, TypeError) as exc:
            raise EvaluationBudgetUnavailable(
                "restored source lacks its verified canonical ledger binding; re-freeze required"
            ) from exc
        if override is None:
            raise EvaluationBudgetUnavailable(
                "restored source requires the original canonical --budget-ledger"
            )
    canonical = canonical.resolve()
    if override is not None:
        if not override.is_absolute() or not override.is_file():
            raise ValueError(
                "--budget-ledger must name the existing absolute canonical campaign ledger"
            )
        if override.resolve() != canonical:
            raise EvaluationBudgetUnavailable(
                "--budget-ledger differs from the original canonical ledger binding"
            )
    return canonical


class EvaluationBudgetExceeded(RuntimeError):
    """An operational evaluation stop; never a retrieval outcome."""

    reason = "evaluation_budget_exhausted"


class EvaluationBudgetUnavailable(RuntimeError):
    """The runner cannot establish a safe, durable attempt boundary."""

    reason = "evaluation_budget_unavailable"


_CURRENT_BUDGET: contextvars.ContextVar[EvaluationBudget | None] = (
    contextvars.ContextVar("evaluation_budget", default=None)
)


def current_evaluation_budget() -> EvaluationBudget | None:
    return _CURRENT_BUDGET.get()


@contextmanager
def bind_evaluation_budget(budget: EvaluationBudget):
    token = _CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _CURRENT_BUDGET.reset(token)


def _identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_.:/-]{1,240}", value
    ):
        raise EvaluationBudgetUnavailable("invalid evaluation identity")
    return value


class EvaluationBudget:
    """Holds the campaign lock through sequential dispatch and artifact creation.

    Reservations are synced before dispatch and are never refunded. An interrupted
    job stays uncertain, requiring reconciliation before it can be run again.
    Provider quota observations are supplied by the coordinator, never inferred.
    """

    def __init__(
        self,
        path: Path,
        *,
        stage: str,
        run_id: str,
        quota: dict[str, Any],
        configuration: dict[str, Any] | None = None,
    ):
        if stage not in {"B0", "final"}:
            raise EvaluationBudgetUnavailable("budget stage must be B0 or final")
        self.path = Path(path).resolve()
        self.stage = stage
        self.run_id = _identity(run_id)
        self.quota = dict(quota) if isinstance(quota, dict) else quota
        self.configuration = configuration
        self._lock = None
        self._state: dict[str, Any] = {}
        self.job_id = ""
        self.fixed_judge_pilot = False
        self.allow_tavily = False
        self._quota_verified = False
        self._quota_day = ""
        self._observation_id = ""
        self._publication_job_ids: list[str] = []
        self.require_registered_observation = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(".lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._lock = None
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        try:
            if self.path.exists():
                self._state = json.loads(self.path.read_text(encoding="utf-8"))
                self._validate_state()
            else:
                self._state = {
                    "version": 1,
                    "ledger_binding": str(self.path),
                    "daily_policy": DAILY_POLICY,
                    "campaign": "portfolio-core",
                    "caps": CAPS,
                    "reservations": [],
                    "jobs": {},
                    "days": {},
                }
            if self.require_registered_observation:
                self._verify_registered_observation()
            self._configure_day()
            if self.configuration is not None:
                bindings = self._state.setdefault("stage_configurations", {})
                previous = bindings.get(self.stage)
                if previous is not None and previous != self.configuration:
                    raise EvaluationBudgetUnavailable(
                        "effective evaluation configuration changed within this stage"
                    )
                if previous is None and any(
                    item["stage"] == self.stage for item in self._state["reservations"]
                ):
                    raise EvaluationBudgetUnavailable(
                        "spent stage is missing its effective configuration binding"
                    )
                bindings[self.stage] = self.configuration
            self._persist()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def _validate_state(self, *, daily_policy=None, caps=None):
        daily_policy = DAILY_POLICY if daily_policy is None else daily_policy
        caps = CAPS if caps is None else caps
        state = self._state
        if not isinstance(state, dict) or state.get("ledger_binding") != str(self.path):
            raise EvaluationBudgetUnavailable(
                "ledger binding is missing or differs from its original canonical path"
            )
        if state.get("daily_policy") != daily_policy:
            raise EvaluationBudgetUnavailable(
                "ledger lacks the approved immutable daily policy; reconciliation required"
            )
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("campaign") != "portfolio-core"
            or state.get("caps") != caps
            or not isinstance(state.get("reservations"), list)
            or not isinstance(state.get("jobs"), dict)
            or not isinstance(state.get("days"), dict)
        ):
            raise EvaluationBudgetUnavailable("invalid evaluation ledger")
        day_counts: dict[str, int] = {}
        request_ids = set()
        for event in state["reservations"]:
            if (
                not isinstance(event, dict)
                or set(event)
                != {"kind", "stage", "run_id", "job_id", "request_id", "reserved_at"}
                or event.get("kind") not in CAPS["overall"]
                or event.get("stage") not in {"B0", "final"}
                or not isinstance(event.get("request_id"), str)
                or event["request_id"] in request_ids
            ):
                raise EvaluationBudgetUnavailable("invalid evaluation reservation")
            request_ids.add(event["request_id"])
            try:
                reserved = datetime.fromisoformat(event["reserved_at"])
                if reserved.tzinfo is None:
                    raise ValueError
                day_key = reserved.astimezone(UTC).date().isoformat()
            except (ValueError, TypeError) as exc:
                raise EvaluationBudgetUnavailable(
                    "invalid reservation timestamp"
                ) from exc
            if event["kind"] in {"llm", "judge"}:
                day_counts[day_key] = day_counts.get(day_key, 0) + 1
        for key, day in state["days"].items():
            if (
                not isinstance(day, dict)
                or any(
                    type(day.get(field)) is not int or day[field] < 0
                    for field in ("used", "batch_limit", "demo_reserve")
                )
                or day["used"] != day_counts.pop(key, 0)
                or not isinstance(day.get("observations"), dict)
                or any(
                    type(count) is not int or not 0 <= count <= day["used"]
                    for count in day["observations"].values()
                )
            ):
                raise EvaluationBudgetUnavailable("invalid daily evaluation counters")
        if day_counts:
            raise EvaluationBudgetUnavailable("reservation daily counters are missing")
        pilot_identity = state.get("pilot_serving_identity")
        if pilot_identity is not None and (
            not isinstance(pilot_identity, dict)
            or set(pilot_identity) != {"model", "provider"}
            or any(
                not isinstance(value, str) or not value
                for value in pilot_identity.values()
            )
        ):
            raise EvaluationBudgetUnavailable("invalid pilot serving identity")
        retry = state.get("pilot_retry")
        if retry is not None and (
            not isinstance(retry, dict)
            or set(retry) != {"job_id", "failed_attempt"}
            or retry.get("job_id") not in state["jobs"]
            or not isinstance(retry.get("failed_attempt"), dict)
        ):
            raise EvaluationBudgetUnavailable("invalid pilot retry state")
        for job in state["jobs"].values():
            if not isinstance(job, dict) or job.get("status") not in {
                "started",
                "completed",
            }:
                raise EvaluationBudgetUnavailable("invalid evaluation job")
            if job["status"] == "completed" and not all(
                isinstance(job.get(key), str) for key in ("artifact", "sha256")
            ):
                raise EvaluationBudgetUnavailable("invalid completed job artifact")

    def _verify_registered_observation(self):
        """Require the exact capture-bound baseline before configuring a day."""
        try:
            if self.stage != "final" or not isinstance(self.quota, dict):
                raise ValueError
            observed = datetime.fromisoformat(self.quota["observed_at"])
            if observed.tzinfo is None:
                raise ValueError
            day_key = observed.astimezone(UTC).date().isoformat()
            identity = hashlib.sha256(
                json.dumps(
                    {
                        key: self.quota[key]
                        for key in ("observed_at", "provider", "remaining_requests")
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            history = self._state.get("quota_observation_registrations", [])
            amendment_history = self._state.get("policy_amendments", [])
            if (
                not isinstance(history, list)
                or not isinstance(amendment_history, list)
                or any(
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("source_proof"), dict)
                    for entry in history
                )
                or any(not isinstance(entry, dict) for entry in amendment_history)
            ):
                raise ValueError
            registrations = [
                entry
                for entry in history
                if entry.get("day") == day_key
                and entry["source_proof"].get("observation_id") == identity
            ]
            amendments = [
                entry
                for entry in amendment_history
                if entry.get("amendment") == "daily-policy-50-20261030"
            ]
            if len(registrations) != 1 or len(amendments) != 1:
                raise ValueError
            proof = registrations[0]["source_proof"]
            binding = amendments[0]["day_change"]["source_proof"]
            if (
                proof["basis"] != "prospective-authority"
                or binding["basis"] != "prospective-authority"
                or proof["proof"]["quota"] != self.quota
                or proof["expected_account_id"] != binding["expected_account_id"]
                or proof["expected_key_id"] != binding["expected_key_id"]
                or self._state["days"][day_key]["observations"][identity]
                != proof["baseline"]
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationBudgetUnavailable(
                "quota observation is not registered with the approved account"
            ) from exc

    def _configure_day(self):
        required = {
            "observed_at",
            "provider",
            "model",
            "remaining_requests",
            "demo_reserve",
            "daily_batch_limit",
        }
        if not isinstance(self.quota, dict) or set(self.quota) != required:
            raise EvaluationBudgetUnavailable(
                "quota metadata must contain only the documented fields"
            )
        try:
            observed = datetime.fromisoformat(self.quota["observed_at"])
            if observed.tzinfo is None:
                raise ValueError
            observed = observed.astimezone(UTC)
            if observed.date() != datetime.now(UTC).date() or not timedelta(
                0
            ) <= datetime.now(UTC) - observed <= timedelta(minutes=15):
                raise ValueError
            for key in ("remaining_requests", "demo_reserve", "daily_batch_limit"):
                if type(self.quota[key]) is not int or self.quota[key] < 0:
                    raise ValueError
            if self.quota["provider"] != "openrouter":
                raise ValueError
            _identity(self.quota["model"])
        except (TypeError, ValueError, EvaluationBudgetUnavailable) as exc:
            raise EvaluationBudgetUnavailable(
                "missing, stale or unsupported provider quota observation"
            ) from exc
        self._quota_verified = True
        self._quota_day = observed.date().isoformat()
        observation = {
            key: self.quota[key]
            for key in ("observed_at", "provider", "remaining_requests")
        }
        self._observation_id = hashlib.sha256(
            json.dumps(observation, sort_keys=True).encode()
        ).hexdigest()
        days = self._state["days"]
        day = days.setdefault(
            self._quota_day,
            {
                "used": 0,
                "batch_limit": min(
                    DAILY_POLICY["llm_attempts_per_utc_day"],
                    self.quota["daily_batch_limit"],
                ),
                "demo_reserve": max(
                    DAILY_POLICY["demo_headroom_requests"], self.quota["demo_reserve"]
                ),
                "observations": {},
            },
        )
        # A changed observation cannot expand today's batch or release demo capacity.
        day["batch_limit"] = min(
            DAILY_POLICY["llm_attempts_per_utc_day"],
            day["batch_limit"],
            self.quota["daily_batch_limit"],
        )
        day["demo_reserve"] = max(
            DAILY_POLICY["demo_headroom_requests"],
            day["demo_reserve"],
            self.quota["demo_reserve"],
        )
        day["observations"].setdefault(self._observation_id, day["used"])

    def require_final_runtime_admission(self):
        """Require a six-attempt turn to fit while the campaign lock is held."""
        if self._lock is None or not self._quota_verified or self.stage != "final":
            raise EvaluationBudgetUnavailable(
                "final runtime admission requires a locked quota session"
            )
        observed = datetime.fromisoformat(self.quota["observed_at"]).astimezone(UTC)
        if observed.date() != datetime.now(UTC).date() or not timedelta(
            0
        ) <= datetime.now(UTC) - observed <= timedelta(minutes=15):
            raise EvaluationBudgetUnavailable(
                "quota observation expired; resume with fresh metadata"
            )
        day = self._state["days"][self._quota_day]
        baseline = day["observations"][self._observation_id]
        events = self._state["reservations"]
        runtime = sum(
            event["stage"] == "final" and event["kind"] == "llm" for event in events
        )
        final = sum(
            event["stage"] == "final" and event["kind"] in {"llm", "judge"}
            for event in events
        )
        overall = sum(event["kind"] in {"llm", "judge"} for event in events)
        if (
            runtime + 6 > 54
            or final + 6 > CAPS["final"]["llm"]
            or overall + 6 > CAPS["overall"]["llm"]
            or day["used"] + 6 > day["batch_limit"]
            or day["used"] - baseline + 6
            > self.quota["remaining_requests"] - day["demo_reserve"]
        ):
            raise EvaluationBudgetExceeded(
                "six-attempt final runtime admission exhausted"
            )

    def _persist(
        self,
        *,
        required_utc_day: str | None = None,
        required_fresh_since: datetime | None = None,
    ):
        if self._lock is None:
            raise EvaluationBudgetUnavailable("evaluation ledger session is not locked")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False, encoding="utf-8"
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(self._state, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            if (
                required_utc_day is not None
                and datetime.now(UTC).date().isoformat() != required_utc_day
            ):
                raise EvaluationBudgetUnavailable(
                    "UTC day changed before policy amendment persistence"
                )
            if required_fresh_since is not None and not timedelta(0) <= datetime.now(
                UTC
            ) - required_fresh_since <= timedelta(minutes=15):
                raise EvaluationBudgetUnavailable(
                    "prospective capacity evidence became stale"
                )
            os.replace(temporary_path, self.path)
            directory = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def is_registered_fixed_judge_quota(self) -> bool:
        """Identify a registered judge-model observation without authorizing it."""
        return self.quota.get("model") == "xingchen-agi/xing4.0-29b-a4b:free" and any(
            isinstance(entry, dict)
            and isinstance(entry.get("source_proof"), dict)
            and entry["source_proof"].get("proof", {}).get("quota") == self.quota
            for entry in self._state.get("quota_observation_registrations", [])
        )

    def reserve(self, kind: str, *, provider: str = "", model: str = "") -> str:
        """Reserves one attempt or logical search before calling its transport."""
        if self._lock is None or kind not in CAPS["overall"]:
            raise EvaluationBudgetUnavailable("invalid or unlocked reservation")
        observed = datetime.fromisoformat(self.quota["observed_at"]).astimezone(UTC)
        if datetime.now(UTC).date().isoformat() != self._quota_day or not timedelta(
            0
        ) <= datetime.now(UTC) - observed <= timedelta(minutes=15):
            raise EvaluationBudgetUnavailable(
                "quota observation expired; resume with fresh metadata"
            )
        if self.fixed_judge_pilot and kind == "judge":
            dispatched = {
                item["job_id"]
                for item in self._state["reservations"]
                if item["kind"] == "judge"
                and self._state["jobs"].get(item["job_id"], {}).get("fixed_judge_pilot")
                is True
            }
            if self.job_id not in dispatched and len(dispatched) >= 4:
                raise EvaluationBudgetExceeded(
                    "fixed judge pilot assessment ceiling exhausted"
                )
        llm_attempt = kind in {"llm", "judge"}
        if (
            (
                self.require_registered_observation
                or self.is_registered_fixed_judge_quota()
            )
            and self.quota.get("model") == "xingchen-agi/xing4.0-29b-a4b:free"
            and (self.stage != "final" or kind != "judge" or not self.fixed_judge_pilot)
        ):
            raise EvaluationBudgetUnavailable(
                "fixed judge quota requires the approved pilot profile"
            )
        if llm_attempt and (
            provider != self.quota["provider"] or model != self.quota["model"]
        ):
            raise EvaluationBudgetUnavailable(
                "provider/model differs from quota observation"
            )
        for scope in (self.stage, "overall"):
            events = [
                item
                for item in self._state["reservations"]
                if scope == "overall" or item["stage"] == scope
            ]
            for counter in ("llm", "judge") if kind == "judge" else (kind,):
                count = sum(
                    item["kind"] in {"llm", "judge"}
                    if counter == "llm"
                    else item["kind"] == counter
                    for item in events
                )
                if count >= CAPS[scope][counter]:
                    raise EvaluationBudgetExceeded(
                        f"{scope} {counter} attempt ceiling exhausted"
                    )
        day = self._state["days"][self._quota_day]
        if llm_attempt:
            if day["used"] >= day["batch_limit"]:
                raise EvaluationBudgetExceeded("daily evaluation batch exhausted")
            since_observation = day["used"] - day["observations"][self._observation_id]
            if (
                since_observation
                >= self.quota["remaining_requests"] - day["demo_reserve"]
            ):
                raise EvaluationBudgetExceeded("provider quota demo reserve reached")
        request_id = str(uuid.uuid4())
        self._state["reservations"].append(
            {
                "kind": kind,
                "stage": self.stage,
                "run_id": self.run_id,
                "job_id": self.job_id,
                "request_id": request_id,
                "reserved_at": datetime.now(UTC).isoformat(),
            }
        )
        if llm_attempt:
            day["used"] += 1
        self._persist()
        return request_id

    def completed_judge_artifacts(self) -> list[tuple[str, Path]]:
        """Return verified completed judge artifacts for campaign reconciliation."""
        artifacts = []
        for job_id, job in self._state["jobs"].items():
            if ":judge:semantic-v2:" not in job_id:
                continue
            if job["status"] != "completed":
                if "fixed_judge_pilot" not in job:
                    raise EvaluationBudgetUnavailable(
                        "legacy judge job has an uncertain profile"
                    )
                continue
            if job.get("fixed_judge_pilot") is False:
                continue
            path = Path(job["artifact"])
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != job["sha256"]
            ):
                raise EvaluationBudgetUnavailable(
                    "completed judge artifact is missing or changed"
                )
            artifacts.append((job_id, path))
        return artifacts

    def pilot_job_profile(self, job_id: str, dimension: str) -> bool:
        """Recover an unmarked legacy profile only from its verified publication."""
        job = self._state["jobs"][job_id]
        marked = job.get("fixed_judge_pilot")
        if type(marked) is bool:
            return marked
        location = job.get("publication_manifest")
        checksum = job.get("publication_sha256")
        if not isinstance(location, str) or not isinstance(checksum, str):
            raise EvaluationBudgetUnavailable(
                "legacy judge profile requires reconciliation"
            )
        path = Path(location)
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != checksum
        ):
            raise EvaluationBudgetUnavailable(
                "legacy judge publication is missing or changed"
            )
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            invocation = manifest["invocation"]
            config = manifest["configuration_snapshot"]["executor_configuration"]
            published = config["evaluation_budget"]["publication_job_ids"]
            fixed = invocation["fixed_judge_pilot"]
            if type(fixed) is not bool or job_id not in published:
                raise ValueError
            if fixed:
                if not (
                    config["judge_profile"] == "fixed-pilot-v1"
                    and config["judge_structured_output_method"] == "json_schema"
                    and config["judge_strict_schema"] is True
                    and config["judge_include_raw"] is True
                    and config["judge_temperature"] == 0
                    and config["judge_reasoning"] == {"effort": "none"}
                    and config["judge_provider_policy"]
                    == {
                        "only": ["Phala"],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                    and config["judge_output_token_limits"][dimension]
                    == {"claim_faithfulness": 8192, "graph_usefulness": 512}[dimension]
                    and config["judge_rubric_versions"][dimension]
                    == "runtime-semantic-v2"
                ):
                    raise ValueError
            elif config["judge_profile"] != "standard-v2":
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationBudgetUnavailable(
                "legacy judge profile is ambiguous"
            ) from exc
        job["fixed_judge_pilot"] = fixed
        self._persist()
        return fixed

    def match_pilot_identity(self, model: str, provider: str):
        """Pin and compare the actual serving identity while holding the ledger lock."""
        if self._lock is None or not model or not provider:
            raise EvaluationBudgetUnavailable("pilot serving identity is missing")
        identity = {"model": model, "provider": provider}
        previous = self._state.get("pilot_serving_identity")
        if previous is not None and previous != identity:
            raise EvaluationBudgetUnavailable(
                "pilot serving identity differs from campaign"
            )
        if previous is None:
            self._state["pilot_serving_identity"] = identity
            self._persist()

    def pilot_retry_identity_matches(self, model: str, provider: str) -> bool:
        """Reject any observed retry identity that conflicts with the campaign pin."""
        previous = self._state.get("pilot_serving_identity")
        retry = self._state.get("pilot_retry")
        failed = retry["failed_attempt"] if retry is not None else {}
        failed_model = failed.get("observed_model", "")
        failed_provider = failed.get("observed_provider", "")
        if not isinstance(failed_model, str) or not isinstance(failed_provider, str):
            return False
        return (
            (previous is None or not model or model == previous["model"])
            and (previous is None or not provider or provider == previous["provider"])
            and (not model or not failed_model or model == failed_model)
            and (not provider or not failed_provider or provider == failed_provider)
        )

    def claim_pilot_retry(self, result: dict[str, Any]) -> bool:
        """Durably spend the shared retry on the currently claimed job."""
        if self._lock is None or not self.job_id:
            raise EvaluationBudgetUnavailable("pilot retry requires a claimed job")
        if self._state.get("pilot_retry") is not None:
            return False
        self._state["pilot_retry"] = {"job_id": self.job_id, "failed_attempt": result}
        self._persist()
        return True

    def begin_job(
        self,
        identity: str,
        *,
        contract_digest: str = "",
        require_setup_recovery: bool = False,
    ) -> Path | None:
        """Returns a verified saved artifact, or claims a new case/repetition job."""
        self.job_id = f"{self.stage}:{_identity(identity)}"
        previous = self._state["jobs"].get(self.job_id)
        if previous is not None:
            if previous.get("contract_digest", "") != contract_digest:
                raise EvaluationBudgetUnavailable(
                    "job contract/configuration differs from saved manifest"
                )
            if self.fixed_judge_pilot and not previous.get("fixed_judge_pilot"):
                previous["fixed_judge_pilot"] = True
                self._persist()
            if previous["status"] != "completed":
                raise EvaluationBudgetUnavailable(
                    "job is uncertain; reconcile its manifest before any redispatch"
                )
            artifact = Path(previous["artifact"])
            if (
                not artifact.is_file()
                or hashlib.sha256(artifact.read_bytes()).hexdigest()
                != previous["sha256"]
            ):
                raise EvaluationBudgetUnavailable(
                    "completed job artifact is missing or changed"
                )
            return artifact
        recovery = None
        recoveries = self._state.get("fixed_judge_recoveries", [])
        has_matching_recovery = isinstance(recoveries, list) and any(
            isinstance(entry, dict) and entry.get("new_job_id") == self.job_id
            for entry in recoveries
        )
        if (
            require_setup_recovery and _FIXED_SETUP_JOB in self._state["jobs"]
        ) or has_matching_recovery:
            entries = self._state.get("fixed_judge_recoveries", [])
            if (
                not self.fixed_judge_pilot
                or not isinstance(entries, list)
                or len(entries) != 1
                or entries[0].get("new_job_id") != self.job_id
                or entries[0].get("new_contract_digest") != contract_digest
                or entries[0].get("consumed") is not False
                or self._state["jobs"].get(_FIXED_SETUP_JOB)
                != entries[0].get("original_job")
                or any(
                    event["job_id"] in {_FIXED_SETUP_JOB, self.job_id}
                    for event in self._state["reservations"]
                )
            ):
                raise EvaluationBudgetUnavailable(
                    "fixed judge setup recovery is required or already consumed"
                )
            recovery = entries[0]
        self._state["jobs"][self.job_id] = {
            "status": "started",
            "run_id": self.run_id,
            "contract_digest": contract_digest,
        }
        if recovery is not None:
            recovery["consumed"] = True
            recovery["consumed_at"] = datetime.now(UTC).isoformat()
            self._state["jobs"][self.job_id]["recovered_from"] = _FIXED_SETUP_JOB
        if identity.startswith("judge:semantic-v2:"):
            self._state["jobs"][self.job_id]["fixed_judge_pilot"] = (
                self.fixed_judge_pilot
            )
        self._persist()
        return None

    def complete_job(self, identity: str, artifact: Path):
        job_id = f"{self.stage}:{_identity(identity)}"
        job = self._state["jobs"].get(job_id)
        if job is None or job["status"] != "started" or job["run_id"] != self.run_id:
            raise EvaluationBudgetUnavailable("job was not claimed by this runner")
        job.update(
            status="completed",
            artifact=str(artifact.resolve()),
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )
        self._persist()

    def is_published(self, identity: str) -> bool:
        """Reconciles the saved publication intent with its immutable run manifest."""
        job_id = f"{self.stage}:{_identity(identity)}"
        job = self._state["jobs"][job_id]
        location = job.get("publication_manifest")
        if location is None:
            return False
        path = Path(location)
        if not path.is_file():
            if "publication_sha256" in job:
                raise EvaluationBudgetUnavailable(
                    "published job manifest is missing; reconciliation required"
                )
            return False
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if job.get("publication_sha256") not in {None, checksum}:
            raise EvaluationBudgetUnavailable("published job manifest changed")
        manifest = json.loads(raw)
        if manifest.get("status") != "completed":
            return False
        published = (
            manifest.get("configuration_snapshot", {})
            .get("executor_configuration", {})
            .get("evaluation_budget", {})
            .get("publication_job_ids", [])
        )
        if job_id not in published:
            raise EvaluationBudgetUnavailable(
                "publication manifest does not contain the claimed job"
            )
        if "publication_sha256" not in job:
            job["publication_sha256"] = checksum
            self._persist()
        return True

    def prepare_publication(self, identities: list[str], manifest: Path) -> None:
        """Persists publication intent before creating any report contribution."""
        self._publication_job_ids = []
        for identity in identities:
            if self.is_published(identity):
                raise EvaluationBudgetUnavailable(
                    "job already has a report contribution"
                )
            job_id = f"{self.stage}:{_identity(identity)}"
            self._state["jobs"][job_id]["publication_manifest"] = str(
                manifest.resolve()
            )
            self._publication_job_ids.append(job_id)
        self._persist()

    def finish_publication(self, identities: list[str]) -> None:
        for identity in identities:
            if not self.is_published(identity):
                raise EvaluationBudgetUnavailable(
                    "publication did not produce a completed manifest"
                )

    def provenance(self) -> dict[str, Any]:
        return {
            "ledger": str(self.path.resolve()),
            "daily_policy": dict(DAILY_POLICY),
            "stage": self.stage,
            "run_id": self.run_id,
            "reservation_ids": [
                item["request_id"]
                for item in self._state.get("reservations", [])
                if item["run_id"] == self.run_id
            ],
            "job_ids": [
                key
                for key, value in self._state.get("jobs", {}).items()
                if value["run_id"] == self.run_id
            ],
            "quota_observation": dict(self.quota) if self._quota_verified else None,
            "quota_max_age_minutes": 15,
            "effective_configuration": self.configuration,
            "publication_job_ids": list(self._publication_job_ids),
        }


def _policy_source_proof(previous: Path, current: Path) -> dict[str, Any]:
    from backend.evaluation import evaluation_reporting as reporting

    snapshots = []
    inventories = []
    original_root = reporting.ROOT
    try:
        for root in (previous.resolve(), current.resolve()):
            reporting.ROOT = root
            snapshots.append(
                reporting.build_evaluation_snapshot({})["fingerprint_input"]
            )
            paths = {
                path.relative_to(root)
                for path in (root / "backend").rglob("*")
                if path.is_file()
                and reporting._is_identity_file(path.relative_to(root / "backend"))
            }
            paths.update(
                Path(relative)
                for relative in reporting.HASHED_RELATIVE_PATHS
                if (root / relative).is_file()
            )
            inventories.append(
                {
                    str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
                    for path in paths
                }
            )
    finally:
        reporting.ROOT = original_root
    changed = sorted(
        path
        for path in inventories[0].keys() | inventories[1].keys()
        if inventories[0].get(path) != inventories[1].get(path)
    )
    allowed_changes = ["backend/evaluation/portfolio_budget.py"]
    runner = "scripts/evaluate_runtime_workflow.py"
    if runner in changed:
        old_runner = (previous / runner).read_bytes()
        new_runner = (current / runner).read_bytes()
        old_help = (
            b'        help="Quota JSON no older than 15 minutes on this UTC day; '
            b"supplied limits can only tighten the fixed 12-attempt daily ceiling "
            b'and 20-request headroom.",'
        )
        if old_runner.count(old_help) != 1 or new_runner != old_runner.replace(
            old_help, old_help.replace(b"12-attempt", b"24-attempt"), 1
        ):
            raise EvaluationBudgetUnavailable(
                "unapproved runner source change in policy amendment"
            )
        allowed_changes.append(runner)
    if changed != allowed_changes:
        raise EvaluationBudgetUnavailable(
            "unapproved source change in policy amendment"
        )
    before, after = snapshots
    expected = before | {"tree_hashes": after["tree_hashes"]}
    if runner in changed:
        expected["file_hashes"] = before["file_hashes"] | {
            runner: after["file_hashes"][runner]
        }
    if expected != after or set(before["tree_hashes"]) != {"backend"}:
        raise EvaluationBudgetUnavailable(
            "policy amendment changed non-policy identity"
        )
    return {
        "previous_fingerprint_input": before,
        "fingerprint_input": after,
        "previous_fingerprint": reporting._hash_json(before),
        "fingerprint": reporting._hash_json(after),
        "changed_files": {
            path: {"before": inventories[0][path], "after": inventories[1][path]}
            for path in changed
        },
    }


def amend_b0_completion_budget(
    *,
    checkout: Path,
    ledger: Path,
    previous_source_root: Path,
    expected_ledger_sha256: str,
    configuration: dict[str, Any],
) -> bool:
    """Applies only the explicitly authorized 12/32/60 to 24/33/59 amendment.

    The exact reviewed prior ledger hash and accepted source must be supplied.
    Only the evaluator fingerprint may change in B0's effective configuration;
    scoped source comparison permits only this module and the exact reviewed
    daily-ceiling help correction in the runtime CLI.
    All prior state is retained, including uncertain jobs and quota observations.
    No normal evaluation session invokes this migration.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha256):
        raise EvaluationBudgetUnavailable("invalid expected ledger SHA256")
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    old_policy = {"llm_attempts_per_utc_day": 12, "demo_headroom_requests": 20}
    completion_policy = {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20}
    old_caps = {
        "B0": {"llm": 32, "judge": 5, "web_search": 3, "web_attempt": 6},
        "final": {"llm": 60, "judge": 5, "web_search": 7, "web_attempt": 14},
        "overall": {"llm": 92, "judge": 10, "web_search": 10, "web_attempt": 20},
    }
    guard = EvaluationBudget(path, stage="B0", run_id="policy-amendment", quota={})
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        raw = path.read_bytes()
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise EvaluationBudgetUnavailable("invalid evaluation ledger")
        guard._state = state
        history = state.get("policy_amendments", [])
        if history:
            guard._validate_state(daily_policy=completion_policy)
            if (
                not isinstance(history, list)
                or len(history) != 1
                or not isinstance(history[0], dict)
                or history[0].get("amendment") != "b0-completion-20261026"
                or history[0].get("previous_ledger_sha256") != expected_ledger_sha256
            ):
                raise EvaluationBudgetUnavailable("different policy amendment history")
            previous_state = history[0]["previous_state"]
        else:
            if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
                raise EvaluationBudgetUnavailable(
                    "reviewed prior ledger SHA256 differs"
                )
            guard._validate_state(daily_policy=old_policy, caps=old_caps)
            previous_state = json.loads(raw)
        bindings = previous_state.get("stage_configurations", {})
        if set(bindings) != {"B0"} or any(
            item["stage"] != "B0" for item in previous_state["reservations"]
        ):
            raise EvaluationBudgetUnavailable(
                "amendment requires the bound B0 campaign"
            )
        previous_configuration = bindings["B0"]
        if not isinstance(
            configuration, dict
        ) or configuration != previous_configuration | {
            "evaluation_fingerprint": configuration.get("evaluation_fingerprint")
        }:
            raise EvaluationBudgetUnavailable("policy amendment changed configuration")
        proof = _policy_source_proof(previous_source_root, checkout)
        if (
            previous_configuration.get("evaluation_fingerprint")
            != proof["previous_fingerprint"]
            or configuration["evaluation_fingerprint"] != proof["fingerprint"]
            or configuration.get("application_configuration")
            != proof["fingerprint_input"]["application_configuration"]
        ):
            raise EvaluationBudgetUnavailable(
                "unproven evaluation configuration identity"
            )
        if history:
            if history[0].get("source_proof") != proof or state.get(
                "stage_configurations"
            ) != {"B0": configuration}:
                raise EvaluationBudgetUnavailable(
                    "amended configuration or proof differs"
                )
            return False
        today = datetime.now(UTC).date().isoformat()
        for key, day in state["days"].items():
            if (
                day["batch_limit"] > 12
                or day["demo_reserve"] < 20
                or day["used"] > day["batch_limit"]
            ):
                raise EvaluationBudgetUnavailable(
                    "invalid prior daily policy enforcement"
                )
            if key == today and day["batch_limit"] == 12:
                day["batch_limit"] = 24
        state["daily_policy"] = dict(completion_policy)
        state["caps"] = json.loads(json.dumps(CAPS))
        state["stage_configurations"]["B0"] = configuration
        state["policy_amendments"] = [
            {
                "amendment": "b0-completion-20261026",
                "applied_at": datetime.now(UTC).isoformat(),
                "previous_ledger_sha256": expected_ledger_sha256,
                "previous_state": previous_state,
                "source_proof": proof,
            }
        ]
        guard._validate_state(daily_policy=completion_policy)
        guard._persist()
        return True
    finally:
        guard.__exit__(None, None, None)


def _daily_policy_source_proof(
    previous: Path,
    current: Path,
    previous_inventory: Path,
    expected_previous_inventory_sha256: str,
    current_inventory: Path,
    expected_current_inventory_sha256: str,
    approved_patch: Path,
    expected_patch_sha256: str,
    *,
    fixed_judge_successor: bool = False,
) -> dict[str, Any]:
    from backend.evaluation import evaluation_reporting as reporting

    inventories = [
        _reviewed_source_inventory(root, inventory, digest)
        for root, inventory, digest in (
            (previous, previous_inventory, expected_previous_inventory_sha256),
            (current, current_inventory, expected_current_inventory_sha256),
        )
    ]
    changed = sorted(
        name
        for name in inventories[0].keys() | inventories[1].keys()
        if inventories[0].get(name) != inventories[1].get(name)
    )
    allowed = {
        "backend/evaluation/portfolio_budget.py",
        "scripts/evaluate_runtime_workflow.py",
        "scripts/amend_portfolio_budget.py",
        "scripts/continue_portfolio_collection.py",
        "tests/test_portfolio_budget.py",
        "tests/test_portfolio_continuation.py",
        "README.md",
        "docs/runtime_evaluation.md",
        "docs/full_evaluation_report_runbook.md",
        "docs/project_overview.md",
    }
    if fixed_judge_successor:
        allowed.add("tests/test_pilot_judge.py")
    if (
        not changed
        or "backend/evaluation/portfolio_budget.py" not in changed
        or set(changed) - allowed
    ):
        raise EvaluationBudgetUnavailable(
            "unapproved source change in daily policy amendment"
        )
    patch = "".join(
        line
        for name in changed
        for line in difflib.unified_diff(
            (previous / name).read_text(encoding="utf-8").splitlines(keepends=True)
            if (previous / name).is_file()
            else [],
            (current / name).read_text(encoding="utf-8").splitlines(keepends=True)
            if (current / name).is_file()
            else [],
            fromfile="a/" + name,
            tofile="b/" + name,
        )
    ).encode("utf-8")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_patch_sha256)
        or hashlib.sha256(patch).hexdigest() != expected_patch_sha256
        or approved_patch.read_bytes() != patch
    ):
        raise EvaluationBudgetUnavailable("reviewed policy patch differs")
    snapshots = []
    original_root = reporting.ROOT
    try:
        for root in (previous.resolve(), current.resolve()):
            reporting.ROOT = root
            snapshots.append(
                reporting.build_evaluation_snapshot({})["fingerprint_input"]
            )
    finally:
        reporting.ROOT = original_root
    before, after = snapshots
    for key in before.keys() | after.keys():
        if key not in {"tree_hashes", "file_hashes"} and before.get(key) != after.get(
            key
        ):
            raise EvaluationBudgetUnavailable(
                "policy amendment changed non-policy identity"
            )
    for name in before["file_hashes"].keys() | after["file_hashes"].keys():
        if (
            before["file_hashes"].get(name) != after["file_hashes"].get(name)
            and name not in changed
        ):
            raise EvaluationBudgetUnavailable("unreviewed fingerprint file changed")
    return {
        "previous_inventory_sha256": expected_previous_inventory_sha256,
        "current_inventory_sha256": expected_current_inventory_sha256,
        "approved_patch_sha256": expected_patch_sha256,
        "changed_files": {
            name: {
                "before": inventories[0].get(name),
                "after": inventories[1].get(name),
            }
            for name in changed
        },
        "previous_fingerprint_input": before,
        "fingerprint_input": after,
        "previous_fingerprint": reporting._hash_json(before),
        "fingerprint": reporting._hash_json(after),
    }


def _reviewed_day_limit(
    previous_state: dict[str, Any],
    proof_path: Path,
    expected_sha256: str,
    *,
    require_today: bool,
) -> tuple[str, int, dict[str, Any]]:
    """Validate complete captured evidence for an independently bounded day."""
    raw = proof_path.read_bytes()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise EvaluationBudgetUnavailable("reviewed quota provenance SHA256 differs")
    proof = json.loads(raw)
    if not isinstance(proof, dict) or set(proof) != {"day", "observations"}:
        raise EvaluationBudgetUnavailable("invalid quota provenance")
    day_key = proof["day"]
    if (
        not isinstance(day_key, str)
        or day_key not in previous_state["days"]
        or (require_today and day_key != datetime.now(UTC).date().isoformat())
    ):
        raise EvaluationBudgetUnavailable("quota provenance is not for the current day")
    day = previous_state["days"][day_key]
    observations = proof["observations"]
    if (
        day["batch_limit"] != 24
        or not isinstance(observations, list)
        or not observations
    ):
        raise EvaluationBudgetUnavailable(
            "quota provenance cannot establish the old policy ceiling"
        )
    identities = set()
    ceilings = [50]
    evidence_hashes = []
    for item in observations:
        if not isinstance(item, dict) or set(item) != {
            "quota",
            "evidence_path",
            "evidence_sha256",
        }:
            raise EvaluationBudgetUnavailable("invalid captured quota evidence")
        q = item["quota"]
        required = {
            "observed_at",
            "provider",
            "model",
            "remaining_requests",
            "demo_reserve",
            "daily_batch_limit",
        }
        if (
            not isinstance(q, dict)
            or set(q) != required
            or q["provider"] != "openrouter"
            or q["daily_batch_limit"] != 24
            or type(q["daily_batch_limit"]) is not int
            or any(
                type(q[key]) is not int or q[key] < 0
                for key in ("remaining_requests", "demo_reserve")
            )
            or q["demo_reserve"] > day["demo_reserve"]
        ):
            raise EvaluationBudgetUnavailable("captured quota differs from stored day")
        try:
            observed = datetime.fromisoformat(q["observed_at"])
            if (
                observed.tzinfo is None
                or observed.astimezone(UTC).date().isoformat() != day_key
            ):
                raise ValueError
            _identity(q["model"])
            evidence_path = Path(item["evidence_path"])
            if not evidence_path.is_absolute() or evidence_path.is_symlink():
                raise ValueError
            evidence_bytes = evidence_path.read_bytes()
            if hashlib.sha256(evidence_bytes).hexdigest() != item["evidence_sha256"]:
                raise ValueError
            evidence = json.loads(evidence_bytes)
            quota_path = Path(evidence["quota_capture_path"])
            provider_path = Path(evidence["provider_capture_path"])
            if any(
                not path.is_absolute() or path.is_symlink()
                for path in (quota_path, provider_path)
            ):
                raise ValueError
            quota_bytes = quota_path.read_bytes()
            provider_bytes = provider_path.read_bytes()
            if (
                hashlib.sha256(quota_bytes).hexdigest()
                != evidence["quota_capture_sha256"]
                or hashlib.sha256(provider_bytes).hexdigest()
                != evidence["provider_capture_sha256"]
                or json.loads(quota_bytes) != q
            ):
                raise ValueError
            provider = json.loads(provider_bytes)["openrouter"]
            account = provider["quota"]["free_model_daily_requests"]
            if (
                provider["status"] != "success"
                or provider["http_status"] != 200
                or provider["method"] != "GET"
                or provider["time"] != q["observed_at"]
                or type(account["limit"]) is not int
                or type(account["remaining"]) is not int
                or account["remaining"] != q["remaining_requests"]
                or account["limit"] < evidence["independent_daily_limit"]
            ):
                raise ValueError
            captured = datetime.fromisoformat(evidence["captured_at"])
            if captured.tzinfo is None or not timedelta(0) <= captured.astimezone(
                UTC
            ) - observed.astimezone(UTC) <= timedelta(minutes=15):
                raise ValueError
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise EvaluationBudgetUnavailable(
                "unverifiable captured quota provenance"
            ) from exc
        identity = hashlib.sha256(
            json.dumps(
                {
                    key: q[key]
                    for key in ("observed_at", "provider", "remaining_requests")
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if (
            not isinstance(evidence, dict)
            or set(evidence)
            != {
                "observation_id",
                "quota_sha256",
                "daily_limit_origin",
                "independent_daily_limit",
                "captured_at",
                "quota_capture_path",
                "quota_capture_sha256",
                "provider_capture_path",
                "provider_capture_sha256",
                "no_other_independent_limit",
            }
            or evidence["observation_id"] != identity
            or evidence["quota_sha256"]
            != hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()
            or evidence["daily_limit_origin"] != "internal_policy"
            or evidence["no_other_independent_limit"] is not True
            or type(evidence["independent_daily_limit"]) is not int
            or evidence["independent_daily_limit"] <= 24
            or identity in identities
        ):
            raise EvaluationBudgetUnavailable("unproven policy-derived day ceiling")
        identities.add(identity)
        ceilings.append(evidence["independent_daily_limit"])
        evidence_hashes.append(item["evidence_sha256"])
    if identities != set(day["observations"]):
        raise EvaluationBudgetUnavailable(
            "quota provenance does not cover every observation"
        )
    return (
        day_key,
        min(ceilings),
        {
            "reviewed_proof_sha256": expected_sha256,
            "proof": proof,
            "captured_evidence_sha256": evidence_hashes,
        },
    )


def capture_prospective_boundary(
    *, checkout: Path, ledger: Path, expected_ledger_sha256: str, destination: Path
) -> dict[str, Any]:
    """Capture read-only quota-boundary evidence under the canonical lock.

    The snapshot is evidence only and must never be opened as a budget ledger.
    Provider and credential captures occur separately after this method returns.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha256):
        raise EvaluationBudgetUnavailable("invalid expected ledger SHA256")
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    destination = Path(destination)
    if not destination.is_absolute():
        raise EvaluationBudgetUnavailable("capture destination must be absolute")
    guard = EvaluationBudget(path, stage="final", run_id="quota-capture", quota={})
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        started_at = datetime.now(UTC).isoformat()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
            raise EvaluationBudgetUnavailable("reviewed prior ledger SHA256 differs")
        guard._state = json.loads(raw)
        policy = (
            guard._state.get("daily_policy") if isinstance(guard._state, dict) else None
        )
        if policy not in (
            {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
            DAILY_POLICY,
        ):
            raise EvaluationBudgetUnavailable("unsupported quota capture policy")
        guard._validate_state(daily_policy=policy)
        destination.mkdir(mode=0o700)
        snapshot = destination / "ledger_snapshot.json"
        boundary = destination / "boundary.json"
        descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        record = {
            "schema": "portfolio-prospective-capture-v1",
            "canonical_ledger": str(path),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "ledger_sha256": expected_ledger_sha256,
            "snapshot_path": str(snapshot),
        }
        descriptor = os.open(boundary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(record, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return record
    finally:
        guard.__exit__(None, None, None)


def _prospective_day_limit(
    previous_state: dict[str, Any],
    proof_path: Path,
    expected_sha256: str,
    expected_account_id: str,
    expected_key_id: str,
    *,
    require_fresh: bool,
    registration: bool = False,
) -> tuple[str, int, dict[str, Any]]:
    """Bind prospective authority to fresh, independently reviewed capacity."""
    raw = proof_path.read_bytes()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise EvaluationBudgetUnavailable("prospective proof SHA256 differs")
    proof = json.loads(raw)
    keys = {
        "authority",
        "capture_started_at",
        "boundary_path",
        "boundary_sha256",
        "capture_ledger_sha256",
        "capture_ledger_path",
        "quota",
        "quota_capture_path",
        "quota_capture_sha256",
        "binding_capture_path",
        "binding_capture_sha256",
        "account_capture_path",
        "account_capture_sha256",
        "route_capture_path",
        "route_capture_sha256",
    }
    if (
        not isinstance(proof, dict)
        or set(proof) != keys
        or proof["authority"] != "daily-50-2026-10-30"
    ):
        raise EvaluationBudgetUnavailable(
            "invalid prospective authority or capture binding"
        )
    try:
        boundary_path = Path(proof["boundary_path"])
        if not boundary_path.is_absolute() or boundary_path.is_symlink():
            raise ValueError
        boundary_bytes = boundary_path.read_bytes()
        if hashlib.sha256(boundary_bytes).hexdigest() != proof["boundary_sha256"]:
            raise ValueError
        boundary = json.loads(boundary_bytes)
        if boundary != {
            "schema": "portfolio-prospective-capture-v1",
            "canonical_ledger": previous_state["ledger_binding"],
            "started_at": proof["capture_started_at"],
            "completed_at": boundary["completed_at"],
            "ledger_sha256": proof["capture_ledger_sha256"],
            "snapshot_path": proof["capture_ledger_path"],
        }:
            raise ValueError
        captured_path = Path(proof["capture_ledger_path"])
        if not captured_path.is_absolute() or captured_path.is_symlink():
            raise ValueError
        captured_bytes = captured_path.read_bytes()
        if hashlib.sha256(captured_bytes).hexdigest() != proof["capture_ledger_sha256"]:
            raise ValueError
        captured_state = json.loads(captured_bytes)
        capture_guard = EvaluationBudget(
            Path(previous_state["ledger_binding"]),
            stage="final",
            run_id="capture-proof",
            quota={},
        )
        capture_guard._state = captured_state
        capture_guard._validate_state(
            daily_policy=DAILY_POLICY
            if registration
            else {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20}
        )
        if (
            captured_state["ledger_binding"] != previous_state["ledger_binding"]
            or captured_state["reservations"]
            != previous_state["reservations"][: len(captured_state["reservations"])]
            or captured_state.get("stage_configurations")
            != previous_state.get("stage_configurations")
            or captured_state.get("policy_amendments", [])
            != previous_state.get("policy_amendments", [])
            or captured_state.get("identity_transitions", [])
            != previous_state.get("identity_transitions", [])
            or any(
                previous_state["jobs"].get(key) != value
                for key, value in captured_state["jobs"].items()
            )
        ):
            raise ValueError
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise EvaluationBudgetUnavailable("invalid captured ledger evidence") from exc
    _identity(expected_account_id)
    _identity(expected_key_id)
    q = proof["quota"]
    judge_model = "xingchen-agi/xing4.0-29b-a4b:free"
    fixed_judge = registration and isinstance(q, dict) and q.get("model") == judge_model
    required = {
        "observed_at",
        "provider",
        "model",
        "remaining_requests",
        "demo_reserve",
        "daily_batch_limit",
    }
    if (
        not isinstance(q, dict)
        or set(q) != required
        or q["provider"] != "openrouter"
        or q["model"]
        not in (
            {"openrouter/free", judge_model} if registration else {"openrouter/free"}
        )
        or any(
            type(q[k]) is not int or q[k] < 0
            for k in ("remaining_requests", "demo_reserve", "daily_batch_limit")
        )
    ):
        raise EvaluationBudgetUnavailable("invalid prospective quota scope")
    try:
        observed = datetime.fromisoformat(q["observed_at"])
        started = datetime.fromisoformat(proof["capture_started_at"])
        if observed.tzinfo is None or started.tzinfo is None:
            raise ValueError
        observed, started = observed.astimezone(UTC), started.astimezone(UTC)
        completed = datetime.fromisoformat(boundary["completed_at"])
        if (
            completed.tzinfo is None
            or not started <= completed.astimezone(UTC) <= observed
        ):
            raise ValueError
        now = datetime.now(UTC)
        if (
            started.date() != observed.date()
            or not timedelta(0) <= observed - started <= timedelta(minutes=15)
            or (
                require_fresh
                and (
                    observed.date() != now.date()
                    or not timedelta(0) <= now - started <= timedelta(minutes=15)
                )
            )
        ):
            raise ValueError
        captures = []
        for name in ("account", "route", "quota", "binding"):
            path = Path(proof[f"{name}_capture_path"])
            if not path.is_absolute() or path.is_symlink():
                raise ValueError
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != proof[f"{name}_capture_sha256"]:
                raise ValueError
            captures.append(json.loads(data))
        account_capture, route, captured_quota, binding = captures
        if captured_quota != q:
            raise ValueError
        account = account_capture["openrouter"]
        account_quota = account["quota"]
        daily = account_quota["free_model_daily_requests"]
        if (
            account["method"] != "GET"
            or account["url"] != "https://openrouter.ai/api/v1/key"
            or type(account["http_status"]) is not int
            or account["http_status"] != 200
            or account["status"] != "success"
            or account["time"] != q["observed_at"]
            or account_quota["is_free_tier"] is not True
            or type(daily["limit"]) is not int
            or type(daily["remaining"]) is not int
            or type(daily["used"]) is not int
            or daily["remaining"] != q["remaining_requests"]
            or not 0 <= daily["used"] <= daily["limit"]
            or not 0 <= daily["remaining"] <= daily["limit"]
            or daily["used"] + daily["remaining"] > daily["limit"]
        ):
            raise ValueError
        route_time = datetime.fromisoformat(route["time"])
        if (
            route_time.tzinfo is None
            or route_time.astimezone(UTC).date() != observed.date()
            or not completed.astimezone(UTC) <= route_time.astimezone(UTC) <= now
            or route["method"] != "GET"
            or route["url"]
            != (
                f"https://openrouter.ai/api/v1/models/{judge_model}/endpoints"
                if fixed_judge
                else "https://openrouter.ai/api/v1/models"
            )
            or type(route["http_status"]) is not int
            or route["http_status"] != 200
            or route["status"] != "success"
            or not isinstance(route["data"], dict if fixed_judge else list)
        ):
            raise ValueError
        if fixed_judge:
            if route["data"].get("id") != judge_model:
                raise ValueError
            matching = [
                entry
                for entry in route["data"].get("endpoints", [])
                if isinstance(entry, dict)
                and entry.get("model_id") == judge_model
                and entry.get("provider_name") == "Phala"
                and entry.get("tag") == "phala"
            ]
            if (
                len(matching) != 1
                or type(matching[0].get("max_completion_tokens")) is not int
                or matching[0]["max_completion_tokens"] < 8192
                or not {
                    "response_format",
                    "structured_outputs",
                    "temperature",
                    "reasoning",
                    "max_tokens",
                }
                <= set(matching[0].get("supported_parameters", []))
            ):
                raise ValueError
            pricing = matching[0].get("pricing", {})
            if not isinstance(pricing, dict) or any(
                Decimal(str(value)) != 0
                for key, value in pricing.items()
                if key != "discount"
            ):
                raise ValueError
            roles = {"judge"}
        else:
            matching = [
                entry
                for entry in route["data"]
                if isinstance(entry, dict) and entry.get("id") == "openrouter/free"
            ]
            roles = {
                "orchestrator",
                "retriever",
                "analyst",
                "mentor",
                "researcher.with_tools",
                "researcher.structured",
            }
        if (
            len(matching) != 1
            or matching[0].get("pricing", {}).get("prompt") != "0"
            or matching[0].get("pricing", {}).get("completion") != "0"
        ):
            raise ValueError
        required_binding = {
            "account_id",
            "key_id",
            "observed_at",
            "provider",
            "model",
            "account_capture_sha256",
            "route_capture_sha256",
            "credential_check",
            "checked_at",
            "sut_role_credentials_match_queried_key",
            "external_daily_limit",
        }
        checked_at = datetime.fromisoformat(binding["checked_at"])
        if (
            checked_at.tzinfo is None
            or not completed.astimezone(UTC) <= checked_at.astimezone(UTC) <= now
        ):
            raise ValueError
        if (
            not isinstance(binding, dict)
            or set(binding) != required_binding
            or binding["account_id"] != expected_account_id
            or binding["key_id"] != expected_key_id
            or binding["observed_at"] != q["observed_at"]
            or binding["provider"] != "openrouter"
            or binding["model"] != q["model"]
            or binding["account_capture_sha256"] != proof["account_capture_sha256"]
            or binding["route_capture_sha256"] != proof["route_capture_sha256"]
            or binding["credential_check"] != "local-equality"
            or not isinstance(binding["sut_role_credentials_match_queried_key"], dict)
            or set(binding["sut_role_credentials_match_queried_key"]) != roles
            or any(
                value is not True
                for value in binding["sut_role_credentials_match_queried_key"].values()
            )
            or type(binding["external_daily_limit"]) is not int
            or not 0 <= binding["external_daily_limit"] <= daily["limit"]
        ):
            raise ValueError
    except (OSError, TypeError, ValueError, KeyError, InvalidOperation) as exc:
        raise EvaluationBudgetUnavailable(
            "unverifiable prospective capacity evidence"
        ) from exc
    day_key = observed.date().isoformat()
    day = previous_state["days"].get(day_key)
    if day is None and registration:
        day = {"used": 0, "batch_limit": 50, "demo_reserve": 20, "observations": {}}
    if not isinstance(day, dict) or (not registration and day["batch_limit"] != 24):
        raise EvaluationBudgetUnavailable(
            "prospective authority requires the approved current day"
        )
    observation_id = hashlib.sha256(
        json.dumps(
            {k: q[k] for k in ("observed_at", "provider", "remaining_requests")},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if observation_id in day["observations"]:
        raise EvaluationBudgetUnavailable(
            "prospective quota observation was already used"
        )
    captured_day = captured_state["days"].get(day_key)
    if captured_day is None and registration:
        captured_day = {
            "used": 0,
            "batch_limit": 50,
            "demo_reserve": 20,
            "observations": {},
        }
    if (
        not isinstance(captured_day, dict)
        or captured_day["batch_limit"] < day["batch_limit"]
        or any(
            day["observations"].get(k) != v
            for k, v in captured_day["observations"].items()
        )
        or day["demo_reserve"] < captured_day["demo_reserve"]
    ):
        raise EvaluationBudgetUnavailable("captured day state differs")
    baseline = 0
    for event in previous_state["reservations"]:
        if event["kind"] in {"llm", "judge"}:
            reserved = datetime.fromisoformat(event["reserved_at"]).astimezone(UTC)
            if reserved.date() == observed.date() and reserved < started:
                baseline += 1
    baseline = min(baseline, captured_day["used"])
    if baseline > day["used"]:
        raise EvaluationBudgetUnavailable("invalid prospective quota baseline")
    limit = min(
        50,
        q["daily_batch_limit"],
        binding["external_daily_limit"],
        day["batch_limit"] if registration else 50,
    )
    reserve = max(20, day["demo_reserve"], q["demo_reserve"])
    allowance = 1 if fixed_judge else 6
    if (
        (not registration and limit <= 24)
        or day["used"] + allowance > limit
        or day["used"] - baseline + allowance > q["remaining_requests"] - reserve
    ):
        raise EvaluationBudgetUnavailable(
            "prospective capacity cannot admit required attempts"
        )
    return (
        day_key,
        limit,
        {
            "basis": "prospective-authority",
            "authority": proof["authority"],
            "effective_at": observed.isoformat(),
            "capture_started_at": started.isoformat(),
            "reviewed_proof_sha256": expected_sha256,
            "proof": proof,
            "observation_id": observation_id,
            "baseline": baseline,
            "effective_reserve": reserve,
            "capture_ledger_sha256": proof["capture_ledger_sha256"],
            "expected_account_id": expected_account_id,
            "expected_key_id": expected_key_id,
        },
    )


def register_prospective_observation(
    *,
    checkout: Path,
    ledger: Path,
    expected_ledger_sha256: str,
    proof_path: Path,
    expected_proof_sha256: str,
) -> bool:
    """Explicitly register one fresh, capture-bound observation under the lock."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha256):
        raise EvaluationBudgetUnavailable("invalid expected ledger SHA256")
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    guard = EvaluationBudget(path, stage="final", run_id="quota-registration", quota={})
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
            raise EvaluationBudgetUnavailable("reviewed prior ledger SHA256 differs")
        state = json.loads(raw)
        guard._state = state
        guard._validate_state()
        history = state.get("policy_amendments", [])
        if not isinstance(history, list):
            raise EvaluationBudgetUnavailable("invalid policy amendment history")
        matches = [
            entry
            for entry in history
            if isinstance(entry, dict)
            and entry.get("amendment") == "daily-policy-50-20261030"
        ]
        if len(matches) != 1:
            raise EvaluationBudgetUnavailable("missing prospective account binding")
        try:
            binding = matches[0]["day_change"]["source_proof"]
            if binding["basis"] != "prospective-authority":
                raise ValueError
            account, key = binding["expected_account_id"], binding["expected_key_id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationBudgetUnavailable(
                "missing prospective account binding"
            ) from exc
        day_key, limit, proof = _prospective_day_limit(
            state,
            proof_path,
            expected_proof_sha256,
            account,
            key,
            require_fresh=True,
            registration=True,
        )
        if datetime.now(UTC).date().isoformat() != day_key:
            raise EvaluationBudgetUnavailable(
                "UTC day changed before quota registration"
            )
        day = state["days"].setdefault(
            day_key,
            {"used": 0, "batch_limit": 50, "demo_reserve": 20, "observations": {}},
        )
        day["batch_limit"] = min(day["batch_limit"], limit)
        day["demo_reserve"] = max(day["demo_reserve"], proof["effective_reserve"])
        day["observations"][proof["observation_id"]] = proof["baseline"]
        registrations = state.setdefault("quota_observation_registrations", [])
        if not isinstance(registrations, list):
            raise EvaluationBudgetUnavailable("invalid quota registration history")
        registrations.append(
            {
                "registered_at": datetime.now(UTC).isoformat(),
                "previous_ledger_sha256": expected_ledger_sha256,
                "day": day_key,
                "source_proof": proof,
            }
        )
        guard._validate_state()
        guard._persist(
            required_utc_day=day_key,
            required_fresh_since=datetime.fromisoformat(proof["capture_started_at"]),
        )
        return True
    finally:
        guard.__exit__(None, None, None)


def transition_final_evaluation_identity(
    *,
    checkout: Path,
    ledger: Path,
    previous_source_root: Path,
    expected_ledger_sha256: str,
    configuration: dict[str, Any],
    previous_inventory: Path,
    expected_previous_inventory_sha256: str,
    current_inventory: Path,
    expected_current_inventory_sha256: str,
    approved_patch: Path,
    expected_patch_sha256: str,
    fixed_judge_successor: bool = False,
) -> bool:
    """Explicitly transition only the final evaluator identity after review."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha256):
        raise EvaluationBudgetUnavailable("invalid expected ledger SHA256")
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    guard = EvaluationBudget(
        path, stage="final", run_id="quota-identity-transition", quota={}
    )
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        raw = path.read_bytes()
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise EvaluationBudgetUnavailable("invalid evaluation ledger")
        transitions = state.get("final_identity_transitions", [])
        if not isinstance(transitions, list) or any(
            not isinstance(entry, dict) for entry in transitions
        ):
            raise EvaluationBudgetUnavailable(
                "invalid final identity transition history"
            )
        transition_name = (
            "fixed-judge-quota-recovery"
            if fixed_judge_successor
            else "quota-input-lifecycle"
        )
        already = bool(
            transitions and transitions[-1].get("transition") == transition_name
        )
        if any(
            entry.get("transition") == transition_name for entry in transitions[:-1]
        ):
            raise EvaluationBudgetUnavailable(
                "duplicate final identity transition history"
            )
        if already:
            prior_text = transitions[-1].get("previous_ledger_text")
            if (
                not isinstance(prior_text, str)
                or hashlib.sha256(prior_text.encode()).hexdigest()
                != expected_ledger_sha256
            ):
                raise EvaluationBudgetUnavailable("different prior ledger identity")
            before = json.loads(prior_text)
        else:
            if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
                raise EvaluationBudgetUnavailable(
                    "reviewed prior ledger SHA256 differs"
                )
            prior_text = raw.decode("utf-8")
            before = json.loads(raw)
        guard._state = before
        guard._validate_state()
        bindings = before.get("stage_configurations", {})
        if (
            not isinstance(bindings, dict)
            or set(bindings) != {"B0", "final"}
            or not isinstance(bindings["final"], dict)
        ):
            raise EvaluationBudgetUnavailable(
                "final identity transition requires both stage bindings"
            )
        old = bindings["final"]
        if not isinstance(configuration, dict) or configuration != old | {
            "evaluation_fingerprint": configuration.get("evaluation_fingerprint")
        }:
            raise EvaluationBudgetUnavailable(
                "final identity transition changed configuration"
            )
        proof = _daily_policy_source_proof(
            previous_source_root,
            checkout,
            previous_inventory,
            expected_previous_inventory_sha256,
            current_inventory,
            expected_current_inventory_sha256,
            approved_patch,
            expected_patch_sha256,
            fixed_judge_successor=fixed_judge_successor,
        )
        if fixed_judge_successor and (
            old.get("evaluation_fingerprint") != _FIXED_PARENT_FINGERPRINT
            or expected_previous_inventory_sha256 != _FIXED_PARENT_INVENTORY
        ):
            raise EvaluationBudgetUnavailable("fixed judge transition parent differs")
        if (
            old.get("evaluation_fingerprint") != proof["previous_fingerprint"]
            or configuration.get("evaluation_fingerprint") != proof["fingerprint"]
            or configuration.get("application_configuration")
            != proof["fingerprint_input"]["application_configuration"]
        ):
            raise EvaluationBudgetUnavailable(
                "unproven final evaluation configuration identity"
            )
        entry = {
            "transition": transition_name,
            "approval": (
                "reviewed fixed-judge quota and recovery successor"
                if fixed_judge_successor
                else "reviewed final evaluator-only quota-input lifecycle"
            ),
            "previous_ledger_sha256": expected_ledger_sha256,
            "previous_ledger_text": prior_text,
            "previous_state": before,
            "source_proof": proof,
        }
        expected = json.loads(json.dumps(before))
        expected["stage_configurations"]["final"] = configuration
        expected.setdefault("final_identity_transitions", []).append(entry)
        if already:
            if state != expected:
                raise EvaluationBudgetUnavailable(
                    "final transition state or proof differs"
                )
            return False
        guard._state = expected
        guard._validate_state()
        guard._persist()
        return True
    finally:
        guard.__exit__(None, None, None)


def amend_daily_policy_budget(
    *,
    checkout: Path,
    ledger: Path,
    previous_source_root: Path,
    expected_ledger_sha256: str,
    configuration: dict[str, Any],
    quota_provenance: Path | None = None,
    prospective_proof: Path | None = None,
    expected_prospective_sha256: str = "",
    expected_account_id: str = "",
    expected_key_id: str = "",
    expected_provenance_sha256: str = "",
    preserve_current_day: bool = False,
    previous_inventory: Path | None = None,
    expected_previous_inventory_sha256: str = "",
    current_inventory: Path | None = None,
    expected_current_inventory_sha256: str = "",
    approved_patch: Path | None = None,
    expected_patch_sha256: str = "",
) -> bool:
    """Explicitly amend daily policy with preserved or proven day limits.

    Expansion requires separately reviewed, complete captured provenance;
    otherwise the existing day limits remain unchanged.
    """
    if (
        sum(
            (
                quota_provenance is not None,
                prospective_proof is not None,
                preserve_current_day,
            )
        )
        != 1
    ):
        raise EvaluationBudgetUnavailable("unproven current-day limit provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha256):
        raise EvaluationBudgetUnavailable("invalid expected ledger SHA256")
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    guard = EvaluationBudget(
        path, stage="final", run_id="daily-policy-amendment", quota={}
    )
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        raw = path.read_bytes()
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise EvaluationBudgetUnavailable("invalid evaluation ledger")
        history = state.get("policy_amendments", [])
        if not isinstance(history, list) or any(
            not isinstance(entry, dict) for entry in history
        ):
            raise EvaluationBudgetUnavailable("invalid policy amendment history")
        already = bool(
            history and history[-1].get("amendment") == "daily-policy-50-20261030"
        )
        if already:
            entry = history[-1]
            if entry.get("previous_ledger_sha256") != expected_ledger_sha256:
                raise EvaluationBudgetUnavailable("different policy amendment history")
            previous_state = entry.get("previous_state")
            previous_text = entry.get("previous_ledger_text")
            if (
                not isinstance(previous_state, dict)
                or not isinstance(previous_text, str)
                or hashlib.sha256(previous_text.encode("utf-8")).hexdigest()
                != expected_ledger_sha256
                or json.loads(previous_text) != previous_state
            ):
                raise EvaluationBudgetUnavailable("invalid previous amendment state")
        else:
            if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
                raise EvaluationBudgetUnavailable(
                    "reviewed prior ledger SHA256 differs"
                )
            previous_state = json.loads(raw)
            previous_text = raw.decode("utf-8")
        guard._state = previous_state
        guard._validate_state(
            daily_policy={"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20}
        )
        if any(
            day["used"] > day["batch_limit"]
            or day["batch_limit"] > 24
            or day["demo_reserve"] < 20
            for day in previous_state["days"].values()
        ):
            raise EvaluationBudgetUnavailable("invalid prior daily policy enforcement")
        bindings = previous_state.get("stage_configurations", {})
        if not isinstance(bindings, dict) or set(bindings) != {"B0", "final"}:
            raise EvaluationBudgetUnavailable(
                "daily amendment requires both stage bindings"
            )
        old = bindings["final"]
        if (
            not isinstance(old, dict)
            or not isinstance(configuration, dict)
            or configuration
            != old
            | {"evaluation_fingerprint": configuration.get("evaluation_fingerprint")}
        ):
            raise EvaluationBudgetUnavailable("policy amendment changed configuration")
        if (
            previous_inventory is None
            or current_inventory is None
            or approved_patch is None
        ):
            raise EvaluationBudgetUnavailable("reviewed source proof is required")
        proof = _daily_policy_source_proof(
            previous_source_root,
            checkout,
            previous_inventory,
            expected_previous_inventory_sha256,
            current_inventory,
            expected_current_inventory_sha256,
            approved_patch,
            expected_patch_sha256,
        )
        if (
            old.get("evaluation_fingerprint") != proof["previous_fingerprint"]
            or configuration.get("evaluation_fingerprint") != proof["fingerprint"]
            or configuration.get("application_configuration")
            != proof["fingerprint_input"]["application_configuration"]
        ):
            raise EvaluationBudgetUnavailable(
                "unproven evaluation configuration identity"
            )
        day_change = None
        if quota_provenance is not None:
            day_key, limit, day_proof = _reviewed_day_limit(
                previous_state,
                quota_provenance,
                expected_provenance_sha256,
                require_today=not already,
            )
            day_change = {
                "day": day_key,
                "previous_limit": 24,
                "new_limit": limit,
                "source_proof": day_proof,
            }
        if prospective_proof is not None:
            day_key, limit, day_proof = _prospective_day_limit(
                previous_state,
                prospective_proof,
                expected_prospective_sha256,
                expected_account_id,
                expected_key_id,
                require_fresh=not already,
            )
            day_change = {
                "day": day_key,
                "previous_limit": 24,
                "new_limit": limit,
                "source_proof": day_proof,
            }
        expected_entry = {
            "amendment": "daily-policy-50-20261030",
            "approval": "prospective daily ceiling 24 to 50; all other limits unchanged",
            "approved_date": "2026-10-30",
            "previous_ledger_sha256": expected_ledger_sha256,
            "previous_ledger_text": previous_text,
            "previous_state": previous_state,
            "source_proof": proof,
            "day_change": day_change,
        }
        if already:
            expected = json.loads(json.dumps(previous_state))
            expected["daily_policy"] = dict(DAILY_POLICY)
            if day_change is not None:
                expected["days"][day_change["day"]]["batch_limit"] = day_change[
                    "new_limit"
                ]
                if prospective_proof is not None:
                    expected["days"][day_change["day"]]["observations"][
                        day_proof["observation_id"]
                    ] = day_proof["baseline"]
                    expected["days"][day_change["day"]]["demo_reserve"] = day_proof[
                        "effective_reserve"
                    ]
            expected["stage_configurations"]["final"] = configuration
            expected.setdefault("policy_amendments", []).append(expected_entry)
            if state != expected:
                raise EvaluationBudgetUnavailable("amended state or proof differs")
            guard._state = state
            guard._validate_state()
            return False
        if (
            day_change is not None
            and datetime.now(UTC).date().isoformat() != day_change["day"]
        ):
            raise EvaluationBudgetUnavailable("UTC day changed before policy amendment")
        state["daily_policy"] = dict(DAILY_POLICY)
        if day_change is not None:
            state["days"][day_change["day"]]["batch_limit"] = day_change["new_limit"]
            if prospective_proof is not None:
                state["days"][day_change["day"]]["observations"][
                    day_proof["observation_id"]
                ] = day_proof["baseline"]
                state["days"][day_change["day"]]["demo_reserve"] = day_proof[
                    "effective_reserve"
                ]
        state["stage_configurations"]["final"] = configuration
        state.setdefault("policy_amendments", []).append(expected_entry)
        guard._state = state
        guard._validate_state()
        guard._persist(
            required_utc_day=day_change["day"] if day_change is not None else None,
            required_fresh_since=datetime.fromisoformat(day_proof["capture_started_at"])
            if prospective_proof is not None
            else None,
        )
        return True
    finally:
        guard.__exit__(None, None, None)


_B0_EVALUATOR_TRANSITION_PATHS = frozenset(
    {
        "backend/evaluation/portfolio_budget.py",
        "backend/evaluation/runtime_evaluation.py",
        "backend/evaluation/runtime_live.py",
        "scripts/evaluate_runtime_workflow.py",
        "scripts/transition_b0_evaluation_identity.py",
        "evals/runtime/scenarios.json",
        "evals/runtime/scenarios_review.md",
        "README.md",
        "docs/runtime_evaluation.md",
        "docs/full_evaluation_report_runbook.md",
        "docs/project_overview.md",
        "tests/test_portfolio_budget.py",
        "tests/test_runtime_evaluation.py",
        "tests/test_b0_identity_transition.py",
        "tests/test_cats_diagnostics.py",
        "tests/test_pilot_judge.py",
        "tests/test_retnet_scenario_contract.py",
    }
)


def _reviewed_source_inventory(root: Path, inventory_path: Path, expected: str):
    from backend.evaluation import evaluation_reporting as reporting

    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise EvaluationBudgetUnavailable("invalid source inventory SHA256")
    raw = inventory_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise EvaluationBudgetUnavailable("reviewed source inventory SHA256 differs")
    files = json.loads(raw)
    if not isinstance(files, dict):
        raise EvaluationBudgetUnavailable("invalid source inventory")
    for name, entry in files.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or not isinstance(entry, dict):
            raise EvaluationBudgetUnavailable("invalid source inventory path")
        path = root / rel
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise EvaluationBudgetUnavailable("source inventory file missing or unsafe")
        data = path.read_bytes()
        if entry != {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}:
            raise EvaluationBudgetUnavailable("source inventory file differs")
    # Reject unlisted code included in the actual fingerprint, even if ignored.
    backend = root / "backend"
    for path in backend.rglob("*"):
        if path.is_file() and reporting._is_identity_file(path.relative_to(backend)):
            if path.relative_to(root).as_posix() not in files:
                raise EvaluationBudgetUnavailable("unreviewed fingerprint source file")
    for name in reporting.HASHED_RELATIVE_PATHS:
        if (root / name).is_file() and name not in files:
            raise EvaluationBudgetUnavailable("unreviewed fingerprint source file")
    return files


def _b0_evaluation_transition_proof(
    previous: Path,
    current: Path,
    previous_inventory: Path,
    expected_previous_inventory_sha256: str,
    current_inventory: Path,
    expected_current_inventory_sha256: str,
):
    from backend.evaluation import evaluation_reporting as reporting

    inventories = [
        _reviewed_source_inventory(root, path, digest)
        for root, path, digest in (
            (previous, previous_inventory, expected_previous_inventory_sha256),
            (current, current_inventory, expected_current_inventory_sha256),
        )
    ]
    changed = {
        name: {"before": inventories[0].get(name), "after": inventories[1].get(name)}
        for name in sorted(inventories[0].keys() | inventories[1].keys())
        if inventories[0].get(name) != inventories[1].get(name)
    }
    if not changed or set(changed) - _B0_EVALUATOR_TRANSITION_PATHS:
        raise EvaluationBudgetUnavailable(
            "unapproved source change in B0 identity transition"
        )
    snapshots = []
    original_root = reporting.ROOT
    try:
        for root in (previous.resolve(), current.resolve()):
            reporting.ROOT = root
            snapshots.append(
                reporting.build_evaluation_snapshot({})["fingerprint_input"]
            )
    finally:
        reporting.ROOT = original_root
    before, after = snapshots
    for key in before.keys() | after.keys():
        if key not in {"tree_hashes", "file_hashes"} and before.get(key) != after.get(
            key
        ):
            raise EvaluationBudgetUnavailable(
                "application, prompt or storage fingerprint changed"
            )
    for name in before["file_hashes"].keys() | after["file_hashes"].keys():
        if (
            before["file_hashes"].get(name) != after["file_hashes"].get(name)
            and name not in changed
        ):
            raise EvaluationBudgetUnavailable("unreviewed fingerprint file changed")
    return {
        "previous_inventory_sha256": expected_previous_inventory_sha256,
        "current_inventory_sha256": expected_current_inventory_sha256,
        "changed_files": changed,
        "previous_fingerprint_input": before,
        "fingerprint_input": after,
        "previous_fingerprint": reporting._hash_json(before),
        "fingerprint": reporting._hash_json(after),
    }


def transition_b0_evaluation_identity(
    *,
    checkout: Path,
    ledger: Path,
    previous_source_root: Path,
    expected_ledger_sha256: str,
    configuration: dict[str, Any],
    previous_inventory: Path,
    expected_previous_inventory_sha256: str,
    current_inventory: Path,
    expected_current_inventory_sha256: str,
) -> bool:
    """Explicitly transition B0 to a reviewed evaluator-only frozen identity.

    All prior reservations, jobs, day observations, policy, and configuration
    details remain intact. This operation is never implicit during a run.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha256):
        raise EvaluationBudgetUnavailable("invalid expected ledger SHA256")
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    guard = EvaluationBudget(path, stage="B0", run_id="identity-transition", quota={})
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
            raise EvaluationBudgetUnavailable("reviewed prior ledger SHA256 differs")
        state = json.loads(raw)
        guard._state = state
        guard._validate_state()
        if state.get("identity_transitions"):
            raise EvaluationBudgetUnavailable("B0 identity transition already recorded")
        if set(state.get("stage_configurations", {})) != {"B0"} or any(
            entry["stage"] != "B0" for entry in state["reservations"]
        ):
            raise EvaluationBudgetUnavailable(
                "identity transition requires bound B0 campaign"
            )
        old = state["stage_configurations"]["B0"]
        if not isinstance(configuration, dict) or configuration != old | {
            "evaluation_fingerprint": configuration.get("evaluation_fingerprint")
        }:
            raise EvaluationBudgetUnavailable(
                "identity transition changed configuration"
            )
        proof = _b0_evaluation_transition_proof(
            previous_source_root,
            checkout,
            previous_inventory,
            expected_previous_inventory_sha256,
            current_inventory,
            expected_current_inventory_sha256,
        )
        if (
            old.get("evaluation_fingerprint") != proof["previous_fingerprint"]
            or configuration.get("evaluation_fingerprint") != proof["fingerprint"]
            or configuration.get("application_configuration")
            != proof["fingerprint_input"]["application_configuration"]
        ):
            raise EvaluationBudgetUnavailable(
                "unproven evaluation configuration identity"
            )
        previous_state = json.loads(raw)
        state["stage_configurations"]["B0"] = configuration
        state["identity_transitions"] = [
            {
                "transition": "b0-cats-evaluator-20261027",
                "applied_at": datetime.now(UTC).isoformat(),
                "previous_ledger_sha256": expected_ledger_sha256,
                "previous_state": previous_state,
                "source_proof": proof,
            }
        ]
        guard._validate_state()
        guard._persist()
        return True
    finally:
        guard.__exit__(None, None, None)


_FIXED_PARENT_FINGERPRINT = "aacdf873baacf605652888e1"
_FIXED_SETUP_JOB = "final:judge:semantic-v2:7c00c8a017a001f7dd970e594cea96cff333a98d4b518061baafd7b96022b978"
_FIXED_SETUP_FREEZE = "19bd9539577cde71c91a7706ca859830828659b95aff30e4a10e909796258330"
_FIXED_PARENT_INVENTORY = (
    "5ba709e9360e1d1d51461dad4e6a158c3f4c723efa1aed28da4d39e92addd969"
)
_FIXED_INPUT = "efa5b77ae9ed40d12220a014f2529dc83c00857817d61fb2f7484c1b1c7a1d73"


def recover_fixed_judge_setup(
    *,
    checkout: Path,
    ledger: Path,
    expected_ledger_sha256: str,
    evidence_dir: Path,
    parent_source: Path,
    parent_inventory: Path,
    input_file: Path,
    contract_pair: Path,
) -> None:
    """Authorize one audited bridge for the specific saved pre-dispatch job."""
    try:
        path = campaign_ledger(checkout, override=ledger)
    except ValueError as exc:
        raise EvaluationBudgetUnavailable(str(exc)) from exc
    guard = EvaluationBudget(
        path, stage="final", run_id="fixed-judge-recovery", quota={}
    )
    guard._lock = path.with_suffix(".lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(guard._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBudgetUnavailable("evaluation ledger is locked") from exc
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
            raise EvaluationBudgetUnavailable("reviewed prior ledger SHA256 differs")
        state = json.loads(raw)
        guard._state = state
        guard._validate_state()
        try:
            freeze_path = evidence_dir / "pilot_setup_stop_freeze.json"
            frozen = freeze_path.read_bytes()
            if hashlib.sha256(frozen).hexdigest() != _FIXED_SETUP_FREEZE:
                raise ValueError
            freeze = json.loads(frozen)
            for name, digest in freeze["files"].items():
                file = evidence_dir / name
                if (
                    file.is_symlink()
                    or file.resolve().is_relative_to(evidence_dir.resolve()) is False
                ):
                    raise ValueError
                if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                    raise ValueError
            _reviewed_source_inventory(
                parent_source, parent_inventory, _FIXED_PARENT_INVENTORY
            )
            if hashlib.sha256(input_file.read_bytes()).hexdigest() != _FIXED_INPUT:
                raise ValueError
            old = json.loads(
                (evidence_dir / "ledger_after_setup_stop.json").read_bytes()
            )
            prior = json.loads(
                (evidence_dir / "ledger_after_registration.json").read_bytes()
            )
            manifest = json.loads(
                (
                    evidence_dir
                    / "runs/01_contextual_mamba_visualization_graph_usefulness/manifest.json"
                ).read_bytes()
            )
            invocation = json.loads((evidence_dir / "01_invocation.json").read_bytes())
            exit_data = json.loads(
                (evidence_dir / "01_invocation_exit.json").read_bytes()
            )
            log = (evidence_dir / "01_invocation.log").read_text()
            original_job = old["jobs"][_FIXED_SETUP_JOB]
            prior_copy = json.loads(json.dumps(prior))
            prior_copy["jobs"][_FIXED_SETUP_JOB] = original_job
            if (
                old != prior_copy
                or original_job
                != {
                    "status": "started",
                    "run_id": "01_contextual_mamba_visualization_graph_usefulness",
                    "contract_digest": _FIXED_SETUP_JOB.rsplit(":", 1)[1],
                    "fixed_judge_pilot": True,
                }
                or exit_data["exit_code"] != 2
                or invocation["input_sha256"] != _FIXED_INPUT
                or manifest["status"] != "failed"
                or manifest["artifacts"] != []
                or manifest["error"]
                != "EvaluationBudgetUnavailable: evaluation model differs from quota observation"
                or "Runtime evaluation failed: evaluation model differs from quota observation"
                not in log
            ):
                raise ValueError
            if (
                state["reservations"] != old["reservations"]
                or state["jobs"] != old["jobs"]
                or any(
                    event["job_id"] == _FIXED_SETUP_JOB
                    for event in state["reservations"]
                )
                or state.get("fixed_judge_recoveries")
            ):
                raise ValueError
            transitions = state.get("final_identity_transitions", [])
            if (
                not transitions
                or transitions[-1].get("transition") != "fixed-judge-quota-recovery"
            ):
                raise ValueError
            pair = json.loads(contract_pair.read_bytes())
            if not isinstance(pair, dict) or set(pair) != {"old", "new"}:
                raise ValueError
            old_contract, new_contract = pair["old"], pair["new"]
            old_digest = hashlib.sha256(
                json.dumps(old_contract, sort_keys=True).encode()
            ).hexdigest()
            new_digest = hashlib.sha256(
                json.dumps(new_contract, sort_keys=True).encode()
            ).hexdigest()
            if (
                old_digest != original_job["contract_digest"]
                or new_digest == old_digest
                or old_contract.get("source_records_file_sha256") != _FIXED_INPUT
                or old_contract.get("dimension") != "graph_usefulness"
                or old_contract.get("judge_profile") != "fixed-pilot-v1"
                or old_contract.get("provider") != "openrouter"
                or old_contract.get("model") != "xingchen-agi/xing4.0-29b-a4b:free"
                or old_contract.get("method") != "json_schema"
                or old_contract.get("configuration")
                != old["stage_configurations"]["final"]
                or new_contract
                != old_contract
                | {"configuration": state["stage_configurations"]["final"]}
                or new_contract["configuration"].get("evaluation_fingerprint")
                != transitions[-1]["source_proof"]["fingerprint"]
            ):
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EvaluationBudgetUnavailable(
                "no-dispatch recovery evidence differs"
            ) from exc
        state["fixed_judge_recoveries"] = [
            {
                "old_job_id": _FIXED_SETUP_JOB,
                "new_job_id": "final:judge:semantic-v2:" + new_digest,
                "old_contract_digest": old_digest,
                "new_contract_digest": new_digest,
                "original_job": original_job,
                "setup_freeze_sha256": _FIXED_SETUP_FREEZE,
                "parent_inventory_sha256": _FIXED_PARENT_INVENTORY,
                "input_sha256": _FIXED_INPUT,
                "contract_pair_sha256": hashlib.sha256(
                    contract_pair.read_bytes()
                ).hexdigest(),
                "recovered_at": datetime.now(UTC).isoformat(),
                "consumed": False,
            }
        ]
        guard._persist()
    finally:
        guard.__exit__(None, None, None)
