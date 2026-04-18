import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langsmith import Client
from pymongo import MongoClient, UpdateOne

from backend.configs.observability import LangSmithSettings
from backend.configs.storage import MongoDBSettings
from backend.utils.helpers import get_logger

logger = get_logger("LangSmith log export")

COLLECTION_NAME = "langsmith_traces"

FREQUENCY_WINDOWS = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
}


def _serialize(value):
    """Recursively converts non-JSON-serializable types to strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _run_to_document(run) -> dict:
    """Converts a LangSmith Run object to a MongoDB-friendly dict."""
    return {
        "_id": str(run.id),
        "session_id": str(run.session_id),
        "name": run.name,
        "run_type": run.run_type,
        "status": run.status,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "inputs": run.inputs,
        "outputs": run.outputs,
        "error": run.error,
        "latency": (run.end_time - run.start_time).total_seconds() if run.start_time and run.end_time else None,
        "total_tokens": run.total_tokens,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "total_cost": str(run.total_cost) if run.total_cost else None,
        "prompt_cost": str(run.prompt_cost) if run.prompt_cost else None,
        "completion_cost": str(run.completion_cost) if run.completion_cost else None,
        "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
        "trace_id": str(run.trace_id) if run.trace_id else None,
        "tags": run.tags,
        "extra": run.extra,
        "feedback_stats": run.feedback_stats,
        "exported_at": datetime.now(timezone.utc),
    }


def _write_json(docs: list[dict], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = output_path / f"langsmith_traces_{timestamp}.json"
    serializable = [_serialize(doc) for doc in docs]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info("JSON export written", path=str(file_path), count=len(docs))


def _upsert_mongodb(docs: list[dict], mongo_settings: MongoDBSettings) -> None:
    mongo_client = MongoClient(mongo_settings.uri.get_secret_value())
    db = mongo_client[mongo_settings.db_name]
    collection = db[COLLECTION_NAME]

    collection.create_index("trace_id")
    collection.create_index("start_time")

    operations = [UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True) for doc in docs]
    result = collection.bulk_write(operations)
    logger.info(
        "MongoDB upsert complete",
        upserted=result.upserted_count,
        modified=result.modified_count,
    )
    mongo_client.close()


def export_traces(frequency: str = "daily", output: str = "mongodb", output_path: Path = Path("./exports")) -> int:
    """Fetches recent LangSmith traces and exports them to MongoDB, JSON, or both.

    Args:
        frequency: One of "hourly", "daily", "weekly". Controls how far back to look.
        output: One of "mongodb", "json", "both".
        output_path: Directory for JSON output (used when output is "json" or "both").

    Returns:
        Number of traces fetched.
    """
    ls_settings = LangSmithSettings()

    if not ls_settings.langsmith_api_key:
        logger.error("LANGSMITH_API_KEY is not set. Aborting export.")
        return 0

    project_name = ls_settings.langsmith_project
    if not project_name:
        logger.error("LANGSMITH_PROJECT is not set. Aborting export.")
        return 0

    window = FREQUENCY_WINDOWS[frequency] + timedelta(minutes=10)
    since = datetime.now(timezone.utc) - window

    logger.info(
        "Starting LangSmith export",
        project=project_name,
        frequency=frequency,
        output=output,
        since=since.isoformat(),
    )

    ls_client = Client(
        api_url=ls_settings.langsmith_endpoint,
        api_key=ls_settings.langsmith_api_key.get_secret_value(),
    )

    docs = [_run_to_document(run) for run in ls_client.list_runs(project_name=project_name, start_time=since)]

    if not docs:
        logger.info("No new traces found.")
        return 0

    if output in ("mongodb", "both"):
        _upsert_mongodb(docs, MongoDBSettings())

    if output in ("json", "both"):
        _write_json(docs, output_path)

    logger.info("Export complete", total=len(docs))
    return len(docs)


def main():
    parser = argparse.ArgumentParser(description="Export LangSmith traces to MongoDB and/or JSON.")
    parser.add_argument(
        "--frequency",
        choices=list(FREQUENCY_WINDOWS.keys()),
        default="daily",
        help="How far back to fetch traces (default: daily).",
    )
    parser.add_argument(
        "--output",
        choices=["mongodb", "json", "both"],
        default="mongodb",
        help="Export destination (default: mongodb).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./exports"),
        help="Directory for JSON output (default: ./exports).",
    )
    args = parser.parse_args()
    export_traces(frequency=args.frequency, output=args.output, output_path=args.output_path)


if __name__ == "__main__":
    main()
