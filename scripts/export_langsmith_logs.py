import argparse
from datetime import datetime, timedelta, timezone

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


def _run_to_document(run) -> dict:
    """Converts a LangSmith Run object to a MongoDB-friendly dict."""
    doc = {
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
    return doc


def export_traces(frequency: str = "daily") -> int:
    """Fetches recent LangSmith traces and upserts them into MongoDB.

    Args:
        frequency: One of "hourly", "daily", "weekly". Controls how far back to look.

    Returns:
        Number of traces upserted.
    """
    ls_settings = LangSmithSettings()
    mongo_settings = MongoDBSettings()

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
        since=since.isoformat(),
    )

    ls_client = Client(
        api_url=ls_settings.langsmith_endpoint,
        api_key=ls_settings.langsmith_api_key.get_secret_value(),
    )

    mongo_client = MongoClient(mongo_settings.uri.get_secret_value())
    db = mongo_client[mongo_settings.db_name]
    collection = db[COLLECTION_NAME]

    collection.create_index("trace_id")
    collection.create_index("start_time")

    runs = ls_client.list_runs(
        project_name=project_name,
        start_time=since,
    )

    operations = []
    for run in runs:
        doc = _run_to_document(run)
        operations.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        )

    if not operations:
        logger.info("No new traces found.")
        mongo_client.close()
        return 0

    result = collection.bulk_write(operations)
    count = result.upserted_count + result.modified_count
    logger.info(
        "Export complete",
        upserted=result.upserted_count,
        modified=result.modified_count,
        total=count,
    )

    mongo_client.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Export LangSmith traces to MongoDB.")
    parser.add_argument(
        "--frequency",
        choices=list(FREQUENCY_WINDOWS.keys()),
        default="daily",
        help="How far back to fetch traces (default: daily).",
    )
    args = parser.parse_args()
    export_traces(frequency=args.frequency)


if __name__ == "__main__":
    main()
