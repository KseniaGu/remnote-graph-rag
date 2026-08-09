from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import SecretStr
from pymongo.errors import DuplicateKeyError

from backend.configs.chat_limits import ChatLimitsSettings
from backend.utils.chat_limits import ChatLimitService


def test_repeated_operation_id_returns_the_existing_counter_value() -> None:
    state = Mock()
    usage = Mock()
    state.find_one_and_update.side_effect = DuplicateKeyError("already reserved")
    state.find_one.return_value = {"count": 3}
    service = ChatLimitService(
        settings=ChatLimitsSettings(shared_quotas_enabled=True, _env_file=None),
        storage_settings=SimpleNamespace(
            uri=SecretStr("mongodb://example.invalid"),
            db_name="chat-tests",
        ),
        state_collection=state,
        usage_collection=usage,
    )

    count = service._conditional_increment(
        "provider_day:ollama:2026-08-09",
        "ollama_day",
        600,
        datetime(2026, 8, 9, tzinfo=UTC),
        "turn-id:llm:1",
    )

    assert count == 3
    update_query = state.find_one_and_update.call_args.args[0]
    update_document = state.find_one_and_update.call_args.args[1]
    assert update_query["operation_ids"] == {"$ne": "turn-id:llm:1"}
    assert update_document["$addToSet"] == {"operation_ids": "turn-id:llm:1"}
    state.find_one.assert_called_once_with(
        {"_id": "provider_day:ollama:2026-08-09", "operation_ids": "turn-id:llm:1"},
        projection={"count": 1},
    )
