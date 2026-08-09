from backend.utils.chat_limits import get_chat_limit_service
from backend.workflows.learner_reflex import get_workflow


def is_healthy() -> bool:
    """Returns True when the workflow and required quota storage are ready."""
    workflow = get_workflow()
    limits = get_chat_limit_service()
    return workflow._graph is not None and limits.ping()
