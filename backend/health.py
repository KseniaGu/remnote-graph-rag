from backend.workflows.learner_reflex import get_workflow


def is_healthy() -> bool:
    """Returns True if the workflow singleton has been fully initialized."""
    workflow = get_workflow()
    return workflow._graph is not None
