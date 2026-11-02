from langchain_core.messages import AIMessage

from backend.workflows.learner_reflex import ReflexLearnerWorkflow


class FakeGraph:
    def __init__(self, error=None, current=None, graph=False, old_agent="ANALYST"):
        self.error = error
        self.current = current
        self.graph = graph
        self.old_agent = old_agent

    def state(self, initial):
        # The prior tagged assistant response represents a checkpointed graph history.
        old = AIMessage(
            content="stale old answer",
            additional_kwargs={"agent": f"[{self.old_agent}]"},
        )
        messages = [old, *initial["messages"]]
        if self.current:
            messages.append(
                AIMessage(
                    content=self.current, additional_kwargs={"agent": "[ANALYST]"}
                )
            )
        return {
            "messages": messages,
            "request_scope": "out_of_scope",
            "next_step": "__end__",
            "visual_artifacts": [{"nodes": []}] if self.graph else [],
        }

    async def ainvoke(self, initial, config):
        if self.error:
            raise self.error
        return self.state(initial)

    async def astream(self, initial, config):
        if self.error:
            raise self.error
        yield {"orchestrator": self.state(initial)}

    async def astream_events(self, initial, config, version):
        if self.error:
            raise self.error
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": self.state(initial)},
        }


def wrapper(graph):
    instance = ReflexLearnerWorkflow()
    instance._graph = graph
    return instance
