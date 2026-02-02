from typing import Annotated, Sequence, Optional, Literal, Any, get_args

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    """Structured output from web research containing findings with source attribution."""
    key_findings: str = Field(
        description="Synthesized summary of the most relevant information found on the web, focusing on what's missing from the knowledge base"
    )
    sources: list[dict[str, str]] = Field(
        description="List of sources consulted, each with 'title', 'url', and 'type' (e.g., 'documentation', 'academic', 'blog', 'news')"
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        description="Confidence in the research findings based on source quality and consistency"
    )
    status: Literal["success", "partial_match", "no_relevant_info"] = Field(
        description="Status of the research operation"
    )
    gap_analysis: Optional[str] = Field(
        default=None,
        description="Brief note on what information was NOT found despite searching"
    )


class RoutingDecision(BaseModel):
    """Determines the next agent to act."""
    next_step: Literal["retriever", "researcher", "analyst", "mentor", "visualizer", "__end__"] = Field(
        description="The target worker node"
    )
    reasoning: str = Field(
        description="Short justification for the routing choice"
    )


class State(BaseModel):
    """Graph workflow state configuration."""
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list, description="The full history of the conversation between user and agents."
    )
    context: str = Field(default="", description="The grounding context retrieved from the knowledge base.")
    visual_artifact: Optional[dict[str, Any]] = Field(
        default=None,
        description="The Plotly figure serialized as a dictionary"
    )
    next_step: Literal["retriever", "researcher", "analyst", "mentor", "visualizer", "__end__"] = Field(
        default="retriever", description="The next node the Orchestrator has decided to activate"
    )
    user_score: Optional[float] = Field(
        default=None, description="The quantitative evaluation of the user's latest answer"
    )

    @classmethod
    def get_literal_values(cls, field_name: str) -> tuple[Any, ...]:
        """Returns the allowed values for a Literal field."""
        if field_name not in cls.model_fields:
            raise AttributeError(f"'{cls.__name__}' has no field '{field_name}'")

        annotation = cls.model_fields[field_name].annotation
        return get_args(annotation)
