from enum import IntEnum, StrEnum, Enum


class ModelRoleType(Enum):
    """A class representing various roles that DL/LL models can take on.

    The role value is a tuple consisting of an integer ID and a boolean value indicating whether the model is being used
    as a system agent.
    """
    other = (0, False)
    embedder = (1, False)
    orchestrator = (2, True)
    retriever = (3, True)
    researcher = (4, True)
    mentor = (5, True)
    analyst = (6, True)
    reranker = (8, False)

    @classmethod
    def get_all_members(cls, is_agent: bool = None):
        if is_agent is None:
            return list(cls.__members__.keys())
        return [key for key in cls.__members__.keys() if getattr(cls, key).value[1] == is_agent]


class PromptType(StrEnum):
    """A class representing different types of prompts used with language models."""
    learner_workflow = "learner_workflow"


class PDFParsingStatus(StrEnum):
    """A class representing the status of PDF file parsing."""
    file_exists = "PDF file is already saved"
    success = "PDF file is saved successfully"
    failed = "Error occurred while trying to save PDF"


class ImageParsingStatus(StrEnum):
    """A class representing the status of image file parsing."""
    file_exists = "Image file is already saved"
    success = "Image file is saved successfully"
    failed = "Error occurred while trying to save image"


class TextParsingStatus(StrEnum):
    """A class representing the status of text file parsing."""
    file_exists = "Text file is already saved"
    success = "Text file is saved successfully"
    failed = "Error occurred while trying to save text"


class ContentType(StrEnum):
    """A class representing different content types for HTTP responses."""
    application_json = "application/json"
    text_html = "text/html"
    text_plain = "text/plain"
    text_xml = "text/xml"
    image = "image"


class StorageType(StrEnum):
    """A class representing different types of storage backends."""
    local = "local"
    redis = "redis"
    neo4j = "neo4j"


class ExternalSourceUrlType(StrEnum):
    """A class representing different types of external URL sources."""
    source = "[SOURCE URL]"
    image = "[IMG URL]"
    article = "[ARTICLE URL]"


class KnowledgeGraphEntity(StrEnum):
    """A class representing entity types in the knowledge graph."""
    model = "MODEL"
    concept = "CONCEPT"
    component = "COMPONENT"
    formula = "FORMULA"
    tool = "TOOL"
    method = "METHOD"
    task = "TASK"
    paper = "PAPER"


class KnowledgeGraphRelation(StrEnum):
    """A class representing relation types between entities in the knowledge graph."""
    is_a = "IS_A"  # Hierarchy (ReLU is_a Activation Function)
    part_of = "PART_OF"  # Structure (Attention part_of Transformer)
    calculates = "CALCULATES"  # Math (Softmax calculates Probability)
    solves = "SOLVES"  # Problem-Solution (LSTM solves Vanishing Gradient)
    proposes = "PROPOSES"  # Citation ("Attention is all you need" proposes Transformer)
    uses = "USES"  # Usage (BERT uses Masked Language Modeling)
    related_to = "RELATED_TO"  # General relation
