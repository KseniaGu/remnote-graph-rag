from pathlib import Path

# BASE
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
ENV_PATH = ROOT_DIR / ".env"
REMNOTE_FOLDER_NAME = "AI Research"
FILENAME_LENGTH_MAX = 100
PDF_PAGES_NUM_MAX = 20
WORKFLOW_LOGGING = "Graph RAG workflow"
DOCUMENT_PARSER_LOGGING = "Documents parsing"
MAX_RETRIES = 3
DEFAULT_EMBEDDING_DIM = 384

# KNOWLEDGE BASE
REMNOTE_IMAGE_HOST_MARKER = "remnote-user-data.s3.amazonaws.com"
ARXIV_ARTICLE_MARKER = "arxiv.org"
HAS_PARSED_URL_MARKER = "has_parsed_url"
HAS_UNPARSED_URL_MARKER = "has_unparsed_url"
EXCLUDED_EMBED_METADATA_KEYS = [
    "source", "child_ids", "depth_level", "original_text", "has_unparsed_url",
    "has_parsed_url", "parsed_from_external_url", "line_number", "path"
]
EXCLUDED_LLM_METADATA_KEYS = [
    "source", "child_ids", "depth_level", "original_text", "has_unparsed_url",
    "has_parsed_url", "parsed_from_external_url", "line_number"
]
MAX_TOKEN_COUNTS_PER_CALL = 8000
MAX_DOCUMENTS_TO_INDEX = 6000
TEST_SOURCES = [
    'почитать про деревья решений', 'AI Expert Roadmap !!!!',
    'Про историю КЗ быстренько глянуть (как там развивались clip, dalle, diffusions etc)',
    '[GRPO Explained] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models',
    'Mixed Precision _ by Benjamin Warner', 'Building Effective AI Agents _ Anthropicпослушать ноутбук',
    'Measuring Massive Multitask Language Understanding', 'Про оптимайзеры тоже почитать',
    'All the Transformer Math You Need to Know _ How To Scale Your Model',
    'Variational Autoencoders (VAEs) вспомнить_',
    'CLIP или SigLIP_База по Computer vision собеседованиям_Middle_Senior',
    '[2404.17625] Alice\'s Adventures in a Differentiable Wonderland -- Volume I, A Tour of the Land',
    'Часть 3_ Diffusion Transformer (DiT) — Stable Diffusion 3 как она есть _ Хабр глянуть тоже',
    '[2205.11487] Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding(IMAGEN)',
    'Почитать про низкоранговое приближение симметричных матриц - часто используется походу',
    'Coding Agents 101: The Art of Actually Getting Things Done',
    'Word Embeddings', 'Paper page - Rope to Nope and Back Again_ A New Hybrid Attention Strategy', 'Algorithms basics',
    'Линейную алгебру ещё раз прогнать всю и законспектировать (обратные матрицы, svd etc  со сложностями)',
    'Backpropagation', 'Super study guide transformers and llm ADD NOTES', 'почитать про DPO, PPO, GRPO',
]

# VISUALIZATION
SPRING_LAYOUT_K = 0.5
SPRING_LAYOUT_ITERATIONS = 50
