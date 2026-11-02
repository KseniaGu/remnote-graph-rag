import hashlib
from pathlib import Path

from scripts.evaluate_retrieval_pipeline import DEFAULT_BENCHMARK_FILE
from scripts.evaluate_runtime_workflow import DEFAULT_SCENARIOS


def test_default_cases_are_beside_evaluation_implementation():
    root = Path(__file__).resolve().parents[1]
    assert DEFAULT_SCENARIOS == root / "backend/evaluation/cases/runtime/scenarios.json"
    assert (
        DEFAULT_BENCHMARK_FILE
        == root / "backend/evaluation/cases/retrieval/benchmark_cases.jsonl"
    )
    assert (
        hashlib.sha256(DEFAULT_SCENARIOS.read_bytes()).hexdigest()
        == "b0dfd1af372437e8c52257c70613e76130efe9721907300252b81d1bc4619f31"
    )
    assert (
        hashlib.sha256(DEFAULT_BENCHMARK_FILE.read_bytes()).hexdigest()
        == "e9b65d365571688e079225325bc6f8e354dee0c1c667483aa661e4da9f775f60"
    )
