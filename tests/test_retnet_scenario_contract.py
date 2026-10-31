from pathlib import Path

from backend.evaluation.runtime_evaluation import (
    RuntimeRecord,
    ToolObservation,
    load_runtime_scenarios,
    score_runtime_record,
)


def test_retnet_web_topic_accepts_neural_network_but_not_ambiguous_or_resnet():
    scenarios = load_runtime_scenarios(Path("evals/runtime/scenarios.json"))
    scenario = next(
        s for s in scenarios if s.id == "in_scope_retnet_local_miss_fallback"
    )
    assert scenario.version == 3
    record = RuntimeRecord(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id="offline-contract-check",
        source_mode="controlled_live",
        input=scenario.input,
    )
    for topic, expected in [
        ("RetNet neural network architecture linearized attention", "pass"),
        ("RetNet", "fail"),
        ("ResNet neural network architecture", "fail"),
        ("RetNet and ResNet neural network architectures", "fail"),
        ("RetNet machine learning architecture", "pass"),
    ]:
        record.tools = [
            ToolObservation(name="deep_web_research", arguments={"topic": topic})
        ]
        result = score_runtime_record(scenario, record)
        check = next(c for c in result.checks if c.name == "deep_web_research.topic")
        assert check.status == expected, topic
