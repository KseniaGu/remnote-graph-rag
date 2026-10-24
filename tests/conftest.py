import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-local-assets",
        action="store_true",
        help="Run integration tests requiring private storage and a local embedding model",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-local-assets"):
        return
    skip = pytest.mark.skip(
        reason="local-assets integration: opt in with --run-local-assets; "
        "requires private storage and a local embedding model"
    )
    for item in items:
        if "local_assets" in item.keywords:
            item.add_marker(skip)
