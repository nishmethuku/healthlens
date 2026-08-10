from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    # Tests must not depend on whether a multi-GB local corpus happens to be
    # present on disk (it's gitignored and environment-specific) — stub the
    # index lookup so startup is instant and deterministic everywhere.
    with (
        patch.object(main, "get_corpus_index", return_value=None),
        TestClient(main.app) as c,
    ):
        yield c
