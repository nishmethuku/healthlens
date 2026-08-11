from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main

TEST_API_KEY = "test-api-key"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def client():
    # Tests must not depend on whether a multi-GB local corpus happens to be
    # present on disk (it's gitignored and environment-specific) — stub the
    # index lookup so startup is instant and deterministic everywhere. Also
    # pin the API key to a known value so tests don't depend on whatever
    # HEALTHLENS_API_KEY happens to be set locally.
    with (
        patch.object(main, "get_corpus_index", return_value=None),
        patch.object(main.settings, "healthlens_api_key", TEST_API_KEY),
        TestClient(main.app) as c,
    ):
        yield c
