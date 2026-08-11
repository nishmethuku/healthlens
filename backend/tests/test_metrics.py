def test_metrics_endpoint_exposes_prometheus_format(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    body = res.text
    assert "request_latency_seconds" in body
    assert "requests_total" in body


def test_metrics_endpoint_does_not_require_api_key(client):
    res = client.get("/metrics")
    assert res.status_code == 200


def test_metrics_reflect_recorded_requests(client):
    client.get("/health/live")
    res = client.get("/metrics")
    body = res.text
    assert 'endpoint="/health/live"' in body
