from __future__ import annotations


def test_summary_endpoint(client):
    response = client.get("/api/summary?window=7d")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert "roi_stats" in payload["data"]


def test_result_detail(client):
    response = client.get("/api/results/2026-04-05_ST_1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["data"]["race_id"] == "2026-04-05_ST_1"
