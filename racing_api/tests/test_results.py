from __future__ import annotations


def test_results_list(client):
    response = client.get("/api/results?status=completed")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert all(item["status"] == "completed" for item in payload["data"])


def test_invalid_route(client):
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "not_found"
