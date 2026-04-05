from __future__ import annotations


def test_predictions_list(client):
    response = client.get("/api/predictions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= 1


def test_prediction_detail(client):
    response = client.get("/api/predictions/2026-04-05_ST_1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["data"]["race_id"] == "2026-04-05_ST_1"
