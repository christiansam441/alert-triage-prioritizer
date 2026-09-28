"""API and persistence tests for the alert triage dashboard."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import DEFAULT_DATA_DIR, create_app


def test_dashboard_root(tmp_path: Path) -> None:
    application = create_app(
        data_dir=DEFAULT_DATA_DIR,
        database_path=tmp_path / "triage.db",
    )
    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "SIGNAL TRIAGE CONSOLE" in response.text
    assert "PRIORITIZED_QUEUE" in response.text
    assert "fetch(\"/api/queue\")" in response.text


def test_queue_returns_ranked_alerts_with_case_state(tmp_path: Path) -> None:
    application = create_app(
        data_dir=DEFAULT_DATA_DIR,
        database_path=tmp_path / "triage.db",
    )
    with TestClient(application) as client:
        response = client.get("/api/queue")

    assert response.status_code == 200
    queue = response.json()
    assert len(queue) == 250
    assert all(alert["status"] == "open" for alert in queue)
    assert all("score_breakdown" in alert for alert in queue)
    assert all("entity_alert_history" in alert for alert in queue)
    assert all("is_true_positive" not in alert for alert in queue)
    scores = [alert["priority_score"] for alert in queue]
    assert scores == sorted(scores, reverse=True)


def test_metrics_returns_status_counts_and_precision(tmp_path: Path) -> None:
    application = create_app(
        data_dir=DEFAULT_DATA_DIR,
        database_path=tmp_path / "triage.db",
    )
    with TestClient(application) as client:
        response = client.get("/api/metrics")

    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total_alerts"] == 250
    assert metrics["counts_by_status"]["open"] == 250
    assert sum(metrics["counts_by_status"].values()) == 250
    assert metrics["random_baseline_trials"] == 1_000
    assert set(metrics["top_k_precision"]) == {
        "top_10_percent",
        "top_25_percent",
        "top_50_percent",
    }
    assert metrics["top_k_precision"]["top_10_percent"] == {
        "fraction": 0.1,
        "alert_count": 25,
        "prioritized": 0.32,
        "raw_order": 0.08,
        "random_mean": 0.1871,
    }


def test_status_update_persists_across_app_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "triage.db"
    first_application = create_app(
        data_dir=DEFAULT_DATA_DIR,
        database_path=database_path,
    )
    with TestClient(first_application) as client:
        alert_id = client.get("/api/queue").json()[0]["alert_id"]
        update_response = client.post(
            f"/api/alerts/{alert_id}/status",
            json={
                "status": "investigating",
                "note": "Review linked activity before disposition.",
            },
        )
        assert update_response.status_code == 200
        assert update_response.json()["status"] == "investigating"

        refreshed_alert = next(
            alert
            for alert in client.get("/api/queue").json()
            if alert["alert_id"] == alert_id
        )
        assert refreshed_alert["status"] == "investigating"
        assert refreshed_alert["analyst_note"] == (
            "Review linked activity before disposition."
        )
        assert refreshed_alert["last_updated"] is not None

    restarted_application = create_app(
        data_dir=DEFAULT_DATA_DIR,
        database_path=database_path,
    )
    with TestClient(restarted_application) as client:
        persisted_alert = next(
            alert
            for alert in client.get("/api/queue").json()
            if alert["alert_id"] == alert_id
        )
        metrics = client.get("/api/metrics").json()

    assert persisted_alert["status"] == "investigating"
    assert persisted_alert["analyst_note"] == (
        "Review linked activity before disposition."
    )
    assert metrics["counts_by_status"]["investigating"] == 1
    assert metrics["counts_by_status"]["open"] == 249
