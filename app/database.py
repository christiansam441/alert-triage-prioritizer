"""SQLite persistence for alert case-management state."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


VALID_STATUSES = (
    "open",
    "investigating",
    "escalated",
    "closed_true_positive",
    "closed_false_positive",
)

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "triage.db"


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the case-state database and schema when needed."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_statuses = ", ".join(f"'{status}'" for status in VALID_STATUSES)
    with _connect(database_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS alert_cases (
                alert_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ({allowed_statuses})),
                analyst_note TEXT,
                last_updated TEXT NOT NULL
            )
            """
        )


def get_case_states(database_path: Path = DEFAULT_DATABASE_PATH) -> dict[str, dict]:
    """Return all persisted states keyed by alert identifier."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT alert_id, status, analyst_note, last_updated
            FROM alert_cases
            """
        ).fetchall()
    return {row["alert_id"]: dict(row) for row in rows}


def get_case_state(
    alert_id: str, database_path: Path = DEFAULT_DATABASE_PATH
) -> dict:
    """Return a persisted case state or the default open state."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT alert_id, status, analyst_note, last_updated
            FROM alert_cases
            WHERE alert_id = ?
            """,
            (alert_id,),
        ).fetchone()
    if row is None:
        return {
            "alert_id": alert_id,
            "status": "open",
            "analyst_note": None,
            "last_updated": None,
        }
    return dict(row)


def update_case_state(
    alert_id: str,
    status: str,
    analyst_note: str | None = None,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict:
    """Insert or update one alert's status and analyst note."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported case status: {status}")
    normalized_note = analyst_note.strip() if analyst_note else None
    last_updated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO alert_cases (alert_id, status, analyst_note, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                status = excluded.status,
                analyst_note = excluded.analyst_note,
                last_updated = excluded.last_updated
            """,
            (alert_id, status, normalized_note, last_updated),
        )
    return get_case_state(alert_id, database_path)
