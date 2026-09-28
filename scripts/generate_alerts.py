"""Generate reproducible synthetic alerts for fraud/risk triage.

The source reliability values below are generation-only ground truth. They shape
the probability that an alert is a true positive, but are intentionally omitted
from alerts.csv so a prioritization engine cannot use them as direct features.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


NUMERIC_SEED = 20260928
TEXT_SEED = 8173
ENTITY_COUNT = 70
ALERT_COUNT = 250
REFERENCE_TIME = datetime(2026, 9, 28, 17, 0, tzinfo=timezone.utc)

# Generation-only baseline likelihood that each source represents a confirmed
# issue before severity, entity history, repeat-alert, and random-noise effects.
# Even the highest-signal source remains noisy in an operational alert queue.
SOURCE_RELIABILITY = {
    "transaction_monitoring": 0.16,
    "kyc_mismatch": 0.20,
    "velocity_rule": 0.065,
    "watchlist_hit": 0.34,
}

SOURCE_FREQUENCY = {
    "transaction_monitoring": 0.38,
    "kyc_mismatch": 0.20,
    "velocity_rule": 0.30,
    "watchlist_hit": 0.12,
}

SEVERITY_EFFECT = {"low": -0.90, "medium": 0.05, "high": 0.95}

DESCRIPTIONS = {
    "transaction_monitoring": {
        "low": [
            "Unusual transaction pattern above peer baseline",
            "Small transfer outside established account behavior",
        ],
        "medium": [
            "Layered transfers across newly observed counterparties",
            "Transaction pattern inconsistent with stated activity",
        ],
        "high": [
            "Large rapid movement of funds through linked accounts",
            "High-value transaction chain with unexplained purpose",
        ],
    },
    "kyc_mismatch": {
        "low": [
            "Minor profile detail differs from verification record",
            "Business classification requires profile review",
        ],
        "medium": [
            "Identity attributes conflict across verification sources",
            "Ownership information differs from submitted profile",
        ],
        "high": [
            "Core identity details failed independent verification",
            "Material beneficial-owner mismatch detected",
        ],
    },
    "velocity_rule": {
        "low": [
            "Activity count modestly exceeded rolling threshold",
            "Short burst of attempts from a familiar channel",
        ],
        "medium": [
            "Rapid sequence of transactions across payment methods",
            "Repeated attempts exceeded the hourly velocity limit",
        ],
        "high": [
            "Extreme transaction burst across multiple channels",
            "Sustained high-frequency activity after repeated declines",
        ],
    },
    "watchlist_hit": {
        "low": [
            "Partial name similarity to a monitored party",
            "Weak transliteration match requires analyst review",
        ],
        "medium": [
            "Name and jurisdiction align with a monitored party",
            "Multiple identity fields match a watchlist record",
        ],
        "high": [
            "Strong identity match to a monitored party",
            "Name, birth year, and location match watchlist data",
        ],
    },
}


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def generate_entities(rng: random.Random) -> list[dict[str, object]]:
    """Create mostly clean entities with a small elevated-risk segment."""
    entities = []
    for number in range(1, ENTITY_COUNT + 1):
        elevated = rng.random() < 0.16
        score = rng.betavariate(5.5, 2.0) if elevated else rng.betavariate(1.25, 7.5)
        entities.append(
            {
                "entity_id": f"ENT-{number:04d}",
                "entity_type": rng.choices(
                    ["merchant", "account"], weights=[0.42, 0.58]
                )[0],
                "risk_history_score": round(score, 4),
            }
        )
    return entities


def generate_alerts(
    entities: list[dict[str, object]],
    rng: random.Random,
    text_rng: random.Random,
) -> list[dict[str, object]]:
    """Create alerts and noisy labels influenced by history, source, and severity."""
    history = [float(entity["risk_history_score"]) for entity in entities]

    # Elevated-history entities are somewhat more likely to generate alerts. A
    # random propensity term prevents alert volume from being a proxy for score.
    alert_propensity = [
        0.65 + (1.8 * score) + rng.gammavariate(1.5, 0.35) for score in history
    ]
    entity_indices = rng.choices(
        range(ENTITY_COUNT), weights=alert_propensity, k=ALERT_COUNT
    )
    entity_alert_counts = Counter(entity_indices)

    source_types = rng.choices(
        list(SOURCE_FREQUENCY), weights=list(SOURCE_FREQUENCY.values()), k=ALERT_COUNT
    )
    severities = rng.choices(
        ["low", "medium", "high"], weights=[0.43, 0.38, 0.19], k=ALERT_COUNT
    )
    timestamps = [
        REFERENCE_TIME
        - timedelta(hours=rng.randrange(1, 24 * 120), minutes=rng.randrange(60))
        for _ in range(ALERT_COUNT)
    ]

    true_positive = []
    descriptions = []
    for entity_index, source_type, severity in zip(
        entity_indices, source_types, severities
    ):
        history_score = float(history[entity_index])
        repeat_count = int(entity_alert_counts[entity_index])

        probability_score = (
            _logit(SOURCE_RELIABILITY[source_type])
            + SEVERITY_EFFECT[severity]
            + 2.25 * (history_score - 0.30)
            + 0.10 * min(max(repeat_count - 3, 0), 7)
            + rng.gauss(0.0, 0.65)
        )
        probability = min(max(_sigmoid(probability_score), 0.03), 0.97)
        true_positive.append(rng.random() < probability)
        descriptions.append(text_rng.choice(DESCRIPTIONS[source_type][severity]))

    alerts = []
    alert_parts = zip(
        entity_indices,
        source_types,
        severities,
        timestamps,
        descriptions,
        true_positive,
    )
    for number, (
        entity_index,
        source_type,
        severity,
        timestamp,
        description,
        label,
    ) in enumerate(
        alert_parts,
        start=1,
    ):
        alerts.append(
            {
                "alert_id": f"ALT-{number:05d}",
                "entity_id": entities[entity_index]["entity_id"],
                "source_type": source_type,
                "severity": severity,
                "raised_at": timestamp.isoformat().replace("+00:00", "Z"),
                "description": description,
                "is_true_positive": label,
            }
        )
    return sorted(
        alerts,
        key=lambda alert: (alert["raised_at"], alert["alert_id"]),
        reverse=True,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    entities: list[dict[str, object]], alerts: list[dict[str, object]]
) -> Path:
    """Write generated CSV files and the evaluation manifest."""
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(data_dir / "entities.csv", entities)
    _write_csv(data_dir / "alerts.csv", alerts)

    true_positive_rate = sum(
        bool(alert["is_true_positive"]) for alert in alerts
    ) / len(alerts)
    manifest = {
        "total_alerts": int(len(alerts)),
        "true_positive_rate": round(true_positive_rate, 4),
        "source_reliability": SOURCE_RELIABILITY,
    }
    with (data_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return data_dir


def main() -> None:
    rng = random.Random(NUMERIC_SEED)
    text_rng = random.Random(TEXT_SEED)
    entities = generate_entities(rng)
    alerts = generate_alerts(entities, rng, text_rng)
    data_dir = write_outputs(entities, alerts)

    print(f"Generated {len(entities)} entities and {len(alerts)} alerts in {data_dir}")
    true_positive_rate = sum(
        bool(alert["is_true_positive"]) for alert in alerts
    ) / len(alerts)
    print(f"True positive rate: {true_positive_rate:.2%}")
    source_counts = Counter(alert["source_type"] for alert in alerts)
    for source_type, count in sorted(source_counts.items()):
        print(f"{source_type}: {count}")


if __name__ == "__main__":
    main()
