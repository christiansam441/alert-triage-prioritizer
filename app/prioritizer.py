"""Explainable alert prioritization using observable alert patterns only."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


# These are the scoring engine's assumed source-signal point values. They are
# deliberately independent of the generator's ground-truth reliability values.
DEFAULT_SOURCE_RELIABILITY_WEIGHTS = {
    "transaction_monitoring": 14,
    "kyc_mismatch": 18,
    "velocity_rule": 8,
    "watchlist_hit": 24,
}

SEVERITY_WEIGHTS = {"low": 8, "medium": 20, "high": 34}
PRIOR_ALERT_POINTS = 4
MAX_PRIOR_ALERT_POINTS = 24
PRIOR_HIGH_SIGNAL_POINTS = 18
EVALUATION_SEED = 20260928
RANDOM_BASELINE_TRIALS = 1_000


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _display_name(value: str) -> str:
    return value.replace("_", " ")


def _is_high_reliability_source(
    source_type: str, source_reliability_weights: Mapping[str, float]
) -> bool:
    if not source_reliability_weights:
        return False
    high_signal_cutoff = max(source_reliability_weights.values()) * 0.75
    return source_reliability_weights.get(source_type, 0) >= high_signal_cutoff


def score_alert(
    alert: Mapping[str, object],
    entity: Mapping[str, object],
    source_reliability_weights: Mapping[str, float],
) -> dict:
    """Score one alert from severity, source, and observable prior alerts.

    ``rank_queue`` supplies ``_prior_alerts`` on its temporary entity context.
    The entity's synthetic ``risk_history_score`` is intentionally never read.
    """
    severity = str(alert.get("severity", "low"))
    source_type = str(alert.get("source_type", "unknown"))
    severity_points = float(SEVERITY_WEIGHTS.get(severity, 0))
    source_points = float(source_reliability_weights.get(source_type, 0))

    prior_alerts_value = entity.get("_prior_alerts", [])
    prior_alerts = (
        list(prior_alerts_value)
        if isinstance(prior_alerts_value, Sequence)
        and not isinstance(prior_alerts_value, (str, bytes))
        else []
    )
    prior_count = len(prior_alerts)
    prior_points = min(prior_count * PRIOR_ALERT_POINTS, MAX_PRIOR_ALERT_POINTS)

    prior_high_signal_count = sum(
        str(prior.get("severity")) == "high"
        and _is_high_reliability_source(
            str(prior.get("source_type", "")), source_reliability_weights
        )
        for prior in prior_alerts
        if isinstance(prior, Mapping)
    )
    prior_high_signal_points = (
        PRIOR_HIGH_SIGNAL_POINTS if prior_high_signal_count else 0
    )

    raw_score = (
        severity_points
        + source_points
        + prior_points
        + prior_high_signal_points
    )
    priority_score = round(min(max(raw_score, 0), 100), 1)

    breakdown = [
        f"{severity} severity (+{severity_points:g})",
        f"{_display_name(source_type)} source (+{source_points:g})",
    ]
    if prior_count:
        breakdown.append(
            f"{prior_count} prior alert{'s' if prior_count != 1 else ''} "
            f"on this entity (+{prior_points:g})"
        )
    else:
        breakdown.append("no prior alerts on this entity (+0)")
    if prior_high_signal_count:
        breakdown.append(
            f"{prior_high_signal_count} prior high-severity, high-reliability "
            f"alert{'s' if prior_high_signal_count != 1 else ''} "
            f"(+{prior_high_signal_points:g})"
        )

    return {
        "priority_score": priority_score,
        "score_breakdown": ", ".join(breakdown),
        "score_components": {
            "severity": severity_points,
            "source_reliability": source_points,
            "prior_alert_count": prior_points,
            "prior_high_signal": prior_high_signal_points,
        },
    }


def rank_queue(alerts: list[dict], entities: list[dict]) -> list[dict]:
    """Return alerts ranked by priority using only history known when raised."""
    entity_by_id = {
        str(entity.get("entity_id")): entity for entity in entities
    }
    alerts_by_entity: dict[str, list[dict]] = defaultdict(list)
    for alert in alerts:
        alerts_by_entity[str(alert.get("entity_id"))].append(alert)

    prior_alerts_by_id: dict[str, list[dict]] = {}
    for entity_alerts in alerts_by_entity.values():
        chronological = sorted(
            entity_alerts,
            key=lambda row: (
                _parse_timestamp(str(row["raised_at"])),
                str(row.get("alert_id", "")),
            ),
        )
        prior_alerts: list[dict] = []
        for alert in chronological:
            prior_alerts_by_id[str(alert.get("alert_id"))] = list(prior_alerts)
            prior_alerts.append(alert)

    ranked = []
    for alert in alerts:
        entity_id = str(alert.get("entity_id"))
        entity_context = dict(entity_by_id.get(entity_id, {"entity_id": entity_id}))
        entity_context.pop("risk_history_score", None)
        entity_context["_prior_alerts"] = prior_alerts_by_id.get(
            str(alert.get("alert_id")), []
        )
        scored = dict(alert)
        scored.update(
            score_alert(
                alert,
                entity_context,
                DEFAULT_SOURCE_RELIABILITY_WEIGHTS,
            )
        )
        ranked.append(scored)

    return sorted(
        ranked,
        key=lambda row: (
            -float(row["priority_score"]),
            -_parse_timestamp(str(row["raised_at"])).timestamp(),
            str(row.get("alert_id", "")),
        ),
    )


def _precision_at_fraction(queue: list[dict], fraction: float) -> tuple[int, float]:
    top_count = math.ceil(len(queue) * fraction)
    selected = queue[:top_count]
    true_positives = sum(
        str(alert.get("is_true_positive", "")).lower() == "true"
        for alert in selected
    )
    return top_count, true_positives / top_count


def _average_random_precision(
    alerts: list[dict], fractions: tuple[float, ...], trials: int
) -> dict[float, float]:
    """Average top-k precision over deterministic random queue orderings."""
    rng = random.Random(EVALUATION_SEED)
    precision_totals = {fraction: 0.0 for fraction in fractions}
    for _ in range(trials):
        shuffled = list(alerts)
        rng.shuffle(shuffled)
        for fraction in fractions:
            _, precision = _precision_at_fraction(shuffled, fraction)
            precision_totals[fraction] += precision
    return {
        fraction: precision_totals[fraction] / trials for fraction in fractions
    }


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    alerts = _load_csv(data_dir / "alerts.csv")
    entities = _load_csv(data_dir / "entities.csv")
    with (data_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if len(alerts) != manifest["total_alerts"]:
        raise ValueError("Alert count does not match manifest")
    observed_rate = sum(
        alert["is_true_positive"].lower() == "true" for alert in alerts
    ) / len(alerts)
    if round(observed_rate, 4) != manifest["true_positive_rate"]:
        raise ValueError("True-positive rate does not match manifest")

    prioritized = rank_queue(alerts, entities)
    raw_order = list(alerts)
    fractions = (0.10, 0.25, 0.50)
    random_precision = _average_random_precision(
        alerts, fractions, RANDOM_BASELINE_TRIALS
    )

    print("Top-k precision comparison")
    print(f"Random baseline: mean of {RANDOM_BASELINE_TRIALS:,} shuffles")
    print("queue_fraction,alert_count,prioritized,raw_order,random_mean")
    for fraction in fractions:
        top_count, prioritized_precision = _precision_at_fraction(
            prioritized, fraction
        )
        _, raw_precision = _precision_at_fraction(raw_order, fraction)
        print(
            f"{fraction:.0%},{top_count},{prioritized_precision:.2%},"
            f"{raw_precision:.2%},{random_precision[fraction]:.2%}"
        )

    example = prioritized[0]
    print("\nHighest-scoring alert")
    print(f"alert_id: {example['alert_id']}")
    print(f"entity_id: {example['entity_id']}")
    print(f"source_type: {example['source_type']}")
    print(f"severity: {example['severity']}")
    print(f"raised_at: {example['raised_at']}")
    print(f"description: {example['description']}")
    print(f"priority_score: {example['priority_score']}")
    print(f"evidence: {example['score_breakdown']}")
    print(f"is_true_positive (evaluation only): {example['is_true_positive']}")


if __name__ == "__main__":
    main()
