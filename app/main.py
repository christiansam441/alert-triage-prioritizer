"""FastAPI dashboard for prioritized fraud and risk alert triage."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.database import (
    DEFAULT_DATABASE_PATH,
    VALID_STATUSES,
    get_case_states,
    initialize_database,
    update_case_state,
)
from app.prioritizer import (
    RANDOM_BASELINE_TRIALS,
    _average_random_precision,
    _precision_at_fraction,
    rank_queue,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
METRIC_FRACTIONS = (0.10, 0.25, 0.50)


class StatusUpdate(BaseModel):
    status: Literal[
        "open",
        "investigating",
        "escalated",
        "closed_true_positive",
        "closed_false_positive",
    ]
    note: str | None = Field(default=None, max_length=2_000)


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def _public_alert(alert: dict) -> dict:
    public = dict(alert)
    public.pop("is_true_positive", None)
    return public


def _build_precision_metrics(raw_alerts: list[dict], ranked: list[dict]) -> dict:
    random_means = _average_random_precision(
        raw_alerts, METRIC_FRACTIONS, RANDOM_BASELINE_TRIALS
    )
    metrics = {}
    for fraction in METRIC_FRACTIONS:
        alert_count, prioritized = _precision_at_fraction(ranked, fraction)
        _, raw_order = _precision_at_fraction(raw_alerts, fraction)
        metrics[f"top_{round(fraction * 100)}_percent"] = {
            "fraction": fraction,
            "alert_count": alert_count,
            "prioritized": round(prioritized, 4),
            "raw_order": round(raw_order, 4),
            "random_mean": round(random_means[fraction], 4),
        }
    return metrics


def create_app(
    data_dir: Path = DEFAULT_DATA_DIR,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> FastAPI:
    data_dir = Path(data_dir)
    database_path = Path(database_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        initialize_database(database_path)
        raw_alerts = _load_csv(data_dir / "alerts.csv")
        entities = _load_csv(data_dir / "entities.csv")
        ranked = rank_queue(raw_alerts, entities)
        alert_ids = {str(alert["alert_id"]) for alert in ranked}
        entity_alerts: dict[str, list[dict]] = defaultdict(list)
        for alert in ranked:
            entity_alerts[str(alert["entity_id"])].append(_public_alert(alert))
        for history in entity_alerts.values():
            history.sort(key=lambda row: str(row["raised_at"]), reverse=True)

        application.state.raw_alerts = raw_alerts
        application.state.ranked = ranked
        application.state.alert_ids = alert_ids
        application.state.entity_alerts = dict(entity_alerts)
        application.state.precision_metrics = _build_precision_metrics(
            raw_alerts, ranked
        )
        yield

    application = FastAPI(
        title="Signal Triage Console",
        version="1.0.0",
        lifespan=lifespan,
    )

    def queue_with_case_state() -> list[dict]:
        cases = get_case_states(database_path)
        joined_queue = []
        for alert in application.state.ranked:
            alert_id = str(alert["alert_id"])
            case = cases.get(
                alert_id,
                {
                    "status": "open",
                    "analyst_note": None,
                    "last_updated": None,
                },
            )
            joined = _public_alert(alert)
            joined.update(
                {
                    "status": case["status"],
                    "analyst_note": case["analyst_note"],
                    "last_updated": case["last_updated"],
                }
            )

            history = []
            for historical_alert in application.state.entity_alerts.get(
                str(alert["entity_id"]), []
            ):
                historical_id = str(historical_alert["alert_id"])
                historical_case = cases.get(
                    historical_id,
                    {
                        "status": "open",
                        "analyst_note": None,
                        "last_updated": None,
                    },
                )
                history.append(
                    {
                        "alert_id": historical_id,
                        "source_type": historical_alert["source_type"],
                        "severity": historical_alert["severity"],
                        "raised_at": historical_alert["raised_at"],
                        "priority_score": historical_alert["priority_score"],
                        "status": historical_case["status"],
                    }
                )
            joined["entity_alert_history"] = history
            joined_queue.append(joined)
        return joined_queue

    @application.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @application.get("/api/queue")
    async def get_queue() -> list[dict]:
        return queue_with_case_state()

    @application.get("/api/metrics")
    async def get_metrics() -> dict:
        cases = get_case_states(database_path)
        counts_by_status = {status: 0 for status in VALID_STATUSES}
        for alert_id in application.state.alert_ids:
            status = cases.get(alert_id, {"status": "open"})["status"]
            counts_by_status[status] += 1
        return {
            "total_alerts": len(application.state.ranked),
            "counts_by_status": counts_by_status,
            "top_k_precision": application.state.precision_metrics,
            "random_baseline_trials": RANDOM_BASELINE_TRIALS,
        }

    @application.post("/api/alerts/{alert_id}/status")
    async def set_status(alert_id: str, update: StatusUpdate) -> dict:
        if alert_id not in application.state.alert_ids:
            raise HTTPException(status_code=404, detail="Alert not found")
        return update_case_state(
            alert_id=alert_id,
            status=update.status,
            analyst_note=update.note,
            database_path=database_path,
        )

    return application


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Signal Triage Console</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #05070d;
      --panel: #0a101a;
      --panel-2: #0e1724;
      --line: #26364b;
      --text: #d7e2f0;
      --muted: #8293a8;
      --cyan: #19e6ff;
      --green: #5dff8b;
      --magenta: #ff4fd8;
      --amber: #ffc247;
      --red: #ff566e;
      --blue: #5f8cff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 18% 5%, rgba(25,230,255,.08), transparent 28rem),
        radial-gradient(circle at 92% 12%, rgba(255,79,216,.07), transparent 25rem),
        var(--bg);
      color: var(--text);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 14px;
    }
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 20;
      background: repeating-linear-gradient(
        to bottom,
        rgba(255,255,255,.018) 0,
        rgba(255,255,255,.018) 1px,
        transparent 1px,
        transparent 4px
      );
    }
    button, select, textarea { font: inherit; }
    button, select {
      color: var(--text);
      background: #0b1420;
      border: 1px solid var(--line);
    }
    button { cursor: pointer; }
    button:hover, button:focus-visible, select:focus-visible, textarea:focus-visible {
      border-color: var(--cyan);
      outline: none;
      box-shadow: 0 0 0 2px rgba(25,230,255,.12);
    }
    .shell { max-width: 1680px; margin: 0 auto; padding: 22px; }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      padding: 3px 0 18px;
      border-bottom: 1px solid var(--cyan);
      box-shadow: 0 8px 18px -18px var(--cyan);
    }
    .eyebrow { color: var(--green); font-size: 11px; letter-spacing: .18em; }
    h1 { margin: 7px 0 0; color: var(--cyan); font-size: clamp(22px, 3vw, 34px); letter-spacing: .07em; }
    .clock { color: var(--amber); white-space: nowrap; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin: 16px 0;
    }
    .metric {
      min-height: 86px;
      padding: 13px;
      background: linear-gradient(145deg, rgba(14,23,36,.95), rgba(7,12,20,.95));
      border: 1px solid var(--line);
      border-top-color: var(--metric-color, var(--cyan));
    }
    .metric:nth-child(2) { --metric-color: var(--magenta); }
    .metric:nth-child(3) { --metric-color: var(--amber); }
    .metric:nth-child(4) { --metric-color: var(--green); }
    .metric-label { color: var(--muted); font-size: 10px; letter-spacing: .11em; text-transform: uppercase; }
    .metric-value { margin-top: 8px; color: var(--metric-color, var(--cyan)); font-size: 23px; }
    .metric-sub { margin-top: 5px; color: var(--muted); font-size: 11px; }
    .status-strip { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 15px; }
    .status-count { padding: 6px 9px; border: 1px solid var(--line); color: var(--muted); }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(370px, .8fr); gap: 14px; }
    .panel { background: rgba(10,16,26,.94); border: 1px solid var(--line); min-width: 0; }
    .panel-heading { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .panel-title { color: var(--magenta); letter-spacing: .1em; font-size: 12px; }
    .terminal-mark { color: var(--green); }
    .filters { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    select { min-width: 145px; padding: 8px; }
    .queue-wrap { overflow: auto; max-height: calc(100vh - 295px); }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #0b1420;
      border-bottom: 1px solid var(--line);
      text-align: left;
      color: var(--muted);
      font-size: 10px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    th button { width: 100%; padding: 10px; border: 0; text-align: left; color: inherit; background: transparent; }
    td { padding: 10px; border-bottom: 1px solid #182436; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: rgba(25,230,255,.055); }
    tbody tr.selected { background: rgba(25,230,255,.10); box-shadow: inset 3px 0 var(--cyan); }
    .score { color: var(--cyan); font-weight: 700; }
    .severity-high { color: var(--red); }
    .severity-medium { color: var(--amber); }
    .severity-low { color: var(--green); }
    .source-watchlist_hit { color: var(--magenta); }
    .source-kyc_mismatch { color: var(--blue); }
    .source-transaction_monitoring { color: var(--cyan); }
    .source-velocity_rule { color: var(--amber); }
    .status-open { color: var(--muted); }
    .status-investigating { color: var(--cyan); }
    .status-escalated { color: var(--amber); }
    .status-closed_true_positive { color: var(--red); }
    .status-closed_false_positive { color: var(--green); }
    .detail { padding: 16px; max-height: calc(100vh - 225px); overflow: auto; }
    .empty { padding: 34px 10px; color: var(--muted); text-align: center; }
    .detail h2 { margin: 0; color: var(--cyan); font-size: 22px; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; margin: 14px 0; }
    .label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 4px; overflow-wrap: anywhere; }
    .evidence { margin: 14px 0; padding: 12px; border-left: 3px solid var(--green); background: rgba(93,255,139,.045); line-height: 1.55; }
    .description { color: #f1f5fa; line-height: 1.55; }
    .section-title { margin: 18px 0 9px; color: var(--amber); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; }
    textarea { width: 100%; min-height: 94px; resize: vertical; padding: 10px; color: var(--text); background: #070d15; border: 1px solid var(--line); line-height: 1.45; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 8px; }
    .actions button { padding: 9px 8px; }
    .actions button[data-status="escalated"] { border-color: #6c552a; color: var(--amber); }
    .actions button[data-status="closed_true_positive"] { border-color: #6f2e3a; color: var(--red); }
    .actions button[data-status="closed_false_positive"] { border-color: #285c3a; color: var(--green); }
    .save-note { grid-column: 1 / -1; border-color: #28556a; color: var(--cyan); }
    .history { display: grid; gap: 6px; }
    .history-item { display: grid; grid-template-columns: 76px 1fr auto; gap: 8px; align-items: center; padding: 8px; border: 1px solid #182436; font-size: 11px; }
    .history-item.current { border-color: var(--cyan); }
    .toast { min-height: 18px; margin-top: 10px; color: var(--green); font-size: 11px; }
    .error { color: var(--red); }
    @media (max-width: 1040px) {
      .workspace { grid-template-columns: 1fr; }
      .queue-wrap, .detail { max-height: none; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 580px) {
      .shell { padding: 12px; }
      header { align-items: flex-start; flex-direction: column; }
      .metrics { grid-template-columns: 1fr; }
      .detail-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <div class="eyebrow">RISK OPERATIONS // PRIORITY ENGINE ONLINE</div>
        <h1>SIGNAL TRIAGE CONSOLE</h1>
      </div>
      <div class="clock" id="clock">UTC --:--:--</div>
    </header>

    <section class="metrics" id="metrics" aria-label="Queue metrics"></section>
    <section class="status-strip" id="status-counts" aria-label="Case counts"></section>

    <section class="workspace">
      <div class="panel">
        <div class="panel-heading">
          <div class="panel-title"><span class="terminal-mark">&gt;</span> PRIORITIZED_QUEUE</div>
          <div id="visible-count">Loading...</div>
        </div>
        <div class="filters">
          <select id="status-filter" aria-label="Filter by status">
            <option value="">ALL STATUSES</option>
          </select>
          <select id="source-filter" aria-label="Filter by source">
            <option value="">ALL SOURCES</option>
          </select>
          <select id="severity-filter" aria-label="Filter by severity">
            <option value="">ALL SEVERITIES</option>
            <option value="high">HIGH</option>
            <option value="medium">MEDIUM</option>
            <option value="low">LOW</option>
          </select>
        </div>
        <div class="queue-wrap">
          <table>
            <thead>
              <tr>
                <th><button data-sort="priority_score">Score</button></th>
                <th><button data-sort="alert_id">Alert</button></th>
                <th><button data-sort="source_type">Source</button></th>
                <th><button data-sort="severity">Severity</button></th>
                <th><button data-sort="entity_id">Entity</button></th>
                <th><button data-sort="status">Status</button></th>
              </tr>
            </thead>
            <tbody id="queue-body"></tbody>
          </table>
        </div>
      </div>

      <aside class="panel">
        <div class="panel-heading">
          <div class="panel-title"><span class="terminal-mark">&gt;</span> CASE_DETAIL</div>
        </div>
        <div class="detail" id="detail">
          <div class="empty">Select an alert to inspect evidence and entity history.</div>
        </div>
      </aside>
    </section>
  </main>

  <script>
    "use strict";

    var queueData = [];
    var selectedAlertId = null;
    var sortKey = "priority_score";
    var sortDirection = -1;
    var statusValues = [
      "open",
      "investigating",
      "escalated",
      "closed_true_positive",
      "closed_false_positive"
    ];

    function formatLabel(value) {
      return String(value).replaceAll("_", " ");
    }

    function formatPercent(value) {
      return (Number(value) * 100).toFixed(1) + "%";
    }

    function element(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function updateClock() {
      document.getElementById("clock").textContent = "UTC " + new Date().toISOString().slice(11, 19);
    }

    function addFilterOptions(selectId, values) {
      var select = document.getElementById(selectId);
      values.forEach(function(value) {
        var option = document.createElement("option");
        option.value = value;
        option.textContent = formatLabel(value).toUpperCase();
        select.appendChild(option);
      });
    }

    function renderMetrics(metrics) {
      var container = document.getElementById("metrics");
      container.replaceChildren();
      var top10 = metrics.top_k_precision.top_10_percent;
      var top25 = metrics.top_k_precision.top_25_percent;
      var cards = [
        ["Total alerts", String(metrics.total_alerts), "Ranked active queue"],
        ["Top 10% precision", formatPercent(top10.prioritized), "Random mean " + formatPercent(top10.random_mean)],
        ["Top 25% precision", formatPercent(top25.prioritized), "Raw order " + formatPercent(top25.raw_order)],
        ["Random trials", metrics.random_baseline_trials.toLocaleString(), "Stable benchmark"]
      ];
      cards.forEach(function(card) {
        var wrapper = element("div", "metric");
        wrapper.appendChild(element("div", "metric-label", card[0]));
        wrapper.appendChild(element("div", "metric-value", card[1]));
        wrapper.appendChild(element("div", "metric-sub", card[2]));
        container.appendChild(wrapper);
      });

      var statusContainer = document.getElementById("status-counts");
      statusContainer.replaceChildren();
      statusValues.forEach(function(status) {
        var count = metrics.counts_by_status[status] || 0;
        var chip = element("div", "status-count status-" + status, formatLabel(status).toUpperCase() + ": " + count);
        statusContainer.appendChild(chip);
      });
    }

    function filteredAndSortedQueue() {
      var status = document.getElementById("status-filter").value;
      var source = document.getElementById("source-filter").value;
      var severity = document.getElementById("severity-filter").value;
      return queueData.filter(function(alert) {
        return (!status || alert.status === status)
          && (!source || alert.source_type === source)
          && (!severity || alert.severity === severity);
      }).sort(function(left, right) {
        var leftValue = left[sortKey];
        var rightValue = right[sortKey];
        if (typeof leftValue === "number") return (leftValue - rightValue) * sortDirection;
        return String(leftValue).localeCompare(String(rightValue)) * sortDirection;
      });
    }

    function renderQueue() {
      var body = document.getElementById("queue-body");
      body.replaceChildren();
      var visible = filteredAndSortedQueue();
      document.getElementById("visible-count").textContent = visible.length + " / " + queueData.length + " ALERTS";
      visible.forEach(function(alert) {
        var row = document.createElement("tr");
        row.dataset.alertId = alert.alert_id;
        if (alert.alert_id === selectedAlertId) row.classList.add("selected");
        row.tabIndex = 0;
        row.appendChild(element("td", "score", Number(alert.priority_score).toFixed(1)));
        row.appendChild(element("td", "", alert.alert_id));
        row.appendChild(element("td", "source-" + alert.source_type, formatLabel(alert.source_type)));
        row.appendChild(element("td", "severity-" + alert.severity, alert.severity.toUpperCase()));
        row.appendChild(element("td", "", alert.entity_id));
        row.appendChild(element("td", "status-" + alert.status, formatLabel(alert.status)));
        row.addEventListener("click", function() { selectAlert(alert.alert_id); });
        row.addEventListener("keydown", function(event) {
          if (event.key === "Enter" || event.key === " ") selectAlert(alert.alert_id);
        });
        body.appendChild(row);
      });
    }

    function detailField(label, value, className) {
      var wrapper = document.createElement("div");
      wrapper.appendChild(element("div", "label", label));
      wrapper.appendChild(element("div", "value " + (className || ""), value));
      return wrapper;
    }

    function selectAlert(alertId) {
      selectedAlertId = alertId;
      renderQueue();
      var alert = queueData.find(function(item) { return item.alert_id === alertId; });
      if (!alert) return;
      var detail = document.getElementById("detail");
      detail.replaceChildren();
      detail.appendChild(element("h2", "", alert.alert_id));
      detail.appendChild(element("div", "description", alert.description));

      var grid = element("div", "detail-grid");
      grid.appendChild(detailField("Priority", Number(alert.priority_score).toFixed(1), "score"));
      grid.appendChild(detailField("Status", formatLabel(alert.status), "status-" + alert.status));
      grid.appendChild(detailField("Source", formatLabel(alert.source_type), "source-" + alert.source_type));
      grid.appendChild(detailField("Severity", alert.severity, "severity-" + alert.severity));
      grid.appendChild(detailField("Entity", alert.entity_id));
      grid.appendChild(detailField("Raised", alert.raised_at));
      detail.appendChild(grid);

      detail.appendChild(element("div", "section-title", "Evidence breakdown"));
      detail.appendChild(element("div", "evidence", alert.score_breakdown));
      detail.appendChild(element("div", "section-title", "Analyst note"));
      var note = document.createElement("textarea");
      note.id = "analyst-note";
      note.maxLength = 2000;
      note.placeholder = "Add investigation context...";
      note.value = alert.analyst_note || "";
      detail.appendChild(note);

      var actions = element("div", "actions");
      statusValues.forEach(function(status) {
        var button = element("button", "", formatLabel(status).toUpperCase());
        button.type = "button";
        button.dataset.status = status;
        button.addEventListener("click", function() { saveCase(status); });
        actions.appendChild(button);
      });
      var saveNote = element("button", "save-note", "SAVE NOTE / KEEP " + formatLabel(alert.status).toUpperCase());
      saveNote.type = "button";
      saveNote.addEventListener("click", function() { saveCase(alert.status); });
      actions.appendChild(saveNote);
      detail.appendChild(actions);
      detail.appendChild(element("div", "toast", ""));

      detail.appendChild(element("div", "section-title", "Entity alert history"));
      var history = element("div", "history");
      alert.entity_alert_history.forEach(function(item) {
        var row = element("div", "history-item" + (item.alert_id === alert.alert_id ? " current" : ""));
        row.appendChild(element("span", "score", Number(item.priority_score).toFixed(1)));
        row.appendChild(element("span", "", item.alert_id + " // " + formatLabel(item.source_type) + " // " + item.severity));
        row.appendChild(element("span", "status-" + item.status, formatLabel(item.status)));
        history.appendChild(row);
      });
      detail.appendChild(history);
    }

    async function saveCase(status) {
      var toast = document.querySelector(".toast");
      var note = document.getElementById("analyst-note").value;
      toast.className = "toast";
      toast.textContent = "Writing case state...";
      try {
        var response = await fetch("/api/alerts/" + encodeURIComponent(selectedAlertId) + "/status", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({status: status, note: note})
        });
        if (!response.ok) throw new Error("Status update failed: " + response.status);
        await loadData();
        selectAlert(selectedAlertId);
        document.querySelector(".toast").textContent = "Case state saved and persisted.";
      } catch (error) {
        toast.className = "toast error";
        toast.textContent = error.message;
      }
    }

    async function loadData() {
      var responses = await Promise.all([fetch("/api/queue"), fetch("/api/metrics")]);
      if (!responses[0].ok || !responses[1].ok) throw new Error("Unable to load dashboard data");
      queueData = await responses[0].json();
      var metrics = await responses[1].json();
      renderMetrics(metrics);
      renderQueue();
    }

    document.querySelectorAll("select").forEach(function(select) {
      select.addEventListener("change", renderQueue);
    });
    document.querySelectorAll("th button").forEach(function(button) {
      button.addEventListener("click", function() {
        var requestedKey = button.dataset.sort;
        if (sortKey === requestedKey) sortDirection *= -1;
        else {
          sortKey = requestedKey;
          sortDirection = requestedKey === "priority_score" ? -1 : 1;
        }
        renderQueue();
      });
    });

    addFilterOptions("status-filter", statusValues);
    addFilterOptions("source-filter", ["transaction_monitoring", "kyc_mismatch", "velocity_rule", "watchlist_hit"]);
    updateClock();
    setInterval(updateClock, 1000);
    loadData().catch(function(error) {
      document.getElementById("visible-count").textContent = error.message;
      document.getElementById("visible-count").className = "error";
      console.error(error);
    });
  </script>
</body>
</html>
"""


app = create_app()
