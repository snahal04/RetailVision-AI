#!/usr/bin/env python3
"""Part E — live terminal dashboard polling store metrics from the API."""

from __future__ import annotations

import argparse
import time

import httpx

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    raise SystemExit("Install rich: pip install rich")


def fetch(api: str, store_id: str) -> dict:
    base = api.rstrip("/")
    with httpx.Client(timeout=10.0) as client:
        metrics = client.get(f"{base}/stores/{store_id}/metrics").json()
        health = client.get(f"{base}/health").json()
    return {"metrics": metrics, "health": health}


def render(store_id: str, data: dict) -> Panel:
    m = data["metrics"]
    h = data["health"]
    table = Table(title=f"Store {store_id} — live metrics", expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Unique visitors (today)", str(m.get("unique_visitors", "—")))
    conv = m.get("conversion_rate")
    table.add_row("Conversion rate", f"{conv:.1%}" if conv is not None else "N/A")
    table.add_row("Queue depth", str(m.get("current_queue_depth", 0)))
    ab = m.get("queue_abandonment_rate")
    table.add_row("Queue abandonment", f"{ab:.1%}" if ab is not None else "N/A")
    table.add_row("As of", str(m.get("as_of", "—")))
    feed = next((s for s in h.get("stores", []) if s["store_id"] == store_id), {})
    table.add_row("Feed status", feed.get("status", "—"))
    return Panel(table, subtitle="Ctrl+C to quit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--store", default="STORE_BLR_002")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    console = Console()
    with Live(console=console, refresh_per_second=4) as live:
        while True:
            try:
                data = fetch(args.api, args.store)
                live.update(render(args.store, data))
            except Exception as e:
                live.update(Panel(f"API error: {e}", style="red"))
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
