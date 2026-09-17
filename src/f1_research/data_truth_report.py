"""Render human-readable data-truth evidence from the machine matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_markdown(matrix: dict[str, Any]) -> str:
    events = matrix.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("matrix must contain at least one event")
    lines = [
        "# Data Truth Report",
        "",
        "Generated only from persisted provider reconciliation artifacts.",
        "",
        f"- Events: {matrix['event_count']}",
        f"- PASS: {matrix['pass_count']}",
        f"- PASS_WITH_GAPS: {matrix['pass_with_gaps_count']}",
        f"- FAIL: {matrix['fail_count']}",
        f"- Hard mismatches: {matrix['hard_mismatch_count']}",
        f"- Provider errors: {matrix['provider_error_count']}",
        "",
        "| Season | Round | OpenF1 session | Status | Hard mismatch | Hard gaps | Secondary gaps | Reconciled | Source SHA-256 |",
        "|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in events:
        lines.append(
            f"| {row['year']} | {row['round_number']} | {row['openf1_session_key']} | "
            f"{row['verification_status']} | {row['hard_mismatch_count']} | "
            f"{row['insufficient_hard_count']} | {row['insufficient_secondary_count']} | "
            f"{'yes' if row['reconciliation_performed'] else 'no'} | `{row['source_sha256']}` |"
        )
    failed = [row for row in events if row["verification_status"] == "FAIL"]
    if failed:
        lines += ["", "## Failure evidence", ""]
        for row in failed:
            provider_keys = ", ".join(row.get("provider_error_keys") or []) or "none"
            identity = "; ".join(row.get("event_identity_failures") or []) or "none"
            lines.append(
                f"- {row['year']} R{row['round_number']}: provider errors={provider_keys}; "
                f"event identity failures={identity}"
            )
    lines += [
        "",
        "## Evidence policy",
        "",
        "- Missing provider evidence remains unknown; it is never converted to zero or false.",
        "- Provider disagreements are not repaired by majority vote.",
        "- A passing artifact requires verified event identity and no provider errors.",
        "- SHA-256 identifies the exact reconciliation artifact summarized by each row.",
        "",
        "Public-provider agreement is evidence of cross-provider consistency, not official FIA certification.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(matrix), encoding="utf-8")


if __name__ == "__main__":
    main()
