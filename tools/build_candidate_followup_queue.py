"""Build the next-action queue for the isolated-event candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tess_assoc.candidate_followup import build_followup_queue


def render(queue: dict) -> str:
    lines = [
        f"# Candidate follow-up queue ({queue['queue_version']})",
        "",
        queue["claim_boundary"],
        "",
        "| Priority | TIC | Decision | Repeat status | Next actions |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for entry in queue["entries"]:
        lines.append(
            f"| {entry['priority']} | {entry['tic_id']} | "
            f"{entry['decision']} | {entry['repeat_status']} | "
            f"{'; '.join(entry['actions'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    reports = Path("reports")
    repeat = json.loads((reports / "candidate_repeat_search.json").read_text())
    pixels = json.loads((reports / "tesscut_pixel_audit.json").read_text())
    queue = build_followup_queue(repeat["candidates"], pixels)
    (reports / "candidate_followup_queue.json").write_text(
        json.dumps(queue, indent=2) + "\n"
    )
    (reports / "candidate_followup_queue.md").write_text(render(queue))
    print(render(queue))
    print("wrote reports/candidate_followup_queue.json and .md")


if __name__ == "__main__":
    main()
