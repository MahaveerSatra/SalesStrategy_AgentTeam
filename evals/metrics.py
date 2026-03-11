"""
Score aggregation, CSV persistence, and delta comparison for eval results.
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("evals/results")
HISTORY_CSV = RESULTS_DIR / "history.csv"

CSV_COLUMNS = [
    "run_id",
    "case_id",
    "agent",  # "all" for end-to-end runs; agent name for per-agent runs
    "timestamp",
    "accuracy",
    "actionability",
    "alignment",
    "safety",
    "overall",
    "safety_concern",
    "det_passed",
    "det_total",
    "notes",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _ensure_csv_exists() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_CSV.exists():
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_to_history(
    run_id: str,
    case_id: str,
    judge_data: dict,
    det_results: list[dict],
    notes: str = "",
    agent: str = "all",
) -> None:
    """Append one row to history.csv."""
    _ensure_csv_exists()
    passed = sum(1 for r in det_results if r["passed"])
    total = len(det_results)

    row = {
        "run_id": run_id,
        "case_id": case_id,
        "agent": agent,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": judge_data.get("accuracy_score", ""),
        "actionability": judge_data.get("actionability_score", ""),
        "alignment": judge_data.get("alignment_score", ""),
        "safety": judge_data.get("safety_score", ""),
        "overall": judge_data.get("overall_score", ""),
        "safety_concern": str(judge_data.get("safety_concern", False)).lower(),
        "det_passed": passed,
        "det_total": total,
        "notes": notes,
    }
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def load_history() -> list[dict]:
    """Return all rows from history.csv, or [] if file doesn't exist."""
    if not HISTORY_CSV.exists():
        return []
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Backward compat: old rows won't have the "agent" column
    for row in rows:
        row.setdefault("agent", "all")
    return rows


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_score_summary(case_id: str, judge_data: dict, det_results: list[dict]) -> None:
    """Print a formatted score summary to stdout."""
    passed = sum(1 for r in det_results if r["passed"])
    total = len(det_results)

    accuracy = judge_data.get("accuracy_score", "?")
    actionability = judge_data.get("actionability_score", "?")
    alignment = judge_data.get("alignment_score", "?")
    safety = judge_data.get("safety_score", "?")
    overall = judge_data.get("overall_score", "?")
    safety_concern = judge_data.get("safety_concern", False)

    print()
    print("=" * 60)
    print(f"  EVAL RESULTS - {case_id}")
    print("=" * 60)
    print(f"  Deterministic checks : {passed}/{total}")
    print(f"  Accuracy             : {accuracy}/5")
    print(f"  Actionability        : {actionability}/5")
    print(f"  Alignment            : {alignment}/5")
    print(f"  Safety & Ethics      : {safety}/5")
    print(f"  Overall              : {overall}/5")

    if safety_concern:
        print()
        print("  [!] SAFETY CONCERN FLAGGED — do not ship until resolved")
        flagged = judge_data.get("safety_flagged_text")
        if flagged:
            print(f'  Flagged text: "{flagged}"')

    print()
    print(f"  Key strength   : {judge_data.get('key_strength', '')}")
    print(f"  Key weakness   : {judge_data.get('key_weakness', '')}")
    print(f"  Improvement    : {judge_data.get('improvement_suggestion', '')}")
    print("=" * 60)


def print_delta_table(case_id: str | None = None) -> None:
    """
    Compare first vs latest run per case and print a delta table.
    If case_id is given, filter to that case only.
    """
    rows = load_history()
    if not rows:
        print("No history found. Run --ingest first.")
        return

    # Group by (case_id, agent)
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        cid = row["case_id"]
        agent = row.get("agent", "all")
        if case_id and cid != case_id:
            continue
        key = f"{cid} [{agent}]" if agent != "all" else cid
        by_case.setdefault(key, []).append(row)

    if not by_case:
        print(f"No history found for case '{case_id}'.")
        return

    metrics = ["accuracy", "actionability", "alignment", "safety", "overall"]
    print()
    print(f"{'CASE':<18} {'METRIC':<16} {'BEFORE':>7} {'AFTER':>7} {'DELTA':>8}")
    print("-" * 63)
    for cid, case_rows in sorted(by_case.items()):
        if len(case_rows) < 2:
            print(f"{cid:<18} (only 1 run - need >= 2 runs to compare)")
            continue
        first = case_rows[0]
        latest = case_rows[-1]
        for metric in metrics:
            try:
                before = float(first[metric])
                after = float(latest[metric])
                delta = after - before
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta:.1f}"
                indicator = " (up)" if delta > 0 else (" (dn)" if delta < 0 else "")
                print(
                    f"{cid:<18} {metric:<16} {before:>7.1f} {after:>7.1f} {delta_str:>8}{indicator}"
                )
            except (ValueError, KeyError):
                print(f"{cid:<18} {metric:<16} (no data)")
        print()
