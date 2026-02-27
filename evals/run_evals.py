"""
Eval framework entry point.

Usage:
    python -m evals.run_evals --case TC-01           # run workflow + det checks + write judge prompt
    python -m evals.run_evals --case TC-01 --mock    # fast: synthetic state, no live API
    python -m evals.run_evals --ingest TC-01         # read judge JSON response, record scores
    python -m evals.run_evals --compare              # delta table for all cases
    python -m evals.run_evals --compare TC-01        # delta table for one case
"""
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from evals.deterministic_checks import run_all_checks
from evals.judge import format_judge_prompt, validate_judge_response
from evals.metrics import (
    RESULTS_DIR,
    append_to_history,
    print_delta_table,
    print_score_summary,
)
from evals.test_cases.registry import get_case

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Mock state builder (--mock fast path)
# ---------------------------------------------------------------------------

def _build_mock_state(case: dict) -> dict:
    """
    Build a synthetic ResearchState-like dict for fast --mock runs.
    Uses realistic-looking but entirely synthetic data.
    No live API calls or Ollama required.
    """
    inp = case["input"]
    account = inp["account_name"]
    industry = inp["industry"]
    seller = inp["seller_name"]

    # Build synthetic signals with metadata URLs and citation-style content
    signals = []
    mock_signal_data = [
        {
            "content": f"{account} is actively hiring simulation engineers and data scientists, "
                       f"indicating investment in technical R&D. [SIG-001]",
            "confidence": 0.85,
            "signal_type": "hiring",
            "metadata": {"url": f"https://careers.{account.lower().replace(' ', '')}.com/jobs"},
        },
        {
            "content": f"{account} announced expanded use of model-based design in their "
                       f"{industry} division. This aligns with MATLAB/Simulink workflows. [SIG-002]",
            "confidence": 0.80,
            "signal_type": "news",
            "metadata": {"url": f"https://news.example.com/{account.lower().replace(' ', '-')}-mbd"},
        },
        {
            "content": f"Job posting at {account}: 'Experience with MATLAB and Simulink required "
                       f"for systems modeling role.' [JOB-001]",
            "confidence": 0.90,
            "signal_type": "hiring",
            "metadata": {"url": f"https://careers.{account.lower().replace(' ', '')}.com/job/001"},
        },
        {
            "content": f"{account} tech stack includes Python, MATLAB, and C++ for embedded "
                       f"systems development. [SIG-003]",
            "confidence": 0.75,
            "signal_type": "tech_stack",
            "metadata": {"url": f"https://stackshare.io/{account.lower().replace(' ', '-')}"},
        },
        {
            "content": f"{account} Q3 earnings mention increased engineering headcount and "
                       f"investment in simulation tooling. [SIG-004]",
            "confidence": 0.70,
            "signal_type": "news",
            "metadata": {"url": f"https://ir.{account.lower().replace(' ', '')}.com/q3-2025"},
        },
        {
            "content": f"LinkedIn: {account} Director of Engineering posted about MBD adoption "
                       f"challenges in {industry} workflows. [SIG-005]",
            "confidence": 0.65,
            "signal_type": "hiring",
            "metadata": {"url": "https://linkedin.com/posts/sample"},
        },
    ]
    for item in mock_signal_data:
        signals.append(item)

    # Validated opportunities with citation-style talking points
    validated_opportunities = [
        {
            "product_name": "MATLAB",
            "confidence": 0.82,
            "target_persona": "Director of Engineering",
            "talking_points": [
                f"Based on {account}'s job postings requiring MATLAB [JOB-001], "
                "MathWorks can directly support their existing workflows.",
                f"Their tech stack already includes MATLAB [SIG-003], "
                "making expansion of licenses a low-friction sale.",
                "MATLAB's data analytics capabilities align with their R&D investment [SIG-004].",
            ],
            "evidence": ["SIG-001", "SIG-003", "JOB-001"],
            "supporting_signals": ["SIG-001", "SIG-003"],
        },
        {
            "product_name": "Simulink",
            "confidence": 0.78,
            "target_persona": "Systems Engineer",
            "talking_points": [
                f"{account} is investing in model-based design [SIG-002] — "
                "Simulink is the industry standard for MBD in {industry}.",
                "Their engineering job listings [JOB-001] explicitly call for Simulink experience.",
            ],
            "evidence": ["SIG-002", "JOB-001"],
            "supporting_signals": ["SIG-002"],
        },
    ]

    return {
        "account_name": account,
        "industry": industry,
        "seller_name": seller,
        "region": inp.get("region"),
        "signals": signals,
        "job_postings": [
            {"title": "Systems Engineer", "company": account, "url": "https://example.com/job1"},
            {"title": "Data Scientist", "company": account, "url": "https://example.com/job2"},
            {"title": "Embedded SW Engineer", "company": account, "url": "https://example.com/job3"},
        ],
        "news_items": [
            {"title": f"{account} expands R&D budget", "url": "https://news.example.com/1"},
        ],
        "tech_stack": ["MATLAB", "Python", "C++", "Simulink", "Git"],
        "opportunities": validated_opportunities,
        "validated_opportunities": validated_opportunities,
        "competitive_risks": [
            f"Ansys offers competing simulation tools in the {industry} space "
            f"— {account} may evaluate alternatives [SIG-002].",
            "Python-based open source tooling could reduce MATLAB license adoption.",
        ],
        "human_feedback": [],
        "waiting_for_human": False,
        "current_report": (
            f"# Sales Intelligence Report: {account}\n\n"
            f"**Seller**: {seller} | **Industry**: {industry}\n\n"
            "## Executive Summary\n"
            f"{account} presents a strong opportunity for MathWorks based on their confirmed "
            "use of MATLAB and active hiring for simulation engineers.\n\n"
            "## Top Opportunities\n"
            "1. **MATLAB** (confidence 0.82) — Expand existing licenses to new teams\n"
            "2. **Simulink** (confidence 0.78) — Support model-based design adoption\n\n"
            "## Recommended Next Steps\n"
            "- Schedule meeting with Director of Engineering (simulation team)\n"
            f"- Reference {account}'s MBD interest as conversation opener\n"
            "- Propose a toolchain assessment workshop\n\n"
            "## Competitive Landscape\n"
            "Ansys is the primary competitor in this space. MathWorks differentiates "
            "through tighter Simulink-to-code-generation workflow.\n"
        ),
        "workflow_iteration": 1,
        "langsmith_url": None,
    }


# ---------------------------------------------------------------------------
# Live workflow runner
# ---------------------------------------------------------------------------

def _run_live_workflow(case: dict) -> dict:
    """Run the real LangGraph workflow for the given test case."""
    from src.graph.workflow import ResearchWorkflow
    from src.models.state import ResearchDepth, create_initial_state

    inp = case["input"]
    case_id = case["id"]
    thread_id = f"eval_{case_id}"

    print(f"  Creating initial state for {inp['account_name']}...")
    initial_state = create_initial_state(
        account_name=inp["account_name"],
        industry=inp["industry"],
        seller_name=inp["seller_name"],
        region=inp.get("region"),
        research_depth=ResearchDepth.QUICK,
    )

    print(f"  Initializing workflow (seller={inp['seller_name']})...")
    workflow = ResearchWorkflow(seller_name=inp["seller_name"])

    print(f"  Running workflow (thread_id={thread_id})...")
    state = workflow.run(initial_state, thread_id=thread_id)

    # Handle human-in-loop interrupt — auto-approve for evals
    iterations = 0
    while state.get("waiting_for_human") and iterations < 5:
        iterations += 1
        print(f"  Workflow paused (iteration {iterations}) — auto-approving report...")
        state = workflow.resume(thread_id, human_input="continue")

    if state.get("waiting_for_human"):
        print("  Warning: workflow still waiting after 5 auto-approvals.")

    return dict(state)


# ---------------------------------------------------------------------------
# --case command
# ---------------------------------------------------------------------------

def cmd_case(case_id: str, mock: bool = False) -> None:
    """Run workflow for a test case, execute det checks, write judge prompt."""
    print(f"\n{'='*60}")
    print(f"  EVAL RUN - {case_id}{'  [MOCK MODE]' if mock else ''}")
    print(f"{'='*60}")

    case = get_case(case_id)
    inp = case["input"]
    print(f"  Account  : {inp['account_name']}")
    print(f"  Industry : {inp['industry']}")
    print(f"  Seller   : {inp['seller_name']}")
    if inp.get("region"):
        print(f"  Region   : {inp['region']}")

    print()
    if mock:
        print("  Building synthetic mock state...")
        state = _build_mock_state(case)
    else:
        state = _run_live_workflow(case)

    # Run deterministic checks
    print("\n  Running deterministic checks...")
    det_results = run_all_checks(state, case)

    print()
    passed_count = sum(1 for r in det_results if r["passed"])
    for r in det_results:
        icon = "PASS" if r["passed"] else "FAIL"
        print(f"    [{icon}]  {r['check']:<30}  {r['detail']}")
    print(f"\n  Result: {passed_count}/{len(det_results)} checks passed")

    # Save det results for --ingest step
    det_path = RESULTS_DIR / f"det_results_{case_id}.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(det_results, f, indent=2)

    # Format and write judge prompt
    print("\n  Formatting judge prompt...")
    prompt = format_judge_prompt(case, state, det_results)

    pending_path = RESULTS_DIR / f"pending_judge_{case_id}.txt"
    with open(pending_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"\n  Judge prompt written to: {pending_path}")
    print()
    print("  --- NEXT STEPS (manual judge) ---------------------------")
    print(f"  1. Open:  {pending_path}")
    print("  2. Copy the full content and paste into Claude Pro (browser)")
    print("  3. Copy the JSON response Claude returns")
    print(f"  4. Save it to: evals/results/judge_response_{case_id}.json")
    print(f"  5. Run:  python -m evals.run_evals --ingest {case_id}")
    print("  ---------------------------------------------------------")


# ---------------------------------------------------------------------------
# --ingest command
# ---------------------------------------------------------------------------

def cmd_ingest(case_id: str) -> None:
    """Read judge JSON response, validate, record scores."""
    print(f"\n{'='*60}")
    print(f"  INGEST - {case_id}")
    print(f"{'='*60}")

    response_path = RESULTS_DIR / f"judge_response_{case_id}.json"
    if not response_path.exists():
        print(f"  Error: judge response file not found: {response_path}")
        print(f"  Run --case {case_id} first, then paste judge output into that file.")
        sys.exit(1)

    with open(response_path, "r", encoding="utf-8") as f:
        try:
            judge_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Error: invalid JSON in {response_path}: {e}")
            sys.exit(1)

    valid, error_msg = validate_judge_response(judge_data)
    if not valid:
        print(f"  Error: judge response failed schema validation: {error_msg}")
        print("  Fix the JSON file and retry.")
        sys.exit(1)

    print("  Judge response validated OK.")

    # Load det results from previous --case run
    det_path = RESULTS_DIR / f"det_results_{case_id}.json"
    if det_path.exists():
        with open(det_path, "r", encoding="utf-8") as f:
            det_results = json.load(f)
    else:
        print("  Warning: no det_results found — run --case first for full results.")
        det_results = []

    run_id = str(uuid.uuid4())[:8]
    append_to_history(run_id, case_id, judge_data, det_results)

    print(f"  Scores recorded (run_id={run_id})")
    print_score_summary(case_id, judge_data, det_results)


# ---------------------------------------------------------------------------
# --compare command
# ---------------------------------------------------------------------------

def cmd_compare(case_id: str | None = None) -> None:
    """Print delta table comparing first vs latest run per case."""
    print(f"\n{'='*60}")
    print(f"  SCORE DELTA{'  ('+case_id+')' if case_id else ''}")
    print(f"{'='*60}")
    print_delta_table(case_id)


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_evals",
        description="Eval framework for SalesStrategy_AgentTeam",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case",
        metavar="CASE_ID",
        help="Run workflow for a test case (e.g. TC-01)",
    )
    group.add_argument(
        "--ingest",
        metavar="CASE_ID",
        help="Ingest judge JSON response and record scores (e.g. TC-01)",
    )
    group.add_argument(
        "--compare",
        nargs="?",
        const="__all__",
        metavar="CASE_ID",
        help="Print score delta table (optionally filter by case ID)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic fixture state instead of live workflow (fast, no API required)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.case:
        cmd_case(args.case, mock=args.mock)
    elif args.ingest:
        cmd_ingest(args.ingest)
    elif args.compare is not None:
        target = None if args.compare == "__all__" else args.compare
        cmd_compare(target)


if __name__ == "__main__":
    main()
