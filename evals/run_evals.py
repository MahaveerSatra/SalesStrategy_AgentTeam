"""
Eval framework entry point.

Usage:
    python -m evals.run_evals --case TC-01              # run workflow + det checks + write judge prompt
    python -m evals.run_evals --case TC-01 --mock       # fast: synthetic state, no live API
    python -m evals.run_evals --case TC-01 --agent gatherer   # per-agent prompt (loads saved state)
    python -m evals.run_evals --ingest TC-01            # read judge JSON response, record scores
    python -m evals.run_evals --ingest TC-01 --agent gatherer # ingest per-agent judge response
    python -m evals.run_evals --compare                 # delta table for all cases
    python -m evals.run_evals --compare TC-01           # delta table for one case
"""
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from evals.deterministic_checks import run_all_checks
from evals.judge import format_agent_judge_prompt, format_judge_prompt, validate_judge_response
from evals.metrics import (
    RESULTS_DIR,
    append_to_history,
    print_delta_table,
    print_score_summary,
)
from evals.test_cases.registry import get_case

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_research_depth(value: str | None):
    """Convert test case JSON string (e.g. 'STANDARD') to ResearchDepth enum."""
    from src.models.state import ResearchDepth
    if value is None:
        return ResearchDepth.STANDARD
    return ResearchDepth(value.lower())


def _serialize_state(state: dict) -> dict:
    """
    Serialize state to a JSON-safe dict.

    Calls model_dump(mode="json") on known Pydantic fields so the output is
    proper nested dicts with enum values as strings, datetimes as ISO strings.
    Never uses default=str (which produces repr strings, not data).
    """
    result = dict(state)
    for key in ("opportunities", "validated_opportunities"):
        result[key] = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in (result.get(key) or [])
        ]
    result["signals"] = [
        sig.model_dump(mode="json") if hasattr(sig, "model_dump") else sig
        for sig in (result.get("signals") or [])
    ]
    if hasattr(result.get("progress"), "model_dump"):
        result["progress"] = result["progress"].model_dump(mode="json")
    # Handle top-level datetime fields (started_at, last_updated)
    for key, val in list(result.items()):
        if isinstance(val, datetime):
            result[key] = val.isoformat()
    return result


def _deserialize_state(data: dict) -> dict:
    """
    Reconstruct Pydantic objects from a JSON-loaded state dict.

    Inverse of _serialize_state — ensures loaded state has the same Pydantic
    type contract as live workflow state, so deterministic checks and judge
    formatters always receive typed objects.
    """
    from src.models.state import Opportunity, ResearchProgress, Signal
    result = dict(data)

    raw_signals = result.get("signals") or []
    result["signals"] = [
        Signal.model_validate(s) if isinstance(s, dict) else s
        for s in raw_signals
    ]

    for key in ("opportunities", "validated_opportunities"):
        opps = []
        for item in (result.get(key) or []):
            if isinstance(item, dict):
                if "evidence" in item and isinstance(item.get("evidence"), list):
                    item = {
                        **item,
                        "evidence": [
                            Signal.model_validate(e) if isinstance(e, dict) else e
                            for e in item["evidence"]
                        ],
                    }
                opps.append(Opportunity.model_validate(item))
            else:
                opps.append(item)
        result[key] = opps

    if isinstance(result.get("progress"), dict):
        result["progress"] = ResearchProgress.model_validate(result["progress"])

    return result


# ---------------------------------------------------------------------------
# Mock state builder (--mock fast path)
# ---------------------------------------------------------------------------

def _build_mock_state(case: dict) -> dict:
    """
    Build a synthetic ResearchState-like dict for fast --mock runs.

    Uses proper Pydantic Signal and Opportunity objects — structurally identical
    to live workflow state. This means det checks and judge formatters work the
    same way on mock and live state, ensuring mock passes are meaningful.
    """
    from src.models.state import (
        Opportunity,
        OpportunityConfidence,
        ResearchDepth,
        ResearchProgress,
        Signal,
    )

    inp = case["input"]
    account = inp["account_name"]
    industry = inp["industry"]
    seller = inp["seller_name"]
    slug = account.lower().replace(" ", "")

    # Build typed Signal objects — Pydantic validates on construction
    signals = [
        Signal(
            source=f"https://careers.{slug}.com/jobs",
            signal_type="hiring",
            content=(
                f"{account} is actively hiring simulation engineers and data scientists, "
                f"indicating investment in technical R&D. [SIG-001]"
            ),
            confidence=0.85,
            metadata={"url": f"https://careers.{slug}.com/jobs"},
        ),
        Signal(
            source=f"https://news.example.com/{account.lower().replace(' ', '-')}-mbd",
            signal_type="news",
            content=(
                f"{account} announced expanded use of model-based design in their "
                f"{industry} division. This aligns with MATLAB/Simulink workflows. [SIG-002]"
            ),
            confidence=0.80,
            metadata={"url": f"https://news.example.com/{account.lower().replace(' ', '-')}-mbd"},
        ),
        Signal(
            source=f"https://careers.{slug}.com/job/001",
            signal_type="hiring",
            content=(
                f"Job posting at {account}: 'Experience with MATLAB and Simulink required "
                f"for systems modeling role.' [JOB-001]"
            ),
            confidence=0.90,
            metadata={"url": f"https://careers.{slug}.com/job/001"},
        ),
        Signal(
            source=f"https://stackshare.io/{account.lower().replace(' ', '-')}",
            signal_type="tech_stack",
            content=(
                f"{account} tech stack includes Python, MATLAB, and C++ for embedded "
                f"systems development. [SIG-003]"
            ),
            confidence=0.75,
            metadata={"url": f"https://stackshare.io/{account.lower().replace(' ', '-')}"},
        ),
        Signal(
            source=f"https://ir.{slug}.com/q3-2025",
            signal_type="news",
            content=(
                f"{account} Q3 earnings mention increased engineering headcount and "
                f"investment in simulation tooling. [SIG-004]"
            ),
            confidence=0.70,
            metadata={"url": f"https://ir.{slug}.com/q3-2025"},
        ),
        Signal(
            source="https://linkedin.com/posts/sample",
            signal_type="hiring",
            content=(
                f"LinkedIn: {account} Director of Engineering posted about MBD adoption "
                f"challenges in {industry} workflows. [SIG-005]"
            ),
            confidence=0.65,
            metadata={"url": "https://linkedin.com/posts/sample"},
        ),
    ]

    # Build typed Opportunity objects — evidence references the Signal objects above
    validated_opportunities = [
        Opportunity(
            product_name="MATLAB",
            rationale=f"Active MATLAB hiring and confirmed tech stack usage at {account}.",
            confidence_score=0.82,
            confidence=OpportunityConfidence.HIGH,
            target_persona="Director of Engineering",
            talking_points=[
                f"Based on {account}'s job postings requiring MATLAB [JOB-001], "
                "MathWorks can directly support their existing workflows.",
                f"Their tech stack already includes MATLAB [SIG-003], "
                "making expansion of licenses a low-friction sale.",
                "MATLAB's data analytics capabilities align with their R&D investment [SIG-004].",
            ],
            evidence=[signals[0], signals[2], signals[3]],  # SIG-001, JOB-001, SIG-003
            risks=["Python-based tooling could reduce MATLAB license adoption."],
        ),
        Opportunity(
            product_name="Simulink",
            rationale=f"{account} is investing in model-based design across {industry} workflows.",
            confidence_score=0.78,
            confidence=OpportunityConfidence.MEDIUM,
            target_persona="Systems Engineer",
            talking_points=[
                f"{account} is investing in model-based design [SIG-002] — "
                f"Simulink is the industry standard for MBD in {industry}.",
                "Their engineering job listings [JOB-001] explicitly call for Simulink experience.",
            ],
            evidence=[signals[1], signals[2]],  # SIG-002, JOB-001
            risks=["Open-source alternatives (OpenModelica) gaining traction."],
        ),
    ]

    research_depth = _parse_research_depth(inp.get("research_depth"))

    return {
        "account_name": account,
        "industry": industry,
        "seller_name": seller,
        "user_context": inp.get("user_context"),
        "research_depth": research_depth,
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
        "progress": ResearchProgress(
            coordinator_complete=True,
            gatherer_complete=True,
            identifier_complete=True,
            validator_complete=True,
        ),
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
        user_context=inp.get("user_context"),
        research_depth=_parse_research_depth(inp.get("research_depth")),
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
        state = workflow.resume(thread_id, human_input="Looks good, the report is approved.")

    if state.get("waiting_for_human"):
        print("  Warning: workflow still waiting after 5 auto-approvals.")

    return dict(state)


# ---------------------------------------------------------------------------
# --case command
# ---------------------------------------------------------------------------

def cmd_case(case_id: str, mock: bool = False, agent: str | None = None) -> None:
    """Run workflow for a test case, execute det checks, write judge prompt."""
    agent_tag = f"  [agent={agent}]" if agent else ""
    mock_tag = "  [MOCK MODE]" if mock else ""
    print(f"\n{'='*60}")
    print(f"  EVAL RUN - {case_id}{mock_tag}{agent_tag}")
    print(f"{'='*60}")

    case = get_case(case_id)
    inp = case["input"]
    print(f"  Account  : {inp['account_name']}")
    print(f"  Industry : {inp['industry']}")
    print(f"  Seller   : {inp['seller_name']}")
    if inp.get("region"):
        print(f"  Region   : {inp['region']}")

    state_path = RESULTS_DIR / f"state_{case_id}.json"

    if agent:
        # Per-agent mode: load saved state rather than re-running workflow
        if state_path.exists():
            print(f"\n  Loading saved state from {state_path} ...")
            with open(state_path, "r", encoding="utf-8") as f:
                state = _deserialize_state(json.load(f))
        else:
            print("\n  No saved state found — running workflow first ...")
            if mock:
                state = _build_mock_state(case)
            else:
                state = _run_live_workflow(case)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(_serialize_state(state), f, indent=2)
    else:
        print()
        if mock:
            print("  Building synthetic mock state...")
            state = _build_mock_state(case)
        else:
            state = _run_live_workflow(case)

        # Always save state so per-agent runs can reuse it
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_state(state), f, indent=2)
        print(f"  State saved to: {state_path}")

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

    # Format and write judge prompt (per-agent or full)
    print("\n  Formatting judge prompt...")
    if agent:
        prompt = format_agent_judge_prompt(agent, case, state, det_results)
        pending_path = RESULTS_DIR / f"pending_judge_{case_id}_{agent}.txt"
        response_path = f"evals/results/judge_response_{case_id}_{agent}.json"
        ingest_cmd = f"python -m evals.run_evals --ingest {case_id} --agent {agent}"
    else:
        prompt = format_judge_prompt(case, state, det_results)
        pending_path = RESULTS_DIR / f"pending_judge_{case_id}.txt"
        response_path = f"evals/results/judge_response_{case_id}.json"
        ingest_cmd = f"python -m evals.run_evals --ingest {case_id}"

    with open(pending_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"\n  Judge prompt written to: {pending_path}")
    print()
    print("  --- NEXT STEPS (manual judge) ---------------------------")
    print(f"  1. Open:  {pending_path}")
    print("  2. Copy the full content and paste into Claude Pro (browser)")
    print("  3. Copy the JSON response Claude returns")
    print(f"  4. Save it to: {response_path}")
    print(f"  5. Run:  {ingest_cmd}")
    print("  ---------------------------------------------------------")


# ---------------------------------------------------------------------------
# --ingest command
# ---------------------------------------------------------------------------

def cmd_ingest(case_id: str, agent: str | None = None) -> None:
    """Read judge JSON response, validate, record scores."""
    agent_tag = f"  [agent={agent}]" if agent else ""
    print(f"\n{'='*60}")
    print(f"  INGEST - {case_id}{agent_tag}")
    print(f"{'='*60}")

    if agent:
        response_path = RESULTS_DIR / f"judge_response_{case_id}_{agent}.json"
    else:
        response_path = RESULTS_DIR / f"judge_response_{case_id}.json"

    if not response_path.exists():
        print(f"  Error: judge response file not found: {response_path}")
        print(f"  Run --case {case_id}{' --agent ' + agent if agent else ''} first, then paste judge output into that file.")
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
    append_to_history(run_id, case_id, judge_data, det_results, agent=agent or "all")

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
    parser.add_argument(
        "--agent",
        choices=["gatherer", "identifier", "validator", "coordinator"],
        default=None,
        help="Generate/ingest per-agent judge prompt (requires --case or --ingest)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.case:
        cmd_case(args.case, mock=args.mock, agent=args.agent)
    elif args.ingest:
        cmd_ingest(args.ingest, agent=args.agent)
    elif args.compare is not None:
        target = None if args.compare == "__all__" else args.compare
        cmd_compare(target)


if __name__ == "__main__":
    main()
