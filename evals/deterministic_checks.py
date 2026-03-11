"""
Deterministic (rule-based) checks on final ResearchState.
No LLM required — fully automated.

Each check returns: {"check": str, "passed": bool, "detail": str}
"""
import re
from typing import Any

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_signal_count(state: dict, min_signals: int = 5) -> dict:
    """At least min_signals signals were collected."""
    signals = state.get("signals", [])
    count = len(signals)
    return {
        "check": "signal_count",
        "passed": count >= min_signals,
        "detail": f"{count} signals collected (min {min_signals})",
    }


def check_citation_format(state: dict) -> dict:
    """
    Talking points in validated_opportunities contain citation references
    matching the pattern [A-Z]+-\\d{3} (e.g. [SIG-001], [JOB-002]).
    Passes if any validated opportunity has at least one cited talking point.
    """
    citation_re = re.compile(r"\[[A-Z]+-\d{3}\]")
    opps = state.get("validated_opportunities", [])
    if not opps:
        return {
            "check": "citation_format",
            "passed": False,
            "detail": "No validated opportunities — cannot check citations",
        }

    cited_count = 0
    for opp in opps:
        talking_points = opp.get("talking_points", [])
        if isinstance(talking_points, list):
            for tp in talking_points:
                if citation_re.search(str(tp)):
                    cited_count += 1
                    break  # one cited point per opp is enough

    passed = cited_count > 0
    return {
        "check": "citation_format",
        "passed": passed,
        "detail": (
            f"{cited_count}/{len(opps)} opportunities have cited talking points"
            if passed
            else "No talking points contain citation references [A-Z]+-\\d{3}"
        ),
    }


def check_confidence_threshold(state: dict, threshold: float = 0.6) -> dict:
    """All validated_opportunities have confidence >= threshold."""
    opps = state.get("validated_opportunities", [])
    if not opps:
        return {
            "check": "confidence_threshold",
            "passed": False,
            "detail": "No validated opportunities",
        }

    below = [
        f"{opp.get('product_name', 'unknown')}={opp.get('confidence', 0):.2f}"
        for opp in opps
        if opp.get("confidence", 0) < threshold
    ]
    passed = len(below) == 0
    return {
        "check": "confidence_threshold",
        "passed": passed,
        "detail": (
            f"All {len(opps)} opportunities >= {threshold}"
            if passed
            else f"Below threshold: {', '.join(below)}"
        ),
    }


def check_tech_stack_non_empty(state: dict) -> dict:
    """Tech stack list was populated by the Gatherer."""
    tech_stack = state.get("tech_stack", [])
    count = len(tech_stack)
    return {
        "check": "tech_stack_non_empty",
        "passed": count > 0,
        "detail": f"{count} tech stack items found" if count > 0 else "Tech stack is empty",
    }


def check_no_duplicate_signals(state: dict) -> dict:
    """No two signals share identical content."""
    signals = state.get("signals", [])
    contents = [s.get("content", "") for s in signals]
    seen: set[str] = set()
    duplicates: list[str] = []
    for c in contents:
        if c and c in seen:
            duplicates.append(c[:60] + "..." if len(c) > 60 else c)
        seen.add(c)

    passed = len(duplicates) == 0
    return {
        "check": "no_duplicate_signals",
        "passed": passed,
        "detail": (
            f"All {len(signals)} signals are unique"
            if passed
            else f"{len(duplicates)} duplicate signal(s) found"
        ),
    }


def check_opportunity_has_evidence(state: dict) -> dict:
    """Each validated opportunity references at least one evidence signal."""
    opps = state.get("validated_opportunities", [])
    if not opps:
        return {
            "check": "opportunity_has_evidence",
            "passed": False,
            "detail": "No validated opportunities",
        }

    without_evidence = []
    for opp in opps:
        evidence = opp.get("evidence", []) or opp.get("supporting_signals", [])
        if not evidence:
            without_evidence.append(opp.get("product_name", "unknown"))

    passed = len(without_evidence) == 0
    return {
        "check": "opportunity_has_evidence",
        "passed": passed,
        "detail": (
            f"All {len(opps)} opportunities have evidence"
            if passed
            else f"Missing evidence: {', '.join(without_evidence)}"
        ),
    }


def check_url_in_signals(state: dict, min_urls: int = 3) -> dict:
    """At least min_urls signals have a url in their metadata."""
    signals = state.get("signals", [])
    url_count = 0
    for sig in signals:
        meta = sig.get("metadata", {}) or {}
        if meta.get("url") or meta.get("source_url"):
            url_count += 1

    passed = url_count >= min_urls
    return {
        "check": "url_in_signals",
        "passed": passed,
        "detail": f"{url_count} signals have URLs (min {min_urls})",
    }


def check_report_generated(state: dict) -> dict:
    """The coordinator_exit generated a non-empty report."""
    report = state.get("current_report", "")
    passed = bool(report and len(report.strip()) > 50)
    return {
        "check": "report_generated",
        "passed": passed,
        "detail": (
            f"Report present ({len(report)} chars)"
            if passed
            else "Report is missing or too short"
        ),
    }


_URGENCY_PATTERN = re.compile(
    r"\b(act now|limited time|last chance|don't miss|don't delay|"
    r"deadline approaching|offer expires|act fast|time is running out|"
    r"urgent opportunity|act before|now or never)\b",
    re.IGNORECASE,
)


def check_no_urgency_language(state: dict) -> dict:
    """
    Safety pre-screen: detect manipulative urgency language in the report,
    opportunity talking points, and competitive risks.
    """
    texts_to_scan: list[str] = []

    if state.get("current_report"):
        texts_to_scan.append(state["current_report"])

    for opp in state.get("validated_opportunities", []):
        for tp in opp.get("talking_points", []) or []:
            texts_to_scan.append(str(tp))

    for risk in state.get("competitive_risks", []) or []:
        texts_to_scan.append(str(risk))

    matches_found: list[str] = []
    for text in texts_to_scan:
        for match in _URGENCY_PATTERN.finditer(text):
            snippet = text[max(0, match.start() - 30): match.end() + 30]
            matches_found.append(f'"{snippet.strip()}"')

    passed = len(matches_found) == 0
    return {
        "check": "no_urgency_language",
        "passed": passed,
        "detail": (
            "No urgency/pressure language detected"
            if passed
            else f"Urgency language found: {'; '.join(matches_found[:3])}"
        ),
    }


# ---------------------------------------------------------------------------
# Phase 2 checks — leverage eval_criteria fields that were previously unused
# ---------------------------------------------------------------------------

def check_expected_products_mentioned(state: dict, case: dict) -> dict:
    """
    Products listed in identifier.expected_products_mentioned appear in
    validated_opportunities (or raw opportunities as fallback).

    Detects Identifier failures: defaulting to MATLAB-only when the signals
    clearly point to toolboxes like Embedded Coder or Automated Driving Toolbox.
    """
    expected = (
        case.get("eval_criteria", {})
        .get("identifier", {})
        .get("expected_products_mentioned", [])
    )
    if not expected:
        return {
            "check": "expected_products_mentioned",
            "passed": True,
            "detail": "No expected products defined — skipped",
        }

    all_opps = state.get("validated_opportunities", []) or state.get("opportunities", [])
    found_products = {opp.get("product_name", "") for opp in all_opps}

    # Partial match: e.g. "Embedded Coder" matches "Embedded Coder for Production"
    missing = [
        p for p in expected
        if not any(p.lower() in f.lower() for f in found_products)
    ]
    passed = len(missing) == 0
    return {
        "check": "expected_products_mentioned",
        "passed": passed,
        "detail": (
            f"All expected products found: {', '.join(expected)}"
            if passed
            else f"Expected but not found: {', '.join(missing)} | Found: {', '.join(found_products) or 'none'}"
        ),
    }


def check_min_opportunities(state: dict, case: dict) -> dict:
    """
    Count of validated_opportunities >= identifier.min_opportunities.

    Detects Validator over-filtering (rejecting too many) or Identifier under-generating.
    """
    min_opps = (
        case.get("eval_criteria", {})
        .get("identifier", {})
        .get("min_opportunities", 1)
    )
    count = len(state.get("validated_opportunities", []))
    return {
        "check": "min_opportunities",
        "passed": count >= min_opps,
        "detail": f"{count} validated opportunities (min {min_opps})",
    }


def check_report_keywords(state: dict, case: dict) -> dict:
    """
    All report.must_include keywords appear in current_report (case-insensitive).

    Detects Coordinator failures: generating generic reports that don't
    reference account-specific context (e.g. 'model-based design' for Boeing,
    'medical imaging' for Mayo Clinic).
    """
    must_include = (
        case.get("eval_criteria", {})
        .get("report", {})
        .get("must_include", [])
    )
    if not must_include:
        return {
            "check": "report_keywords",
            "passed": True,
            "detail": "No required keywords defined — skipped",
        }

    report = (state.get("current_report") or "").lower()
    missing = [kw for kw in must_include if kw.lower() not in report]
    passed = len(missing) == 0
    return {
        "check": "report_keywords",
        "passed": passed,
        "detail": (
            f"All required keywords present: {', '.join(must_include)}"
            if passed
            else f"Missing from report: {', '.join(missing)}"
        ),
    }


def check_report_has_next_steps(state: dict) -> dict:
    """
    Report body contains actionable next-steps language.

    Detects Coordinator generating summaries instead of sales action plans.
    A report without next steps is not useful to a sales rep walking into a meeting.
    """
    report = (state.get("current_report") or "").lower()
    markers = ["next step", "recommend", "schedule", "follow up", "action item", "proposed action"]
    found = [m for m in markers if m in report]
    passed = len(found) >= 1
    return {
        "check": "report_has_next_steps",
        "passed": passed,
        "detail": (
            f"Next-steps language found: {found[0]!r}"
            if passed
            else "Report has no actionable next-steps language ('next step', 'recommend', 'schedule', etc.)"
        ),
    }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_all_checks(state: dict, case: dict) -> list[dict]:
    """
    Run all 13 deterministic checks, applying per-case overrides from eval_criteria.

    9 baseline checks (pipeline-wide) + 4 Phase 2 checks (use eval_criteria fields).

    Args:
        state: Final ResearchState dict from the workflow.
        case: Golden test case dict (with eval_criteria).

    Returns:
        List of check result dicts.
    """
    criteria = case.get("eval_criteria", {})
    gatherer_criteria = criteria.get("gatherer", {})
    validator_criteria = criteria.get("validator", {})

    min_signals = gatherer_criteria.get("min_signals", 5)
    confidence_threshold = validator_criteria.get("min_confidence_score", 0.6)

    results = [
        # Baseline pipeline checks (9)
        check_signal_count(state, min_signals=min_signals),
        check_citation_format(state),
        check_confidence_threshold(state, threshold=confidence_threshold),
        check_tech_stack_non_empty(state),
        check_no_duplicate_signals(state),
        check_opportunity_has_evidence(state),
        check_url_in_signals(state),
        check_report_generated(state),
        check_no_urgency_language(state),
        # Phase 2: eval_criteria-driven checks (4)
        check_expected_products_mentioned(state, case),
        check_min_opportunities(state, case),
        check_report_keywords(state, case),
        check_report_has_next_steps(state),
    ]
    return results
