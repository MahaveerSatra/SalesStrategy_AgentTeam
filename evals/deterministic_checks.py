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
# Public runner
# ---------------------------------------------------------------------------

def run_all_checks(state: dict, case: dict) -> list[dict]:
    """
    Run all 9 deterministic checks, applying per-case overrides from eval_criteria.

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
        check_signal_count(state, min_signals=min_signals),
        check_citation_format(state),
        check_confidence_threshold(state, threshold=confidence_threshold),
        check_tech_stack_non_empty(state),
        check_no_duplicate_signals(state),
        check_opportunity_has_evidence(state),
        check_url_in_signals(state),
        check_report_generated(state),
        check_no_urgency_language(state),
    ]
    return results
