"""
Output formatters for CLI.

Provides functions to format research results in different formats:
- Terminal output (rich formatting)
- Markdown reports
- JSON exports
"""
import json
from datetime import datetime
from typing import Any

from ..models.state import ResearchState, Opportunity, Signal


def format_terminal_summary(state: ResearchState) -> str:
    """
    Format research state as a terminal-friendly summary.

    Args:
        state: Research state to format

    Returns:
        Formatted string for terminal display
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"RESEARCH SUMMARY: {state['account_name']}")
    lines.append("=" * 70)
    lines.append("")

    # Basic info
    lines.append(f"Industry: {state['industry']}")
    if state.get('region'):
        lines.append(f"Region: {state['region']}")
    lines.append(f"Research Depth: {state['research_depth'].value}")
    lines.append(f"Started: {state['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Progress
    progress = state['progress']
    completed = progress.get_completed_agents()
    lines.append(f"Completed Agents: {', '.join(completed) if completed else 'None'}")
    lines.append("")

    # Data collected
    lines.append("DATA COLLECTED:")
    lines.append(f"  - Signals: {len(state.get('signals', []))}")
    lines.append(f"  - Job Postings: {len(state.get('job_postings', []))}")
    lines.append(f"  - News Items: {len(state.get('news_items', []))}")
    lines.append(f"  - Tech Stack Items: {len(state.get('tech_stack', []))}")
    lines.append("")

    # Opportunities
    opportunities = state.get('validated_opportunities', [])
    if opportunities:
        lines.append(f"OPPORTUNITIES FOUND: {len(opportunities)}")
        lines.append("")
        for i, opp in enumerate(opportunities, 1):
            lines.append(f"{i}. {opp.product_name}")
            lines.append(f"   Confidence: {opp.confidence.value.upper()} ({opp.confidence_score:.2f})")
            lines.append(f"   Rationale: {opp.rationale[:100]}...")
            if opp.target_persona:
                lines.append(f"   Target: {opp.target_persona}")
            lines.append("")
    else:
        lines.append("OPPORTUNITIES FOUND: 0")
        lines.append("")

    # Risks
    risks = state.get('competitive_risks', [])
    if risks:
        lines.append(f"COMPETITIVE RISKS: {len(risks)}")
        for risk in risks[:3]:  # Show top 3
            lines.append(f"  - {risk}")
        if len(risks) > 3:
            lines.append(f"  ... and {len(risks) - 3} more")
        lines.append("")

    # Waiting for human?
    if state.get('waiting_for_human'):
        lines.append("STATUS: Waiting for human input")
        if state.get('human_question'):
            lines.append(f"Question: {state['human_question']}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_markdown_report(state: ResearchState) -> str:
    """
    Format research state as a comprehensive markdown report.

    Args:
        state: Research state to format

    Returns:
        Markdown-formatted report
    """
    lines = []

    # Header
    lines.append(f"# Enterprise Account Research: {state['account_name']}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Industry**: {state['industry']}")
    if state.get('region'):
        lines.append(f"**Region**: {state['region']}")
    lines.append(f"**Research Depth**: {state['research_depth'].value}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    opportunities = state.get('validated_opportunities', [])
    high_conf = [o for o in opportunities if o.confidence.value == 'high']
    lines.append(f"- **Opportunities Identified**: {len(opportunities)}")
    lines.append(f"- **High Confidence**: {len(high_conf)}")
    lines.append(f"- **Data Points Collected**: {len(state.get('signals', []))}")
    lines.append(f"- **Job Postings Analyzed**: {len(state.get('job_postings', []))}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Opportunities
    if opportunities:
        lines.append("## Opportunities")
        lines.append("")

        # Sort by confidence score
        sorted_opps = sorted(opportunities, key=lambda o: o.confidence_score, reverse=True)

        for i, opp in enumerate(sorted_opps, 1):
            lines.append(f"### {i}. {opp.product_name}")
            lines.append("")
            lines.append(f"**Confidence**: {opp.confidence.value.upper()} ({opp.confidence_score:.2%})")
            lines.append("")
            lines.append("**Rationale**:")
            lines.append(f"{opp.rationale}")
            lines.append("")

            if opp.target_persona:
                lines.append(f"**Target Persona**: {opp.target_persona}")
                lines.append("")

            if opp.talking_points:
                lines.append("**Talking Points**:")
                for point in opp.talking_points:
                    lines.append(f"- {point}")
                lines.append("")

            if opp.evidence:
                lines.append(f"**Supporting Evidence**: {len(opp.evidence)} signals")
                for signal in opp.evidence[:3]:  # Show top 3
                    lines.append(f"- [{signal.signal_type}] {signal.content[:100]}...")
                if len(opp.evidence) > 3:
                    lines.append(f"- ... and {len(opp.evidence) - 3} more signals")
                lines.append("")

            if opp.risks:
                lines.append("**Potential Risks**:")
                for risk in opp.risks:
                    lines.append(f"- {risk}")
                lines.append("")

            if opp.estimated_value:
                lines.append(f"**Estimated Value**: {opp.estimated_value}")
                lines.append("")

            lines.append("---")
            lines.append("")
    else:
        lines.append("## Opportunities")
        lines.append("")
        lines.append("No opportunities identified in this research.")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Competitive Risks
    risks = state.get('competitive_risks', [])
    if risks:
        lines.append("## Competitive Risks")
        lines.append("")
        for risk in risks:
            lines.append(f"- {risk}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Technology Stack
    tech_stack = state.get('tech_stack', [])
    if tech_stack:
        lines.append("## Technology Stack")
        lines.append("")
        lines.append("Technologies identified from job postings and signals:")
        lines.append("")
        for tech in sorted(tech_stack):
            lines.append(f"- {tech}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Data Sources Summary
    lines.append("## Research Methodology")
    lines.append("")
    lines.append(f"- Web search queries: {len([s for s in state.get('signals', []) if s.signal_type == 'web_search'])}")
    lines.append(f"- Job postings analyzed: {len(state.get('job_postings', []))}")
    lines.append(f"- News articles reviewed: {len(state.get('news_items', []))}")
    lines.append(f"- Total signals collected: {len(state.get('signals', []))}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by Enterprise Account Research System*")
    lines.append("")

    return "\n".join(lines)


def format_json_export(state: ResearchState) -> str:
    """
    Format research state as JSON export.

    Args:
        state: Research state to export

    Returns:
        JSON string
    """
    # Build exportable dict
    export_data = {
        "account_name": state['account_name'],
        "industry": state['industry'],
        "region": state.get('region'),
        "research_depth": state['research_depth'].value,
        "started_at": state['started_at'].isoformat(),
        "last_updated": state['last_updated'].isoformat(),
        "progress": {
            "coordinator_complete": state['progress'].coordinator_complete,
            "gatherer_complete": state['progress'].gatherer_complete,
            "identifier_complete": state['progress'].identifier_complete,
            "validator_complete": state['progress'].validator_complete
        },
        "data_collected": {
            "signals_count": len(state.get('signals', [])),
            "job_postings_count": len(state.get('job_postings', [])),
            "news_items_count": len(state.get('news_items', [])),
            "tech_stack_count": len(state.get('tech_stack', []))
        },
        "opportunities": [
            {
                "product_name": opp.product_name,
                "rationale": opp.rationale,
                "confidence": opp.confidence.value,
                "confidence_score": opp.confidence_score,
                "target_persona": opp.target_persona,
                "talking_points": opp.talking_points,
                "estimated_value": opp.estimated_value,
                "risks": opp.risks,
                "evidence_count": len(opp.evidence)
            }
            for opp in state.get('validated_opportunities', [])
        ],
        "competitive_risks": state.get('competitive_risks', []),
        "tech_stack": state.get('tech_stack', []),
        "waiting_for_human": state.get('waiting_for_human', False),
        "human_question": state.get('human_question')
    }

    return json.dumps(export_data, indent=2)


def format_opportunity_list(opportunities: list[Opportunity]) -> str:
    """
    Format a list of opportunities for quick display.

    Args:
        opportunities: List of opportunities

    Returns:
        Formatted string
    """
    if not opportunities:
        return "No opportunities found."

    lines = []
    sorted_opps = sorted(opportunities, key=lambda o: o.confidence_score, reverse=True)

    for i, opp in enumerate(sorted_opps, 1):
        conf_display = f"{opp.confidence.value.upper()} ({opp.confidence_score:.0%})"
        lines.append(f"{i}. {opp.product_name} - {conf_display}")
        lines.append(f"   {opp.rationale[:80]}...")

    return "\n".join(lines)


def format_progress_bar(state: ResearchState) -> str:
    """
    Format a simple text progress indicator.

    Args:
        state: Research state

    Returns:
        Progress bar string
    """
    progress = state['progress']

    steps = [
        ("Coordinator", progress.coordinator_complete),
        ("Gatherer", progress.gatherer_complete),
        ("Identifier", progress.identifier_complete),
        ("Validator", progress.validator_complete)
    ]

    bar = []
    for name, complete in steps:
        if complete:
            bar.append(f"[✓] {name}")
        else:
            bar.append(f"[ ] {name}")

    return " → ".join(bar)


def save_report(state: ResearchState, output_path: str, format: str = "markdown") -> None:
    """
    Save report to file.

    Args:
        state: Research state to save
        output_path: Path to save file
        format: Output format ('markdown' or 'json')
    """
    if format == "markdown":
        content = format_markdown_report(state)
    elif format == "json":
        content = format_json_export(state)
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
