"""
Output formatters for CLI.

Provides functions to format research results in different formats:
- Terminal output (rich formatting)
- Markdown reports
- JSON exports

Formatting limits are configurable via settings:
- report_max_evidence_signals: Max evidence signals per opportunity
- report_evidence_char_limit: Character limit per evidence signal
- report_rationale_char_limit: Character limit for rationale
- report_show_full_content: Show full content without truncation
"""
import json
from datetime import datetime
from typing import Any

from ..config import settings
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

        # Use configurable limit for rationale
        rationale_limit = (
            None if settings.report_show_full_content
            else settings.report_rationale_char_limit
        )

        for i, opp in enumerate(opportunities, 1):
            lines.append(f"{i}. {opp.product_name}")
            lines.append(f"   Confidence: {opp.confidence.value.upper()} ({opp.confidence_score:.2f})")

            # Format rationale with configurable truncation
            rationale_display = opp.rationale
            if rationale_limit and len(opp.rationale) > rationale_limit:
                rationale_display = opp.rationale[:rationale_limit] + "..."
            lines.append(f"   Rationale: {rationale_display}")

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
                # Use configurable limits for evidence display
                max_signals = (
                    len(opp.evidence) if settings.report_show_full_content
                    else settings.report_max_evidence_signals
                )
                char_limit = (
                    None if settings.report_show_full_content
                    else settings.report_evidence_char_limit
                )

                lines.append(f"**Supporting Evidence**: {len(opp.evidence)} signals")
                for signal in opp.evidence[:max_signals]:
                    # Format content with configurable truncation
                    content = signal.content
                    if char_limit and len(content) > char_limit:
                        content = content[:char_limit] + "..."
                    lines.append(f"- [{signal.signal_type}] {content}")

                    # Show additional metadata in full content mode
                    if settings.report_show_full_content and signal.metadata:
                        if "url" in signal.metadata:
                            lines.append(f"  - Source: {signal.metadata['url']}")
                        if "buying_signals" in signal.metadata:
                            bs = signal.metadata["buying_signals"]
                            if isinstance(bs, dict):
                                if bs.get("technologies"):
                                    techs = bs["technologies"][:5]
                                    lines.append(f"  - Technologies: {', '.join(techs)}")
                                if bs.get("hiring_for"):
                                    roles = bs["hiring_for"][:3]
                                    lines.append(f"  - Hiring for: {', '.join(roles)}")

                remaining = len(opp.evidence) - max_signals
                if remaining > 0:
                    lines.append(f"- ... and {remaining} more signals")
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

    # Data Collection Issues (if any)
    error_messages = state.get('error_messages', [])
    if error_messages:
        lines.append("## Data Collection Issues")
        lines.append("")
        lines.append("The following issues occurred during data collection:")
        lines.append("")
        for error in error_messages[:10]:  # Limit to first 10 errors
            lines.append(f"- {error}")
        if len(error_messages) > 10:
            lines.append(f"- ... and {len(error_messages) - 10} more issues")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Data Sources Summary
    lines.append("## Research Methodology")
    lines.append("")
    web_search_count = len([s for s in state.get('signals', []) if s.signal_type == 'web_search'])
    news_count = len(state.get('news_items', []))
    lines.append(f"- Web search queries: {web_search_count}")
    lines.append(f"- Job postings analyzed: {len(state.get('job_postings', []))}")
    lines.append(f"- News articles reviewed: {news_count}")
    lines.append(f"- Total signals collected: {len(state.get('signals', []))}")

    # Add warnings for missing data sources
    if web_search_count == 0:
        lines.append("- WARNING: No web search results - MCP search may have failed")
    if news_count == 0:
        lines.append("- WARNING: No news articles - news search may have failed")
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


def format_detailed_evidence(opportunities: list[Opportunity]) -> str:
    """
    Format detailed evidence for all opportunities.

    Shows full evidence content with all metadata. Useful for
    comprehensive analysis or debugging data quality issues.

    Args:
        opportunities: List of opportunities

    Returns:
        Markdown-formatted detailed evidence section
    """
    lines = []
    lines.append("# Detailed Evidence Report")
    lines.append("")

    for opp in opportunities:
        lines.append(f"## {opp.product_name}")
        lines.append("")
        lines.append(f"**Confidence**: {opp.confidence.value.upper()} ({opp.confidence_score:.2%})")
        lines.append(f"**Target**: {opp.target_persona or 'Not specified'}")
        lines.append("")

        if not opp.evidence:
            lines.append("*No supporting evidence collected.*")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        lines.append(f"### Evidence ({len(opp.evidence)} signals)")
        lines.append("")

        for i, signal in enumerate(opp.evidence, 1):
            lines.append(f"#### Signal {i}: {signal.signal_type}")
            lines.append("")
            lines.append(f"**Content**: {signal.content}")
            lines.append("")
            lines.append(f"- Confidence: {signal.confidence:.2f}")
            lines.append(f"- Source: {signal.source}")
            lines.append(f"- Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}")

            if signal.metadata:
                lines.append("- **Metadata**:")
                for key, value in signal.metadata.items():
                    if key in ("original_snippet",):  # Skip redundant fields
                        continue
                    if isinstance(value, dict):
                        lines.append(f"  - {key}:")
                        for k, v in value.items():
                            if v:
                                if isinstance(v, list):
                                    lines.append(f"    - {k}: {', '.join(str(x) for x in v[:5])}")
                                else:
                                    lines.append(f"    - {k}: {v}")
                    elif isinstance(value, list) and value:
                        lines.append(f"  - {key}: {', '.join(str(x) for x in value[:5])}")
                    elif value:
                        lines.append(f"  - {key}: {value}")

            lines.append("")

        lines.append("---")
        lines.append("")

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
