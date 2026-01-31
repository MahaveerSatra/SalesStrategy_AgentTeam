"""Tests for CLI formatters."""
import json
import os
import tempfile
from datetime import datetime
import pytest

from src.cli.formatters import (
    format_terminal_summary,
    format_markdown_report,
    format_json_export,
    format_opportunity_list,
    format_progress_bar,
    save_report
)
from src.models.state import OpportunityConfidence
from .fixtures import (
    create_minimal_state,
    create_complete_state,
    create_empty_opportunities_state,
    create_paused_state,
    create_partial_progress_state,
    create_state_with_risks
)


class TestFormatTerminalSummary:
    """Tests for format_terminal_summary function."""

    def test_minimal_state(self):
        """Test formatting minimal state."""
        state = create_minimal_state()
        result = format_terminal_summary(state)

        assert "Test Company" in result
        assert "technology" in result
        assert "STANDARD" in result.upper()
        assert "2026-01-30 10:00:00" in result
        assert "OPPORTUNITIES FOUND: 0" in result

    def test_complete_state(self):
        """Test formatting complete state with all data."""
        state = create_complete_state()
        result = format_terminal_summary(state)

        assert "Test Company" in result
        assert "automotive" in result
        assert "North America" in result
        assert "Signals: 3" in result
        assert "Job Postings: 2" in result
        assert "News Items: 1" in result
        assert "Tech Stack Items: 5" in result
        assert "OPPORTUNITIES FOUND: 2" in result
        assert "Automated Driving Toolbox" in result
        assert "Simulink" in result

    def test_shows_confidence_scores(self):
        """Test that confidence scores are displayed."""
        state = create_complete_state()
        result = format_terminal_summary(state)

        assert "HIGH" in result
        assert "0.85" in result
        assert "MEDIUM" in result
        assert "0.65" in result

    def test_shows_competitive_risks(self):
        """Test that competitive risks are displayed."""
        state = create_state_with_risks()
        result = format_terminal_summary(state)

        assert "COMPETITIVE RISKS: 5" in result
        assert "Competitor A" in result
        # Should show top 3
        assert "Budget frozen" in result
        assert "Recent executive" in result
        # Should indicate more exist
        assert "and 2 more" in result

    def test_shows_waiting_status(self):
        """Test that waiting for human status is displayed."""
        state = create_paused_state()
        result = format_terminal_summary(state)

        assert "Waiting for human input" in result
        assert "Should I gather more data" in result

    def test_shows_completed_agents(self):
        """Test that completed agents are listed."""
        state = create_complete_state()
        result = format_terminal_summary(state)

        assert "coordinator" in result
        assert "gatherer" in result
        assert "identifier" in result
        assert "validator" in result

    def test_empty_opportunities(self):
        """Test formatting when no opportunities found."""
        state = create_empty_opportunities_state()
        result = format_terminal_summary(state)

        assert "OPPORTUNITIES FOUND: 0" in result

    def test_partial_progress(self):
        """Test formatting with partial progress."""
        state = create_partial_progress_state()
        result = format_terminal_summary(state)

        assert "coordinator" in result
        assert "gatherer" in result
        # Should show only completed agents


class TestFormatMarkdownReport:
    """Tests for format_markdown_report function."""

    def test_report_structure(self):
        """Test basic markdown report structure."""
        state = create_complete_state()
        result = format_markdown_report(state)

        # Check headers
        assert "# Enterprise Account Research: Test Company" in result
        assert "## Executive Summary" in result
        assert "## Opportunities" in result
        assert "## Competitive Risks" in result
        assert "## Technology Stack" in result
        assert "## Research Methodology" in result

    def test_executive_summary_metrics(self):
        """Test executive summary contains correct metrics."""
        state = create_complete_state()
        result = format_markdown_report(state)

        assert "**Opportunities Identified**: 2" in result
        assert "**High Confidence**: 1" in result
        assert "**Data Points Collected**: 3" in result
        assert "**Job Postings Analyzed**: 2" in result

    def test_opportunities_sorted_by_confidence(self):
        """Test that opportunities are sorted by confidence score."""
        state = create_complete_state()
        result = format_markdown_report(state)

        # High confidence (0.85) should appear before medium (0.65)
        auto_driving_idx = result.index("Automated Driving Toolbox")
        simulink_idx = result.index("### 2. Simulink")
        assert auto_driving_idx < simulink_idx

    def test_opportunity_details(self):
        """Test that opportunity details are included."""
        state = create_complete_state()
        result = format_markdown_report(state)

        assert "**Confidence**: HIGH (85.00%)" in result
        assert "**Rationale**:" in result
        assert "**Target Persona**: VP of Engineering" in result
        assert "**Talking Points**:" in result
        assert "Accelerate autonomous driving" in result
        assert "**Estimated Value**: $500K-$1M" in result
        assert "**Potential Risks**:" in result

    def test_supporting_evidence(self):
        """Test that supporting evidence is shown."""
        state = create_complete_state()
        result = format_markdown_report(state)

        assert "**Supporting Evidence**: 2 signals" in result
        assert "[hiring]" in result or "[product_launch]" in result

    def test_competitive_risks_section(self):
        """Test competitive risks section."""
        state = create_state_with_risks()
        result = format_markdown_report(state)

        assert "## Competitive Risks" in result
        assert "Competitor A has 5-year partnership" in result
        assert "Budget frozen until Q3" in result

    def test_technology_stack_section(self):
        """Test technology stack section."""
        state = create_complete_state()
        result = format_markdown_report(state)

        assert "## Technology Stack" in result
        assert "- MATLAB" in result
        assert "- Python" in result
        assert "- Simulink" in result

    def test_research_methodology(self):
        """Test research methodology section."""
        state = create_complete_state()
        result = format_markdown_report(state)

        assert "## Research Methodology" in result
        assert "- Job postings analyzed: 2" in result
        assert "- News articles reviewed: 1" in result
        assert "- Total signals collected: 3" in result

    def test_empty_opportunities_message(self):
        """Test message when no opportunities found."""
        state = create_empty_opportunities_state()
        result = format_markdown_report(state)

        assert "No opportunities identified in this research." in result

    def test_footer(self):
        """Test report footer."""
        state = create_complete_state()
        result = format_markdown_report(state)

        assert "Generated by Enterprise Account Research System" in result


class TestFormatJsonExport:
    """Tests for format_json_export function."""

    def test_valid_json(self):
        """Test that output is valid JSON."""
        state = create_complete_state()
        result = format_json_export(state)

        # Should parse without error
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_basic_fields(self):
        """Test that basic fields are included."""
        state = create_complete_state()
        result = format_json_export(state)
        data = json.loads(result)

        assert data["account_name"] == "Test Company"
        assert data["industry"] == "automotive"
        assert data["region"] == "North America"
        assert data["research_depth"] == "deep"

    def test_timestamps_iso_format(self):
        """Test that timestamps are in ISO format."""
        state = create_complete_state()
        result = format_json_export(state)
        data = json.loads(result)

        # Should be ISO format strings
        assert "T" in data["started_at"]
        assert "T" in data["last_updated"]
        # Should be parseable
        datetime.fromisoformat(data["started_at"])

    def test_progress_tracking(self):
        """Test that progress is included."""
        state = create_complete_state()
        result = format_json_export(state)
        data = json.loads(result)

        assert data["progress"]["coordinator_complete"] is True
        assert data["progress"]["gatherer_complete"] is True
        assert data["progress"]["identifier_complete"] is True
        assert data["progress"]["validator_complete"] is True

    def test_data_collected_counts(self):
        """Test that data collection counts are correct."""
        state = create_complete_state()
        result = format_json_export(state)
        data = json.loads(result)

        assert data["data_collected"]["signals_count"] == 3
        assert data["data_collected"]["job_postings_count"] == 2
        assert data["data_collected"]["news_items_count"] == 1
        assert data["data_collected"]["tech_stack_count"] == 5

    def test_opportunities_structure(self):
        """Test that opportunities are properly structured."""
        state = create_complete_state()
        result = format_json_export(state)
        data = json.loads(result)

        assert len(data["opportunities"]) == 2
        opp = data["opportunities"][0]
        assert "product_name" in opp
        assert "rationale" in opp
        assert "confidence" in opp
        assert "confidence_score" in opp
        assert "target_persona" in opp
        assert "talking_points" in opp
        assert "estimated_value" in opp
        assert "risks" in opp
        assert "evidence_count" in opp

    def test_competitive_risks(self):
        """Test that competitive risks are included."""
        state = create_state_with_risks()
        result = format_json_export(state)
        data = json.loads(result)

        assert len(data["competitive_risks"]) == 5

    def test_tech_stack(self):
        """Test that tech stack is included."""
        state = create_complete_state()
        result = format_json_export(state)
        data = json.loads(result)

        assert "Python" in data["tech_stack"]
        assert "MATLAB" in data["tech_stack"]

    def test_waiting_for_human_flag(self):
        """Test that waiting_for_human flag is included."""
        state = create_paused_state()
        result = format_json_export(state)
        data = json.loads(result)

        assert data["waiting_for_human"] is True
        assert data["human_question"] == "Should I gather more data about their tech stack?"


class TestFormatOpportunityList:
    """Tests for format_opportunity_list function."""

    def test_empty_list(self):
        """Test formatting empty opportunity list."""
        result = format_opportunity_list([])
        assert result == "No opportunities found."

    def test_single_opportunity(self):
        """Test formatting single opportunity."""
        state = create_complete_state()
        opps = state['validated_opportunities'][:1]
        result = format_opportunity_list(opps)

        assert "1. Automated Driving Toolbox" in result
        assert "HIGH (85%)" in result

    def test_multiple_opportunities(self):
        """Test formatting multiple opportunities."""
        state = create_complete_state()
        opps = state['validated_opportunities']
        result = format_opportunity_list(opps)

        assert "1. Automated Driving Toolbox" in result
        assert "2. Simulink" in result

    def test_sorted_by_confidence(self):
        """Test that opportunities are sorted by confidence."""
        state = create_complete_state()
        opps = state['validated_opportunities']
        result = format_opportunity_list(opps)

        # High confidence should be first
        lines = result.split('\n')
        assert "Automated Driving Toolbox" in lines[0]

    def test_shows_rationale_truncated(self):
        """Test that rationale is truncated."""
        state = create_complete_state()
        opps = state['validated_opportunities']
        result = format_opportunity_list(opps)

        # Should truncate at 80 chars
        assert "..." in result


class TestFormatProgressBar:
    """Tests for format_progress_bar function."""

    def test_no_progress(self):
        """Test progress bar with no agents complete."""
        state = create_minimal_state()
        result = format_progress_bar(state)

        assert "[ ] Coordinator" in result
        assert "[ ] Gatherer" in result
        assert "[ ] Identifier" in result
        assert "[ ] Validator" in result

    def test_partial_progress(self):
        """Test progress bar with partial completion."""
        state = create_partial_progress_state()
        result = format_progress_bar(state)

        assert "[✓] Coordinator" in result
        assert "[✓] Gatherer" in result
        assert "[ ] Identifier" in result
        assert "[ ] Validator" in result

    def test_complete_progress(self):
        """Test progress bar with all agents complete."""
        state = create_complete_state()
        result = format_progress_bar(state)

        assert "[✓] Coordinator" in result
        assert "[✓] Gatherer" in result
        assert "[✓] Identifier" in result
        assert "[✓] Validator" in result

    def test_arrow_separator(self):
        """Test that steps are separated by arrows."""
        state = create_complete_state()
        result = format_progress_bar(state)

        assert "→" in result


class TestSaveReport:
    """Tests for save_report function."""

    def test_save_markdown(self):
        """Test saving markdown report."""
        state = create_complete_state()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name

        try:
            save_report(state, temp_path, format="markdown")

            # Check file was created
            assert os.path.exists(temp_path)

            # Check content
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert "# Enterprise Account Research: Test Company" in content
            assert "## Opportunities" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_json(self):
        """Test saving JSON report."""
        state = create_complete_state()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            save_report(state, temp_path, format="json")

            # Check file was created
            assert os.path.exists(temp_path)

            # Check content is valid JSON
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data["account_name"] == "Test Company"
            assert len(data["opportunities"]) == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        state = create_complete_state()

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unknown format"):
                save_report(state, temp_path, format="xml")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_creates_utf8_encoding(self):
        """Test that files are saved with UTF-8 encoding."""
        state = create_complete_state()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name

        try:
            save_report(state, temp_path, format="markdown")

            # Read with UTF-8 encoding
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Should not raise encoding errors
            assert len(content) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
