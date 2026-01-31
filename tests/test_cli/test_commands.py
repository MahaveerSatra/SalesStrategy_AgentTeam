"""Tests for CLI commands."""
import os
import sqlite3
import tempfile
from unittest.mock import Mock, MagicMock, patch, call
import pytest

from src.cli.commands import (
    research_command,
    resume_command,
    list_runs_command,
    _run_with_human_loop,
    _resume_with_human_loop,
    _save_reports
)
from src.models.state import ResearchDepth
from .fixtures import (
    create_minimal_state,
    create_complete_state,
    create_paused_state
)


class TestResearchCommand:
    """Tests for research_command function."""

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._run_with_human_loop')
    @patch('builtins.print')
    def test_basic_research(self, mock_print, mock_run_loop, mock_workflow_class):
        """Test basic research command flow."""
        # Setup mocks
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow
        result_state = create_complete_state()
        mock_run_loop.return_value = result_state

        # Run command
        research_command(
            account_name="Boeing",
            industry="aerospace"
        )

        # Verify workflow created
        mock_workflow_class.assert_called_once()

        # Verify run_with_human_loop called
        mock_run_loop.assert_called_once()
        args = mock_run_loop.call_args[0]
        assert args[0] == mock_workflow
        assert args[1]['account_name'] == "Boeing"
        assert args[1]['industry'] == "aerospace"
        # Thread ID should be auto-generated
        thread_id = args[2]
        assert thread_id.startswith("research_Boeing_")

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._run_with_human_loop')
    @patch('src.cli.commands._save_reports')
    @patch('builtins.print')
    def test_research_with_output_dir(self, mock_print, mock_save, mock_run_loop, mock_workflow_class):
        """Test research with output directory specified."""
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow
        result_state = create_complete_state()
        result_state['waiting_for_human'] = False
        mock_run_loop.return_value = result_state

        research_command(
            account_name="Tesla",
            industry="automotive",
            output_dir="./reports"
        )

        # Verify save_reports was called
        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        assert args[0] == result_state
        assert args[1] == "./reports"

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.print')
    def test_invalid_research_depth(self, mock_print, mock_workflow_class):
        """Test handling of invalid research depth."""
        research_command(
            account_name="Boeing",
            industry="aerospace",
            research_depth="invalid"
        )

        # Should print error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Invalid research depth" in str(c) for c in print_calls)

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.print')
    def test_workflow_initialization_error(self, mock_print, mock_workflow_class):
        """Test handling of workflow initialization error."""
        mock_workflow_class.side_effect = Exception("Database connection failed")

        research_command(
            account_name="Boeing",
            industry="aerospace"
        )

        # Should print error
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error initializing workflow" in str(c) for c in print_calls)

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._run_with_human_loop')
    @patch('builtins.print')
    def test_custom_thread_id(self, mock_print, mock_run_loop, mock_workflow_class):
        """Test research with custom thread ID."""
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow
        result_state = create_complete_state()
        mock_run_loop.return_value = result_state

        research_command(
            account_name="Boeing",
            industry="aerospace",
            thread_id="custom_thread_123"
        )

        # Verify custom thread ID used
        args = mock_run_loop.call_args[0]
        assert args[2] == "custom_thread_123"

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._run_with_human_loop')
    @patch('builtins.print')
    def test_all_optional_parameters(self, mock_print, mock_run_loop, mock_workflow_class):
        """Test research with all optional parameters."""
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow
        result_state = create_complete_state()
        mock_run_loop.return_value = result_state

        research_command(
            account_name="Rivian",
            industry="automotive",
            region="North America",
            research_depth="deep",
            output_dir="./reports",
            thread_id="test_123"
        )

        # Verify state created with all parameters
        args = mock_run_loop.call_args[0]
        state = args[1]
        assert state['account_name'] == "Rivian"
        assert state['industry'] == "automotive"
        assert state['region'] == "North America"
        assert state['research_depth'] == ResearchDepth.DEEP

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._run_with_human_loop')
    @patch('builtins.print')
    def test_research_with_user_context(self, mock_print, mock_run_loop, mock_workflow_class):
        """Test research with user context for strategic advice."""
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow
        result_state = create_complete_state()
        mock_run_loop.return_value = result_state

        context = """Sales Objective: Prepare for Q1 QBR meeting
Relationship: Existing customer since 2020 - MATLAB + Simulink site license
Known Initiatives: Autonomous vehicle program launching Q2
Pain Points: Simulation too slow, need HIL testing
Competitive Threat: Ansys SCADE being evaluated"""

        research_command(
            account_name="Boeing",
            industry="aerospace",
            user_context=context
        )

        # Verify state created with context
        args = mock_run_loop.call_args[0]
        state = args[1]
        assert state['account_name'] == "Boeing"
        assert state['user_context'] == context

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._run_with_human_loop')
    @patch('builtins.print')
    def test_research_without_context_defaults_none(self, mock_print, mock_run_loop, mock_workflow_class):
        """Test that user_context defaults to None when not provided."""
        mock_workflow = Mock()
        mock_workflow_class.return_value = mock_workflow
        result_state = create_complete_state()
        mock_run_loop.return_value = result_state

        research_command(
            account_name="Tesla",
            industry="automotive"
        )

        # Verify state created without context
        args = mock_run_loop.call_args[0]
        state = args[1]
        assert state['user_context'] is None


class TestResumeCommand:
    """Tests for resume_command function."""

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.print')
    def test_thread_not_found(self, mock_print, mock_workflow_class):
        """Test resuming non-existent thread."""
        mock_workflow = Mock()
        mock_workflow.get_state.return_value = None
        mock_workflow_class.return_value = mock_workflow

        resume_command(thread_id="nonexistent_123")

        # Should print error
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("No research found" in str(c) for c in print_calls)

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.print')
    def test_not_paused(self, mock_print, mock_workflow_class):
        """Test resuming research that's not paused."""
        mock_workflow = Mock()
        state = create_complete_state()
        state['waiting_for_human'] = False
        mock_workflow.get_state.return_value = state
        mock_workflow_class.return_value = mock_workflow

        resume_command(thread_id="test_123")

        # Should indicate nothing to resume
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("not paused" in str(c) for c in print_calls)

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('src.cli.commands._resume_with_human_loop')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_resume_with_input(self, mock_print, mock_input, mock_resume_loop, mock_workflow_class):
        """Test successful resume with user input."""
        mock_workflow = Mock()
        paused_state = create_paused_state()
        mock_workflow.get_state.return_value = paused_state
        mock_workflow_class.return_value = mock_workflow

        completed_state = create_complete_state()
        completed_state['waiting_for_human'] = False
        mock_resume_loop.return_value = completed_state

        mock_input.return_value = "yes, continue"

        resume_command(thread_id="test_123")

        # Verify resume was called with input
        mock_resume_loop.assert_called_once()
        args = mock_resume_loop.call_args[0]
        assert args[0] == mock_workflow
        assert args[1] == "test_123"
        assert args[2] == "yes, continue"

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_cancel_resume(self, mock_print, mock_input, mock_workflow_class):
        """Test canceling resume."""
        mock_workflow = Mock()
        paused_state = create_paused_state()
        mock_workflow.get_state.return_value = paused_state
        mock_workflow_class.return_value = mock_workflow

        mock_input.return_value = "cancel"

        resume_command(thread_id="test_123")

        # Should print cancelled message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Cancelled" in str(c) for c in print_calls)

    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_empty_input(self, mock_print, mock_input, mock_workflow_class):
        """Test empty user input."""
        mock_workflow = Mock()
        paused_state = create_paused_state()
        mock_workflow.get_state.return_value = paused_state
        mock_workflow_class.return_value = mock_workflow

        mock_input.return_value = ""

        resume_command(thread_id="test_123")

        # Should print no input message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("No input provided" in str(c) for c in print_calls)


class TestListRunsCommand:
    """Tests for list_runs_command function."""

    @patch('src.cli.commands.settings')
    @patch('builtins.print')
    def test_no_database(self, mock_print, mock_settings):
        """Test when checkpoint database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.checkpoint_dir = tmpdir

            list_runs_command()

            # Should print no database message
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("No checkpoint database found" in str(c) for c in print_calls)

    @patch('src.cli.commands.settings')
    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.print')
    def test_empty_database(self, mock_print, mock_workflow_class, mock_settings):
        """Test when database exists but has no runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.checkpoint_dir = tmpdir
            db_path = os.path.join(tmpdir, "checkpoints.db")

            # Create empty database with table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE checkpoints (
                    thread_id TEXT,
                    checkpoint_id INTEGER
                )
            """)
            conn.commit()
            conn.close()

            list_runs_command()

            # Should print no runs message
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("No research runs found" in str(c) for c in print_calls)

    @patch('src.cli.commands.settings')
    @patch('src.cli.commands.ResearchWorkflow')
    @patch('builtins.print')
    def test_list_runs_with_data(self, mock_print, mock_workflow_class, mock_settings):
        """Test listing runs with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.checkpoint_dir = tmpdir
            db_path = os.path.join(tmpdir, "checkpoints.db")

            # Create database with test data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE checkpoints (
                    thread_id TEXT,
                    checkpoint_id INTEGER
                )
            """)
            cursor.execute("INSERT INTO checkpoints VALUES ('thread_1', 1)")
            cursor.execute("INSERT INTO checkpoints VALUES ('thread_2', 2)")
            conn.commit()
            conn.close()

            # Mock workflow to return states
            mock_workflow = Mock()
            state1 = create_complete_state()
            state1['account_name'] = "Company A"
            state2 = create_paused_state()
            state2['account_name'] = "Company B"

            mock_workflow.get_state.side_effect = [state1, state2]
            mock_workflow_class.return_value = mock_workflow

            list_runs_command()

            # Should display both runs
            print_calls = [str(call) for call in mock_print.call_args_list]
            output = ' '.join([str(c) for c in print_calls])
            assert "Company A" in output
            assert "Company B" in output
            assert "thread_1" in output
            assert "thread_2" in output

    @patch('src.cli.commands.settings')
    @patch('builtins.print')
    def test_database_without_checkpoints_table(self, mock_print, mock_settings):
        """Test database without checkpoints table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.checkpoint_dir = tmpdir
            db_path = os.path.join(tmpdir, "checkpoints.db")

            # Create empty database without table
            conn = sqlite3.connect(db_path)
            conn.close()

            list_runs_command()

            # Should print no table message
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("No checkpoint table found" in str(c) for c in print_calls)


class TestRunWithHumanLoop:
    """Tests for _run_with_human_loop helper."""

    def test_no_human_interaction_needed(self):
        """Test workflow that completes without human input."""
        mock_workflow = Mock()
        state = create_complete_state()
        state['waiting_for_human'] = False
        mock_workflow.run.return_value = state

        result = _run_with_human_loop(mock_workflow, state, "thread_123")

        assert result == state
        mock_workflow.run.assert_called_once_with(state, "thread_123")
        # Resume should not be called
        mock_workflow.resume.assert_not_called()

    @patch('builtins.input')
    @patch('builtins.print')
    def test_single_human_interaction(self, mock_print, mock_input):
        """Test workflow with single human input."""
        mock_workflow = Mock()

        # First returns waiting, then complete
        waiting_state = create_paused_state()
        complete_state = create_complete_state()
        complete_state['waiting_for_human'] = False
        mock_workflow.run.return_value = waiting_state
        mock_workflow.resume.return_value = complete_state

        mock_input.return_value = "yes, continue"

        result = _run_with_human_loop(mock_workflow, waiting_state, "thread_123")

        assert result == complete_state
        mock_workflow.resume.assert_called_once_with("thread_123", "yes, continue")

    @patch('builtins.input')
    @patch('builtins.print')
    def test_user_saves_workflow(self, mock_print, mock_input):
        """Test user saving workflow mid-execution."""
        mock_workflow = Mock()
        waiting_state = create_paused_state()
        mock_workflow.run.return_value = waiting_state

        mock_input.return_value = "save"

        result = _run_with_human_loop(mock_workflow, waiting_state, "thread_123")

        assert result['waiting_for_human'] is True
        # Resume should not be called
        mock_workflow.resume.assert_not_called()

    @patch('builtins.input')
    @patch('builtins.print')
    def test_max_iterations_protection(self, mock_print, mock_input):
        """Test that max iterations prevents infinite loops."""
        mock_workflow = Mock()
        waiting_state = create_paused_state()
        # Always return waiting state
        mock_workflow.run.return_value = waiting_state
        mock_workflow.resume.return_value = waiting_state

        mock_input.return_value = "continue"

        result = _run_with_human_loop(mock_workflow, waiting_state, "thread_123")

        # Should stop after 10 iterations
        assert mock_workflow.resume.call_count == 10


class TestResumeWithHumanLoop:
    """Tests for _resume_with_human_loop helper."""

    @patch('builtins.input')
    @patch('builtins.print')
    def test_resume_completes_immediately(self, mock_print, mock_input):
        """Test resume that completes after first input."""
        mock_workflow = Mock()
        complete_state = create_complete_state()
        complete_state['waiting_for_human'] = False
        mock_workflow.resume.return_value = complete_state

        result = _resume_with_human_loop(mock_workflow, "thread_123", "yes")

        assert result == complete_state
        mock_workflow.resume.assert_called_once_with("thread_123", "yes")

    @patch('builtins.input')
    @patch('builtins.print')
    def test_resume_with_additional_interactions(self, mock_print, mock_input):
        """Test resume with multiple interactions."""
        mock_workflow = Mock()
        waiting_state = create_paused_state()
        complete_state = create_complete_state()
        complete_state['waiting_for_human'] = False

        # First resume returns waiting, second returns complete
        mock_workflow.resume.side_effect = [waiting_state, complete_state]
        mock_input.return_value = "more details please"

        result = _resume_with_human_loop(mock_workflow, "thread_123", "initial input")

        assert result == complete_state
        assert mock_workflow.resume.call_count == 2


class TestSaveReports:
    """Tests for _save_reports helper."""

    def test_save_both_formats(self):
        """Test saving both markdown and JSON reports."""
        state = create_complete_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_reports(state, tmpdir, "test_thread_123")

            # Check both files created
            md_path = os.path.join(tmpdir, "test_thread_123_report.md")
            json_path = os.path.join(tmpdir, "test_thread_123_data.json")

            assert os.path.exists(md_path)
            assert os.path.exists(json_path)

            # Verify content
            with open(md_path, 'r') as f:
                md_content = f.read()
            assert "Test Company" in md_content

            with open(json_path, 'r') as f:
                import json
                json_data = json.load(f)
            assert json_data["account_name"] == "Test Company"

    def test_sanitize_thread_id(self):
        """Test that thread ID is sanitized for filenames."""
        state = create_complete_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use thread ID with invalid filename characters
            _save_reports(state, tmpdir, "thread/with\\slashes")

            # Should replace slashes with underscores
            md_path = os.path.join(tmpdir, "thread_with_slashes_report.md")
            assert os.path.exists(md_path)

    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        state = create_complete_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "reports")

            _save_reports(state, nested_dir, "test_123")

            # Directory should be created
            assert os.path.exists(nested_dir)
            # Files should exist
            md_path = os.path.join(nested_dir, "test_123_report.md")
            assert os.path.exists(md_path)
