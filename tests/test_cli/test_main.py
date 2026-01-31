"""Tests for CLI main entry point."""
import sys
from unittest.mock import patch, Mock
import pytest

from src.cli.main import create_parser, main


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_created(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "python -m src.cli"

    def test_research_command_exists(self):
        """Test that research command is available."""
        parser = create_parser()
        args = parser.parse_args(['research', 'Boeing', '--industry', 'aerospace'])

        assert args.command == 'research'
        assert args.account_name == 'Boeing'
        assert args.industry == 'aerospace'

    def test_resume_command_exists(self):
        """Test that resume command is available."""
        parser = create_parser()
        args = parser.parse_args(['resume', 'thread_123'])

        assert args.command == 'resume'
        assert args.thread_id == 'thread_123'

    def test_list_runs_command_exists(self):
        """Test that list-runs command is available."""
        parser = create_parser()
        args = parser.parse_args(['list-runs'])

        assert args.command == 'list-runs'

    def test_research_required_arguments(self):
        """Test that research command requires account_name and industry."""
        parser = create_parser()

        # Missing industry should fail
        with pytest.raises(SystemExit):
            parser.parse_args(['research', 'Boeing'])

    def test_research_optional_arguments(self):
        """Test research command optional arguments."""
        parser = create_parser()
        args = parser.parse_args([
            'research', 'Tesla',
            '--industry', 'automotive',
            '--region', 'North America',
            '--depth', 'deep',
            '--output', './reports',
            '--thread-id', 'custom_123'
        ])

        assert args.account_name == 'Tesla'
        assert args.industry == 'automotive'
        assert args.region == 'North America'
        assert args.depth == 'deep'
        assert args.output == './reports'
        assert args.thread_id == 'custom_123'

    def test_research_depth_choices(self):
        """Test that research depth only accepts valid choices."""
        parser = create_parser()

        # Valid choices
        for depth in ['quick', 'standard', 'deep']:
            args = parser.parse_args(['research', 'Boeing', '--industry', 'aerospace', '--depth', depth])
            assert args.depth == depth

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            parser.parse_args(['research', 'Boeing', '--industry', 'aerospace', '--depth', 'invalid'])

    def test_research_short_flags(self):
        """Test that short flags work for research command."""
        parser = create_parser()
        args = parser.parse_args([
            'research', 'Boeing',
            '-i', 'aerospace',
            '-r', 'Global',
            '-d', 'quick',
            '-o', './out',
            '-t', 'test_123'
        ])

        assert args.industry == 'aerospace'
        assert args.region == 'Global'
        assert args.depth == 'quick'
        assert args.output == './out'
        assert args.thread_id == 'test_123'

    def test_research_context_flag(self):
        """Test that --context flag is available."""
        parser = create_parser()
        context = "Sales objective: Discovery call. Existing customer with MATLAB."
        args = parser.parse_args([
            'research', 'Boeing',
            '--industry', 'aerospace',
            '--context', context
        ])

        assert args.context == context

    def test_research_context_short_flag(self):
        """Test that -c short flag works for context."""
        parser = create_parser()
        context = "Preparing for QBR meeting."
        args = parser.parse_args([
            'research', 'Boeing',
            '-i', 'aerospace',
            '-c', context
        ])

        assert args.context == context

    def test_research_context_default_none(self):
        """Test that context defaults to None if not provided."""
        parser = create_parser()
        args = parser.parse_args([
            'research', 'Boeing',
            '--industry', 'aerospace'
        ])

        assert args.context is None

    def test_resume_optional_output(self):
        """Test resume command with optional output."""
        parser = create_parser()
        args = parser.parse_args(['resume', 'thread_123', '--output', './reports'])

        assert args.thread_id == 'thread_123'
        assert args.output == './reports'

    def test_no_command_fails(self):
        """Test that no command results in error."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_help_text_available(self):
        """Test that help text is generated."""
        parser = create_parser()
        help_text = parser.format_help()

        assert 'Enterprise Account Research System' in help_text
        assert 'research' in help_text
        assert 'resume' in help_text
        assert 'list-runs' in help_text

    def test_epilog_with_examples(self):
        """Test that epilog contains usage examples."""
        parser = create_parser()
        help_text = parser.format_help()

        assert 'Examples:' in help_text
        assert 'Boeing' in help_text
        assert 'Tesla' in help_text


class TestMain:
    """Tests for main function."""

    @patch('src.cli.main.research_command')
    def test_research_command_dispatch(self, mock_research_cmd):
        """Test that research command is dispatched correctly."""
        exit_code = main(['research', 'Boeing', '--industry', 'aerospace'])

        mock_research_cmd.assert_called_once_with(
            account_name='Boeing',
            industry='aerospace',
            region=None,
            research_depth='standard',
            output_dir=None,
            thread_id=None,
            user_context=None,
            seller_name='MathWorks'
        )
        assert exit_code == 0

    @patch('src.cli.main.research_command')
    def test_research_with_all_args(self, mock_research_cmd):
        """Test research command with all arguments."""
        exit_code = main([
            'research', 'Tesla',
            '--industry', 'automotive',
            '--region', 'NA',
            '--depth', 'deep',
            '--output', './out',
            '--thread-id', 'test_123'
        ])

        mock_research_cmd.assert_called_once_with(
            account_name='Tesla',
            industry='automotive',
            region='NA',
            research_depth='deep',
            output_dir='./out',
            thread_id='test_123',
            user_context=None,
            seller_name='MathWorks'
        )
        assert exit_code == 0

    @patch('src.cli.main.research_command')
    def test_research_with_context(self, mock_research_cmd):
        """Test research command with context argument."""
        context = """Sales Objective: QBR preparation
Relationship: Existing customer since 2020
Current Products: MATLAB, Simulink
Known Initiatives: Autonomous vehicle program"""

        exit_code = main([
            'research', 'Boeing',
            '--industry', 'aerospace',
            '--context', context
        ])

        mock_research_cmd.assert_called_once_with(
            account_name='Boeing',
            industry='aerospace',
            region=None,
            research_depth='standard',
            output_dir=None,
            thread_id=None,
            user_context=context,
            seller_name='MathWorks'
        )
        assert exit_code == 0

    @patch('src.cli.main.resume_command')
    def test_resume_command_dispatch(self, mock_resume_cmd):
        """Test that resume command is dispatched correctly."""
        exit_code = main(['resume', 'thread_123'])

        mock_resume_cmd.assert_called_once_with(
            thread_id='thread_123',
            output_dir=None
        )
        assert exit_code == 0

    @patch('src.cli.main.resume_command')
    def test_resume_with_output(self, mock_resume_cmd):
        """Test resume command with output directory."""
        exit_code = main(['resume', 'thread_123', '--output', './reports'])

        mock_resume_cmd.assert_called_once_with(
            thread_id='thread_123',
            output_dir='./reports'
        )
        assert exit_code == 0

    @patch('src.cli.main.list_runs_command')
    def test_list_runs_dispatch(self, mock_list_cmd):
        """Test that list-runs command is dispatched correctly."""
        exit_code = main(['list-runs'])

        mock_list_cmd.assert_called_once()
        assert exit_code == 0

    @patch('src.cli.main.research_command')
    @patch('builtins.print')
    def test_keyboard_interrupt_handling(self, mock_print, mock_research_cmd):
        """Test handling of KeyboardInterrupt."""
        mock_research_cmd.side_effect = KeyboardInterrupt()

        exit_code = main(['research', 'Boeing', '--industry', 'aerospace'])

        assert exit_code == 130  # Standard exit code for Ctrl+C
        # Should print interrupted message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Interrupted by user" in str(c) for c in print_calls)

    @patch('src.cli.main.research_command')
    @patch('builtins.print')
    def test_exception_handling(self, mock_print, mock_research_cmd):
        """Test handling of general exceptions."""
        mock_research_cmd.side_effect = Exception("Something went wrong")

        exit_code = main(['research', 'Boeing', '--industry', 'aerospace'])

        assert exit_code == 1
        # Should print error message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error:" in str(c) for c in print_calls)

    @patch('builtins.print')
    def test_invalid_command_handling(self, mock_print):
        """Test handling of invalid command."""
        # Note: argparse will handle this before we reach the command dispatch
        # This test ensures we have fallback logic

        # Mock the parser to simulate unknown command getting through
        with patch('src.cli.main.create_parser') as mock_create_parser:
            mock_parser = Mock()
            mock_args = Mock()
            mock_args.command = 'unknown_command'
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            exit_code = main(['unknown_command'])

            assert exit_code == 1

    @patch('src.cli.main.research_command')
    def test_main_with_none_argv(self, mock_research_cmd):
        """Test main with None argv (uses sys.argv)."""
        # Temporarily modify sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ['cli', 'research', 'Boeing', '--industry', 'aerospace']
            exit_code = main(None)

            mock_research_cmd.assert_called_once()
            assert exit_code == 0
        finally:
            sys.argv = original_argv

    @patch('src.cli.main.research_command')
    def test_success_returns_zero(self, mock_research_cmd):
        """Test that successful execution returns 0."""
        exit_code = main(['research', 'Boeing', '--industry', 'aerospace'])
        assert exit_code == 0

    @patch('src.cli.main.research_command')
    def test_exception_returns_one(self, mock_research_cmd):
        """Test that exceptions return exit code 1."""
        mock_research_cmd.side_effect = ValueError("Invalid input")
        exit_code = main(['research', 'Boeing', '--industry', 'aerospace'])
        assert exit_code == 1

    def test_invalid_arguments_exit(self):
        """Test that invalid arguments cause exit."""
        # Missing required argument
        with pytest.raises(SystemExit):
            main(['research', 'Boeing'])  # Missing --industry

    @patch('src.cli.main.research_command')
    @patch('builtins.print')
    def test_traceback_printed_on_error(self, mock_print, mock_research_cmd):
        """Test that traceback is printed on error."""
        mock_research_cmd.side_effect = RuntimeError("Test error")

        exit_code = main(['research', 'Boeing', '--industry', 'aerospace'])

        # Should print traceback
        assert exit_code == 1
        # Traceback should be printed (via traceback.print_exc())


class TestMainIntegration:
    """Integration tests for main entry point."""

    def test_help_flag(self):
        """Test that --help flag works."""
        with pytest.raises(SystemExit) as exc_info:
            main(['--help'])

        # Help should exit with 0
        assert exc_info.value.code == 0

    def test_research_help(self):
        """Test that research --help works."""
        with pytest.raises(SystemExit) as exc_info:
            main(['research', '--help'])

        assert exc_info.value.code == 0

    def test_resume_help(self):
        """Test that resume --help works."""
        with pytest.raises(SystemExit) as exc_info:
            main(['resume', '--help'])

        assert exc_info.value.code == 0

    def test_list_runs_help(self):
        """Test that list-runs --help works."""
        with pytest.raises(SystemExit) as exc_info:
            main(['list-runs', '--help'])

        assert exc_info.value.code == 0
