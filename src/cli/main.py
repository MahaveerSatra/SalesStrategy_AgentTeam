"""
Main CLI entry point.

Usage:
    python -m src.cli research "Boeing" --industry aerospace
    python -m src.cli resume <thread_id>
    python -m src.cli list-runs
"""
import argparse
import sys
from typing import Optional

from .commands import research_command, resume_command, list_runs_command


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Enterprise Account Research System - Multi-agent AI research tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new research
  python -m src.cli research "Boeing" --industry aerospace

  # Start with custom depth and region
  python -m src.cli research "Tesla" --industry automotive --region "North America" --depth deep

  # Resume interrupted research
  python -m src.cli resume research_Tesla_20260130_143022

  # List all previous runs
  python -m src.cli list-runs

  # Save reports to custom directory
  python -m src.cli research "Rivian" --industry automotive --output ./reports
        """
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # ─────────────────────────────────────────────────────────────────────
    # RESEARCH COMMAND
    # ─────────────────────────────────────────────────────────────────────
    research_parser = subparsers.add_parser(
        "research",
        help="Start new research on a company",
        description="Start a new research workflow for an enterprise account"
    )
    research_parser.add_argument(
        "account_name",
        type=str,
        help="Company name to research (e.g., 'Boeing', 'Tesla')"
    )
    research_parser.add_argument(
        "--industry",
        "-i",
        type=str,
        required=True,
        help="Industry vertical (e.g., 'aerospace', 'automotive')"
    )
    research_parser.add_argument(
        "--region",
        "-r",
        type=str,
        default=None,
        help="Geographic region (e.g., 'North America', 'Europe')"
    )
    research_parser.add_argument(
        "--depth",
        "-d",
        type=str,
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Research depth: quick (2-3 min), standard (3-5 min), deep (5-10 min)"
    )
    research_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for reports (markdown + JSON)"
    )
    research_parser.add_argument(
        "--thread-id",
        "-t",
        type=str,
        default=None,
        help="Custom thread ID (defaults to auto-generated)"
    )
    research_parser.add_argument(
        "--context",
        "-c",
        type=str,
        default=None,
        help="""Additional context for strategic research. Include details like:
- Sales objective (discovery, QBR, renewal)
- Relationship status (new prospect, existing customer)
- Current products they own
- Known initiatives or pain points
- Competitive threats
- Budget timing
If not provided and request is generic, system will ask clarifying questions."""
    )

    # ─────────────────────────────────────────────────────────────────────
    # RESUME COMMAND
    # ─────────────────────────────────────────────────────────────────────
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume interrupted research",
        description="Resume a paused research workflow by thread ID"
    )
    resume_parser.add_argument(
        "thread_id",
        type=str,
        help="Thread ID to resume (e.g., 'research_Boeing_20260130_143022')"
    )
    resume_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for reports (markdown + JSON)"
    )

    # ─────────────────────────────────────────────────────────────────────
    # LIST-RUNS COMMAND
    # ─────────────────────────────────────────────────────────────────────
    list_parser = subparsers.add_parser(
        "list-runs",
        help="List all previous research runs",
        description="Show all research runs stored in checkpoint database"
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "research":
            research_command(
                account_name=args.account_name,
                industry=args.industry,
                region=args.region,
                research_depth=args.depth,
                output_dir=args.output,
                thread_id=args.thread_id,
                user_context=args.context
            )
            return 0

        elif args.command == "resume":
            resume_command(
                thread_id=args.thread_id,
                output_dir=args.output
            )
            return 0

        elif args.command == "list-runs":
            list_runs_command()
            return 0

        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
