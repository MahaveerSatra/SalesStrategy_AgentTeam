"""
Main CLI entry point.

Usage:
    python -m src.cli research "Boeing" --industry aerospace
    python -m src.cli resume <thread_id>
    python -m src.cli list-runs
"""
import argparse
import io
import sys
from typing import Optional

from .commands import research_command, resume_command, list_runs_command, setup_catalog_command


def _configure_utf8_output():
    """
    Configure UTF-8 output for Windows console.

    Windows console may use cp1252 or other encodings that can't handle
    all Unicode characters. This ensures UTF-8 output with error handling.
    """
    if sys.platform == "win32":
        # Reconfigure stdout/stderr to use UTF-8 with error replacement
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


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
    research_parser.add_argument(
        "--seller",
        "-s",
        type=str,
        default="MathWorks",
        help="""Your company name (the seller). Default: MathWorks.
Products for this company must be indexed first using 'setup-catalog' command.
Example: --seller "Salesforce" """
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

    # ─────────────────────────────────────────────────────────────────────
    # SETUP-CATALOG COMMAND
    # ─────────────────────────────────────────────────────────────────────
    catalog_parser = subparsers.add_parser(
        "setup-catalog",
        help="Index product catalog for a seller company",
        description="""Index product catalog for semantic matching.

This must be run ONCE before using the research command with a seller.
Products can be loaded from:
  1. Built-in catalog (MathWorks has 147 products pre-configured)
  2. JSON file with product definitions
  3. Web page URL to scrape for products (uses LLM to extract)
  4. Text/Markdown document with product information
"""
    )
    catalog_parser.add_argument(
        "--seller",
        "-s",
        type=str,
        required=True,
        help="Seller company name (e.g., 'MathWorks', 'Salesforce')"
    )
    catalog_parser.add_argument(
        "--catalog-file",
        "-f",
        type=str,
        default=None,
        help="""Path to product catalog file. Supported formats:
- JSON: Array of product objects with name, category, description, key_features, use_cases, target_personas
- TXT/MD: Text description of products (will use LLM to extract structured data)"""
    )
    catalog_parser.add_argument(
        "--catalog-url",
        "-u",
        type=str,
        default=None,
        help="URL of product catalog page to scrape (e.g., company products page)"
    )
    catalog_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if catalog already exists"
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
    # Configure UTF-8 output for Windows console
    _configure_utf8_output()

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
                user_context=args.context,
                seller_name=args.seller
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

        elif args.command == "setup-catalog":
            setup_catalog_command(
                seller_name=args.seller,
                catalog_file=args.catalog_file,
                catalog_url=args.catalog_url,
                force=args.force
            )
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
