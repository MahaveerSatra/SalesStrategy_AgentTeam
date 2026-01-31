"""
CLI command implementations.

Provides the core command logic for:
- Starting new research
- Resuming interrupted research
- Listing previous runs
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional
import structlog

from ..graph.workflow import ResearchWorkflow
from ..models.state import create_initial_state, ResearchDepth, ResearchState
from .formatters import (
    format_terminal_summary,
    format_markdown_report,
    format_progress_bar,
    save_report
)
from ..config import settings

logger = structlog.get_logger(__name__)


def research_command(
    account_name: str,
    industry: str,
    region: Optional[str] = None,
    research_depth: str = "standard",
    output_dir: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_context: Optional[str] = None,
    seller_name: str = "MathWorks"
) -> None:
    """
    Start a new research workflow.

    Args:
        account_name: Company name to research (the TARGET customer)
        industry: Industry vertical
        region: Geographic region (optional)
        research_depth: Research depth (quick/standard/deep)
        output_dir: Directory to save reports (optional)
        thread_id: Custom thread ID (optional, defaults to research_{account_name})
        user_context: Additional strategic context for the research (optional).
            If not provided and request is generic, CoordinatorAgent will ask
            clarifying questions to gather context for practical strategic advice.
        seller_name: Your company name (the SELLER). Products for this company
            must be indexed first using 'setup-catalog' command.
    """
    print(f"\n{'='*70}")
    print(f"Starting research for: {account_name}")
    print(f"{'='*70}\n")

    # Parse research depth
    try:
        depth_enum = ResearchDepth(research_depth.lower())
    except ValueError:
        print(f"Invalid research depth: {research_depth}")
        print("Valid options: quick, standard, deep")
        return

    # Create initial state
    initial_state = create_initial_state(
        account_name=account_name,
        industry=industry,
        region=region,
        user_context=user_context,
        research_depth=depth_enum
    )

    # Generate thread ID
    if not thread_id:
        # Sanitize account name for thread ID
        sanitized_name = account_name.replace(" ", "_").replace("/", "_")
        thread_id = f"research_{sanitized_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Thread ID: {thread_id}")
    print(f"Research Depth: {depth_enum.value}")
    print(f"Seller: {seller_name}")
    print()

    # Create workflow with seller's product catalog
    try:
        workflow = ResearchWorkflow(seller_name=seller_name)
    except ValueError as e:
        # Product catalog not indexed
        print(f"\nError: {e}")
        print(f"\nTo fix this, run:")
        print(f"  python -m src.cli setup-catalog --seller \"{seller_name}\"")
        return
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        logger.error("workflow_initialization_failed", error=str(e))
        return

    # Run workflow with human-in-loop handling
    try:
        result = _run_with_human_loop(workflow, initial_state, thread_id)
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
        print(f"Resume later with: python -m src.cli resume {thread_id}")
        return
    except Exception as e:
        print(f"\nError during research: {e}")
        logger.error("research_failed", error=str(e), thread_id=thread_id)
        return

    # Display results
    print("\n" + format_terminal_summary(result))

    # Save report if output directory specified
    if output_dir:
        _save_reports(result, output_dir, thread_id)

    # Show next steps
    if not result.get('waiting_for_human'):
        print(f"\n✓ Research complete!")
        print(f"  Thread ID: {thread_id}")
        if output_dir:
            print(f"  Reports saved to: {output_dir}")
    else:
        print(f"\n⏸  Research paused for feedback.")
        print(f"  Resume with: python -m src.cli resume {thread_id}")


def resume_command(
    thread_id: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Resume an interrupted research workflow.

    Args:
        thread_id: Thread ID to resume
        output_dir: Directory to save reports (optional)
    """
    print(f"\n{'='*70}")
    print(f"Resuming research: {thread_id}")
    print(f"{'='*70}\n")

    # Create workflow
    try:
        workflow = ResearchWorkflow()
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        logger.error("workflow_initialization_failed", error=str(e))
        return

    # Get current state
    try:
        current_state = workflow.get_state(thread_id)
        if not current_state:
            print(f"No research found with thread ID: {thread_id}")
            print("\nUse 'list-runs' to see available threads.")
            return
    except Exception as e:
        print(f"Error loading state: {e}")
        logger.error("state_load_failed", error=str(e), thread_id=thread_id)
        return

    # Show current status
    print(f"Account: {current_state['account_name']}")
    print(f"Progress: {format_progress_bar(current_state)}")
    print()

    # Check if waiting for human
    if not current_state.get('waiting_for_human'):
        print("Research is not paused. Nothing to resume.")
        print("\nCurrent status:")
        print(format_terminal_summary(current_state))
        return

    # Show question/report if available
    if current_state.get('human_question'):
        print("Question from system:")
        print(f"  {current_state['human_question']}")
        print()
    elif current_state.get('current_report'):
        print("Current Report:")
        print(current_state['current_report'])
        print()

    # Get human input
    print("Enter your response (or 'cancel' to exit):")
    try:
        user_input = input("> ").strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return

    if user_input.lower() == 'cancel':
        print("Cancelled.")
        return

    if not user_input:
        print("No input provided. Cancelled.")
        return

    # Resume with human input
    print(f"\nResuming with your feedback...")
    print()

    try:
        result = _resume_with_human_loop(workflow, thread_id, user_input)
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
        print(f"Resume later with: python -m src.cli resume {thread_id}")
        return
    except Exception as e:
        print(f"\nError during research: {e}")
        logger.error("resume_failed", error=str(e), thread_id=thread_id)
        return

    # Display results
    print("\n" + format_terminal_summary(result))

    # Save report if output directory specified
    if output_dir:
        _save_reports(result, output_dir, thread_id)

    # Show next steps
    if not result.get('waiting_for_human'):
        print(f"\n✓ Research complete!")
        print(f"  Thread ID: {thread_id}")
        if output_dir:
            print(f"  Reports saved to: {output_dir}")
    else:
        print(f"\n⏸  Research paused for feedback.")
        print(f"  Resume with: python -m src.cli resume {thread_id}")


def list_runs_command() -> None:
    """
    List all previous research runs from checkpoint database.
    """
    print(f"\n{'='*70}")
    print("Previous Research Runs")
    print(f"{'='*70}\n")

    # Get checkpoint database path
    db_path = os.path.join(settings.checkpoint_dir, "checkpoints.db")

    if not os.path.exists(db_path):
        print("No checkpoint database found.")
        print("Start a research with: python -m src.cli research <company> --industry <industry>")
        return

    # Query checkpoints
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='checkpoints'
        """)
        if not cursor.fetchone():
            conn.close()
            print("No checkpoint table found.")
            print("Start a research with: python -m src.cli research <company> --industry <industry>")
            return

        # Get distinct thread IDs (simpler query to avoid JSON parsing issues)
        cursor.execute("""
            SELECT DISTINCT thread_id
            FROM checkpoints
            ORDER BY checkpoint_id DESC
        """)

        thread_rows = cursor.fetchall()
        conn.close()

        if not thread_rows:
            print("No research runs found.")
            return

        # Display runs
        print(f"Found {len(thread_rows)} research run(s):\n")

        # Create workflow to fetch states
        try:
            workflow = ResearchWorkflow()
        except Exception as e:
            print(f"Error initializing workflow: {e}")
            return

        count = 0
        for (thread_id,) in thread_rows:
            count += 1

            # Try to get state for this thread
            try:
                state = workflow.get_state(thread_id)

                if state:
                    account_name = state.get('account_name', 'Unknown')
                    industry = state.get('industry', 'Unknown')
                    waiting = state.get('waiting_for_human', False)
                    started_at = state.get('started_at')

                    # Parse status
                    status = "⏸ Paused" if waiting else "✓ Complete"

                    # Format started time
                    try:
                        if started_at:
                            time_str = started_at.strftime('%Y-%m-%d %H:%M')
                        else:
                            time_str = "Unknown"
                    except:
                        time_str = "Unknown"

                    print(f"{count}. [{status}] {account_name}")
                    print(f"   Industry: {industry}")
                    print(f"   Started: {time_str}")
                    print(f"   Thread ID: {thread_id}")
                    print()
                else:
                    # State not found, just show thread ID
                    print(f"{count}. [?] Thread: {thread_id}")
                    print()

            except Exception as e:
                # Error getting state, just show thread ID
                print(f"{count}. [?] Thread: {thread_id} (error loading state)")
                print()

        print(f"Resume a run with: python -m src.cli resume <thread_id>")

    except Exception as e:
        print(f"Error reading checkpoint database: {e}")
        logger.error("checkpoint_read_failed", error=str(e))


# ─────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def _run_with_human_loop(
    workflow: ResearchWorkflow,
    state: ResearchState,
    thread_id: str
) -> ResearchState:
    """
    Run workflow with human-in-loop handling.

    Args:
        workflow: Workflow instance
        state: Initial state
        thread_id: Thread ID for checkpointing

    Returns:
        Final state (may be paused)
    """
    result = workflow.run(state, thread_id)

    # Handle human-in-loop iterations
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while result.get('waiting_for_human') and iteration < max_iterations:
        iteration += 1

        # Show question/report
        if result.get('human_question'):
            print(f"\n{'─'*70}")
            print("System Question:")
            print(f"  {result['human_question']}")
            print(f"{'─'*70}\n")
        elif result.get('current_report'):
            print(f"\n{'─'*70}")
            print("Report for Review:")
            print(result['current_report'])
            print(f"{'─'*70}\n")

        # Get human input
        print("Enter your response (or 'save' to pause):")
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nPaused. Resume with: python -m src.cli resume {thread_id}")
            return result

        if user_input.lower() == 'save':
            print(f"\nPaused. Resume with: python -m src.cli resume {thread_id}")
            return result

        if not user_input:
            print("No input provided. Please try again.")
            continue

        # Resume with input
        print(f"\n{format_progress_bar(result)}")
        print("Processing your feedback...\n")

        result = workflow.resume(thread_id, user_input)

    if iteration >= max_iterations:
        print("\nWarning: Maximum iterations reached. Stopping.")

    return result


def _resume_with_human_loop(
    workflow: ResearchWorkflow,
    thread_id: str,
    initial_input: str
) -> ResearchState:
    """
    Resume workflow with human-in-loop handling.

    Args:
        workflow: Workflow instance
        thread_id: Thread ID to resume
        initial_input: First human input

    Returns:
        Final state (may be paused)
    """
    result = workflow.resume(thread_id, initial_input)

    # Handle additional human-in-loop iterations
    max_iterations = 10
    iteration = 0

    while result.get('waiting_for_human') and iteration < max_iterations:
        iteration += 1

        # Show question/report
        if result.get('human_question'):
            print(f"\n{'─'*70}")
            print("System Question:")
            print(f"  {result['human_question']}")
            print(f"{'─'*70}\n")
        elif result.get('current_report'):
            print(f"\n{'─'*70}")
            print("Report for Review:")
            print(result['current_report'])
            print(f"{'─'*70}\n")

        # Get human input
        print("Enter your response (or 'save' to pause):")
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\nPaused. Resume with: python -m src.cli resume {thread_id}")
            return result

        if user_input.lower() == 'save':
            print(f"\nPaused. Resume with: python -m src.cli resume {thread_id}")
            return result

        if not user_input:
            print("No input provided. Please try again.")
            continue

        # Resume with input
        print(f"\n{format_progress_bar(result)}")
        print("Processing your feedback...\n")

        result = workflow.resume(thread_id, user_input)

    if iteration >= max_iterations:
        print("\nWarning: Maximum iterations reached. Stopping.")

    return result


def _save_reports(state: ResearchState, output_dir: str, thread_id: str) -> None:
    """
    Save markdown and JSON reports.

    Args:
        state: Research state
        output_dir: Output directory
        thread_id: Thread ID for filename
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize thread ID for filename
    safe_thread_id = thread_id.replace("/", "_").replace("\\", "_")

    # Save markdown report
    md_path = os.path.join(output_dir, f"{safe_thread_id}_report.md")
    try:
        save_report(state, md_path, format="markdown")
        print(f"  ✓ Markdown report: {md_path}")
    except Exception as e:
        print(f"  ✗ Failed to save markdown: {e}")
        logger.error("markdown_save_failed", error=str(e), path=md_path)

    # Save JSON export
    json_path = os.path.join(output_dir, f"{safe_thread_id}_data.json")
    try:
        save_report(state, json_path, format="json")
        print(f"  ✓ JSON export: {json_path}")
    except Exception as e:
        print(f"  ✗ Failed to save JSON: {e}")
        logger.error("json_save_failed", error=str(e), path=json_path)


def setup_catalog_command(
    seller_name: str,
    catalog_file: Optional[str] = None,
    catalog_url: Optional[str] = None,
    force: bool = False
) -> None:
    """
    Index product catalog for a seller company.

    Args:
        seller_name: Name of the seller company
        catalog_file: Optional path to catalog file (JSON, TXT, MD)
        catalog_url: Optional URL to scrape for products
        force: Force re-indexing even if catalog exists
    """
    import asyncio
    from ..data_sources.product_catalog import ProductCatalogIndexer, ProductMatcher

    print(f"\n{'='*70}")
    print(f"Setting up product catalog for: {seller_name}")
    print(f"{'='*70}\n")

    # Check if catalog already exists
    if not force:
        try:
            matcher = ProductMatcher(company_name=seller_name)
            product_count = matcher.collection.count()
            print(f"Catalog already exists with {product_count} products.")
            print(f"Use --force to re-index.")
            return
        except Exception:
            pass  # Catalog doesn't exist, proceed with indexing

    # Create indexer
    indexer = ProductCatalogIndexer(
        company_name=seller_name,
        catalog_file=catalog_file
    )

    async def run_indexing():
        # Load products from various sources
        if catalog_url:
            print(f"Fetching products from URL: {catalog_url}")
            products = await indexer.build_catalog_from_url(catalog_url)
        elif catalog_file and catalog_file.endswith('.json'):
            print(f"Loading products from JSON: {catalog_file}")
            products = await indexer.build_catalog()
        elif catalog_file:
            print(f"Extracting products from document: {catalog_file}")
            products = await indexer.build_catalog_from_document(catalog_file)
        else:
            print(f"Loading built-in catalog for {seller_name}...")
            products = await indexer.build_catalog()

        if not products:
            print(f"\nNo products found!")
            if seller_name.lower() != "mathworks":
                print(f"\nFor custom companies, provide a catalog source:")
                print(f"  --catalog-file <path>  : JSON or text file with products")
                print(f"  --catalog-url <url>    : Web page with product information")
            return

        print(f"Found {len(products)} products")

        # Index products
        print("Indexing products in ChromaDB...")
        await indexer.index_products(products)

        print(f"\n✓ Successfully indexed {len(products)} products for {seller_name}")
        print(f"  Collection: {indexer.collection_name}")
        print(f"  Database: {indexer.db_path}")

        # Show sample products
        print(f"\nSample products:")
        for product in products[:5]:
            print(f"  - {product.name} ({product.category})")
        if len(products) > 5:
            print(f"  ... and {len(products) - 5} more")

    # Run async indexing
    try:
        asyncio.run(run_indexing())
    except Exception as e:
        print(f"\nError during indexing: {e}")
        logger.error("catalog_indexing_failed", error=str(e), seller=seller_name)
