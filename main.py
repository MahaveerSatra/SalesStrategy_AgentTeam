"""
Example usage of the account research system.
Use this to test Phase 1 foundation.

IMPORTANT: This file should be in the PROJECT ROOT
Location: SalesStrategy_AgentTeam/main.py
"""
import asyncio
from pathlib import Path

from src.models.state import create_initial_state, ResearchDepth
from src.graph.workflow import ResearchWorkflow
from src.core.model_router import router
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def test_model_router():
    """Test the model router with different complexity levels."""
    print("\n=== Testing Model Router ===\n")
    
    # Test 1: Simple task (local model)
    print("Test 1: Simple task (complexity=2)")
    try:
        response = await router.generate(
            prompt="What is 2+2? Answer in one word.",
            complexity=2,
            max_tokens=50
        )
        print(f"✓ Model: {response.model}")
        print(f"✓ Response: {response.content}")
        print(f"✓ Latency: {response.latency_ms:.0f}ms")
        print(f"✓ Cached: {response.cached}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Test 2: Medium task (external 8B model if API key configured)
    print("Test 2: Medium task (complexity=5)")
    try:
        response = await router.generate(
            prompt="List 3 key applications of simulation software in aerospace.",
            complexity=5,
            max_tokens=200
        )
        print(f"✓ Model: {response.model}")
        print(f"✓ Response: {response.content[:100]}...")
        print(f"✓ Latency: {response.latency_ms:.0f}ms")
    except Exception as e:
        print(f"✗ Error (might need GROQ_API_KEY): {e}")
    print()
    
    # Test 3: Test caching
    print("Test 3: Testing cache (repeat same request)")
    try:
        response = await router.generate(
            prompt="What is 2+2? Answer in one word.",
            complexity=2,
            max_tokens=50
        )
        print(f"✓ Cached: {response.cached}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Show metrics
    try:
        metrics = router.get_metrics()
        print("Router Metrics:")
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Cache hit rate: {metrics['cache_stats']['hit_rate']:.1%}")
        print(f"  Requests by model: {metrics['requests_by_model']}")
    except Exception as e:
        print(f"✗ Error getting metrics: {e}")
    print()


def test_workflow():
    """Test the basic workflow structure (SYNCHRONOUS)."""
    print("\n=== Testing Workflow ===\n")
    
    try:
        # Create initial state
        state = create_initial_state(
            account_name="Boeing",
            industry="aerospace",
            region="North America",
            user_context="Existing MATLAB customer, interested in autonomous systems",
            research_depth=ResearchDepth.STANDARD
        )
        
        print(f"✓ Initial State Created:")
        print(f"  Account: {state['account_name']}")
        print(f"  Industry: {state['industry']}")
        print(f"  Depth: {state['research_depth']}")
        print()
        
        # Create and run workflow
        workflow = ResearchWorkflow()
        
        print("Running workflow (placeholder nodes)...")
        # NOTE: No 'await' - this is synchronous now
        result = workflow.run(state, thread_id="test_boeing_001")
        
        print("\n✓ Workflow Complete!")
        print(f"  Progress: {result['progress'].get_completed_agents()}")
        print(f"  All complete: {result['progress'].is_complete()}")
        
    except Exception as e:
        print(f"✗ Workflow error: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_checkpoint_resume():
    """Test checkpoint and resume functionality (SYNCHRONOUS)."""
    print("\n=== Testing Checkpoint Resume ===\n")
    
    try:
        workflow = ResearchWorkflow()
        
        # Run initial workflow
        state = create_initial_state(
            account_name="Tesla",
            industry="automotive",
            research_depth=ResearchDepth.QUICK
        )
        
        thread_id = "test_tesla_001"
        print(f"✓ Starting workflow with thread_id: {thread_id}")
        
        # NOTE: No 'await' - synchronous
        result = workflow.run(state, thread_id=thread_id)
        
        print(f"✓ Initial run complete. Completed agents: {result['progress'].get_completed_agents()}")
        
        # Simulate resume
        print("\nSimulating resume from checkpoint...")
        resumed = workflow.resume(thread_id=thread_id)
        
        print(f"✓ Resumed successfully!")
        print(f"  Account: {resumed['account_name']}")
        print(f"  Progress maintained: {resumed['progress'].get_completed_agents()}")
        
    except Exception as e:
        print(f"✗ Checkpoint error: {e}")
        import traceback
        traceback.print_exc()
    print()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Enterprise Account Research System - Phase 1 Tests")
    print("=" * 60)
    
    try:
        # Test 1: Model Router (ASYNC)
        await test_model_router()
        
        # Test 2: Workflow (SYNC - no await)
        test_workflow()
        
        # Test 3: Checkpointing (SYNC - no await)
        test_checkpoint_resume()
        
        print("\n" + "=" * 60)
        print("✅ All Phase 1 tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. ✓ Verify models are working correctly")
        print("  2. ✓ Check data/checkpoints/ for SQLite database")
        print("  3. → Ready to integrate MCP tools")
        print("  4. → Ready to move to Phase 2: Data Layer")
        print("\nTroubleshooting:")
        print("  - If Ollama errors: Run 'ollama serve' in another terminal")
        print("  - If API errors: Add GROQ_API_KEY to .env file")
        print("  - Check logs in console or data/ directory")
        
    except Exception as e:
        logger.error("test_failed", error=str(e), error_type=type(e).__name__)
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())