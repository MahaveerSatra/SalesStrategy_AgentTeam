"""
Example usage of the account research system.
Use this to test Phase 1 foundation.
"""
import asyncio
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
    response = await router.generate(
        prompt="What is 2+2? Answer in one word.",
        complexity=2,
        max_tokens=50
    )
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Cached: {response.cached}")
    print()
    
    # Test 2: Medium task (external 8B model)
    print("Test 2: Medium task (complexity=5)")
    response = await router.generate(
        prompt="List 3 key applications of MathWorks simulation software in aerospace.",
        complexity=5,
        max_tokens=200
    )
    print(f"Model: {response.model}")
    print(f"Response: {response.content[:100]}...")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print()
    
    # Test 3: Test caching
    print("Test 3: Testing cache (repeat same request)")
    response = await router.generate(
        prompt="What is 2+2? Answer in one word.",
        complexity=2,
        max_tokens=50
    )
    print(f"Cached: {response.cached}")
    print()
    
    # Show metrics
    metrics = router.get_metrics()
    print("Router Metrics:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Cache hit rate: {metrics['cache_stats']['hit_rate']:.1%}")
    print(f"  Requests by model: {metrics['requests_by_model']}")
    print()


async def test_workflow():
    """Test the basic workflow structure."""
    print("\n=== Testing Workflow ===\n")
    
    # Create initial state
    state = create_initial_state(
        account_name="Boeing",
        industry="aerospace",
        region="North America",
        user_context="Existing MathWorks customer, interested in safety systems",
        research_depth=ResearchDepth.STANDARD
    )
    
    print(f"Initial State Created:")
    print(f"  Account: {state['account_name']}")
    print(f"  Industry: {state['industry']}")
    print(f"  Depth: {state['research_depth']}")
    print()
    
    # Create and run workflow
    workflow = ResearchWorkflow()
    
    print("Running workflow (placeholder nodes)...")
    result = await workflow.run(state, thread_id="test_boeing_001")
    
    print("\nWorkflow Complete!")
    print(f"  Progress: {result['progress'].get_completed_agents()}")
    print(f"  All complete: {result['progress'].is_complete()}")
    print()


async def test_checkpoint_resume():
    """Test checkpoint and resume functionality."""
    print("\n=== Testing Checkpoint Resume ===\n")
    
    workflow = ResearchWorkflow()
    
    # Run initial workflow
    state = create_initial_state(
        account_name="Tesla",
        industry="automotive",
        research_depth=ResearchDepth.QUICK
    )
    
    thread_id = "test_tesla_001"
    print(f"Starting workflow with thread_id: {thread_id}")
    
    result = await workflow.run(state, thread_id=thread_id)
    
    print(f"Initial run complete. Completed agents: {result['progress'].get_completed_agents()}")
    
    # Simulate resume (in practice, you'd restart the app)
    print("\nSimulating resume from checkpoint...")
    resumed = await workflow.resume(thread_id=thread_id)
    
    print(f"Resumed successfully!")
    print(f"  Account: {resumed['account_name']}")
    print(f"  Progress maintained: {resumed['progress'].get_completed_agents()}")
    print()


async def main():
    """Run all tests."""
    try:
        # Test 1: Model Router
        await test_model_router()
        
        # Test 2: Workflow
        await test_workflow()
        
        # Test 3: Checkpointing
        await test_checkpoint_resume()
        
        print("\n✅ All Phase 1 tests passed!")
        print("\nNext steps:")
        print("  1. Verify models are working correctly")
        print("  2. Check data/checkpoints/ for SQLite database")
        print("  3. Ready to move to Phase 2: Data Layer")
        
    except Exception as e:
        logger.error("test_failed", error=str(e), error_type=type(e).__name__)
        raise


if __name__ == "__main__":
    asyncio.run(main())