"""
Abstract base class for all research agents.
Staff-level: Clean abstraction with monitoring, error handling, and type safety.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any
from datetime import datetime
import time
import structlog

from ..models.state import ResearchState
from ..models.domain import AgentResult
from ..core.exceptions import AgentError

# Type variables for input/output
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')

logger = structlog.get_logger(__name__)


class BaseAgent(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for all research agents.
    
    Provides:
    - Standardized execution interface
    - Built-in monitoring and logging
    - Error handling patterns
    - Performance tracking
    """
    
    def __init__(self, name: str):
        """
        Initialize agent.
        
        Args:
            name: Unique identifier for this agent
        """
        self.name = name
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        self.logger = logger.bind(agent=name)
    
    @abstractmethod
    async def execute(self, state: ResearchState) -> TOutput:
        """
        Execute the agent's core logic.
        
        Args:
            state: Current research state
            
        Returns:
            Agent-specific output
            
        Raises:
            AgentError: If execution fails
        """
        pass
    
    @abstractmethod
    def get_complexity(self, state: ResearchState) -> int:
        """
        Determine task complexity for model routing.
        
        Args:
            state: Current research state
            
        Returns:
            Complexity score (1-10)
            1-3: Simple (local model)
            4-7: Medium (external 8B model)
            8-10: Complex (external 70B model)
        """
        pass
    
    async def execute_with_monitoring(
        self, 
        state: ResearchState
    ) -> AgentResult:
        """
        Execute agent with monitoring, error handling, and metrics.
        
        This is the public interface that should be called.
        Wraps the abstract execute() method.
        """
        start_time = time.time()
        self.execution_count += 1
        
        self.logger.info(
            "agent_started",
            execution_count=self.execution_count,
            account=state.get("account_name")
        )
        
        try:
            # Execute core logic
            output = await self.execute(state)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            self.logger.info(
                "agent_completed",
                execution_time=execution_time,
                success=True
            )
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=self._serialize_output(output),
                execution_time_seconds=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time
            
            self.logger.error(
                "agent_failed",
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time
            )
            
            # Re-raise as AgentError
            raise AgentError(
                agent_name=self.name,
                message=str(e),
                original_error=e
            )
    
    def _serialize_output(self, output: TOutput) -> dict[str, Any]:
        """
        Convert output to dictionary for AgentResult.
        Override if custom serialization needed.
        """
        if isinstance(output, dict):
            return output
        elif hasattr(output, 'model_dump'):
            # Pydantic model
            return output.model_dump()
        else:
            return {"result": str(output)}
    
    def get_metrics(self) -> dict[str, Any]:
        """Return performance metrics for this agent."""
        avg_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 
            else 0
        )
        
        return {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_time,
            "success_rate": (
                (self.execution_count - self.error_count) / self.execution_count
                if self.execution_count > 0
                else 0
            )
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class StatelessAgent(BaseAgent[ResearchState, ResearchState]):
    """
    Agent that updates state in-place.
    Most agents will inherit from this.
    """
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """
        Execute agent logic and return updated state.
        
        Default implementation calls process() and updates state.
        """
        await self.process(state)
        
        # Update last_updated timestamp
        state["last_updated"] = datetime.now()
        
        return state
    
    @abstractmethod
    async def process(self, state: ResearchState) -> None:
        """
        Process the state. Override this in subclasses.
        
        Modify state dict in-place. Don't return anything.
        """
        pass


# Example usage in a concrete agent:
"""
from src.core.base_agent import StatelessAgent
from src.models.state import ResearchState

class MyAgent(StatelessAgent):
    def __init__(self):
        super().__init__(name="my_agent")
    
    async def process(self, state: ResearchState) -> None:
        # Your logic here
        state["some_field"] = "some_value"
    
    def get_complexity(self, state: ResearchState) -> int:
        return 5  # Medium complexity
"""