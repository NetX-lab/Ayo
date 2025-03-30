from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)

if TYPE_CHECKING:
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node

class OPT_Pass(ABC):
    """Base class for all optimization passes
    
    An optimization pass takes a DAG as input, performs specific optimizations,
    and returns the optimized DAG. Each pass should focus on a specific type
    of optimization (e.g., pruning dependencies, batching, splitting).
    """
    
    def __init__(self, name: str):
        """Initialize the optimization pass
        
        Args:
            name: Unique identifier for this optimization pass
        """
        self.name = name
        self.enabled = True
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def run(self, dag: 'DAG') -> 'DAG':
        """Execute the optimization pass on the given DAG
        
        Args:
            dag: Input DAG to optimize
            
        Returns:
            Optimized DAG
            
        This method must be implemented by all concrete optimization passes.
        """
        pass
    
    def configure(self, **kwargs) -> None:
        """Configure the optimization pass
        
        Args:
            **kwargs: Configuration parameters specific to this pass
        """
        self.config.update(kwargs)
    
    def enable(self) -> None:
        """Enable this optimization pass"""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this optimization pass"""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if this pass is enabled"""
        return self.enabled
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
        """
        return self.config.get(key, default)
    
    def validate_dag(self, dag: 'DAG') -> bool:
        """Validate DAG before optimization
        
        Args:
            dag: DAG to validate
            
        Returns:
            True if DAG is valid for this optimization
        """
        return True
    
    def get_applicable_nodes(self, dag: 'DAG') -> List['Node']:
        """Get nodes that this pass can optimize
        
        Args:
            dag: Input DAG
            
        Returns:
            List of nodes that can be optimized by this pass
        """
        return []
    
    def log_optimization(self, message: str) -> None:
        """Log optimization information
        
        Args:
            message: Message to log
        """
        logger.info(f"[{self.name}] {message}")
    
    def __str__(self) -> str:
        return f"OPT_Pass(name={self.name}, enabled={self.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()
