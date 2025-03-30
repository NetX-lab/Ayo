from typing import Dict, List, Any
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema

class BaseModule:
    """
    Base class for all modules
    """
    
    def __init__(self, 
                 input_format: Dict[str, Any] = None,
                 output_format: Dict[str, Any] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the module
        
        Args:
            input_format: Input format definition
            output_format: Output format definition
            config: Module configuration parameters
        """
        self.input_format = input_format or {}
        self.output_format = output_format or {}
        self.config = config or {}

        self.pre_dependencies = []
        self.post_dependencies = []
        

    def __rshift__(self, other):
        self.post_dependencies.append(other) 
        other.pre_dependencies.append(self)
        return self
    
    
    def to_primitive_nodes(self) -> List[Node]:
        """
        Convert the module to a list of primitive nodes
        
        Returns:
            List[Node]: List of primitive nodes
        """
        raise NotImplementedError("Subclasses must implement the to_primitive_nodes method")
    
    def validate_io_schema(self) -> bool:
        """
        Validate the input and output format
        
        Returns:
            bool: Validation result
        """
        # Default implementation, subclasses can override this method for more detailed validation
        return len(self.input_format) > 0 and len(self.output_format) > 0
    
    
    def __str__(self) -> str:
        """Return the string representation of the module"""
        return f"{self.__class__.__name__}(input={self.input_format}, output={self.output_format})"
    
    def __repr__(self) -> str:
        """Return the detailed string representation of the module"""
        return f"{self.__class__.__name__}(input={self.input_format}, output={self.output_format}, config={self.config})" 