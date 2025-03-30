

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class NodeAnnotation(str, Enum):
    """Node annotations for optimization hints"""
    SPLITTABLE = "splittable"    # Node output can be split
    BATCHABLE = "batchable"      # Node can process batched inputs
    NONE = "none"                # No special optimization

class NodeOps(str, Enum):
    """Node operations"""
    INPUT = "input"
    OUTPUT = "output"
    EMBEDDING = "embedding"
    VECTORDB_INGESTION = "vectordb_ingestion" 
    VECTORDB_SEARCHING = "vectordb_searching"
    RERANKING = "reranking"
    LLM_PREFILLING = "llm_prefilling"
    LLM_DECODING = "llm_decoding"
    LLM_PARTIAL_PREFILLING = "llm_partial_prefilling" 
    LLM_FULL_PREFILLING = "llm_full_prefilling"
    LLM_PARTIAL_DECODING = "llm_parallel_decoding"
    AGGREGATOR = "aggregator" 
    


class NodeType(Enum):
    """Node types in DAG"""
    INPUT = "input"      # Input node that holds query inputs
    COMPUTE = "compute"  # Computation node that performs operations
    OUTPUT = "output"    # Output node that collects results


@dataclass
class NodeConfig:
    """Configuration for node execution"""
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # Add other configuration parameters as needed

@dataclass
class NodeIOSchema:
    """Define the input and output schema for the node"""
    input_format: Dict[str, type]  # input field name and type
    output_format: Dict[str, type] # output field name and type
    