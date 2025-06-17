from enum import Enum
from typing import Any, Dict

import ray


class QueryStatus(Enum):
    """Query execution states"""

    INIT = "init"  # Query is initialized
    PENDING = "pending"  # Query is waiting to be processed
    RUNNING = "running"  # Query is being processed
    COMPLETED = "completed"  # Query completed successfully
    FAILED = "failed"  # Query failed during execution
    TIMEOUT = "timeout"  # Query exceeded time limit


@ray.remote
class QueryStates:
    """Ray Actor for managing query states and intermediate results

    Handles both:
    1. Global variables and intermediate results for DAG nodes
    2. Service-specific results and query states
    """

    def __init__(self):
        # For storing node-specific variables and results
        self.global_var_idx = {}

        # For storing query states and metadata
        self.states: Dict[str, Dict[str, Any]] = {}

        # For storing node-specific results
        self.node_results: Dict[str, Any] = {}

    def set_global_var(self, var, node_name):
        """Set global variable for a node"""
        self.global_var_idx[node_name] = var

    def get_global_var(self, node_name):
        """Get global variable for a node"""
        if node_name not in self.global_var_idx:
            return None
        return self.global_var_idx[node_name]

    def get_node_results(self):
        """Get all node results"""
        return self.node_results

    def set_node_result(self, node_name: str, result):
        """Set the result of a specific node"""
        if node_name not in self.node_results:
            self.node_results[node_name] = {}
        self.node_results[node_name] = result

    def get_node_result(self, node_name: str):
        """Get the result of a specific node"""
        if node_name not in self.node_results:
            return None
        return self.node_results[node_name]

    def clear_node_result(self, node_name: str):
        """Clear the result of a specific node"""
        if node_name in self.node_results:
            self.node_results.pop(node_name, None)

    def clear_query(self, query_id: str):
        """Clear all data related to a query"""
        # Clear query state
        if query_id in self.states:
            del self.states[query_id]

        # Clear any service results
        for service_results in self.service_results.values():
            service_results.pop(query_id, None)
