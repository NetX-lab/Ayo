import uuid
from collections import deque
from typing import Any, Dict, List, Optional

import ray

from Ayo.dags.node import Node, NodeIOSchema
from Ayo.dags.node_commons import NodeStatus, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.opt_pass.base_pass import OPT_Pass

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class DAG:
    def __init__(
        self,
        dag_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """initialize DAG

        Args:
            dag_id: DAG's unique identifier, if not provided, it will be generated automatically
            name: DAG's name
            description: DAG's description
            metadata: DAG's metadata
        """
        # basic attributes
        self.id = dag_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.metadata = metadata or {}

        # node related
        self.nodes: List[Node] = []
        self.node_map: Dict[str, Node] = {}
        self.in_degree: Dict[Node, int] = {}
        self.topo_list: List[Node] = []

        # input and output related
        self.input_nodes: Dict[str, Node] = {}
        self.output_nodes: Dict[str, Node] = {}
        self.query_id: Optional[str] = None
        self._query_inputs: Dict[str, Any] = {}

        # execution status
        self.is_completed = False
        self.error_nodes: List[Node] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.execution_stats: Dict[str, Any] = {}

        # Flag to track if topology needs to be recalculated
        self._topo_dirty = True

    def get_attr(self, key: str) -> Any:
        """get attribute of the DAG"""
        return getattr(self, key)

    def set_attr(self, key: str, value: Any) -> None:
        """set attribute of the DAG"""
        setattr(self, key, value)

    def from_chain(self, *nodes: Node) -> None:
        """build DAG from node chain"""
        self.nodes.clear()

        for node in nodes:
            node.clear_dependencies()

        for i in range(len(nodes) - 1):
            nodes[i] >> nodes[i + 1]

        self.nodes.clear()
        self.nodes.extend(nodes)

    def register_nodes(self, *nodes: Node) -> None:
        """Register nodes to the DAG"""

        self.nodes.clear()
        self.node_map.clear()
        self.in_degree.clear()
        self.topo_list.clear()

        # check the uniqueness of node names
        node_names = set()
        for node in nodes:
            if node.name in node_names:
                raise ValueError(f"Duplicate node name found: {node.name}")
            node_names.add(node.name)

        self.nodes.extend(nodes)
        self.node_map.update({node.name: node for node in nodes})

        # Create input nodes if query inputs are available
        if self._query_inputs:
            self.create_input_nodes()

        # Create output nodes for leaf nodes
        self.create_output_nodes()

        self.in_degree = {node: len(node.parents) for node in self.nodes}
        self._mark_topo_dirty()

    @property
    def query_inputs(self) -> Dict[str, Any]:
        """Get query inputs

        Returns:
            Dict[str, Any]: Query inputs dictionary
        """
        return self._query_inputs

    def set_query_inputs(self, inputs: Dict[str, Any]) -> None:
        """Set query inputs and update related nodes

        Args:
            inputs: Query inputs dictionary
        """
        if not isinstance(inputs, dict):
            raise TypeError("Query inputs must be a dictionary")

        self._query_inputs = inputs

        # If there are already registered nodes, update input nodes
        if self.nodes:
            self.create_input_nodes()

    def create_input_nodes(self) -> None:
        """Create input nodes for query inputs and establish connections"""
        if not self._query_inputs:
            return

        for input_name, input_value in self._query_inputs.items():
            # check if the input node with the same name already exists
            input_node_name = f"input_{input_name}"
            if input_node_name in self.node_map:
                continue

            input_node = Node(
                name=input_node_name,
                engine_type=EngineType.INPUT,
                io_schema=NodeIOSchema(
                    input_format={input_name: type(input_value)},
                    output_format={input_name: type(input_value)},
                ),
                node_type=NodeType.INPUT,
            )

            input_node.input_values[input_name] = input_value
            input_node.input_kwargs[input_name] = input_value
            self.input_nodes[input_node_name] = input_node
            self.nodes.append(input_node)
            self.node_map[input_node_name] = input_node

            # Find nodes that need this input
            for node in self.nodes:
                if (
                    node.node_type == NodeType.COMPUTE
                    and input_name in node.input_kwargs
                ):
                    input_node.add_child(node)

        # Update in_degree after adding connections
        self.in_degree = {node: len(node.parents) for node in self.nodes}

    def create_output_nodes(self) -> None:
        """Create output nodes for leaf nodes in the DAG"""
        leaf_nodes = [
            node
            for node in self.nodes
            if not node.children and node.node_type == NodeType.COMPUTE
        ]

        for leaf_node in leaf_nodes:
            for output_name in leaf_node.output_names:
                # check if the output node with the same name already exists
                output_node_name = f"output_{output_name}"
                if output_node_name in self.node_map:
                    continue

                output_node = Node(
                    name=output_node_name,
                    engine_type=EngineType.OUTPUT,
                    io_schema=NodeIOSchema(
                        input_format={output_name: None}, output_format={}
                    ),
                    node_type=NodeType.OUTPUT,
                )
                self.output_nodes[output_name] = output_node
                self.nodes.append(output_node)
                self.node_map[output_node_name] = output_node

                # connect the leaf node to the output node
                leaf_node.add_child(output_node)

        # update in_degree after adding connections
        self.in_degree = {node: len(node.parents) for node in self.nodes}

    def topological_sort(self) -> List[Node]:
        """Perform topological sort on the DAG

        Returns:
            List[Node]: Sorted nodes in topological order

        Raises:
            ValueError: If a cycle is detected in the DAG
        """
        # copy in_degree dictionary to avoid modifying original data

        # recompuate the in_degree
        self.in_degree = {node: len(node.parents) for node in self.nodes}

        in_degree = self.in_degree.copy()

        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        sorted_list = []

        # for detecting cycle
        visited_edges = set()

        while queue:
            node = queue.popleft()
            sorted_list.append(node)

            for child in node.children:
                # record visited edges
                edge = (node.name, child.name)
                visited_edges.add(edge)

                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_list) != len(self.nodes):
            # find nodes involved in cycle
            remaining_nodes = set(self.nodes) - set(sorted_list)
            cycle_nodes = [node.name for node in remaining_nodes]

            # build detailed error information
            error_msg = (
                f"Cycle detected in DAG. The following nodes are involved in a cycle: "
                f"{', '.join(cycle_nodes)}. "
                f"Total nodes: {len(self.nodes)}, Sorted nodes: {len(sorted_list)}"
            )
            raise ValueError(error_msg)

        # update the topological sort result of the class
        self.topo_list = sorted_list

        # Update node depth, the deeper the node, the smaller the depth
        total_nodes = len(sorted_list)
        for idx, node in enumerate(sorted_list):
            node.depth = total_nodes - idx - 1

        # verify the correctness of the sorted result
        node_indices = {node: idx for idx, node in enumerate(sorted_list)}
        for node in sorted_list:
            for child in node.children:
                if node_indices[child] <= node_indices[node]:
                    raise ValueError(
                        f"Invalid topological sort: Node '{node.name}' appears before "
                        f"its child '{child.name}'"
                    )

        return sorted_list

    def optimize(self, passes: List[OPT_Pass]) -> "DAG":
        """Apply optimization passes to the DAG

        Args:
            passes: List of optimization passes to apply

        Returns:
            Optimized DAG
        """

        logger.info(f"Optimizing DAG for query {self.query_id} with passes {passes}")

        self._ensure_topo_sort()  # Ensure topology is up to date

        for opt_pass in passes:
            if not opt_pass.is_enabled():
                continue

            try:
                opt_pass.log_optimization("Starting optimization")
                opt_pass.run(self)
                opt_pass.log_optimization("Optimization completed")
            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                opt_pass.log_optimization(
                    f"Optimization failed: {str(e)}\nStack trace: {stack_trace}"
                )
                continue

            self._mark_topo_dirty()

        return self

    def get_ready_nodes(self) -> List[Node]:
        """Get ready nodes"""
        return [node for node in self.nodes if node.is_ready]

    def is_failed(self) -> bool:
        """Check if DAG execution failed"""
        return bool(self.error_nodes)

    def check_completion(self) -> bool:
        """Check if DAG execution is completed"""
        all_completed = all(node.status == NodeStatus.COMPLETED for node in self.nodes)
        self.is_completed = all_completed
        return all_completed

    def __str__(self) -> str:
        self._ensure_topo_sort()  # Ensure topology is up to date
        return f"\nDAG({self.id})\nNodes({[x.name for x in self.topo_list]})"

    def __repr__(self) -> str:
        return str(self)

    def __enter__(self) -> "DAG":
        self.nodes.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return

    def get_full_dag_nodes_info(self) -> List[Dict[str, Any]]:
        """Get full DAG nodes info"""
        self._ensure_topo_sort()
        return [node for node in self.topo_list]

    def collect_outputs(self) -> Dict[str, Any]:
        """Collect results from output nodes"""
        results = {}
        for output_name, output_node in self.output_nodes.items():
            if output_node.status == NodeStatus.COMPLETED:
                # Get result from parent node
                parent_node = output_node.parents[0]  # Should only have one parent
                if parent_node.result_ref is not None:
                    results[output_name] = ray.get(parent_node.result_ref)
        return results

    def add_node(self, node: Node) -> None:
        """Add node to the DAG"""
        if node.name in self.node_map:
            raise ValueError(f"Node with name {node.name} already exists")
        self.nodes.append(node)
        self.node_map[node.name] = node
        self.in_degree[node] = len(node.parents)
        self._mark_topo_dirty()  # Mark as dirty after structure change

    def get_node(self, node_name: str) -> Optional[Node]:
        """Get node by name"""
        return self.node_map.get(node_name)

    def remove_node(self, node_name: str) -> None:
        """Remove node from the DAG"""
        if node_name not in self.node_map:
            return
        node = self.node_map[node_name]

        # Update dependencies
        for parent in node.parents:
            parent.children.remove(node)
        for child in node.children:
            child.parents.remove(node)
            self.in_degree[child] -= 1

        self.nodes.remove(node)
        del self.node_map[node_name]
        del self.in_degree[node]
        self._mark_topo_dirty()  # Mark as dirty after structure change

    def validate(self) -> bool:
        """Validate the DAG"""
        try:
            # check basic attributes
            if not self.nodes:
                raise ValueError("DAG has no nodes")

            # check the validity of node types
            for node in self.nodes:
                if node.node_type not in NodeType:
                    raise ValueError(f"Invalid node type for node {node.name}")

                # check the special constraints of input and output nodes
                if node.node_type == NodeType.INPUT and node.parents:
                    raise ValueError(
                        f"Input node {node.name} should not have parent nodes"
                    )
                if node.node_type == NodeType.OUTPUT and node.children:
                    raise ValueError(
                        f"Output node {node.name} should not have child nodes"
                    )

            # Only perform topological sort if necessary
            self._ensure_topo_sort()

            # check the validity of node connections
            for node in self.nodes:
                if node.node_type == NodeType.COMPUTE:
                    # verify if all inputs can be obtained from parent nodes
                    required_inputs = set(node.input_kwargs.keys())
                    available_inputs = set()
                    for parent in node.parents:
                        available_inputs.update(parent.output_names)

                    if not required_inputs.issubset(available_inputs):
                        missing = required_inputs - available_inputs
                        raise ValueError(f"Node {node.name} missing inputs: {missing}")

            return True
        except Exception as e:
            logger.error(f"DAG validation failed: {str(e)}")
            return False

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get DAG execution statistics"""
        if not self.is_completed:
            return {}

        stats = {
            "total_time": self.end_time - self.start_time if self.end_time else None,
            "node_stats": {},
            "error_count": len(self.error_nodes),
            "successful_nodes": sum(
                1 for node in self.nodes if node.status == NodeStatus.COMPLETED
            ),
        }

        for node in self.nodes:
            stats["node_stats"][node.name] = {
                "status": node.status.value,
                "error": node.error_message if hasattr(node, "error_message") else None,
            }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to a dictionary representation, for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": self.metadata,
            "is_completed": self.is_completed,
            "execution_stats": self.get_execution_stats(),
        }

    def _mark_topo_dirty(self):
        """Mark the topological sort as outdated"""
        self._topo_dirty = True

    def _ensure_topo_sort(self):
        """Ensure the topological sort is up to date"""
        if self._topo_dirty:
            self.topological_sort()
            self._topo_dirty = False
