from typing import List, TYPE_CHECKING
from Ayo.dags.node_commons import NodeType
from Ayo.opt_pass.base_pass import OPT_Pass
from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)

if TYPE_CHECKING:
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node

class PruningDependencyPass(OPT_Pass):
    """Optimization Pass: Clean up invalid dependencies in DAG"""
    
    def __init__(self):
        super().__init__(name="PruningDependencyPass")
    
    def run(self, dag: 'DAG') -> 'DAG':
        """Execute the optimization pass"""
        # save the reference of DAG
        self.dag = dag
        
        # Ensure topological sort is up to date
        dag._ensure_topo_sort()
        
        # Check and add missing parent node connections
        self._add_missing_connections(dag)
        
        # Identify essential parents for each node based on topological structure
        self._identify_essential_parents(dag)
        
        # Finally prune invalid connections
        for node in dag.nodes:
            self._prune_node_dependencies(node)
        
        # clear the reference of DAG, avoid memory leak
        self.dag = None
        
        return dag
    
    def get_applicable_nodes(self, dag: 'DAG') -> List['Node']:
        """Get nodes that can be pruned"""
        return [node for node in dag.nodes if len(node.parents) > 0]
    
    def validate_dag(self, dag: 'DAG') -> bool:
        """Validate if DAG can be pruned"""
        return len(dag.nodes) > 0
    
    def _prune_node_dependencies(self, node: 'Node') -> None:
        """Clean up invalid dependencies for a single node"""
        invalid_parents = []
        
        for parent in node.parents:
            if not self._has_valid_connection(parent, node):
                invalid_parents.append(parent)
        
        for parent in invalid_parents:
            self._remove_connection(parent, node)
    
    def _has_valid_connection(self, parent: 'Node', child: 'Node') -> bool:
        """Check if there is valid data flow between parent and child nodes"""
        if parent.name in child.input_key_from_parents:
            output_key = child.input_key_from_parents[parent.name]
            if (output_key in parent.output_names and 
                output_key in child.input_names):
                return True
        return False
    
    def _remove_connection(self, parent: 'Node', child: 'Node') -> None:
        """Remove connection between two nodes"""
        if child in parent.children:
            parent.children.remove(child)
        
        if parent in child.parents:
            child.parents.remove(parent)
        
        if parent.name in child.input_key_from_parents:
            del child.input_key_from_parents[parent.name]
        
        # update the in_degree information of DAG
        if hasattr(self, 'dag'):
            if child in self.dag.in_degree:
                self.dag.in_degree[child] -= 1
            # mark the topological sort need to be recalculated
            self.dag._mark_topo_dirty()
    
    def _add_missing_connections(self, dag: 'DAG') -> None:
        """Check and add missing parent node connections"""
        # preprocess: create the mapping from output name to node
        output_providers = {}
        for node in dag.topo_list:
            for output_name in node.output_names:
                if output_name not in output_providers:
                    output_providers[output_name] = []
                output_providers[output_name].append(node)
        
        # process the nodes in topological order
        for node in dag.topo_list:
            if node.node_type == NodeType.INPUT:
                continue
                
            # check each input needed by the node
            for input_name in node.input_names:
                # if this input has no provider
                # here we assume the output name from different nodes are unique  
                logger.info(f"{node.name}, {input_name}, {node.input_key_from_parents.values()}")
                if (input_name  not in node.input_key_from_parents.values()):
                    logger.warning(f"input_name: {input_name} not in node.input_key_from_parents.values(): {node.input_key_from_parents.values()}")
                    # find the node that can provide this input
                    potential_parents = output_providers.get(input_name, [])
                    for potential_parent in potential_parents:
                        if (potential_parent != node and 
                            potential_parent not in node.parents):
                            # add the connection
                            node.add_parent(potential_parent)
                            node.input_key_from_parents[potential_parent.name] = input_name
                            break
    
    def _identify_essential_parents(self, dag: 'DAG') -> None:
        """Identify essential parents for each node based on topological structure"""
        # process the nodes in topological order
        for node in dag.topo_list:
            if node.node_type == NodeType.INPUT or len(node.parents) <= 1:
                continue
                
            self._find_essential_parents(node)
    
    def _find_essential_parents(self, node: 'Node') -> None:
        """Find essential parents for a node, removing redundant parents"""
        # Record each input provider
        input_providers = {}  # input name -> best parent node
        redundant_parents = []
        
        # ensure we know each parent's input
        for parent in node.parents:
            if parent.name in node.input_key_from_parents:
                output_key = node.input_key_from_parents[parent.name]
                
                # if this input has no provider, record current parent
                if output_key not in input_providers:
                    input_providers[output_key] = parent
                else:
                    # there is already a node providing this input
                    # here we can implement the logic to select the best parent node, for example:
                    # 1. select the node with earlier topological order (may result in an earlier availability)
                    # 2. select the node based on other criteria
                    # default: keep the first encountered parent node
                    redundant_parents.append(parent)
        
        # remove the redundant parent node connections
        for parent in redundant_parents:
            self._remove_connection(parent, node)


if __name__ == "__main__":
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node
    from Ayo.dags.node_commons import NodeType, NodeIOSchema, NodeAnnotation
    from Ayo.engines.engine_types import EngineType
    from typing import Any, Dict

    dag = DAG()

    dag.set_query_inputs({"query": "What is the capital of France?", "passages": ["Paris is the capital of France.", "Paris is the capital of France.", "Paris is the capital of France."]})

    embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str]},
            output_format={"embeddings_passages": List[Any]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    reranker_node = Node(
        name="Reranker",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.RERANKER,
        io_schema=NodeIOSchema(
            input_format={
                "query": str,
                "passages": List[str]
            },
            output_format={
                "ranked_results": List[str]
            }
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    llm_node = Node(
        name="LLM",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        io_schema=NodeIOSchema(
            input_format={"query": str, "ranked_results": List[str]},
            output_format={"answer": str}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }   
    )

    embedding_node>>reranker_node >> llm_node

    dag.register_nodes(embedding_node, reranker_node, llm_node)

    print(dag.get_full_dag_nodes_info())

    dag.optimize([PruningDependencyPass()])


    print(dag.get_full_dag_nodes_info())
    
