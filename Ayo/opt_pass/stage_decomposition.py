import copy
import math
import time
from typing import List, Dict, Set, TYPE_CHECKING
from Ayo.configs.model_config import get_aggregator_config, get_aggregator_config_for_parent_node
from Ayo.opt_pass.base_pass import OPT_Pass
from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL
from Ayo.dags.node_commons import NodeType, NodeOps

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)

if TYPE_CHECKING:
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node

class StageDecompositionPass(OPT_Pass):
    """Optimization Pass: Decompose computation-intensive nodes into sub-nodes
    
    This pass identifies batchable operations and splits them into multiple 
    smaller operations to enable parallel processing
    """
    
    def __init__(self, batch_size: int = 100):
        super().__init__(name="stage_decomposition")
        self.configure(batch_size=batch_size)
        
        # Define which operator types are batchable
        self.batchable_ops = {
            NodeOps.VECTORDB_INGESTION,
            NodeOps.VECTORDB_SEARCHING,
            NodeOps.RERANKING,  
            NodeOps.EMBEDDING
        }

        self.splittable_ops = {
            NodeOps.VECTORDB_SEARCHING,
            NodeOps.EMBEDDING,
            NodeOps.LLM_DECODING
        }
        
    def run(self, dag: 'DAG') -> 'DAG':
        """Execute the optimization pass
        
        Args:
            dag: Input DAG
            
        Returns:
            Optimized DAG with decomposed nodes
        """
        processed_nodes = set()

        to_decompose_nodes = dag.topological_sort()

        for node in to_decompose_nodes:
            if self._is_decomposable(node) and node not in processed_nodes:
                logger.debug(f"Decomposing node: {node.name}")
                self._decompose_node(dag, node, processed_nodes)

        return dag
    
    def get_applicable_nodes(self, dag: 'DAG') -> List['Node']:
        """Get nodes that can be decomposed"""
        return [node for node in dag.nodes 
                if node.op_type in self.batchable_ops]
    
    def validate_dag(self, dag: 'DAG') -> bool:
        """Validate if DAG has decomposable nodes"""
        return any(node.is_batchable() for node in dag.nodes)
    
    def _is_decomposable(self, node: 'Node') -> bool:
        """Check if a node can be decomposed
        
        A node is decomposable if:
        1. Its operator type is batchable and splittable
        2. Its child node is batchable
        3. Its parent node is batchable and splittable or is input node
        4. Its input size exceeds batch_size
        """
            
        if node.op_type not in self.batchable_ops or node.op_type not in self.splittable_ops:
            return False
            
        if len(node.children) !=1:
            return False 
        
        if node.children[0].op_type not in self.batchable_ops:
            return False
        
        parent_nodes=node.parents

        for parent_node in parent_nodes:
            if parent_node.node_type == NodeType.INPUT:
                continue
            if parent_node.op_type not in self.batchable_ops or parent_node.op_type not in self.splittable_ops:
                return False
        
        return True
        
    def _get_num_sub_nodes(self, node: 'Node', input_size: int, **kwargs) -> int:
        """Get the number of sub-nodes needed for a node"""

        #FIXME: hardcode some heuristic values, to update the auto-tuning of the micro-batch size 
        if node.op_type == NodeOps.EMBEDDING:
            heuristic_micro_batch_size = 128
            sub_nodes = math.ceil(input_size / heuristic_micro_batch_size)

            if sub_nodes == 0:
                input_size,sub_nodes = 1

            return heuristic_micro_batch_size,sub_nodes
        
        elif node.op_type == NodeOps.VECTORDB_SEARCHING:
            heuristic_micro_batch_size = 64 
            sub_nodes = math.ceil(input_size / heuristic_micro_batch_size)

            if sub_nodes == 0:
                input_size,sub_nodes = 1

            return heuristic_micro_batch_size,sub_nodes
        
        elif node.op_type == NodeOps.RERANKING:
            return 64,3
        else:
            return 64,1


    def _get_batchable_input_field_for_node(self, node: 'Node', **kwargs) -> List[str]:
        """Get the batchable input field for a node"""

        if node.op_type == NodeOps.EMBEDDING:
            return list(node.input_kwargs.keys())
        
        elif node.op_type == NodeOps.VECTORDB_SEARCHING:

            batchable_input_fields = []
            for key in node.input_kwargs.keys():
                if key.lower() in ['query','queries','question','questions','query_vector','query_vectors','query_vectors_list','query_vector_list','query_vector_lists']:
                    batchable_input_fields.append(key)
            if len(batchable_input_fields) == 0:
                raise ValueError(f"Missing batchable input field for node {node.name} vector_db_searching")
            return batchable_input_fields
        
        elif node.op_type == NodeOps.RERANKING:
            batchable_input_fields = []
            for key in node.input_kwargs.keys():
                if key.lower() in ['query','queries','question','questions']:
                    batchable_input_fields.append(key)
            if len(batchable_input_fields) == 0:
                raise ValueError(f"Missing batchable input field for node {node.name} reranking")
            return batchable_input_fields
        
        elif node.op_type == NodeOps.VECTORDB_INGESTION:
            return list(node.input_kwargs.keys())
        else:
            raise ValueError(f"Unsupported node type: {node.op_type}")
    

    
    def _decompose_node(self, dag: 'DAG', node: 'Node', processed_nodes: set) -> None:
        """Decompose a node into multiple sub-nodes
        
        Args:
            dag: The DAG being optimized
            node: Node to decompose
            processed_nodes: Set of already processed nodes
        """

        #Steps:
        # 1. for the target node, calculate the number of sub-nodes needed
        # 2. according to the number of sub-nodes, create sub-nodes, each sub-node needs to update the corresponding variables, and set self.decomposed: bool = True
        # self.input_shards_mapping
        # 3. update its connections with the parent node
        # 4. process the child node, the processing of the child node needs to ensure that the child node is in the batchable ops, otherwise directly connect the child node to the sub-node
        # for the child node, according to the current number of sub-nodes, to correspondingly decompose the child node into multiple sub-nodes, each sub-node of the child node directly connects to the sub-node
        # 5. delete the original node, check if the child node is splittable, if it is, process the child node's child node according to the above steps. Otherwise, add an aggregate node, and aggregate the sub-nodes results of the child node
        

        # Calculate number of sub-nodes needed
        batchable_input_fields = self._get_batchable_input_field_for_node(node) 

        # Get the corresponding parent node
        input_sizes = set()

        for batchable_input_field in batchable_input_fields:
            parent_node = None
            if batchable_input_field in node.input_key_to_parent:
                parent_node = node.input_key_to_parent[batchable_input_field]
            
            if parent_node is None:
                # If the parent node is not found, try to get the data from the query_inputs of the DAG
                if batchable_input_field in dag.query_inputs:
                    input_data = dag.query_inputs[batchable_input_field]
                    if isinstance(input_data, list):
                        batch_size = len(input_data)
                    else:
                        batch_size = 1
                    output_shape_info = (batch_size,)
                else:
                    raise ValueError(f"cannot find the source of input {batchable_input_field} for node {node.name}")
            else:
                parent_node.update_output_shape_info()
                # Get the output shape info from the parent node
                if hasattr(parent_node, 'output_shape_info') and batchable_input_field in parent_node.output_shape_info:
                    output_shape_info = parent_node.output_shape_info[batchable_input_field]["shape"]
                else:
                    # If there is no shape info, try to infer the shape from the output of the parent node
                    raise ValueError(f"cannot find the shape info of input {batchable_input_field} for node {node.name}")

            batch_size = output_shape_info[0]
            input_sizes.add(batch_size)

        assert len(input_sizes) == 1, f"different input sizes for batchable input fields for node {node.name}: {input_sizes}"

        batch_size = input_sizes.pop()
        
        logger.debug(f"batch_size: {batch_size}")

        
        sub_nodes = []

        micro_batch_size,num_sub_nodes = self._get_num_sub_nodes(node,input_size=batch_size)

        if num_sub_nodes == 1:
            return 

        
        for i in range(num_sub_nodes):

            start_idx = i * micro_batch_size
            end_idx = min((i + 1) * micro_batch_size, batch_size)

            from Ayo.dags.node import Node

            sub_node = Node(
                op_type=node.op_type,
                name=f"{node.name}-sub-{i}",
                engine_type=node.engine_type,
                node_type=node.node_type,
                io_schema=node.io_schema,
            )
            
            sub_node.decomposed = True

            sub_node.input_shards_mapping = {
                batchable_input_field: slice(start_idx, end_idx) 
                for batchable_input_field in batchable_input_fields
            }
            
            sub_nodes.append(sub_node)
            dag.add_node(sub_node)

        self._update_upstream_connections(dag, node, sub_nodes)
        
        decomposed_nodes_list = [node]  

        # After updating the connections, process the aggregator node

        #child = node.children[0] 

        # FIXME: Here we assume that each compute node has only one child node

        while True:
            child = node.children[0]
        
            if not self._is_decomposable(child) and child.op_type in self.batchable_ops:
            # child is a batchable ops, then it needs to be decomposed into multiple sub-nodes, the number is the same as the number of sub_nodes
                num_sub_nodes = len(sub_nodes)

                sub_nodes_for_child = [] 

                other_parents = [parent for parent in child.parents if parent != node]

                for i in range(num_sub_nodes):

                    io_schema = copy.deepcopy(child.io_schema)

                    logger.debug(f"io_schema: {io_schema} for {child.name}")
                    io_schema.output_format = {
                        k+"-sub-"+str(i): v for k, v in io_schema.output_format.items()
                    }


                    sub_node_for_child = Node(
                        op_type=child.op_type,
                        name=f"{child.name}-sub-{i}",
                        engine_type=child.engine_type,
                        node_type=child.node_type,
                        # the output should be have a suffix of "_sub_{i}"
                        io_schema=io_schema,
                    )

                    sub_node_for_child.decomposed = True
                    
                    sub_nodes_for_child.append(sub_node_for_child)

                    dag.add_node(sub_node_for_child)
                    
                # Make the connections between the sub_nodes_for_child and sub_nodes
                for sub_node_child, sub_node in zip(sub_nodes_for_child, sub_nodes):
                    sub_node_child.add_parent(sub_node)
                    sub_node.add_child(sub_node_child) 

                    input_key_intersect = set(sub_node_child.input_kwargs.keys()) & set(sub_node.input_kwargs.keys()) 

                    if sub_node.input_shards_mapping:
                        for input_key in input_key_intersect:
                            sub_node_child.input_shards_mapping[input_key] = sub_node.input_shards_mapping[input_key]

                for other_parent in other_parents:
                    for sub_node_child in sub_nodes_for_child:
                        other_parent.add_child(sub_node_child)
                        sub_node_child.add_parent(other_parent)

                decomposed_nodes_list.append(child)


                agg_io_schema = copy.deepcopy(child.io_schema)
                output_format = {k+"-sub-"+str(i): v for i in range(num_sub_nodes) 
                            for k, v in child.io_schema.output_format.items()}
                agg_io_schema.input_format = output_format

                # Create the aggregator node
                agg_config = get_aggregator_config_for_parent_node(child)

               
                agg_node = Node(
                    op_type=NodeOps.AGGREGATOR,
                    name=f"{child.name}-aggregator",
                    engine_type=child.engine_type,
                    node_type=child.node_type,
                    io_schema=agg_io_schema,
                    config=agg_config
                )
                dag.add_node(agg_node)
                
                # Connect the aggregator node
                for sub_node in sub_nodes_for_child:
                    sub_node.add_child(agg_node)
                    agg_node.add_parent(sub_node)
                
                # Connect the original child node
                agg_node.add_child(child.children[0])
                child.children[0].add_parent(agg_node)

                break

            elif self._is_decomposable(child):

                num_sub_nodes = len(sub_nodes)

                sub_nodes_for_child = [] 

                other_parents = [parent for parent in child.parents if parent != node]

                for i in range(num_sub_nodes):

                    io_schema = copy.deepcopy(child.io_schema)

                    logger.debug(f"io_schema: {io_schema} for {child.name}")

                    sub_node_for_child = Node(
                        op_type=child.op_type,
                        name=f"{child.name}-sub-{i}",
                        engine_type=child.engine_type,
                        node_type=child.node_type,
                        io_schema=io_schema,
                    )

                    sub_nodes_for_child.append(sub_node_for_child)

                    dag.add_node(sub_node_for_child)    
                    
                    sub_node_for_child.decomposed = True

                for sub_node_child, sub_node in zip(sub_nodes_for_child, sub_nodes):
                    sub_node_child.add_parent(sub_node)
                    sub_node.add_child(sub_node_child) 

                for other_parent in other_parents:
                    for sub_node_child in sub_nodes_for_child:
                        other_parent.add_child(sub_node_child)
                        sub_node_child.add_parent(other_parent)

                decomposed_nodes_list.append(child)




            else:
                break

            node = child
            sub_nodes=sub_nodes_for_child





        # remove the original node
        dag.remove_node(node.name)
        processed_nodes.add(node)
        for child in decomposed_nodes_list:
            dag.remove_node(child.name)
            processed_nodes.add(child)




    
    
    def _update_upstream_connections(self, dag: 'DAG', original_node: 'Node', sub_nodes: List['Node']):
        """Update connections for decomposed nodes
        
        For each parent of the original node:
        - Connect it to all sub-nodes
        """
        # Handle parent connections
        for parent in original_node.parents:
            for sub_node in sub_nodes:
                parent.add_child(sub_node)
                sub_node.add_parent(parent)
                

if __name__ == "__main__":
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node
    from Ayo.dags.node_commons import NodeType, NodeIOSchema, NodeAnnotation
    from Ayo.engines.engine_types import EngineType
    from typing import Any, Dict

    dag = DAG(dag_id="test_dag_stage_decomposition")

    dag.set_query_inputs({"query": "What is the capital of France?", 
                          "questions": ["Paris is the capital of France.", 
                                      "France is a country in Europe.", 
                                      "China is a country in Asia.", 
                                      "Asia is a continent in the world.", 
                                      "Europe is a continent in the world.", 
                                      "America is a continent in the world."]*128})
    

    embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        op_type=NodeOps.EMBEDDING,
        io_schema=NodeIOSchema(
            input_format={"questions": List[str]},
            output_format={"embeddings_questions": List[Any]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    search_node = Node(
        name="Search",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        op_type=NodeOps.VECTORDB_SEARCHING,
        io_schema=NodeIOSchema(
            input_format={"embeddings_questions": List[Any]},
            output_format={"search_results": List[str]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    reranking_node = Node(
        name="Reranking",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.RERANKER,
        op_type=NodeOps.RERANKING,
        io_schema=NodeIOSchema( 
            input_format={"query": List[Any], "search_results": List[str]},
            output_format={"reranked_results": List[str]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    embedding_node >> search_node >> reranking_node

    dag.register_nodes(embedding_node, search_node, reranking_node)

    print(dag.get_full_dag_nodes_info())

    begin_time = time.time()

    dag.optimize([StageDecompositionPass()])

    end_time = time.time()

    print(f"Time taken: {end_time - begin_time} seconds")

    print(dag.get_full_dag_nodes_info())


    from Ayo.vis.vis_graph import  visualize_dag_with_node_types

    visualize_dag_with_node_types(dag, "test_dag_stage_decomposition_optimized.pdf")
