from copy import deepcopy
from typing import List, Dict, Set, TYPE_CHECKING, Optional 
import uuid
if TYPE_CHECKING:
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node

from Ayo.dags.node_commons import NodeAnnotation, NodeIOSchema, NodeType, NodeOps
from Ayo.engines.engine_types import EngineType
from Ayo.opt_pass.base_pass import OPT_Pass
from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)

class LLMDecodingPipeliningPass(OPT_Pass):
    """Optimization Pass: Split LLM decoding operations into parallel sub-operations
    
    Split LLM decoding operations into parallel sub-operations
    """
    
    def __init__(self):
        super().__init__(name="llm_decoding_split")
        self.num_splits = None
        # Define which operator types are splittable

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
            Optimized DAG with split decoding nodes
        """

        processed_nodes = set()

        to_decompose_nodes = dag.topological_sort()

        for node in to_decompose_nodes:
            if self._is_splittable_llm_decoding(node) and node not in processed_nodes:
                print(f"Splitting pipeline from node: {node.name}")
                self._split_pipeline(dag, node, processed_nodes)

        return dag
    

    def _get_num_splits(self,node):

        config=node.config 

        for k,v in config.items():
            if 'num' in k:
                self.num_splits = v
                break

        return self.num_splits
    
    def _is_splittable_llm_decoding(self, node: 'Node') -> bool:
        """Check if the node is splittable
        LLM decoding node is splittable if its child node is batchable; 
        and actually it is the start of the pipeline if return True
        """
        if node.op_type==NodeOps.LLM_DECODING and node.children[0].op_type in self.batchable_ops:
            num_split=self._get_num_splits(node)
            assert num_split is not None, f"num_split is not set for node {node.name}" 
            return True
        return False
    
    def _is_pipeline_end(self, node: 'Node') -> bool:
        """Check if the node is the end of the pipeline
        Pipeline end node is a batchable node and it is not splittable;
        and actually it is the end of the pipeline if return True, 
        and an aggregator node would be added 
        """
        if node.op_type in self.batchable_ops and node.op_type not in self.splittable_ops:
            return True
        return False
    
    def _is_in_pipeline(self, node: 'Node') -> bool:
        """Check if the node is in the pipeline
        Pipeline node is a splittable node and a batchable node that is not the end of the pipeline
        """
        if node.op_type in self.batchable_ops and node.op_type in self.splittable_ops:
            return True
        return False
    
    def _split_pipeline(self, dag: 'DAG', start_node: 'Node', processed_nodes: Set['Node']) -> None:
        """Split the entire pipeline starting from an LLM decoding node
        
        Args:
            dag: The DAG being optimized
            start_node: Starting LLM decoding node
            processed_nodes: Set of already processed nodes
        """
        from Ayo.dags.node import Node

        current_node = start_node
        current_sub_nodes = None
        decomposed_nodes_list = []



        while True:
            # 1. Create the sub-nodes of the current node
            num_sub_nodes = self.num_splits if current_sub_nodes is None else len(current_sub_nodes)
            sub_nodes = []
            
            for i in range(num_sub_nodes):
                io_schema = deepcopy(current_node.io_schema)
                # Modify the output format, adding the split identifier

                # if len(decomposed_nodes_list) == 0:
                #     io_schema.output_format = {
                #         f"{k}-sub-{i}": v 
                #         for k, v in current_node.io_schema.output_format.items()
                #     }

                # Get the appropriate operation type based on the current node type
                if current_node.op_type == NodeOps.LLM_DECODING:
                    sub_op_type = NodeOps.LLM_PARTIAL_DECODING
                    config = deepcopy(current_node.config) 
                    config["llm_partial_decoding_idx"] = i
                    config['partial_output'] = True 

                else:
                    # Keep the original operation type
                    sub_op_type = current_node.op_type
                    config = deepcopy(current_node.config)
                
                sub_node = Node(
                    name=f"{current_node.name}-sub-{i}",
                    node_type=current_node.node_type,
                    engine_type=current_node.engine_type,
                    op_type=sub_op_type,
                    io_schema=io_schema,
                    config=config
                )
                sub_node.decomposed = True
                sub_nodes.append(sub_node)
                dag.add_node(sub_node)

            if current_node.op_type == NodeOps.LLM_DECODING: 
                # modify the prefilling parent node's config
                for parent in current_node.parents:
                    if parent.op_type in [NodeOps.LLM_FULL_PREFILLING, NodeOps.LLM_PARTIAL_PREFILLING, NodeOps.LLM_PREFILLING]:
                        parent.config['partial_output'] = True

            # 2. Update the connection relationship
            # Connect the parent node
            if current_sub_nodes is None:
                # The first node, connect the original parent node
                for parent in current_node.parents:
                    parent.add_child(sub_nodes[0])
                    sub_nodes[0].add_parent(parent)
            
                # Establish the serial dependency between sub-nodes: A0->B0->C0
                for i in range(len(sub_nodes)-1):
                    sub_nodes[i].add_child(sub_nodes[i+1])
                    sub_nodes[i+1].add_parent(sub_nodes[i])
            else:
                # Connect the previous group of sub-nodes
        
                for sub_node_prev, sub_node in zip(current_sub_nodes, sub_nodes):
                    sub_node.add_parent(sub_node_prev)
                    sub_node_prev.add_child(sub_node)

                other_parents = [p for p in current_node.parents if p != sub_nodes[0].parents[0]]
                
                # Connect the other parent nodes
                for other_parent in other_parents:
                    for sub_node in sub_nodes:
                        other_parent.add_child(sub_node)
                        sub_node.add_parent(other_parent)

            # Record the processed nodes
            decomposed_nodes_list.append(current_node)
            
                
            #child = current_node.children[0]
            
            if not self._is_in_pipeline(current_node) and self._is_pipeline_end(current_node):
                logger.warning(f"encounter a node at the end of pipeline: {current_node.name}")
                if current_node.op_type in self.batchable_ops:
                    # First split the child node

                    # for i, node in enumerate(sub_nodes):
                    #     # add the suffix to the output format
                    #     node.io_schema.output_format = {
                    #         f"{k}-sub-{i}": v 
                    #         for k, v in current_node.io_schema.output_format.items()
                    #     }
                    #     node.refresh_io_schema(node.io_schema)


                    # The child node is batchable but not splittable, add the aggregator node
                    agg_io_schema = deepcopy(current_node.io_schema)
                    input_format = {}
                    for i in range(num_sub_nodes):
                        for k, v in current_node.io_schema.output_format.items():
                            input_format[f"{k}-sub-{i}"] = v
                    agg_io_schema.input_format = input_format
                    
                    agg_node = Node(
                        name=f"aggregator-{current_node.name}",
                        node_type=NodeType.COMPUTE,
                        engine_type=EngineType.AGGREGATOR,
                        op_type=NodeOps.AGGREGATOR,
                        io_schema=agg_io_schema,
                        anno=NodeAnnotation.NONE
                    )
                    dag.add_node(agg_node)
                    
                    # Connect the aggregator node
                    for i, sub_node in enumerate(sub_nodes):
                        io_schema = deepcopy(sub_node.io_schema)
                        io_schema.output_format = {
                            f"{k}-sub-{i}": v 
                            for k, v in current_node.io_schema.output_format.items()
                        }
                        sub_node.refresh_io_schema(io_schema)
                        sub_node.add_child(agg_node)
                        agg_node.add_parent(sub_node)
                    
                    # Connect the original child node
                    agg_node.add_child(current_node.children[0])
                    current_node.children[0].add_parent(agg_node)
                break
                
            # Continue to process the next node
            current_node = current_node.children[0]
            current_sub_nodes = sub_nodes

        # 4. Delete the original node
        for node in decomposed_nodes_list:
            dag.remove_node(node.name)
            processed_nodes.add(node)
    
        
    def _create_split_io_schema(self, original_schema, split_idx: int) -> NodeIOSchema:
        """Create the IO schema for the split node"""
        new_schema = deepcopy(original_schema)
        # Modify the output name, adding the split identifier
        new_schema.output_format = {
            f"{k}_{split_idx}": v 
            for k, v in original_schema.output_format.items()
        }
        return new_schema
        
    def _create_aggregator_node(self, dag: 'DAG', sub_nodes: List['Node'], 
                              next_node: 'Node') -> 'Node':
        """Create an aggregator node"""
        agg_node = Node(
            name=f"aggregator_{next_node.name}",
            node_type=NodeType.COMPUTE,
            engine_type=EngineType.AGGREGATOR,
            io_schema=self._create_aggregator_schema(sub_nodes, next_node),
            anno=NodeAnnotation.NONE
        )
        dag.add_node(agg_node)
        return agg_node
        
    def _create_aggregator_schema(self, sub_nodes: List['Node'], 
                                next_node: 'Node') -> 'NodeIOSchema':
        """Create the IO schema for the aggregator node"""
        input_format = {}
        for sub_node in sub_nodes:
            input_format.update(sub_node.io_schema.output_format)
            
        return NodeIOSchema(
            input_format=input_format,
            output_format=next_node.io_schema.input_format
        )
        
    def _update_connections(self, dag: 'DAG', original_node: 'Node', 
                          sub_nodes: List['Node']):
        """Update the connections between nodes"""
        # Connect parent nodes
        for parent in original_node.parents:
            for sub_node in sub_nodes:
                parent.add_child(sub_node)
                sub_node.add_parent(parent)
                if parent.name in original_node.input_key_from_parents:
                    sub_node.input_key_from_parents[parent.name] = \
                        original_node.input_key_from_parents[parent.name]
                        
        # Establish dependencies between sub-nodes
        for i in range(len(sub_nodes)-1):
            sub_nodes[i].add_child(sub_nodes[i+1])
            
    def _connect_aggregator(self, dag: 'DAG', sub_nodes: List['Node'], 
                          agg_node: 'Node', next_node: 'Node'):
        """Connect the aggregator node"""
        # Connect sub-nodes to the aggregator node
        for sub_node in sub_nodes:
            sub_node.add_child(agg_node)
            agg_node.add_parent(sub_node)
            
        # Connect the aggregator node to the next node
        agg_node.add_child(next_node)
        next_node.add_parent(agg_node)




if __name__ == "__main__":

    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node
    from Ayo.dags.node_commons import NodeType, NodeIOSchema, NodeAnnotation
    from Ayo.engines.engine_types import EngineType
    from typing import Any, Dict

    dag = DAG(dag_id="test_dag_llm_decoding_pipeling")

    dag.set_query_inputs({"prefilled": True, 
                          'query':'What is the capital of France?'})
    
    uuid_str = str(uuid.uuid4())
    
    prefilling_node = Node(
        name="Prefilling",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        op_type=NodeOps.LLM_PREFILLING,
        io_schema=NodeIOSchema(
            input_format={"query": str},
            output_format={"prefilled": bool}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
            'prompt_template': "xxxxxxx",
            'prompt': "xxxxxxx",
            'expanded_query_num': 2,
            'partial_output': False,
            'partial_prefilling': False,
            'llm_internal_id': uuid_str
        }
    )

    llm_decoding_node = Node(
        name="LLM_Decoding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        op_type=NodeOps.LLM_DECODING,
        io_schema=NodeIOSchema(
            input_format={"prefilled": bool,},
            output_format={"expanded_queries": List[str]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
            'prompt_template': "xxxxxxx",
            'prompt': "xxxxxxx",
            'expanded_query_num': 2,
            'partial_output': False,
            'partial_prefilling': False,
            'llm_partial_decoding_idx': -1,
            'llm_internal_id': uuid_str
        }
    )

    embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        op_type=NodeOps.EMBEDDING,
        io_schema=NodeIOSchema(
            input_format={"expanded_queries": List[str]},
            output_format={"embeddings_expanded_queries": List[Any]}
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
            input_format={"embeddings_expanded_queries": List[Any]},
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

    prefilling_node >> llm_decoding_node >> embedding_node >> search_node >> reranking_node

    dag.register_nodes(prefilling_node, llm_decoding_node, embedding_node, search_node, reranking_node)

    print(dag.get_full_dag_nodes_info())

    dag.optimize([LLMDecodingPipeliningPass()])

    print(dag.get_full_dag_nodes_info())

    from Ayo.vis.vis_graph import visualize_dag_with_node_types

    visualize_dag_with_node_types(dag, "test_dag_llm_decoding_pipeling.pdf")


    