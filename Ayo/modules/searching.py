from typing import List
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeType, NodeOps, NodeIOSchema
from Ayo.engines.engine_types import EngineType
from Ayo.modules.base_module import BaseModule

class SearchingModule(BaseModule):
    '''
    The searching module is used to caluate the embedding for the query and search the vector database for the most relevant documents.
    '''
    def __init__(self, 
                 input_format: dict={
                    "queries": List[str],
                    'index_status': bool
                 },
                 output_format: dict={
                    'search_results': List[str]
                 },
                 config: dict={
                    'top_k': 3
                 }):
        
        """Initialize the Searching Module.
        
        This module is responsible for calculating embeddings for queries and searching
        the vector database for the most relevant documents based on embedding similarity.
        
        Args:
            input_format (dict): Input format definition, defaults to:
                - queries (List[str]): List of queries to search for
                - index_status (bool): Status indicating whether the index is ready
            output_format (dict): Output format definition, defaults to:
                - search_results (List[str]): List of retrieved documents
            config (dict): Configuration parameters, including:
                - top_k (int): Number of top results to return per query
        """
        super().__init__(input_format, output_format, config)

    def to_primitive_nodes(self):
        
        query_embedd_input_key=None
        index_input_key=None
        for key in self.input_format.keys():   
            if 'query' in key or 'queries' in key or 'expanded_queries' in key or 'passages' in key or 'expanded_passages' in key or 'expanded_query' in key:
                query_embedd_input_key=key 

            elif 'index' in key or 'index_status' in key:
                index_input_key=key
 

        query_embedding_node = Node(
            name="QueryEmbedding",
            io_schema=NodeIOSchema(
                input_format={query_embedd_input_key: self.input_format[query_embedd_input_key]},
                output_format={"queries_embeddings": List[float]}
            ),
            op_type=NodeOps.EMBEDDING,
            engine_type=EngineType.EMBEDDER,
            node_type=NodeType.COMPUTE,
            config={}
        )
        
        vector_db_searching_node = Node(
            name="VectorDBSearching",
            io_schema=NodeIOSchema(
                input_format={
                    "queries_embeddings": List[float],
                    index_input_key: self.input_format[index_input_key]
                },
                output_format=self.output_format
            ),
            op_type=NodeOps.VECTORDB_SEARCHING,
            engine_type=EngineType.VECTOR_DB,
            node_type=NodeType.COMPUTE,
            config={
                "top_k": self.config["top_k"],
            }
        )

        query_embedding_node >> vector_db_searching_node

        return [query_embedding_node, vector_db_searching_node]

        
