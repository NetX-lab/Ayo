from typing import List

from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.modules.base_module import BaseModule


class IndexingModule(BaseModule):
    def __init__(
        self,
        input_format: dict = {"passages": List[str]},
        output_format: dict = {"index_status": bool},
        config: dict = None,
    ):
        """Initialize the Indexing Module.

        This module is responsible for embedding passages and ingesting them into a vector database.
        It creates an index that can be used for vector similarity search.

        Args:
            input_format (dict): Input format definition, defaults to:
                - passages (List[str]): List of text passages to be indexed
            output_format (dict): Output format definition, defaults to:
                - index_status (bool): Status indicating whether indexing was successful
            config (dict, optional): Configuration parameters for the indexing process
        """
        super().__init__(input_format, output_format, config)

    def to_primitive_nodes(self):
        # create embedding node
        embedding_node = Node(
            name="EmbeddingForIndex",
            io_schema=NodeIOSchema(
                input_format={"passages": List[str]},
                output_format={"passages_embeddings": List[float]},
            ),
            op_type=NodeOps.EMBEDDING,
            engine_type=EngineType.EMBEDDER,
            node_type=NodeType.COMPUTE,
            config=self.config,
        )

        # create ingestion node
        ingestion_node = Node(
            name="IngestionForIndex",
            io_schema=NodeIOSchema(
                input_format={
                    "passages": List[str],
                    "passages_embeddings": List[float],
                },
                output_format={"index_status": bool},
            ),
            op_type=NodeOps.VECTORDB_INGESTION,
            engine_type=EngineType.VECTOR_DB,
            node_type=NodeType.COMPUTE,
            config=self.config,
        )

        # connect nodes
        embedding_node >> ingestion_node

        return [embedding_node, ingestion_node]
