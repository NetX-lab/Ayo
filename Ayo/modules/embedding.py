from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeType, NodeOps, NodeIOSchema
from Ayo.engines.engine_types import EngineType
from Ayo.modules.base_module import BaseModule


class EmbeddingModule(BaseModule):
    def __init__(self, 
                 input_format: dict={
                    "text": list[str]
                 },
                 output_format: dict={
                    "embeddings": list
                 },
                 config: dict=None):
        """Initialize the Embedding Module.
        
        This module is responsible for converting text into vector embeddings
        using an embedding model.
        
        Args:
            input_format (dict): Input format definition, defaults to:
                - text (list[str]): List of text strings to be embedded
            output_format (dict): Output format definition, defaults to:
                - embeddings (list): List of vector embeddings
            config (dict, optional): Configuration parameters for the embedding process
        """
        super().__init__(input_format, output_format, config)

    def to_primitive_nodes(self):
        return [
            Node(
                name="Embedding",
                io_schema=NodeIOSchema(
                    input_format=self.input_format,
                    output_format=self.output_format
                ),
                op_type=NodeOps.EMBEDDING,
                engine_type=EngineType.EMBEDDING,
                node_type=NodeType.COMPUTE,
                config=self.config
            )
        ]
