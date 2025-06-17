from typing import List

from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.modules.base_module import BaseModule


class RerankingModule(BaseModule):
    def __init__(
        self,
        input_format: dict = {"query": str, "passages": List[str]},
        output_format: dict = {"passages": List[str]},
        config: dict = {"top_k": 10},
    ):
        """Initialize the Reranking Module.

        This module is responsible for reranking the passages based on the query.

        Args:
            input_format (dict): Input format definition, defaults to:
                - query (str): The query to rerank the passages
                - passages (List[str]): The passages to rerank
            output_format (dict): Output format definition, defaults to:
                - passages (List[str]): The reranked passages
            config (dict, optional): Configuration parameters for the reranking process
        """

        super().__init__(input_format, output_format, config)

    def to_primitive_nodes(self):
        return [
            Node(
                name="Reranking",
                io_schema=NodeIOSchema(
                    input_format=self.input_format, output_format=self.output_format
                ),
                op_type=NodeOps.RERANKING,
                engine_type=EngineType.RERANKER,
                node_type=NodeType.COMPUTE,
                config=self.config,
            )
        ]
