import uuid
from typing import List

from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.modules.base_module import BaseModule


class QueryExpandingModule(BaseModule):

    prompt_template = """Please rewrite the following question into {refine_question_number} more refined one. \
        You should keep the original meaning of the question, but make it more suitable and clear for context retrieval. \
        The original question is: {question}? \
        Please output your answer in json format. \
        It should contain {refine_question_number} new refined questions.\
        For example, if the expaned number is 3, the json output should be like this: \
        {{\
            "revised question1": "[refined question 1]",\
            "revised question2": "[refined question 2]",\
            "revised question3": "[refined question 3]"\
        }}\
        You just need to output the json string, do not output any other information or additional text!!! \
        The json output:"""

    # temperature=0.9, top_p=0.95 for llama2-7b-chat-hf

    # prompt
    # support_partial_output = node.config.get('partial_output', False)
    # support_partial_prefilling = node.config.get('partial_prefilling', False)

    # llm_partial_decoding_idx = node.config.get("llm_partial_decoding_idx", -1)

    def __init__(
        self,
        input_format: dict = {"query": str},
        output_format: dict = {"expanded_queries": List[str]},
        config: dict = {
            "expanded_query_num": 3,
            "prompt_template": prompt_template,
            "parse_json": True,
            "prompt": prompt_template,
            "partial_output": False,
            "partial_prefilling": False,
            "llm_partial_decoding_idx": -1,
        },
    ):
        """Initialize the Query Expanding Module.

        This module is responsible for expanding a single query into multiple refined queries
        using a language model, which can improve retrieval performance by capturing different
        aspects of the original query.

        Args:
            input_format (dict): Input format definition, defaults to:
                - query (str): Original user query
            output_format (dict): Output format definition, defaults to:
                - expanded_queries (List[str]): List of expanded/refined queries
            config (dict): Configuration parameters, including:
                - expanded_query_num (int): Number of queries to generate
                - prompt_template (str): Template for query expansion prompt
                - parse_json (bool): Whether to parse JSON output
                - prompt (str): Complete prompt string
                - partial_output (bool): Whether to enable partial output
                - partial_prefilling (bool): Whether to enable partial prefilling
                - llm_partial_decoding_idx (int): Partial decoding index
        """
        super().__init__(input_format, output_format, config)

    def to_primitive_nodes(self) -> List[Node]:
        # create LLM prefilling node

        llm_internal_id = f"query_expanding_{uuid.uuid4()}"

        llm_prefilling_node = Node(
            name="QueryExpandingPrefilling",
            io_schema=NodeIOSchema(
                input_format={"query": str}, output_format={"prefill_state": dict}
            ),
            op_type=NodeOps.LLM_PREFILLING,
            engine_type=EngineType.LLM,
            node_type=NodeType.COMPUTE,
            config={
                "prompt_template": self.config.get(
                    "prompt_template", self.prompt_template
                ),
                "prompt": self.config.get("prompt", self.prompt_template),
                "expanded_query_num": self.config.get("expanded_query_num", 3),
                "parse_json": self.config.get("parse_json", True),
                "partial_output": self.config.get("partial_output", False),
                "partial_prefilling": self.config.get("partial_prefilling", False),
                "llm_partial_decoding_idx": self.config.get(
                    "llm_partial_decoding_idx", -1
                ),
                "llm_internal_id": llm_internal_id,
            },
        )

        # create LLM decoding node
        llm_decoding_node = Node(
            name="QueryExpandingDecoding",
            io_schema=NodeIOSchema(
                input_format={"query": str, "prefill_state": dict},
                output_format={"expanded_queries": List[str]},
            ),
            op_type=NodeOps.LLM_DECODING,
            engine_type=EngineType.LLM,
            node_type=NodeType.COMPUTE,
            config={
                "prompt_template": self.config.get(
                    "prompt_template", self.prompt_template
                ),
                "prompt": self.config.get("prompt", self.prompt_template),
                "expanded_query_num": self.config.get("expanded_query_num", 3),
                "parse_json": self.config.get("parse_json", True),
                "partial_output": self.config.get("partial_output", False),
                "partial_prefilling": self.config.get("partial_prefilling", False),
                "llm_partial_decoding_idx": self.config.get(
                    "llm_partial_decoding_idx", -1
                ),
                "llm_internal_id": llm_internal_id,
            },
        )

        # Connect the nodes
        llm_prefilling_node >> llm_decoding_node

        return [llm_prefilling_node, llm_decoding_node]

    def format_prompt(self, question: str) -> str:
        refine_question_number = self.config.get("expanded_query_num", None)

        assert refine_question_number is not None, "expanded_query_num is not set"
        # keys = ", ".join([f"question{i+1}" for i in range(refine_question_number)])
        # json_example = "{\n      " + "\n      ".join([f"\"question{i+1}\": \"[refined version {i+1}]\"" + ("," if i < refine_question_number-1 else "") for i in range(refine_question_number)]) + "\n    }"

        return self.prompt_template.format(
            refine_question_number=refine_question_number,
            question=question,
        )


if __name__ == "__main__":
    query_expanding_module = QueryExpandingModule()
    question = "What is the capital of France?"
    print(query_expanding_module.format_prompt(question))
