import re
from copy import deepcopy
from typing import List, Set, Tuple

from Ayo.dags.dag import DAG
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import EngineType
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.opt_pass.base_pass import OPT_Pass

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class PrefillingSpiltPass(OPT_Pass):
    """Optimize Pass: Make prefilling operation split into partial prefilling and full prefilling

    Based on input dependency, calculate the part that does not depend on subsequent inputs early, to improve the overall efficiency.
    For RAG template, split the part before {question} as partial prefilling, and the part after {question} as full prefilling.
    """

    def __init__(self):
        super().__init__(name="prefilling_split")

    def run(self, dag: DAG) -> DAG:
        """Execute the optimize pass

        Args:
            dag: Input DAG

        Returns:
            Optimized DAG, containing split prefilling nodes

        """
        processed_nodes = set()

        # Get the list of nodes in topological order
        to_process_nodes = dag.topological_sort()

        for node in to_process_nodes:
            if self._is_splittable_prefilling(node) and node not in processed_nodes:
                print(f"Splitting prefilling node: {node.name}")
                self._split_prefilling(dag, node, processed_nodes)

        return dag

    def _is_splittable_prefilling(self, node: Node) -> bool:
        """Check if the node can be split

        If the node is of prefilling type and has multiple input dependencies, it can be split
        """
        if node.op_type == NodeOps.LLM_PREFILLING:
            # Check if the node has multiple input dependencies
            if len(node.parents) == 1:
                return False
            # Or check if the input format of the node contains multiple fields
            if len(node.io_schema.input_format) <= 1:
                return False
            # Check if there is a template string and contains placeholders
            parents = node.parents
            for parent in parents:
                if (
                    parent.name in node.input_key_from_parents
                    and abs(node.depth - parent.depth) > 1
                ):
                    return True

        return False

    def _analyze_template(self, template: str) -> Tuple[str, str, List[str], List[str]]:
        """Analyze the template string and split it into partial prefilling and full prefilling parts

        Args:
            template: The template string

        Returns:
            partial_template: The partial prefilling template
            full_template: The full prefilling template
            partial_dependencies: The input fields that partial prefilling depends on
            full_dependencies: The input fields that full prefilling depends on
        """
        # Extract all input placeholders
        placeholders = re.findall(r"\{([^{}]+)\}", template)
        unique_placeholders = [p for p in placeholders]

        if not unique_placeholders:
            # If there are no placeholders, do not split
            return template, "", [], []

        logger.info(f"Unique placeholders: {unique_placeholders}")
        # Use the first placeholder as the split point by default
        split_placeholder = unique_placeholders[0]

        # Find the split point position
        split_pos = template.find(f"{{{split_placeholder}}}")
        if split_pos == -1:
            # If the split point is not found, do not split
            return template, "", unique_placeholders, []

        # Find the position of the next placeholder after the split point
        next_placeholder_pos = -1
        for placeholder in unique_placeholders:
            if placeholder == split_placeholder:
                continue

            placeholder_pattern = f"{{{placeholder}}}"
            pos = template.find(
                placeholder_pattern, split_pos + len(f"{{{split_placeholder}}}")
            )
            if pos != -1 and (next_placeholder_pos == -1 or pos < next_placeholder_pos):
                next_placeholder_pos = pos
                logger.info(f"Next placeholder position: {next_placeholder_pos}")

        # If the next placeholder is not found, find the sentence end marker
        if next_placeholder_pos == -1:
            logger.warning(
                f"Cannot find next placeholder for {split_placeholder} in template"
            )
            end_markers = [".", "?", "!"]
            end_pos = -1

            for marker in end_markers:
                marker_pos = template.find(marker, split_pos)
                if marker_pos != -1 and (end_pos == -1 or marker_pos < end_pos):
                    end_pos = marker_pos

            # If the sentence end marker is not found, use the placeholder end position
            if end_pos == -1:
                end_pos = split_pos + len(f"{{{split_placeholder}}}")
            else:
                end_pos = end_pos + 1  # Include the sentence end marker

        else:
            # Use the position of the next placeholder as the split point (not including the next placeholder)
            end_pos = next_placeholder_pos

        # Split the template
        partial_template = template[
            :end_pos
        ]  # Include the content before the second placeholder
        full_template = template[
            end_pos:
        ]  # Include the content from the first placeholder

        # Analyze the dependencies
        partial_dependencies = []
        full_dependencies = []

        # Check each placeholder in which part appears
        for placeholder in unique_placeholders:
            placeholder_pattern = f"{{{placeholder}}}"
            if placeholder_pattern in partial_template:
                partial_dependencies.append(placeholder)
            if placeholder_pattern in full_template:
                full_dependencies.append(placeholder)

        return partial_template, full_template, partial_dependencies, full_dependencies

    def _split_prefilling(
        self, dag: DAG, node: Node, processed_nodes: Set[Node]
    ) -> None:
        """Split the prefilling node into partial prefilling and full prefilling

        Args:
            dag: The DAG being optimized
            node: The prefilling node to split
            processed_nodes: The set of processed nodes
        """
        # Get the template
        template = node.config.get("template", None) or node.config.get(
            "prompt_template", None
        )

        if not isinstance(template, str) or not template:
            # If there is no template or the template is not a string, use the default analysis method
            raise ValueError(f"No template found for node {node.name}")

        # Analyze the RAG template
        partial_template, full_template, partial_deps, full_deps = (
            self._analyze_template(template)
        )

        if not full_template or not partial_deps:
            # If it cannot be split, do not process
            logger.warning(
                f"Cannot split prefilling node {node.name} because it cannot be split"
            )
            processed_nodes.add(node)
            return

        logger.info(f"Partial dependencies: {partial_deps}")
        logger.info(f"Full dependencies: {full_deps}")
        # Create partial prefilling node
        partial_io_schema = deepcopy(node.io_schema)
        # Only keep the input that partial prefilling needs
        partial_io_schema.input_format = {
            k: v for k, v in node.io_schema.input_format.items() if k in partial_deps
        }
        # Set the output format
        partial_io_schema.output_format = {"partial_prefilling_done": bool}

        partial_config = deepcopy(node.config)
        partial_config["prompt_template"] = partial_template
        partial_config["prompt"] = None
        partial_config["is_partial"] = True
        partial_config["partial_output"] = False
        partial_config["partial_prefilling"] = True

        partial_node = Node(
            name=f"{node.name}-partial",
            node_type=node.node_type,
            engine_type=node.engine_type,
            op_type=NodeOps.LLM_PARTIAL_PREFILLING,
            io_schema=partial_io_schema,
            config=partial_config,
            anno=node.anno,
        )

        dag.add_node(partial_node)

        # Create full prefilling node
        full_io_schema = deepcopy(node.io_schema)
        # Add partial_result as input
        full_io_schema.input_format = {
            k: v for k, v in node.io_schema.input_format.items() if k in full_deps
        }
        full_io_schema.input_format["partial_prefilling_done"] = bool

        full_config = deepcopy(node.config)
        full_config["prompt_template"] = full_template
        full_config["prompt"] = full_template
        full_config["is_full"] = True
        full_config["partial_output"] = False
        full_config["partial_prefilling"] = True
        full_config["llm_partial_decoding_idx"] = -1

        full_node = Node(
            name=f"{node.name}-full",
            node_type=node.node_type,
            engine_type=node.engine_type,
            op_type=NodeOps.LLM_FULL_PREFILLING,
            io_schema=full_io_schema,
            config=full_config,
            anno=node.anno,
        )

        dag.add_node(full_node)

        # Connect partial and full nodes
        partial_node.add_child(full_node)
        full_node.add_parent(partial_node)
        full_node.input_key_from_parents[partial_node.name] = "partial_prefilling_done"

        # Connect parent nodes
        for parent in node.parents:
            input_key = node.input_key_from_parents.get(parent.name)
            logger.info(
                f"original prefilling's parent: {parent.name}, input_key: {input_key}"
            )
            if input_key in partial_deps:
                # This parent node provides the input that partial prefilling needs
                parent.add_child(partial_node)
                partial_node.add_parent(parent)
                partial_node.input_key_from_parents[parent.name] = input_key

            elif input_key in full_deps:
                # This parent node provides the input that full prefilling needs
                parent.add_child(full_node)
                full_node.add_parent(parent)
                full_node.input_key_from_parents[parent.name] = input_key

            else:
                raise ValueError(
                    f"Cannot find similar dependency for {input_key} in {full_deps+partial_deps}"
                )

        # Connect child nodes
        for child in node.children:
            full_node.add_child(child)
            child.add_parent(full_node)
            # Update the input key mapping of the child node
            for output_key in node.io_schema.output_format:
                if (
                    node.name in child.input_key_from_parents
                    and child.input_key_from_parents[node.name] == output_key
                ):
                    child.input_key_from_parents[full_node.name] = output_key

        # Remove the original node from the DAG
        dag.remove_node(node.name)
        processed_nodes.add(node)

    def get_applicable_nodes(self, dag: DAG) -> List[Node]:
        """Get the nodes that can apply this optimization"""
        return [node for node in dag.nodes if self._is_splittable_prefilling(node)]

    def validate_dag(self, dag: DAG) -> bool:
        """Validate if the DAG can apply this optimization"""
        return len(self.get_applicable_nodes(dag)) > 0


if __name__ == "__main__":
    # Test code

    # Create a test DAG
    dag = DAG(dag_id="test_dag_prefilling_split")

    # RAG template
    RAG_PROMPT_TEMPLATE = """\
      You are an AI assistant specialized in Retrieval-Augmented Generation (RAG). Your responses
      must be based strictly on the retrieved documents provided to you. Follow these guidelines:
      1. Use Retrieved Information Only - Your responses must rely solely on the retrieved documents.
      If the retrieved documents do not contain relevant information, explicitly state: 'Based on the
      available information, I cannot determine the answer.'\n"
      2. Response Formatting - Directly answer the question using the retrieved data. If multiple
      sources provide information, synthesize them in a coherent manner. If no relevant information
      is found, clearly state that.\n"
      3. Clarity and Precision - Avoid speculative language such as 'I think' or 'It might be.'
      Maintain a neutral and factual tone.\n"
      4. Information Transparency - Do not fabricate facts or sources. If needed, summarize the
      retrieved information concisely.\n"
      5. Handling Out-of-Scope Queries - If a question is outside the retrieved data (e.g., opinions,
      unverifiable claims), state: 'The retrieved documents do not provide information on this topic.'\n
      ---\n
      Example Interactions:\n
      User Question: Who founded Apple Inc.?\n
      Retrieved Context: 'Apple Inc. was co-founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.'\n
      Model Answer: 'Apple Inc. was co-founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.'\n
      ---\n
      User Question: When was the first iPhone released, and what were its key features?\n"
      Retrieved Context: 'The first iPhone was announced by Steve Jobs on January 9, 2007, and released on June 29, 2007.' "
      "'The original iPhone featured a 3.5-inch touchscreen display, a 2-megapixel camera, and ran on iOS.'\n"
      Model Answer: 'The first iPhone was announced on January 9, 2007, and released on June 29, 2007. "
      "It featured a 3.5-inch touchscreen display, a 2-megapixel camera, and ran on iOS.'\
      This ensures accuracy, reliability, and transparency in all responses. And you should directly answer the question based on the retrieved context and keep it concise as possible.
      Here is the question: {question}?
      Here is the retrieved context: {context}
      Here is your answer:
     """

    search_node = Node(
        name="Search",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        op_type=NodeOps.VECTORDB_SEARCHING,
        io_schema=NodeIOSchema(
            input_format={"question": str}, output_format={"context": str}
        ),
        config={},
    )

    # Create prefilling node
    prefilling_node = Node(
        name="RAGPrefilling",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        op_type=NodeOps.LLM_PREFILLING,
        io_schema=NodeIOSchema(
            input_format={"question": str, "context": str},
            output_format={"prefilled_done": bool},
        ),
        config={"template": RAG_PROMPT_TEMPLATE},
    )

    # create decoding node
    decoding_node = Node(
        name="Decoding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        op_type=NodeOps.LLM_DECODING,
        io_schema=NodeIOSchema(
            input_format={"prefilled_done": bool}, output_format={"answer": str}
        ),
        config={},
    )

    # Connect nodes
    search_node >> prefilling_node
    prefilling_node >> decoding_node

    # Register nodes
    dag.register_nodes(search_node, prefilling_node, decoding_node)

    dag.set_query_inputs({"question": "What is the capital of France?"})

    # Print the original DAG information
    print("Original DAG:")
    print(dag.get_full_dag_nodes_info())

    # Apply the optimization
    dag.optimize([PrefillingSpiltPass()])

    # Print the optimized DAG information
    print("\nOptimized DAG:")
    print(dag.get_full_dag_nodes_info())

    for node in dag.nodes:
        if node.op_type in [
            NodeOps.LLM_PARTIAL_PREFILLING,
            NodeOps.LLM_FULL_PREFILLING,
        ]:
            print(f"Node: {node.name}")
            print(node.config.get("template", ""))

    # Visualize the DAG
    try:
        from Ayo.vis.vis_graph import visualize_dag_with_node_types

        visualize_dag_with_node_types(dag, "test_dag_prefilling_split.pdf")
        print("\nDAG visualization image generated: test_dag_prefilling_split.pdf")
    except ImportError:
        print(
            "\nFailed to generate visualization image, missing necessary dependencies"
        )
