from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from Ayo.dags.node_commons import NodeOps
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.utils import (
    check_unfilled_placeholders_in_prompt_template,
    fill_prompt_template_with_placeholdersname_approximations,
)
from vllm import SamplingParams

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)

if TYPE_CHECKING:
    from Ayo.dags.node import Node


class RequestType(str, Enum):
    """Types of vector database operations"""

    INGESTION = "ingestion"  # Insert vectors and texts
    SEARCHING = "searching"  # Search for similar vectors


class PayloadTransformer(ABC):
    """Base class for payload transformers"""

    @abstractmethod
    def transform(self, input_kwargs: Dict[str, Any]) -> Any:
        """Transform input data to engine required format

        Args:
            input_kwargs: Original input parameters

        Returns:
            Transformed data
        """
        pass


class VectorDBTransformer(PayloadTransformer):
    """Payload transformer for vector database"""

    def transform(self, node: "Node") -> Tuple[RequestType, Any]:
        """Transform vector database input

        Handle two types of requests:
        1. Ingestion request: requires embeddings and texts
        2. Search request: requires query_vector and top_k

        Args:
            input_kwargs: Dictionary containing request parameters

        Returns:
            Tuple[RequestType, Any]: Request type and transformed data
        """
        # Check if operation type is provided
        request_type = (
            RequestType.INGESTION
            if node.op_type == NodeOps.VECTORDB_INGESTION
            else RequestType.SEARCHING
        )

        if not request_type:
            raise ValueError("Missing operation type in input")

        if request_type == RequestType.INGESTION:
            # Handle ingestion request
            embeddings = None
            texts = None

            for key, value in node.input_kwargs.items():
                if "embed" in key.lower():
                    embeddings = value
                    break

            for key, value in node.input_kwargs.items():
                if "embed" not in key.lower() and (
                    "text" in key.lower() or "passage" in key.lower()
                ):
                    texts = value
                    break

            if embeddings is None or texts is None:
                raise ValueError("Missing embeddings or texts for ingestion")

            if len(embeddings) != len(texts):
                raise ValueError("Number of embeddings must match number of texts")

            if isinstance(texts, list):
                num_text_embeddings_pairs = len(texts)
            else:
                num_text_embeddings_pairs = 1

            if isinstance(embeddings, list):
                assert (
                    len(embeddings) == num_text_embeddings_pairs
                ), "Number of embeddings must match number of texts"
                assert isinstance(
                    embeddings[0], list
                ), "Embeddings must be a list of lists"

            elif isinstance(embeddings, np.ndarray):
                assert (
                    len(embeddings) == num_text_embeddings_pairs
                ), "Length of embeddings np array must match number of texts"
                embeddings = embeddings.tolist()
            else:
                raise ValueError("Invalid embeddings format")

            # Ensure texts is a list of strings
            if not isinstance(texts, list):
                texts = [str(texts)]
            texts = [str(text) for text in texts]

            # （embeddings, texts） pair
            data = [(embedding, text) for embedding, text in zip(embeddings, texts)]

            logger.debug(f"{self.__class__.__name__} len of data: {len(data)}")

            logger.debug(f"vector db transformer data: len of data: {len(data)}")

            return {"request_type": RequestType.INGESTION, "data": data}

        elif request_type == RequestType.SEARCHING:
            # Handle search request
            query_vectors = None

            for key, value in node.input_kwargs.items():
                if key.lower() in [
                    "query_vectors",
                    "query_vector",
                    "query_embedding",
                    "query_embeddings",
                    "queries_embeddings",
                    "embeddings",
                    "embedding",
                    "query_vector_list",
                    "query_vector_lists",
                    "queries",
                    "query_list",
                    "query_lists",
                ]:
                    query_vectors = value
                    break

            top_k = node.config.get("top_k", None)

            # Ensure query_vector is a list of numpy arrays
            if isinstance(query_vectors, list):
                if len(query_vectors) == 0:
                    raise ValueError(
                        f"Empty query_vectors list for node {node.name} vector db searching"
                    )

                if isinstance(query_vectors[0], list):
                    # Convert list of lists to list of numpy arrays
                    try:
                        query_vectors = [
                            np.array(query_vector, dtype=float)
                            for query_vector in query_vectors
                        ]
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Failed to convert query vectors to numpy arrays for node {node.name} vector db searching"
                        )

                elif isinstance(query_vectors[0], np.ndarray):
                    # Already a list of numpy arrays, ensure they are 1D
                    for i, qv in enumerate(query_vectors):
                        if qv.ndim != 1:
                            raise ValueError(
                                f"Query vector at index {i} has {qv.ndim} dimensions, expected 1D for node {node.name} vector db searching"
                            )

                else:
                    raise ValueError(
                        f"Invalid query_vector format: expected list of lists or numpy arrays for node {node.name} vector db searching"
                    )

            elif isinstance(query_vectors, np.ndarray):
                # Handle single or multiple query vectors in a numpy array
                if query_vectors.ndim == 1:
                    # Single 1D vector
                    query_vectors = [query_vectors]
                elif query_vectors.ndim == 2:
                    # Multiple vectors in a 2D array, convert to list of 1D arrays
                    query_vectors = [
                        query_vectors[i] for i in range(query_vectors.shape[0])
                    ]
                else:
                    raise ValueError(
                        f"Query vector numpy array has {query_vectors.ndim} dimensions, expected 1D or 2D for node {node.name} vector db searching"
                    )

            else:
                raise ValueError(
                    f"Invalid query_vector format: expected list or numpy array for node {node.name} vector db searching"
                )

            # （query_vector, top_k） pair
            data = (query_vectors, top_k)

            assert (
                top_k is not None
            ), f"Missing top_k for node {node.name} vector db searching"
            assert (
                query_vectors is not None
            ), f"Missing query_vector for node {node.name} vector db searching"

            return {"request_type": RequestType.SEARCHING, "data": data}

        else:
            raise ValueError(f"Unknown operation type: {request_type}")


class EmbedderTransformer(PayloadTransformer):
    def transform(self, node: "Node") -> Dict:
        texts = []
        for value in node.input_kwargs.values():
            if isinstance(value, (list, tuple)):
                texts.extend(str(item) for item in value)
            else:
                texts.append(str(value))
        return {"texts": texts}


class RerankerTransformer(PayloadTransformer):
    def transform(self, node: "Node") -> Dict:

        # TODO: fix the input_kwargs for any possible input names
        # Add the top_k config
        top_k = node.config.get("top_k", None)
        query = None
        passages = None

        for key, value in node.input_kwargs.items():
            if key.lower() in [
                "query",
                "queries",
                "query_text",
                "query_texts",
                "query_texts_list",
            ]:
                query = value
            elif key.lower() in [
                "text",
                "texts",
                "passages",
                "documents",
                "context",
                "document",
                "contexts",
                "docs",
                "search_results",
                "search_result_list",
                "search_result_lists",
                "searching_results",
                "searching_result_list",
                "searching_result_lists",
            ]:
                passages = value

        assert query is not None, f"Missing query for node {node.name} reranking"
        assert passages is not None, f"Missing passages for node {node.name} reranking"
        assert top_k is not None, f"Missing top_k for node {node.name} reranking"

        if isinstance(query, list):
            if len(query) == 1 and isinstance(query[0], str):
                query = query[0]
            else:
                raise ValueError(
                    f"Query is a list for node {node.name} reranking, only string is supported. please check the input_kwargs"
                )
        elif isinstance(query, str):
            pass
        else:
            raise ValueError(f"Invalid query format for node {node.name} reranking")

        if isinstance(passages, list):
            if len(passages) == 0:
                raise ValueError(f"Empty passages list for node {node.name} reranking")

            if isinstance(passages[0], list):
                # Process nested lists
                merged_passages = []
                for i, passage_list in enumerate(passages):
                    if not isinstance(passage_list, list):
                        raise ValueError(
                            f"Inconsistent passages format for node {node.name} reranking"
                        )

                    # Skip empty lists
                    if len(passage_list) == 0:
                        raise ValueError(
                            f"Empty passages list at index {i} for node {node.name} reranking"
                        )

                    # Check the first element type and process the whole list accordingly type and process the whole list accordingly
                    if passage_list and isinstance(passage_list[0], str):
                        # If it's a list of strings, add non-None elements (filter out  of )add non-None elements (filter out None)
                        merged_passages.extend(
                            [str(p) for p in passage_list if p is not None]
                        )
                    elif passage_list and isinstance(passage_list[0], dict):
                        # If it's a list of dicts, find the text key and extract
                        text_key = None
                        # Only check the first dict's keys, assuming all dicts have the same key structure
                        for key in passage_list[0].keys():
                            if key.lower() in [
                                "text",
                                "passage",
                                "doc",
                                "document",
                                "context",
                                "content",
                                "texts",
                                "passages",
                                "docs",
                                "documents",
                                "contexts",
                            ]:
                                text_key = key
                                break

                        if text_key is None:
                            raise ValueError(
                                f"Missing text key in nested dict list for node {node.name} reranking"
                            )

                        # Extract all texts at once
                        merged_passages.extend(
                            str(p_dict[text_key]) for p_dict in passage_list
                        )
                    else:
                        raise ValueError(
                            f"Passages contains unsupported element types for node {node.name} reranking"
                        )

                passages = merged_passages
            elif isinstance(passages[0], str):
                # process single layer string list
                passages = [str(p) for p in passages if p is not None]
            elif isinstance(passages[0], dict):
                # a possible format for the passages is a list of dicts, each dict contains a "text"-related key and a "score"-related key
                # process nested list
                text_key = None
                # TODO: handle the score key
                merged_passages = []
                for key in passages[0].keys():
                    if key.lower() in [
                        "text",
                        "passage",
                        "doc",
                        "document",
                        "context",
                        "content",
                        "texts",
                        "passages",
                        "docs",
                        "documents",
                        "contexts",
                    ]:
                        text_key = key
                    elif key.lower() in [
                        "score",
                        "similarity_score",
                        "cosine_score",
                        "cosine_similarity_score",
                    ]:
                        pass
                        # TODO: handle the score key, not used for now
                    else:
                        raise ValueError(
                            f"Invalid passage format for node {node.name} reranking"
                        )

                if text_key is None:
                    raise ValueError(f"Missing text key for node {node.name} reranking")
                else:
                    for passage in passages:
                        merged_passages.append(passage[text_key])
                    passages = merged_passages

            else:
                raise ValueError(
                    f"Invalid passages format {type(passages[0])} for node {node.name} reranking"
                )

            # ensure the processed list is not empty
            if len(passages) == 0:
                raise ValueError(
                    f"No valid passages after processing for node {node.name} reranking"
                )

        elif isinstance(passages, str):
            passages = [passages]
            logger.warning(
                f"Passages is a single string for node {node.name} reranking, convert to list"
            )
        else:
            raise ValueError(f"Invalid passages format for node {node.name} reranking")

        return {"query": query, "passages": passages, "top_k": top_k}


class DefaultTransformer(PayloadTransformer):
    def transform(self, node: "Node") -> Dict[str, Any]:
        return node.input_kwargs


class AggregatorTransformer(PayloadTransformer):
    def transform(self, node: "Node") -> Dict[str, Any]:
        """Transform the input for the aggregator engine"""
        agg_mode = node.config.get("agg_mode", "concat")
        data_sources = node.config.get("data_sources", [])

        return {"agg_mode": agg_mode, "data_sources": data_sources}


class LLMTransformer(PayloadTransformer):
    def transform(self, node: "Node") -> Dict[str, Any]:
        # llm_internal_id: str,
        # prompt: str,
        # llm_op_type: NodeOps,
        # llm_partial_decoding_idx: int = -1,
        # sampling_params: SamplingParams = None

        llm_internal_id = node.config.get("llm_internal_id", "")

        # TODO: make the prompt is complete without any unfilled placeholders

        potential_prompt_template = None
        potential_prompt = None
        prompt = None

        for key, value in node.config.items():
            if key.lower() in [
                "prompt_template",
                "prompt_templates",
                "template",
                "templates",
            ]:
                potential_prompt_template = value
            elif key.lower() in ["prompt", "prompts", "prompt_list", "prompt_lists"]:
                potential_prompt = value
            else:
                continue

        if node.op_type in [
            NodeOps.LLM_PREFILLING,
            NodeOps.LLM_PARTIAL_PREFILLING,
            NodeOps.LLM_FULL_PREFILLING,
        ]:
            if potential_prompt is None and potential_prompt_template is None:
                raise ValueError(
                    f"Missing prompt for node {node.name} llm prefilling-related operation {node.op_type}"
                )
            else:
                if potential_prompt_template is not None:
                    # we should fill the prompt template with the input_kwargs
                    prompt = fill_prompt_template_with_placeholdersname_approximations(
                        potential_prompt_template, node.input_kwargs
                    )

                    # first find the placeholders in the prompt template
                else:
                    prompt = potential_prompt
                    # check if the prompt is complete without any unfilled placeholders
                    if not check_unfilled_placeholders_in_prompt_template(prompt):
                        raise ValueError(
                            f"Prompt is not complete without any unfilled placeholders for node {node.name} llm prefilling-related operation {node.op_type}"
                        )

        # prepare the prompt for the LLM

        max_tokens = node.config.get("max_tokens", 150)
        llm_op_type = node.op_type

        support_partial_output = node.config.get("partial_output", False)
        support_partial_prefilling = node.config.get("partial_prefilling", False)

        llm_partial_decoding_idx = node.config.get("llm_partial_decoding_idx", -1)

        # FIXME: this sampling_params is not used for decoding-related operations
        sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=max_tokens,
            support_partial_output=support_partial_output,
            support_partial_prefilling=support_partial_prefilling,
        )

        assert (
            llm_internal_id != ""
        ), f"Missing llm_internal_id for node {node.name} llm"
        assert llm_op_type in [
            NodeOps.LLM_PREFILLING,
            NodeOps.LLM_PARTIAL_PREFILLING,
            NodeOps.LLM_FULL_PREFILLING,
            NodeOps.LLM_DECODING,
            NodeOps.LLM_PARTIAL_DECODING,
        ], f"Invalid llm_op_type for node {node.name} llm"

        if llm_op_type in [
            NodeOps.LLM_PREFILLING,
            NodeOps.LLM_PARTIAL_PREFILLING,
            NodeOps.LLM_FULL_PREFILLING,
        ]:
            assert prompt is not None, f"Missing prompt for node {node.name} llm"

        return {
            "llm_internal_id": llm_internal_id,
            "prompt": prompt,
            "llm_op_type": llm_op_type,
            "llm_partial_decoding_idx": llm_partial_decoding_idx,
            "sampling_params": sampling_params,
        }


# Transformer registry
TRANSFORMER_REGISTRY = {
    "embedder": EmbedderTransformer(),
    "reranker": RerankerTransformer(),
    "vector_db": VectorDBTransformer(),
    "aggregator": AggregatorTransformer(),
    "llm": LLMTransformer(),
    # Add new transformers here
}
