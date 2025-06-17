from dataclasses import dataclass
from enum import Enum
from typing import Dict

from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeOps


@dataclass
class EmbeddingModelConfig:
    """embedding model config"""

    model_name: str
    dimension: int
    max_length: int = 512
    device: str = "cuda"
    batch_size: int = 1024
    vector_dim: int = 1024


@dataclass
class LLMConfig:
    """llm model config"""

    model_name: str
    temperature: float = 0.9
    top_p: float = 0.95
    device: str = "cuda"


@dataclass
class VectorDBConfig:
    """vector database config"""

    db_path: str
    dimension: int
    index_type: str = "HNSW"  # or "IVF", "HNSW" etc.
    metric_type: str = "cosine"  # or "IP", "cosine" etc.
    nprobe: int = 10


class AggMode(str, Enum):
    """aggregator mode"""

    DUMMY = "dummy"
    MERGE = "merge"
    TOP_K = "top_k"


def get_aggregator_config(node: Node, **kwargs) -> Dict:
    """get the aggregator config"""

    assert (
        node.op_type == NodeOps.AGGREGATOR
    ), f"node {node.name} is not an aggregator node"
    if node.parents[0].op_type == NodeOps.EMBEDDING:
        return {"agg_mode": AggMode.DUMMY}
    elif node.parents[0].op_type == NodeOps.VECTORDB_SEARCHING:
        return {"agg_mode": AggMode.MERGE}
    elif node.parents[0].op_type == NodeOps.RERANKING:

        agg_config = {"agg_mode": AggMode.TOP_K}
        agg_config.update(
            {
                node.config.get("topk", {})
                or node.config.get("top_k", {})
                or node.config.get("k", {})
                or 5
            }
        )
        return agg_config
    elif node.parents[0].op_type == NodeOps.VECTORDB_INGESTION:
        return {"agg_mode": AggMode.DUMMY}
    else:
        raise ValueError(f"Unsupported node op type: {node.op_type}")


def get_aggregator_config_for_parent_node(node: Node, **kwargs) -> Dict:
    if node.op_type == NodeOps.EMBEDDING:
        return {"agg_mode": AggMode.DUMMY}
    elif node.op_type == NodeOps.VECTORDB_SEARCHING:
        return {"agg_mode": AggMode.MERGE}
    elif node.op_type == NodeOps.RERANKING:
        return {"agg_mode": AggMode.TOP_K}
    elif node.op_type == NodeOps.VECTORDB_INGESTION:
        return {"agg_mode": AggMode.DUMMY}
    else:
        raise ValueError(f"Unsupported node op type: {node.op_type}")
