from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

from Ayo.engines.base_engine import BaseEngine


class EngineType(str, Enum):
    """Supported engine types"""

    INPUT = "input"
    OUTPUT = "output"
    EMBEDDER = "embedder"
    VECTOR_DB = "vector_db"
    RERANKER = "reranker"
    LLM = "llm"
    AGGREGATOR = "aggregator"
    DUMMY = "dummy"

    @classmethod
    def list(cls) -> list:
        """Get list of all engine types"""
        return list(cls)

    @classmethod
    def validate(cls, engine_type: str) -> bool:
        """Validate if engine type is supported"""
        return engine_type in cls.__members__.values()


@dataclass
class EngineSpec:
    """Engine specifications"""

    engine_class: Any  # 使用Any代替具体类型
    default_config: Dict
    description: str


class EngineRegistry:
    """Central registry for engine types and specifications"""

    def __init__(self):
        self._registry: Dict[str, EngineSpec] = {}
        self._register_default_engines()

    def _register_default_engines(self):
        """Register built-in engines"""
        # Lazy import to avoid circular imports
        from Ayo.engines.aggregator import AggregateEngine
        from Ayo.engines.base_engine import BaseEngine
        from Ayo.engines.embedder import EmbeddingEngine
        from Ayo.engines.llm import LLMEngine
        from Ayo.engines.reranker import RerankerEngine
        from Ayo.engines.vector_db import VectorDBEngine

        self.register(
            EngineType.INPUT,
            EngineSpec(
                engine_class=BaseEngine,
                default_config={},
                description="The dummy engine for input node",
            ),
        )

        self.register(
            EngineType.OUTPUT,
            EngineSpec(
                engine_class=BaseEngine,
                default_config={},
                description="The dummy engine for output node",
            ),
        )

        self.register(
            EngineType.EMBEDDER,
            EngineSpec(
                engine_class=EmbeddingEngine,
                default_config={
                    "model_name": "BAAI/bge-large-en-v1.5",
                    "max_batch_size": 1024,
                    "vector_dim": 1024,
                },
                description="Text embedding engine using BGE model",
            ),
        )

        self.register(
            EngineType.VECTOR_DB,
            EngineSpec(
                engine_class=VectorDBEngine,
                default_config={
                    "host": "localhost",
                    "user": "asplos25",
                    "password": "123456",
                    "database": "database_asplos",
                    "port": 5432,
                    "max_batch_size": 1000,
                },
                description="Vector database engine using pgvector",
            ),
        )

        self.register(
            EngineType.RERANKER,
            EngineSpec(
                engine_class=RerankerEngine,
                default_config={
                    "model_name": "BAAI/bge-reranker-large",
                    "max_batch_size": 512,
                },
                description="Cross-encoder reranking engine",
            ),
        )

        self.register(
            EngineType.LLM,
            EngineSpec(
                engine_class=LLMEngine,
                default_config={
                    "model_name": "meta-llama/Llama-2-7b-chat-hf",
                    "tensor_parallel_size": 1,
                    "max_num_seqs": 256,
                    "max_queue_size": 1000,
                    "trust_remote_code": False,
                    "dtype": "auto",
                },
                description="Large language model engine",
            ),
        )

        self.register(
            EngineType.AGGREGATOR,
            EngineSpec(
                engine_class=AggregateEngine,
                default_config={"max_batch_size": 32, "max_queue_size": 1000},
                description="Data Aggregation Engine, supports multiple aggregation modes",
            ),
        )

    def register(self, engine_type: str, spec: EngineSpec) -> None:
        """Register a new engine type"""
        if engine_type in self._registry:
            raise ValueError(f"Engine type {engine_type} already registered")
        self._registry[engine_type] = spec

    def unregister(self, engine_type: str) -> None:
        """Unregister an engine type"""
        if engine_type not in self._registry:
            raise ValueError(f"Engine type {engine_type} not registered")
        del self._registry[engine_type]

    def get_spec(self, engine_type: str) -> Optional[EngineSpec]:
        """Get engine specifications"""
        return self._registry.get(engine_type)

    def get_engine_class(self, engine_type: str) -> Optional[Type[BaseEngine]]:
        """Get engine class for given type"""
        spec = self.get_spec(engine_type)
        return spec.engine_class if spec else None

    def get_default_config(self, engine_type: str) -> Optional[Dict]:
        """Get default configuration for engine type"""
        spec = self.get_spec(engine_type)
        return spec.default_config if spec else None

    def list_engines(self) -> Dict[str, str]:
        """List all registered engines and their descriptions"""
        return {
            engine_type: spec.description
            for engine_type, spec in self._registry.items()
        }


# Global engine registry instance
ENGINE_REGISTRY = EngineRegistry()
