from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EngineConfig:
    """configuration for execution engines"""

    name: str
    engine_type: str  # e.g., "embedder", "llm", "vector_db"
    num_gpus: int = 0
    num_cpus: int = 1
    resources: Optional[Dict[str, int]] = None  # e.g., {"GPU": 2}
    instances: int = 1
    model_config: Optional[Dict] = None
    latency_profile: Optional[Dict] = None

    def dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)


@dataclass
class AppConfig:
    """application config"""

    engines: Dict[str, EngineConfig]
    optimization_passes: List[str] = None
    workflow_template: Dict[str, Any] = None
