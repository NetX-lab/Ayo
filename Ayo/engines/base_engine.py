import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ray

from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class BaseRequest:
    """Base data class for all engine requests"""

    request_id: str
    query_id: str
    callback_ref: Any
    timestamp: float = field(default_factory=time.time)


class BaseEngine(ABC):
    """Base class for all Ray Actor engines

    Features:
    - Async request handling
    - Request queuing and batching
    - Request tracking by query_id
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_queue_size: int = 1000,
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):

        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.scheduler_ref = scheduler_ref

        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)

        self.query_requests: Dict[str, List[BaseRequest]] = {}

        self.loop = asyncio.get_event_loop()

        self.running = True
        self.tasks = [
            self.loop.create_task(self._batch_requests()),
            self.loop.create_task(self._process_batches()),
        ]

    @abstractmethod
    def _load_model(self):
        """Load the model - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def submit_request(
        self, request_id: str, query_id: str, **kwargs
    ) -> ray.ObjectRef:
        """Submit a new request - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def _get_next_batch(self):
        """Get next batch of requests - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def _process_batch(self, batch_data):
        """Process a batch of requests - must be implemented by subclasses"""
        pass

    async def _batch_requests(self):
        """Async task for batching requests"""
        while self.running:
            try:
                batch_data = await self._get_next_batch()
                if batch_data:
                    await self.batch_queue.put(batch_data)
                else:
                    await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Error in batching task: {e}")
                continue

    async def _process_batches(self):
        """Async task for processing batches"""
        while self.running:
            try:
                try:
                    batch_data = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_batch(batch_data)

            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                continue

    async def shutdown(self):
        """Shutdown the engine"""
        self.running = False
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
