from typing import Dict, List, Optional, Any
import ray
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BaseRequest:
    """Base data class for all engine requests"""
    request_id: str
    query_id: str  # Group requests from same query
    callback_ref: Any  # Ray ObjectRef for result
    timestamp: float = time.time()


class BaseEngine(ABC):
    """Base class for all Ray Actor engines
    
    Features:
    - Async request handling
    - Request queuing and batching
    - Request tracking by query_id
    """
    
    def __init__(self,
                 max_batch_size: int = 32,
                 max_queue_size: int = 1000,
                 scheduler_ref: Optional[ray.actor.ActorHandle] = None,
                 **kwargs):
        
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.scheduler_ref = scheduler_ref
        
        # Async queues
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Track requests by query_id
        self.query_requests: Dict[str, List[BaseRequest]] = {}
        
        # Create event loop
        self.loop = asyncio.get_event_loop()
        
        # Start processing tasks
        self.running = True
        self.tasks = [
            self.loop.create_task(self._batch_requests()),
            self.loop.create_task(self._process_batches())
        ]

    @abstractmethod
    def _load_model(self):
        """Load the model - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def submit_request(self, request_id: str, query_id: str, **kwargs) -> ray.ObjectRef:
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
                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in batching task: {e}")
                continue

    async def _process_batches(self):
        """Async task for processing batches"""
        while self.running:
            try:
                try:
                    batch_data = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_batch(batch_data)

            except Exception as e:
                print(f"Error in process loop: {e}")
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