import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray

from Ayo.configs.config import EngineConfig
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.queries.query import Query

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class EngineRequest:
    """Data class for engine requests"""

    request_id: str  # unique id for each request, {query_id}_{node_name}
    query_id: str
    query: Query
    payload: Any  # Request payload (e.g., texts for embedding), generated from the payload_transformer, could be different for different engines
    result_ref: Optional[ray.ObjectRef] = None
    timestamp: float = time.time()


class BaseEngineScheduler(ABC):
    """Abstract base class for engine schedulers"""

    @abstractmethod
    async def submit_request(self, request: EngineRequest):
        """Submit a request to the engine"""
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the scheduler and all engine instances"""
        pass


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies"""

    @abstractmethod
    def get_next_engine(
        self, engines: List[Any], current_idx: int
    ) -> "tuple[Any, int]":
        """Get the next available engine

        Args:
            engines: list of engine instances
            current_idx: current engine index

        Returns:
            tuple[engine, new_idx]: selected engine and updated index
        """
        pass


class RoundRobinStrategy(SchedulingStrategy):
    """Round-robin scheduling strategy"""

    def get_next_engine(
        self, engines: List[Any], current_idx: int
    ) -> "tuple[Any, int]":
        if not engines:
            raise ValueError("No engines available")
        engine = engines[current_idx]
        new_idx = (current_idx + 1) % len(engines)
        return engine, new_idx


@ray.remote
class EngineScheduler(BaseEngineScheduler):
    def __init__(
        self,
        engine_class,
        engine_config: EngineConfig,  # use EngineConfig as config
        **engine_kwargs,
    ):

        self.engine_class = engine_class
        self.name = engine_config.name
        self.num_instances = engine_config.instances
        self.num_gpus = engine_config.num_gpus
        self.num_cpus = engine_config.num_cpus
        self.resources = engine_config.resources
        self.engine_kwargs = {
            **engine_kwargs,
            **(engine_config.model_config or {}),  # merge model_config
        }
        self.engines = []
        self._create_engines()

        # set scheduling strategy, default using round robin
        self.scheduling_strategy = RoundRobinStrategy()

        # create request queue
        self.request_queue = asyncio.Queue(maxsize=1000)

        # initialize current engine index
        self.current_engine_idx = 0

        self.pending_requests: Dict[str, EngineRequest] = {}

        # Start processing tasks
        self.running = True
        self.loop = asyncio.get_event_loop()
        self.submit_task = self.loop.create_task(self._submit_requests())
        self.result_task = self.loop.create_task(self._process_results())
        self._is_ready = True  # initialization complete flag

        # The queue for completed requests
        self.complete_queue = asyncio.Queue(maxsize=1000)

    def _create_engines(self):
        """create engine instances based on EngineConfig"""
        for i in range(self.num_instances):
            # create options dict, only include custom resources in resources
            options = {}
            if self.resources:
                custom_resources = {
                    k: v
                    for k, v in self.resources.items()
                    if k.upper() not in ["CPU", "GPU"]
                }
                if custom_resources:
                    options["resources"] = custom_resources

            if self.num_gpus:
                options["num_gpus"] = self.num_gpus
            if self.num_cpus:
                options["num_cpus"] = self.num_cpus
            else:
                options["num_cpus"] = 1

            # directly use the current actor's handle
            scheduler_handle = ray.runtime_context.get_runtime_context().current_actor

            # when creating engine instance, pass in scheduler handle
            logger.info(
                f"try to create engine instance with name: {self.name}_{i} for class: {self.engine_class} with options: {options}"
            )

            engine = self.engine_class.options(**options).remote(
                name=f"{self.name}_{i}",
                scheduler_ref=scheduler_handle,  # directly use actor handle
                **self.engine_kwargs,
            )

            _ = ray.get(engine.is_ready.remote())
            self.engines.append(engine)
            logger.info(
                f"created engine instance with name: {self.name}_{i} for class: {self.engine_class}"
            )

    def _get_next_engine(self):
        """use scheduling strategy to get next available engine"""
        engine, new_idx = self.scheduling_strategy.get_next_engine(
            self.engines, self.current_engine_idx
        )
        self.current_engine_idx = new_idx
        return engine

    async def add_engine(self):
        """add a new engine instance dynamically"""
        engine = self.engine_class.remote(
            name=f"{self.name}_{self.num_instances}",
            resource=self.resource,
            scheduler_ref=self,
            **self.engine_kwargs,
        )
        self.engines.append(engine)
        self.num_instances += 1
        return len(self.engines)

    async def remove_engine(self, index: Optional[int] = None):
        """remove an engine instance

        Args:
            index: the index of the engine to remove, if None, remove the last one
        """
        if not self.engines or self.num_instances <= 1:
            raise ValueError("Cannot remove the last engine instance")

        if index is None:
            index = len(self.engines) - 1

        if 0 <= index < len(self.engines):
            engine = self.engines.pop(index)
            await engine.shutdown.remote()
            self.num_instances -= 1

            # adjust current engine index
            if self.current_engine_idx >= len(self.engines):
                self.current_engine_idx = 0

            return len(self.engines)
        else:
            raise ValueError(f"Invalid engine index: {index}")

    async def _submit_requests(self):
        """Task for submitting requests to engines"""
        while self.running:
            try:
                # timeout 0.1 second to avoid busy waiting
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.05)
                logger.info(f"Processing request: {request.request_id}")

                # get available engine
                engine = self._get_next_engine()

                logger.info(
                    f"submit request {request.request_id} with payload: {request.payload.keys()}"
                )
                # submit request to engine
                await engine.submit_request.remote(
                    request_id=request.request_id,
                    query_id=request.query_id,
                    # texts=request.payload
                    **request.payload,
                )

                # track result
                self.pending_requests[request.request_id] = request
                self.request_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error submitting request for {request.request_id}: {e}")
                await asyncio.sleep(0.001)

    async def _process_results(self):
        """Task for processing completed results"""
        while self.running:
            try:
                request = await self.complete_queue.get()

                if request.result_ref is not None:
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, ray.get, request.result_ref
                        )
                        logger.debug(f"result in engine scheduler: {result}")

                        query_states = request.query.query_state
                        # Use ray.get to wait for result completion
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            ray.get,
                            query_states.set_node_result.remote(
                                request.request_id.split("::")[
                                    -1
                                ],  # here we assume the request_id is in the format of {query_id}::{node_name} as in the graph scheduler submit_node method
                                request.result_ref,
                            ),
                        )

                        logger.info(
                            f"Successfully set result for request {request.request_id}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Error processing result for request {request.request_id}: {e}"
                        )

                    finally:
                        # clean up completed request
                        self.complete_queue.task_done()
                        if request.request_id in self.pending_requests:
                            del self.pending_requests[request.request_id]

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in result processing loop: {e}")
                await asyncio.sleep(0.01)

    async def submit_request(self, request: EngineRequest):
        """Submit a request to the engine"""
        # Enqueue request instead of submitting directly
        try:
            logger.info(f"Enqueued request: {request.request_id}")
            await self.request_queue.put(request)
            self.pending_requests[request.request_id] = request
            logger.info(f"Enqueued request: {request.request_id}")

        except asyncio.QueueFull:
            logger.error(f"Request queue full, cannot enqueue {request.request_id}")
            raise

    async def on_result(
        self, request_id: str, query_id: str, result_ref_from_engine: Any
    ):
        """Handle result callback from engine"""
        if not isinstance(result_ref_from_engine, ray.ObjectRef):
            result_ref_from_engine = ray.put(result_ref_from_engine)
        try:
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                request.result_ref = result_ref_from_engine

                # put completed request into complete queue
                await self.complete_queue.put(request)
                logger.info(f"Moved request {request_id} to complete queue")

                # Clean up pending_requests
                del self.pending_requests[request_id]
            else:
                logger.warning(f"Received result for unknown request {request_id}")

        except Exception as e:
            logger.error(f"Error processing result callback: {e}")

    async def shutdown(self):
        """Shutdown the scheduler and all engine instances"""
        self.running = False

        # Cancel processing tasks
        for task in [self.submit_task, self.result_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown all engine instances
        for engine in self.engines:
            await engine.shutdown.remote()

        # Clear engine list
        self.engines.clear()
        self.num_instances = 0

    async def is_ready(self) -> bool:
        """Check if the scheduler is fully initialized

        Returns:
            bool: if the scheduler is fully initialized and ready to process requests
        """
        # check if all necessary components are initialized
        if not self.engines:
            return False

        # check if async tasks are running
        if not self.submit_task or self.submit_task.done():
            return False
        if not self.result_task or self.result_task.done():
            return False

        return self._is_ready
