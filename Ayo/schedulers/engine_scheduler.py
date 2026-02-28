import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    timestamp: float = field(default_factory=time.time)
    node_name: str = ""
    node_depth: int = 0
    op_type: Optional[str] = None
    arrival_ts: float = field(default_factory=time.time)


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
        scheduler_cfg = engine_kwargs.pop("scheduler_config", None) or {}

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
        self.pending_by_query: Dict[str, Dict[str, List[EngineRequest]]] = {}
        self.pending_by_query_ts: Dict[str, float] = {}

        engine_scheduler_cfg = self._get_engine_scheduler_config(
            scheduler_cfg, engine_config.engine_type
        )
        # scheduler config: default follows global toggle; per-engine override is optional.
        global_topology_aware = bool(scheduler_cfg.get("topology_aware", False))
        if "topology_aware" in engine_scheduler_cfg:
            self.topology_aware = bool(engine_scheduler_cfg.get("topology_aware"))
        else:
            self.topology_aware = global_topology_aware
        self.batch_policy = (
            scheduler_cfg.get("batch_policy") or "strict_compat"
        ).lower()
        if self.batch_policy not in {"strict_compat", "engine_aligned"}:
            self.batch_policy = "strict_compat"

        self.max_batch_size = self._resolve_scheduler_max_batch_size(
            scheduler_cfg=scheduler_cfg,
            engine_scheduler_cfg=engine_scheduler_cfg,
            engine_type=engine_config.engine_type,
            model_config=engine_config.model_config or {},
        )
        self.max_wait_ms = self._resolve_max_wait_ms(
            scheduler_cfg=scheduler_cfg,
            engine_scheduler_cfg=engine_scheduler_cfg,
        )
        fill_strategy = (
            engine_scheduler_cfg.get("fill_strategy")
            if "fill_strategy" in engine_scheduler_cfg
            else scheduler_cfg.get("fill_strategy")
        )
        fill_strategy = (fill_strategy or "none").lower()
        self.fill_strategy = (
            fill_strategy
            if fill_strategy in {"none", "same_query", "same_op"}
            else "none"
        )
        self.min_pending_queries_for_topology = (
            self._resolve_min_pending_queries_for_topology(
                scheduler_cfg=scheduler_cfg,
                engine_scheduler_cfg=engine_scheduler_cfg,
            )
        )

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
                if not self.topology_aware:
                    # Preserve original behavior when feature is disabled.
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=0.05
                    )
                    logger.info(f"Processing request: {request.request_id}")

                    engine = self._get_next_engine()
                    logger.info(
                        f"submit request {request.request_id} with payload: {request.payload.keys()}"
                    )
                    await engine.submit_request.remote(
                        request_id=request.request_id,
                        query_id=request.query_id,
                        **request.payload,
                    )
                    self.pending_requests[request.request_id] = request
                    self.request_queue.task_done()
                    continue

                # Topology-aware path with bounded queue drain and optional short wait.
                drain_limit = max(self.max_batch_size * 4, 64)
                drained_count = self._bounded_drain_from_queue(
                    self.request_queue, self._enqueue_request, drain_limit
                )

                if drained_count == 0 and not self.pending_by_query:
                    await asyncio.sleep(0.001)
                    continue

                pending_count = self._pending_request_count()
                pending_query_count = self._pending_query_count()
                if (
                    drained_count > 0
                    and self.max_wait_ms > 0
                    and pending_count < self.max_batch_size
                ):
                    await asyncio.sleep(self.max_wait_ms / 1000.0)
                    drained_count += self._bounded_drain_from_queue(
                        self.request_queue,
                        self._enqueue_request,
                        drain_limit - drained_count,
                    )

                pending_query_count = self._pending_query_count()
                if pending_query_count < self.min_pending_queries_for_topology:
                    batch = self._select_batch_fifo_from_pending(
                        self.pending_by_query,
                        self.pending_by_query_ts,
                    )
                else:
                    batch = self._select_batch_topology()
                if not batch:
                    await asyncio.sleep(0.001)
                    continue

                for request in batch:
                    logger.info(f"Processing request: {request.request_id}")
                    engine = self._get_next_engine()
                    logger.info(
                        f"submit request {request.request_id} with payload: {request.payload.keys()}"
                    )
                    await engine.submit_request.remote(
                        request_id=request.request_id,
                        query_id=request.query_id,
                        **request.payload,
                    )
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

    def _enqueue_request(self, request: EngineRequest) -> None:
        query_id = request.query_id
        node_name = request.node_name or request.request_id
        if query_id not in self.pending_by_query:
            self.pending_by_query[query_id] = {}
        if query_id not in self.pending_by_query_ts:
            self.pending_by_query_ts[query_id] = request.arrival_ts or request.timestamp

        node_bucket = self.pending_by_query[query_id].setdefault(node_name, [])
        node_bucket.append(request)

        ts = request.arrival_ts or request.timestamp
        if ts < self.pending_by_query_ts[query_id]:
            self.pending_by_query_ts[query_id] = ts

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_engine_scheduler_config(
        scheduler_cfg: Dict[str, Any], engine_type: str
    ) -> Dict[str, Any]:
        per_engine = scheduler_cfg.get("per_engine", {})
        if not isinstance(per_engine, dict):
            return {}
        engine_type_value = getattr(engine_type, "value", engine_type)
        engine_cfg = per_engine.get(engine_type)
        if engine_cfg is None:
            engine_cfg = per_engine.get(engine_type_value)
        if engine_cfg is None:
            engine_cfg = per_engine.get(str(engine_type))
        return engine_cfg if isinstance(engine_cfg, dict) else {}

    @staticmethod
    def _resolve_scheduler_max_batch_size(
        scheduler_cfg: Dict[str, Any],
        engine_scheduler_cfg: Dict[str, Any],
        engine_type: str,
        model_config: Dict[str, Any],
    ) -> int:
        policy = (scheduler_cfg.get("batch_policy") or "strict_compat").lower()
        if policy not in {"strict_compat", "engine_aligned"}:
            policy = "strict_compat"

        hard_cap = max(
            1,
            EngineScheduler._safe_int(
                scheduler_cfg.get("max_batch_size_cap", 256), 256
            ),
        )

        explicit = (
            engine_scheduler_cfg.get("max_batch_size")
            if "max_batch_size" in engine_scheduler_cfg
            else scheduler_cfg.get("max_batch_size")
        )
        if explicit is not None:
            return max(1, min(EngineScheduler._safe_int(explicit, 1), hard_cap))

        if policy == "engine_aligned":
            engine_type_value = getattr(engine_type, "value", engine_type)
            inferred = model_config.get("max_batch_size")
            if inferred is None and engine_type_value == "llm":
                inferred = model_config.get("max_num_seqs")
            if inferred is not None:
                return max(1, min(EngineScheduler._safe_int(inferred, 1), hard_cap))

        return 1

    @staticmethod
    def _resolve_max_wait_ms(
        scheduler_cfg: Dict[str, Any], engine_scheduler_cfg: Dict[str, Any]
    ) -> int:
        explicit = (
            engine_scheduler_cfg.get("max_wait_ms")
            if "max_wait_ms" in engine_scheduler_cfg
            else scheduler_cfg.get("max_wait_ms", 0)
        )
        return max(0, EngineScheduler._safe_int(explicit, 0))

    @staticmethod
    def _resolve_min_pending_queries_for_topology(
        scheduler_cfg: Dict[str, Any], engine_scheduler_cfg: Dict[str, Any]
    ) -> int:
        explicit = (
            engine_scheduler_cfg.get("min_pending_queries_for_topology")
            if "min_pending_queries_for_topology" in engine_scheduler_cfg
            else scheduler_cfg.get("min_pending_queries_for_topology", 1)
        )
        return max(1, EngineScheduler._safe_int(explicit, 1))

    @staticmethod
    def _bounded_drain_from_queue(
        queue: asyncio.Queue,
        on_request: Any,
        limit: int,
    ) -> int:
        if limit <= 0:
            return 0
        drained = 0
        while drained < limit:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            on_request(request)
            drained += 1
        return drained

    async def get_scheduler_settings(self) -> Dict[str, Any]:
        return {
            "topology_aware": self.topology_aware,
            "batch_policy": self.batch_policy,
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "fill_strategy": self.fill_strategy,
            "min_pending_queries_for_topology": self.min_pending_queries_for_topology,
        }

    @staticmethod
    def _normalize_depth(node_depth: Optional[int]) -> int:
        if node_depth is None:
            return 0
        return -int(node_depth)

    @staticmethod
    def _select_batch_fifo_from_pending(
        pending_by_query: Dict[str, Dict[str, List[EngineRequest]]],
        pending_by_query_ts: Dict[str, float],
    ) -> List[EngineRequest]:
        if not pending_by_query:
            return []

        query_order = sorted(
            pending_by_query.keys(),
            key=lambda q: pending_by_query_ts.get(q, 0.0),
        )
        for query_id in query_order:
            nodes = pending_by_query.get(query_id, {})
            if not nodes:
                continue

            oldest = []
            for node_name, node_reqs in nodes.items():
                if not node_reqs:
                    continue
                arrival_ts = node_reqs[0].arrival_ts or node_reqs[0].timestamp
                oldest.append((arrival_ts, node_name))
            if not oldest:
                pending_by_query.pop(query_id, None)
                pending_by_query_ts.pop(query_id, None)
                continue

            oldest.sort()
            _, node_name = oldest[0]
            request = nodes[node_name].pop(0)
            if not nodes[node_name]:
                nodes.pop(node_name, None)
            if not nodes:
                pending_by_query.pop(query_id, None)
                pending_by_query_ts.pop(query_id, None)
            return [request]

        return []

    @staticmethod
    def _select_batch_topology_from_pending(
        pending_by_query: Dict[str, Dict[str, List[EngineRequest]]],
        pending_by_query_ts: Dict[str, float],
        max_batch_size: int,
        fill_strategy: str = "none",
    ) -> List[EngineRequest]:
        batch: List[EngineRequest] = []
        if max_batch_size <= 0:
            return batch

        fill_strategy = (fill_strategy or "none").lower()
        query_order = sorted(
            pending_by_query.keys(),
            key=lambda q: pending_by_query_ts.get(q, 0.0),
        )

        for query_id in query_order:
            if len(batch) >= max_batch_size:
                break

            nodes = pending_by_query.get(query_id, {})
            if not nodes:
                continue

            deepest_rank = None
            for _, node_reqs in nodes.items():
                if not node_reqs:
                    continue
                depth_rank = EngineScheduler._normalize_depth(node_reqs[0].node_depth)
                if deepest_rank is None or depth_rank > deepest_rank:
                    deepest_rank = depth_rank

            if deepest_rank is None:
                continue

            candidates = []
            for node_name, node_reqs in nodes.items():
                if not node_reqs:
                    continue
                depth_rank = EngineScheduler._normalize_depth(node_reqs[0].node_depth)
                if depth_rank == deepest_rank:
                    arrival_ts = node_reqs[0].arrival_ts or node_reqs[0].timestamp
                    candidates.append((arrival_ts, node_name))

            candidates.sort()
            slots = max_batch_size - len(batch)
            for _, node_name in candidates:
                if slots <= 0:
                    break
                node_reqs = nodes[node_name]
                take = min(len(node_reqs), slots)
                batch.extend(node_reqs[:take])
                del node_reqs[:take]
                slots -= take
                if not node_reqs:
                    nodes.pop(node_name, None)

            if fill_strategy == "same_query" and slots > 0:
                remaining = []
                for node_name, node_reqs in nodes.items():
                    if not node_reqs:
                        continue
                    depth_rank = EngineScheduler._normalize_depth(
                        node_reqs[0].node_depth
                    )
                    arrival_ts = node_reqs[0].arrival_ts or node_reqs[0].timestamp
                    remaining.append((-depth_rank, arrival_ts, node_name))

                remaining.sort()
                for _, __, node_name in remaining:
                    if slots <= 0:
                        break
                    node_reqs = nodes[node_name]
                    take = min(len(node_reqs), slots)
                    batch.extend(node_reqs[:take])
                    del node_reqs[:take]
                    slots -= take
                    if not node_reqs:
                        nodes.pop(node_name, None)

            if not nodes:
                pending_by_query.pop(query_id, None)
                pending_by_query_ts.pop(query_id, None)

        if fill_strategy == "same_op" and len(batch) < max_batch_size:
            EngineScheduler._fill_batch_same_op_from_pending(
                pending_by_query=pending_by_query,
                pending_by_query_ts=pending_by_query_ts,
                batch=batch,
                max_batch_size=max_batch_size,
            )

        return batch

    @staticmethod
    def _fill_batch_same_op_from_pending(
        pending_by_query: Dict[str, Dict[str, List[EngineRequest]]],
        pending_by_query_ts: Dict[str, float],
        batch: List[EngineRequest],
        max_batch_size: int,
    ) -> None:
        if not batch or len(batch) >= max_batch_size:
            return

        op_counts: Dict[Any, int] = {}
        first_seen: Dict[Any, int] = {}
        for idx, req in enumerate(batch):
            op = req.op_type
            if op is None:
                continue
            op_counts[op] = op_counts.get(op, 0) + 1
            if op not in first_seen:
                first_seen[op] = idx
        if not op_counts:
            return

        target_op = min(
            op_counts.keys(), key=lambda op: (-op_counts[op], first_seen[op])
        )

        candidates = []
        query_order = sorted(
            pending_by_query.keys(),
            key=lambda q: pending_by_query_ts.get(q, 0.0),
        )
        for query_id in query_order:
            nodes = pending_by_query.get(query_id, {})
            for node_name, node_reqs in nodes.items():
                if not node_reqs:
                    continue
                if node_reqs[0].op_type != target_op:
                    continue
                depth_rank = EngineScheduler._normalize_depth(node_reqs[0].node_depth)
                arrival_ts = node_reqs[0].arrival_ts or node_reqs[0].timestamp
                candidates.append((-depth_rank, arrival_ts, query_id, node_name))

        candidates.sort()
        slots = max_batch_size - len(batch)
        for _, __, query_id, node_name in candidates:
            if slots <= 0:
                break
            nodes = pending_by_query.get(query_id)
            if not nodes or node_name not in nodes:
                continue
            node_reqs = nodes[node_name]
            if not node_reqs or node_reqs[0].op_type != target_op:
                continue

            take = min(len(node_reqs), slots)
            batch.extend(node_reqs[:take])
            del node_reqs[:take]
            slots -= take

            if not node_reqs:
                nodes.pop(node_name, None)
            if not nodes:
                pending_by_query.pop(query_id, None)
                pending_by_query_ts.pop(query_id, None)

    def _select_batch_topology(self) -> List[EngineRequest]:
        return self._select_batch_topology_from_pending(
            self.pending_by_query,
            self.pending_by_query_ts,
            self.max_batch_size,
            self.fill_strategy,
        )

    def _pending_request_count(self) -> int:
        total = 0
        for query_nodes in self.pending_by_query.values():
            for node_reqs in query_nodes.values():
                total += len(node_reqs)
        return total

    def _pending_query_count(self) -> int:
        return sum(1 for qnodes in self.pending_by_query.values() if qnodes)

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

    async def cleanup_query(self, query_id: str) -> None:
        """Best-effort per-query cleanup on underlying engines."""
        for engine in self.engines:
            cleanup_method = getattr(engine, "cleanup_query", None)
            if cleanup_method is None:
                continue
            try:
                await cleanup_method.remote(query_id)
            except Exception as e:
                logger.warning(
                    f"cleanup_query failed for query {query_id} on "
                    f"engine scheduler {self.name}: {e}"
                )

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
