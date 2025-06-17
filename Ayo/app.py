import asyncio
from asyncio import Future, Lock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Set, Type

import ray

from Ayo.configs.config import EngineConfig
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node
from Ayo.engines.base_engine import BaseEngine
from Ayo.engines.engine_types import ENGINE_REGISTRY, EngineType
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.opt_pass.base_pass import OPT_Pass
from Ayo.opt_pass.pass_manager import PassManager
from Ayo.queries.query import Query
from Ayo.queries.query_state import QueryStatus
from Ayo.schedulers.engine_scheduler import EngineScheduler
from Ayo.schedulers.graph_scheduler import GraphScheduler

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class QueryTask:
    query: Query
    config: Optional[Dict]
    future: Future
    timestamp: float = field(default_factory=time)  # Query arrival time
    status: str = "pending"
    timeout: float = 1.0  # Query timeout in seconds


class APP:
    """Main entry point for Ayo framework"""

    def __init__(self):
        """Initialize Ayo application"""
        # Configure logging

        self.engines: Dict[str, BaseEngine] = {}
        self.engine_schedulers: Dict[str, EngineScheduler] = {}
        self.optimization_passes: List[OPT_Pass] = []
        self.workflow_template: Dict[str, Node] = {}
        self.graph_scheduler = None
        self.pass_manager = PassManager()

        # Online serving configurations
        self.max_concurrent_queries = 100
        self.query_timeout = 1.0

        # Queue and concurrency control
        self.query_queue = asyncio.Queue()
        self.query_semaphore = asyncio.Semaphore(self.max_concurrent_queries)
        self.query_lock = asyncio.Lock()

        # Add task tracking
        self.tasks: Set[asyncio.Task] = set()

        self.base_dag = None

        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "timeout_queries": 0,
            "avg_latency": 0.0,
        }
        self.metrics_lock = Lock()

        # Add these attributes
        self.is_running = False
        self.active_queries = {}
        self.thread_pool = None  # Thread pool for DAG optimization

    @classmethod
    def init(cls):
        """Initialize a new Ayo application"""
        if not ray.is_initialized():
            ray.init()
        app = cls()
        app.graph_scheduler = GraphScheduler.remote({})
        return app

    def register_engine(self, config: EngineConfig) -> None:
        """Register an execution engine with proper configuration"""
        engine_type = config.engine_type
        try:
            if not EngineType.validate(engine_type):
                raise ValueError(f"Unsupported engine type: {engine_type}")

            engine_cls = self._get_engine_class(engine_type)

            # Get the default config and merge it with the user config
            default_config = ENGINE_REGISTRY.get_default_config(engine_type) or {}

            # First ensure the basic config fields are correct
            base_config = {
                "name": config.name,
                "engine_type": config.engine_type,
                "num_gpus": config.num_gpus,
                "num_cpus": config.num_cpus,
                "resources": config.resources,
                "instances": config.instances,
            }

            # Handle model_config separately
            model_config = {
                **(default_config.get("model_config", {}) or {}),
                **(config.model_config or {}),
            }
            base_config["model_config"] = model_config

            # Handle latency_profile separately
            if config.latency_profile or default_config.get("latency_profile"):
                base_config["latency_profile"] = {
                    **(default_config.get("latency_profile", {}) or {}),
                    **(config.latency_profile or {}),
                }

            if engine_type in self.engine_schedulers:
                raise ValueError(f"Engine scheduler {engine_type} already exists")

            # Create EngineConfig instance with correctly merged config
            engine_config = EngineConfig(**base_config)

            # Create the scheduler with the merged config
            scheduler_actor = EngineScheduler.remote(
                engine_class=engine_cls, engine_config=engine_config
            )

            # wait for the scheduler to be fully initialized
            _ = ray.get(scheduler_actor.is_ready.remote())

            logger.info(
                f"Registered engine {engine_type} with scheduler {scheduler_actor}"
            )

            self.engine_schedulers[engine_type] = scheduler_actor

            if self.graph_scheduler:
                ray.get(
                    self.graph_scheduler.update_schedulers.remote(
                        self.engine_schedulers
                    )
                )

        except Exception as e:
            logger.error(f"Failed to register engine {engine_type}: {str(e)}")
            raise

    def _get_engine_class(self, engine_type: str) -> Type[BaseEngine]:
        """Get engine class based on type"""
        engine_class = ENGINE_REGISTRY.get_engine_class(engine_type)
        if not engine_class:
            raise ValueError(f"Unknown engine type: {engine_type}")
        return engine_class

    def register_optimization(self, opt_pass: OPT_Pass) -> None:
        """Register an optimization pass

        Args:
            opt_pass: Optimization pass to register
        """
        self.pass_manager.register_pass(opt_pass)

    def update_template(self, dag: DAG) -> None:
        """Update workflow template with new nodes"""
        self.base_dag = dag

    async def submit_query(
        self,
        query: Query,
        config: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Future:

        logger.info(f"Received new query request: query_id={query.query_id}")
        # print(f"Received new query request: query_id={query.uuid}")
        future = Future()
        task = QueryTask(
            query=query,
            config=config,
            future=future,
            timeout=timeout or self.query_timeout,
        )

        async with self.query_lock:
            self.active_queries[query.query_id] = task
            await self.update_metrics("total_queries")
            logger.info(f"Query added to active queries: query_id={query.query_id}")

        await self.query_queue.put(task)
        logger.info(
            f"Query enqueued: query_id={query.query_id}, queue_size={self.query_queue.qsize()}"
        )
        return future

    async def start(self):
        """Start the application"""
        self.is_running = True
        self.thread_pool = ThreadPoolExecutor(max_workers=100)

        # Wait for all schedulers to be ready
        await asyncio.gather(
            *[
                scheduler.is_ready.remote()
                for scheduler in self.engine_schedulers.values()
            ]
        )

        # Create and track the queue processing task
        queue_task = asyncio.create_task(self._process_queue())
        self.tasks.add(queue_task)
        queue_task.add_done_callback(self.tasks.discard)

    async def _process_queue(self):
        """Main loop for processing query queue"""
        while self.is_running:
            try:
                try:
                    query_task = await asyncio.wait_for(
                        self.query_queue.get(), timeout=0.1
                    )
                    logger.info(
                        f"Retrieved query from queue: query_id={query_task.query.query_id}"
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in Retrieving query from queue: {str(e)}")
                    await asyncio.sleep(5)
                    continue

                if time() - query_task.timestamp > query_task.timeout:
                    logger.warning(
                        f"Query timeout detected: query_id={query_task.query.query_id}"
                    )
                    await self.handle_timeout(query_task)
                    self.query_queue.task_done()
                    continue

                logger.info(
                    f"Starting query processing: query_id={query_task.query.query_id}"
                )
                task = asyncio.create_task(
                    self._handle_query(query_task),
                    name=f"query_{query_task.query.query_id}",
                )
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

                self.query_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue processing: {str(e)}")
                await asyncio.sleep(5)

    async def _handle_query(self, query_task: QueryTask):
        query_id = query_task.query.query_id
        try:
            async with self.query_semaphore:
                logger.info(f"Starting DAG optimization: query_id={query_id}")
                await self.optimize_dag(query_task)

                logger.info(f"Submitting query to graph scheduler: query_id={query_id}")
                scheduler_query_id = await self.graph_scheduler.submit_query.remote(
                    query=query_task.query,
                    config=query_task.config or {},
                    # engine_schedulers=self.engine_schedulers
                )

                logger.info(f"Creating query monitor task: query_id={query_id}")
                asyncio.create_task(
                    self._monitor_query_status(query_task, scheduler_query_id),
                    name=f"monitor_{query_id}",
                )

        except Exception as e:
            logger.error(f"Error processing query: query_id={query_id}, error={str(e)}")
            await self.handle_error(query_task, e)

    async def _monitor_query_status(self, query_task: QueryTask, query_id: str):

        task_id = query_task.query.query_id
        logger.info(f"Starting query status monitoring: query_id={task_id}")

        try:
            start_time = time()

            while True:
                if time() - start_time > query_task.timeout:
                    logger.warning(f"Query monitoring timeout: query_id={task_id}")
                    await self.handle_timeout(query_task)
                    break

                status = ray.get(
                    self.graph_scheduler.get_query_status.remote(
                        query_task.query.query_id
                    )
                )

                # logging.info(f"Query status update: query_id={task_id}, status={status}")

                if status == QueryStatus.COMPLETED:
                    latency = time() - start_time
                    await self.update_metrics("avg_latency", latency)
                    logger.info(
                        f"Query completed successfully: query_id={task_id}, duration={latency:.3f}s"
                    )
                    print(f"query_task.query.results: {query_task.query.results}")

                    node_results = ray.get(
                        query_task.query.query_state.get_node_results.remote()
                    )
                    query_task.future.set_result(node_results)
                    await self.update_metrics("successful_queries")
                    break
                elif status == QueryStatus.FAILED:
                    logger.error(f"Query execution failed: query_id={task_id}")
                    raise Exception(f"Query failed: {query_task.query.error_message}")

                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(
                f"Error in query monitoring: query_id={task_id}, error={str(e)}"
            )
            await self.handle_error(query_task, e)
        finally:
            async with self.query_lock:
                self.active_queries.pop(query_task.query.query_id, None)
                logger.info(f"Query removed from active list: query_id={task_id}")

    async def optimize_dag(self, query_task: QueryTask):
        """Execute DAG optimization asynchronously"""

        def _optimize():
            query_task.query.start()
            enabled_passes = self.pass_manager.get_enabled_passes()
            query_task.query.DAG.optimize(enabled_passes)

        await asyncio.get_event_loop().run_in_executor(self.thread_pool, _optimize)

    async def handle_timeout(self, query_task: QueryTask):
        """Handle query timeout"""
        query_task.query.set_timeout()
        query_task.future.set_exception(
            TimeoutError(f"Query {query_task.query.query_id} timed out")
        )
        await self.update_metrics("timeout_queries")

    async def handle_error(self, query_task: QueryTask, error: Exception):
        """Handle query error"""
        query_task.query.fail(str(error))
        query_task.future.set_exception(error)
        await self.update_metrics("failed_queries")

    async def update_metrics(self, metric_name: str, value: Any = 1):
        """Update performance metrics"""
        async with self.metrics_lock:
            if metric_name == "avg_latency":
                self.metrics["avg_latency"] = (
                    0.9 * self.metrics["avg_latency"] + 0.1 * value
                )
            else:
                self.metrics[metric_name] += value

    async def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        async with self.metrics_lock:
            return self.metrics.copy()

    def shutdown(self):
        """Shutdown the application and all its components"""
        # Shutdown graph scheduler
        if self.graph_scheduler:
            ray.get(self.graph_scheduler.shutdown.remote())

        # Shutdown all engine schedulers
        for scheduler in self.engine_schedulers.values():
            ray.get(scheduler.shutdown.remote())

        # Shutdown ray
        ray.shutdown()

    async def stop(self):
        """Stop the application gracefully"""
        self.is_running = False

        # Cancel all tracked tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Clean up thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        # Clean up remaining queries
        while not self.query_queue.empty():
            try:
                task = self.query_queue.get_nowait()
                task.future.set_exception(RuntimeError("Application shutting down"))
            except asyncio.QueueEmpty:
                break
