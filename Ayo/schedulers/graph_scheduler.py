import asyncio
import os
import time
import traceback
from collections import deque
from typing import Any, Dict, Optional

import ray

from Ayo.configs.model_config import AggMode, get_aggregator_config
from Ayo.dags.node import Node, NodeOps, NodeStatus, NodeType
from Ayo.engines.payload_transformers import TRANSFORMER_REGISTRY, DefaultTransformer
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from Ayo.queries.query import Query
from Ayo.queries.query_state import QueryStatus
from Ayo.schedulers.engine_scheduler import EngineRequest

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class QueryRunner:
    """A class for running a query"""

    def __init__(
        self,
        query: Query,
        config: Dict,
        engine_schedulers: Dict[str, ray.actor.ActorHandle],
    ):

        self.query = query
        self.config = config
        self.dag = query.DAG
        self.engine_schedulers = engine_schedulers

        self.running = deque()
        self.ready = deque()
        self.pending = deque()

        self.nodes_outputs = {}
        self.cancelled = False

        self.input_nodes = []
        self.compute_nodes = []
        self.output_nodes = []

        logger.info(f"query.query_state in query runner: {self.query.query_state}")

    def initialize(self):
        """Initialize the runner"""
        self.dag.topological_sort()

        for node in self.dag.topo_list:
            if node.node_type == NodeType.INPUT:
                self.input_nodes.append(node)
            elif node.node_type == NodeType.COMPUTE:
                self.compute_nodes.append(node)
            elif node.node_type == NodeType.OUTPUT:
                self.output_nodes.append(node)

        for node in self.input_nodes:
            node.status = NodeStatus.COMPLETED
            self.nodes_outputs[node.name] = node.input_values

        self.pending.extend(self.compute_nodes)
        self.pending.extend(self.output_nodes)

        logger.info(f"DAG input nodes: {self.input_nodes}")
        logger.info(f"DAG compute nodes: {self.compute_nodes}")
        logger.info(f"DAG output nodes: {self.output_nodes}")

    def prepare_engine_payload(self, node: Node) -> Any:
        """prepare the data to send to the engine"""
        if node.node_type != NodeType.COMPUTE:
            return node.input_kwargs

        transformer = TRANSFORMER_REGISTRY.get(node.engine_type, DefaultTransformer())

        return transformer.transform(node)

    async def check_node_ready(self, node: Node) -> bool:
        """Check if a node is ready to execute"""
        if node.node_type == NodeType.INPUT:
            return True

        elif node.op_type in [NodeOps.LLM_DECODING, NodeOps.LLM_PARTIAL_DECODING]:
            parent_op_types = [parent.op_type for parent in node.parents]
            if NodeOps.LLM_PARTIAL_DECODING in parent_op_types:
                return node.parents[
                    parent_op_types.index(NodeOps.LLM_PARTIAL_DECODING)
                ].status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
            else:
                potential_prefilling_status = [
                    parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
                    for parent in node.parents
                    if parent.op_type == NodeOps.LLM_PREFILLING
                ]
                potential_full_prefilling_status = [
                    parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
                    for parent in node.parents
                    if parent.op_type == NodeOps.LLM_FULL_PREFILLING
                ]

                # Check if at least one parent node is prefilling type, and all such parent nodes are ready
                has_prefilling_parents = len(potential_prefilling_status) > 0
                has_full_prefilling_parents = len(potential_full_prefilling_status) > 0

                if (has_prefilling_parents and all(potential_prefilling_status)) or (
                    has_full_prefilling_parents
                    and all(potential_full_prefilling_status)
                ):
                    return True
                else:
                    return False

        elif node.op_type == NodeOps.LLM_FULL_PREFILLING:
            llm_partial_prefilling_status = [
                parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
                for parent in node.parents
                if parent.op_type == NodeOps.LLM_PARTIAL_PREFILLING
            ]
            other_parents_status = [
                parent.status in [NodeStatus.COMPLETED]
                for parent in node.parents
                if parent.op_type not in [NodeOps.LLM_PARTIAL_PREFILLING]
            ]
            if all(llm_partial_prefilling_status) and all(other_parents_status):
                return True
            else:
                return False

        else:
            return all(parent.status == NodeStatus.COMPLETED for parent in node.parents)

    async def submit_node(self, node: Node):
        """submit a node to the engine scheduler"""
        try:
            logger.info(f"submit node: {node.name}")
            if node.node_type == NodeType.INPUT:
                return

            if node.node_type == NodeType.OUTPUT:
                if (
                    len(node.parents) == 1
                    and node.parents[0].status == NodeStatus.COMPLETED
                ):
                    node.status = NodeStatus.COMPLETED
                    node.update_input_kwargs(self.nodes_outputs)
                    self.nodes_outputs[node.name] = node.input_kwargs
                return

            if node.op_type == NodeOps.AGGREGATOR:
                # we do the in-place aggregation here, do not need to submit to the aggregator engine which would be discarded later
                aggregator_mode = get_aggregator_config(node)["agg_mode"]

                logger.debug(f"aggregator node {node}, agg mode {aggregator_mode}")

                if aggregator_mode == AggMode.DUMMY:
                    node.status = NodeStatus.COMPLETED
                    self.nodes_outputs[node.name] = {k: True for k in node.output_names}

                    await self.query.query_state.set_node_result.remote(
                        node.name, self.nodes_outputs[node.name]
                    )

                    return
                elif aggregator_mode == AggMode.MERGE:
                    node.status = NodeStatus.COMPLETED
                    node.update_input_kwargs(self.nodes_outputs)
                    if isinstance(list(node.input_kwargs.values())[0], dict):
                        output = {}
                    elif isinstance(list(node.input_kwargs.values())[0], list):
                        output = []
                    else:
                        raise ValueError(
                            f"Unsupported input type: {type(list(node.input_kwargs.values())[0])}"
                        )

                    for k, v in node.input_kwargs.items():
                        if isinstance(v, dict):
                            output.update(v)
                        elif isinstance(v, list):
                            output.extend(v)
                        else:
                            raise ValueError(f"Unsupported input type: {type(v)}")

                    self.nodes_outputs[node.name] = output
                    await self.query.query_state.set_node_result.remote(
                        node.name, self.nodes_outputs[node.name]
                    )
                    return

                elif aggregator_mode == AggMode.TOP_K:
                    # assume each output is a list of dicts or tuples
                    # if is dict, {"text": "xxx", "score": 0.9}
                    # if is tuple, ("xxx", 0.9)
                    node.status = NodeStatus.COMPLETED
                    node.update_input_kwargs(self.nodes_outputs)

                    logger.debug(
                        f"aggregator: {node.name} input_kwargs: {node.input_kwargs}"
                    )
                    top_k = node.config.get("top_k", None)
                    if top_k is None:
                        raise ValueError(
                            f"Missing top_k for node {node.name} aggregator"
                        )
                    output = []
                    for k, v in node.input_kwargs.items():
                        if isinstance(v, list):
                            if isinstance(v[0], dict):
                                keys = v[0].keys()
                                if "score" not in keys:
                                    raise ValueError(
                                        f"Dict keys do not contain 'score': {keys}"
                                    )
                                score_key = "score"
                                # Assume there is only one non-"score" field is the text
                                text_keys = [key for key in keys if key != "score"]
                                if not text_keys:
                                    raise ValueError(
                                        f"No text key found in dict: {keys}"
                                    )
                                text_key = text_keys[0]

                                output.extend(
                                    (v[i][text_key], v[i][score_key])
                                    for i in range(len(v))
                                )
                            elif isinstance(v[0], tuple):
                                if isinstance(v[0][0], str):
                                    text_key = 0
                                    score_key = 1
                                else:
                                    text_key = 1
                                    score_key = 0
                                output.extend(
                                    (v[i][text_key], v[i][score_key])
                                    for i in range(len(v))
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported input type: {type(v[0])}"
                                )
                        else:
                            raise ValueError(f"Unsupported input type: {type(v)}")

                    # sort the output by score
                    output.sort(key=lambda x: x[1], reverse=True)
                    # keep the top k
                    output = output[:top_k]
                    # only keep the text
                    output = [x[0] for x in output]
                    self.nodes_outputs[node.name] = output
                    logger.info(
                        f"top {top_k} aggregation output for node: {node.name} output: {output}"
                    )
                    await self.query.query_state.set_node_result.remote(
                        node.name, self.nodes_outputs[node.name]
                    )
                    return
                else:
                    raise ValueError(
                        f"Unsupported aggregator mode: {get_aggregator_config(node)}"
                    )

            # Address the compute node
            node.update_input_kwargs(self.nodes_outputs)

            payload = self.prepare_engine_payload(node)

            engine_type = node.engine_type
            scheduler = self.engine_schedulers.get(engine_type)

            if scheduler:
                engine_request = EngineRequest(
                    request_id=f"{self.query.query_id}::{node.name}",  # the engine scheduler will use the request_id.split("::")[-1] to get the node_name
                    query_id=self.query.query_id,
                    query=self.query,
                    payload=payload,
                    node_name=node.name,
                    node_depth=node.depth,
                    op_type=node.op_type,
                    arrival_ts=time.time(),
                )

                await scheduler.submit_request.remote(engine_request)
                node.status = NodeStatus.RUNNING
            else:
                raise ValueError(f"No scheduler found for engine type: {engine_type}")

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(
                f"Error submitting node {node.name}: {str(e)}\nTraceback:\n{error_trace}"
            )
            node.status = NodeStatus.FAILED
            node.error_message = f"{str(e)}\nTraceback:\n{error_trace}"
            self.dag.error_nodes.append(node)

    async def cleanup_runtime_context(self):
        """Cleanup runtime context"""
        self.input_nodes.clear()
        self.compute_nodes.clear()
        self.output_nodes.clear()
        self.running.clear()
        self.ready.clear()
        self.pending.clear()


@ray.remote
class GraphScheduler:
    """Scheduler for managing multiple queries and their DAG execution"""

    def __init__(self, engine_schedulers: Dict[str, ray.actor.ActorHandle]):
        self.engine_schedulers = engine_schedulers
        self.query_runners: Dict[str, QueryRunner] = {}
        self.query_status_cache: Dict[str, QueryStatus] = {}
        self._dag_tasks = set()
        self._query_futures: Dict[str, asyncio.Future] = {}
        self._query_outcomes: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
        self._outcome_ttl_s = float(os.getenv("AYO_QUERY_OUTCOME_TTL_S", "300"))
        self._cleanup_interval_s = float(
            os.getenv("AYO_QUERY_OUTCOME_CLEANUP_INTERVAL_S", "60")
        )

    async def update_schedulers(
        self, engine_schedulers: Dict[str, ray.actor.ActorHandle]
    ) -> None:
        """Update engine schedulers

        Args:
            engine_schedulers: Dictionary of engine type to scheduler actor handle
        """
        self.engine_schedulers = engine_schedulers

    async def submit_query(self, query: Query, config: Optional[Dict] = None) -> str:
        """Submit a new query asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            self._query_futures[query.query_id] = loop.create_future()
            if self._cleanup_task is None:
                self._cleanup_task = loop.create_task(self._ttl_cleanup_loop())
            runner = QueryRunner(
                query=query, config=config, engine_schedulers=self.engine_schedulers
            )
            self.query_runners[query.query_id] = runner

            runner.initialize()
            dag_task = asyncio.create_task(self._process_dag(runner))
            self._dag_tasks.add(dag_task)
            dag_task.add_done_callback(self._dag_tasks.discard)

            return query.query_id

        except Exception as e:
            logger.error(f"Failed to submit query {query.query_id}: {str(e)}")
            future = self._query_futures.pop(query.query_id, None)
            if future and not future.done():
                future.set_exception(e)
            raise

    async def _process_dag(self, runner: QueryRunner):
        """Process DAG execution asynchronously"""
        dag_error = None
        try:
            last_pending_nodes_names = []
            last_running_nodes_names = []
            while not runner.dag.check_completion():
                if runner.cancelled:
                    runner.query.status = QueryStatus.TIMEOUT
                    break
                ready_nodes = [
                    node
                    for node in runner.pending
                    if await runner.check_node_ready(node)
                ]

                for node in ready_nodes:
                    runner.ready.append(node)
                    runner.pending.remove(node)

                while runner.ready:
                    nodes_to_submit = []
                    while runner.ready:
                        nodes_to_submit.append(runner.ready.popleft())

                    await asyncio.gather(
                        *[runner.submit_node(node) for node in nodes_to_submit]
                    )
                    runner.running.extend(nodes_to_submit)

                current_pending_nodes_names = [node.name for node in runner.pending]
                current_running_nodes_names = [node.name for node in runner.running]

                if (
                    last_pending_nodes_names != current_pending_nodes_names
                    or last_running_nodes_names != current_running_nodes_names
                ):
                    logger.info(f"pending nodes: {current_pending_nodes_names}")
                    logger.info(f"running nodes: {current_running_nodes_names}")

                    last_pending_nodes_names = current_pending_nodes_names
                    last_running_nodes_names = current_running_nodes_names

                if runner.running:
                    try:
                        completed_nodes = []
                        results = await asyncio.gather(
                            *[
                                asyncio.wrap_future(
                                    runner.query.query_state.get_node_result.remote(
                                        node.name
                                    ).future()
                                )
                                for node in runner.running
                            ],
                            return_exceptions=True,
                        )

                        for node, result in zip(runner.running, results):
                            if isinstance(result, Exception):
                                node.status = NodeStatus.FAILED
                                node.error_message = str(result)
                                runner.dag.error_nodes.append(node)
                                logger.error(
                                    f"Node execution failed: {node.name}: {str(result)}"
                                )
                                raise Exception(
                                    f"DAG execution terminated due to node failure: {node.name} - {str(result)}"
                                )

                            elif result is not None:
                                logger.info(
                                    f"result have been received for node: {node.name} in graph scheduler"
                                )
                                runner.nodes_outputs[node.name] = {
                                    node.output_names[0]: result
                                }
                                node.status = NodeStatus.COMPLETED
                                completed_nodes.append(node)

                        for node in completed_nodes:
                            runner.running.remove(node)

                    except Exception as e:
                        logger.error(f"Error in DAG execution: {str(e)}")
                        for node in runner.running:
                            node.status = NodeStatus.FAILED
                            node.error_message = str(e)
                            runner.dag.error_nodes.append(node)
                        runner.running.clear()
                        runner.query.status = QueryStatus.FAILED
                        dag_error = e
                        break

                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(
                f"DAG execution failed for query {runner.query.query_id}: {str(e)}"
            )
            runner.dag.error_nodes.extend(runner.running)
            for node in runner.running:
                node.status = NodeStatus.FAILED
                node.error_message = str(e)
            runner.query.status = QueryStatus.FAILED
            dag_error = e
        finally:
            if (
                not runner.dag.error_nodes
                and runner.query.status != QueryStatus.TIMEOUT
            ):
                runner.query.status = QueryStatus.COMPLETED
            final_result = None
            if runner.query.status == QueryStatus.COMPLETED:
                try:
                    final_result = await asyncio.wrap_future(
                        runner.query.query_state.get_node_results.remote().future()
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch node_results for query {runner.query.query_id}: {e}"
                    )
                    final_result = runner.nodes_outputs

            future = self._query_futures.get(runner.query.query_id)
            if future and not future.done():
                if runner.query.status == QueryStatus.COMPLETED:
                    future.set_result(final_result)
                else:
                    err = dag_error or Exception(
                        f"DAG failed: {[n.name for n in runner.dag.error_nodes]}"
                    )
                    future.set_exception(err)

            self._query_outcomes[runner.query.query_id] = {
                "status": runner.query.status,
                "result": (
                    final_result
                    if runner.query.status == QueryStatus.COMPLETED
                    else None
                ),
                "error": str(dag_error) if dag_error else None,
                "finished_ts": time.time(),
            }
            await self.cleanup_query(runner.query.query_id)

    async def get_query_status(self, query_id: str) -> QueryStatus:
        """Get query status"""
        if query_id in self.query_runners:
            runner = self.query_runners[query_id]
            return runner.query.status
        return self.query_status_cache.get(query_id)

    async def wait_for_query(
        self, query_id: str, timeout_s: Optional[float] = None
    ) -> Any:
        future = self._query_futures.get(query_id)
        if future is not None:
            if timeout_s is None:
                return await future
            return await asyncio.wait_for(future, timeout=timeout_s)

        outcome = self._query_outcomes.get(query_id)
        if outcome is None:
            raise ValueError(f"Unknown query_id: {query_id}")
        if outcome["status"] == QueryStatus.COMPLETED:
            return outcome["result"]
        raise RuntimeError(outcome.get("error") or f"Query {query_id} failed")

    async def cancel_query(self, query_id: str):
        runner = self.query_runners.get(query_id)
        if runner:
            runner.cancelled = True
            runner.query.status = QueryStatus.TIMEOUT
        future = self._query_futures.get(query_id)
        if future and not future.done():
            future.set_exception(asyncio.CancelledError(f"Query {query_id} cancelled"))

    async def cleanup_query(self, query_id: str):
        """Clean up query resources"""
        if query_id in self.query_runners:
            runner = self.query_runners[query_id]
            self.query_status_cache[query_id] = runner.query.status
            await runner.cleanup_runtime_context()
            del self.query_runners[query_id]
        cleanup_tasks = []
        for scheduler in self.engine_schedulers.values():
            cleanup_method = getattr(scheduler, "cleanup_query", None)
            if cleanup_method is None:
                continue
            cleanup_tasks.append(cleanup_method.remote(query_id))

        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(
                        f"Engine cleanup_query failed for query {query_id}: {result}"
                    )

    async def _ttl_cleanup_loop(self):
        while True:
            await asyncio.sleep(self._cleanup_interval_s)
            now = time.time()
            expired = [
                qid
                for qid, outcome in self._query_outcomes.items()
                if now - outcome.get("finished_ts", 0) > self._outcome_ttl_s
            ]
            for qid in expired:
                self._query_outcomes.pop(qid, None)
                self._query_futures.pop(qid, None)
                self.query_status_cache.pop(qid, None)

    async def shutdown(self):
        """Shutdown the graph scheduler"""
        for query_id, future in list(self._query_futures.items()):
            if not future.done():
                future.set_exception(
                    RuntimeError(
                        f"GraphScheduler shutting down, query {query_id} cancelled"
                    )
                )
        for query_id in list(self.query_runners.keys()):
            await self.cleanup_query(query_id)
        for task in list(self._dag_tasks):
            if not task.done():
                task.cancel()
        if self._dag_tasks:
            await asyncio.gather(*self._dag_tasks, return_exceptions=True)
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        self.query_runners = {}
        self.query_status_cache = {}
        self._query_futures = {}
        self._query_outcomes = {}
