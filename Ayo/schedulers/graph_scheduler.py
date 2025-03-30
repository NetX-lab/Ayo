import ray
import asyncio
import threading
from typing import Dict, List, Optional, Any
from queue import Queue
from collections import deque
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node, NodeStatus, NodeType, NodeOps
from Ayo.engines.payload_transformers import TRANSFORMER_REGISTRY, DefaultTransformer
from Ayo.queries.query_state import QueryStatus
from Ayo.schedulers.engine_scheduler import EngineRequest
from Ayo.queries.query import Query
from Ayo.configs.model_config import get_aggregator_config, AggMode
import traceback

from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class QueryRunner:
    """A class for running a query"""
    def __init__(self, 
                 query: Query,
                 config: Dict,
                 engine_schedulers: Dict[str, ray.actor.ActorHandle]):
        
        self.query = query
        self.config = config
        self.dag = query.DAG
        self.engine_schedulers = engine_schedulers
        
        # Task queues
        self.running = deque()  # Running nodes
        self.ready = Queue()    # Ready nodes
        self.pending = deque()  # Pending nodes
        
        # Result store
        self.nodes_outputs = {}
        
        # Add tracking for different node types
        self.input_nodes = []
        self.compute_nodes = []
        self.output_nodes = []

        logger.info(f"query.query_state in query runner: {self.query.query_state}")
        
    def initialize(self):
        """Initialize the runner"""
        # Get topological sort
        self.dag.topological_sort()
        
        # Categorize nodes by type
        for node in self.dag.topo_list:
            if node.node_type == NodeType.INPUT:
                self.input_nodes.append(node)
            elif node.node_type == NodeType.COMPUTE:
                self.compute_nodes.append(node)
            elif node.node_type == NodeType.OUTPUT:
                self.output_nodes.append(node)
        
        # Initialize input nodes with values
        for node in self.input_nodes:
            node.status = NodeStatus.COMPLETED
            self.nodes_outputs[node.name] = node.input_values
            
        # Add compute and output nodes to pending
        self.pending.extend(self.compute_nodes)
        self.pending.extend(self.output_nodes)

        #log the dag's input nodes
        logger.info(f"DAG input nodes: {self.input_nodes}") 
        logger.info(f"DAG compute nodes: {self.compute_nodes}")
        logger.info(f"DAG output nodes: {self.output_nodes}")

    def prepare_engine_payload(self, node: Node) -> Any:
        """prepare the data to send to the engine"""
        if node.node_type != NodeType.COMPUTE:
            return node.input_kwargs
            
        # get the corresponding transformer, if not, use the default transformer
        transformer = TRANSFORMER_REGISTRY.get(
            node.engine_type, 
            DefaultTransformer()
        )

        return transformer.transform(node)

    async def check_node_ready(self, node: Node) -> bool:
        """Check if a node is ready to execute"""
        # Input nodes are always ready
        if node.node_type == NodeType.INPUT:
            return True
        
        elif node.op_type in [NodeOps.LLM_DECODING, NodeOps.LLM_PARTIAL_DECODING]:
            parent_op_types=[parent.op_type for parent in node.parents]
            if NodeOps.LLM_PARTIAL_DECODING in parent_op_types:
                return node.parents[parent_op_types.index(NodeOps.LLM_PARTIAL_DECODING)].status in [NodeStatus.RUNNING, NodeStatus.COMPLETED]
            else:
                potential_prefilling_status=[parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED] for parent in node.parents if parent.op_type == NodeOps.LLM_PREFILLING] 
                potential_full_prefilling_status=[parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED] for parent in node.parents if parent.op_type == NodeOps.LLM_FULL_PREFILLING]
                
                # 检查是否至少有一个父节点是预填充类型，并且所有这类父节点都已准备好
                has_prefilling_parents = len(potential_prefilling_status) > 0
                has_full_prefilling_parents = len(potential_full_prefilling_status) > 0
                
                if (has_prefilling_parents and all(potential_prefilling_status)) or \
                   (has_full_prefilling_parents and all(potential_full_prefilling_status)):
                    return True
                else:
                    return False
                
        elif node.op_type == NodeOps.LLM_FULL_PREFILLING: 
            llm_partial_prefilling_status=[parent.status in [NodeStatus.RUNNING, NodeStatus.COMPLETED] for parent in node.parents if parent.op_type == NodeOps.LLM_PARTIAL_PREFILLING] 
            other_parents_status=[parent.status in [NodeStatus.COMPLETED] for parent in node.parents if parent.op_type not in [NodeOps.LLM_PARTIAL_PREFILLING]]
            if all(llm_partial_prefilling_status) and all(other_parents_status):
                return True
            else:
                return False
            
        else:
            # For compute and output nodes, check parent completion
            return all(
                parent.status == NodeStatus.COMPLETED 
                for parent in node.parents
            )

    async def submit_node(self, node: Node):
        """submit a node to the engine scheduler"""
        try:
            logger.info(f"submit node: {node.name}")
            if node.node_type == NodeType.INPUT:
                return
            
            if node.node_type == NodeType.OUTPUT:
                if len(node.parents) == 1 and node.parents[0].status == NodeStatus.COMPLETED:
                    node.status = NodeStatus.COMPLETED
                    parent_node = node.parents[0]
                    node.update_input_kwargs(self.nodes_outputs)
                    self.nodes_outputs[node.name] = node.input_kwargs
                return
            
            if node.op_type == NodeOps.AGGREGATOR:
                # we do the in-place aggregation here, do not need to submit to the aggregator engine which would be discarded later
                aggregator_mode = get_aggregator_config(node)["agg_mode"]
                
                if aggregator_mode == AggMode.DUMMY:
                    node.status = NodeStatus.COMPLETED 
                    self.nodes_outputs[node.name] = {
                        k:True for k in node.output_names
                    }

                    await self.query.query_state.set_node_result.remote(node.name, self.nodes_outputs[node.name])

                    return 
                elif aggregator_mode == AggMode.MERGE:
                    node.status = NodeStatus.COMPLETED 
                    node.update_input_kwargs(self.nodes_outputs) 
                    if isinstance(list(node.input_kwargs.values())[0], dict):
                        output= {}
                    elif isinstance(list(node.input_kwargs.values())[0], list):
                        output= []
                    else:
                        raise ValueError(f"Unsupported input type: {type(list(node.input_kwargs.values())[0])}")
                    
                    for k,v in node.input_kwargs.items():
                        if isinstance(v, dict):
                            output.update(v)
                        elif isinstance(v, list):
                            output.extend(v)
                        else:
                            raise ValueError(f"Unsupported input type: {type(v)}")

                    self.nodes_outputs[node.name] = output
                    await self.query.query_state.set_node_result.remote(node.name, self.nodes_outputs[node.name])
                    return  
                
                elif aggregator_mode == AggMode.TOP_K:
                    # assume each output is a list of dicts or tuples 
                    # if is dict, {"text": "xxx", "score": 0.9}
                    # if is tuple, ("xxx", 0.9) 
                    node.status = NodeStatus.COMPLETED 
                    node.update_input_kwargs(self.nodes_outputs)
                    top_k = node.config.get("top_k", None)
                    if top_k is None:
                        raise ValueError(f"Missing top_k for node {node.name} aggregator")
                    output= []
                    for k,v in node.input_kwargs.items():
                        if isinstance(v, list):
                            if isinstance(v[0], dict):
                                keys=v[0].keys() 
                                for i,key in enumerate(keys): 
                                    if key == 'score':
                                        score_key = i 
                                        text_key = 1-i 
                                        break
                                output.extend((v[i][text_key], v[i][score_key]) for i in range(len(v)))
                            elif isinstance(v[0], tuple):
                                if isinstance(v[0][0], str):
                                    text_key = 0
                                    score_key = 1
                                else:
                                    text_key = 1
                                    score_key = 0
                                output.extend((v[i][text_key], v[i][score_key]) for i in range(len(v)))
                            else:
                                raise ValueError(f"Unsupported input type: {type(v[0])}")
                        else:
                            raise ValueError(f"Unsupported input type: {type(v)}")
                        
                    # sort the output by score
                    output.sort(key=lambda x: x[1], reverse=True)
                    # keep the top k
                    output = output[:top_k]
                    # only keep the text
                    output = [x[0] for x in output]
                    self.nodes_outputs[node.name] = output
                    logger.info(f"top {top_k} aggregation output for node: {node.name}", output)
                    await self.query.query_state.set_node_result.remote(node.name, self.nodes_outputs[node.name])
                    return 
                else:
                    raise ValueError(f"Unsupported aggregator mode: {get_aggregator_config(node)}")
            
            # Address the compute node
            node.update_input_kwargs(self.nodes_outputs)

            
            payload = self.prepare_engine_payload(node)

            #print(f"payload: {payload} for node: {node.name}")
            
            engine_type = node.engine_type
            scheduler = self.engine_schedulers.get(engine_type)
            
            if scheduler:
                engine_request = EngineRequest(
                    request_id=f"{self.query.query_id}::{node.name}", # the engine scheduler will use the request_id.split("::")[-1] to get the node_name 
                    query_id=self.query.query_id,
                    query=self.query,
                    payload=payload,
                )
                
                await scheduler.submit_request.remote(engine_request)
                node.status = NodeStatus.RUNNING
            else:
                raise ValueError(f"No scheduler found for engine type: {engine_type}")
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error submitting node {node.name}: {str(e)}\nTraceback:\n{error_trace}")
            node.status = NodeStatus.FAILED
            node.error_message = f"{str(e)}\nTraceback:\n{error_trace}"
            self.dag.error_nodes.append(node)

    async def cleanup_runtime_context(self):
        """Cleanup runtime context"""
        self.input_nodes.clear()
        self.compute_nodes.clear()
        self.output_nodes.clear()
        self.running.clear()
        self.ready.queue.clear()
        self.pending.clear()


@ray.remote
class GraphScheduler:
    """Scheduler for managing multiple queries and their DAG execution"""
    
    def __init__(self, engine_schedulers: Dict[str, ray.actor.ActorHandle]):
        self.engine_schedulers = engine_schedulers
        self.query_runners: Dict[str, QueryRunner] = {}
        
    async def update_schedulers(self, 
                              engine_schedulers: Dict[str, ray.actor.ActorHandle]) -> None:
        """Update engine schedulers
        
        Args:
            engine_schedulers: Dictionary of engine type to scheduler actor handle
        """
        self.engine_schedulers = engine_schedulers
        
    async def submit_query(self, 
                          query: Query, 
                          config: Optional[Dict] = None) -> str:
        """Submit a new query asynchronously"""
        try:
            runner = QueryRunner(
                query=query,
                config=config,
                engine_schedulers=self.engine_schedulers
            )
            self.query_runners[query.query_id] = runner
            
            # Initialize and start processing
            runner.initialize()
            asyncio.create_task(self._process_dag(runner))
            
            return query.query_id
            
        except Exception as e:
            logger.error(f"Failed to submit query {query.query_id}: {str(e)}")
            raise

    async def _process_dag(self, runner: QueryRunner):
        """Process DAG execution asynchronously"""
        try:
            last_pending_nodes_names=[]
            last_running_nodes_names=[]
            while not runner.dag.check_completion():
                # check the ready nodes
                ready_nodes = [
                    node for node in runner.pending
                    if await runner.check_node_ready(node)
                ]
                
                # batch update the queue
                for node in ready_nodes:
                    runner.ready.put(node)
                    runner.pending.remove(node)

            

                
                # batch submit the nodes
                while not runner.ready.empty():
                    nodes_to_submit = []
                    while not runner.ready.empty():
                        nodes_to_submit.append(runner.ready.get())
                    
                    # parallel submit the nodes
                    await asyncio.gather(*[
                        runner.submit_node(node) 
                        for node in nodes_to_submit
                    ])
                    runner.running.extend(nodes_to_submit)
                
                # print(f"runner.running: {[node.name for node in runner.running]}")

                current_pending_nodes_names=[ node.name for node in runner.pending]
                current_running_nodes_names=[ node.name for node in runner.running]

                if last_pending_nodes_names != current_pending_nodes_names or last_running_nodes_names != current_running_nodes_names:
                    logger.info(f"pending nodes: {current_pending_nodes_names}")
                    logger.info(f"running nodes: {current_running_nodes_names}")
                    
                    last_pending_nodes_names=current_pending_nodes_names
                    last_running_nodes_names=current_running_nodes_names

                # batch check the running nodes
                if runner.running:
                    try:
                        completed_nodes = []  # add: store the completed nodes
                        results = await asyncio.gather(*[
                            asyncio.wrap_future(runner.query.query_state.get_node_result.remote(node.name).future())
                            for node in runner.running
                        ], return_exceptions=True)
                        
                        # check the errors in the results
                        for node, result in zip(runner.running, results):
                            if isinstance(result, Exception):
                                # handle the failed nodes
                                node.status = NodeStatus.FAILED
                                node.error_message = str(result)
                                runner.dag.error_nodes.append(node)
                                logger.error(f"Node execution failed: {node.name}: {str(result)}")
                                # terminate the DAG execution immediately
                                raise Exception(f"DAG execution terminated due to node failure: {node.name} - {str(result)}")
                            
                            elif result is not None:
                                # handle the successful nodes
                                logger.info(f"result have been received for node: {node.name} in graph scheduler")
                                runner.nodes_outputs[node.name] = {node.output_names[0]: result}
                                node.status = NodeStatus.COMPLETED
                                completed_nodes.append(node)  # add to the completed nodes list
                        
                        # remove the completed nodes
                        for node in completed_nodes:
                            runner.running.remove(node)

                    except Exception as e:
                        logger.error(f"Error in DAG execution: {str(e)}")
                        # mark all the running nodes as failed
                        for node in runner.running:
                            node.status = NodeStatus.FAILED
                            node.error_message = str(e)
                            runner.dag.error_nodes.append(node)
                        runner.running.clear()
                        # set the query status to failed and raise the exception
                        runner.query.status = QueryStatus.FAILED
                        raise
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"DAG execution failed for query {runner.query.query_id}: {str(e)}")
            # ensure all the running nodes are marked as failed
            runner.dag.error_nodes.extend(runner.running)
            for node in runner.running:
                node.status = NodeStatus.FAILED
                node.error_message = str(e)
            runner.query.status = QueryStatus.FAILED
            # raise the exception to the upper layer
            raise Exception(f"DAG execution failed: {str(e)}")
        finally:
            # only set the query status to completed when there is no error
            if not runner.dag.error_nodes:
                runner.query.status = QueryStatus.COMPLETED
            await self.cleanup_query(runner.query.query_id)

    async def get_query_status(self, query_id: str) -> QueryStatus:
        """Get query status"""
        if query_id in self.query_runners:
            runner = self.query_runners[query_id]
            return runner.query.status
        else:
            return None
                  

    async def cleanup_query(self, query_id: str):
        """Clean up query resources"""
        if query_id in self.query_runners:
            runner = self.query_runners[query_id]
            await runner.cleanup_runtime_context()

    async def shutdown(self):
        """Shutdown the graph scheduler"""
        for query_id, runner in self.query_runners.items():
            await self.cleanup_query(query_id)
        self.query_runners = {}
        
